//! Integration checks for the Responses route and tool-call compatibility.

use super::*;

#[tokio::test]
async fn responses_conversation_is_normalized_without_leaking_into_chat_body() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "conversation": {"id": "conversation-7"},
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests[0].agent_session(), Some("conversation-7"));
    assert_eq!(
        requests[0].agent_session_source(),
        Some("responses.conversation")
    );
    assert!(!requests[0].extra.contains_key("conversation"));
}

#[tokio::test]
async fn conflicting_header_and_responses_conversation_fail_closed() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for_with_config(
        backend.clone(),
        InferenceFrontendConfig::default()
            .with_agent_session_header(HeaderName::from_static("x-litellm-session-id")),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("x-litellm-session-id", "header-session")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "conversation": {"id": "body-session"},
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(backend.requests.lock().unwrap().is_empty());
}

/// The same rung-ladder marker must ride out of the non-streaming
/// `/v1/responses` leg too: it calls the same `chat_completion_with_context`
/// hook path as `/v1/chat/completions`, but translates the response into a
/// new `Response` on its way out, which previously dropped the marker
/// extension before `frontend_lifecycle_middleware` could read it.
#[tokio::test]
async fn hook_minted_capsule_marker_is_exposed_on_non_streaming_responses() {
    let backend = crate::hooks::HookedInferenceBackend::new(
        Arc::new(FakeBackend),
        Arc::new(CapsuleMintingHook),
    );
    let app = router_for(Arc::new(backend));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-mesh",
                        "input": "hi"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let header = response
        .headers()
        .get("x-capsule-id")
        .expect("X-Capsule-Id header present")
        .to_str()
        .unwrap();
    assert!(header.starts_with("capsule-chatcmpl"));
}

#[tokio::test]
async fn responses_stream_frames_backend_errors_without_completed_tail() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "stream-error",
            "input": "hi",
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: error"));
    assert!(body.contains(r#""message":"stream backend failed""#));
    assert!(body.contains(r#""code":"service_unavailable""#));
    assert!(!body.contains("event: response.completed"));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn responses_route_translates_to_chat_and_back() {
    let response = post_json(
            "/v1/responses",
            json!({
                "model": "test-model",
                "instructions": "be concise",
                "input": "hi",
                "max_output_tokens": 12,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
                "logprobs": true,
                "top_logprobs": 1
            }),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "response");
    assert_eq!(body["output_text"], "echo: be concise\nhi");
    assert_eq!(body["usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn responses_route_preserves_tool_calls_and_logprobs() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "tool-call",
            "input": "hi",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "logprobs": true,
            "top_logprobs": 1
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["output_text"], "calling lookup");
    assert_eq!(
        body["output"][0]["content"][0]["logprobs"]["content"][0]["token"],
        "calling"
    );
    assert_eq!(body["output"][1]["type"], "function_call");
    assert_eq!(body["output"][1]["call_id"], "call_123");
    assert_eq!(body["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn responses_route_replays_mixed_prior_output_with_function_result() {
    let previous = post_json(
        "/v1/responses",
        json!({"model": "tool-call", "input": "hi"}),
    )
    .await;
    assert_eq!(previous.status(), StatusCode::OK);
    let previous = response_body_json(previous).await;
    let mut input = previous["output"].as_array().unwrap().clone();
    assert_eq!(input[0]["content"][0]["type"], "output_text");
    assert_eq!(input[1]["type"], "function_call");
    let call_id = input[1]["call_id"].clone();
    input.push(json!({"type": "function_call_output",
        "call_id": call_id, "output": "sunny"}));

    let response = post_json(
        "/v1/responses",
        json!({"model": "test-model", "input": input}),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["output_text"], "echo: calling lookup\nsunny");
}

#[tokio::test]
async fn responses_route_accepts_function_call_output_after_its_call() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": "weather"},
                {"type": "function_call", "call_id": "call_1", "name": "lookup",
                    "arguments": "{\"city\":\"Sydney\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}
            ]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["output_text"], "echo: weather\nsunny");
}

#[tokio::test]
async fn responses_route_rejects_unrecognized_typed_input_items() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "item_reference", "id": "msg_1"}
            ]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("item_reference")
    );
}

#[tokio::test]
async fn responses_route_preserves_backend_unsupported_errors() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "unsupported-feature",
            "input": "hi",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}}
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "unsupported_model_feature");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("structured output")
    );
}

// The guarded chat tool call is also the shape projected by Responses.
#[tokio::test]
async fn guarded_chat_rescues_tool_call_text() {
    let backend = Arc::new(GuardrailRescueBackend::default());
    let app = guarded_test_app(backend.clone());

    let response = post_json_with_app_and_request_id(
        app,
        "/v1/chat/completions",
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }),
        Some("a35bb624-5c07-431f-92a9-9c884472ca95"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-request-id"],
        "a35bb624-5c07-431f-92a9-9c884472ca95"
    );
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"][0]["message"]["content"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
    assert!(!serde_json::to_string(&body).unwrap().contains("_mesh_"));

    let seen_requests = backend.seen_chat_requests.lock().unwrap();
    assert_eq!(seen_requests.len(), 1);
    let seen_tools = seen_requests[0]
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .cloned()
        .expect("guarded backend should receive tools");
    assert_eq!(seen_tools.len(), 2);
    assert_eq!(seen_tools[0]["function"]["name"], "lookup");
    assert_eq!(seen_tools[1]["function"]["name"], "_mesh_respond");
}

#[tokio::test]
async fn responses_stream_route_returns_responses_sse() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "test-model",
            "input": "hi",
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.output_item.added"));
    assert!(body.contains("event: response.content_part.added"));
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains("event: response.output_text.done"));
    assert!(body.contains("event: response.content_part.done"));
    assert!(body.contains("event: response.output_item.done"));
    assert!(body.contains("event: response.completed"));
    assert!(body.contains(r#""sequence_number":1"#));
    assert!(body.contains(r#""output_text":"hello""#));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn responses_stream_route_emits_live_function_call_without_empty_text() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "stream-tool-calls",
            "input": "weather",
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("data: [DONE]"));
    let events = response_sse_payloads(&body);
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .expect("function call item is announced");
    assert_eq!(added["item"]["type"], "function_call");
    assert_eq!(added["item"]["call_id"], "call_live_1");
    assert_eq!(added["item"]["name"], "lookup");
    let deltas: Vec<&str> = events
        .iter()
        .filter(|event| event["type"] == "response.function_call_arguments.delta")
        .filter_map(|event| event["delta"].as_str())
        .collect();
    assert_eq!(deltas, ["{\"city\":", "\"Sydney\"}"]);
    let done = events
        .iter()
        .find(|event| event["type"] == "response.function_call_arguments.done")
        .expect("function call arguments finish");
    assert_eq!(done["arguments"], "{\"city\":\"Sydney\"}");

    let completed = events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("stream completes");
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(
        output.len(),
        1,
        "no empty assistant text item may be emitted"
    );
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["id"], added["item"]["id"]);
    assert_eq!(output[0]["call_id"], "call_live_1");
    assert_eq!(output[0]["name"], "lookup");
    assert_eq!(output[0]["arguments"], "{\"city\":\"Sydney\"}");
    assert_eq!(completed["response"]["usage"]["input_tokens"], 3);
    assert!(
        !events
            .iter()
            .any(|event| event["type"] == "response.output_text.done")
    );
    let sequence_numbers: Vec<i64> = events
        .iter()
        .filter_map(|event| event["sequence_number"].as_i64())
        .collect();
    assert!(sequence_numbers.windows(2).all(|pair| pair[0] < pair[1]));
}

fn response_sse_payloads(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE data is JSON"))
        .collect()
}

#[tokio::test]
async fn responses_stream_route_preserves_logprobs() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "stream-logprobs",
            "input": "hi",
            "stream": true,
            "logprobs": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains(r#""logprobs":{"content":[{"logprob":-0.1,"token":"tok"}]}"#));
}
