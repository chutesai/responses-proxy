use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::sse::{Event, Sse},
};
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{sync::RwLock, task};
use tokio_stream::wrappers::ReceiverStream;

/// Maximum size for error response bodies to prevent DoS (10KB)
const MAX_ERROR_BODY_SIZE: usize = 10 * 1024;

/// Maximum size for input content to prevent memory exhaustion (5MB)
const MAX_INPUT_CONTENT_SIZE: usize = 5 * 1024 * 1024;
const REALTIME_ITEM_OBJECT: &str = "realtime.item";
use crate::models::{
    App, ChatCompletionChunk, IncompleteDetails, OutputContent, OutputItem, Response,
    ResponseReasoningState, ResponseRequest, StreamEvent, TokenDetails, Usage,
};
use crate::services::{
    build_model_list_content, convert_to_chat_completions, extract_client_key,
    format_backend_error, get_available_models, mask_token, model_supports_feature,
    normalize_model_name, SseEventParser,
};
use crate::utils::{
    dump_backend_chunk, dump_backend_request, dump_request, dump_stream_event,
    extract_xml_tool_calls,
};

/// Track state of a tool call as it streams
#[derive(Debug, Clone)]
struct ToolCallState {
    call_id: String,
    item_id: String,
    type_: String,
    name: Option<String>,
    arguments: String,
    item_added: bool,     // Whether we've sent the output_item.added event
    end_emitted: bool,    // Whether we've emitted output_tool_call.end/legacy done events
    pending_args: String, // Arguments buffered before name arrives
}

/// Helper to assign monotonic event and sequence identifiers
struct EventSequencer {
    next_event_id: u64,
    next_sequence: u32,
}

impl EventSequencer {
    fn new() -> Self {
        Self {
            next_event_id: 0,
            next_sequence: 0,
        }
    }

    fn prepare(
        &mut self,
        mut event: StreamEvent,
        response_id: &str,
    ) -> Result<(String, u32), serde_json::Error> {
        self.next_event_id = self.next_event_id.saturating_add(1);
        self.next_sequence = self.next_sequence.saturating_add(1);

        let event_id = format!("evt_{response_id}_{:016x}", self.next_event_id);
        event.event_id = Some(event_id);
        event.response_id = Some(response_id.to_string());
        event.sequence_number = Some(self.next_sequence);

        let sequence_number = self.next_sequence;
        let json = serde_json::to_string(&event)?;

        Ok((json, sequence_number))
    }
}

/// Helper to serialize, log, and dispatch stream events
async fn dispatch_event(
    tx: &tokio::sync::mpsc::Sender<Event>,
    sequencer: &mut EventSequencer,
    response_id: &str,
    request_id: &str,
    event: StreamEvent,
) {
    let event_type = event.type_.clone();
    match sequencer.prepare(event, response_id) {
        Ok((json, sequence_number)) => {
            dump_stream_event(&json, request_id, sequence_number);
            let _ = tx.send(Event::default().data(json)).await;
        }
        Err(err) => {
            log::error!("❌ Failed to serialize stream event {}: {err}", event_type);
        }
    }
}

async fn emit_tool_call_begin_events(
    tx: &tokio::sync::mpsc::Sender<Event>,
    sequencer: &mut EventSequencer,
    response_id: &str,
    request_id: &str,
    item_id: &str,
    call_id: &str,
    function_name: &str,
    output_index: u32,
) {
    let begin_event = StreamEvent {
        type_: "response.output_tool_call.begin".to_string(),
        response: None,
        event_id: None,
        response_id: None,
        item_id: Some(item_id.to_string()),
        output_index: Some(output_index),
        content_index: None,
        delta: None,
        text: None,
        item: None,
        sequence_number: None,
        call_id: Some(call_id.to_string()),
        name: Some(function_name.to_string()),
        arguments: None,
        error: None,
    };

    dispatch_event(tx, sequencer, response_id, request_id, begin_event).await;

    let legacy_event = StreamEvent {
        type_: "response.output_item.added".to_string(),
        response: None,
        event_id: None,
        response_id: None,
        item_id: Some(item_id.to_string()),
        output_index: Some(output_index),
        content_index: None,
        delta: None,
        text: None,
        item: Some(OutputItem {
            id: item_id.to_string(),
            object: REALTIME_ITEM_OBJECT.to_string(),
            type_: "function_call".to_string(),
            status: "in_progress".to_string(),
            role: None,
            content: None,
            call_id: Some(call_id.to_string()),
            name: Some(function_name.to_string()),
            arguments: Some(String::new()),
            output: None,
        }),
        sequence_number: None,
        call_id: Some(call_id.to_string()),
        name: None,
        arguments: None,
        error: None,
    };

    dispatch_event(tx, sequencer, response_id, request_id, legacy_event).await;
}

async fn emit_tool_call_delta_events(
    tx: &tokio::sync::mpsc::Sender<Event>,
    sequencer: &mut EventSequencer,
    response_id: &str,
    request_id: &str,
    item_id: &str,
    call_id: &str,
    output_index: u32,
    delta: &str,
) {
    let delta_string = delta.to_string();

    let modern_event = StreamEvent {
        type_: "response.output_tool_call.delta".to_string(),
        response: None,
        event_id: None,
        response_id: None,
        item_id: Some(item_id.to_string()),
        output_index: Some(output_index),
        content_index: None,
        delta: Some(delta_string.clone()),
        text: None,
        item: None,
        sequence_number: None,
        call_id: Some(call_id.to_string()),
        name: None,
        arguments: None,
        error: None,
    };

    dispatch_event(tx, sequencer, response_id, request_id, modern_event).await;

    let legacy_event = StreamEvent {
        type_: "response.function_call_arguments.delta".to_string(),
        response: None,
        event_id: None,
        response_id: None,
        item_id: Some(item_id.to_string()),
        output_index: Some(output_index),
        content_index: None,
        delta: Some(delta_string),
        text: None,
        item: None,
        sequence_number: None,
        call_id: Some(call_id.to_string()),
        name: None,
        arguments: None,
        error: None,
    };

    dispatch_event(tx, sequencer, response_id, request_id, legacy_event).await;
}

async fn emit_tool_call_end_event(
    tx: &tokio::sync::mpsc::Sender<Event>,
    sequencer: &mut EventSequencer,
    response_id: &str,
    request_id: &str,
    item_id: &str,
    call_id: &str,
    output_index: u32,
    function_name: &str,
    arguments: &str,
) {
    let end_event = StreamEvent {
        type_: "response.output_tool_call.end".to_string(),
        response: None,
        event_id: None,
        response_id: None,
        item_id: Some(item_id.to_string()),
        output_index: Some(output_index),
        content_index: None,
        delta: None,
        text: None,
        item: None,
        sequence_number: None,
        call_id: Some(call_id.to_string()),
        name: Some(function_name.to_string()),
        arguments: Some(arguments.to_string()),
        error: None,
    };

    dispatch_event(tx, sequencer, response_id, request_id, end_event).await;
}

/// Record a circuit breaker failure asynchronously
#[inline]
fn record_circuit_breaker_failure(cb: Arc<RwLock<crate::models::CircuitBreakerState>>) {
    task::spawn(async move {
        cb.write().await.record_failure();
    });
}

fn warn_unsupported_features(req: &ResponseRequest) {
    if let Some(include) = &req.include {
        if !include.is_empty() {
            log::warn!(
                "⚠️  'include' values {:?} are not supported by this proxy and will be ignored",
                include
            );
        }
    }

    if let Some(stream_options) = &req.stream_options {
        if stream_options.include_obfuscation.is_some() {
            log::warn!("⚠️  stream_options.include_obfuscation is not supported");
        }
    }

    if req.conversation.is_some() {
        log::warn!("⚠️  conversation references are ignored (proxy is stateless)");
    }

    if req.previous_response_id.is_some() {
        log::warn!("⚠️  previous_response_id is ignored (proxy is stateless)");
    }

    if let Some(reasoning) = &req.reasoning {
        if reasoning.summary.is_some() || reasoning.generate_summary.is_some() {
            log::warn!("⚠️  reasoning summary preferences are not supported and will be ignored");
        }
    }

    if req.max_tool_calls.is_some() {
        log::warn!("⚠️  max_tool_calls is not enforced");
    }

    if let Some(text) = &req.text {
        if text.verbosity.is_some() {
            log::warn!("⚠️  text.verbosity is not supported");
        }
    }

    if req.safety_identifier.is_some() {
        log::warn!("⚠️  safety_identifier is not forwarded to the backend");
    }

    if req.prompt_cache_key.is_some() {
        log::warn!("⚠️  prompt_cache_key is not forwarded to the backend");
    }

    if req.service_tier.is_some() {
        log::warn!("⚠️  service_tier overrides are not supported");
    }
}

pub async fn create_response(
    State(app): State<App>,
    headers: HeaderMap,
    body: String,
) -> Result<
    (
        HeaderMap,
        Sse<impl Stream<Item = Result<Event, Infallible>>>,
    ),
    (StatusCode, &'static str),
> {
    let request_start = SystemTime::now();
    let request_id = format!(
        "{:x}",
        request_start.duration_since(UNIX_EPOCH).unwrap().as_nanos()
    );

    // Dump full request to logs
    dump_request(&body, &request_id);

    // Log request body for debugging (truncate if too large)
    if log::log_enabled!(log::Level::Debug) {
        let preview = if body.len() > 1000 {
            format!("{}... ({} total bytes)", &body[..1000], body.len())
        } else {
            body.clone()
        };
        log::debug!("📥 [{}] Incoming request body: {}", request_id, preview);
    }

    // Parse request - detect if it's Chat Completions or Responses format
    let is_chat_completions_format = body.contains("\"messages\"") && !body.contains("\"input\"");

    let req: ResponseRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            log::error!("❌ Failed to parse request: {}", e);
            log::error!(
                "❌ Request body (first 500 chars): {}",
                &body[..body.len().min(500)]
            );
            return Err((StatusCode::UNPROCESSABLE_ENTITY, "invalid_request_format"));
        }
    };

    if is_chat_completions_format {
        log::info!("📨 Detected Chat Completions format request (using messages field)");
    }

    if req.store.unwrap_or(false) {
        log::warn!("⚠️  'store' flag requested but persistence is not supported; ignoring");
    }

    if req.background.unwrap_or(false) {
        log::error!("❌ Background responses are not supported by this proxy");
        return Err((StatusCode::BAD_REQUEST, "background_not_supported"));
    }

    if req.prompt.is_some() {
        log::error!("❌ Prompt template references are not supported by this proxy");
        return Err((StatusCode::BAD_REQUEST, "prompt_reference_not_supported"));
    }

    // Circuit breaker check
    {
        let mut cb = app.circuit_breaker.write().await;
        if !cb.should_allow_request() {
            log::error!("🔴 Circuit breaker is open - rejecting request");
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "backend_unavailable_circuit_open",
            ));
        }
    }

    // Request validation
    if let Some(crate::models::ResponseInput::Array(items)) = &req.input {
        if items.len() > 1000 {
            log::warn!(
                "❌ Validation failed: too many input items ({})",
                items.len()
            );
            return Err((StatusCode::BAD_REQUEST, "too_many_messages"));
        }
    }

    // Validate max_output_tokens if provided
    if let Some(max_tokens) = req.max_output_tokens {
        if !(1..=100_000).contains(&max_tokens) {
            log::warn!(
                "❌ Validation failed: max_output_tokens out of range ({})",
                max_tokens
            );
            return Err((StatusCode::BAD_REQUEST, "invalid_max_tokens"));
        }
    }

    // Validate instructions length if provided
    if let Some(ref instructions) = req.instructions {
        if instructions.len() > 100 * 1024 {
            // 100KB limit
            log::warn!(
                "❌ Validation failed: instructions too large ({} bytes)",
                instructions.len()
            );
            return Err((StatusCode::BAD_REQUEST, "instructions_too_large"));
        }
    }

    // Validate input content size to prevent memory exhaustion
    if let Some(ref input) = req.input {
        let input_size = estimate_input_size(input);
        if input_size > MAX_INPUT_CONTENT_SIZE {
            log::warn!(
                "❌ Validation failed: input content too large ({} bytes, max {} bytes)",
                input_size,
                MAX_INPUT_CONTENT_SIZE
            );
            return Err((StatusCode::PAYLOAD_TOO_LARGE, "input_content_too_large"));
        }
    }

    if let Some(top_logprobs) = req.top_logprobs {
        if top_logprobs > 20 {
            log::warn!(
                "❌ Validation failed: top_logprobs out of range ({})",
                top_logprobs
            );
            return Err((StatusCode::BAD_REQUEST, "invalid_top_logprobs"));
        }
    }

    warn_unsupported_features(&req);

    // Extract and validate auth
    let client_key = extract_client_key(&headers);

    if let Some(key) = &client_key {
        log::info!("🔑 Client API Key: Bearer {}", mask_token(key));
    } else {
        log::warn!("❌ No client API key provided");
        return Err((StatusCode::UNAUTHORIZED, "missing_api_key"));
    }

    // Extract and normalize model name
    let requested_model = req
        .model
        .clone()
        .ok_or((StatusCode::BAD_REQUEST, "model_required"))?;

    // Normalize model name (use Arc to avoid string clones for error/metrics)
    let backend_model: Arc<str> = Arc::from(normalize_model_name(&requested_model, &app).await);
    let backend_model_for_error = Arc::clone(&backend_model);
    let backend_model_for_metrics = Arc::clone(&backend_model);

    // Check model capability for tool calling
    let supports_native_tools = model_supports_feature(&backend_model, "tools", &app).await
        || model_supports_feature(&backend_model, "function_calling", &app).await;

    if req.tools.is_some() {
        if !supports_native_tools {
            log::warn!(
                "⚠️ Model '{}' may not support native tool calling - XML tool instruction will be injected",
                backend_model
            );
        } else {
            log::debug!("✅ Model '{}' supports native tool calling", backend_model);
        }
    }

    // Convert Responses API request to Chat Completions format
    let chat_req = match convert_to_chat_completions(&req, supports_native_tools) {
        Ok(mut cr) => {
            // Ensure the normalized model name is used in the converted request
            cr.model = backend_model.to_string();
            cr
        }
        Err(e) => {
            log::error!("❌ Request conversion failed: {}", e);
            return Err((StatusCode::BAD_REQUEST, "invalid_request"));
        }
    };

    // Add detailed tool logging for debugging
    if let Some(ref tools) = req.tools {
        log::info!("🔧 Original request contains {} tool(s)", tools.len());
        for tool in tools {
            let func_def = tool.function_def();
            log::debug!("   - Tool type: {}, name: {}", tool.type_(), func_def.name);
        }
    }

    // Log what tools are being sent to backend
    if let Some(ref tools) = chat_req.tools {
        log::info!("🔧 Sending {} tool(s) to backend", tools.len());
        for tool in tools {
            let crate::models::ChatTool::Function { type_, .. } = tool;
            log::debug!("   - Backend tool type: {}", type_);
        }
    } else {
        log::info!("🔧 No tools being sent to backend");
    }

    // Check for non-function tools that might not be supported
    if let Some(ref tools) = req.tools {
        let non_function_tools: Vec<String> = tools
            .iter()
            .filter_map(|tool| match tool {
                crate::models::Tool::Nested { type_, .. } if type_ != "function" => {
                    Some(type_.clone())
                }
                crate::models::Tool::Flat { type_, .. } if type_ != "function" => {
                    Some(type_.clone())
                }
                _ => None,
            })
            .collect();

        if !non_function_tools.is_empty() {
            log::warn!("⚠️ Request contains non-function tools: {} - these may not work with Chat Completions API backends like Chutes.ai", non_function_tools.join(", "));
            log::warn!(
                "💡 For reliable tool calling, use only 'function' type tools with this proxy"
            );
        }
    }

    log::info!(
        "📨 Request: model={}, messages={}, stream={}, backend={}",
        backend_model.as_ref(),
        chat_req.messages.len(),
        chat_req.stream,
        app.backend_url
    );

    // Build the backend request
    let mut backend_req = app
        .client
        .post(&app.backend_url)
        .header("content-type", "application/json");

    // Forward client headers, skipping hop-by-hop and headers we set ourselves
    for (name, value) in headers.iter() {
        match name.as_str() {
            "host" | "connection" | "keep-alive" | "transfer-encoding" | "upgrade"
            | "proxy-authenticate" | "proxy-authorization" | "te" | "trailers"
            | "content-type" | "content-length" => {}
            _ => {
                backend_req = backend_req.header(name, value);
            }
        }
    }

    if let Some(host) = &app.backend_host_header {
        backend_req = backend_req.header("host", host);
    }

    if client_key.is_some() {
        log::info!("🔄 Auth: Forwarding client key to backend");
    }

    // Send request to backend
    log::debug!(
        "🚀 [{}] Sending request to backend with {} messages",
        request_id,
        chat_req.messages.len()
    );

    // Dump backend request
    if let Ok(backend_body) = serde_json::to_string(&chat_req) {
        dump_backend_request(&backend_body, &request_id);
    }

    let res = backend_req.json(&chat_req).send().await.map_err(|e| {
        log::error!("❌ Backend connection failed: {}", e);
        record_circuit_breaker_failure(app.circuit_breaker.clone());
        (StatusCode::BAD_GATEWAY, "backend_unavailable")
    })?;

    let status = res.status();
    log::debug!("📥 Backend response status: {}", status);

    // Handle non-success responses
    if !status.is_success() {
        record_circuit_breaker_failure(app.circuit_breaker.clone());

        let error_body = read_bounded_error(res).await;

        log::error!(
            "❌ Backend returned error: {} {} - {} ({} bytes)",
            status.as_u16(),
            status.canonical_reason().unwrap_or(""),
            &error_body[..error_body.len().min(200)], // Log first 200 chars
            error_body.len()
        );

        // Create error stream for non-success responses
        let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);

        // Handle 404 with model list
        if status == StatusCode::NOT_FOUND {
            let models = get_available_models(&app).await;
            if !models.is_empty() {
                log::info!(
                    "💡 Model '{}' not found - sending model list",
                    backend_model_for_error
                );
                send_error_response(
                    tx,
                    backend_model_for_error.to_string(),
                    build_model_list_content(&backend_model_for_error, &models),
                    "model_not_found".to_string(),
                );
            } else {
                send_error_response(
                    tx,
                    backend_model_for_error.to_string(),
                    format_backend_error(&error_body, &error_body),
                    "backend_error".to_string(),
                );
            }
        } else {
            send_error_response(
                tx,
                backend_model_for_error.to_string(),
                format_backend_error(&error_body, &error_body),
                "backend_error".to_string(),
            );
        }

        let mut out_headers = HeaderMap::new();
        out_headers.insert("cache-control", "no-cache".parse().unwrap());
        out_headers.insert("connection", "keep-alive".parse().unwrap());
        out_headers.insert("x-accel-buffering", "no".parse().unwrap());
        out_headers.insert("content-type", "text/event-stream".parse().unwrap());

        let stream = ReceiverStream::new(rx).map(Ok::<Event, Infallible>);
        return Ok((out_headers, Sse::new(stream)));
    }

    log::info!("✅ Backend responded successfully ({})", status);

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let model_for_response = Arc::clone(&backend_model);

    // Clone request parameters to echo back in response
    let req_instructions = req.instructions.clone();
    let req_tools = req.tools.clone();
    let req_tool_choice = req.tool_choice.clone();
    let req_parallel_tool_calls = req.parallel_tool_calls;
    let req_temperature = req.temperature;
    let req_top_p = req.top_p;
    let req_max_output_tokens = req.max_output_tokens;
    let req_metadata = req.metadata.clone();
    let req_store = Some(false);
    let req_previous_response_id = req.previous_response_id.clone();
    let req_reasoning_state = req.reasoning.as_ref().map(ResponseReasoningState::from);
    let req_background = req.background;
    let req_max_tool_calls = req.max_tool_calls;
    let req_text = req.text.clone();
    let req_prompt = req.prompt.clone();
    let req_truncation = req.truncation.clone();
    let req_conversation = req.conversation.clone();
    let req_top_logprobs = req.top_logprobs;
    let req_user = req.user.clone();
    let req_safety_identifier = req.safety_identifier.clone();
    let req_prompt_cache_key = req.prompt_cache_key.clone();
    let req_service_tier = req.service_tier.clone();

    // Clone request_id for logging in spawn
    let request_id_clone = request_id.clone();

    // Spawn streaming task
    tokio::spawn(async move {
        let request_id = request_id_clone;
        log::debug!("🎬 Streaming task started");

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let created_at = timestamp.as_secs();
        let id_seed = format!("{}_{}", request_id, timestamp.as_nanos());
        let response_id = format!("resp_{}", request_id);
        let message_id = format!("msg_{}", id_seed);
        let reasoning_id_seed = format!("reasoning_{}", id_seed);
        let mut sequencer = EventSequencer::new();

        // Send response.created event
        let created_event = StreamEvent {
            type_: "response.created".to_string(),
            response: Some(Response {
                id: response_id.clone(),
                object: "response".to_string(),
                created_at,
                status: "in_progress".to_string(),
                error: None,
                incomplete_details: None,
                model: Some(model_for_response.to_string()),
                output: vec![],
                usage: None,
                metadata: req_metadata.clone(),
                // Echo back request parameters
                instructions: req_instructions.clone(),
                tools: req_tools.clone(),
                tool_choice: req_tool_choice.clone(),
                parallel_tool_calls: req_parallel_tool_calls,
                temperature: req_temperature,
                top_p: req_top_p,
                max_output_tokens: req_max_output_tokens,
                store: req_store,
                previous_response_id: req_previous_response_id.clone(),
                reasoning: req_reasoning_state.clone(),
                background: req_background,
                max_tool_calls: req_max_tool_calls,
                text: req_text.clone(),
                prompt: req_prompt.clone(),
                truncation: req_truncation.clone(),
                conversation: req_conversation.clone(),
                top_logprobs: req_top_logprobs,
                user: req_user.clone(),
                safety_identifier: req_safety_identifier.clone(),
                prompt_cache_key: req_prompt_cache_key.clone(),
                service_tier: req_service_tier.clone(),
            }),
            event_id: None,
            response_id: None,
            item_id: None,
            output_index: None,
            content_index: None,
            delta: None,
            text: None,
            item: None,
            sequence_number: None,
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };
        dispatch_event(
            &tx,
            &mut sequencer,
            &response_id,
            &request_id,
            created_event,
        )
        .await;

        // Send output_item.added event
        let item_added_event = StreamEvent {
            type_: "response.output_item.added".to_string(),
            response: None,
            item_id: Some(message_id.clone()),
            output_index: Some(0),
            content_index: None,
            delta: None,
            text: None,
            item: Some(OutputItem {
                id: message_id.clone(),
                object: REALTIME_ITEM_OBJECT.to_string(),
                type_: "message".to_string(),
                status: "in_progress".to_string(),
                role: Some("assistant".to_string()),
                content: Some(vec![]),
                call_id: None,
                name: None,
                arguments: None,
                output: None,
            }),
            event_id: None,
            response_id: None,
            sequence_number: None,
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };
        dispatch_event(
            &tx,
            &mut sequencer,
            &response_id,
            &request_id,
            item_added_event,
        )
        .await;

        // Send content_part.added event
        let content_added_event = StreamEvent {
            type_: "response.content_part.added".to_string(),
            response: None,
            item_id: Some(message_id.clone()),
            output_index: Some(0),
            content_index: Some(0),
            delta: None,
            text: None,
            item: None,
            event_id: None,
            response_id: None,
            sequence_number: None,
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };
        dispatch_event(
            &tx,
            &mut sequencer,
            &response_id,
            &request_id,
            content_added_event,
        )
        .await;

        let mut bytes_stream = res.bytes_stream();
        let mut sse_parser = SseEventParser::new();
        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut reasoning_started = false;
        let mut reasoning_item_id: Option<String> = None;
        let mut done = false;
        let mut final_status = "completed";
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;
        let mut backend_chunk_num = 0u32;

        // Tool call tracking
        use std::collections::HashMap;
        let mut tool_calls: HashMap<usize, ToolCallState> = HashMap::new();
        let mut next_xml_index: usize = 0; // Track next available index for XML tool calls

        // XML buffering - track if we're waiting for closing tag
        let mut xml_buffering = false;

        // Process streaming response
        while let Some(item) = bytes_stream.next().await {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(e) => {
                    log::error!("❌ Error reading chunk from stream: {}", e);
                    break;
                }
            };

            for payload in sse_parser.push_and_drain_events(&chunk) {
                let data = payload.trim();

                // Dump backend chunk
                backend_chunk_num += 1;
                dump_backend_chunk(data, &request_id, backend_chunk_num);

                if data == "[DONE]" {
                    log::debug!("🏁 [{}] Received [DONE] marker from backend", request_id);
                    done = true;
                    break;
                }
                if data.is_empty() {
                    continue;
                }

                let parsed: Result<ChatCompletionChunk, _> = serde_json::from_str(data);

                let chunk = match parsed {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("⚠️  Failed to parse chunk: {}", e);
                        continue;
                    }
                };

                // Handle error in chunk
                if let Some(error) = &chunk.error {
                    log::error!("❌ Backend returned error in chunk: {:?}", error);
                    final_status = "failed";
                    done = true;
                    break;
                }

                if chunk.choices.is_empty() {
                    continue;
                }

                let choice = &chunk.choices[0];

                // Update final status based on finish_reason
                if let Some(reason) = &choice.finish_reason {
                    final_status = crate::services::translate_finish_reason(Some(reason));
                    log::debug!(
                        "📍 Backend finish_reason: {} → status: {}",
                        reason,
                        final_status
                    );
                }

                // Capture usage if provided
                if let Some(usage) = &chunk.usage {
                    if let Some(prompt) = usage.prompt_tokens {
                        total_input_tokens = prompt;
                    }
                    if let Some(completion) = usage.completion_tokens {
                        total_output_tokens = completion;
                    }
                }

                // Handle complete message (non-streaming fallback)
                if let Some(message) = &choice.message {
                    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                        accumulated_text.push_str(content);

                        // Send delta event
                        let delta_event = StreamEvent {
                            type_: "response.output_text.delta".to_string(),
                            response: None,
                            event_id: None,
                            response_id: None,
                            item_id: Some(message_id.clone()),
                            output_index: Some(0),
                            content_index: Some(0),
                            delta: Some(content.to_string()),
                            text: None,
                            item: None,
                            sequence_number: None,
                            call_id: None,
                            name: None,
                            arguments: None,
                            error: None,
                        };

                        dispatch_event(&tx, &mut sequencer, &response_id, &request_id, delta_event)
                            .await;
                    }
                    continue;
                }

                // Handle streaming delta
                if let Some(delta) = &choice.delta {
                    // Handle reasoning content (for reasoning models)
                    if let Some(reasoning) = &delta.reasoning_content {
                        if !reasoning.is_empty() {
                            accumulated_reasoning.push_str(reasoning);

                            // Start reasoning item if not started
                            if !reasoning_started {
                                reasoning_item_id = Some(reasoning_id_seed.clone());
                                reasoning_started = true;
                                log::info!(
                                    "🧠 Reasoning content detected, emitting reasoning events"
                                );
                            }

                            // Send reasoning delta event
                            let reasoning_delta_event = StreamEvent {
                                type_: "response.reasoning_text.delta".to_string(),
                                response: None,
                                event_id: None,
                                response_id: None,
                                item_id: reasoning_item_id.clone(),
                                output_index: Some(0),
                                content_index: Some(0),
                                delta: Some(reasoning.clone()),
                                text: None,
                                item: None,
                                sequence_number: None,
                                call_id: None,
                                name: None,
                                arguments: None,
                                error: None,
                            };

                            dispatch_event(
                                &tx,
                                &mut sequencer,
                                &response_id,
                                &request_id,
                                reasoning_delta_event,
                            )
                            .await;
                        }
                    }

                    // Handle regular text content
                    if let Some(content) = &delta.content {
                        if let Some(content_text) = extract_text_delta(content) {
                            if !content_text.is_empty() {
                                accumulated_text.push_str(&content_text);

                                // Check if we should start XML buffering
                                if !xml_buffering && content_text.contains("<function=") {
                                    xml_buffering = true;
                                    log::debug!(
                                        "🔍 Started XML buffering - detected <function= tag"
                                    );
                                }

                                // If buffering, check if we have the closing tag
                                if xml_buffering {
                                    // Check if we now have a complete XML tool call (has closing tag)
                                    if accumulated_text.contains("</tool_call>")
                                        || accumulated_text.contains("</function>")
                                    {
                                        log::debug!(
                                            "🔍 Found closing tag - extracting XML tool calls"
                                        );

                                        // Extract and convert XML to function calls
                                        let (cleaned, xml_calls) =
                                            extract_xml_tool_calls(&accumulated_text);

                                        if !xml_calls.is_empty() {
                                            log::warn!(
                                                "⚠️ Converted {} XML-style tool call(s) to proper function calls",
                                                xml_calls.len()
                                            );

                                            // Replace accumulated text with cleaned version
                                            accumulated_text = cleaned;

                                            // Convert each XML call to function call events
                                            for xml_call in xml_calls.into_iter() {
                                                // Find next available index to avoid collisions with native tool calls
                                                while tool_calls.contains_key(&next_xml_index) {
                                                    next_xml_index += 1;
                                                }
                                                let call_idx = next_xml_index;
                                                next_xml_index += 1;

                                                let call_id =
                                                    format!("call_xml_{}_{}", request_id, call_idx);
                                                let item_id = call_id.clone();

                                                let call_state = ToolCallState {
                                                    call_id: call_id.clone(),
                                                    item_id: item_id.clone(),
                                                    type_: "function".to_string(),
                                                    name: Some(xml_call.name.clone()),
                                                    arguments: xml_call.arguments.clone(),
                                                    item_added: true,
                                                    end_emitted: false,
                                                    pending_args: String::new(),
                                                };

                                                tool_calls.insert(call_idx, call_state.clone());

                                                let output_idx = (call_idx + 1) as u32;
                                                emit_tool_call_begin_events(
                                                    &tx,
                                                    &mut sequencer,
                                                    &response_id,
                                                    &request_id,
                                                    &item_id,
                                                    &call_id,
                                                    &xml_call.name,
                                                    output_idx,
                                                )
                                                .await;

                                                emit_tool_call_delta_events(
                                                    &tx,
                                                    &mut sequencer,
                                                    &response_id,
                                                    &request_id,
                                                    &item_id,
                                                    &call_id,
                                                    output_idx,
                                                    &xml_call.arguments,
                                                )
                                                .await;

                                                emit_tool_call_end_event(
                                                    &tx,
                                                    &mut sequencer,
                                                    &response_id,
                                                    &request_id,
                                                    &item_id,
                                                    &call_id,
                                                    output_idx,
                                                    &xml_call.name,
                                                    &xml_call.arguments,
                                                )
                                                .await;

                                                let args_done_event = StreamEvent {
                                                    type_: "response.function_call_arguments.done"
                                                        .to_string(),
                                                    response: None,
                                                    event_id: None,
                                                    response_id: None,
                                                    item_id: Some(item_id.clone()),
                                                    output_index: Some(output_idx),
                                                    content_index: None,
                                                    delta: None,
                                                    text: None,
                                                    item: None,
                                                    sequence_number: None,
                                                    call_id: Some(call_id.clone()),
                                                    name: Some(xml_call.name.clone()),
                                                    arguments: Some(xml_call.arguments.clone()),
                                                    error: None,
                                                };

                                                dispatch_event(
                                                    &tx,
                                                    &mut sequencer,
                                                    &response_id,
                                                    &request_id,
                                                    args_done_event,
                                                )
                                                .await;

                                                let call_done_event = StreamEvent {
                                                    type_: "response.output_item.done".to_string(),
                                                    response: None,
                                                    event_id: None,
                                                    response_id: None,
                                                    item_id: Some(item_id.clone()),
                                                    output_index: Some(output_idx),
                                                    content_index: None,
                                                    delta: None,
                                                    text: None,
                                                    item: Some(OutputItem {
                                                        id: item_id.clone(),
                                                        object: REALTIME_ITEM_OBJECT.to_string(),
                                                        type_: "function_call".to_string(),
                                                        status: "completed".to_string(),
                                                        role: None,
                                                        content: None,
                                                        call_id: Some(call_id.clone()),
                                                        name: Some(xml_call.name.clone()),
                                                        arguments: Some(xml_call.arguments.clone()),
                                                        output: None,
                                                    }),
                                                    sequence_number: None,
                                                    call_id: Some(call_id.clone()),
                                                    name: None,
                                                    arguments: None,
                                                    error: None,
                                                };

                                                dispatch_event(
                                                    &tx,
                                                    &mut sequencer,
                                                    &response_id,
                                                    &request_id,
                                                    call_done_event,
                                                )
                                                .await;

                                                if let Some(entry) = tool_calls.get_mut(&call_idx) {
                                                    entry.end_emitted = true;
                                                }

                                                log::info!(
                                                    "🔧 Converted XML tool: {}",
                                                    xml_call.name
                                                );
                                            }

                                            // Done buffering
                                            xml_buffering = false;

                                            // Skip emitting the XML as text since we converted it
                                            continue;
                                        } else {
                                            // Had closing tag but parser failed - fall through to emit
                                            log::warn!("Found closing tag but XML parser failed - emitting as text");
                                            xml_buffering = false;
                                        }
                                    } else {
                                        // No closing tag yet - keep buffering, don't emit anything
                                        log::debug!("🔍 Buffering XML ({} bytes) - waiting for </tool_call>", accumulated_text.len());
                                        continue;
                                    }
                                }

                                // Legacy path removed - buffering path above handles all XML conversion

                                // Only emit text delta if we have actual text content AND we're not buffering XML
                                if !content_text.is_empty() && !xml_buffering {
                                    let delta_str = content_text.clone();
                                    let delta_event = StreamEvent {
                                        type_: "response.output_text.delta".to_string(),
                                        response: None,
                                        event_id: None,
                                        response_id: None,
                                        item_id: Some(message_id.clone()),
                                        output_index: Some(0),
                                        content_index: Some(0),
                                        delta: Some(delta_str.clone()),
                                        text: None,
                                        item: None,
                                        sequence_number: None,
                                        call_id: None,
                                        name: None,
                                        arguments: None,
                                        error: None,
                                    };

                                    dispatch_event(
                                        &tx,
                                        &mut sequencer,
                                        &response_id,
                                        &request_id,
                                        delta_event,
                                    )
                                    .await;
                                }
                            }
                        } else {
                            log::debug!("⚠️ Unhandled content delta shape: {:?}", content);
                        }
                    }

                    // Handle tool_calls (function calling)
                    if let Some(tool_calls_delta) = &delta.tool_calls {
                        for tc in tool_calls_delta {
                            let call_state = tool_calls.entry(tc.index).or_insert_with(|| {
                                let fallback_id = format!("call_{}_{}", request_id, tc.index);
                                let call_id = tc.id.clone().unwrap_or_else(|| fallback_id.clone());
                                ToolCallState {
                                    call_id: call_id.clone(),
                                    item_id: call_id,
                                    type_: tc
                                        .type_
                                        .clone()
                                        .unwrap_or_else(|| "function".to_string()),
                                    name: None,
                                    arguments: String::new(),
                                    item_added: false,
                                    end_emitted: false,
                                    pending_args: String::new(),
                                }
                            });

                            // Update ID if provided
                            if let Some(ref id) = tc.id {
                                call_state.call_id = id.clone();
                                call_state.item_id = id.clone();
                            }

                            // Update type if provided
                            if let Some(ref type_) = tc.type_ {
                                call_state.type_ = type_.clone();
                            }

                            // Handle function call delta
                            if let Some(ref func) = tc.function {
                                // Update name if provided
                                if let Some(ref name) = func.name {
                                    call_state.name = Some(name.clone());

                                    // Send output_item.added when we first get the function name
                                    if !call_state.item_added {
                                        call_state.item_added = true;

                                        let output_idx = tc.index as u32 + 1; // +1 because message is at index 0

                                        let function_name =
                                            call_state.name.as_deref().unwrap_or("function_call");
                                        log::info!(
                                            "🔧 Tool call started: {} (index {})",
                                            function_name,
                                            tc.index
                                        );

                                        emit_tool_call_begin_events(
                                            &tx,
                                            &mut sequencer,
                                            &response_id,
                                            &request_id,
                                            &call_state.item_id,
                                            &call_state.call_id,
                                            function_name,
                                            output_idx,
                                        )
                                        .await;

                                        // If we buffered arguments before the name arrived, replay them now
                                        if !call_state.pending_args.is_empty() {
                                            log::info!(
                                                "🔧 Replaying {} buffered argument bytes for {}",
                                                call_state.pending_args.len(),
                                                function_name
                                            );

                                            emit_tool_call_delta_events(
                                                &tx,
                                                &mut sequencer,
                                                &response_id,
                                                &request_id,
                                                &call_state.item_id,
                                                &call_state.call_id,
                                                output_idx,
                                                &call_state.pending_args,
                                            )
                                            .await;

                                            // Move pending to arguments
                                            call_state.arguments.push_str(&call_state.pending_args);
                                            call_state.pending_args.clear();
                                        }
                                    }
                                }

                                // Update arguments if provided
                                if let Some(ref args) = func.arguments {
                                    if call_state.item_added {
                                        // Name already sent, emit delta immediately
                                        call_state.arguments.push_str(args);

                                        let output_idx = tc.index as u32 + 1;

                                        emit_tool_call_delta_events(
                                            &tx,
                                            &mut sequencer,
                                            &response_id,
                                            &request_id,
                                            &call_state.item_id,
                                            &call_state.call_id,
                                            output_idx,
                                            args,
                                        )
                                        .await;
                                    } else {
                                        // Name not yet received, buffer the arguments
                                        call_state.pending_args.push_str(args);
                                        log::debug!(
                                            "🔍 Buffering {} argument bytes for tool index {} (name not yet received)",
                                            args.len(),
                                            tc.index
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if done {
                break;
            }
        }

        // Send reasoning.done event if reasoning was emitted
        if reasoning_started {
            let reasoning_done_event = StreamEvent {
                type_: "response.reasoning_text.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: reasoning_item_id.clone(),
                output_index: Some(0),
                content_index: Some(0),
                delta: None,
                text: Some(accumulated_reasoning.clone()),
                item: None,
                sequence_number: None,
                call_id: None,
                name: None,
                arguments: None,
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                reasoning_done_event,
            )
            .await;

            log::info!(
                "🧠 Reasoning content complete ({} chars)",
                accumulated_reasoning.len()
            );
        }

        // Send output_text.done event only if we have text content
        if !accumulated_text.is_empty() {
            let text_done_event = StreamEvent {
                type_: "response.output_text.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: Some(message_id.clone()),
                output_index: Some(0),
                content_index: Some(0),
                delta: None,
                text: Some(accumulated_text.clone()),
                item: None,
                sequence_number: None,
                call_id: None,
                name: None,
                arguments: None,
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                text_done_event,
            )
            .await;

            // Send content_part.done event
            let content_done_event = StreamEvent {
                type_: "response.content_part.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: Some(message_id.clone()),
                output_index: Some(0),
                content_index: Some(0),
                delta: None,
                text: None,
                item: None,
                sequence_number: None,
                call_id: None,
                name: None,
                arguments: None,
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                content_done_event,
            )
            .await;
        }

        // Send output_item.done event for the message (only if we have text)
        if !accumulated_text.is_empty() {
            let item_done_event = StreamEvent {
                type_: "response.output_item.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: Some(message_id.clone()),
                output_index: Some(0),
                content_index: None,
                delta: None,
                text: None,
                item: Some(OutputItem {
                    id: message_id.clone(),
                    object: REALTIME_ITEM_OBJECT.to_string(),
                    type_: "message".to_string(),
                    status: "completed".to_string(),
                    role: Some("assistant".to_string()),
                    content: Some(vec![OutputContent::OutputText {
                        text: accumulated_text.clone(),
                        annotations: vec![],
                    }]),
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                }),
                sequence_number: None,
                call_id: None,
                name: None,
                arguments: None,
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                item_done_event,
            )
            .await;
        }

        // Collect and sort tool calls for processing
        let mut sorted_calls: Vec<_> = tool_calls.into_iter().collect();
        sorted_calls.sort_by_key(|(idx, _)| *idx);

        // Clone tool calls for later use in final response
        let sorted_calls_clone = sorted_calls.clone();

        // Send function_call_arguments.done and output_item.done for each tool call
        // Tool calls always start at index 1 (message is at index 0)
        for (idx, call_state) in sorted_calls {
            let output_idx = idx as u32 + 1;
            let function_name = call_state
                .name
                .clone()
                .unwrap_or_else(|| "function_call".to_string());

            if call_state.end_emitted {
                continue;
            }

            emit_tool_call_end_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                &call_state.item_id,
                &call_state.call_id,
                output_idx,
                &function_name,
                &call_state.arguments,
            )
            .await;

            // Send function_call_arguments.done
            let args_done_event = StreamEvent {
                type_: "response.function_call_arguments.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: Some(call_state.item_id.clone()),
                output_index: Some(output_idx),
                content_index: None,
                delta: None,
                text: None,
                item: None,
                sequence_number: None,
                call_id: Some(call_state.call_id.clone()),
                name: Some(function_name.clone()),
                arguments: Some(call_state.arguments.clone()),
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                args_done_event,
            )
            .await;

            log::info!(
                "🔧 Tool call complete: {} - {} bytes of args",
                function_name,
                call_state.arguments.len()
            );

            // Send output_item.done for the function call
            let call_done_event = StreamEvent {
                type_: "response.output_item.done".to_string(),
                response: None,
                event_id: None,
                response_id: None,
                item_id: Some(call_state.item_id.clone()),
                output_index: Some(output_idx),
                content_index: None,
                delta: None,
                text: None,
                item: Some(OutputItem {
                    id: call_state.item_id.clone(),
                    object: REALTIME_ITEM_OBJECT.to_string(),
                    type_: "function_call".to_string(),
                    status: "completed".to_string(),
                    role: None,
                    content: None,
                    call_id: Some(call_state.call_id.clone()),
                    name: Some(function_name.clone()),
                    arguments: Some(call_state.arguments.clone()),
                    output: None,
                }),
                sequence_number: None,
                call_id: Some(call_state.call_id.clone()),
                name: None,
                arguments: None,
                error: None,
            };

            dispatch_event(
                &tx,
                &mut sequencer,
                &response_id,
                &request_id,
                call_done_event,
            )
            .await;
        }

        // Send response.completed/done events
        let mut final_reasoning_state = req_reasoning_state.clone();
        if final_reasoning_state.is_none() && reasoning_started {
            final_reasoning_state = Some(ResponseReasoningState::default());
        }

        let mut output_items = vec![];

        // Add reasoning item if present
        if reasoning_started && !accumulated_reasoning.is_empty() {
            output_items.push(OutputItem {
                id: reasoning_item_id.unwrap_or_else(|| reasoning_id_seed.clone()),
                object: REALTIME_ITEM_OBJECT.to_string(),
                type_: "reasoning".to_string(),
                status: "completed".to_string(),
                role: Some("assistant".to_string()),
                content: Some(vec![OutputContent::Reasoning {
                    text: accumulated_reasoning.clone(),
                }]),
                call_id: None,
                name: None,
                arguments: None,
                output: None,
            });
        }

        // Add text message item (always include at index 0 for consistent indices)
        output_items.push(OutputItem {
            id: message_id.clone(),
            object: REALTIME_ITEM_OBJECT.to_string(),
            type_: "message".to_string(),
            status: "completed".to_string(),
            role: Some("assistant".to_string()),
            content: Some(vec![OutputContent::OutputText {
                text: accumulated_text.clone(),
                annotations: vec![],
            }]),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        });

        // Reconstruct the sorted tool calls for the final response
        let mut final_tool_calls: Vec<_> = sorted_calls_clone
            .iter()
            .map(|(_idx, call_state)| OutputItem {
                id: call_state.item_id.clone(),
                object: REALTIME_ITEM_OBJECT.to_string(),
                type_: "function_call".to_string(),
                status: "completed".to_string(),
                role: None,
                content: None,
                call_id: Some(call_state.call_id.clone()),
                name: call_state.name.clone(),
                arguments: Some(call_state.arguments.clone()),
                output: None,
            })
            .collect();

        // Add all tool calls to output
        output_items.append(&mut final_tool_calls);

        // Determine incomplete_details if status is incomplete
        let incomplete_details = if final_status == "incomplete" {
            Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            })
        } else {
            None
        };

        let final_response = Response {
            id: response_id.clone(),
            object: "response".to_string(),
            created_at,
            status: final_status.to_string(),
            error: None,
            incomplete_details,
            model: Some(model_for_response.to_string()),
            output: output_items,
            usage: Some(Usage {
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
                total_tokens: total_input_tokens + total_output_tokens,
                input_tokens_details: Some(TokenDetails {
                    cached_tokens: 0,
                    reasoning_tokens: 0,
                }),
                output_tokens_details: Some(TokenDetails {
                    cached_tokens: 0,
                    reasoning_tokens: 0,
                }),
            }),
            metadata: req_metadata.clone(),
            // Echo back request parameters
            instructions: req_instructions.clone(),
            tools: req_tools.clone(),
            tool_choice: req_tool_choice.clone(),
            parallel_tool_calls: req_parallel_tool_calls,
            temperature: req_temperature,
            top_p: req_top_p,
            max_output_tokens: req_max_output_tokens,
            store: req_store,
            previous_response_id: req_previous_response_id.clone(),
            reasoning: final_reasoning_state.clone(),
            background: req_background,
            max_tool_calls: req_max_tool_calls,
            text: req_text.clone(),
            prompt: req_prompt.clone(),
            truncation: req_truncation.clone(),
            conversation: req_conversation.clone(),
            top_logprobs: req_top_logprobs,
            user: req_user.clone(),
            safety_identifier: req_safety_identifier.clone(),
            prompt_cache_key: req_prompt_cache_key.clone(),
            service_tier: req_service_tier.clone(),
        };

        let completed_event = StreamEvent {
            type_: "response.completed".to_string(),
            event_id: None,
            response_id: None,
            response: Some(final_response.clone()),
            item_id: None,
            output_index: None,
            content_index: None,
            delta: None,
            text: None,
            item: None,
            sequence_number: None,
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };

        dispatch_event(
            &tx,
            &mut sequencer,
            &response_id,
            &request_id,
            completed_event,
        )
        .await;

        let done_event = StreamEvent {
            type_: "response.done".to_string(),
            event_id: None,
            response_id: None,
            response: Some(final_response),
            item_id: None,
            output_index: None,
            content_index: None,
            delta: None,
            text: None,
            item: None,
            sequence_number: None,
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };

        dispatch_event(&tx, &mut sequencer, &response_id, &request_id, done_event).await;

        log::debug!("🏁 Streaming task completed");

        // Record circuit breaker success
        let cb_clone = app.circuit_breaker.clone();
        tokio::spawn(async move {
            cb_clone.write().await.record_success();
        });

        // Log metrics
        if let Ok(elapsed) = request_start.elapsed() {
            log::info!(target: "metrics",
                "request_completed: model={}, duration_ms={}, status={}",
                backend_model_for_metrics, elapsed.as_millis(), final_status
            );
        }
    });

    let mut out_headers = HeaderMap::new();
    out_headers.insert("cache-control", "no-cache".parse().unwrap());
    out_headers.insert("connection", "keep-alive".parse().unwrap());
    out_headers.insert("x-accel-buffering", "no".parse().unwrap());
    out_headers.insert("content-type", "text/event-stream".parse().unwrap());

    let stream = ReceiverStream::new(rx).map(Ok::<Event, Infallible>);
    Ok((out_headers, Sse::new(stream)))
}

/// Estimate size of input content to prevent memory exhaustion
fn estimate_input_size(input: &crate::models::ResponseInput) -> usize {
    use crate::models::{ContentPart, ResponseContent, ResponseInput, ResponseInputItem};

    match input {
        ResponseInput::String(s) => s.len(),
        ResponseInput::Array(items) => items
            .iter()
            .map(|item| match item {
                ResponseInputItem::Message {
                    content,
                    role,
                    attachments,
                    ..
                } => {
                    let content_size = match content {
                        ResponseContent::String(s) => s.len(),
                        ResponseContent::Array(parts) => parts
                            .iter()
                            .map(|p| match p {
                                ContentPart::InputText { text }
                                | ContentPart::OutputText { text } => text.len(),
                                ContentPart::ToolOutput { body, .. } => body.len(),
                                ContentPart::InputImage { image_url } => image_url.url.len(),
                                ContentPart::InputFile {
                                    file_id,
                                    filename,
                                    file_url,
                                    file_data,
                                } => {
                                    file_id.as_ref().map(|s| s.len()).unwrap_or(0)
                                        + filename.as_ref().map(|s| s.len()).unwrap_or(0)
                                        + file_url.as_ref().map(|s| s.len()).unwrap_or(0)
                                        + file_data.as_ref().map(|s| s.len()).unwrap_or(0)
                                }
                                ContentPart::Reasoning {
                                    text,
                                    encrypted_content,
                                } => {
                                    text.len()
                                        + encrypted_content.as_ref().map(|e| e.len()).unwrap_or(0)
                                }
                                ContentPart::EncryptedContent { encrypted_content } => {
                                    encrypted_content.len()
                                }
                            })
                            .sum(),
                    };

                    let attachments_size = attachments
                        .as_ref()
                        .map(|list| {
                            list.iter()
                                .map(|att| {
                                    let tools_len: usize =
                                        att.tools.iter().map(|tool| tool.to_string().len()).sum();
                                    att.file_id.len() + tools_len
                                })
                                .sum()
                        })
                        .unwrap_or(0);

                    role.len() + content_size + attachments_size
                }
                ResponseInputItem::Reasoning {
                    text,
                    encrypted_content,
                } => {
                    text.as_ref().map(|t| t.len()).unwrap_or(0)
                        + encrypted_content.as_ref().map(|e| e.len()).unwrap_or(0)
                }
                ResponseInputItem::ItemReference { id } => id.len(),
                ResponseInputItem::AgentMessage {
                    author,
                    recipient,
                    content,
                } => {
                    let content_size = match content {
                        ResponseContent::String(s) => s.len(),
                        ResponseContent::Array(parts) => parts
                            .iter()
                            .map(|p| match p {
                                ContentPart::InputText { text }
                                | ContentPart::OutputText { text } => text.len(),
                                ContentPart::EncryptedContent { encrypted_content } => {
                                    encrypted_content.len()
                                }
                                _ => 0,
                            })
                            .sum(),
                    };
                    author.as_ref().map(|a| a.len()).unwrap_or(0)
                        + recipient.as_ref().map(|r| r.len()).unwrap_or(0)
                        + content_size
                }
                ResponseInputItem::FunctionCall {
                    call_id,
                    id,
                    name,
                    arguments,
                } => {
                    call_id.as_ref().map(|c| c.len()).unwrap_or(0)
                        + id.as_ref().map(|i| i.len()).unwrap_or(0)
                        + name.len()
                        + arguments.as_ref().map(|a| a.len()).unwrap_or(0)
                }
                ResponseInputItem::FunctionCallOutput {
                    call_id,
                    id,
                    output,
                    name,
                } => {
                    call_id.as_ref().map(|c| c.len()).unwrap_or(0)
                        + id.as_ref().map(|i| i.len()).unwrap_or(0)
                        + name.as_ref().map(|n| n.len()).unwrap_or(0)
                        + match output {
                            Value::String(s) => s.len(),
                            Value::Null => 0,
                            other => other.to_string().len(),
                        }
                }
            })
            .sum(),
    }
}

fn extract_text_delta(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => {
            let type_field = map.get("type").and_then(Value::as_str).unwrap_or("");
            if type_field == "text" || type_field == "output_text" {
                map.get("text")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            } else {
                None
            }
        }
        Value::Array(items) => {
            let mut combined = String::new();
            for item in items {
                if let Some(segment) = extract_text_delta(item) {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&segment);
                }
            }
            if combined.is_empty() {
                None
            } else {
                Some(combined)
            }
        }
        _ => None,
    }
}

/// Read error response body with size limit to prevent DoS
async fn read_bounded_error(res: reqwest::Response) -> String {
    let mut body = res.bytes_stream();
    let mut bytes = Vec::with_capacity(4096);
    let mut total = 0;

    while let Some(chunk_result) = body.next().await {
        if let Ok(chunk) = chunk_result {
            let remaining = MAX_ERROR_BODY_SIZE.saturating_sub(total);
            if remaining == 0 {
                log::warn!(
                    "⚠️  Error body exceeded {} bytes, truncating",
                    MAX_ERROR_BODY_SIZE
                );
                bytes.extend_from_slice(b"... (truncated)");
                break;
            }
            let to_take = chunk.len().min(remaining);
            bytes.extend_from_slice(&chunk[..to_take]);
            total += to_take;
        }
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Create an error response as a channel sender
fn send_error_response(
    tx: tokio::sync::mpsc::Sender<Event>,
    model: String,
    error_message: String,
    error_code: String,
) {
    tokio::spawn(async move {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let created_at = timestamp.as_secs();
        let response_id = format!("resp_{:x}", timestamp.as_nanos());

        let error_event = StreamEvent {
            type_: "response.failed".to_string(),
            event_id: Some(format!("evt_{response_id}_0001")),
            response_id: Some(response_id.clone()),
            response: Some(Response {
                id: response_id,
                object: "response".to_string(),
                created_at,
                status: "failed".to_string(),
                error: Some(crate::models::ResponseError {
                    code: error_code,
                    message: error_message,
                }),
                incomplete_details: None,
                model: Some(model),
                output: vec![],
                usage: None,
                metadata: None,
                instructions: None,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                store: Some(false),
                previous_response_id: None,
                reasoning: None,
                background: None,
                max_tool_calls: None,
                text: None,
                prompt: None,
                truncation: None,
                conversation: None,
                top_logprobs: None,
                user: None,
                safety_identifier: None,
                prompt_cache_key: None,
                service_tier: None,
            }),
            item_id: None,
            output_index: None,
            content_index: None,
            delta: None,
            text: None,
            item: None,
            sequence_number: Some(1),
            call_id: None,
            name: None,
            arguments: None,
            error: None,
        };

        if let Ok(json) = serde_json::to_string(&error_event) {
            dump_stream_event(&json, "error", 1);
            let _ = tx.send(Event::default().data(json)).await;
        }
    });
}
