use crate::models::{
    ChatCompletionRequest, ChatFunction, ChatMessage, ChatTool, ContentPart, ResponseContent,
    ResponseInput, ResponseInputItem, ResponseRequest,
};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};

/// Convert OpenAI Responses API request to Chat Completions format
pub fn convert_to_chat_completions(
    req: &ResponseRequest,
    supports_native_tools: bool,
) -> Result<ChatCompletionRequest, String> {
    let model = req.model.as_ref().ok_or("Model is required")?.clone();

    let mut messages = Vec::new();

    // Prepare tool overrides
    let native_tool_override = "\n\n---\n\nIMPORTANT: Tool Calling Format Override\n\
When calling functions/tools, you MUST use the standard OpenAI Chat Completions JSON format, NOT any XML or custom syntax. \
The system will automatically handle tool execution. Never output tool calls as text - use the native function calling mechanism.";

    let xml_tool_override = "\n\n---\n\nIMPORTANT: Tool Calling Format Override\n\
To call a function, you MUST use the following XML format:\n\
<function=function_name>\n\
<parameter=param_name>value</parameter>\n\
...\n\
</function>\n\
\n\
Do not use JSON tool calls. Use the XML format above.";

    let file_ops_guidance = "\n\nFile Operation Best Practices:\n\
- Use relative paths (e.g. 'test.py', 'src/main.rs') for files in the workspace\n\
- Read each file ONCE before editing - do not re-read files you've already successfully read\n\
- After receiving file contents from read_file, proceed directly to editing without redundant reads\n\
- For apply_patch, include 3-5 lines of surrounding context for reliable matching\n\
- Never announce \"I will read the file\" after you've already read it - just use the content you received";

    // Determine which instructions to use
    let mut system_instructions = req.instructions.clone().unwrap_or_default();

    // Only append overrides if tools are actually present or requested
    if req.tools.is_some() {
        if supports_native_tools {
            system_instructions.push_str(native_tool_override);
        } else {
            system_instructions.push_str(xml_tool_override);
        }
        // Append general guidance
        system_instructions.push_str(file_ops_guidance);
    }

    // Add instructions as system message if not empty
    if !system_instructions.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(json!(system_instructions)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Passthrough messages if provided (hybrid Chat Completions compatibility)
    // This allows advanced users to send pre-formatted messages while using the Responses endpoint
    if let Some(req_messages) = &req.messages {
        log::debug!(
            "📨 Processing {} pre-formatted messages (hybrid mode)",
            req_messages.len()
        );
        for msg in req_messages {
            if let Ok(chat_msg) = serde_json::from_value::<ChatMessage>(msg.clone()) {
                messages.push(chat_msg);
            }
        }
    }

    // Convert input to messages
    if let Some(input) = &req.input {
        match input {
            ResponseInput::String(text) => {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some(json!(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            ResponseInput::Array(items) => {
                let mut accumulated_reasoning: Vec<String> = Vec::new();
                let mut pending_tool_calls: Vec<Value> = Vec::new();
                // call_id -> function name, so `tool` messages can carry `name`
                let mut tool_names_by_call_id: HashMap<String, String> = HashMap::new();
                // FIFO of (call_id, name) for outputs that don't reference a known call_id
                let mut unmatched_calls: VecDeque<(String, String)> = VecDeque::new();
                let mut synthetic_call_seq: usize = 0;

                for item in items {
                    match item {
                        ResponseInputItem::Message {
                            role,
                            content,
                            tool_call_id,
                            attachments,
                            ..
                        } => {
                            if let Some(attached) = attachments {
                                if !attached.is_empty() {
                                    let file_ids: Vec<_> =
                                        attached.iter().map(|a| a.file_id.as_str()).collect();
                                    log::error!(
                                        "❌ Attachments are not supported in stateless mode (files: {:?})",
                                        file_ids
                                    );
                                    return Err("attachments_not_supported".to_string());
                                }
                            }

                            if role == "tool" {
                                let call_id = tool_call_id.clone().ok_or_else(|| {
                                    log::error!("❌ Tool role message missing tool_call_id");
                                    "tool_message_missing_tool_call_id".to_string()
                                })?;

                                let tool_payload = extract_tool_message_body(content)?;

                                // A tool result closes the assistant turn that issued the
                                // call, so emit the assistant `tool_calls` message first.
                                flush_pending_tool_calls(
                                    &mut messages,
                                    &mut pending_tool_calls,
                                    &mut accumulated_reasoning,
                                );

                                let (call_id, name) = resolve_tool_call(
                                    Some(&call_id),
                                    None,
                                    &mut tool_names_by_call_id,
                                    &mut unmatched_calls,
                                );

                                messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: Some(json!(tool_payload)),
                                    name,
                                    tool_calls: None,
                                    tool_call_id: call_id,
                                });

                                continue;
                            }

                            // Any non-assistant message ends the assistant turn as well.
                            if role != "assistant" {
                                flush_pending_tool_calls(
                                    &mut messages,
                                    &mut pending_tool_calls,
                                    &mut accumulated_reasoning,
                                );
                            }

                            let (mut msg_content, content_reasoning) =
                                convert_response_content(content)?;

                            // If content has inline reasoning, accumulate it
                            if let Some(content_think) = content_reasoning {
                                accumulated_reasoning.push(content_think);
                            }

                            // If assistant message and we have accumulated reasoning, prepend as <think> tags
                            if role == "assistant" && !accumulated_reasoning.is_empty() {
                                let thinking_text = accumulated_reasoning.join("\n");
                                let original_content = msg_content.as_str().unwrap_or("");
                                let combined = format!(
                                    "<think>{}</think>\n{}",
                                    thinking_text, original_content
                                );
                                msg_content = json!(combined);
                                log::info!("🧠 INPUT: Prepended {} reasoning part(s) ({} chars) to assistant message as <think> tags", 
                                    accumulated_reasoning.len(), thinking_text.len());
                                accumulated_reasoning.clear();
                            }

                            // If assistant message and we have pending tool calls, add them to the message
                            if role == "assistant" && !pending_tool_calls.is_empty() {
                                log::info!(
                                    "🔧 Added {} tool call(s) to assistant message",
                                    pending_tool_calls.len()
                                );
                                messages.push(ChatMessage {
                                    role: role.clone(),
                                    content: Some(msg_content),
                                    name: None,
                                    tool_calls: Some(pending_tool_calls.clone()),
                                    tool_call_id: None,
                                });
                                pending_tool_calls.clear();
                            } else {
                                messages.push(ChatMessage {
                                    role: role.clone(),
                                    content: Some(msg_content),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                });
                            }
                        }
                        // An inter-agent message. Chat Completions has no such
                        // role, so it becomes a `user` message: it is always
                        // information arriving from *another* agent — a task
                        // handed down, or a child's answer handed back — never
                        // something this model said.
                        ResponseInputItem::AgentMessage {
                            author,
                            recipient,
                            content,
                        } => {
                            // Like any non-assistant message, it closes the
                            // assistant turn that preceded it.
                            flush_pending_tool_calls(
                                &mut messages,
                                &mut pending_tool_calls,
                                &mut accumulated_reasoning,
                            );

                            let (msg_content, content_reasoning) =
                                convert_response_content(content)?;
                            if let Some(content_think) = content_reasoning {
                                accumulated_reasoning.push(content_think);
                            }
                            log::info!(
                                "🤝 INPUT: agent_message {} -> {}",
                                author.as_deref().unwrap_or("?"),
                                recipient.as_deref().unwrap_or("?")
                            );
                            messages.push(ChatMessage {
                                role: "user".to_string(),
                                content: Some(msg_content),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                        ResponseInputItem::FunctionCall {
                            call_id,
                            id,
                            name,
                            arguments,
                        } => {
                            let call_id = call_id.clone().or_else(|| id.clone()).unwrap_or_else(|| {
                                synthetic_call_seq += 1;
                                let generated = format!("call_{}_{}", name, synthetic_call_seq);
                                log::warn!(
                                    "⚠️  function_call for '{}' has no call_id/id - generating '{}'",
                                    name,
                                    generated
                                );
                                generated
                            });

                            let arguments = arguments
                                .clone()
                                .filter(|a| !a.trim().is_empty())
                                .unwrap_or_else(|| "{}".to_string());

                            // Accumulate tool calls; they are flushed into a single assistant
                            // message as soon as the turn ends (tool output, other message, or
                            // end of input).
                            pending_tool_calls.push(json!({
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                }
                            }));
                            tool_names_by_call_id.insert(call_id.clone(), name.clone());
                            unmatched_calls.push_back((call_id.clone(), name.clone()));
                            log::info!(
                                "🔧 INPUT: Found function_call {} ({}) - will emit assistant tool_calls message",
                                name,
                                call_id
                            );
                        }
                        ResponseInputItem::FunctionCallOutput {
                            call_id,
                            id,
                            output,
                            name,
                        } => {
                            // The output may be a plain string, a JSON string carrying Codex's
                            // {"output":"...","metadata":{...}} envelope, or structured JSON.
                            let content_str = function_output_to_string(output);

                            // Emit the assistant `tool_calls` message that this output answers
                            // BEFORE the tool message, otherwise the backend sees a `tool`
                            // message with nothing to match against.
                            flush_pending_tool_calls(
                                &mut messages,
                                &mut pending_tool_calls,
                                &mut accumulated_reasoning,
                            );

                            let (resolved_call_id, resolved_name) = resolve_tool_call(
                                call_id.as_deref().or(id.as_deref()),
                                name.as_deref(),
                                &mut tool_names_by_call_id,
                                &mut unmatched_calls,
                            );

                            if resolved_name.is_none() {
                                log::warn!(
                                    "⚠️  function_call_output (call_id: {:?}) has no resolvable function name",
                                    resolved_call_id
                                );
                            }

                            log::info!(
                                "🔧 INPUT: Added function_call_output (call_id: {:?}, name: {:?}, {} bytes)",
                                resolved_call_id,
                                resolved_name,
                                content_str.len()
                            );

                            messages.push(ChatMessage {
                                role: "tool".to_string(),
                                content: Some(json!(content_str)),
                                name: resolved_name,
                                tool_calls: None,
                                tool_call_id: resolved_call_id,
                            });
                        }
                        ResponseInputItem::Reasoning {
                            text,
                            encrypted_content,
                        } => {
                            // Accumulate reasoning to prepend to next assistant message
                            if let Some(reasoning_text) = text {
                                accumulated_reasoning.push(reasoning_text.clone());
                                log::info!("🧠 INPUT: Found reasoning item ({} chars), will prepend to next assistant message", reasoning_text.len());
                            } else if encrypted_content.is_some() {
                                log::warn!("⚠️  Encrypted reasoning content not supported (stateless mode), skipping");
                            }
                        }
                        ResponseInputItem::ItemReference { id } => {
                            log::warn!("⚠️  Item references (id: {}) are not supported in stateless mode, skipping", id);
                        }
                    }
                }

                // Trailing tool calls (an assistant turn with no tool result yet) still need
                // their assistant message, otherwise they are silently dropped.
                if !pending_tool_calls.is_empty() {
                    log::warn!(
                        "⚠️  {} trailing tool call(s) with no function_call_output - emitting assistant message anyway",
                        pending_tool_calls.len()
                    );
                    flush_pending_tool_calls(
                        &mut messages,
                        &mut pending_tool_calls,
                        &mut accumulated_reasoning,
                    );
                }

                // If reasoning items remain without an assistant message, log warning
                if !accumulated_reasoning.is_empty() {
                    log::warn!("⚠️  {} reasoning item(s) found but no following assistant message to attach to", accumulated_reasoning.len());
                }
            }
        }
    }

    let response_format = req
        .text
        .as_ref()
        .and_then(|t| t.format.clone())
        .or_else(|| req.response_format.clone());

    // Handle logprobs - support both Responses API (top_logprobs) and Chat Completions (logprobs + top_logprobs)
    let (logprobs, top_logprobs) = match (req.logprobs, req.top_logprobs) {
        (_, Some(0)) => {
            log::warn!("⚠️ top_logprobs=0 requested - ignoring logprob request");
            (None, None)
        }
        (Some(true), tl) => (Some(true), tl.or(Some(5))), // Default to 5 if logprobs=true but no top_logprobs
        (_, Some(value)) => (Some(true), Some(value)),
        (None, None) => (None, None),
        (Some(false), None) => (None, None),
    };

    // Convert tools if provided - ONLY function tools are supported
    // Simply forward tools from the client; no injection needed
    let tools = if let Some(tools_vec) = req.tools.as_ref() {
        // Filter to only function tools; others are not supported in Chat Completions API
        let non_function_tools: Vec<_> = tools_vec
            .iter()
            .filter(|t| t.type_() != "function")
            .map(|t| t.type_())
            .collect();

        if !non_function_tools.is_empty() {
            log::debug!(
                "⚠️ Skipping non-function tools (not supported by Chat Completions API): {}",
                non_function_tools.join(", ")
            );
        }

        tools_vec
            .iter()
            .filter_map(|t| {
                if t.type_() == "function" {
                    let f = t.function_def();
                    Some(ChatTool::Function {
                        type_: "function".to_string(),
                        function: ChatFunction {
                            name: f.name.clone(),
                            description: f.description.clone(),
                            parameters: f.parameters.clone(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // Tool injection has been removed. Codex CLI now properly sends all tools
    // (read_file, list_dir, grep_files, etc.) when experimental_supported_tools
    // is configured in the model family. The proxy simply forwards whatever
    // tools the client provides.

    let tools = if tools.is_empty() { None } else { Some(tools) };

    // Convert tool_choice to Value for backend
    let tool_choice = req.tool_choice.as_ref().map(|tc| {
        use crate::models::ToolChoice;
        match tc {
            ToolChoice::String(s) => json!(s),
            ToolChoice::Specific(spec) => json!(spec),
        }
    });

    Ok(ChatCompletionRequest {
        model,
        messages,
        max_tokens: req.max_output_tokens.or(req.max_tokens), // Support both field names
        temperature: req.temperature,
        top_p: req.top_p,
        response_format,
        tools,
        tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
        user: req.user.clone(),
        logprobs,
        top_logprobs,
        stream: true, // Always stream — the Responses API always returns SSE
        stop: req.stop.clone(),
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        seed: req.seed,
        logit_bias: req.logit_bias.clone(),
        metadata: req.metadata.clone(),
        service_tier: req.service_tier.clone(),
        store: req.store,
        n: req.n,
        stream_options: Some(req
            .stream_options
            .as_ref()
            .map(|so| serde_json::to_value(so).unwrap_or(json!({})))
            .unwrap_or(json!({"include_usage": true}))),
        max_completion_tokens: req.max_completion_tokens,
        modalities: req.modalities.clone(),
        prediction: req.prediction.clone(),
        reasoning_effort: req.reasoning_effort.clone(),
        verbosity: req.verbosity.clone(),
        safety_identifier: req.safety_identifier.clone(),
        prompt_cache_key: req.prompt_cache_key.clone(),
        web_search_options: req.web_search_options.clone(),
        function_call: req.function_call.clone(),
        functions: req.functions.clone(),
    })
}

/// Emit the pending `function_call` items as a single assistant `tool_calls` message.
///
/// In the Responses API a `function_call` item *is* the assistant turn — there is no
/// separate assistant message that owns it. Chat Completions, however, requires an
/// `{"role":"assistant","tool_calls":[...]}` message before any `{"role":"tool"}`
/// message; without it backends (e.g. Kimi K3) reject the request because the tool
/// message has no call to match against.
///
/// Consecutive `function_call` items are merged into one assistant message, matching
/// the OpenAI schema for parallel tool calls.
fn flush_pending_tool_calls(
    messages: &mut Vec<ChatMessage>,
    pending_tool_calls: &mut Vec<Value>,
    accumulated_reasoning: &mut Vec<String>,
) {
    if pending_tool_calls.is_empty() {
        return;
    }

    let tool_calls = std::mem::take(pending_tool_calls);
    let count = tool_calls.len();

    let reasoning = if accumulated_reasoning.is_empty() {
        None
    } else {
        let joined = accumulated_reasoning.join("\n");
        accumulated_reasoning.clear();
        Some(joined)
    };

    // If the previous message is a plain assistant message, attach the calls to it
    // instead of emitting two assistant messages back to back.
    if let Some(last) = messages.last_mut() {
        if last.role == "assistant" && last.tool_calls.is_none() {
            if let Some(think) = &reasoning {
                let existing = match last.content.as_ref() {
                    Some(Value::String(s)) => s.clone(),
                    None | Some(Value::Null) => String::new(),
                    // Non-string (multimodal) content: leave it alone.
                    Some(_) => {
                        log::warn!("⚠️  Dropping reasoning for assistant message with structured content");
                        String::new()
                    }
                };
                if matches!(last.content, Some(Value::String(_)) | None | Some(Value::Null)) {
                    last.content = Some(json!(format!("<think>{}</think>\n{}", think, existing)));
                }
            }
            last.tool_calls = Some(tool_calls);
            log::info!(
                "🔧 INPUT: Attached {} tool call(s) to preceding assistant message",
                count
            );
            return;
        }
    }

    let content = match &reasoning {
        Some(think) => Some(json!(format!("<think>{}</think>", think))),
        // Explicit null (not omitted): matches what the Chat Completions API expects
        // for an assistant message that only makes tool calls.
        None => Some(Value::Null),
    };

    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content,
        name: None,
        tool_calls: Some(tool_calls),
        tool_call_id: None,
    });
    log::info!(
        "🔧 INPUT: Emitted assistant message with {} tool call(s)",
        count
    );
}

/// Resolve the `(tool_call_id, name)` pair for a tool result.
///
/// Priority: the `call_id` recorded from a preceding `function_call` item, then a
/// name the client put on the output item itself, then the oldest unanswered call
/// (positional matching, mirroring what lenient backends do).
fn resolve_tool_call(
    call_id: Option<&str>,
    explicit_name: Option<&str>,
    tool_names_by_call_id: &mut HashMap<String, String>,
    unmatched_calls: &mut VecDeque<(String, String)>,
) -> (Option<String>, Option<String>) {
    if let Some(call_id) = call_id.filter(|c| !c.is_empty()) {
        if let Some(name) = tool_names_by_call_id.remove(call_id) {
            if let Some(pos) = unmatched_calls.iter().position(|(id, _)| id == call_id) {
                unmatched_calls.remove(pos);
            }
            return (Some(call_id.to_string()), Some(name));
        }

        let name = explicit_name.map(|n| n.to_string()).or_else(|| {
            unmatched_calls
                .pop_front()
                .map(|(_, name)| name)
        });
        return (Some(call_id.to_string()), name);
    }

    // No usable call_id: fall back to the oldest unanswered call entirely.
    match unmatched_calls.pop_front() {
        Some((id, name)) => {
            tool_names_by_call_id.remove(&id);
            (Some(id), Some(name))
        }
        None => (None, explicit_name.map(|n| n.to_string())),
    }
}

/// Flatten a `function_call_output.output` value into the string Chat Completions wants.
///
/// Handles: plain strings, Codex's `{"output":"...","metadata":{...}}` envelope (sent as
/// a JSON *string*), and structured JSON sent directly. JSON that is not an object with
/// an `output` key is preserved verbatim so tool results that legitimately are JSON
/// documents survive the round trip.
fn function_output_to_string(output: &Value) -> String {
    match output {
        Value::Null => String::new(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => extract_output_envelope(&parsed).unwrap_or_else(|| s.clone()),
            Err(_) => s.clone(),
        },
        other => extract_output_envelope(other).unwrap_or_else(|| other.to_string()),
    }
}

/// Pull the inner `output` field out of a Codex-style envelope, if that's what this is.
fn extract_output_envelope(value: &Value) -> Option<String> {
    let obj = value.as_object()?;
    let inner = obj.get("output")?;
    Some(match inner {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

/// Convert ResponseContent to JSON value for Chat Completions
/// Returns (content_value, extracted_reasoning_text)
fn convert_response_content(content: &ResponseContent) -> Result<(Value, Option<String>), String> {
    match content {
        ResponseContent::String(text) => Ok((json!(text), None)),
        ResponseContent::Array(parts) => {
            let mut reasoning_text = String::new();
            let mut converted: Vec<Value> = Vec::new();

            for part in parts {
                match part {
                    ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                        converted.push(json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::ToolOutput { body, .. } => {
                        converted.push(json!({
                            "type": "text",
                            "text": body
                        }));
                    }
                    ContentPart::EncryptedContent { encrypted_content } => {
                        converted.push(json!({
                            "type": "text",
                            "text": encrypted_content
                        }));
                    }
                    ContentPart::InputImage { image_url } => {
                        converted.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url.url
                            }
                        }));
                    }
                    ContentPart::InputFile { .. } => {
                        return Err("input_file_content_not_supported".to_string());
                    }
                    ContentPart::Reasoning { text, .. } => {
                        // Reasoning within message content - accumulate for <think> tags
                        if !reasoning_text.is_empty() {
                            reasoning_text.push('\n');
                        }
                        reasoning_text.push_str(text);
                        log::info!(
                            "🧠 INPUT: Found reasoning in message content ({} chars)",
                            text.len()
                        );
                    }
                }
            }

            // If all text parts (no images), concatenate into string
            let has_images = parts
                .iter()
                .any(|p| matches!(p, ContentPart::InputImage { .. }));
            let has_reasoning = !reasoning_text.is_empty();

            if !has_images && !converted.is_empty() {
                let text: String = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                            Some(text.as_str())
                        }
                        ContentPart::ToolOutput { body, .. } => Some(body.as_str()),
                        ContentPart::EncryptedContent { encrypted_content } => {
                            Some(encrypted_content.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok((
                    json!(text),
                    if has_reasoning {
                        Some(reasoning_text)
                    } else {
                        None
                    },
                ))
            } else {
                Ok((
                    json!(converted),
                    if has_reasoning {
                        Some(reasoning_text)
                    } else {
                        None
                    },
                ))
            }
        }
    }
}

/// Extract tool role content into a plain string suitable for Chat Completions
fn extract_tool_message_body(content: &ResponseContent) -> Result<String, String> {
    match content {
        ResponseContent::String(text) => Ok(text.clone()),
        ResponseContent::Array(parts) => {
            let mut combined = String::new();

            for part in parts {
                match part {
                    ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                        if !combined.is_empty() {
                            combined.push('\n');
                        }
                        combined.push_str(text);
                    }
                    ContentPart::ToolOutput { body, .. } => {
                        if !combined.is_empty() {
                            combined.push('\n');
                        }
                        combined.push_str(body);
                    }
                    other => {
                        log::error!(
                            "❌ Tool message content part not supported in proxy: {:?}",
                            other
                        );
                        return Err("tool_output_content_not_supported".to_string());
                    }
                }
            }

            if combined.is_empty() {
                Err("tool_output_empty".to_string())
            } else {
                Ok(combined)
            }
        }
    }
}

/// Translate Chat Completions finish_reason to Responses API status
pub fn translate_finish_reason(finish_reason: Option<&str>) -> &'static str {
    match finish_reason {
        Some("stop") => "completed",
        Some("length") => "incomplete",
        Some("content_filter") => "failed",
        Some("tool_calls") => "completed",
        Some(_) => "completed",
        None => "in_progress",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Convert a raw Responses API request body and return the generated chat
    /// messages, minus the injected system prompt (which is asserted separately).
    fn messages_for(body: Value) -> Vec<Value> {
        let req: ResponseRequest =
            serde_json::from_value(body).expect("request body should deserialize");
        let chat = convert_to_chat_completions(&req, true).expect("conversion should succeed");
        chat.messages
            .iter()
            .map(|m| serde_json::to_value(m).expect("message should serialize"))
            .filter(|m| m["role"] != "system")
            .collect()
    }

    fn python_tools() -> Value {
        json!([{
            "type": "function",
            "name": "python",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }])
    }

    /// The exact request that used to fail with
    /// "Kimi K3 tool messages need a resolvable tool name".
    #[test]
    fn second_turn_emits_assistant_tool_calls_before_tool_message() {
        let messages = messages_for(json!({
            "model": "moonshotai/Kimi-K3-TEE",
            "instructions": "You are an agent.",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Run print(1+1), then tell me the answer."}]},
                {"type": "function_call", "call_id": "python:0", "name": "python",
                 "arguments": "{\"code\":\"print(1+1)\"}"},
                {"type": "function_call_output", "call_id": "python:0", "output": "2"}
            ],
            "tools": python_tools(),
            "max_output_tokens": 800,
            "stream": true
        }));

        assert_eq!(messages.len(), 3, "expected user + assistant + tool: {messages:#?}");
        assert_eq!(messages[0]["role"], "user");

        // The assistant tool_calls message is what was missing entirely.
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], Value::Null);
        let calls = messages[1]["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "python:0");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "python");
        assert_eq!(calls[0]["function"]["arguments"], "{\"code\":\"print(1+1)\"}");

        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "python:0");
        assert_eq!(messages[2]["name"], "python", "tool message must carry the function name");
        assert_eq!(messages[2]["content"], "2");
    }

    #[test]
    fn parallel_tool_calls_merge_into_one_assistant_message() {
        let messages = messages_for(json!({
            "model": "moonshotai/Kimi-K3-TEE",
            "input": [
                {"type": "message", "role": "user", "content": "list and read"},
                {"type": "function_call", "call_id": "c1", "name": "list_dir", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "read_file", "arguments": "{\"p\":\"a\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a\nb"},
                {"type": "function_call_output", "call_id": "c2", "output": "hello"}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages.len(), 4, "{messages:#?}");
        assert_eq!(messages[1]["role"], "assistant");
        let calls = messages[1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2, "consecutive function_calls must merge");
        assert_eq!(calls[0]["id"], "c1");
        assert_eq!(calls[1]["id"], "c2");

        assert_eq!(messages[2]["tool_call_id"], "c1");
        assert_eq!(messages[2]["name"], "list_dir");
        assert_eq!(messages[3]["tool_call_id"], "c2");
        assert_eq!(messages[3]["name"], "read_file");
    }

    #[test]
    fn interleaved_turns_produce_alternating_assistant_and_tool_messages() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "1"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "2"}
            ],
            "tools": python_tools()
        }));

        let roles: Vec<&str> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "assistant", "tool"]);
        assert_eq!(messages[1]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(messages[3]["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn reasoning_item_before_function_call_lands_on_that_assistant_message() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "reasoning", "text": "I should run the code."},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "2"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "The answer is 2."}]}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            "<think>I should run the code.</think>",
            "reasoning must attach to the tool-calling assistant turn, not leak to a later one"
        );
        assert!(messages[1]["tool_calls"].is_array());
        // The later assistant message must NOT inherit the tool calls.
        assert_eq!(messages[3]["role"], "assistant");
        assert_eq!(messages[3]["content"], "The answer is 2.");
        assert!(messages[3]["tool_calls"].is_null());
    }

    #[test]
    fn assistant_text_then_function_call_merges_into_a_single_message() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Let me run that."}]},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "2"}
            ],
            "tools": python_tools()
        }));

        let roles: Vec<&str> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
        assert_eq!(messages[1]["content"], "Let me run that.");
        assert_eq!(messages[1]["tool_calls"].as_array().unwrap()[0]["id"], "c1");
    }

    #[test]
    fn codex_output_envelope_is_unwrapped_but_plain_json_is_preserved() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1",
                 "output": "{\"output\":\"total 4\\n\",\"metadata\":{\"exit_code\":0}}"},
                {"type": "function_call", "call_id": "c2", "name": "api", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "{\"rows\":[1,2,3]}"},
                {"type": "function_call", "call_id": "c3", "name": "api", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c3", "output": {"structured": true}}
            ],
            "tools": python_tools()
        }));

        let tool_messages: Vec<&Value> =
            messages.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_messages.len(), 3);
        // Codex envelope: inner output extracted
        assert_eq!(tool_messages[0]["content"], "total 4\n");
        // A JSON document that is not an envelope must survive verbatim
        assert_eq!(tool_messages[1]["content"], "{\"rows\":[1,2,3]}");
        // Structured (non-string) output is stringified rather than rejected
        assert_eq!(tool_messages[2]["content"], "{\"structured\":true}");
    }

    #[test]
    fn tool_role_message_form_also_gets_a_name() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"},
                {"type": "message", "role": "tool", "tool_call_id": "c1",
                 "content": [{"type": "input_text", "text": "2"}]}
            ],
            "tools": python_tools()
        }));

        let roles: Vec<&str> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
        assert_eq!(messages[2]["name"], "python");
        assert_eq!(messages[2]["tool_call_id"], "c1");
    }

    #[test]
    fn output_with_unknown_call_id_falls_back_to_positional_match() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "mismatched", "output": "2"}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["name"], "python");
    }

    #[test]
    fn function_call_without_call_id_still_pairs_with_its_output() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "id": "fc_123", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "fc_123", "output": "2"}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages[1]["tool_calls"].as_array().unwrap()[0]["id"], "fc_123");
        assert_eq!(messages[2]["tool_call_id"], "fc_123");
        assert_eq!(messages[2]["name"], "python");
    }

    #[test]
    fn trailing_function_call_is_not_dropped() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"].as_array().unwrap()[0]["id"], "c1");
    }

    #[test]
    fn pending_tool_call_is_flushed_before_a_following_user_message() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "2"},
                {"type": "message", "role": "user", "content": "and again"},
                {"type": "function_call", "call_id": "c2", "name": "python", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "4"}
            ],
            "tools": python_tools()
        }));

        let roles: Vec<&str> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool", "user", "assistant", "tool"]
        );
    }

    #[test]
    fn plain_conversation_without_tools_is_unchanged() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "bye"}]}
            ]
        }));

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["content"], "hi");
        assert_eq!(messages[1]["content"], "hello");
        assert!(messages[0]["name"].is_null(), "name must not be emitted for plain messages");
    }

    #[test]
    fn empty_arguments_become_an_empty_json_object() {
        let messages = messages_for(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "now"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"}
            ],
            "tools": python_tools()
        }));

        assert_eq!(
            messages[1]["tool_calls"].as_array().unwrap()[0]["function"]["arguments"],
            "{}"
        );
    }

    // ------------------------------------------------- inter-agent messages

    /// The exact first request a spawned sub-agent makes, copied from a real
    /// chutescoder rollout. Before `ResponseInputItem::AgentMessage` existed
    /// this failed to deserialize and the proxy answered `422
    /// invalid_request_format`, so *every* sub-agent died on its first turn.
    #[test]
    fn a_sub_agents_task_message_converts_instead_of_being_rejected() {
        let messages = messages_for(json!({
            "model": "moonshotai/Kimi-K3-TEE",
            "input": [{
                "type": "agent_message",
                "id": "amsg_019fd6f4-077c-7a50-b796-663fa222ffe7",
                "author": "/root",
                "recipient": "/root/counter",
                "content": [
                    {"type": "input_text",
                     "text": "Message Type: NEW_TASK\nTask name: /root/counter\nSender: /root\nPayload:\n"},
                    {"type": "encrypted_content",
                     "encrypted_content": "Count the .py files. Reply with ONLY the number."}
                ]
            }],
            "tools": python_tools()
        }));

        assert_eq!(messages.len(), 1, "{messages:#?}");
        assert_eq!(messages[0]["role"], "user");
        let text = messages[0]["content"].as_str().expect("flattened to text");
        // Both halves must survive: the header says who is asking, and the
        // body *is* the task. Dropping the body would hand the sub-agent an
        // empty assignment, which is worse than an error.
        assert!(text.contains("Sender: /root"), "{text}");
        assert!(text.contains("Count the .py files"), "{text}");
    }

    /// The mirror image: the child's answer, injected back into the parent's
    /// history by the completion watcher.
    #[test]
    fn a_childs_final_answer_converts_in_the_parents_history() {
        let messages = messages_for(json!({
            "model": "moonshotai/Kimi-K3-TEE",
            "input": [
                {"type": "message", "role": "user", "content": "delegate the count"},
                {"type": "agent_message", "author": "/root/counter", "recipient": "/root",
                 "content": [{"type": "input_text",
                              "text": "Message Type: FINAL_ANSWER\nSender: /root/counter\nPayload:\n5"}]}
            ],
            "tools": python_tools()
        }));

        assert_eq!(messages.len(), 2, "{messages:#?}");
        assert_eq!(messages[1]["role"], "user");
        assert!(
            messages[1]["content"].as_str().unwrap().contains("5"),
            "{messages:#?}"
        );
    }

    /// An agent message closes the assistant turn before it, exactly like any
    /// other non-assistant message — otherwise the pending `tool_calls` would
    /// be emitted *after* it and Kimi would reject the ordering.
    #[test]
    fn an_agent_message_flushes_pending_tool_calls_first() {
        let messages = messages_for(json!({
            "model": "moonshotai/Kimi-K3-TEE",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "python",
                 "arguments": "{\"code\":\"1\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "1"},
                {"type": "agent_message", "author": "/root/kid", "recipient": "/root",
                 "content": [{"type": "input_text", "text": "done"}]}
            ],
            "tools": python_tools()
        }));

        let roles: Vec<_> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "user"], "{messages:#?}");
    }
}
