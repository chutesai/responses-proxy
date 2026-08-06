use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- Request Models (OpenAI Responses API) ----------

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseInput {
    String(String),
    Array(Vec<ResponseInputItem>),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ResponseInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponseContent,
        #[serde(default)]
        tool_call_id: Option<String>,
        #[allow(dead_code)]
        #[serde(default)]
        metadata: Option<Value>,
        #[serde(default)]
        attachments: Option<Vec<MessageAttachment>>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    #[serde(rename = "item_reference")]
    ItemReference { id: String },
    #[serde(rename = "function_call")]
    FunctionCall {
        /// Responses API items carry both an item `id` (`fc_...`) and a `call_id`.
        /// Only `call_id` is meaningful for pairing with the output, but accept
        /// clients that only send `id`.
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        id: Option<String>,
        name: String,
        #[serde(default)]
        arguments: Option<String>,
    },
    /// An inter-agent message, as Codex's multi-agent runtime records it in a
    /// session's history.
    ///
    /// Codex writes one of these both ways: into a freshly spawned sub-agent's
    /// history (its task) and back into the parent's history (the child's final
    /// answer). Without this variant the tagged enum fails to deserialize and
    /// the whole request is rejected as `invalid_request_format` — which makes
    /// *every* sub-agent request fail, since a sub-agent's history begins with
    /// exactly this item.
    #[serde(rename = "agent_message")]
    AgentMessage {
        #[serde(default)]
        author: Option<String>,
        #[serde(default)]
        recipient: Option<String>,
        content: ResponseContent,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        id: Option<String>,
        /// Usually a string, but tolerate clients that send structured JSON.
        #[serde(default)]
        output: Value,
        /// Some clients echo the function name back on the output item.
        #[serde(default, alias = "tool")]
        name: Option<String>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseContent {
    String(String),
    Array(Vec<ContentPart>),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")] // Accept output_text in input (for multi-turn)
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: ImageUrl },
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(default)]
        file_id: Option<String>,
        #[serde(default)]
        filename: Option<String>,
        #[serde(default)]
        file_url: Option<String>,
        #[serde(default)]
        file_data: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        text: String,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    /// The body of an inter-agent message.
    ///
    /// Named "encrypted" because on OpenAI's own backend it is opaque to the
    /// client; against a plain OpenAI-compatible upstream it is the literal
    /// text, and dropping it would send a sub-agent its task with the task
    /// removed. Treated as text for exactly that reason.
    #[serde(rename = "encrypted_content")]
    EncryptedContent { encrypted_content: String },
    #[serde(rename = "output")]
    ToolOutput {
        #[allow(dead_code)]
        content_type: String,
        body: String,
    },
}

#[derive(Deserialize, Debug)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Deserialize, Debug)]
pub struct MessageAttachment {
    pub file_id: String,
    #[serde(default)]
    pub tools: Vec<Value>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum Tool {
    /// Responses API format: {type: "function", function: {name, description, parameters}}
    Nested {
        #[serde(rename = "type")]
        type_: String,
        function: FunctionDef,
    },
    /// Chat Completions format (also accepted by Responses API): {type: "function", name, description, parameters, strict}
    /// Also catches non-function tools like web_search, custom, etc.
    Flat {
        #[serde(rename = "type")]
        type_: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(default)]
        strict: bool,
        #[serde(flatten)]
        extra: Value,
    },
}

impl Tool {
    pub fn type_(&self) -> &str {
        match self {
            Tool::Nested { type_, .. } => type_,
            Tool::Flat { type_, .. } => type_,
        }
    }

    pub fn function_def(&self) -> FunctionDef {
        match self {
            Tool::Nested { function, .. } => function.clone(),
            Tool::Flat {
                name,
                description,
                parameters,
                type_,
                extra,
                ..
            } => {
                // For flat tools, use provided fields or fall back to type/extra
                let func_name = name
                    .clone()
                    .or_else(|| {
                        extra
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| type_.clone());

                let func_desc = description.clone().or_else(|| {
                    extra
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });

                let func_params = parameters
                    .clone()
                    .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

                FunctionDef {
                    name: func_name,
                    description: func_desc,
                    parameters: func_params,
                }
            }
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "auto", "none", "required"
    Specific(ToolChoiceSpecific),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ToolChoiceSpecific {
    #[serde(rename = "type")]
    pub type_: String, // "function"
    pub function: FunctionChoice,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FunctionChoice {
    pub name: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum ConversationReference {
    Id(String),
    Object { id: String },
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub generate_summary: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub verbosity: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ResponsePrompt {
    pub id: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub variables: Option<Value>,
}

#[derive(Serialize, Debug, Clone, Default)]
pub struct ResponseReasoningState {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

impl From<&ReasoningConfig> for ResponseReasoningState {
    fn from(config: &ReasoningConfig) -> Self {
        Self {
            effort: config.effort.clone(),
            summary: config.summary.clone(),
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct ResponseRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<ResponseInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<u32>, // Chat Completions compatibility (alias for max_output_tokens)
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub conversation: Option<ConversationReference>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    #[serde(default)]
    pub prompt: Option<ResponsePrompt>,
    #[serde(default)]
    pub truncation: Option<String>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub logprobs: Option<bool>, // Chat Completions compatibility
    #[serde(default)]
    pub n: Option<u32>, // Chat Completions: number of choices to generate
    #[serde(default)]
    pub max_completion_tokens: Option<u32>, // Chat Completions: newer max tokens field
    #[serde(default)]
    pub modalities: Option<Vec<String>>, // Chat Completions: output modalities
    #[serde(default)]
    pub prediction: Option<Value>, // Chat Completions: predicted output config
    #[serde(default)]
    pub reasoning_effort: Option<String>, // Chat Completions: reasoning effort level
    #[serde(default)]
    pub verbosity: Option<String>, // Chat Completions: response verbosity
    #[serde(default)]
    pub safety_identifier: Option<String>, // Chat Completions: safety tracking
    #[serde(default)]
    pub prompt_cache_key: Option<String>, // Chat Completions: cache optimization
    #[serde(default)]
    pub web_search_options: Option<Value>, // Chat Completions: web search config
    #[serde(default)]
    pub function_call: Option<Value>, // Chat Completions: deprecated, use tool_choice
    #[serde(default)]
    pub functions: Option<Vec<Value>>, // Chat Completions: deprecated, use tools
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<Value>>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub logit_bias: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

// ---------- Response Models (OpenAI Responses API) ----------

#[derive(Serialize, Debug, Clone)]
pub struct Response {
    pub id: String,
    pub object: String, // "response"
    pub created_at: u64,
    pub status: String, // "completed", "failed", "in_progress", "incomplete", "cancelled", "queued"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub output: Vec<OutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    // Echo back request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct IncompleteDetails {
    pub reason: String, // "max_output_tokens", "content_filter"
}

#[derive(Serialize, Debug, Clone)]
pub struct OutputItem {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    #[serde(rename = "type")]
    pub type_: String, // "message", "function_call", "function_call_output", "reasoning", "refusal"
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<OutputContent>>,
    // For function_call items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    // For function_call_output items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum OutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Value>,
    },
    #[serde(rename = "reasoning")]
    Reasoning { text: String },
}

#[derive(Serialize, Debug, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<TokenDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<TokenDetails>,
}

#[derive(Serialize, Debug, Clone)]
pub struct TokenDetails {
    pub cached_tokens: u32,
    pub reasoning_tokens: u32,
}

// ---------- Streaming Events ----------

#[derive(Serialize, Debug)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Response>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<OutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_number: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    // For function call events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    // For error events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}
