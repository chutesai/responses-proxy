use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- Chat Completions Request (to Chutes.ai) ----------

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>, // String or Array for multimodal
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>, // Function name for tool role messages (required by some templates)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>, // For tool role messages
}

#[derive(Serialize, Debug)]
pub struct ChatFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum ChatTool {
    Function {
        #[serde(rename = "type")]
        type_: String,
        function: ChatFunction,
    },
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>, // String or Array
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>, // Map<token_id, bias>
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>, // Number of choices to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>, // Stream options for SSE
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>, // New field replacing max_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>, // Output modalities (text, audio)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Value>, // Predicted output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>, // Reasoning effort level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>, // Response verbosity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>, // Safety tracking identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>, // Cache optimization key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<Value>, // Web search configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Value>, // Deprecated: use tool_choice
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Value>>, // Deprecated: use tools
}

// ---------- Chat Completions Response (from Chutes.ai) ----------

#[derive(Deserialize, Debug, Default)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    #[serde(rename = "id")]
    pub _id: Option<String>,
    #[serde(default)]
    #[serde(rename = "object")]
    pub _object: Option<String>,
    #[serde(default)]
    #[serde(rename = "created")]
    pub _created: Option<i64>,
    #[serde(default)]
    #[serde(rename = "model")]
    pub _model: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Deserialize, Debug, Default)]
pub struct Choice {
    #[serde(default)]
    #[serde(rename = "index")]
    pub _index: usize,
    #[serde(default)]
    pub delta: Option<Delta>,
    #[serde(default)]
    pub message: Option<Value>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct Delta {
    #[serde(default)]
    #[serde(rename = "role")]
    pub _role: Option<String>,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    // Extended reasoning content (for reasoning models like DeepSeek-R1)
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub type_: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Deserialize, Debug)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    #[serde(rename = "total_tokens")]
    pub _total_tokens: Option<u32>,
    #[serde(default)]
    #[serde(rename = "prompt_tokens_details")]
    pub _prompt_tokens_details: Option<Value>,
    #[serde(default)]
    #[serde(rename = "completion_tokens_details")]
    pub _completion_tokens_details: Option<Value>,
}
