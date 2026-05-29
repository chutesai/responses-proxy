use crate::models::{App, ModelInfo};
use serde_json::Value;

/// Build `/v1/models` URL from backend chat completions URL.
fn models_url_from_backend_url(backend_url: &str) -> String {
    // best-effort: replace trailing `/v1/chat/completions` with `/v1/models`
    if let Some(idx) = backend_url.rfind("/v1/chat/completions") {
        let mut s = String::with_capacity(backend_url.len());
        s.push_str(&backend_url[..idx]);
        s.push_str("/v1/models");
        s
    } else {
        // fallback: assume same host, standard path
        format!("{}/../models", backend_url.trim_end_matches('/'))
    }
}

/// Refresh the models cache from backend
pub async fn refresh_models_cache(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let models_url = models_url_from_backend_url(&app.backend_url);
    log::info!("🔄 Fetching available models from {}", models_url);

    // Models endpoint is public (no auth required)
    let mut req = app.client.get(&models_url);
    if let Some(host) = &app.backend_host_header {
        req = req.header("host", host);
    }
    let res = req.send().await?;
    let status = res.status();
    if !status.is_success() {
        // Read error body for debugging
        let error_text = res.text().await.unwrap_or_else(|_| "".into());
        log::warn!(
            "❌ Models endpoint returned {} - response: {}",
            status,
            if error_text.len() > 200 {
                &error_text[..200]
            } else {
                &error_text
            }
        );
        return Err(format!("Models endpoint returned {}", status).into());
    }

    let data: Value = res.json().await?;
    let models: Vec<ModelInfo> = data["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m["id"].as_str()?.to_string();
                    let input_price = m["price"]["input"]["usd"]
                        .as_f64()
                        .or_else(|| m["pricing"]["prompt"].as_f64());
                    let output_price = m["price"]["output"]["usd"]
                        .as_f64()
                        .or_else(|| m["pricing"]["completion"].as_f64());
                    let supported_features = m["supported_features"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(ModelInfo {
                        id,
                        input_price_usd: input_price,
                        output_price_usd: output_price,
                        supported_features,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    log::info!("✅ Cached {} models from backend", models.len());
    let mut cache = app.models_cache.write().await;
    *cache = Some(models);
    Ok(())
}

/// Get cached models or fetch if not available
pub async fn get_available_models(app: &App) -> Vec<ModelInfo> {
    {
        let cache = app.models_cache.read().await;
        if let Some(models) = cache.as_ref() {
            return models.clone();
        }
    }
    if let Err(e) = refresh_models_cache(app).await {
        log::warn!("Failed to fetch models: {}", e);
        return vec![];
    }
    let cache = app.models_cache.read().await;
    cache.as_ref().cloned().unwrap_or_default()
}

/// Normalize model name with case-correction from cache
pub async fn normalize_model_name(model: &str, app: &App) -> String {
    let model_lower = model.to_lowercase();
    let cache = app.models_cache.read().await;
    if let Some(models) = cache.as_ref() {
        if models.iter().any(|m| m.id == model) {
            return model.to_string();
        }
        if let Some(matched) = models.iter().find(|m| m.id.to_lowercase() == model_lower) {
            log::info!("🔄 Model: {} → {} (case-corrected)", model, matched.id);
            return matched.id.clone();
        }
    }
    model.to_string()
}

/// Check if a model supports a specific feature from backend capability list
///
/// Example usage:
/// ```rust
/// // Auto-detect reasoning models
/// if model_supports_feature(&model, "thinking", &app).await {
///     log::info!("🧠 Model supports thinking/reasoning");
/// }
///
/// // Check for vision support
/// if model_supports_feature(&model, "vision", &app).await {
///     log::info!("👁️ Model supports image input");
/// }
/// ```
pub async fn model_supports_feature(model: &str, feature: &str, app: &App) -> bool {
    let cache = app.models_cache.read().await;
    if let Some(models) = cache.as_ref() {
        if let Some(model_info) = models.iter().find(|m| m.id.eq_ignore_ascii_case(model)) {
            return model_info
                .supported_features
                .iter()
                .any(|f| f.eq_ignore_ascii_case(feature));
        }
    }
    false
}
