use log::warn;
use reqwest::Client;
use std::{sync::Arc, time::SystemTime};
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub input_price_usd: Option<f64>,
    pub output_price_usd: Option<f64>,
    pub supported_features: Vec<String>,
}

// ---------- App with cached models and circuit breaker ----------

#[derive(Clone)]
pub struct App {
    pub client: Client,
    pub backend_url: String,
    pub backend_host_header: Option<String>,
    pub models_cache: Arc<RwLock<Option<Vec<ModelInfo>>>>,
    pub circuit_breaker: Arc<RwLock<CircuitBreakerState>>,
}

// ---------- Circuit breaker state ----------

#[derive(Clone, Debug)]
pub struct CircuitBreakerState {
    pub consecutive_failures: u32,
    pub last_failure_time: Option<SystemTime>,
    pub is_open: bool,
    pub enabled: bool,
}

impl CircuitBreakerState {
    pub fn new(enabled: bool) -> Self {
        Self {
            consecutive_failures: 0,
            last_failure_time: None,
            is_open: false,
            enabled,
        }
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.is_open = false;
        self.last_failure_time = None;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure_time = Some(SystemTime::now());
        if self.consecutive_failures >= 5 {
            self.is_open = true;
            warn!(
                "🔴 Circuit breaker opened after {} consecutive failures",
                self.consecutive_failures
            );
        }
    }

    pub fn should_allow_request(&mut self) -> bool {
        if !self.enabled {
            return true;
        }
        if !self.is_open {
            return true;
        }
        // Try to recover after 30 seconds
        if let Some(last_fail) = self.last_failure_time {
            if let Ok(elapsed) = SystemTime::now().duration_since(last_fail) {
                if elapsed.as_secs() >= 30 {
                    log::info!("🟡 Circuit breaker attempting half-open state");
                    self.is_open = false;
                    self.consecutive_failures = 0;
                    return true;
                }
            }
        }
        false
    }
}
