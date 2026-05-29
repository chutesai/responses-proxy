use axum::{
    routing::{get, post},
    Router,
};
use log::info;
use std::{env, sync::Arc, time::Duration};
use tokio::sync::RwLock;

// Import our modules
mod handlers;
mod models;
mod services;
mod utils;

use models::{App, CircuitBreakerState};
use services::refresh_models_cache;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let backend_url = env::var("BACKEND_URL")
        .unwrap_or_else(|_| "https://llm.chutes.ai/v1/chat/completions".into());
    let backend_host_header = env::var("BACKEND_HOST_HEADER").ok();
    let backend_timeout_secs = env::var("BACKEND_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(600);
    let log_volume_enabled = env::var("ENABLE_LOG_VOLUME")
        .ok()
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);

    info!("🚀 OpenAI Responses Proxy for Chutes.ai starting...");
    info!("   Backend URL: {}", backend_url);
    info!("   Backend Timeout: {}s", backend_timeout_secs);
    info!("   Circuit Breaker: enabled");
    info!(
        "   Log Volume: {}",
        if log_volume_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );

    // Initialize logging directory if enabled
    if log_volume_enabled {
        if let Err(e) = utils::init_log_dir() {
            log::warn!("⚠️  Failed to initialize logging directory: {}", e);
        }
    }

    let models_cache = Arc::new(RwLock::new(None));
    let circuit_breaker = Arc::new(RwLock::new(CircuitBreakerState::new(true)));

    let app = App {
        client: reqwest::Client::builder()
            .pool_max_idle_per_host(1024)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(backend_timeout_secs))
            .build()
            .unwrap(),
        backend_url: backend_url.clone(),
        backend_host_header: backend_host_header.clone(),
        models_cache: models_cache.clone(),
        circuit_breaker: circuit_breaker.clone(),
    };

    // Initial model cache load
    info!("🔄 Loading initial model cache...");
    if let Err(e) = refresh_models_cache(&app).await {
        log::warn!(
            "⚠️  Failed to load initial model cache: {}. Continuing anyway.",
            e
        );
    }

    // Background model cache refresh (every 60s) with graceful shutdown
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let cache_task = {
        let app_clone = app.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = refresh_models_cache(&app_clone).await {
                    log::warn!("Failed to refresh models cache: {}", e);
                }

                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {
                        // Continue loop
                    }
                    _ = shutdown_rx.recv() => {
                        info!("🛑 Model cache refresh task shutting down gracefully");
                        break;
                    }
                }
            }
        })
    };

    let router = Router::new()
        .route("/health", get(handlers::health_check))
        .route("/v1/responses", post(handlers::create_response))
        .layer(axum::extract::DefaultBodyLimit::max(10 * 1024 * 1024)) // 10MB limit
        .layer(tower_http::compression::CompressionLayer::new())
        .with_state(app);

    let port = env::var("HOST_PORT")
        .unwrap_or_else(|_| "8282".into())
        .parse::<u16>()
        .unwrap_or(8282);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    info!("   Listening on: 0.0.0.0:{}", port);

    // Graceful shutdown
    let server = axum::serve(listener, router).with_graceful_shutdown(async {
        tokio::signal::ctrl_c().await.ok();
        info!("🛑 Received shutdown signal, draining connections...");
    });

    if let Err(e) = server.await {
        log::error!("Server error: {}", e);
    }

    // Cleanup background tasks
    info!("🧹 Cleaning up background tasks...");
    let _ = shutdown_tx.send(()).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), cache_task).await;
    info!("✅ Shutdown complete");
}
