mod auth;
mod config;
mod engine;
mod error;
mod routes;
mod version;

use std::sync::Arc;

use routes::health::{build_router, AppState};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let config = config::Config::from_env();
    let port = config.port;
    let auth_token = config.auth_token.clone().map(Arc::new);

    tracing::info!(
        device = %config.device,
        models_dir = %config.models_dir.display(),
        max_body_mb = config.max_body_mb,
        max_concurrent_inference = config.max_concurrent_inference,
        "vision-ext 启动中"
    );

    if config.detection_model.is_some() {
        tracing::info!("已配置 detection 模型");
    }
    if config.face_det_model.is_some() {
        tracing::info!("已配置 face_detection 模型");
    }
    if config.ocr_det_model.is_some() {
        tracing::info!("已配置 ocr 模型");
    }

    let engines = engine::EngineManager::new(config.clone());
    let state = AppState::new(config, engines);

    let app = build_router(state, auth_token);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!("vision-ext 监听 http://0.0.0.0:{port}");
    axum::serve(listener, app).await?;

    Ok(())
}
