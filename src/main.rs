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
        version = %crate::version::version_string(),
        device = %config.device,
        models_dir = %config.models_dir.display(),
        max_body_mb = config.max_body_mb,
        max_concurrent_inference = config.max_concurrent_inference,
        "vision-ext 启动中"
    );

    tracing::info!("已配置模型能力: {:?}", config.configured_engines());

    let engines = engine::EngineManager::new(config.clone());
    let state = AppState::new(config, engines);

    let app = build_router(state, auth_token);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!("vision-ext 监听 http://0.0.0.0:{port}");

    // 优雅停机：SIGINT/SIGTERM 后停止接受新连接，等待在途请求（含长推理）完成；
    // 超时保底强退，防止个别悬挂连接导致进程无法退出。
    // 超时取 3s：浏览器 keep-alive 连接不会主动关闭，长超时只会拉长停机窗口
    //（共享机上被外部周期性终止时，60s 等待意味着每轮 1 分钟不可用）。
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
    match tokio::time::timeout(std::time::Duration::from_secs(3), server).await {
        Ok(result) => result?,
        Err(_) => tracing::error!("优雅停机超时（60s），强制退出"),
    }
    tracing::info!("vision-ext 已停止 ({})", crate::version::version_string());

    Ok(())
}

/// 等待终止信号（Ctrl+C 或 SIGTERM），触发 axum 优雅停机。
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let ctrl_c = tokio::signal::ctrl_c();
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("安装 SIGTERM 处理器失败")
                .recv()
                .await;
        };
        tokio::select! {
            _ = ctrl_c => tracing::info!("收到 SIGINT（Ctrl+C），开始优雅停机"),
            _ = terminate => tracing::info!("收到 SIGTERM，开始优雅停机"),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.expect("安装 Ctrl+C 处理器失败");
        tracing::info!("收到 SIGINT（Ctrl+C），开始优雅停机");
    }
}
