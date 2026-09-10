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

    // GPU 显存上限:必须在创建任何 ONNX Session 之前设置(进程级 override,
    // 各引擎 OnnxRuntimeConfig::defaults() 自动读取)。
    // CUDA arena 只增不减:不设上限曾与 vLLM 共存膨胀至 81GB 打死服务。
    if config.gpu_mem_limit_mb > 0 {
        rust_onnx_infer::set_global_gpu_mem_limit(config.gpu_mem_limit_mb);
        tracing::info!(
            gpu_mem_limit_mb = config.gpu_mem_limit_mb,
            "GPU 显存上限已设置(CUDA arena 防雪崩)"
        );
    }

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
    let shutdown_timeout = std::time::Duration::from_secs(config.shutdown_timeout_secs);
    let state = AppState::new(config, engines);

    let app = build_router(state, auth_token);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!("vision-ext 监听 http://0.0.0.0:{port}");

    // 优雅停机：SIGINT/SIGTERM 后停止接受新连接，等待在途请求（含长推理）完成。
    // keep-alive 连接不会主动关闭，axum drain 可能永远不完成——超时上限只从
    // **收到停机信号后**起算（曾误写成 timeout(上限, server)：超时从进程启动
    // 倒计时，服务每上限秒无信号自杀且 exit 0，长期被误判为外部周期性终止，
    // 见 tests/lifecycle.rs 回归防护）。上限可配（默认 10s）：需覆盖最长推理
    // （SAHI/depth ~1.5s，池化排队翻倍），过大只会拉长停机窗口。
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(());
    });
    tokio::select! {
        result = server => result?,
        _ = async move {
            let _ = shutdown_rx.await;                  // 信号已到，axum 进入 drain
            tokio::time::sleep(shutdown_timeout).await; // drain 收尾上限
        } => {
            tracing::error!(
                "优雅停机超时（{}s），强制退出",
                shutdown_timeout.as_secs()
            );
        }
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
