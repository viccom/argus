use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::auth_middleware;
use crate::config::Config;
use crate::engine::EngineManager;

/// 应用共享状态，通过 axum State 提取器注入到所有 handler。
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub engines: Arc<EngineManager>,
    /// 推理背压信号量：限制并发推理数，防止 spawn_blocking 堆积导致 runtime 饥饿。
    /// max_concurrent_inference=0 时容量为 usize::MAX（等效无限制）。
    pub inference_semaphore: Arc<tokio::sync::Semaphore>,
    pub started_at: std::time::Instant,
}

impl AppState {
    pub fn new(config: Config, engines: EngineManager) -> Self {
        let permits = if config.max_concurrent_inference > 0 {
            config.max_concurrent_inference
        } else {
            tokio::sync::Semaphore::MAX_PERMITS
        };
        Self {
            config: Arc::new(config),
            engines: Arc::new(engines),
            inference_semaphore: Arc::new(tokio::sync::Semaphore::new(permits)),
            started_at: std::time::Instant::now(),
        }
    }
}

/// /health 查询参数。deep=true 时额外返回 configured（已配置模型的能力清单）。
#[derive(Deserialize)]
pub struct HealthQuery {
    pub deep: Option<bool>,
}

/// GET /health[?deep=true]
///
/// 浅检返回已加载引擎（`engines`）；deep=true 追加:
/// - `configured`:配置了模型路径且在白名单内的能力清单(与 engines 对照可见
///   "配置了未加载";白名单外能力被过滤——它们永远不会加载,列出只会造成
///   "配了却没起来"的沉默困惑);
/// - `engine_health`:对每个已加载引擎发 dummy 推理,返回 ok/degraded 状态数组。
///   动机:arena OOM 后 cuBLAS 永久失败,浅检(端口活)不可见——僵尸态。
pub async fn health(
    State(state): State<AppState>,
    Query(query): Query<HealthQuery>,
) -> Json<Value> {
    let engines = state.engines.list_loaded();
    let mut body = json!({
        "status": "healthy",
        "device": state.config.device,
        "engines": engines,
        "version": crate::version::version_string(),
        "uptime_s": state.started_at.elapsed().as_secs(),
    });
    if query.deep.unwrap_or(false) {
        let configured: Vec<_> = state
            .config
            .configured_engines()
            .into_iter()
            .filter(|c| state.config.capability_enabled(c))
            .collect();
        body["configured"] = json!(configured);
        let health_results = state.engines.deep_health_check().await;
        body["engine_health"] = json!(health_results);
    }
    Json(body)
}

/// GET / — 服务信息端点：服务名、版本、端点列表。
///
/// 内容协商：浏览器地址栏访问（Accept 含 text/html）302 跳转 `/ui`（省去手动
/// 拼 URL）；curl/fetch 等非浏览器客户端（无 Accept 或 */*/application/json）
/// 保持 JSON——API 语义不变。
pub async fn service_info(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    let accept = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if accept.contains("text/html") {
        return axum::response::Redirect::to("/ui").into_response();
    }
    Json(json!({
        "service": "argus",
        "version": crate::version::version_string(),
        "uptime_s": state.started_at.elapsed().as_secs(),
        "endpoints": {
            "GET /health": "健康检查",
            "GET /models": "能力键控的模型清单（configured/loaded/models，每能力实际使用的模型文件名）",
            "POST /detect": "目标检测（multipart: file/sahi/slice_w/slice_h/overlap/confidence?/target_classes?/roi?）",
            "POST /detect/batch": "批量检测（JSON: image_paths[服务端本地路径]/confidence?/target_classes?/roi?，逐图独立背压与错误隔离）",
            "POST /classify": "图像分类（multipart: file/top_k?）",
            "POST /detect/obb": "OBB 旋转框检测（multipart: file）",
            "POST /face/detect": "人脸检测（multipart: file）",
            "POST /face/recognize": "人脸识别（multipart: file/known_embeddings）",
            "POST /ocr": "OCR 文字识别（multipart: file）",
            "POST /license-plate": "车牌识别（multipart: file）",
            "POST /qr": "二维码识别（multipart: file）",
            "POST /quality": "图像质量评估（multipart: file）",
            "POST /pose": "姿态估计（multipart: file）",
            "POST /depth": "深度估计（multipart: file）",
            "POST /human-parsing": "人体解析（multipart: file）"
        }
    }))
    .into_response()
}

/// 构建应用路由（提取自 main 以便测试）。
pub fn build_router(state: AppState, auth_state: Option<Arc<String>>) -> axum::Router {
    use axum::middleware::from_fn_with_state;
    use axum::routing::{get, post};
    use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
    use tower_http::trace::TraceLayer;
    use tower_http::cors::CorsLayer;

    let max_body = state.config.max_body_mb * 1024 * 1024;
    axum::Router::new()
        .route("/", get(service_info))
        .route("/health", get(health))
        .route("/models", get(crate::routes::models::list_models))
        .route("/ui", get(crate::routes::ui::ui))
        .route("/detect", post(crate::routes::detection::detect))
        .route("/detect/batch", post(crate::routes::batch::detect_batch))
        .route("/detect/obb", post(crate::routes::detection::detect_obb))
        .route("/segmentation", post(crate::routes::segmentation::segmentation))
        .route("/ppe", post(crate::routes::scene::ppe_detect))
        .route("/fire-smoke", post(crate::routes::scene::fire_smoke_detect))
        .route("/classify", post(crate::routes::classification::classify))
        .route("/face/detect", post(crate::routes::face::face_detect))
        .route("/face/recognize", post(crate::routes::face::face_recognize))
        .route("/reid", post(crate::routes::reid::reid_embed))
        .route("/ocr", post(crate::routes::text::ocr))
        .route("/license-plate", post(crate::routes::text::license_plate))
        .route("/qr", post(crate::routes::text::qr))
        .route("/quality", post(crate::routes::analysis::quality))
        .route("/pose", post(crate::routes::analysis::pose))
        .route("/depth", post(crate::routes::analysis::depth))
        .route("/human-parsing", post(crate::routes::analysis::human_parsing))
        .layer(axum::extract::DefaultBodyLimit::max(max_body))
        .layer(from_fn_with_state(auth_state, auth_middleware))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<_>| {
            let request_id = req
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            tracing::info_span!(
                "http",
                request_id = %request_id,
                method = %req.method(),
                uri = %req.uri(),
            )
        })
        .on_response(|resp: &axum::http::Response<_>, latency: std::time::Duration, _span: &tracing::Span| {
            tracing::info!(
                status = resp.status().as_u16(),
                latency_ms = latency.as_millis() as u64,
                "请求完成"
            );
        }))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let config = Config::default();
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn body_text(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    #[tokio::test]
    async fn root_should_return_service_info_and_list_endpoints() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("argus"), "应包含服务名: {text}");
        assert!(text.contains("/detect"), "应列出端点: {text}");
        assert!(text.contains("/face"), "应列出人脸端点: {text}");
        assert!(text.contains("/ocr"), "应列出 OCR 端点: {text}");
    }

    #[tokio::test]
    async fn health_shallow_should_return_healthy() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("healthy"), "浅检应返回 healthy: {text}");
        assert!(text.contains("uptime_s"), "应含 uptime: {text}");
        assert!(text.contains("version"), "应含 version: {text}");
    }

    #[tokio::test]
    async fn health_should_list_loaded_engines() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("engines"), "应含 engines 字段: {text}");
    }

    #[tokio::test]
    async fn health_deep_should_list_configured_engines() {
        let config = Config {
            detection_model: Some(std::path::PathBuf::from("/models/yolov8n.onnx")),
            ..Default::default()
        };
        let mgr = EngineManager::new(config.clone());
        let app = build_router(AppState::new(config, mgr), None);
        let resp = app
            .oneshot(Request::get("/health?deep=true").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("configured"), "deep 检应含 configured 字段: {text}");
        assert!(text.contains("detection"), "configured 应列出已配置的 detection: {text}");
    }

    #[tokio::test]
    async fn health_shallow_should_not_include_configured() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(!text.contains("configured"), "浅检不应含 configured 字段: {text}");
    }

    #[tokio::test]
    async fn service_info_should_advertise_detect_filter_params() {
        // confidence/target_classes/roi 已是 /detect 生效参数(后过滤),接口描述须宣传;
        // 历史上 confidence 曾被解析但未生效,当时的反向断言随实现落地翻转
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        for p in ["confidence", "target_classes", "roi", "/detect/batch"] {
            assert!(text.contains(p), "service_info 应宣传 {p}: {text}");
        }
    }

    #[tokio::test]
    async fn ui_should_serve_html_even_when_auth_configured() {
        // 浏览器无法为页面加载本身携带 API key,控制台页面须豁免鉴权;
        // 页面内的 API 调用仍需 key(页面提供输入框)
        let app = build_router(test_state(), Some(Arc::new("secret".into())));
        let resp = app
            .oneshot(Request::get("/ui").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "/ui 应可访问");
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        assert!(ct.contains("text/html"), "应为 HTML: {ct}");
        let text = body_text(resp).await;
        assert!(text.contains("argus"), "页面应含服务标识: {text}");
    }

    #[tokio::test]
    async fn root_should_redirect_browsers_to_ui() {
        // 浏览器地址栏访问 Accept 含 text/html → 302 /ui
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::get("/")
                    .header("accept", "text/html,application/xhtml+xml,*/*;q=0.8")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // axum Redirect::to = 303 See Other(浏览器 GET 重定向标准语义,等价 302)
        assert_eq!(resp.status(), StatusCode::SEE_OTHER, "浏览器访问 / 应重定向到 /ui");
        assert_eq!(
            resp.headers().get("location").and_then(|v| v.to_str().ok()),
            Some("/ui")
        );
    }

    #[tokio::test]
    async fn root_should_keep_json_for_non_browser_clients() {
        // curl/fetch(无 Accept 或非 html)保持 JSON API 语义
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::get("/")
                    .header("accept", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = body_text(resp).await;
        assert!(text.contains("argus"), "非浏览器应返回 service_info JSON: {text}");
    }

    #[tokio::test]
    async fn response_should_carry_request_id_header() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(
            resp.headers().contains_key("x-request-id"),
            "响应缺少 x-request-id 头"
        );
    }

    #[tokio::test]
    async fn auth_should_reject_when_token_configured_but_missing() {
        let app = build_router(test_state(), Some(Arc::new("secret".into())));
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("缺少"), "应拒绝无认证请求: {text}");
    }
}
