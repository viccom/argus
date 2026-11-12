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
    /// 推理背压观测计数（B5）：in_flight/queued_total/rejected_total/acquire_timeout_total。
    pub metrics: Arc<crate::routes::common::InferenceMetrics>,
    /// 进程内指标导出（D3）：disabled 时全部记录方法 no-op；main 在
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` 非空时替换为启用实例。
    pub otel: crate::otel_metrics::OtelMetrics,
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
            metrics: Arc::new(crate::routes::common::InferenceMetrics::default()),
            otel: crate::otel_metrics::OtelMetrics::disabled(),
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
    use std::sync::atomic::Ordering::Relaxed;
    let m = &state.metrics;
    let mut body = json!({
        "status": "healthy",
        "device": state.config.device,
        "engines": engines,
        "version": crate::version::version_string(),
        "uptime_s": state.started_at.elapsed().as_secs(),
        "backpressure": {
            "max_concurrent": state.config.max_concurrent_inference,
            "in_flight": m.in_flight.load(Relaxed),
            "queued_total": m.queued_total.load(Relaxed),
            "rejected_total": m.rejected_total.load(Relaxed),
            "acquire_timeout_total": m.acquire_timeout_total.load(Relaxed),
        },
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
        // 动态槽深检字段（设计 C3：engines/engine_health 现有字段不动，
        // 只在 deep 分支追加——浅检信封零变化）
        body["dynamic_engines"] = json!(state.engines.dynamic_loaded());
        body["dynamic_slot_health"] = json!(state.engines.dynamic_slot_health().await);
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
            "POST /detect": "目标检测（multipart: file/sahi/slice_w/slice_h/overlap/confidence?/target_classes?/roi?/model?[动态槽名]）",
            "POST /detect/batch": "批量检测（JSON: image_paths[服务端本地路径]/confidence?/target_classes?/roi?/model?[动态槽名]，逐图独立背压与错误隔离）",
            "POST /classify": "图像分类（multipart: file/top_k?）",
            "POST /detect/obb": "OBB 旋转框检测（multipart: file）",
            "POST /face/detect": "人脸检测（multipart: file）",
            "POST /face/recognize": "人脸识别（multipart: file/known_embeddings?/threshold?/min_face_px?/max_faces?）",
            "POST /face/recognize/batch": "批量人脸识别（multipart: 多 file×≤16/共享参数），单图失败带 error 不炸整批",
            "POST /ocr": "OCR 文字识别（multipart: file）",
            "POST /license-plate": "车牌识别（multipart: file）",
            "POST /qr": "二维码识别（multipart: file）",
            "POST /quality": "图像质量评估（multipart: file）",
            "POST /pose": "姿态估计（multipart: file）",
            "POST /depth": "深度估计（multipart: file）",
            "POST /human-parsing": "人体解析（multipart: file）",
            "POST /embed/text": "文本 Embedding（JSON: texts[1..128]/mode?/normalize?，返回归一化向量+dim+pipeline）",
            "POST /rerank": "文本重排（JSON: query/texts[1..32 默认，ARGUS_TEXT_RERANKING_MAX_TEXTS 可调 1..=64]，返回 scores 输入序+ranking 降序，预算 B×max_pair_len≤8192）"
        }
    }))
    .into_response()
}

/// 探活/信息类路径:其"请求完成"日志按 DEBUG 记(04-log-spec §5 噪声治理)。
///
/// 实测噪声占比:2026-11-10 全天 4317 行里 2482 行是探活(57.5%)。
/// 判定只看 path(不含 query)——`/health?deep=true` 同样是探活。
const NOISE_PATHS: [&str; 4] = ["/", "/health", "/api/health", "/status"];

/// 该 path 是否属于探活/噪声路径。
fn is_noise_path(path: &str) -> bool {
    NOISE_PATHS.contains(&path)
}

/// `x-request-id` 来件形态校验：8–64 位 `[A-Za-z0-9_-]`（与 safebot-go 的
/// `internal/infra/requestid` 同规则）。合法值由 `SetRequestIdLayer` 原样采纳；
/// 非法值必须**在进入该层之前**从请求头移除——tower-http 0.6 的 `SetRequestId`
/// 只在头缺失时才调用 `MakeRequestId`，头存在就原样采纳
/// （`tower-http-0.6.11/src/request_id.rs:331-340`）。否则超长/带空格的值会被
/// 原样写进日志 span 与响应头：未认证请求即可把 8KB 头放大成 16KB 日志行。
fn is_valid_request_id(s: &str) -> bool {
    (8..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// 在 `SetRequestIdLayer` 外层剔除非法的 `x-request-id`，随后的 SetRequestId
/// 因头缺失而生成 UUID v4；合法来件仍被原样采纳（下游贯穿语义不变）。
async fn sanitize_request_id(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let invalid = req
        .headers()
        .get("x-request-id")
        .is_some_and(|v| v.to_str().map(|s| !is_valid_request_id(s)).unwrap_or(true));
    if invalid {
        req.headers_mut().remove("x-request-id");
    }
    next.run(req).await
}

/// HTTP 入站指标中间件(D3,obs-platform 05 §4.5):记 requests_total/duration_ms;
/// argus 错误信封恒 HTTP 200,失败经 `ErrorMetric` 响应扩展标记在此关联
/// MatchedPath 路由模板落 `argus.inference.errors_total`。挂在最外层
/// (SetRequestIdLayer 之后 = 全部分层的最外圈),404 未命中路由无 MatchedPath 不计。
async fn http_metrics_middleware(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let endpoint = req
        .extensions()
        .get::<axum::extract::MatchedPath>()
        .map(|p| p.as_str().to_owned());
    let t0 = std::time::Instant::now();
    let resp = next.run(req).await;
    if let Some(endpoint) = endpoint {
        state.otel.observe_http(&endpoint, t0.elapsed().as_millis());
        if let Some(marker) = resp.extensions().get::<crate::error::ErrorMetric>() {
            state.otel.record_inference_error(&endpoint, marker.0);
        }
    }
    resp
}

/// 构建应用路由（提取自 main 以便测试）。
pub fn build_router(state: AppState, auth_state: Option<Arc<String>>) -> axum::Router {
    use axum::middleware::{from_fn, from_fn_with_state};
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
        .route("/face/recognize/batch", post(crate::routes::face::face_recognize_batch))
        .route("/reid", post(crate::routes::reid::reid_embed))
        .route("/embed/text", post(crate::routes::embed::embed_text))
        .route("/rerank", post(crate::routes::rerank::rerank_text))
        .route("/ocr", post(crate::routes::text::ocr))
        .route("/license-plate", post(crate::routes::text::license_plate))
        .route("/qr", post(crate::routes::text::qr))
        .route("/quality", post(crate::routes::analysis::quality))
        .route("/pose", post(crate::routes::analysis::pose))
        .route("/pose-rules", post(crate::routes::pose_rules::pose_rules))
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
            // 探活请求的 span 单独命名:on_response 靠 span 名判断该记 INFO 还是 DEBUG。
            // span 保持 INFO 级别(而非 debug_span!)——RUST_LOG=info 时 debug span 会
            // 被完全禁用,metadata() 变 None,级别判断就失效了。
            if is_noise_path(req.uri().path()) {
                tracing::info_span!(
                    "http_noise",
                    request_id = %request_id,
                    method = %req.method(),
                    uri = %req.uri(),
                )
            } else {
                tracing::info_span!(
                    "http",
                    request_id = %request_id,
                    method = %req.method(),
                    uri = %req.uri(),
                )
            }
        })
        .on_response(|resp: &axum::http::Response<_>, latency: std::time::Duration, span: &tracing::Span| {
            let status = resp.status().as_u16();
            let latency_ms = latency.as_millis() as u64;
            // tracing 的 event! 级别必须编译期常量(callsite 是静态),故分两支。
            if span.metadata().is_some_and(|m| m.name() == "http_noise") {
                tracing::event!(target: "http", tracing::Level::DEBUG, status, latency_ms, "请求完成");
            } else {
                tracing::event!(target: "http", tracing::Level::INFO, status, latency_ms, "请求完成");
            }
        }))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        // 紧贴 SetRequestId 的外层:非法来件 ID 先剔除(头缺失 → SetRequestId 自生成);
        // 合法来件仍由 SetRequestId 原样沿用(04 §4 贯穿语义不变)
        .layer(from_fn(sanitize_request_id))
        // 最外层:HTTP 指标(D3)——看得到全部内层(含 AppError 扩展标记)与最终响应
        .layer(from_fn_with_state(
            state.clone(),
            http_metrics_middleware,
        ))
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

    #[tokio::test]
    async fn health_should_report_backpressure_counters() {
        // B5：客户端要能区分"排队饱和"与"拒绝风暴"，浅检必须带计数
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        for field in ["backpressure", "in_flight", "queued_total", "rejected_total", "acquire_timeout_total"] {
            assert!(text.contains(field), "浅检应含 {field}: {text}");
        }
    }

    // ---- F0 golden：/health 形状锁定（动态化只允许 deep 增字段） ----

    #[tokio::test]
    async fn health_shallow_envelope_should_be_exact_golden() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let mut body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        // volatile 字段替换哨兵后整体等值
        body["version"] = json!("X");
        body["uptime_s"] = json!(0);
        body["backpressure"] = json!({
            "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
            "rejected_total": 0, "acquire_timeout_total": 0,
        });
        assert_eq!(
            body,
            json!({
                "status": "healthy", "device": "cpu", "engines": [],
                "version": "X", "uptime_s": 0,
                "backpressure": {
                    "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
                    "rejected_total": 0, "acquire_timeout_total": 0,
                },
            }),
            "浅检信封与 golden 不一致"
        );
    }

    #[tokio::test]
    async fn health_deep_with_detection_configured_should_be_exact_golden() {
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
        let mut body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        body["version"] = json!("X");
        body["uptime_s"] = json!(0);
        body["backpressure"] = json!({
            "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
            "rejected_total": 0, "acquire_timeout_total": 0,
        });
        assert_eq!(
            body,
            json!({
                "status": "healthy", "device": "cpu", "engines": [],
                "version": "X", "uptime_s": 0,
                "backpressure": {
                    "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
                    "rejected_total": 0, "acquire_timeout_total": 0,
                },
                "configured": ["detection"],
                "engine_health": [],
                "dynamic_engines": [],
                "dynamic_slot_health": [],
            }),
            "深检信封与 golden 不一致"
        );
    }

    // ---- 动态槽 deep 字段（F2，设计 C3：engines/engine_health 不变） ----

    #[tokio::test]
    async fn health_deep_should_carry_dynamic_fields_and_shallow_should_not() {
        let app = build_router(test_state(), None);
        let resp = app
            .clone()
            .oneshot(Request::get("/health?deep=true").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body["dynamic_engines"], serde_json::json!([]), "无槽应为空数组");
        assert_eq!(body["dynamic_slot_health"], serde_json::json!([]), "{body}");

        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(!text.contains("dynamic_engines"), "浅检不得新增字段: {text}");
    }

    /// 04 §4：合法来件 ID 原样采纳（router 级行为）。
    ///
    /// 注：采纳由 tower-http `SetRequestId`（头存在即沿用）完成，`AdoptRequestId`
    /// 那条"生成策略"分支在生产路径不可达（头缺失才会被调用）——2026-11-12 深度
    /// 审查实证；非法值的拦截由 `sanitize_request_id` 在路由层负责（见下两个用例）。
    #[test]
    fn request_id_form_should_be_validated() {
        assert!(!is_valid_request_id("short"), "少于 8 位应拒绝");
        assert!(!is_valid_request_id("bad id with spaces"), "空白应拒绝");
        assert!(!is_valid_request_id(&"a".repeat(65)), "超过 64 位应拒绝");
        assert!(!is_valid_request_id(&"é".repeat(8)), "非 ASCII 应拒绝");
        assert!(is_valid_request_id("12345678"));
        assert!(is_valid_request_id("ec07c7c7-5d4a-41a4-a7f0-98e1df837478"));
        assert!(is_valid_request_id(&"a".repeat(64)));
    }

    /// 路由层：响应头回写来件 ID（既有行为回归保护）。
    #[tokio::test]
    async fn response_carries_upstream_request_id() {
        const INBOUND: &str = "ec07c7c7-5d4a-41a4-a7f0-98e1df837478";
        let app = build_router(test_state(), None);
        let req = Request::builder()
            .uri("/health")
            .header("x-request-id", INBOUND)
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("x-request-id").and_then(|v| v.to_str().ok()),
            Some(INBOUND)
        );
    }

    /// 非法来件 ID 不得进入响应头/日志:router 级断言(tower-http 的 SetRequestId
    /// 对已存在的头不调用 MakeRequestId,直测生成器证明不了这条路径)。
    #[tokio::test]
    async fn invalid_upstream_request_id_should_be_replaced_not_adopted() {
        let bad_values = vec![
            "short".to_string(),
            "bad id with spaces".to_string(),
            "a".repeat(65),
            "c".repeat(8000),
        ];
        for bad in bad_values {
            let app = build_router(test_state(), None);
            let req = Request::builder()
                .uri("/health")
                .header("x-request-id", &bad)
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            let got = resp
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();
            assert_ne!(got, bad, "非法来件 ID 被采纳(len={})", bad.len());
            assert_eq!(got.len(), 36, "应替换为 UUID v4,实际: {got}");
        }
    }

    /// 缺失来件 ID → 自生成 UUID v4。
    #[tokio::test]
    async fn absent_request_id_should_generate_uuid() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let got = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert_eq!(got.len(), 36, "应为带连字符的 UUID v4: {got}");
        assert!(got.chars().all(|c| c.is_ascii_hexdigit() || c == '-'), "{got}");
    }

}
