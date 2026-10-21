//! POST /detect/batch — 逐图批量检测（服务端本地路径）。
//!
//! 契约源自 yolo-rs（detector.py 消费方）：`image_paths` 为**服务端本地路径**（内网文件语义，
//! 非上传），逐图 for 循环（非张量 batch，真 batch 是后续吞吐优化项）。
//!
//! 与 yolo-rs 差异（吸收计划阶段 3 明确要求）：
//! - **错误隔离**：单图失败该条目带 `error`，不炸整批（yolo-rs 是 `?` 直接中断）；
//! - **每图独立背压 permit**：过载时按 `ARGUS_ACQUIRE_TIMEOUT_MS` 语义排队或 fail-fast；
//! - `model` 可选：命名动态槽（slots.toml），缺省用预置 detection 槽；

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::engine::EnginePool;
use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image};
use crate::routes::detect_filter::{DetectFilter, Polygon};
use crate::routes::detection::DetectionDto;
use crate::routes::health::AppState;

#[derive(Deserialize)]
pub struct BatchRequest {
    image_paths: Vec<String>,
    /// 可选动态槽名（slots.toml 注册）；缺省用预置 detection 槽
    model: Option<String>,
    confidence: Option<f64>,
    target_classes: Option<Vec<i32>>,
    /// 归一化多边形 `[[x,y],...]`
    roi: Option<Vec<[f64; 2]>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum BatchItem {
    Ok {
        image_path: String,
        detections: Vec<DetectionDto>,
        count: usize,
        inference_ms: u128,
    },
    Err {
        image_path: String,
        error: String,
    },
}

/// POST /detect/batch
pub async fn detect_batch(
    State(state): State<AppState>,
    body: Result<Json<BatchRequest>, JsonRejection>,
) -> Result<Json<Value>, AppError> {
    // JSON 解析失败走服务统一信封（HTTP 200 + status=error），而非 axum 默认的 4xx 纯文本
    let Json(req) = body.map_err(|e| AppError::BadRequest(format!("JSON 请求体无效: {e}")))?;
    if req.image_paths.is_empty() {
        return Err(AppError::BadRequest("image_paths 不能为空".into()));
    }
    let roi = req.roi.map(Polygon::from_points).transpose()?;
    let filter = DetectFilter::new(req.confidence, req.target_classes, roi)?;
    // 模型未配置是整批级错误（无引擎可用），不做逐图隔离；动态槽在入口解析一次，
    // 逐图复用同一池与同一溯源（单图失败仍按现有语义隔离）
    let model = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let (pool, dyn_slot) = match model {
        None => (state.engines.detection().await?, None),
        Some(name) => {
            let (slot, pool) = state.engines.resolve_dynamic_slot(name).await?;
            (pool, Some(slot))
        }
    };

    let t0 = std::time::Instant::now();
    let mut results = Vec::with_capacity(req.image_paths.len());
    let mut failed = 0usize;
    for path in req.image_paths {
        match detect_one(&state, &pool, &filter, &path).await {
            Ok(item) => results.push(item),
            Err(e) => {
                failed += 1;
                tracing::warn!(image_path = %path, error = %e, "批量检测单图失败（已隔离）");
                results.push(BatchItem::Err {
                    image_path: path,
                    error: e.to_string(),
                });
            }
        }
    }
    let total = results.len();
    let elapsed = t0.elapsed().as_millis();
    tracing::info!(
        endpoint = "detect_batch",
        total,
        failed,
        confidence = filter.confidence,
        elapsed_ms = elapsed as u64,
        "批量检测完成"
    );

    let (model, pipeline) = match &dyn_slot {
        Some(slot) => match slot.model_info().await {
            Some(m) => (json!(m.model), json!(m.pipeline)),
            None => (Value::Null, Value::Null),
        },
        None => state.engines.provenance("detection").await,
    };
    Ok(Json(json!({
        "status": "ok",
        "data": {
            "model": model,
            "pipeline": pipeline,
            "results": results,
            "total": total,
            "succeeded": total - failed,
            "failed": failed,
            "elapsed_ms": elapsed,
        }
    })))
}

/// 单图流程：读文件 → 解码 → 取 permit → 推理 → 过滤。任一步失败由调用方隔离。
async fn detect_one(
    state: &AppState,
    pool: &EnginePool<DetectionEngine>,
    filter: &DetectFilter,
    path: &str,
) -> Result<BatchItem, AppError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| AppError::BadRequest(format!("读取 {path} 失败: {e}")))?;
    let image = decode_image(bytes, state.config.max_image_pixels).await?;
    let (img_w, img_h) = (image.width(), image.height());
    let _permit = acquire_permit(state).await?;
    let t0 = std::time::Instant::now();
    let dets = pool.run(move |e| e.predict(&image)).await?;
    let inference_ms = t0.elapsed().as_millis();
    let dets = filter.apply(dets, img_w, img_h);
    let detections: Vec<DetectionDto> = dets.iter().map(DetectionDto::from).collect();
    Ok(BatchItem::Ok {
        image_path: path.to_string(),
        count: detections.len(),
        detections,
        inference_ms,
    })
}

#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::engine::EngineManager;
    use crate::routes::health::{build_router, AppState};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let config = Config::default();
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn post_batch(body: &str) -> (StatusCode, String) {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect/batch")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    #[tokio::test]
    async fn batch_with_empty_paths_returns_bad_request() {
        let (_, text) = post_batch(r#"{"image_paths": []}"#).await;
        assert!(text.contains("error") && text.contains("不能为空"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_model_unknown_slot_returns_unknown_model() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": "nope"}"#).await;
        assert!(text.contains("\"code\":\"unknown_model\""), "{text}");
    }

    #[tokio::test]
    async fn batch_with_reserved_model_returns_bad_request_hint() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": "detection"}"#).await;
        assert!(text.contains("静态能力键") && text.contains("端点"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_empty_model_should_fall_back_to_static() {
        // 空 model = 未传：走静态 detection（未配模型 → 整批"模型未配置"）
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": ""}"#).await;
        assert!(text.contains("未配置") || text.contains("model"), "{text}");
        assert!(!text.contains("unknown_model"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_confidence_below_floor_returns_bad_request() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "confidence": 0.01}"#).await;
        assert!(text.contains("error") && text.contains("0.1"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_malformed_roi_returns_bad_request() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "roi": [[0,0],[1,1]]}"#).await;
        assert!(text.contains("3 个点"), "{text}");
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "roi": [[0,0],[2,0],[1,1]]}"#).await;
        assert!(text.contains("越界"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_malformed_json_uses_error_envelope() {
        // 不应落到 axum 默认的 4xx 纯文本,消费方按统一信封解析
        let (status, text) = post_batch(r#"{"image_paths": [}"#).await;
        assert_eq!(status, StatusCode::OK, "应走 HTTP 200 信封: {text}");
        assert!(text.contains("\"status\":\"error\"") && text.contains("JSON"), "{text}");
    }

    #[tokio::test]
    async fn batch_when_model_not_configured_fails_whole_batch() {
        // 无引擎可用属整批级错误,不进入逐图隔离
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg", "/tmp/b.jpg"]}"#).await;
        assert!(text.contains("未配置") || text.contains("model"), "{text}");
        assert!(!text.contains("results"), "整批失败不应返回 results: {text}");
    }
}
