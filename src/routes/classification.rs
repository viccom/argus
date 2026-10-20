//! 图像分类路由：`POST /classify`。
//!
//! 引擎：`ClassificationEngine::for_yolo`（YOLO-cls 系列，pixel/255 归一化）。
//! 低于引擎置信度阈值时返回空 predictions（与 /detect 空结果语义一致，非错误）。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::model::ClassificationResult;

use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image, field_text, parse_field_text};
use crate::routes::health::AppState;

/// top_k 默认值（ImageNet 惯例 top-5）与上限（防误传大值拖慢序列化）。
const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 50;
/// confidence 默认值：复现引擎构造期 0.5 阈值时代的 top-1 门控行为。
const DEFAULT_CONFIDENCE: f32 = 0.5;

/// top-1 门控：复现引擎在构造期阈值下的怪癖语义。
///
/// 引擎旧路径（base 默认 0.5）：top-1 不过 → 整列表空；top-1 过 → `from_scores`
/// 返回**全量** top-k（不做逐项过滤，低分尾巴保留）。引擎阈值降为 0 后由本函数
/// 按请求阈值复现同一语义——不是逐项过滤，两者在 top-1 过线时输出不同（有测试锁定）。
pub(crate) fn apply_top1_gate(
    preds: Vec<rust_onnx_infer::model::ClassificationResult>,
    threshold: f32,
) -> Vec<rust_onnx_infer::model::ClassificationResult> {
    match preds.first() {
        Some(top1) if (top1.confidence as f32) >= threshold => preds,
        _ => Vec::new(),
    }
}

// ---- DTO ----

#[derive(Serialize)]
struct ClassificationDto {
    class: String,
    class_id: usize,
    confidence: f64,
}

impl From<&ClassificationResult> for ClassificationDto {
    fn from(r: &ClassificationResult) -> Self {
        Self {
            class: r.class_name.clone(),
            class_id: r.class_id,
            confidence: r.confidence,
        }
    }
}

// ---- multipart 参数 ----

struct ClassifyParams {
    file: Vec<u8>,
    top_k: Option<usize>,
    confidence: Option<f32>,
}

async fn parse_classify_multipart(mut multipart: Multipart) -> Result<ClassifyParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut top_k: Option<usize> = None;
    let mut confidence: Option<f32> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| crate::routes::common::map_field_err(e, "multipart 解析失败"))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| crate::routes::common::map_field_err(e, "读取文件失败"))?
                        .to_vec(),
                );
            }
            "top_k" => {
                top_k = parse_field_text(&name, field).await?;
            }
            "confidence" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    let v: f32 = text
                        .parse()
                        .map_err(|_| AppError::BadRequest(format!("confidence 无法解析: {text}")))?;
                    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                        return Err(AppError::BadRequest(format!(
                            "confidence 越界: {v}（范围 [0.0, 1.0]）"
                        )));
                    }
                    confidence = Some(v);
                }
            }
            _ => {
                let _ = field_text(field).await?; // 消费未知字段，保持流一致
            }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(ClassifyParams { file, top_k, confidence })
}

// ---- handler ----

/// POST /classify — 图像分类（multipart: file, top_k?=5, confidence?=0.5）。
///
/// `confidence` 为 top-1 门控阈值：top-1 低于它时返回空列表（默认 0.5，
/// 与历史行为逐字段一致）；显式传低值（如 0.05）可拿到带各自 conf 的完整 top-k。
pub async fn classify(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_classify_multipart(multipart).await?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;
    let top_k = params.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let confidence = params.confidence.unwrap_or(DEFAULT_CONFIDENCE);

    let pool = state.engines.classification().await?;

    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool.run(move |e| e.predict_top_k(&image, top_k)).await?;
    let results = apply_top1_gate(results, confidence);
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<ClassificationDto> = results.iter().map(ClassificationDto::from).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "classify", count, top_k, inference_ms = elapsed as u64, "推理完成");

    let (model, pipeline) = state.engines.provenance("classification").await;
    Ok(Json(json!({
        "status": "ok",
        "data": {
            "model": model,
            "pipeline": pipeline,
            "predictions": dtos,
            "count": count,
            "inference_ms": elapsed,
        }
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::EngineManager;
    use crate::routes::health::{build_router, AppState};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let config = Config::default();
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn body_text(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// 构造 multipart body：有效 1x1 PNG file 字段 + 可选附加字段。
    fn multipart_with_png(extra: &[(&str, &str)]) -> Vec<u8> {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();

        let boundary = "----testboundary";
        let mut full = Vec::new();
        for (name, value) in extra {
            full.extend_from_slice(
                format!(
                    "--{boundary}\r\n\
                     Content-Disposition: form-data; name=\"{name}\"\r\n\r\n\
                     {value}\r\n"
                )
                .as_bytes(),
            );
        }
        full.extend_from_slice(
            format!(
                "--{boundary}\r\n\
                 Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n\
                 Content-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        full.extend_from_slice(&img_bytes);
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        full
    }

    fn multipart_invalid_image() -> Vec<u8> {
        let boundary = "----testboundary";
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n\
             Content-Type: image/jpeg\r\n\r\n\
             not-an-image\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
    }

    fn multipart_without_file() -> Vec<u8> {
        let boundary = "----testboundary";
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"top_k\"\r\n\r\n\
             5\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
    }

    async fn post(app: axum::Router, body: Vec<u8>) -> String {
        let resp = app
            .oneshot(
                Request::post("/classify")
                    .header(
                        "content-type",
                        "multipart/form-data; boundary=----testboundary",
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn classify_without_file_returns_bad_request() {
        let text = post(build_router(test_state(), None), multipart_without_file()).await;
        assert!(text.contains("error"), "应返回错误: {text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn classify_with_invalid_image_returns_image_error() {
        let text = post(build_router(test_state(), None), multipart_invalid_image()).await;
        assert!(text.contains("error"), "应返回错误: {text}");
    }

    #[tokio::test]
    async fn classify_when_model_not_configured_returns_error() {
        let text = post(build_router(test_state(), None), multipart_with_png(&[])).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[tokio::test]
    async fn classify_with_invalid_topk_returns_bad_request() {
        let text = post(
            build_router(test_state(), None),
            multipart_with_png(&[("top_k", "abc")]),
        )
        .await;
        assert!(text.contains("top_k"), "应指出 top_k 无法解析: {text}");
    }

    #[test]
    fn classification_dto_should_serialize_full_fields() {
        let r = rust_onnx_infer::model::ClassificationResult::new("minibus", 654, 0.5759);
        let dto = ClassificationDto::from(&r);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["class"], "minibus");
        assert_eq!(json["class_id"], 654);
        assert_eq!(json["confidence"], 0.5759);
    }

    /// top-1 门控语义（必须逐字段复现引擎 0.5 阈值时代的怪癖）：
    /// top-1 过阈值 → 返回**全部** top-k（含低分尾巴）；
    /// top-1 不过 → 空列表。这不是逐项过滤——两者在 top-1 过线时输出不同。
    #[test]
    fn top1_gate_should_keep_low_confidence_tail_when_top1_passes() {
        let preds = vec![
            rust_onnx_infer::model::ClassificationResult::new("a", 1, 0.8),
            rust_onnx_infer::model::ClassificationResult::new("b", 2, 0.3),
        ];
        let gated = apply_top1_gate(preds, 0.5);
        assert_eq!(gated.len(), 2, "top-1 过线时低分尾巴必须保留: {gated:?}");
    }

    #[test]
    fn top1_gate_should_return_empty_when_top1_below_threshold() {
        let preds = vec![
            rust_onnx_infer::model::ClassificationResult::new("a", 1, 0.4),
            rust_onnx_infer::model::ClassificationResult::new("b", 2, 0.2),
        ];
        assert!(apply_top1_gate(preds, 0.5).is_empty());
    }

    #[test]
    fn top1_gate_with_zero_threshold_should_always_return_all() {
        // 显式低值（如 0.05）的用途：拿到带各自 conf 的真实 top-k
        let preds = vec![rust_onnx_infer::model::ClassificationResult::new("a", 1, 0.07)];
        assert_eq!(apply_top1_gate(preds, 0.0).len(), 1);
    }

    #[test]
    fn top1_gate_on_empty_input_should_stay_empty() {
        assert!(apply_top1_gate(Vec::new(), 0.5).is_empty());
    }

    #[tokio::test]
    async fn classify_with_invalid_confidence_returns_bad_request() {
        let text = post(
            build_router(test_state(), None),
            multipart_with_png(&[("confidence", "1.5")]),
        )
        .await;
        assert!(text.contains("confidence"), "应指出 confidence 越界: {text}");
        assert!(text.contains("bad_request"), "应为 bad_request 码: {text}");
    }
}
