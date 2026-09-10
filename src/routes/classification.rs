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
}

async fn parse_classify_multipart(mut multipart: Multipart) -> Result<ClassifyParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut top_k: Option<usize> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("multipart 解析失败: {e}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| AppError::BadRequest(format!("读取文件失败: {e}")))?
                        .to_vec(),
                );
            }
            "top_k" => {
                top_k = parse_field_text(&name, field).await?;
            }
            _ => {
                let _ = field_text(field).await?; // 消费未知字段，保持流一致
            }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(ClassifyParams { file, top_k })
}

// ---- handler ----

/// POST /classify — 图像分类（multipart: file, top_k?=5）。
pub async fn classify(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_classify_multipart(multipart).await?;
    let image = decode_image(params.file).await?;
    let top_k = params.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);

    let pool = state.engines.classification().await?;

    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool.run(move |e| e.predict_top_k(&image, top_k)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<ClassificationDto> = results.iter().map(ClassificationDto::from).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "classify", count, top_k, inference_ms = elapsed as u64, "推理完成");

    Ok(Json(json!({
        "status": "ok",
        "data": {
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
}
