//! 实例分割路由:`POST /segmentation`(YOLO-seg 系列,检测框 + 逐实例 float 掩码)。
//!
//! 响应契约:`data.segments[]`,每项含检测字段 + `area_ratio`(掩码前景占掩码全图比例)
//! + `mask_preview`(最多 64×64 均值池化预览,行优先,`mask_cols` 为列数)。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::model::Segmentation;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::error::AppError;
use crate::routes::analysis::downsample_grid;
use crate::routes::common::{acquire_permit, decode_image, field_text};
use crate::routes::health::AppState;

/// 掩码预览目标边长(与 depth 的 64×64 预览口径一致)。
const MASK_PREVIEW_TARGET: usize = 64;

// ---- DTO ----

#[derive(Serialize)]
struct SegmentationDto {
    class: String,
    class_id: i32,
    confidence: f64,
    bbox: [f64; 4],
    /// 掩码前景占掩码全图像素比例(0~1,按引擎掩码阈值二值;无掩码为 0)
    area_ratio: f64,
    /// 下采样掩码预览(行优先 0~1;空 = 该实例无掩码)
    mask_preview: Vec<f32>,
    /// mask_preview 的列数(前端重建网格必需)
    mask_cols: usize,
}

impl SegmentationDto {
    fn from_seg(s: &Segmentation) -> Self {
        let d = &s.detection;
        let (mask_preview, mask_cols) = match &s.mask {
            Some(m) if !m.is_empty() => {
                downsample_grid(m.data(), m.width(), m.height(), MASK_PREVIEW_TARGET)
            }
            _ => (Vec::new(), 0),
        };
        let total = match &s.mask {
            Some(m) if !m.is_empty() => (m.width() * m.height()) as f64,
            _ => 0.0,
        };
        let area_ratio = if total > 0.0 { s.mask_area() / total } else { 0.0 };
        Self {
            class: d.class_name.clone(),
            class_id: d.class_id,
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
            area_ratio,
            mask_preview,
            mask_cols,
        }
    }
}

// ---- multipart 参数 ----

async fn parse_file_multipart(mut multipart: Multipart) -> Result<Vec<u8>, AppError> {
    let mut file: Option<Vec<u8>> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("multipart 解析失败: {e}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            file = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("读取文件失败: {e}")))?
                    .to_vec(),
            );
        } else {
            let _ = field_text(field).await?;
        }
    }
    file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))
}

// ---- handler ----

/// POST /segmentation — 实例分割(传统/End2End 布局由引擎按输出 shape 自动识别)。
pub async fn segmentation(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let file = parse_file_multipart(multipart).await?;
    let image = decode_image(file).await?;

    let pool = state.engines.segmentation().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool.run(move |e| e.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<SegmentationDto> = results.iter().map(SegmentationDto::from_seg).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "segmentation", count, inference_ms = elapsed as u64, "推理完成");

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "segments": dtos,
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
    use rust_onnx_infer::imaging::FloatMask;
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

    fn multipart_png() -> Vec<u8> {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();

        let boundary = "----testboundary";
        let mut full = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n\
             Content-Type: image/png\r\n\r\n"
        )
        .into_bytes();
        full.extend_from_slice(&img_bytes);
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        full
    }

    fn multipart_without_file() -> Vec<u8> {
        let boundary = "----testboundary";
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"confidence\"\r\n\r\n\
             0.5\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
    }

    async fn post(body: Vec<u8>) -> String {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/segmentation")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn segmentation_without_file_returns_bad_request() {
        let text = post(multipart_without_file()).await;
        assert!(text.contains("error"), "{text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn segmentation_when_model_not_configured_returns_error() {
        let text = post(multipart_png()).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[tokio::test]
    async fn segmentation_with_invalid_image_returns_image_error() {
        let boundary = "----testboundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n\
             Content-Type: image/jpeg\r\n\r\n\
             not-an-image\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes();
        let text = post(body).await;
        assert!(text.contains("error"), "{text}");
    }

    #[test]
    fn segmentation_dto_should_serialize_with_mask() {
        let mask = FloatMask::from_raw(
            4,
            4,
            vec![0.0, 0.0, 0.9, 0.9, 0.0, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let seg = Segmentation::new("person", 0, 10.0, 20.0, 110.0, 220.0, 0.95, Some(mask));
        let dto = SegmentationDto::from_seg(&seg);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["class"], "person");
        assert_eq!(json["confidence"], 0.95);
        assert_eq!(json["bbox"][3], 220.0);
        // 4x4 掩码阈值 0.35 二值:0.9/0.9/0.8/0.9 共 4 个前景 / 16 总像素
        let expect = 4.0 / 16.0;
        assert!((json["area_ratio"].as_f64().unwrap() - expect).abs() < 1e-6);
        assert!(!json["mask_preview"].as_array().unwrap().is_empty());
    }

    #[test]
    fn segmentation_dto_without_mask_should_zero_area() {
        let seg = Segmentation::new("car", 2, 0.0, 0.0, 50.0, 50.0, 0.8, None);
        let dto = SegmentationDto::from_seg(&seg);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["area_ratio"], 0.0);
        assert_eq!(json["mask_preview"].as_array().unwrap().len(), 0);
        assert_eq!(json["mask_cols"], 0);
    }
}
