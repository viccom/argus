//! 目标检测路由：`/detect`（含 SAHI 切片）与 `/detect/obb`（旋转框）。

use std::time::Duration;

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::engines::obb_detection::ObbResult;
use rust_onnx_infer::model::Detection;
use rust_onnx_infer::sahi::config::SahiConfig;
use rust_onnx_infer::sahi::sliced_predictor::predict_sliced_detection;
use rust_onnx_infer::{Image, OnnxInferenceEngine};

use crate::error::AppError;
use crate::routes::health::AppState;

/// 背压排队超时：超过该时间仍未拿到推理许可则拒绝请求。
const INFERENCE_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);

// ---- DTO ----

#[derive(Serialize)]
struct DetectionDto {
    class: String,
    class_id: i32,
    confidence: f64,
    bbox: [f64; 4],
}

impl From<&Detection> for DetectionDto {
    fn from(d: &Detection) -> Self {
        Self {
            class: d.class_name.clone(),
            class_id: d.class_id,
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
        }
    }
}

#[derive(Serialize)]
struct ObbDto {
    class: String,
    class_id: i32,
    confidence: f64,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle: f32,
    corners: [[f32; 2]; 4],
}

impl From<&ObbResult> for ObbDto {
    fn from(o: &ObbResult) -> Self {
        Self {
            class: o.class_name.clone(),
            class_id: o.class_id,
            confidence: o.confidence,
            cx: o.cx,
            cy: o.cy,
            w: o.w,
            h: o.h,
            angle: o.angle_rad,
            corners: o.corners,
        }
    }
}

// ---- multipart 参数 ----

/// /detect 的 multipart 参数。
struct DetectParams {
    file: Vec<u8>,
    #[allow(dead_code)] // 后续版本支持运行时调整置信度
    confidence: Option<f32>,
    sahi: bool,
    slice_w: Option<i32>,
    slice_h: Option<i32>,
    overlap: Option<f64>,
}

async fn parse_detect_multipart(mut multipart: Multipart) -> Result<DetectParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut confidence: Option<f32> = None;
    let mut sahi = false;
    let mut slice_w: Option<i32> = None;
    let mut slice_h: Option<i32> = None;
    let mut overlap: Option<f64> = None;

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
            "confidence" => {
                confidence = parse_field_text(&name, field).await?;
            }
            "sahi" => {
                let text = field_text(field).await?;
                sahi = text.trim().eq_ignore_ascii_case("true") || text == "1";
            }
            "slice_w" => {
                slice_w = parse_field_text(&name, field).await?;
            }
            "slice_h" => {
                slice_h = parse_field_text(&name, field).await?;
            }
            "overlap" => {
                overlap = parse_field_text(&name, field).await?;
            }
            _ => { /* 忽略未知字段 */ }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(DetectParams {
        file,
        confidence,
        sahi,
        slice_w,
        slice_h,
        overlap,
    })
}

/// 通用单字段文本提取 + 泛型解析（复用于 confidence / slice_w / slice_h / overlap）。
async fn parse_field_text<T>(
    field_name: &str,
    field: axum::extract::multipart::Field<'_>,
) -> Result<Option<T>, AppError>
where
    T: std::str::FromStr,
{
    let text = field_text(field).await?;
    if text.trim().is_empty() {
        return Ok(None);
    }
    text.parse::<T>()
        .map(Some)
        .map_err(|_| AppError::BadRequest(format!("字段 {field_name} 无法解析: {text}")))
}

pub(crate) async fn field_text(field: axum::extract::multipart::Field<'_>) -> Result<String, AppError> {
    let bytes = field
        .bytes()
        .await
        .map_err(|e| AppError::BadRequest(format!("读取字段失败: {e}")))?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

// ---- 图像解码 ----

pub(crate) fn decode_image(bytes: &[u8]) -> Result<Image, AppError> {
    let dyn_img = image::load_from_memory(bytes)?;
    Ok(Image::from_dynamic(&dyn_img))
}

// ---- 背压 ----

pub(crate) async fn acquire_permit(state: &AppState) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
    let permit = tokio::time::timeout(
        INFERENCE_ACQUIRE_TIMEOUT,
        state.inference_semaphore.clone().acquire_owned(),
    )
    .await
    .map_err(|_| AppError::Overload("推理排队超时，请稍后重试".into()))?
    .map_err(|_| AppError::Internal("信号量已关闭".into()))?;
    Ok(permit)
}

// ---- handlers ----

/// POST /detect — 目标检测（可选 SAHI 切片）。
pub async fn detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_detect_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.detection().await?;

    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let (detections, num_slices) = if params.sahi {
        let mut cfg = SahiConfig::default();
        if let Some(w) = params.slice_w {
            cfg.slice_width = Some(w);
        }
        if let Some(h) = params.slice_h {
            cfg.slice_height = Some(h);
        }
        if let Some(o) = params.overlap {
            cfg.overlap_width_ratio = o;
            cfg.overlap_height_ratio = o;
        }
        let result = crate::engine::run_blocking(move || {
            predict_sliced_detection(engine.as_ref(), engine.as_ref(), &image, &cfg)
        })
        .await?;
        (result.detections, Some(result.num_slices))
    } else {
        let dets = crate::engine::run_blocking(move || engine.predict(&image)).await?;
        (dets, None)
    };
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<DetectionDto> = detections.iter().map(DetectionDto::from).collect();
    let count = dtos.len();

    let mut data = json!({
        "detections": dtos,
        "count": count,
        "inference_ms": elapsed,
    });
    if let Some(n) = num_slices {
        data["sahi_slices"] = json!(n);
    }

    Ok(Json(json!({ "status": "ok", "data": data })))
}

/// POST /detect/obb — OBB 旋转框检测。
pub async fn detect_obb(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_detect_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.obb().await?;

    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results =
        crate::engine::run_blocking(move || engine.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<ObbDto> = results.iter().map(ObbDto::from).collect();
    let count = dtos.len();

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "detections": dtos,
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

    /// 构造一个只含 file 字段的 multipart body（无效图像内容）。
    fn multipart_no_file() -> Vec<u8> {
        let boundary = "----testboundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"confidence\"\r\n\r\n\
             0.5\r\n\
             --{boundary}--\r\n"
        );
        body.into_bytes()
    }

    fn multipart_with_invalid_image() -> Vec<u8> {
        let boundary = "----testboundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n\
             Content-Type: image/jpeg\r\n\r\n\
             not-an-image\r\n\
             --{boundary}--\r\n"
        );
        body.into_bytes()
    }

    #[tokio::test]
    async fn detect_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_no_file()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("error"), "应返回错误: {text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn detect_with_invalid_image_returns_image_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_with_invalid_image()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("error"), "应返回错误: {text}");
    }

    #[tokio::test]
    async fn detect_when_model_not_configured_returns_error() {
        // 用一个 1x1 红色像素的有效 PNG
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();

        let boundary = "----testboundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n\
             Content-Type: image/png\r\n\r\n"
        );
        let mut full = body.into_bytes();
        full.extend_from_slice(&img_bytes);
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(full))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[tokio::test]
    async fn obb_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect/obb")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_no_file()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("error"), "应返回错误: {text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn obb_when_model_not_configured_returns_error() {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();

        let boundary = "----testboundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n\
             Content-Type: image/png\r\n\r\n"
        );
        let mut full = body.into_bytes();
        full.extend_from_slice(&img_bytes);
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect/obb")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(full))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[test]
    fn detection_dto_should_serialize_full_fields() {
        let d = Detection::new("person", 0, 10.0, 20.0, 110.0, 220.0, 0.95);
        let dto = DetectionDto::from(&d);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["class"], "person");
        assert_eq!(json["class_id"], 0);
        assert_eq!(json["confidence"], 0.95);
        assert_eq!(json["bbox"][0], 10.0);
        assert_eq!(json["bbox"][3], 220.0);
    }

    #[test]
    fn obb_dto_should_serialize_full_fields() {
        let o = ObbResult {
            cx: 50.0,
            cy: 60.0,
            w: 30.0,
            h: 20.0,
            angle_rad: 0.5,
            corners: [[0.0; 2]; 4],
            class_name: "ship".into(),
            class_id: 1,
            confidence: 0.88,
        };
        let dto = ObbDto::from(&o);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["class"], "ship");
        assert_eq!(json["confidence"], 0.88);
        assert_eq!(json["angle"], 0.5);
        assert_eq!(json["cx"], 50.0);
    }
}
