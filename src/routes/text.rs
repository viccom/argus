//! 文字识别路由：`/ocr`（OCR 文字识别）、`/license-plate`（车牌识别）、`/qr`（二维码识别）。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::engines::license_plate::PlateResult;
use rust_onnx_infer::engines::ocr::{TextLine, TextRegion};
use rust_onnx_infer::engines::qr_detector::QrDetection;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::error::AppError;
use crate::routes::detection::{acquire_permit, decode_image, field_text};
use crate::routes::health::AppState;

// ---- DTO ----

#[derive(Serialize)]
struct TextRegionDto {
    /// 四角点 [x, y]
    quad: [[f32; 2]; 4],
    score: f32,
    text: String,
    text_score: f32,
}

impl From<&(TextRegion, TextLine)> for TextRegionDto {
    fn from((region, line): &(TextRegion, TextLine)) -> Self {
        Self {
            quad: region.quad,
            score: region.score,
            text: line.text.clone(),
            text_score: line.score,
        }
    }
}

#[derive(Serialize)]
struct PlateDto {
    /// 车牌外接框 xyxy
    bbox: [f32; 4],
    text: String,
    score: f64,
    plate_type: Option<String>,
}

impl From<&PlateResult> for PlateDto {
    fn from(p: &PlateResult) -> Self {
        Self {
            bbox: p.r#box,
            text: p.text.clone(),
            score: p.score,
            plate_type: p.plate_type.clone(),
        }
    }
}

#[derive(Serialize)]
struct QrDto {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    score: f32,
}

impl From<&QrDetection> for QrDto {
    fn from(q: &QrDetection) -> Self {
        Self {
            x: q.x,
            y: q.y,
            w: q.w,
            h: q.h,
            score: q.score,
        }
    }
}

// ---- multipart 参数 ----

struct FileParams {
    file: Vec<u8>,
}

async fn parse_file_multipart(mut multipart: Multipart) -> Result<FileParams, AppError> {
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
            let _ = field_text(field).await;
        }
    }
    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(FileParams { file })
}

// ---- handlers ----

/// POST /ocr — OCR 文字识别（PP-OCR 检测 + 识别）。
pub async fn ocr(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.ocr().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = crate::engine::run_blocking(move || engine.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<TextRegionDto> = results.iter().map(TextRegionDto::from).collect();
    let count = dtos.len();
    let full_text: Vec<&str> = dtos.iter().map(|d| d.text.as_str()).collect();

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "regions": dtos,
            "count": count,
            "text": full_text.join("\n"),
            "inference_ms": elapsed,
        }
    })))
}

/// POST /license-plate — 车牌识别。
pub async fn license_plate(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.license_plate().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = crate::engine::run_blocking(move || engine.recognize(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<PlateDto> = results.iter().map(PlateDto::from).collect();
    let count = dtos.len();

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "plates": dtos,
            "count": count,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /qr — 二维码检测（仅定位，不解码内容）。
pub async fn qr(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.qr().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = crate::engine::run_blocking(move || engine.detect(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<QrDto> = results.iter().map(QrDto::from).collect();
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

    fn multipart_no_file() -> Vec<u8> {
        let boundary = "----testboundary";
        format!("--{boundary}--\r\n").into_bytes()
    }

    fn multipart_with_png() -> Vec<u8> {
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
        full
    }

    #[tokio::test]
    async fn ocr_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/ocr")
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
    async fn ocr_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/ocr")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_with_png()))
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
    async fn license_plate_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/license-plate")
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
    async fn license_plate_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/license-plate")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_with_png()))
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
    async fn qr_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/qr")
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
    async fn qr_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/qr")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(multipart_with_png()))
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
    fn text_region_dto_should_serialize_all_fields() {
        let region = TextRegion {
            quad: [[0.0, 0.0], [100.0, 0.0], [100.0, 30.0], [0.0, 30.0]],
            score: 0.9,
        };
        let line = TextLine {
            text: "hello".into(),
            score: 0.95,
        };
        let dto = TextRegionDto::from(&(region, line));
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["text"], "hello");
        assert!((json["text_score"].as_f64().unwrap() - 0.95).abs() < 1e-5);
        assert!((json["score"].as_f64().unwrap() - 0.9).abs() < 1e-5);
        assert_eq!(json["quad"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn plate_dto_should_serialize_all_fields() {
        let p = PlateResult {
            r#box: [10.0, 20.0, 110.0, 60.0],
            text: "皖A12345".into(),
            score: 0.92,
            plate_type: Some("蓝牌".into()),
        };
        let dto = PlateDto::from(&p);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["text"], "皖A12345");
        assert_eq!(json["score"], 0.92);
        assert_eq!(json["plate_type"], "蓝牌");
        assert_eq!(json["bbox"][0], 10.0);
    }

    #[test]
    fn qr_dto_should_serialize_all_fields() {
        let q = QrDetection {
            x: 10,
            y: 20,
            w: 100,
            h: 100,
            score: 0.88,
        };
        let dto = QrDto::from(&q);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["x"], 10);
        assert_eq!(json["w"], 100);
        assert!((json["score"].as_f64().unwrap() - 0.88).abs() < 1e-5);
    }
}
