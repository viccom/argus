//! 图像分析路由：`/quality`（质量评估）、`/pose`（姿态估计）、`/depth`（深度估计）、`/human-parsing`（人体解析）。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::engines::human_parsing::{CLOTHES_CLASSES, ParsingMap};
use rust_onnx_infer::engines::image_quality::{ImageQualityReport, QualityAssessor};
use rust_onnx_infer::imaging::FloatMask;
use rust_onnx_infer::model::{coco, PoseResult};
use rust_onnx_infer::OnnxInferenceEngine;

use crate::error::AppError;
use crate::routes::detection::{acquire_permit, decode_image, field_text};
use crate::routes::health::AppState;

// ---- DTO ----

#[derive(Serialize)]
struct QualityDto {
    blur_score: f32,
    brightness: f32,
    contrast: f32,
    noise: f32,
    overall: f32,
}

impl From<&ImageQualityReport> for QualityDto {
    fn from(r: &ImageQualityReport) -> Self {
        Self {
            blur_score: r.blur_score,
            brightness: r.brightness,
            contrast: r.contrast,
            noise: r.noise,
            overall: r.overall,
        }
    }
}

#[derive(Serialize)]
struct KeypointDto {
    x: f32,
    y: f32,
    score: f32,
    name: &'static str,
}

#[derive(Serialize)]
struct PoseDto {
    confidence: f64,
    bbox: [f64; 4],
    keypoints: Vec<KeypointDto>,
}

impl From<&PoseResult> for PoseDto {
    fn from(p: &PoseResult) -> Self {
        let d = &p.detection;
        let kps = p
            .keypoints
            .iter()
            .enumerate()
            .map(|(i, k)| KeypointDto {
                x: k.x,
                y: k.y,
                score: k.score,
                name: coco::NAMES.get(i).copied().unwrap_or("?"),
            })
            .collect();
        Self {
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
            keypoints: kps,
        }
    }
}

#[derive(Serialize)]
struct DepthDto {
    width: usize,
    height: usize,
    /// 最小深度值
    min: f32,
    /// 最大深度值
    max: f32,
    /// 平均深度值
    mean: f32,
    /// 下采样后的深度值（最多 64x64，用于可视化预览）
    preview: Vec<f32>,
}

impl From<&FloatMask> for DepthDto {
    fn from(mask: &FloatMask) -> Self {
        let data = mask.data();
        let w = mask.width();
        let h = mask.height();

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for &v in data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        let mean = if data.is_empty() { 0.0 } else { sum / data.len() as f32 };

        // 下采样到最多 64x64
        let target = 64usize;
        let step_w = (w + target - 1) / target.max(1);
        let step_h = (h + target - 1) / target.max(1);
        let mut preview = Vec::new();
        let step_w = step_w.max(1);
        let step_h = step_h.max(1);
        let mut y = 0;
        while y < h {
            let mut x = 0;
            while x < w {
                let mut s = 0.0;
                let mut n = 0;
                let y_end = (y + step_h).min(h);
                let x_end = (x + step_w).min(w);
                for yy in y..y_end {
                    for xx in x..x_end {
                        s += data[yy * w + xx];
                        n += 1;
                    }
                }
                preview.push(if n > 0 { s / n as f32 } else { 0.0 });
                x += step_w;
            }
            y += step_h;
        }

        Self {
            width: w,
            height: h,
            min,
            max,
            mean,
            preview,
        }
    }
}

#[derive(Serialize)]
struct HumanParsingDto {
    width: usize,
    height: usize,
    /// 各部件像素占比（不含 Background）
    parts: Vec<PartRatio>,
}

#[derive(Serialize)]
struct PartRatio {
    name: &'static str,
    ratio: f32,
}

impl From<&ParsingMap> for HumanParsingDto {
    fn from(map: &ParsingMap) -> Self {
        let total = map.classes.len();
        let mut counts = [0usize; 256];
        for &c in &map.classes {
            counts[c as usize] += 1;
        }

        let mut parts: Vec<PartRatio> = Vec::new();
        for (id, &count) in counts.iter().enumerate().take(CLOTHES_CLASSES.len()).skip(1) {
            if count > 0 {
                let ratio = if total > 0 {
                    count as f32 / total as f32
                } else {
                    0.0
                };
                parts.push(PartRatio {
                    name: map.class_name(id),
                    ratio,
                });
            }
        }
        parts.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            width: map.width,
            height: map.height,
            parts,
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

/// POST /quality — 图像质量评估（纯算法，不依赖模型文件）。
pub async fn quality(
    State(_state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    // QualityAssessor 为纯算法评估器，不加载 ONNX 模型；CPU 密集但轻量，
    // 仍走 spawn_blocking 避免阻塞 runtime。
    let t0 = std::time::Instant::now();
    let report = crate::engine::run_blocking(move || QualityAssessor::new().assess(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dto = QualityDto::from(&report);

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "quality": dto,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /pose — 姿态估计（COCO 17 关键点）。
pub async fn pose(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.pose().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = crate::engine::run_blocking(move || engine.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<PoseDto> = results.iter().map(PoseDto::from).collect();
    let count = dtos.len();

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "poses": dtos,
            "count": count,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /depth — 单目深度估计。
pub async fn depth(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.depth().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let mask = crate::engine::run_blocking(move || engine.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dto = DepthDto::from(&mask);

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "depth": dto,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /human-parsing — 人体解析（衣物部件分割）。
pub async fn human_parsing(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(&params.file)?;

    let engine = state.engines.human_parsing().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let map = crate::engine::run_blocking(move || engine.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dto = HumanParsingDto::from(&map);

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "parsing": dto,
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
    use rust_onnx_infer::model::{Detection, Keypoint};
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
    async fn quality_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/quality")
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
    async fn quality_returns_report_without_model() {
        // QualityAssessor 为纯算法，不需要模型文件即可工作
        // 用 8x8 图像（laplacian_variance 要求 >= 3x3）
        let img = image::DynamicImage::new_rgb8(8, 8);
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
                Request::post("/quality")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(full))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("ok"), "纯算法质量评估应成功: {text}");
        assert!(text.contains("blur_score"), "应含清晰度: {text}");
        assert!(text.contains("overall"), "应含综合分: {text}");
    }

    #[tokio::test]
    async fn pose_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/pose")
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
    async fn pose_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/pose")
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
    async fn depth_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/depth")
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
    async fn depth_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/depth")
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
    async fn human_parsing_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/human-parsing")
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
    async fn human_parsing_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/human-parsing")
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
    fn quality_dto_should_serialize_all_fields() {
        let r = ImageQualityReport {
            blur_score: 100.0,
            brightness: 128.0,
            contrast: 50.0,
            noise: 0.5,
            overall: 0.8,
        };
        let dto = QualityDto::from(&r);
        let json = serde_json::to_value(&dto).unwrap();
        assert!((json["blur_score"].as_f64().unwrap() - 100.0).abs() < 1e-5);
        assert!((json["overall"].as_f64().unwrap() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn pose_dto_should_include_coco_keypoint_names() {
        let det = Detection::new("person", 0, 10.0, 20.0, 110.0, 220.0, 0.9);
        let kps: Vec<Keypoint> = (0..17)
            .map(|i| Keypoint::new(i as f32 * 10.0, i as f32 * 5.0, 0.8))
            .collect();
        let pose = PoseResult::new(det, kps);
        let dto = PoseDto::from(&pose);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["keypoints"].as_array().unwrap().len(), 17);
        assert_eq!(json["keypoints"][0]["name"], "nose");
        assert_eq!(json["keypoints"][16]["name"], "right_ankle");
    }

    #[test]
    fn depth_dto_should_compute_stats_and_downsample() {
        let mask = FloatMask::from_raw(128, 128, vec![0.5; 128 * 128]).unwrap();
        let dto = DepthDto::from(&mask);
        assert_eq!(dto.width, 128);
        assert_eq!(dto.height, 128);
        assert!((dto.min - 0.5).abs() < 1e-5);
        assert!((dto.mean - 0.5).abs() < 1e-5);
        // 128 / 64 向上取整步长 = 2, preview 应为 64*64
        assert!(dto.preview.len() <= 64 * 64);
        assert!(!dto.preview.is_empty());
    }

    #[test]
    fn depth_dto_should_handle_small_mask() {
        let mask = FloatMask::from_raw(4, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                                   9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]).unwrap();
        let dto = DepthDto::from(&mask);
        assert!((dto.min - 1.0).abs() < 1e-5);
        assert!((dto.max - 16.0).abs() < 1e-5);
        assert!((dto.mean - 8.5).abs() < 1e-4);
    }
}
