//! 目标检测路由：`/detect`（含 SAHI 切片）与 `/detect/obb`（旋转框）。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::engines::obb_detection::ObbResult;
use rust_onnx_infer::model::Detection;
use rust_onnx_infer::sahi::config::SahiConfig;
use rust_onnx_infer::sahi::sliced_predictor::predict_sliced_detection;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image, field_text, parse_field_text};
use crate::routes::detect_filter::{parse_target_classes, DetectFilter, Polygon};
use crate::routes::health::AppState;

// ---- DTO ----

#[derive(Serialize)]
pub(crate) struct DetectionDto {
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
///
/// `confidence`/`target_classes`/`roi` 为请求级后过滤（语义见 `detect_filter`），
/// 引擎以低阈值构造后按请求过滤，SAHI 路径经 `SahiConfig::min_confidence` 合并前过滤。
struct DetectParams {
    file: Vec<u8>,
    sahi: bool,
    slice_w: Option<i32>,
    slice_h: Option<i32>,
    overlap: Option<f64>,
    confidence: Option<f64>,
    target_classes: Option<Vec<i32>>,
    roi: Option<Polygon>,
}

async fn parse_detect_multipart(mut multipart: Multipart) -> Result<DetectParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut sahi = false;
    let mut slice_w: Option<i32> = None;
    let mut slice_h: Option<i32> = None;
    let mut overlap: Option<f64> = None;
    let mut confidence: Option<f64> = None;
    let mut target_classes: Option<Vec<i32>> = None;
    let mut roi: Option<Polygon> = None;

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
            "confidence" => {
                confidence = parse_field_text(&name, field).await?;
            }
            "target_classes" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    target_classes = Some(parse_target_classes(&text)?);
                }
            }
            "roi" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    roi = Some(Polygon::parse(&text)?);
                }
            }
            _ => { /* 忽略未知字段 */ }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(DetectParams {
        file,
        sahi,
        slice_w,
        slice_h,
        overlap,
        confidence,
        target_classes,
        roi,
    })
}

// ---- handlers ----

/// 检测族能力:ppe/fire_smoke 复用 DetectionEngine 与本文件的全部参数语义
/// (SAHI/confidence/target_classes/roi),仅模型与能力键不同。
pub(crate) enum DetectCap {
    Detection,
    Ppe,
    FireSmoke,
}

impl DetectCap {
    async fn pool(
        &self,
        state: &AppState,
    ) -> Result<std::sync::Arc<crate::engine::EnginePool<rust_onnx_infer::engines::detection::DetectionEngine>>, AppError>
    {
        match self {
            Self::Detection => state.engines.detection().await,
            Self::Ppe => state.engines.ppe().await,
            Self::FireSmoke => state.engines.fire_smoke().await,
        }
    }

    /// 能力键（A4 溯源与 /models 一致）。
    fn key(&self) -> &'static str {
        match self {
            Self::Detection => "detection",
            Self::Ppe => "ppe",
            Self::FireSmoke => "fire_smoke",
        }
    }
}

/// 检测族共享实现:参数校验先于图像解码与引擎加载,非法参数不消耗推理资源。
pub(crate) async fn run_detect(
    state: AppState,
    multipart: Multipart,
    cap: DetectCap,
    endpoint: &'static str,
) -> Result<Json<Value>, AppError> {
    let params = parse_detect_multipart(multipart).await?;
    let filter = DetectFilter::new(params.confidence, params.target_classes, params.roi)?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;
    let (img_w, img_h) = (image.width(), image.height());

    let pool = cap.pool(&state).await?;

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
        // 合并前过滤:与引擎直接以该阈值构造严格等价,避免低分碎片先放大高分框几何
        cfg.min_confidence = Some(filter.confidence as f32);
        let result = pool
            .run(move |e| predict_sliced_detection(e, e, &image, &cfg))
            .await?;
        (result.detections, Some(result.num_slices))
    } else {
        let dets = pool.run(move |e| e.predict(&image)).await?;
        (dets, None)
    };
    let elapsed = t0.elapsed().as_millis();
    let detections = filter.apply(detections, img_w, img_h);

    let dtos: Vec<DetectionDto> = detections.iter().map(DetectionDto::from).collect();
    let count = dtos.len();
    tracing::info!(
        endpoint,
        count,
        sahi = params.sahi,
        confidence = filter.confidence,
        inference_ms = elapsed as u64,
        "推理完成"
    );

    let mut data = json!({
        "detections": dtos,
        "count": count,
        "inference_ms": elapsed,
    });
    if let Some(n) = num_slices {
        data["sahi_slices"] = json!(n);
    }
    let (model, pipeline) = state.engines.provenance(cap.key()).await;
    data["model"] = model;
    data["pipeline"] = pipeline;

    Ok(Json(json!({ "status": "ok", "data": data })))
}

/// POST /detect — 目标检测（可选 SAHI 切片 + 请求级 confidence/target_classes/roi 过滤）。
pub async fn detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    run_detect(state, multipart, DetectCap::Detection, "detect").await
}

/// POST /detect/obb — OBB 旋转框检测。
pub async fn detect_obb(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_detect_multipart(multipart).await?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;

    let pool = state.engines.obb().await?;

    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool.run(move |e| e.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<ObbDto> = results.iter().map(ObbDto::from).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "detect_obb", count, inference_ms = elapsed as u64, "推理完成");

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

    /// 1x1 有效 PNG + 任意文本字段的 multipart body。
    fn multipart_png_with_fields(fields: &[(&str, &str)]) -> Vec<u8> {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();

        let boundary = "----testboundary";
        let mut full = Vec::new();
        for (name, value) in fields {
            full.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
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

    async fn post_detect(body: Vec<u8>) -> String {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/detect")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn detect_with_confidence_below_floor_returns_bad_request() {
        // 校验先于引擎加载:未配模型也应报参数错误而非"模型未配置"
        let text = post_detect(multipart_png_with_fields(&[("confidence", "0.05")])).await;
        assert!(text.contains("error"), "{text}");
        assert!(text.contains("0.1"), "应提示下限 0.1: {text}");
        assert!(!text.contains("未配置"), "参数错误应先于模型检查: {text}");
    }

    #[tokio::test]
    async fn detect_with_malformed_roi_returns_bad_request() {
        let text = post_detect(multipart_png_with_fields(&[("roi", "0.1,0.1,0.9,0.9")])).await;
        assert!(text.contains("error") && text.contains("roi"), "{text}");
        let text = post_detect(multipart_png_with_fields(&[("roi", "[[0,0],[1,1]]")])).await;
        assert!(text.contains("3 个点"), "{text}");
    }

    #[tokio::test]
    async fn detect_with_malformed_target_classes_returns_bad_request() {
        let text = post_detect(multipart_png_with_fields(&[("target_classes", "person,car")])).await;
        assert!(text.contains("error") && text.contains("target_classes"), "{text}");
        let text = post_detect(multipart_png_with_fields(&[("target_classes", "[]")])).await;
        assert!(text.contains("不能为空"), "{text}");
    }

    #[tokio::test]
    async fn detect_with_valid_filter_params_should_reach_engine() {
        // 合法参数全部通过校验,走到引擎加载(未配模型 → 模型未配置错误,证明参数被接受)
        let text = post_detect(multipart_png_with_fields(&[
            ("confidence", "0.6"),
            ("target_classes", "[0, 2]"),
            ("roi", "[[0,0],[1,0],[1,1],[0,1]]"),
        ]))
        .await;
        assert!(text.contains("未配置") || text.contains("model"), "应到达引擎加载: {text}");
    }

    #[tokio::test]
    async fn detect_with_empty_optional_fields_should_be_treated_as_absent() {
        // 前端表单常提交空字符串,须视为未传而非解析失败
        let text = post_detect(multipart_png_with_fields(&[
            ("confidence", ""),
            ("target_classes", ""),
            ("roi", ""),
        ]))
        .await;
        assert!(text.contains("未配置") || text.contains("model"), "空字段应视为未传: {text}");
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
