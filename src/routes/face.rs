//! 人脸路由：`/face/detect`（人脸检测 + 5 关键点）与 `/face/recognize`（提取嵌入 + 余弦比对）。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rust_onnx_infer::engines::face_detection::FaceDetectResult;
use rust_onnx_infer::model::Keypoint;

use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image, field_text};
use crate::routes::health::AppState;

// ---- DTO ----

#[derive(Serialize)]
struct KeypointDto {
    x: f32,
    y: f32,
    score: f32,
}

impl From<Keypoint> for KeypointDto {
    fn from(k: Keypoint) -> Self {
        Self {
            x: k.x,
            y: k.y,
            score: k.score,
        }
    }
}

#[derive(Serialize)]
struct FaceDetectDto {
    confidence: f64,
    bbox: [f64; 4],
    landmarks: Vec<KeypointDto>,
}

impl From<&FaceDetectResult> for FaceDetectDto {
    fn from(f: &FaceDetectResult) -> Self {
        let d = &f.detection;
        Self {
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
            landmarks: f.landmarks.iter().copied().map(KeypointDto::from).collect(),
        }
    }
}

#[derive(Serialize)]
struct FaceMatchDto {
    /// 比对到的已知人脸标签（无匹配时为 None）。
    label: Option<String>,
    /// 余弦相似度 [-1, 1]。
    similarity: f32,
    /// 本脸在结果列表中的索引。
    face_index: usize,
    confidence: f64,
    bbox: [f64; 4],
}

// ---- /face/recognize 的已知人脸库条目 ----

#[derive(Deserialize)]
struct KnownFace {
    label: String,
    /// 嵌入向量（与识别模型输出维度一致）。
    embedding: Vec<f32>,
}

// ---- multipart 参数 ----

struct FaceDetectParams {
    file: Vec<u8>,
}

async fn parse_face_multipart(mut multipart: Multipart) -> Result<FaceDetectParams, AppError> {
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
            // 消费未知字段避免 body 残留
            let _ = field.bytes().await;
        }
    }
    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(FaceDetectParams { file })
}

struct FaceRecognizeParams {
    file: Vec<u8>,
    known: Vec<KnownFace>,
    /// 余弦相似度匹配阈值，低于该值不认作匹配。默认 0.5。
    threshold: f32,
}

async fn parse_recognize_multipart(
    mut multipart: Multipart,
) -> Result<FaceRecognizeParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut known_json: Option<String> = None;
    let mut threshold: Option<f32> = None;

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
            "known_embeddings" => {
                known_json = Some(field_text(field).await?);
            }
            "threshold" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    threshold = Some(
                        text.parse::<f32>()
                            .map_err(|_| AppError::BadRequest(format!("threshold 无法解析: {text}")))?,
                    );
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    let known: Vec<KnownFace> = match known_json {
        Some(ref json_str) if !json_str.trim().is_empty() => serde_json::from_str(json_str)
            .map_err(|e| AppError::BadRequest(format!("known_embeddings JSON 解析失败: {e}")))?,
        _ => Vec::new(),
    };

    Ok(FaceRecognizeParams {
        file,
        known,
        threshold: threshold.unwrap_or(0.5),
    })
}

// ---- 余弦相似度 ----

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---- handlers ----

/// POST /face/detect — 人脸检测，返回每张人脸的框 + 5 关键点。
pub async fn face_detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_face_multipart(multipart).await?;
    let image = decode_image(params.file).await?;

    let pool = state.engines.face_detection().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let faces = pool.run(move |e| e.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<FaceDetectDto> = faces.iter().map(FaceDetectDto::from).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "face_detect", count, inference_ms = elapsed as u64, "推理完成");

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "faces": dtos,
            "count": count,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /face/recognize — 对图中每张人脸提取嵌入，与已知库做余弦比对。
///
/// `known_embeddings` 为 JSON 数组：`[{"label":"alice","embedding":[...]}]`。
/// 无已知库时仅返回嵌入向量（可用于注册）。
pub async fn face_recognize(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_recognize_multipart(multipart).await?;
    let image = decode_image(params.file).await?;

    let det_pool = state.engines.face_detection().await?;
    let rec_pool = state.engines.face_recognition().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let image_clone = image.clone();
    let faces = det_pool.run(move |e| e.predict(&image_clone)).await?;

    // 逐脸提取嵌入(每次经池取实例,天然公平)
    let mut matches: Vec<FaceMatchDto> = Vec::with_capacity(faces.len());
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(faces.len());

    for (idx, face) in faces.iter().enumerate() {
        let landmarks = face.landmarks.clone();
        let img = image.clone();
        let emb = rec_pool.run(move |e| e.extract(&img, &landmarks)).await?;
        embeddings.push(emb.clone());

        // 在已知库中找最高相似度
        let (best_label, best_sim) = params
            .known
            .iter()
            .map(|k| (k.label.as_str(), cosine_similarity(&emb, &k.embedding)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, sim)| {
                (
                    if sim >= params.threshold {
                        Some(label.to_string())
                    } else {
                        None
                    },
                    sim,
                )
            })
            .unwrap_or((None, 0.0));

        let d = &face.detection;
        matches.push(FaceMatchDto {
            label: best_label,
            similarity: best_sim,
            face_index: idx,
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
        });
    }
    let elapsed = t0.elapsed().as_millis();
    tracing::info!(endpoint = "face_recognize", count = matches.len(), inference_ms = elapsed as u64, "推理完成");

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "matches": matches,
            "count": matches.len(),
            "embeddings": embeddings,
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
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"threshold\"\r\n\r\n\
             0.5\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
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
    async fn face_detect_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/face/detect")
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
    async fn face_detect_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/face/detect")
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
    async fn face_recognize_without_file_returns_bad_request() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/face/recognize")
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
    async fn face_recognize_when_model_not_configured_returns_error() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/face/recognize")
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
    fn cosine_similarity_should_be_1_for_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_should_be_0_for_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_should_handle_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn face_detect_dto_should_include_landmarks() {
        use rust_onnx_infer::model::Detection;
        let det = Detection::new("face", 0, 10.0, 20.0, 110.0, 220.0, 0.95);
        let lms = vec![
            Keypoint::new(30.0, 40.0, 0.9),
            Keypoint::new(50.0, 40.0, 0.9),
            Keypoint::new(40.0, 60.0, 0.8),
            Keypoint::new(35.0, 80.0, 0.7),
            Keypoint::new(45.0, 80.0, 0.7),
        ];
        let face = FaceDetectResult::new(det, lms);
        let dto = FaceDetectDto::from(&face);
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["confidence"], 0.95);
        assert_eq!(json["landmarks"].as_array().unwrap().len(), 5);
        assert_eq!(json["landmarks"][0]["x"], 30.0);
    }
}
