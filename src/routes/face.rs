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
    /// 5 关键点（B1）：与 /face/detect 同源同值，消费方做姿态/尺寸质量门控
    /// （低头/侧脸不进簇心）无需二次调用检测端点。
    landmarks: Vec<KeypointDto>,
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
    gate: FaceGate,
}

async fn parse_face_multipart(mut multipart: Multipart) -> Result<FaceDetectParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut gate_raw = (None, None);
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
            "min_face_px" => gate_raw.0 = parse_gate_u32(&name, field).await?,
            "max_faces" => gate_raw.1 = parse_gate_usize(&name, field).await?,
            _ => {
                // 消费未知字段避免 body 残留
                let _ = field.bytes().await;
            }
        }
    }
    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(FaceDetectParams { file, gate: FaceGate::from_raw(gate_raw)? })
}

struct FaceRecognizeParams {
    file: Vec<u8>,
    known: Vec<KnownFace>,
    /// 余弦相似度匹配阈值，低于该值不认作匹配。默认 0.5。
    threshold: f32,
    gate: FaceGate,
}

async fn parse_recognize_multipart(
    mut multipart: Multipart,
) -> Result<FaceRecognizeParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut known_json: Option<String> = None;
    let mut threshold: Option<f32> = None;
    let mut gate_raw = (None, None);

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
            "min_face_px" => gate_raw.0 = parse_gate_u32(&name, field).await?,
            "max_faces" => gate_raw.1 = parse_gate_usize(&name, field).await?,
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
        gate: FaceGate::from_raw(gate_raw)?,
    })
}

// ---- 数量/尺寸门控 ----

/// 人脸门控参数（`/face/detect` 与 `/face/recognize` 共用）。
///
/// 成本语义：门控发生在**提向量之前**——集会照曾为 38 张背景小脸逐一提向量
/// （1475ms），消费方不需要它们。默认全关，行为与历史完全一致。
#[derive(Clone, Copy, Default)]
struct FaceGate {
    /// bbox 短边绝对像素下限，低于该值的脸被丢弃。0 = 不限。
    min_face_px: u32,
    /// 按检测置信度取前 N 张脸。0 = 不限。
    max_faces: usize,
}

impl FaceGate {
    fn from_raw(raw: (Option<u32>, Option<usize>)) -> Result<Self, AppError> {
        Ok(Self {
            min_face_px: raw.0.unwrap_or(0),
            max_faces: raw.1.unwrap_or(0),
        })
    }

    fn is_off(&self) -> bool {
        self.min_face_px == 0 && self.max_faces == 0
    }
}

async fn parse_gate_u32(
    field_name: &str,
    field: axum::extract::multipart::Field<'_>,
) -> Result<Option<u32>, AppError> {
    parse_gate_num::<u32>(field_name, field).await
}

async fn parse_gate_usize(
    field_name: &str,
    field: axum::extract::multipart::Field<'_>,
) -> Result<Option<usize>, AppError> {
    parse_gate_num::<usize>(field_name, field).await
}

async fn parse_gate_num<T: std::str::FromStr>(
    field_name: &str,
    field: axum::extract::multipart::Field<'_>,
) -> Result<Option<T>, AppError> {
    let text = field_text(field).await?;
    if text.trim().is_empty() {
        return Ok(None);
    }
    text.parse::<T>()
        .map(Some)
        .map_err(|_| AppError::BadRequest(format!("{field_name} 无法解析: {text}")))
}

/// 应用门控：先按 bbox 短边过滤，再按检测置信度降序取前 N（稳定排序，
/// 同分保持原检测顺序）。开启 max_faces 时返回序列为置信度降序。
fn gate_faces(faces: Vec<FaceDetectResult>, gate: FaceGate) -> Vec<FaceDetectResult> {
    if gate.is_off() {
        return faces;
    }
    let mut kept: Vec<FaceDetectResult> = faces
        .into_iter()
        .filter(|f| {
            gate.min_face_px == 0 || {
                let d = &f.detection;
                let short_side = (d.x2() - d.x1()).min(d.y2() - d.y1());
                short_side >= f64::from(gate.min_face_px)
            }
        })
        .collect();
    if gate.max_faces > 0 && kept.len() > gate.max_faces {
        kept.sort_by(|a, b| {
            b.detection
                .confidence()
                .partial_cmp(&a.detection.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kept.truncate(gate.max_faces);
    }
    kept
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

// ---- B3 批量上传 ----

/// 单请求文件数上限。与 `ARGUS_MAX_BODY_MB` 协同：16 张原尺寸家庭照易超 20MB
/// body 上限，客户端应送 JPEG 派生图（错误码 too_large 会明确提示）。
pub(crate) const MAX_BATCH_FILES: usize = 16;

#[derive(Serialize)]
#[serde(untagged)]
enum FaceBatchItem {
    Ok {
        file_index: usize,
        matches: Vec<FaceMatchDto>,
        count: usize,
        embeddings: Vec<Vec<f32>>,
        embed_dim: Option<usize>,
        inference_ms: u128,
    },
    Err {
        file_index: usize,
        error: String,
    },
}

/// 批量参数：多个 file 字段 + 共享的 known_embeddings/threshold/门控。
struct FaceBatchParams {
    files: Vec<Vec<u8>>,
    known: Vec<KnownFace>,
    threshold: f32,
    gate: FaceGate,
}

async fn parse_face_batch_multipart(mut multipart: Multipart) -> Result<FaceBatchParams, AppError> {
    let mut files: Vec<Vec<u8>> = Vec::new();
    let mut known_json: Option<String> = None;
    let mut threshold: Option<f32> = None;
    let mut gate_raw = (None, None);

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| crate::routes::common::map_field_err(e, "multipart 解析失败"))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| crate::routes::common::map_field_err(e, "读取文件失败"))?
                    .to_vec();
                files.push(bytes);
            }
            "known_embeddings" => known_json = Some(field_text(field).await?),
            "threshold" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    threshold = Some(
                        text.parse::<f32>().map_err(|_| {
                            AppError::BadRequest(format!("threshold 无法解析: {text}"))
                        })?,
                    );
                }
            }
            "min_face_px" => gate_raw.0 = parse_gate_u32(&name, field).await?,
            "max_faces" => gate_raw.1 = parse_gate_usize(&name, field).await?,
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    if files.is_empty() {
        return Err(AppError::BadRequest("缺少 file 字段（至少一张图片）".into()));
    }
    if files.len() > MAX_BATCH_FILES {
        return Err(AppError::BadRequest(format!(
            "单请求最多 {MAX_BATCH_FILES} 个 file 字段，当前 {}（批量应送压缩后的派生图）",
            files.len()
        )));
    }
    let known: Vec<KnownFace> = match known_json {
        Some(ref json_str) if !json_str.trim().is_empty() => serde_json::from_str(json_str)
            .map_err(|e| AppError::BadRequest(format!("known_embeddings JSON 解析失败: {e}")))?,
        _ => Vec::new(),
    };

    Ok(FaceBatchParams {
        files,
        known,
        threshold: threshold.unwrap_or(0.5),
        gate: FaceGate::from_raw(gate_raw)?,
    })
}

/// 单图识别全流程（detect → 门控 → 逐脸提向量 → 比对），单端点与批量共享。
/// permit 在此获取：整段（检测 + 逐脸 extract）受同一背压保护。
#[allow(clippy::too_many_arguments)]
async fn recognize_faces(
    state: &AppState,
    image: rust_onnx_infer::Image,
    known: &[KnownFace],
    threshold: f32,
    gate: FaceGate,
) -> Result<(Vec<FaceMatchDto>, Vec<Vec<f32>>), AppError> {
    let det_pool = state.engines.face_detection().await?;
    let rec_pool = state.engines.face_recognition().await?;

    let _permit = acquire_permit(state).await?;

    let t0 = std::time::Instant::now();
    let image_clone = image.clone();
    let faces = det_pool.run(move |e| e.predict(&image_clone)).await?;
    // 门控必须发生在逐脸提向量**之前**——省的就是每脸一次 rec 池推理
    let faces = gate_faces(faces, gate);

    let mut matches: Vec<FaceMatchDto> = Vec::with_capacity(faces.len());
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(faces.len());
    for face in &faces {
        let landmarks = face.landmarks.clone();
        let img = image.clone();
        let emb = rec_pool.run(move |e| e.extract(&img, &landmarks)).await?;
        embeddings.push(emb.clone());

        let (best_label, best_sim) = known
            .iter()
            .map(|k| (k.label.as_str(), cosine_similarity(&emb, &k.embedding)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, sim)| {
                (
                    if sim >= threshold {
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
            face_index: embeddings.len() - 1,
            confidence: d.confidence(),
            bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
            landmarks: face
                .landmarks
                .iter()
                .copied()
                .map(KeypointDto::from)
                .collect(),
        });
    }
    let _ = t0;
    Ok((matches, embeddings))
}

// ---- handlers ----

/// POST /face/detect — 人脸检测，返回每张人脸的框 + 5 关键点。
pub async fn face_detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_face_multipart(multipart).await?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;

    let pool = state.engines.face_detection().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let faces = pool.run(move |e| e.predict(&image)).await?;
    // 门控在响应组装前：尺寸/数量过滤按需裁剪检出列表
    let faces = gate_faces(faces, params.gate);
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<FaceDetectDto> = faces.iter().map(FaceDetectDto::from).collect();
    let count = dtos.len();
    tracing::info!(endpoint = "face_detect", count, inference_ms = elapsed as u64, "推理完成");

    let (model, pipeline) = state.engines.provenance("face_detection").await;
    Ok(Json(json!({
        "status": "ok",
        "data": {
            "model": model,
            "pipeline": pipeline,
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
    let image = decode_image(params.file, state.config.max_image_pixels).await?;

    let t0 = std::time::Instant::now();
    let (matches, embeddings) = recognize_faces(
        &state,
        image,
        &params.known,
        params.threshold,
        params.gate,
    )
    .await?;
    let elapsed = t0.elapsed().as_millis();
    tracing::info!(endpoint = "face_recognize", count = matches.len(), inference_ms = elapsed as u64, "推理完成");

    let (model, pipeline) = state.engines.provenance("face_recognition").await;
    Ok(Json(json!({
        "status": "ok",
        "data": {
            "model": model,
            "pipeline": pipeline,
            "matches": matches,
            "count": matches.len(),
            "embeddings": embeddings,
            "inference_ms": elapsed,
        }
    })))
}

/// POST /face/recognize/batch — 多图上传批量人脸识别（B3）。
///
/// 语义照抄 `/detect/batch`：逐图串行（省的是 HTTP 往返与请求开销，非推理并行）、
/// 单图失败带 `error` 不炸整批、每图独立背压 permit；模型未配置是整批级错误。
/// 共享参数：known_embeddings/threshold/min_face_px/max_faces 对全部图生效。
pub async fn face_recognize_batch(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_face_batch_multipart(multipart).await?;
    // 模型未配置是整批级错误（无引擎可用），不进入逐图隔离
    let _det_pool = state.engines.face_detection().await?;
    let _rec_pool = state.engines.face_recognition().await?;

    let t0 = std::time::Instant::now();
    let mut results = Vec::with_capacity(params.files.len());
    let mut failed = 0usize;
    for (idx, bytes) in params.files.into_iter().enumerate() {
        let item = match decode_image(bytes, state.config.max_image_pixels).await {
            Ok(image) => {
                let per_image_t0 = std::time::Instant::now();
                match recognize_faces(
                    &state,
                    image,
                    &params.known,
                    params.threshold,
                    params.gate,
                )
                .await
                {
                    Ok((matches, embeddings)) => {
                        let embed_dim = embeddings.first().map(|e| e.len());
                        FaceBatchItem::Ok {
                            file_index: idx,
                            count: matches.len(),
                            matches,
                            embeddings,
                            embed_dim,
                            inference_ms: per_image_t0.elapsed().as_millis(),
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        tracing::warn!(file_index = idx, error = %e, "批量识别单图失败（已隔离）");
                        FaceBatchItem::Err {
                            file_index: idx,
                            error: e.to_string(),
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                tracing::warn!(file_index = idx, error = %e, "批量识别单图解码失败（已隔离）");
                FaceBatchItem::Err {
                    file_index: idx,
                    error: e.to_string(),
                }
            }
        };
        results.push(item);
    }
    let total = results.len();
    let elapsed = t0.elapsed().as_millis();
    tracing::info!(
        endpoint = "face_recognize_batch",
        total,
        failed,
        elapsed_ms = elapsed as u64,
        "批量识别完成"
    );

    let (model, pipeline) = state.engines.provenance("face_recognition").await;
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
    async fn oversized_body_should_report_too_large_code() {
        // body 超限曾被报成"读取文件失败: Error parsing multipart"（与坏文件不可区分）；
        // axum 对 DefaultBodyLimit 超限报 413，据此映射独立的 too_large 错误码
        let config = Config {
            max_body_mb: 1,
            ..Default::default()
        };
        let app = build_router(AppState::new(config.clone(), EngineManager::new(config)), None);

        // 构造 >1MB 的 file 字段
        let big = vec![b'a'; 1_100_000];
        let boundary = "----testboundary";
        let mut body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"big.jpg\"\r\n\r\n"
        )
        .into_bytes();
        body.extend_from_slice(&big);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let resp = app
            .oneshot(
                Request::post("/face/detect")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("\"code\":\"too_large\""), "应报 too_large: {text}");
        assert!(text.contains("ARGUS_MAX_BODY_MB"), "应提示调整上限: {text}");
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


    /// 通用 multipart 构造：若干文本字段 + 1x1 PNG file 字段；post 到指定路径。
    async fn post_multipart(path: &str, text_fields: &[(&str, &str)]) -> String {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let boundary = "----testboundary";
        let mut full = Vec::new();
        for (name, value) in text_fields {
            full.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        full.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"t.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        full.extend_from_slice(&buf.into_inner());
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post(path)
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(full))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    // ---- A3 门控单测 ----

    fn gate_face(x1: f64, y1: f64, x2: f64, y2: f64, conf: f64) -> FaceDetectResult {
        let det = rust_onnx_infer::model::Detection::new("face", 0, x1, y1, x2, y2, conf);
        FaceDetectResult::new(det, Vec::new())
    }

    #[test]
    fn gate_off_by_default_should_keep_all_faces_in_original_order() {
        // 默认全关：行为与历史完全一致（顺序、数量均不动）
        let faces = vec![
            gate_face(0., 0., 100., 200., 0.6),
            gate_face(0., 0., 10., 10., 0.9),
        ];
        let gated = gate_faces(faces, FaceGate::default());
        assert_eq!(gated.len(), 2);
        assert_eq!(gated[0].detection.confidence(), 0.6, "不开 max_faces 不得重排");
    }

    #[test]
    fn gate_min_face_px_should_drop_small_boxes_by_short_side() {
        let faces = vec![
            gate_face(0., 0., 100., 200., 0.6), // 短边 100
            gate_face(0., 0., 79., 300., 0.9),  // 短边 79 < 80
            gate_face(0., 0., 80., 80., 0.5),   // 短边恰为 80（边界含）
        ];
        let gated = gate_faces(
            faces,
            FaceGate { min_face_px: 80, max_faces: 0 },
        );
        assert_eq!(gated.len(), 2, "短边 79 应被丢弃，80 保留（>= 语义）");
    }

    #[test]
    fn gate_max_faces_should_keep_top_n_by_confidence() {
        let faces = vec![
            gate_face(0., 0., 100., 100., 0.6),
            gate_face(0., 0., 100., 100., 0.95),
            gate_face(0., 0., 100., 100., 0.8),
        ];
        let gated = gate_faces(faces, FaceGate { min_face_px: 0, max_faces: 2 });
        assert_eq!(gated.len(), 2);
        assert_eq!(gated[0].detection.confidence(), 0.95, "按置信度降序");
        assert_eq!(gated[1].detection.confidence(), 0.8);
    }

    #[test]
    fn gate_combined_should_filter_then_truncate() {
        let faces = vec![
            gate_face(0., 0., 200., 200., 0.7),  // 过尺寸
            gate_face(0., 0., 10., 200., 0.99),  // 被尺寸过滤
            gate_face(0., 0., 150., 150., 0.9),  // 过尺寸
        ];
        let gated = gate_faces(faces, FaceGate { min_face_px: 50, max_faces: 1 });
        assert_eq!(gated.len(), 1);
        assert_eq!(gated[0].detection.confidence(), 0.9, "先滤后取，高置信小脸不得挤掉大脸");
    }

    #[test]
    fn gate_max_faces_zero_or_equal_should_not_reorder() {
        let faces = vec![gate_face(0., 0., 9., 9., 0.6), gate_face(0., 0., 9., 9., 0.9)];
        let gated = gate_faces(faces, FaceGate { min_face_px: 5, max_faces: 0 });
        assert_eq!(gated[0].detection.confidence(), 0.6, "max_faces=0 只滤不排");
    }

    #[tokio::test]
    async fn face_detect_with_invalid_min_face_px_returns_bad_request() {
        let text = post_multipart("/face/detect", &[("min_face_px", "-5"), ("file", "@png")]).await;
        assert!(text.contains("min_face_px"), "应指出参数无法解析: {text}");
        assert!(text.contains("bad_request"), "应为 bad_request 码: {text}");
    }

    #[test]
    fn face_match_dto_should_carry_landmarks_for_quality_gating() {
        // B1：消费方靠 landmarks 做姿态门控（眼距/roll 角），缺字段即断
        let dto = FaceMatchDto {
            label: None,
            similarity: 0.0,
            face_index: 0,
            confidence: 0.9,
            bbox: [1.0, 2.0, 3.0, 4.0],
            landmarks: vec![
                KeypointDto { x: 1.5, y: 2.0, score: 0.9 },
                KeypointDto { x: 2.5, y: 2.0, score: 0.5 },
            ],
        };
        let json = serde_json::to_value(&dto).unwrap();
        let lms = json["landmarks"].as_array().unwrap();
        assert_eq!(lms.len(), 2);
        assert_eq!(lms[0]["x"], 1.5);
        assert_eq!(lms[1]["score"], 0.5);
    }

    // ---- B3 批量端点测试 ----

    async fn post_batch_files(n_files: usize, extra: &[(&str, &str)]) -> String {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let boundary = "----testboundary";
        let mut body = Vec::new();
        for (name, value) in extra {
            body.extend_from_slice(
                format!(
                    "--{b}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n",
                    b = boundary
                )
                .as_bytes(),
            );
        }
        for _ in 0..n_files {
            body.extend_from_slice(
                format!(
                    "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"t.png\"\r\nContent-Type: image/png\r\n\r\n",
                    b = boundary
                )
                .as_bytes(),
            );
            body.extend_from_slice(&png);
            body.extend_from_slice(format!("\r\n--{b}--\r\n", b = boundary).as_bytes());
            // 修正：终止段只应出现一次，去掉上面这行的终止符，循环外加
        }
        // 移除误加的中间终止段，统一以最终终止段收尾
        // （上面循环里每图后都写了 --boundary--，需要去掉前面的，只保留最后一个）
        // 简化处理：重建 body
        let mut body2 = Vec::new();
        for (name, value) in extra {
            body2.extend_from_slice(
                format!(
                    "--{b}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n",
                    b = boundary
                )
                .as_bytes(),
            );
        }
        for _ in 0..n_files {
            body2.extend_from_slice(
                format!(
                    "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"t.png\"\r\nContent-Type: image/png\r\n\r\n",
                    b = boundary
                )
                .as_bytes(),
            );
            body2.extend_from_slice(&png);
            body2.extend_from_slice(b"\r\n");
        }
        body2.extend_from_slice(format!("--{b}--\r\n", b = boundary).as_bytes());

        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post("/face/recognize/batch")
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body2))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn batch_without_files_returns_bad_request() {
        let text = post_batch_files(0, &[("threshold", "0.5")]).await;
        assert!(text.contains("file"), "应提示缺 file: {text}");
    }

    #[tokio::test]
    async fn batch_exceeding_max_files_returns_bad_request() {
        let text = post_batch_files(17, &[]).await;
        assert!(text.contains("16"), "应提示上限 16: {text}");
    }

    #[tokio::test]
    async fn batch_when_model_not_configured_fails_whole_batch() {
        // 无引擎可用属整批级错误，不进入逐图隔离（与 /detect/batch 同语义）
        let text = post_batch_files(3, &[]).await;
        assert!(text.contains("未配置") || text.contains("model"), "{text}");
        assert!(!text.contains("results"), "整批失败不应返回 results: {text}");
    }
}
