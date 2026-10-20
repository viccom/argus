//! 行人重识别路由:`POST /reid`(OSNet x1.0,512 维 L2 归一化 embedding)。
//!
//! 输入整图或行人框列表(`boxes`,像素坐标 `[[x1,y1,x2,y2],...]`),逐框裁剪提取特征。
//! 框会 clamp 到原图界,clamp 后退化(宽/高 < 1px)的框报 400;未传 boxes 视为整图单区域。
//! 跨框/跨图相似度由消费方对 embedding 做余弦(引擎同款 `ReidEngine::cosine_similarity`)。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::imaging::Rect;

use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image, field_text};
use crate::routes::health::AppState;

// ---- DTO ----

#[derive(Serialize)]
struct RegionEmbeddingDto {
    /// 实际参与推理的裁剪框 (x1, y1, x2, y2),已 clamp 到原图界(像素坐标)
    bbox: [f32; 4],
    /// L2 归一化特征(512 维,OSNet x1.0)
    embedding: Vec<f32>,
}

// ---- multipart 参数 ----

struct ReidParams {
    file: Vec<u8>,
    boxes: Option<Vec<[f64; 4]>>,
}

async fn parse_reid_multipart(mut multipart: Multipart) -> Result<ReidParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    let mut boxes: Option<Vec<[f64; 4]>> = None;

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
            "boxes" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    boxes = Some(serde_json::from_str(&text).map_err(|e| {
                        AppError::BadRequest(format!(
                            "字段 boxes 无法解析(应为 JSON [[x1,y1,x2,y2],...]): {e}"
                        ))
                    })?);
                }
            }
            _ => {
                let _ = field_text(field).await?; // 消费未知字段，保持流一致
            }
        }
    }

    let file = file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?;
    Ok(ReidParams { file, boxes })
}

/// 校验像素框并 clamp 到原图界;clamp 后退化的框(宽/高 < 1px)报 400。
fn clamp_boxes(boxes: &[[f64; 4]], width: usize, height: usize) -> Result<Vec<Rect>, AppError> {
    let (w, h) = (width as f64, height as f64);
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let x1 = b[0].clamp(0.0, w);
            let y1 = b[1].clamp(0.0, h);
            let x2 = b[2].clamp(0.0, w);
            let y2 = b[3].clamp(0.0, h);
            let (bw, bh) = (x2 - x1, y2 - y1);
            if bw < 1.0 || bh < 1.0 {
                return Err(AppError::BadRequest(format!(
                    "boxes[{i}] 在图 {width}x{height} 内退化(宽 {bw:.1} / 高 {bh:.1}),请检查坐标"
                )));
            }
            Ok(Rect::new(
                x1.round() as i32,
                y1.round() as i32,
                bw.round() as i32,
                bh.round() as i32,
            ))
        })
        .collect()
}

// ---- handler ----

/// POST /reid — 行人重识别特征提取(multipart: file, boxes?)。
pub async fn reid_embed(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_reid_multipart(multipart).await?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;

    let rects = match &params.boxes {
        Some(boxes) if !boxes.is_empty() => {
            clamp_boxes(boxes, image.width(), image.height())?
        }
        _ => vec![Rect::new(0, 0, image.width() as i32, image.height() as i32)],
    };
    let count = rects.len();

    let pool = state.engines.reid().await?;
    let _permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool
        .run(move |e| {
            rects
                .iter()
                .map(|r| {
                    e.embed(&image, *r).map(|embedding| {
                        (
                            [
                                r.x as f32,
                                r.y as f32,
                                (r.x + r.width) as f32,
                                (r.y + r.height) as f32,
                            ],
                            embedding,
                        )
                    })
                })
                .collect::<rust_onnx_infer::Result<Vec<_>>>()
        })
        .await?;
    let elapsed = t0.elapsed().as_millis();

    let dim = results.first().map(|(_, emb)| emb.len()).unwrap_or(0);
    let dtos: Vec<RegionEmbeddingDto> = results
        .into_iter()
        .map(|(bbox, embedding)| RegionEmbeddingDto { bbox, embedding })
        .collect();
    tracing::info!(endpoint = "reid", count, dim, inference_ms = elapsed as u64, "推理完成");

    Ok(Json(json!({
        "status": "ok",
        "data": {
            "regions": dtos,
            "count": count,
            "dim": dim,
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

    /// 构造 multipart body:有效 1x1 PNG file 字段 + 可选附加字段。
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

    fn multipart_without_file() -> Vec<u8> {
        let boundary = "----testboundary";
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"boxes\"\r\n\r\n\
             []\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
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

    async fn post(app: axum::Router, body: Vec<u8>) -> String {
        let resp = app
            .oneshot(
                Request::post("/reid")
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
    async fn reid_without_file_returns_bad_request() {
        let text = post(build_router(test_state(), None), multipart_without_file()).await;
        assert!(text.contains("error"), "应返回错误: {text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn reid_when_model_not_configured_returns_error() {
        let text = post(build_router(test_state(), None), multipart_with_png(&[])).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[tokio::test]
    async fn reid_with_invalid_image_returns_image_error() {
        let text = post(build_router(test_state(), None), multipart_invalid_image()).await;
        assert!(text.contains("error"), "应返回错误: {text}");
    }

    #[tokio::test]
    async fn reid_with_broken_boxes_json_returns_bad_request() {
        let text = post(
            build_router(test_state(), None),
            multipart_with_png(&[("boxes", "[[0,0," )]),
        )
        .await;
        assert!(text.contains("boxes"), "应指出 boxes 无法解析: {text}");
    }

    #[tokio::test]
    async fn reid_with_wrong_coord_count_returns_bad_request() {
        let text = post(
            build_router(test_state(), None),
            multipart_with_png(&[("boxes", "[[1,2,3]]")]),
        )
        .await;
        assert!(text.contains("boxes"), "应指出 boxes 坐标数不对: {text}");
    }

    #[tokio::test]
    async fn reid_with_degenerate_box_returns_bad_request() {
        // 1x1 图上 clamp 后宽高 < 1px
        let text = post(
            build_router(test_state(), None),
            multipart_with_png(&[("boxes", "[[10,10,10,10]]")]),
        )
        .await;
        assert!(text.contains("退化"), "应指出框退化: {text}");
    }

    #[test]
    fn clamp_boxes_should_clamp_and_reject_degenerate() {
        let boxes = vec![[5.0, 5.0, 50.0, 60.0]];
        let rects = clamp_boxes(&boxes, 40, 30).unwrap();
        assert_eq!(rects.len(), 1);
        assert_eq!((rects[0].x, rects[0].y), (5, 5));
        assert_eq!((rects[0].width, rects[0].height), (35, 25));

        // 完全出界 → clamp 后退化 → 400
        assert!(clamp_boxes(&[[100.0, 100.0, 120.0, 130.0]], 40, 30).is_err());
        // 负坐标 clamp 到 0
        let rects = clamp_boxes(&[[-10.0, -10.0, 20.0, 20.0]], 40, 30).unwrap();
        assert_eq!((rects[0].x, rects[0].y), (0, 0));
        assert_eq!((rects[0].width, rects[0].height), (20, 20));
    }

    #[test]
    fn region_embedding_dto_should_serialize_all_fields() {
        let dto = RegionEmbeddingDto {
            bbox: [0.0, 0.0, 128.0, 256.0],
            embedding: vec![0.25; 512],
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["bbox"][2], 128.0);
        assert_eq!(json["embedding"].as_array().unwrap().len(), 512);
        assert_eq!(json["embedding"][0], 0.25);
    }
}
