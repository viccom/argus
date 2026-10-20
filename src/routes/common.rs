//! 路由层公共工具：multipart 字段解析、图像解码、推理背压。
//!
//! 各能力端点（detection/classification/face/text/analysis）共用，
//! 与具体端点解耦，新增引擎接入直接从这里引用。

use std::time::Duration;

use image::ImageDecoder;
use rust_onnx_infer::Image;

use crate::error::AppError;
use crate::routes::health::AppState;

/// multipart 流错误分类：body 超限与解析错误必须可区分。
///
/// axum 的 `DefaultBodyLimit` 超限在 multipart 流上表现为 413（LengthLimitError），
/// 其余为 400。曾一律映射为"解析失败"，客户端无法判别该缩图还是该修格式。
pub(crate) fn map_field_err(
    e: axum::extract::multipart::MultipartError,
    context: &str,
) -> AppError {
    if e.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        AppError::TooLarge(format!("请求体超过上限（请缩图或调大 ARGUS_MAX_BODY_MB）: {e}"))
    } else {
        AppError::BadRequest(format!("{context}: {e}"))
    }
}

/// 读取单个 multipart 文本字段。
pub(crate) async fn field_text(field: axum::extract::multipart::Field<'_>) -> Result<String, AppError> {
    let bytes = field
        .bytes()
        .await
        .map_err(|e| map_field_err(e, "读取字段失败"))?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

/// 通用单字段文本提取 + 泛型解析（空文本视为未传，返回 None）。
pub(crate) async fn parse_field_text<T>(
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

/// 在 spawn_blocking 中解码图像字节：应用 EXIF Orientation 摆正 + 像素总数闸门。
///
/// JPEG/PNG 解码是 CPU 密集操作（大图可达百毫秒级），与推理一样必须隔离到
/// 阻塞线程池，否则并发大图请求会占住 tokio async worker（拖死 health 等轻端点）。
///
/// `image::load_from_memory` 不处理 EXIF Orientation（`image` crate 立场：不隐式变换），
/// 实测 orientation=6 的照片检测 2 脸 vs 摆正后 38 脸——所有端点必须在解码边界统一摆正。
/// `max_image_pixels > 0` 时在像素缓冲分配**之前**用 `decoder.dimensions()` 闸门
/// （20MB 以内的 PNG 可达上亿像素，解码成 GB 级 RGBA）；0 = 不限。
pub(crate) async fn decode_image(bytes: Vec<u8>, max_image_pixels: u64) -> Result<Image, AppError> {
    tokio::task::spawn_blocking(move || {
        decode_image_inner(bytes, max_image_pixels).map(|dyn_img| Image::from_dynamic(&dyn_img))
    })
    .await
    .map_err(|e| AppError::Internal(format!("图像解码任务失败: {e}")))?
}

fn decode_image_inner(bytes: Vec<u8>, max_image_pixels: u64) -> Result<image::DynamicImage, AppError> {
    let reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| AppError::Image(format!("图像格式探测失败: {e}")))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(AppError::from)?;
    let (w, h) = decoder.dimensions();
    if max_image_pixels > 0 && u64::from(w) * u64::from(h) > max_image_pixels {
        return Err(AppError::TooLarge(format!(
            "图像 {w}x{h} 共 {} 像素，超过上限 {max_image_pixels}（ARGUS_MAX_IMAGE_PIXELS）",
            u64::from(w) * u64::from(h)
        )));
    }
    let orientation = decoder.orientation().unwrap_or_else(|e| {
        // EXIF 损坏不应让整张图不可用：按无取向处理，留 debug 痕迹
        tracing::debug!("EXIF orientation 读取失败，按无变换处理: {e}");
        image::metadata::Orientation::NoTransforms
    });
    let mut dyn_img = image::DynamicImage::from_decoder(decoder).map_err(AppError::from)?;
    // NoTransforms 分支是空操作：无 EXIF 的摄像头帧/PNG 零开销，行为逐字节不变
    dyn_img.apply_orientation(orientation);
    Ok(dyn_img)
}

/// 获取推理许可（全局并发背压信号量）。
///
/// 超时由 `ARGUS_ACQUIRE_TIMEOUT_MS` 控制（默认 30000ms 排队式）：
/// - `> 0`：在信号量上等待,超时返回 Overload(内网低负载排队更友好);
/// - `0`：try_acquire fail-fast——耗尽即拒,不排队(生产 GPU 防 arena 雪崩,
///   与 max_concurrent=pool_size 对齐)。
pub(crate) async fn acquire_permit(state: &AppState) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
    let timeout_ms = state.config.acquire_timeout_ms;
    if timeout_ms == 0 {
        // fail-fast:try_acquire 不阻塞,拿不到立即拒绝
        state
            .inference_semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| AppError::Overload("推理并发已满(fail-fast),请稍后重试".into()))
    } else {
        tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            state.inference_semaphore.clone().acquire_owned(),
        )
        .await
        .map_err(|_| AppError::Overload("推理排队超时，请稍后重试".into()))?
        .map_err(|_| AppError::Internal("信号量已关闭".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x2 基准图：左半红、右半蓝。不对称，取向变换能从像素判别。
    fn base_test_image() -> image::DynamicImage {
        let mut img = image::DynamicImage::new_rgb8(4, 2);
        for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if x < 2 {
                image::Rgb([220, 20, 20])
            } else {
                image::Rgb([20, 20, 220])
            };
        }
        img
    }

    fn encode_jpeg(base: &image::DynamicImage) -> Vec<u8> {
        let mut buf = Vec::new();
        base.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)
            .unwrap();
        buf
    }

    /// 向 JPEG 字节注入携带 orientation 的 EXIF APP1 段（SOI 之后）。
    /// TIFF 小端、IFD0 单条目 tag 274（SHORT）——与 image crate 解析路径对齐。
    fn inject_exif_orientation(jpeg: Vec<u8>, orientation: u16) -> Vec<u8> {
        let mut exif: Vec<u8> = Vec::new();
        exif.extend_from_slice(&[0x49, 0x49, 0x2A, 0x00]); // "II" + 42
        exif.extend_from_slice(&8u32.to_le_bytes()); // IFD0 偏移
        exif.extend_from_slice(&1u16.to_le_bytes()); // 条目数
        exif.extend_from_slice(&0x0112u16.to_le_bytes()); // tag 274
        exif.extend_from_slice(&3u16.to_le_bytes()); // 类型 SHORT
        exif.extend_from_slice(&1u32.to_le_bytes()); // count
        exif.extend_from_slice(&orientation.to_le_bytes()); // 值内联（前 2 字节）
        exif.extend_from_slice(&0u16.to_le_bytes()); // 值字段补齐
        exif.extend_from_slice(&0u32.to_le_bytes()); // next IFD

        let mut out = Vec::with_capacity(jpeg.len() + exif.len() + 16);
        out.extend_from_slice(&jpeg[..2]); // SOI
        out.extend_from_slice(&[0xFF, 0xE1]); // APP1
        out.extend_from_slice(&((exif.len() + 8) as u16).to_be_bytes()); // 段长（含自身 2 字节）
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(&exif);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    fn expected_after(img: &image::DynamicImage, orientation: u16) -> image::DynamicImage {
        use image::metadata::Orientation as O;
        let mut expect = img.clone();
        expect.apply_orientation(match orientation {
            2 => O::FlipHorizontal,
            3 => O::Rotate180,
            4 => O::FlipVertical,
            5 => O::Rotate90FlipH,
            6 => O::Rotate90,
            7 => O::Rotate270FlipH,
            8 => O::Rotate270,
            _ => O::NoTransforms,
        });
        expect
    }

    /// 8 种 EXIF 取向全部按 image crate 语义摆正；对照侧是"同一份 JPEG 字节解码后
    /// 手工 apply"——两侧共享解码结果，规避 JPEG 有损压缩带来的像素差。
    #[test]
    fn decode_should_apply_each_exif_orientation() {
        let base = base_test_image();
        let jpeg = encode_jpeg(&base);
        for v in 1u16..=8 {
            let with_exif = inject_exif_orientation(jpeg.clone(), v);
            let got = decode_image_inner(with_exif, 0).unwrap();
            // 对照必须基于同一解码基线：先解无标签版本再 apply
            let plain = decode_image_inner(jpeg.clone(), 0).unwrap();
            let expect = expected_after(&plain, v);
            assert_eq!(
                (got.width(), got.height()),
                (expect.width(), expect.height()),
                "orientation={v} 尺寸不符"
            );
            assert_eq!(
                got.to_rgb8().into_raw(),
                expect.to_rgb8().into_raw(),
                "orientation={v} 像素不符"
            );
        }
    }

    /// 无 EXIF 的 PNG（safebot 摄像头帧同型）：行为逐字节不变。
    #[test]
    fn decode_should_leave_png_without_exif_untouched() {
        let base = base_test_image();
        let mut png = Vec::new();
        base.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let got = decode_image_inner(png, 0).unwrap();
        assert_eq!((got.width(), got.height()), (4, 2));
        assert_eq!(got.to_rgb8().into_raw(), base.to_rgb8().into_raw());
    }

    /// 像素闸门在解码前拒绝（decoder.dimensions() 预检），错误码 too_large。
    #[test]
    fn pixel_gate_should_reject_before_decoding() {
        let base = base_test_image(); // 4x2 = 8 像素
        let mut png = Vec::new();
        base.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let err = decode_image_inner(png.clone(), 7).unwrap_err();
        assert_eq!(err.code(), "too_large");
        assert!(err.to_string().contains("ARGUS_MAX_IMAGE_PIXELS"));
        // 0 = 不限（默认，复现既有行为）
        assert!(decode_image_inner(png, 0).is_ok());
    }
}
