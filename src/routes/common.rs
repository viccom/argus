//! 路由层公共工具：multipart 字段解析、图像解码、推理背压。
//!
//! 各能力端点（detection/classification/face/text/analysis）共用，
//! 与具体端点解耦，新增引擎接入直接从这里引用。

use std::time::Duration;

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

/// 在 spawn_blocking 中解码图像字节。
///
/// JPEG/PNG 解码是 CPU 密集操作（大图可达百毫秒级），与推理一样必须隔离到
/// 阻塞线程池，否则并发大图请求会占住 tokio async worker（拖死 health 等轻端点）。
pub(crate) async fn decode_image(bytes: Vec<u8>) -> Result<Image, AppError> {
    tokio::task::spawn_blocking(move || {
        image::load_from_memory(&bytes).map(|dyn_img| Image::from_dynamic(&dyn_img))
    })
    .await
    .map_err(|e| AppError::Internal(format!("图像解码任务失败: {e}")))?
    .map_err(AppError::from)
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
