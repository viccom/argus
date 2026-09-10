//! 路由层公共工具：multipart 字段解析、图像解码、推理背压。
//!
//! 各能力端点（detection/classification/face/text/analysis）共用，
//! 与具体端点解耦，新增引擎接入直接从这里引用。

use std::time::Duration;

use rust_onnx_infer::Image;

use crate::error::AppError;
use crate::routes::health::AppState;

/// 背压排队超时：超过该时间仍未拿到推理许可则拒绝请求。
const INFERENCE_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);

/// 读取单个 multipart 文本字段。
pub(crate) async fn field_text(field: axum::extract::multipart::Field<'_>) -> Result<String, AppError> {
    let bytes = field
        .bytes()
        .await
        .map_err(|e| AppError::BadRequest(format!("读取字段失败: {e}")))?;
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
