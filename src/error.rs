use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// 应用错误类型。
///
/// 与 yolo-rs 契约一致：逻辑错误返回 HTTP 200 + `{"status":"error","data":{"error":...}}`，
/// 使消费方按 JSON body 的 status 字段判断结果。
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("请求格式错误: {0}")]
    BadRequest(String),

    #[error("模型未配置: {0}（请在环境变量中设置对应模型路径）")]
    ModelNotConfigured(String),

    #[error("能力未启用: {0}（不在 ARGUS_ENABLED_CAPABILITIES 白名单中）")]
    CapabilityDisabled(String),

    #[error("推理失败: {0}")]
    Inference(String),

    #[error("图像处理失败: {0}")]
    Image(String),

    #[error("内部错误: {0}")]
    Internal(String),

    #[error("推理过载: {0}")]
    Overload(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // 服务错误统一返回 HTTP 200（信封语义），TraceLayer 按状态码分级失效，
        // 因此必须在响应构造处主动落日志，否则生产环境无失败痕迹可查。
        match &self {
            // 客户端输入问题：量可能大且非服务故障，降为 debug 不刷屏
            AppError::BadRequest(_) => tracing::debug!("{}", self.to_string()),
            // 背压拒绝属预期内的过载信号
            AppError::Overload(_) => tracing::warn!("请求被背压拒绝: {}", self.to_string()),
            // 服务侧故障（模型未配置/推理失败/图像异常/内部错误）必须留痕
            _ => tracing::error!("请求处理失败: {}", self.to_string()),
        }
        let body = json!({
            "status": "error",
            "data": { "error": self.to_string() }
        });
        (StatusCode::OK, axum::Json(body)).into_response()
    }
}

impl From<rust_onnx_infer::VisionError> for AppError {
    fn from(e: rust_onnx_infer::VisionError) -> Self {
        AppError::Inference(e.to_string())
    }
}

impl From<image::ImageError> for AppError {
    fn from(e: image::ImageError) -> Self {
        AppError::Image(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bad_request_should_return_200_with_error_json() {
        let resp = AppError::BadRequest("missing file".into()).into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "error");
        assert!(json["data"]["error"].as_str().unwrap().contains("missing file"));
    }

    #[test]
    fn model_not_configured_should_mention_capability() {
        let err = AppError::ModelNotConfigured("face_detect".into());
        assert!(err.to_string().contains("face_detect"));
    }

    #[test]
    fn capability_disabled_should_mention_whitelist_for_troubleshooting() {
        // 与"模型未配置"必须可区分:排错时一眼分清"缺配置"还是"被白名单挡住"
        let err = AppError::CapabilityDisabled("ocr".into());
        let msg = err.to_string();
        assert!(msg.contains("ocr") && msg.contains("白名单"), "实际: {msg}");
    }

    #[test]
    fn vision_error_should_convert_to_inference() {
        let ve = rust_onnx_infer::VisionError::image("test error");
        let app: AppError = ve.into();
        assert!(matches!(app, AppError::Inference(_)));
    }
}
