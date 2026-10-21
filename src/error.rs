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

    /// 请求体或图像内容超过上限（body 字节数 / 像素总数）。
    /// axum 的 DefaultBodyLimit 超限在 multipart 错误里报告为 413，可与之区分。
    #[error("内容过大: {0}")]
    TooLarge(String),

    /// `model` 参数指向未注册/已删除的动态槽。消息内嵌可用槽名清单，
    /// 不泄露文件系统路径（槽名来自 URL，是外部输入）。
    #[error("未知模型槽: {0}")]
    UnknownModel(String),
}

impl AppError {
    /// 机器可读错误码。消费方按 `data.code` 做重试分类，不解析 error 原文。
    pub fn code(&self) -> &'static str {
        match self {
            AppError::BadRequest(_) => "bad_request",
            AppError::ModelNotConfigured(_) => "model_not_configured",
            AppError::CapabilityDisabled(_) => "capability_disabled",
            AppError::Inference(_) => "inference_failed",
            AppError::Image(_) => "unsupported_image",
            AppError::Internal(_) => "internal",
            AppError::Overload(_) => "overload",
            AppError::TooLarge(_) => "too_large",
            AppError::UnknownModel(_) => "unknown_model",
        }
    }

    /// 客户端可否安全重试。过载（背压拒绝/排队超时）是唯一预期内可重试项；
    /// 超限重发同样的请求体必然再超限（应缩图或调 ARGUS_MAX_BODY_MB）。
    pub fn retryable(&self) -> bool {
        matches!(self, AppError::Overload(_))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // 服务错误统一返回 HTTP 200（信封语义），TraceLayer 按状态码分级失效，
        // 因此必须在响应构造处主动落日志，否则生产环境无失败痕迹可查。
        match &self {
            // 客户端输入问题：量可能大且非服务故障，降为 debug 不刷屏
            AppError::BadRequest(_) | AppError::TooLarge(_) | AppError::UnknownModel(_) => {
                tracing::debug!("{}", self.to_string())
            }
            // 背压拒绝属预期内的过载信号
            AppError::Overload(_) => tracing::warn!("请求被背压拒绝: {}", self.to_string()),
            // 服务侧故障（模型未配置/推理失败/图像异常/内部错误）必须留痕
            _ => tracing::error!("请求处理失败: {}", self.to_string()),
        }
        let body = json!({
            "status": "error",
            "data": {
                "error": self.to_string(),
                "code": self.code(),
                "retryable": self.retryable(),
            }
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

    #[tokio::test]
    async fn error_envelope_should_carry_machine_readable_code() {
        // 消费方(homepix client.go)按 data.code 分类重试策略,不猜 error 原文
        let resp = AppError::BadRequest("x".into()).into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"]["code"], "bad_request");
        assert_eq!(json["data"]["retryable"], false);
    }

    #[tokio::test]
    async fn overload_should_be_the_only_retryable_code() {
        let resp = AppError::Overload("busy".into()).into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"]["code"], "overload");
        assert_eq!(json["data"]["retryable"], true);
    }

    #[test]
    fn every_variant_should_have_distinct_stable_code() {
        // 码表是对外契约:两变体撞码或改码都会打断消费方的重试分类
        let samples = [
            AppError::BadRequest("".into()),
            AppError::ModelNotConfigured("".into()),
            AppError::CapabilityDisabled("".into()),
            AppError::Inference("".into()),
            AppError::Image("".into()),
            AppError::Internal("".into()),
            AppError::Overload("".into()),
            AppError::TooLarge("".into()),
            AppError::UnknownModel("".into()),
        ];
        let mut codes: Vec<&str> = samples.iter().map(|e| e.code()).collect();
        let dedup: std::collections::HashSet<&str> = codes.drain(..).collect();
        assert_eq!(dedup.len(), samples.len(), "错误码必须互不相同");
        assert!(!samples.iter().any(|e| e.code().is_empty()));
    }

    #[test]
    fn too_large_should_not_be_retryable() {
        // 超限重试同一请求体必然再超限:客户端要缩图或调上限,不是重试
        let err = AppError::TooLarge("body".into());
        assert_eq!(err.code(), "too_large");
        assert!(!err.retryable());
    }

    #[test]
    fn unknown_model_should_have_stable_code_and_not_retryable() {
        let err = AppError::UnknownModel("nope（可用动态槽: a, b）".into());
        assert_eq!(err.code(), "unknown_model");
        assert!(!err.retryable());
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
