//! 场景检测路由:`/ppe`(工地 PPE 违章检测)与 `/fire-smoke`(火焰/烟雾检测)。
//!
//! 两者复用 DetectionEngine 与 `/detect` 的全部参数语义(SAHI/confidence/target_classes/roi),
//! 仅模型与能力键不同——共享实现在 [`crate::routes::detection::run_detect`]。

use axum::extract::{Multipart, State};
use axum::Json;
use serde_json::Value;

use crate::error::AppError;
use crate::routes::detection::{run_detect, DetectCap};
use crate::routes::health::AppState;

/// POST /ppe — PPE 检测(10 类:Hardhat/Mask/NO-Hardhat/NO-Mask/NO-Safety Vest/
/// Person/Safety Cone/Safety Vest/machinery/vehicle,含违章反类)。
pub async fn ppe_detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    run_detect(state, multipart, DetectCap::Ppe, "ppe").await
}

/// POST /fire-smoke — 火焰/烟雾检测(fire/smoke 两类)。
pub async fn fire_smoke_detect(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    run_detect(state, multipart, DetectCap::FireSmoke, "fire_smoke").await
}

#[cfg(test)]
mod tests {
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

    /// 只含 confidence 字段、无 file 的 multipart body(测缺 file 路径)。
    fn multipart_no_file() -> Vec<u8> {
        let boundary = "----testboundary";
        format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"confidence\"\r\n\r\n\
             0.5\r\n\
             --{boundary}--\r\n"
        )
        .into_bytes()
    }

    async fn post(path: &'static str, body: Vec<u8>) -> String {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                Request::post(path)
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn ppe_without_file_returns_bad_request() {
        // 参数校验先于引擎加载:无 file 应报参数错误而非"模型未配置"
        let text = post("/ppe", multipart_no_file()).await;
        assert!(text.contains("error"), "{text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn ppe_when_model_not_configured_returns_error() {
        let text = post("/ppe", multipart_png_with_fields(&[])).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }

    #[tokio::test]
    async fn ppe_with_invalid_confidence_returns_bad_request() {
        let text = post("/ppe", multipart_png_with_fields(&[("confidence", "0.05")])).await;
        assert!(text.contains("0.1"), "应提示下限 0.1: {text}");
    }

    #[tokio::test]
    async fn fire_smoke_without_file_returns_bad_request() {
        let text = post("/fire-smoke", multipart_no_file()).await;
        assert!(text.contains("error"), "{text}");
        assert!(text.contains("file"), "应提及缺少 file: {text}");
    }

    #[tokio::test]
    async fn fire_smoke_when_model_not_configured_returns_error() {
        let text = post("/fire-smoke", multipart_png_with_fields(&[])).await;
        assert!(
            text.contains("未配置") || text.contains("model"),
            "应提示模型未配置: {text}"
        );
    }
}
