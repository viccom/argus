use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// API key 鉴权中间件（轻量化设计，不引入用户/组体系）。
///
/// token 未配置时全放行（宽松兜底）。
/// 配置后，请求需通过以下任一方式携带 key：
/// - `Authorization: Bearer {key}`
/// - `X-API-Key: {key}`
pub async fn auth_middleware(
    axum::extract::State(token): axum::extract::State<Option<Arc<String>>>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(ref expected) = token {
        let provided = extract_api_key(&req);
        match provided {
            Some(key) => {
                if !constant_time_eq(&key, expected) {
                    return unauthorized("无效的 API key");
                }
            }
            None => return unauthorized("缺少 API key（Authorization: Bearer <key> 或 X-API-Key: <key>）"),
        }
    }

    next.run(req).await
}

/// 从请求头提取 API key：优先 Authorization: Bearer，其次 X-API-Key。
fn extract_api_key(req: &Request) -> Option<String> {
    if let Some(h) = req.headers().get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = h.strip_prefix("Bearer ") {
            return Some(token.to_string());
        }
    }
    req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn unauthorized(message: &str) -> Response {
    tracing::warn!("鉴权失败: {message}");
    let body = serde_json::json!({
        "status": "error",
        "data": { "error": message }
    });
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// 常量时间字节比较，防止 key 校验的 timing side-channel。
/// 长度不同直接返回 false（key 长度不作为秘密保护）。
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn test_router(token: Option<Arc<String>>) -> Router {
        let handler = || get(|| async { "ok" });
        Router::new()
            .route("/health", handler())
            .layer(from_fn_with_state(token, auth_middleware))
    }

    async fn body_text(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    #[tokio::test]
    async fn no_token_configured_should_pass_all() {
        let app = test_router(None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_text(resp).await, "ok");
    }

    #[tokio::test]
    async fn bearer_header_with_correct_key_should_pass() {
        let app = test_router(Some(Arc::new("my-secret".into())));
        let resp = app
            .oneshot(
                Request::get("/health")
                    .header("authorization", "Bearer my-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(body_text(resp).await.contains("ok"));
    }

    #[tokio::test]
    async fn x_api_key_header_with_correct_key_should_pass() {
        let app = test_router(Some(Arc::new("my-secret".into())));
        let resp = app
            .oneshot(
                Request::get("/health")
                    .header("x-api-key", "my-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(body_text(resp).await.contains("ok"));
    }

    #[tokio::test]
    async fn missing_auth_header_when_token_configured_should_reject() {
        let app = test_router(Some(Arc::new("my-secret".into())));
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("缺少"), "实际: {text}");
    }

    #[tokio::test]
    async fn wrong_key_should_be_rejected() {
        let app = test_router(Some(Arc::new("my-secret".into())));
        let resp = app
            .oneshot(
                Request::get("/health")
                    .header("authorization", "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(text.contains("无效"), "实际: {text}");
    }

    #[test]
    fn constant_time_eq_should_match_identical() {
        assert!(constant_time_eq("abc123", "abc123"));
    }

    #[test]
    fn constant_time_eq_should_reject_different_same_length() {
        assert!(!constant_time_eq("abc123", "abc124"));
    }

    #[test]
    fn constant_time_eq_should_reject_different_lengths() {
        assert!(!constant_time_eq("short", "a-much-longer-token"));
        assert!(!constant_time_eq("", "token"));
    }
}
