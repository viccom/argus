//! Web 控制台路由:`GET /ui` 返回内嵌的单文件静态页面(web/index.html)。
//!
//! 页面与 API 同源部署(由本服务直接提供),无 CORS 依赖、无外部 CDN,
//! 适用于离线/内网工业环境。页面本身在 auth 中间件中豁免——浏览器加载页面
//! 时无法携带自定义 header,页面内的 API 调用仍需 API key。
//!
//! API key 输入框默认值由服务端注入(占位符 `__DEFAULT_API_KEY__` 替换为
//! 配置的 auth_token):测试环境免手动输入。代价是任何能访问 /ui 的人可见
//! token——/ui 本就免鉴权,内网测试可接受;生产暴露公网时建议关闭注入或
//! 换用独立前端。

use axum::extract::State;
use axum::response::Html;

use crate::routes::health::AppState;

pub async fn ui(State(state): State<AppState>) -> Html<String> {
    let html = include_str!("../../web/index.html");
    let key = state.config.auth_token.as_deref().unwrap_or("");
    Html(html.replace("__DEFAULT_API_KEY__", key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::EngineManager;
    use crate::routes::health::build_router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn state_with_token(token: Option<&str>) -> AppState {
        let config = Config {
            auth_token: token.map(str::to_string),
            ..Default::default()
        };
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn body(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    #[tokio::test]
    async fn ui_should_inject_configured_token_as_default_key() {
        let app = build_router(state_with_token(Some("secret-key")), None);
        let resp = app
            .oneshot(Request::get("/ui").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let html = body(resp).await;
        assert!(
            html.contains(r#"value="secret-key""#),
            "应注入配置 token 作默认 key"
        );
    }

    #[tokio::test]
    async fn ui_should_not_leak_placeholder_when_no_token() {
        let app = build_router(state_with_token(None), None);
        let resp = app
            .oneshot(Request::get("/ui").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = body(resp).await;
        assert!(
            !html.contains("__DEFAULT_API_KEY__"),
            "未配置 token 时不应残留占位符"
        );
    }
}
