//! Web 控制台路由:`GET /ui` 返回内嵌的单文件静态页面(web/index.html)。
//!
//! 页面与 API 同源部署(由本服务直接提供),无 CORS 依赖、无外部 CDN,
//! 适用于离线/内网工业环境。页面本身在 auth 中间件中豁免——浏览器加载页面
//! 时无法携带自定义 header,页面内的 API 调用仍需 API key(页面提供输入框)。

use axum::response::Html;

pub async fn ui() -> Html<&'static str> {
    Html(include_str!("../../web/index.html"))
}
