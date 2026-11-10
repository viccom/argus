//! 日志初始化(04-log-spec §2:统一 JSON 行)。
//!
//! 全体系目标格式是 JSON 行:AI/Agent 可直接 `jq` 抽字段、无正则歧义、token 更省。
//! 改造前 argus 用 tracing 默认文本格式(`tracing-subscriber` 的 fmt 层),
//! 采集侧(collector)只能靠正则解析且字段名与 slog 不一致 —— 本模块统一为 JSON。
//!
//! 字段对照(tracing JSON 层):
//!   timestamp → RFC3339 带时区(04 §2 要求 time 带偏移);level → 大写级别名;
//!   fields.message → 消息体;target → 模块路径;span → 请求上下文(request_id/method/uri)。

use std::io;

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// 未设 `RUST_LOG` 时的默认过滤级别(与改造前一致:info)。
pub fn default_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
}

/// 组装 JSON 行 subscriber。生产与测试共用这一份格式配置,
/// 避免"测试断言的格式"与"线上实际格式"分叉。
pub fn json_subscriber<W>(writer: W, filter: EnvFilter) -> impl tracing::Subscriber
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .with_writer(writer)
        .finish()
}

/// 生产初始化:JSON 行 → stdout(warden 负责落盘与轮转)。
pub fn init() {
    json_subscriber(io::stdout, default_filter()).init();
}