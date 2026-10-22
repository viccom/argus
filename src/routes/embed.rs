//! 文本 Embedding 路由：`POST /embed/text`。
//!
//! 引擎：`TextEmbeddingEngine`（jina-embeddings-v2-base-zh，MaskedMean 池化 +
//! L2 归一化 + maxlen 512，双语 A/B 选型见 fork 计划表 2026-10-21）。
//! 逐条 `embed()` 循环（非 `embed_batch`）：本机实测逐条更快（2026-10-21 T8）且
//! 严格跨批逐位确定；空字符串在 HTTP 层前置校验（引擎兜底是 inference_failed，
//! 语义码不对——fork 审查移交事项）。

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AppError;
use crate::routes::common::{acquire_permit, map_json_rejection};
use crate::routes::health::AppState;

/// 批量条数上限（防单请求压垮池：128×2000 字符已有界）。
const MAX_TEXTS: usize = 128;
/// 单条字符上限。512 token 截断上界之上留足余量（≈500 汉字已截断），
/// 2000 只防滥用不伤正常文档；常量不放 env（YAGNI，设计 §3.5）。
const MAX_TEXT_CHARS: usize = 2000;

// ---- DTO ----

#[derive(Deserialize)]
pub struct EmbedTextRequest {
    texts: Vec<String>,
    /// `passage`（缺省）/`query`，两值等价实现（jina v2 短查询无需指令前缀）。
    mode: Option<String>,
    /// 仅接受 true/缺省——引擎构造期已归一化，v1 不支持关闭（显式报错优于静默忽略）。
    normalize: Option<bool>,
}

/// 成功响应组装（独立函数便于无模型单测 json 形状）。
fn ok_response(
    queue_ms: u128,
    model: Value,
    pipeline: Value,
    dim: usize,
    embeddings: Vec<Vec<f32>>,
    inference_ms: u128,
) -> Value {
    json!({
        "status": "ok",
        "data": {
            "queue_ms": queue_ms,
            "model": model,
            "pipeline": pipeline,
            "dim": dim,
            "normalized": true,
            "count": embeddings.len(),
            "embeddings": embeddings,
            "inference_ms": inference_ms,
        }
    })
}

// ---- handler ----

/// POST /embed/text — 批量文本 Embedding（JSON: texts[1..128]/mode?/normalize?）。
///
/// 校验顺序（硬纪律 4：校验先于懒加载触发）：parse → 逐条校验 → mode/normalize →
/// 引擎 → permit → 逐条推理；计时在 permit 之后（B5 纪律）。
pub async fn embed_text(
    State(state): State<AppState>,
    body: Result<Json<EmbedTextRequest>, JsonRejection>,
) -> Result<Json<Value>, AppError> {
    let Json(req) = body.map_err(map_json_rejection)?;

    if req.texts.is_empty() {
        return Err(AppError::BadRequest("texts 不能为空（1..128 条）".into()));
    }
    if req.texts.len() > MAX_TEXTS {
        return Err(AppError::BadRequest(format!(
            "texts 数量超过上限（最多 {MAX_TEXTS} 条，当前 {} 条）",
            req.texts.len()
        )));
    }
    // 逐条校验：消息用下标定位，绝不回显原文（隐私，需求 §7）
    for (i, t) in req.texts.iter().enumerate() {
        if t.is_empty() {
            return Err(AppError::BadRequest(format!("texts[{i}] 为空字符串")));
        }
        let chars = t.chars().count();
        if chars > MAX_TEXT_CHARS {
            return Err(AppError::BadRequest(format!(
                "texts[{i}] 超过 {MAX_TEXT_CHARS} 字符上限（当前 {chars} 字符）"
            )));
        }
    }
    // mode 两值等价（设计 §3.4），不进 pipeline 串；非法值显式拒绝
    if let Some(m) = &req.mode {
        if m != "passage" && m != "query" {
            return Err(AppError::BadRequest("mode 非法（仅支持 passage/query）".into()));
        }
    }
    if req.normalize == Some(false) {
        return Err(AppError::BadRequest(
            "normalize=false 暂不支持（引擎构造期归一化）".into(),
        ));
    }

    let pool = state.engines.text_embedding().await?;

    let permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let mut embeddings = Vec::with_capacity(req.texts.len());
    for text in &req.texts {
        let t = text.clone();
        embeddings.push(pool.run(move |e| e.embed(&t)).await?);
    }
    let elapsed = t0.elapsed().as_millis();
    // dim 用 env 声明值：loader 加载期已校验与引擎一致，此处可信且免触碰池实例
    let dim = state.config.text_embedding_dim.unwrap_or_default();
    let count = embeddings.len();
    tracing::info!(endpoint = "embed_text", count, dim, inference_ms = elapsed as u64, "推理完成");

    let (model, pipeline) = state.engines.provenance("text_embedding").await;
    Ok(Json(ok_response(permit.queue_ms, model, pipeline, dim, embeddings, elapsed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::EngineManager;
    use crate::routes::health::{build_router, AppState};
    use axum::body::Body;
    use axum::http::Request;
    use serde_json::Value;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let config = Config::default();
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    /// 三件套齐备但白名单只留 detection：能力被关。
    fn disabled_state() -> AppState {
        let config = Config {
            text_embedding_model: Some("/nonexistent/model.onnx".into()),
            text_embedding_tokenizer: Some("/nonexistent/tokenizer.json".into()),
            text_embedding_dim: Some(512),
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn post_json(state: AppState, body: String) -> Value {
        let app = build_router(state, None);
        let resp = app
            .oneshot(
                Request::post("/embed/text")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    }

    /// 断言错误信封：status=error + data.code 精确匹配。
    fn assert_code(body: &Value, code: &str) {
        assert_eq!(body["status"], "error", "应为错误信封: {body}");
        assert_eq!(body["data"]["code"], code, "实际: {body}");
    }

    #[tokio::test]
    async fn missing_texts_field_returns_bad_request() {
        let body = post_json(test_state(), r#"{}"#.into()).await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn empty_texts_returns_bad_request() {
        let body = post_json(test_state(), r#"{"texts": []}"#.into()).await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("texts"),
            "错误消息应指向 texts: {body}"
        );
    }

    #[tokio::test]
    async fn too_many_texts_returns_bad_request() {
        let texts: Vec<String> = (0..129).map(|i| format!("文本{i}")).collect();
        let body = post_json(
            test_state(),
            serde_json::json!({ "texts": texts }).to_string(),
        )
        .await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("128"),
            "应提及 128 条上限: {body}"
        );
    }

    #[tokio::test]
    async fn texts_at_128_limit_should_pass_count_validation() {
        // 128 条是合法上界：计数校验放行后落到"模型未配置"（而非 bad_request）
        let texts: Vec<String> = (0..128).map(|i| format!("文本{i}")).collect();
        let body = post_json(
            test_state(),
            serde_json::json!({ "texts": texts }).to_string(),
        )
        .await;
        assert_code(&body, "model_not_configured");
    }

    #[tokio::test]
    async fn empty_string_item_returns_bad_request() {
        let body = post_json(
            test_state(),
            r#"{"texts": ["正常文本", ""]}"#.into(),
        )
        .await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("texts[1]"),
            "应以下标定位空字符串: {body}"
        );
    }

    #[tokio::test]
    async fn oversize_item_returns_bad_request() {
        // 2001 个汉字：超过 2000 字符上限；错误消息不得回显原文
        let long = "汉".repeat(2001);
        let body = post_json(
            test_state(),
            serde_json::json!({ "texts": [long] }).to_string(),
        )
        .await;
        assert_code(&body, "bad_request");
        let msg = body["data"]["error"].as_str().unwrap();
        assert!(msg.contains("texts[0]") && msg.contains("2000"), "实际: {msg}");
        assert!(!msg.contains("汉"), "错误消息不得包含原文: {msg}");
    }

    #[tokio::test]
    async fn invalid_mode_returns_bad_request() {
        let body = post_json(
            test_state(),
            r#"{"texts": ["文本"], "mode": "doc"}"#.into(),
        )
        .await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn normalize_false_returns_bad_request() {
        // v1 不支持关闭归一化：显式报错优于静默忽略（契约谎言）
        let body = post_json(
            test_state(),
            r#"{"texts": ["文本"], "normalize": false}"#.into(),
        )
        .await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("normalize"),
            "应说明 normalize=false 不支持: {body}"
        );
    }

    #[tokio::test]
    async fn malformed_json_returns_bad_request() {
        // 坏 JSON 与缺字段同走 bad_request 信封（非 axum 默认 4xx 纯文本）
        let body = post_json(test_state(), r#"{"texts": ["#.into()).await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn non_string_texts_item_returns_bad_request() {
        // texts 元素必须是字符串（serde 类型校验）
        let body = post_json(test_state(), r#"{"texts": [123]}"#.into()).await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn body_over_limit_returns_too_large() {
        // JsonRejection 的 413 分支 → too_large（ARGUS_MAX_BODY_MB 超限）
        let config = Config {
            max_body_mb: 1,
            ..Config::default()
        };
        let mgr = EngineManager::new(config.clone());
        let state = AppState::new(config, mgr);
        let body = format!(r#"{{"texts": ["{}"]}}"#, "a".repeat(1_100_000));
        let resp = post_json(state, body).await;
        assert_code(&resp, "too_large");
    }

    #[tokio::test]
    async fn valid_request_unconfigured_returns_model_not_configured() {
        // 合法请求（mode=query 是合法值）→ 校验放行 → 落到引擎"模型未配置"；
        // 同时证明校验先于懒加载触发（硬纪律 4）
        let body = post_json(
            test_state(),
            r#"{"texts": ["一家人围在桌前吹蜡烛"], "mode": "query", "normalize": true}"#.into(),
        )
        .await;
        assert_code(&body, "model_not_configured");
    }

    #[tokio::test]
    async fn capability_disabled_returns_capability_disabled() {
        // 三件套已配置但白名单不含 text_embedding：capability_disabled
        let body = post_json(
            disabled_state(),
            r#"{"texts": ["文本"]}"#.into(),
        )
        .await;
        assert_code(&body, "capability_disabled");
    }

    // ---- DTO / 响应组装（json 形状） ----

    #[test]
    fn embed_request_should_treat_mode_and_normalize_as_optional() {
        let req: EmbedTextRequest =
            serde_json::from_str(r#"{"texts": ["文本"]}"#).unwrap();
        assert_eq!(req.texts, ["文本"]);
        assert!(req.mode.is_none() && req.normalize.is_none());
    }

    #[test]
    fn embed_request_should_reject_non_string_texts_item() {
        assert!(serde_json::from_str::<EmbedTextRequest>(r#"{"texts": [123]}"#).is_err());
    }

    #[test]
    fn ok_response_should_carry_full_envelope() {
        let resp = ok_response(
            3,
            json!("model.onnx#abc123"),
            json!("bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512"),
            512,
            vec![vec![0.5, 0.5], vec![1.0, 0.0]],
            7,
        );
        assert_eq!(resp["status"], "ok");
        let data = &resp["data"];
        assert_eq!(data["queue_ms"], 3);
        assert_eq!(data["model"], "model.onnx#abc123");
        assert_eq!(data["pipeline"], "bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512");
        assert_eq!(data["dim"], 512);
        assert_eq!(data["normalized"], true, "引擎构造期归一化，恒 true");
        assert_eq!(data["count"], 2);
        assert_eq!(data["embeddings"][0].as_array().unwrap().len(), 2);
        assert_eq!(data["inference_ms"], 7);
    }
}
