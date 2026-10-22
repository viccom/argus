//! 文本 Reranking 路由：`POST /rerank`（JSON，设计 §6.1）。
//!
//! cross-encoder 单查询批模式：query 固定、texts 批量，一批一前向后逐条
//! sigmoid 得 relevance 分数（输入序）。校验顺序是硬纪律 4 的**窄例外**：
//! 形状/长度/条数校验先于懒加载，token 预算校验在引擎之后——tokenizer 在
//! `TextSessionRunner` 私有字段内，只有懒加载后才存在（设计 §5/§6.1）。
//! 预算用 `B × max_pair_len` 而非 Σ token：批内按最长 padding，真实计算量
//! 是 B×L（2026-10-22 实测 31 短文本 + 1×512 → Σ 仅 884 却 9.7s）。

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AppError;
use crate::routes::common::{acquire_permit, map_json_rejection};
use crate::routes::health::AppState;

/// query 与单条 text 的字符上限（同 embed：512 token 截断上界之上留足余量，
/// 防滥用不伤正常文本；常量不放 env，设计 §6.1）。
const MAX_CHARS: usize = 2000;
/// token 计算量预算上限：`B × max_pair_len ≤ 8192`（设计 §6.1，动机见模块注释）。
const TOKEN_BUDGET: usize = 8192;

// ---- DTO ----

#[derive(Deserialize)]
pub struct RerankRequest {
    query: String,
    texts: Vec<String>,
}

/// 预算判定（纯函数供单测）：`B × max_pair_len > 8192` 即超。
/// saturating_mul 防溢出回绕（回绕会让超巨请求伪装成小值绕过预算）。
fn budget_exceeded(batch: usize, max_pair_len: usize) -> bool {
    batch.saturating_mul(max_pair_len) > TOKEN_BUDGET
}

/// ranking 构造（纯函数供单测）：按 score 降序，同分按 index 升序
/// （确定性 tie-break，设计 §6.1）。调用前已过 is_finite 后校验，无 NaN 入参。
fn ranking_of(scores: &[f32]) -> Vec<Value> {
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .map(|(index, score)| json!({ "index": index, "score": score }))
        .collect()
}

/// 成功响应组装（独立函数便于无模型单测 json 形状）。
/// 无 sigmoid 字段——pipeline 串已含 `|sigmoid|`（设计 §6.1）。
fn ok_response(
    queue_ms: u128,
    model: Value,
    pipeline: Value,
    count: usize,
    scores: Vec<f32>,
    ranking: Vec<Value>,
    inference_ms: u128,
) -> Value {
    json!({
        "status": "ok",
        "data": {
            "queue_ms": queue_ms,
            "model": model,
            "pipeline": pipeline,
            "count": count,
            "scores": scores,
            "ranking": ranking,
            "inference_ms": inference_ms,
        }
    })
}

// ---- handler ----

/// POST /rerank — 文本重排（JSON: query[1..2000 字符]/texts[1..=max_texts]
/// 单条 1..2000 字符，max_texts 默认 32 经 `ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调）。
///
/// 校验顺序（设计 §6.1，硬纪律 4 的窄例外）：形状/长度/条数先于懒加载 →
/// 引擎 → token 预算（tokenizer 在引擎内，只能在后）→ permit → 推理；
/// 计时在 permit 之后（B5 纪律）。
pub async fn rerank_text(
    State(state): State<AppState>,
    body: Result<Json<RerankRequest>, JsonRejection>,
) -> Result<Json<Value>, AppError> {
    let Json(req) = body.map_err(map_json_rejection)?;

    // ---- 懒加载前校验（无需 tokenizer，硬纪律 4） ----
    let query_chars = req.query.chars().count();
    if query_chars == 0 {
        return Err(AppError::BadRequest("query 不能为空（1..2000 字符）".into()));
    }
    if query_chars > MAX_CHARS {
        return Err(AppError::BadRequest(format!(
            "query 超过 {MAX_CHARS} 字符上限（当前 {query_chars} 字符）"
        )));
    }
    let max_texts = state.config.text_reranking_max_texts.unwrap_or(32);
    if req.texts.is_empty() || req.texts.len() > max_texts {
        return Err(AppError::BadRequest(format!(
            "texts 数量须在 1..={max_texts} 条（当前 {} 条）",
            req.texts.len()
        )));
    }
    // 逐条校验：消息用下标定位，绝不回显原文（隐私，需求 §7）
    for (i, t) in req.texts.iter().enumerate() {
        if t.is_empty() {
            return Err(AppError::BadRequest(format!("texts[{i}] 为空字符串")));
        }
        let chars = t.chars().count();
        if chars > MAX_CHARS {
            return Err(AppError::BadRequest(format!(
                "texts[{i}] 超过 {MAX_CHARS} 字符上限（当前 {chars} 字符）"
            )));
        }
    }

    let pool = state.engines.text_reranking().await?;

    // ---- token 预算（B × max_pair_len ≤ 8192；tokenize 是 CPU 密集，
    // pool.run 内部 spawn_blocking，硬纪律 1） ----
    let batch = req.texts.len();
    let (q, docs) = (req.query.clone(), req.texts.clone());
    let max_len = pool
        .run(move |e| {
            let docs: Vec<&str> = docs.iter().map(String::as_str).collect();
            e.max_pair_len(&q, &docs)
        })
        .await?;
    if budget_exceeded(batch, max_len) {
        // 消息只含 B/max_len/乘积/上限，不含 query/texts 原文（隐私）
        return Err(AppError::BadRequest(format!(
            "计算量超预算：{batch} 条 × 最长 pair {max_len} token = {}，超过上限 {TOKEN_BUDGET}（缩短 query/texts 或减少条数）",
            batch.saturating_mul(max_len)
        )));
    }

    let permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let (q, docs) = (req.query.clone(), req.texts.clone());
    let scores = pool
        .run(move |e| {
            let docs: Vec<&str> = docs.iter().map(String::as_str).collect();
            e.rerank(&q, &docs)
        })
        .await?;
    let elapsed = t0.elapsed().as_millis();

    // ---- 后校验（fail loud：引擎返回形状不符/非有限分数不外发） ----
    if scores.len() != batch {
        return Err(AppError::Inference(format!(
            "rerank 返回 {} 个分数，与输入 {batch} 条不符",
            scores.len()
        )));
    }
    if let Some(i) = scores.iter().position(|s| !s.is_finite()) {
        return Err(AppError::Inference(format!(
            "rerank 返回非有限分数（NaN/Inf，scores[{i}]）"
        )));
    }

    let ranking = ranking_of(&scores);
    // 日志不含 query/texts 原文（隐私）
    tracing::info!(endpoint = "rerank_text", count = batch, inference_ms = elapsed as u64, "推理完成");

    let (model, pipeline) = state.engines.provenance("text_reranking").await;
    Ok(Json(ok_response(
        permit.queue_ms,
        model,
        pipeline,
        batch,
        scores,
        ranking,
        elapsed,
    )))
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

    /// 两件套齐备但白名单只留 detection：能力被关（reranking 无 dim env，设计 §6.2）。
    fn disabled_state() -> AppState {
        let config = Config {
            text_reranking_model: Some("/nonexistent/model.onnx".into()),
            text_reranking_tokenizer: Some("/nonexistent/tokenizer.json".into()),
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
                Request::post("/rerank")
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

    // ---- oneshot 错误矩阵 ----

    #[tokio::test]
    async fn missing_query_field_returns_bad_request() {
        let body = post_json(test_state(), r#"{"texts": ["文本"]}"#.into()).await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn empty_query_returns_bad_request() {
        let body = post_json(test_state(), r#"{"query": "", "texts": ["文本"]}"#.into()).await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("query"),
            "错误消息应指向 query: {body}"
        );
    }

    #[tokio::test]
    async fn oversize_query_returns_bad_request() {
        // 2001 个汉字超过 2000 字符上限；错误消息不得回显原文
        let long = "汉".repeat(2001);
        let body = post_json(
            test_state(),
            serde_json::json!({ "query": long, "texts": ["文本"] }).to_string(),
        )
        .await;
        assert_code(&body, "bad_request");
        let msg = body["data"]["error"].as_str().unwrap();
        assert!(msg.contains("query") && msg.contains("2000"), "实际: {msg}");
        assert!(!msg.contains("汉"), "错误消息不得包含原文: {msg}");
    }

    #[tokio::test]
    async fn empty_texts_returns_bad_request() {
        let body = post_json(test_state(), r#"{"query": "q", "texts": []}"#.into()).await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains("texts"),
            "错误消息应指向 texts: {body}"
        );
    }

    #[tokio::test]
    async fn empty_string_item_returns_bad_request() {
        let body = post_json(
            test_state(),
            r#"{"query": "q", "texts": ["正常文本", ""]}"#.into(),
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
            serde_json::json!({ "query": "q", "texts": [long] }).to_string(),
        )
        .await;
        assert_code(&body, "bad_request");
        let msg = body["data"]["error"].as_str().unwrap();
        assert!(msg.contains("texts[0]") && msg.contains("2000"), "实际: {msg}");
        assert!(!msg.contains("汉"), "错误消息不得包含原文: {msg}");
    }

    #[tokio::test]
    async fn non_string_texts_item_returns_bad_request() {
        // texts 元素必须是字符串（serde 类型校验）
        let body = post_json(test_state(), r#"{"query": "q", "texts": [123]}"#.into()).await;
        assert_code(&body, "bad_request");
    }

    #[tokio::test]
    async fn unknown_field_should_be_ignored_not_rejected() {
        // serde 默认忽略未知字段：应正常解析并过校验，落到"模型未配置"
        // （证明未知字段不炸，与显式参数校验的拒绝语义区分）
        let body = post_json(
            test_state(),
            r#"{"query": "q", "texts": ["文本"], "foo": 1}"#.into(),
        )
        .await;
        assert_code(&body, "model_not_configured");
    }

    #[tokio::test]
    async fn malformed_json_returns_bad_request() {
        // 坏 JSON 走 bad_request 信封（非 axum 默认 4xx 纯文本）
        let body = post_json(test_state(), r#"{"query": "q""#.into()).await;
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
        let body = format!(
            r#"{{"query": "q", "texts": ["{}"]}}"#,
            "a".repeat(1_100_000)
        );
        let resp = post_json(state, body).await;
        assert_code(&resp, "too_large");
    }

    #[tokio::test]
    async fn valid_request_unconfigured_returns_model_not_configured() {
        // 合法请求 → 校验放行 → 落到引擎"模型未配置"；
        // 同时证明校验先于懒加载触发（硬纪律 4）
        let body = post_json(
            test_state(),
            r#"{"query": "一家人围在桌前吹蜡烛", "texts": ["全家福照片描述"]}"#.into(),
        )
        .await;
        assert_code(&body, "model_not_configured");
    }

    #[tokio::test]
    async fn capability_disabled_returns_capability_disabled() {
        // 两件套已配置但白名单不含 text_reranking：capability_disabled
        let body = post_json(
            disabled_state(),
            r#"{"query": "q", "texts": ["文本"]}"#.into(),
        )
        .await;
        assert_code(&body, "capability_disabled");
    }

    #[tokio::test]
    async fn max_texts_boundary_should_enforce_env_cap() {
        // text_reranking_max_texts=2 时 3 条即拒（ARGUS_TEXT_RERANKING_MAX_TEXTS 可调）
        let config = Config {
            text_reranking_max_texts: Some(2),
            ..Config::default()
        };
        let mgr = EngineManager::new(config.clone());
        let state = AppState::new(config, mgr);
        let body = post_json(
            state,
            r#"{"query": "q", "texts": ["a", "b", "c"]}"#.into(),
        )
        .await;
        assert_code(&body, "bad_request");
        assert!(
            body["data"]["error"].as_str().unwrap().contains('2'),
            "应提及上限 2 条: {body}"
        );
    }

    #[tokio::test]
    async fn texts_at_default_limit_should_pass_count_validation() {
        // 默认 32 条是合法上界：计数校验放行后落到"模型未配置"（而非 bad_request）
        let texts: Vec<String> = (0..32).map(|i| format!("文本{i}")).collect();
        let body = post_json(
            test_state(),
            serde_json::json!({ "query": "q", "texts": texts }).to_string(),
        )
        .await;
        assert_code(&body, "model_not_configured");
    }

    // ---- 预算判定（纯函数） ----

    /// 16×512=8192 恰在预算内（全 512 token pair 时的有效条数上限，设计 §6.1）。
    #[test]
    fn budget_at_exact_limit_should_not_exceed() {
        assert!(!budget_exceeded(16, 512));
    }

    /// 17×512 超预算：条数上限随文本长度收缩。
    #[test]
    fn budget_over_limit_should_exceed() {
        assert!(budget_exceeded(17, 512));
    }

    /// 32 条短文本（~12 token）远在预算内：默认 32 条对短文本不设卡。
    #[test]
    fn short_texts_should_not_exceed() {
        assert!(!budget_exceeded(32, 12));
    }

    /// saturating_mul 边界：巨值不因乘法回绕伪装成小值（溢出 saturate 到 MAX 仍判超）。
    #[test]
    fn budget_should_saturate_instead_of_wrapping() {
        assert!(budget_exceeded(usize::MAX, 2));
        assert!(budget_exceeded(2, usize::MAX));
        assert!(!budget_exceeded(0, usize::MAX), "零批不占预算");
    }

    // ---- DTO / 响应组装（json 形状） ----

    #[test]
    fn rerank_request_should_require_query_and_texts() {
        let req: RerankRequest =
            serde_json::from_str(r#"{"query": "查询", "texts": ["文本"]}"#).unwrap();
        assert_eq!(req.query, "查询");
        assert_eq!(req.texts, ["文本"]);
        // 缺任一必填字段必须拒绝
        assert!(serde_json::from_str::<RerankRequest>(r#"{"query": "q"}"#).is_err());
        assert!(serde_json::from_str::<RerankRequest>(r#"{"texts": ["t"]}"#).is_err());
    }

    #[test]
    fn ranking_should_sort_desc_with_index_tiebreak() {
        // 同分 0.5：index 0 先于 2（确定性 tie-break，设计 §6.1）；
        // 分数经 f32 存储，断言用 f32 展宽后的值精确比较
        let ranking = ranking_of(&[0.5, 0.9, 0.5]);
        assert_eq!(ranking[0]["index"], 1);
        assert_eq!(ranking[0]["score"].as_f64().unwrap(), 0.9f32 as f64);
        assert_eq!(ranking[1]["index"], 0);
        assert_eq!(ranking[2]["index"], 2);
        assert_eq!(ranking[1]["score"].as_f64().unwrap(), 0.5f32 as f64);
    }

    #[test]
    fn ok_response_should_carry_full_envelope() {
        let resp = ok_response(
            3,
            json!("model.onnx#abc123"),
            json!("jina-reranker-v2-base-multilingual|sigmoid|maxlen512"),
            2,
            vec![0.91, 0.02],
            ranking_of(&[0.91, 0.02]),
            37,
        );
        assert_eq!(resp["status"], "ok");
        let data = &resp["data"];
        assert_eq!(data["queue_ms"], 3);
        assert_eq!(data["model"], "model.onnx#abc123");
        assert_eq!(data["pipeline"], "jina-reranker-v2-base-multilingual|sigmoid|maxlen512");
        assert_eq!(data["count"], 2);
        assert_eq!(data["scores"][0].as_f64().unwrap(), 0.91f32 as f64);
        assert_eq!(data["scores"][1].as_f64().unwrap(), 0.02f32 as f64);
        assert_eq!(data["ranking"][0]["index"], 0);
        assert_eq!(data["ranking"][0]["score"].as_f64().unwrap(), 0.91f32 as f64);
        assert_eq!(data["ranking"][1]["index"], 1);
        assert_eq!(data["inference_ms"], 37);
        assert!(data.get("sigmoid").is_none(), "不单列 sigmoid 字段（pipeline 串已含 |sigmoid|，设计 §6.1）");
    }
}
