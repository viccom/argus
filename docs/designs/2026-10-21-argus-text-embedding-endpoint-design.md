# argus 文本 Embedding 端点设计（批 D1：`POST /embed/text`）

> 状态：implemented（2026-10-21；计划 D0-D6 全销账，测试 258→282，E2E 6 项过；生产 bundle 已部署 argus-0.1.0-c9ef409，2026-10-22 验收全过）
> 日期：2026-10-21
> 基线：argus @ `1b9adaa`（动态槽已合并 + 模型清单制度）；rust-onnx-infer @ `4b4615e`（`fdd7beb` 引擎 + 审查修正，已 push）
> 权威源：需求 homepix 批 D1（`text-embedding-requirements.md`，homepix 侧 2026-10-21 成文）；引擎侧设计 [2026-10-21-rust-onnx-infer-text-embedding-design.md](2026-10-21-rust-onnx-infer-text-embedding-design.md)（implemented）；实施进度 [plans/2026-10-21-argus-text-embedding-endpoint-plan.md](../plans/2026-10-21-argus-text-embedding-endpoint-plan.md)；端点/env/模型表 **README.md**
> 取代关系：无

## 1. 背景与目标

### 1.1 背景

fork 侧 `TextEmbeddingEngine`（`4b4615e`）已就绪：BERT 分词 → int64 三输入直跑 → CLS 池化 → L2，实测 p95 最差 9.00ms（ur5900，2026-10-21）、与 python 参照位级一致、Arc 并发逐位一致。本设计把它接入 argus 成为首条文本能力。

### 1.2 目标

- `POST /embed/text`：批量中文文本 → 归一化向量，满足 homepix 批 D1 验收清单 6 项；
- 第 16 个能力键 `text_embedding`，/models、白名单、per-capability 池自然覆盖；
- 零回归：现有 258 测试全绿、clippy 零警告、golden 同步更新。

### 1.3 非目标

- 不做 reranking / 跨模态 / ANN（需求 §8 明确排除）；
- 不做 /ui 卡片（控制台是图像拖放流，**显式豁免三道门第 3 关的 /ui 项**，E2E 以 curl+余弦验证替代——写入计划）；
- 不做 query 指令前缀（bge v1.5 官方短查询无需指令；`mode` 字段接受且等价实现，需求 §2 允许）；
- 不引入向量存储（argus 只算不存，需求 §6）。

## 2. 代码现状（基线 `1b9adaa`；fork 侧事实 @ `4b4615e`）

| 事实 | 证据 |
|---|---|
| 能力键 15 个，无文本能力 | `src/config.rs:8` `CAPABILITY_KEYS: [&str; 15]` |
| 接入形态判定：非检测族+新契约 → 三道门代码接入 | AGENTS.md「检测族新模型优先走动态槽……仅非检测族或需要新契约形态的能力才走三道门」（2026-10-21 新增） |
| fork 引擎 API | `TextEmbeddingEngine::new(model, tokenizer, device)`（Cls+normalize+512）；`embed(&str)`/`embed_batch(&[&str])`/`dim()`/`max_length()`；**未实现 `OnnxInferenceEngine` trait**（Send+Sync 有 const 断言，fork `4b4615e`） |
| fork 空字符串契约 | `embed_batch` 对空字符串返回 `InvalidArgument` → argus `From<VisionError>` 映射 `inference_failed`——**HTTP 层必须前置校验返回 `bad_request`**（fork 审查移交事项） |
| 吞吐策略依据 | T8 实测（2026-10-21）：本机 64 条 batch 402ms **慢于**逐条循环 323ms（批内 pad 白付计算）；逐条循环兼得严格跨批逐位确定（引擎设计 §4.4） |
| EnginePool 泛型约束 | `E: Send + Sync + 'static`（`src/engine.rs` EnginePool），不要求推理 trait |
| 预热 helper 不适用 | `warmup_engine` 依赖 `OnnxInferPredict`（读 input_size 造图像 dummy），文本引擎无该 trait；`pool.warmup` 本身接受任意闭包（engine.rs:778） |
| 深检宏不适用 | `check_engine!` 发图像 dummy 走 `predict`——文本引擎无此方法，需独立文本分支 |
| JSON 端点信封先例 | `/detect/batch`：`Result<Json<T>, JsonRejection>` → `BadRequest`（`src/routes/batch.rs:56-59`）；413 分支映射 `TooLarge` 尚无先例（本设计新增） |
| 多文件能力的 configured 判定先例 | ocr 三件套（det+rec+dict 全配才算 configured，dict 不进 models 清单）——tokenizer.json 同理（`src/config.rs:276`、`src/routes/models.rs:131`） |
| golden 基线锁定 | `tests/golden/models.json` 锁 15 能力全量输出；复比 mask `loaded`/`engines`/`engine_health`（AGENTS.md golden 约定，2026-10-21） |
| 断言 15 的测试 | `config.rs`（`CAPABILITY_KEYS: [&str; 15]` 类型本身）、`models.rs::empty_config_should_list_all_capabilities_unconfigured` 断言 `caps.len()==15`、golden |
| 模型清单制度 | `docs/model-inventory.md`：新模型过任一验证环节即回填（2026-10-21 立） |

## 3. 决策与备选

### 3.1 路由层逐条 `embed()` 循环（不用 `embed_batch`）

依据：T8 实测本机逐条循环更快（323 vs 402ms/64 条）+ 严格跨批逐位确定性（任意批组成下同输入同向量，homepix `source_hash` 幂等最稳形态）。128 条上限下最坏 ~128×9ms≈1.2s（61 tok 长文），批量预算内。被否：`embed_batch`（本机更慢且引入跨批 ulp 口径负担）。

### 3.2 tokenizer 路径：显式 env（`ARGUS_TEXT_EMBEDDING_TOKENIZER`）

`ARGUS_TEXT_EMBEDDING_MODEL` 指向 onnx、`ARGUS_TEXT_EMBEDDING_TOKENIZER` 指向 tokenizer.json、`ARGUS_TEXT_EMBEDDING_DIM` 声明维度——三件套齐备才算 configured（缺任一 → model_not_configured）。依据：仓库既有非模型附件先例 `ARGUS_OCR_DICT`（显式 env，无路径推导）；推导默认值会引入新配置模式且部署 env 本就逐项显式（bundle 布局 `./models` 相对路径约定不变）。~~初稿曾设计"同目录推导 + env 覆盖"，实施时按计划走显式三件套，本节已对齐实现（2026-10-21 纠偏，见变更记录）~~。

### 3.3 `dim` 三态合一：env 声明 + /models 静态回报 + 加载期校验

`ARGUS_TEXT_EMBEDDING_DIM` 必填（与 model+tokenizer 构成 configured 三件套）→ /models 配置态静态可得（需求 §9 期望，/models 刻意不触碰 session）；引擎加载期 `engine.dim() != env` 即拒载（fail loud，防模型文件被悄悄替换成不同维度）。被否：从 ONNX 文件解析输出 shape（过度工程）/ 加载后才可知（/models 拿不到）。

### 3.4 `mode` 接受且等价

`passage`（默认）/`query` 都按原文编码（bge v1.5 官方：短查询无需检索指令前缀）。字段校验非法值 → `bad_request`。pipeline 串不体现 mode（等价实现，无语义分支）。

### 3.5 单条字符上限常量 `MAX_TEXT_CHARS = 2000`

需求 §2 要求"单条超上限 → bad_request"但未给数。512 token 截断上界之上留足余量（≈500 汉字已截断），2000 只防滥用不伤正常文档（homepix 文档列最长约几百字符，需求 §3）。常量不放 env（YAGNI）。

## 4. 架构与数据流

```
POST /embed/text (JSON: {texts: [1..128], mode?, normalize?})
  → serde 解析失败 → bad_request；body 超 ARGUS_MAX_BODY_MB → too_large（JsonRejection 413 分支）
  → 逐条校验：空字符串/超 2000 字符/mode 非法 → bad_request（消息用下标表述，不含原文）
  → engines.text_embedding()（白名单→路径→spawn_blocking 构造→池→warmup 短文本）
  → acquire_permit（全局背压，overload 语义不变）
  → 逐条 pool.run(|e| e.embed(text))（每条独立池准入；计时在 permit 之后，B5 纪律）
  → 可选跳过归一化（normalize=false 时引擎需支持？——见 §5 契约）
  → data{model, pipeline, dim, normalized, count, embeddings, queue_ms, inference_ms}
```

预热：`pool.warmup(|e| e.embed("warmup").map(|_| ()))`（短文本，不走 warmup_engine helper）。
深检：`deep_health_check` 加独立分支——已加载时 `embed("ping")` 成功即 ok（不进 `check_engine!` 宏）。

### 4.1 normalize=false 的实现

fork 引擎 normalize 是**构造参数**（`with_options`），不是每次调用参数。argus 池内引擎按 `normalize=true` 构造（需求默认）；`normalize=false` 请求的处理：**v1 不支持——传入 `normalize:false` 返回 `bad_request`**（明确报错优于静默忽略）。依据：homepix 明确只用 true（点积退化余弦，需求 §2）；fork 支持每请求归一化切换需双引擎或改 fork，为无消费者的场景付复杂度不值。README 契约注明。

## 5. 契约与兼容性

**请求**（新端点，纯新增）：
```json
{"texts": ["一家人围在桌前吹蜡烛"], "mode": "passage", "normalize": true}
```
`texts` 必填 1..128；`mode` ∈ {passage, query}（缺省 passage）；`normalize` 仅接受 true/缺省。

**响应**（只增字段）：
```json
{"status":"ok","data":{"model":"model.onnx#<sha12>","pipeline":"bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512",
 "dim":512,"normalized":true,"count":1,"embeddings":[[...]],"queue_ms":0,"inference_ms":3}}
```
`latency_ms` 不设——沿用 B5「queue_ms 与 inference_ms 分开」契约（2026-10-20 审查修复项）。

**错误码**（全部既有，零新增）：bad_request（空数组/超 128/非字符串（serde）/空字符串/超 2000 字符/mode 非法/normalize=false/坏 JSON）、too_large（413 分支）、model_not_configured、capability_disabled、inference_failed、overload（唯一 retryable）。

**pipeline 串**：`PIPELINE_TABLE` 加 `("text_embedding", "bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512")`。对齐需求示例格式（模型族|池化-归一化|截断）；换模型/换池化/换 maxlen 必改此串（既有纪律，撞串测试自动覆盖）。

**/models**：16 项；`CapabilityModels` 增 `dim: Option<usize>`（`skip_serializing_if` None，仅 text_embedding 项出现——"只加不改"，其他 15 项序列化输出逐字节不变）；configured 三件套判定（model+tokenizer+dim）。golden `models.json` 同步。

**config**：`CAPABILITY_KEYS` 15→16（`text_embedding`）；新字段 `text_embedding_model/text_embedding_tokenizer/text_embedding_dim`；`model_paths_for`/`configured_engines` 各加分支；`ARGUS_TEXT_EMBEDDING_POOL` 经既有循环自动生效。断言 15 的测试与 golden 随之改 16。

**/health deep**：`engine_health` 数组在文本引擎已加载时多一项 `text_embedding`（ok/degraded）；浅检 `engines` 列表同理。golden `health-deep.json` 若在生产实例重抓则同步（复比 mask 约定下，未加载路径不出现该项——**golden 用未配置实例抓取则零变化**，见计划 D4）。

**鉴权/日志**：router 层中间件自然覆盖；访问日志只记 count/dim/耗时（TraceLayer 只见 URI/状态）；handler 日志与错误消息不含任何输入文本（隐私，需求 §7）。

## 6. 错误处理与安全边界

- 校验顺序遵循硬纪律 4：parse → 逐条校验 → engine（懒加载触发点在校验之后）→ permit → run；
- `From<VisionError>` 兜底存在但空字符串/长度校验**必须在 handler 前置**（fork 兜底映射的是 inference_failed，语义码不对——fork 审查移交事项，测试锁定）；
- 错误消息模式：`"texts[3] 为空字符串"`（下标定位，无原文回显）。

## 7. 风险登记

| 风险 | 级别 | 缓解 | 验证 |
|---|---|---|---|
| golden/断言 15 的测试漏改 → 假失败 | 低 | 计划 D4 集中清单（models.rs 断言、config 数组、golden 重抓） | cargo test 全绿 |
| normalize=false 静默忽略（契约谎言） | 中 | 显式 bad_request + 测试锁定 | oneshot 测试 |
| 空字符串漏前置校验 → inference_failed 漏码 | 中 | handler 校验 + 测试锁定 | oneshot 测试 |
| 128 条×长文本请求耗时压垮池 | 低 | 上限 128×2000 字符已有界；逐条独立池准入可被并发打断 | E2E |
| 深检误入 check_engine! 宏（编译期即暴露） | 低 | 宏要求 predict 方法，编译失败即发现 | 编译 |

## 8. 测试策略

- config：16 键/三件套 env 解析/dim 必填判定/池覆盖（照既有 config 测试风格，env 锁串行）；
- engine：loader 未配置→model_not_configured、白名单外→capability_disabled（照既有 loader 测试）；PIPELINE_TABLE 覆盖测试自动扩展（16 串互异）；
- 路由 oneshot 组（照 classification.rs 范式）：缺 texts/空数组/超 128/空字符串/超 2000 字符/mode 非法/normalize=false/坏 JSON/未配置模型/白名单关——全部断言 `data.code`；DTO 序列化测试（字段齐全、f32 序列化）；
- golden：未配置实例重抓 `models.json`（16 项，text_embedding 项 configured=false）——复比 mask 约定下零运行态字段；
- E2E（真模型，:7099 临时实例）：验收清单 6 项（能力位/批量+模长/确定性逐位/语义余弦/错误形态/鉴权）。

## 9. 实施阶段与门禁

见 [plans/2026-10-21-argus-text-embedding-endpoint-plan.md](../plans/2026-10-21-argus-text-embedding-endpoint-plan.md)。门禁：cargo test 全绿 + clippy 零警告 + E2E 6 项 + golden 同步，缺一不可销账。

## 10. 验收标准

```bash
cargo test --jobs 6 && cargo clippy --all-targets --jobs 6        # 双绿，基线 258 → 新基线全绿
# E2E（模型就位，ARGUS_TEXT_EMBEDDING_MODEL=... DIM=512）：
curl -s localhost:7099/models | jq '.data.capabilities.text_embedding'   # configured/enabled/pipeline/dim 四项
curl -s localhost:7099/embed/text -H 'content-type: application/json' -d '{"texts":["在海边玩","沙滩上堆城堡"]}' \
  | jq '.data | {dim,count,normalized,pipeline,first_norm:(.embeddings[0]|map(.*.)|add)}'
# 确定性：同输入两次逐位相等；语义：cos(海边玩,堆城堡) > cos(海边玩,财务报表)
# 错误形态：空数组→bad_request；未配→model_not_configured；并发压满→overload+retryable；无 key→拒
```

## 11. 回滚

单批次提交（代码+golden+文档），revert 即回滚；fork 依赖 `4b4615e` 已在远端，回滚 argus 提交后 Cargo.lock 仍指向 `4b4615e`（引擎代码不激活即零影响——未配置能力不加载）。

## 12. 未决项

- 部署 bundle 何时带 bge 模型（E2E 通过后由部署流程纳入 `models/bge-small-zh-v1.5/` 两文件——运维动作，不入库）；
- 批 D 段回填 homepix-integration-plan.md（D1 完成时一并）；
- model-inventory.md 回填 bge 行（E2E 后按制度登记为现役）。

## 13. 变更记录

- 2026-10-21：建立（draft）。汇总前序全部论证（需求评估/gate-1/T8/fork 审查两轮）为 argus 侧接入设计。
- 2026-10-21（实施完成，draft→implemented）：D0-D6 全销账（证据见计划表）。实施偏差两处：①§3.2 由"同目录推导+env 覆盖"改为**显式 `ARGUS_TEXT_EMBEDDING_TOKENIZER`**（对齐实现与 `ARGUS_OCR_DICT` 既有先例，本节正文已改）；②路由注册提前至 D3（D3 oneshot 组转绿的必要条件）。E2E 实测（:7099）：语义余弦与 gate-1 参照一致（0.4899/0.1927）、同输入逐位相等、HTTP 端到端 p95 3.6ms（预算 30ms 的 8 倍余量）。
