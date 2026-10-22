# 设计：文本 reranking 能力 + 同家族模型对（embedding × reranker）env 级组合

> 状态：approved（2026-10-22 用户批准：texts 上限默认 32、`ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调至 64；预算校验位于懒加载后已接受）
> 日期：2026-10-22
> 基线：argus @ `45e0bc9`（测试 282 全绿 + clippy 0）；rust-onnx-infer @ `49b8585`
> 权威源：模型事实 = [model-inventory.md](../model-inventory.md) #19/#22/#23；端点与 env 落地后回写 README
> 取代关系：无

## 1. 背景与目标

产品将开源，终端用户语言场景不同，需按场景选择 **同家族模型对**（一个 argus 实例内 embedding + reranker 稳定共存）：

- **jina 对**（双语）：jina-embeddings-v2-base-zh（mean 池化 768 维）+ jina-reranker-v2-base-multilingual **int8**
- **bge 对**（中文）：bge-small-zh-v1.5（CLS 池化 512 维）+ bge-reranker-base

要求：低风险、低成本；后期新模型对零/低成本接入（仅 env + 模型文件，不改代码）。

实测依据（2026-10-22，model-inventory #22/#23）：两族 reranker 同接口形态（XLM-R seq-cls 单 logit、两输入、`<s>q</s></s>d</s>`、sigmoid）；bge 重排在中文家庭场景更稳，jina 重排双语+近重复纠偏强；int8 与 fp32 精度无差；重排净占内存 int8 621MB / fp32 1587MB。

## 2. 非目标

- 不做 `PooledOutput` 池化的 env 暴露（无目标模型需要，YAGNI）
- **不做单实例同时运行两对模型**：env 单值语义（一个 `ARGUS_TEXT_EMBEDDING_MODEL`）结构上就只支持一对；切换 = 改 env + 重启。若将来真要并存，需另设机制（多能力键或命名槽），不在本期。注：两段本身无耦合，**跨族混搭**（如 jina embedding + bge reranker）在 env 上天然成立（model-inventory #23 实测同为满分），"同家族"是文档推荐约定而非强制
- 不做 reranker 分数校准/阈值过滤（消费方自行取 top-N）
- 不改既有 15 个图像能力的行为；`text_embedding` 的**默认 env 行为**不变（新 env 不设 = 现状）
- **不做控制台 UI 卡片**：`/ui` 是图像工作台（`web/index.html` 的 TASKS 全为图像任务，text_embedding 当初亦未加卡片）。文本能力以 curl/脚本 E2E 为准；本设计显式记录该豁免，并回改 AGENTS.md 硬纪律措辞（T8）

## 3. 代码现状（锚点，基线 45e0bc9 / fork 49b8585）

| 事实 | 位置（基线） |
|---|---|
| `CAPABILITY_KEYS: [&str; 16]` | src/config.rs:8 |
| 路由注册点（`.route("/embed/text", ...)`） | src/routes/health.rs:164 |
| embedding 加载器硬编码 MaskedMean + maxlen 512 | src/engine.rs loader（`with_options`） |
| `PIPELINE_TABLE` 硬编码单串（jina）；`pipeline_for()` 返回 `&'static str` | src/engine.rs:55-81 |
| `ModelInfo.pipeline: &'static str`（**类型装不下运行时 env 串**） | src/engine.rs:87 |
| `model_paths_for` / `configured_engines` 无 text_reranking 分支 | src/config.rs:302-331 / 335-361 |
| `list_loaded` 无 reranking 分支 | src/engine.rs:310-329 |
| `deep_health_check` 的 text_embedding 独立 if-let 分支（可照抄） | src/engine.rs:354-367 |
| `EnginePool::warmup` 要求 `Fn(&E) -> Result<()> + Send + Sync`（**须 `.map(|_| ())`**） | src/engine.rs:970-972 |
| fork：`runner::new` **强制要求 `last_hidden_state` 输出并解析末维作 dim**（reranker 只有 `logits`，直接复用必失败） | fork runner.rs:154-229 |
| fork：`PAD_ID = 0`（BERT 约定），但两个 reranker tokenizer 实测 `<pad>=1`、`<s>=0` | fork runner.rs:29 + 本机 tokenizer dump |
| fork：`encode_texts` 是 `pub(crate)` 自由函数，需 `&Tokenizer`（runner 内私有） | fork runner.rs:49-53 |
| fork：`PoolingKind` 是 `pub`，argus 已在用全路径构造 | fork embedding.rs:32-45 |
| 两族 reranker 输出 `logits [B,1]`（gate-1 实测） | 模型目录 gate1 脚本 |

## 4. 决策与备选

| 决策 | 备选 | 为什么 |
|---|---|---|
| 池化/maxlen 暴露为 env（`POOLING`/`MAXLEN`，默认 = 现状） | 保持硬编码 | bge 对需要 CLS；默认值保证零回归 |
| `PIPELINE` 串 env 覆盖，且 `ModelInfo.pipeline` 从 `&'static str` 改 `String` | 保持 `&'static str` | 运行时 env 串装不进 `&'static str`；不改类型 T5 无法实现（审查 B2） |
| override 由 `Config` 解析，经 `EngineManager::pipeline_for(&self, cap) -> String` 统一出口 | 让自由函数 `pipeline_for` 直接读 env | 自由函数无 Config 且在测试中与 env 锁冲突（flaky）；单出口保证 `/models` 与推理响应同源 |
| reranker 输出恒 sigmoid，响应不再单列 `sigmoid` 字段 | 加开关 / 加 `sigmoid:true` 字段 | 两族官方约定一致，YAGNI；pipeline 串已含 `\|sigmoid\|`，冗余字段无消费方 |
| **texts 上限默认 32，`ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调（上限 64）** | 固定 32 / 固定 64 | **实测（2026-10-22，int8，intra_op=4）**：64×512 → **17.7s/+4.7GB**；32×512 → 8.8s；fp32 64×512 → 28.3s/+5.5GB。**`Σ pair_len` 是错误指标**：31 条短文本 + 1 条 512 token（Σ 仅 884，远低于 8192）实测 **9.7s**——padding 到 L=512、B=32，真实计算量 `B×L=16384`。故预算必须用 `B × max_pair_len`（`max_pair_len` 含 query 与特殊 token）。默认 32+8192 最坏约 4.9s（16×512）；用户显式调到 64 时自担更长的单请求占用（预算仍生效：64 条短文本 B×L=768 依旧瞬时）。**用户拍板（2026-10-22）：默认 32、env 可调、上限封顶 64** |
| **token 预算校验在懒加载之后**（硬纪律 4 的窄例外；**用户已接受**） | 加载前校验 | 计数需要 tokenizer，而 tokenizer 在 `TextSessionRunner` 私有字段内、只有懒加载后才存在（fork 无任何 tokenizer 访问器，2026-10-22 核实）。**顺序**：形状/长度/条数校验（无需 tokenizer，仍先于懒加载）→ 引擎 → **预算校验** → permit → 推理。超预算请求会触发一次模型加载，代价可接受（加载一次性） |
| `ARGUS_TEXT_RERANKING_MAXLEN` 独立 env（默认 512） | 复用 embedding 的 MAXLEN | 两族 reranker 支持更长上下文（jina 1026 / bge 514），且与 embedding 取值本就该独立（审查 O6） |
| pad id 从 tokenizer 解析（不硬编码 0） | 复用 `PAD_ID = 0` | reranker 实测 `<pad>=1`；int8 下 padding 长度/pad 值会造成 logits 0.08-0.33 抖动（审查 B5） |
| fork runner 增 `new_for_output(...)` 通用构造器 | 复用 `new`（硬编码 last_hidden_state） | reranker 无 `last_hidden_state`，复用 `new` 构造即失败（审查 B1） |
| rerank 单查询批模式（query 固定 + texts 批量） | 逐对循环 | cross-encoder 一批一前向是效率所在；确定性契约见 §7 |
| `logits` 末维构造期校验 =1 + 运行期 `[B,1]`/长度校验（fail loud） | 运行期才发现 | 照 embedding dim 校验先例，防模型换错文件 |

## 5. 架构与数据流

```
POST /rerank {"query","texts"}
  → ① 懒加载前校验：JSON 形状 / query 长度 / texts 条数 1..32 / 单条 1..2000 字符 / 空串下标
  → ② EngineManager::text_reranking()（OnceCell 懒加载，ensure_enabled 白名单门）
  → ③ 预算校验：B × max_pair_len ≤ 8192（需 tokenizer，故在引擎之后；见 §6.1）
  → permit（B5 计时纪律）
  → pool.run(|e| e.rerank(query, texts))   // 一批一前向 → sigmoid → Vec<f32>（输入序）
  → 响应：scores（输入序）+ ranking（按分降序的 {index,score}）
```

fork 侧（纯增量，不动既有引擎行为）：

```
runner.rs:  + 构造期解析 pad id 存字段 pad_id（tokenizer 取 <pad>/[PAD]，回退 0）
              —— 作用域仅 encode_pairs；encode_texts 保持 PAD_ID=0 不变
              （jina embedding tokenizer 也是 <pad>=1，若一并改会改变既有输入张量，
               与"纯增量"冲突；MaskedMean 屏蔽 pad 位故输出无差，但按纪律不动）
            + new_for_output(model, tokenizer, device, max_length, ort_cfg,
                             output_name: &'static str, expect_last_dim: Option<usize>)
              —— 与既有 new 共用初始化；dim 字段保持 usize（embedding 路径语义不变），
                 expect_last_dim 仅作构造期额外校验（reranker 传 Some(1)，embedding 传 None）
              —— new(...) = new_for_output(OUT_LAST_HIDDEN_STATE, None) 的委托
            + encode_pairs 自由函数（tokenizer.encode(query, doc)，批内最长 padding，用 pad_id）
            + OUT_LOGITS 常量
            run(batch, output_name) 复用不变
reranking.rs（新文件）: TextRerankingEngine{runner}::with_options(...)
            构造期校验 logits 声明末维 == 1；sigmoid 纯函数；Send/Sync const 断言
            运行期校验 shape == [B,1] 且 data.len()==B（不符 → inference 错）
            pub fn max_pair_len(&self, query, texts) -> Result<usize>  // 供 HTTP 层预算校验
```

**注意 1**：`TextSessionRunner::new` 硬编码 `last_hidden_state`（fork runner.rs:152-214），reranker 必须走 `new_for_output("logits", Some(1))`。

**注意 2**：`dim` **保持 `usize`**（审查 C8）——`new_for_output` 在 `expect_last_dim: None` 时仍解析静态末维（embedding 路径语义不变）；reranker 传 `Some(1)` 额外校验。这样 `TextEmbeddingEngine::dim() -> usize`、argus loader 链路、`TextSessionRunner::dim()` 全部零改动，也不引入 `expect` panic 路径。

**注意 3**：`max_pair_len` 的 tokenization 是 CPU 密集操作，实现须在 `spawn_blocking` 内（硬纪律 1）；且与随后 `rerank` 内部编码重复一次（可接受：短文本微秒级，换取"校验与推理解耦"；如后续成为瓶颈，可在引擎内缓存编码结果）。

## 6. 契约与兼容性

### 6.1 新端点 `POST /rerank`（JSON）

请求：`{"query":"...","texts":[".."]}`；query 1..2000 字符；texts 1..**32** 条（`ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调，取值 1..=64，非法值启动失败走 `Config::validate`）、单条 1..2000 字符、空字符串拒绝。

**计算量预算（替代单纯条数上限）**：`B × max_pair_len ≤ 8192`，其中 `B = texts.len()`，`max_pair_len` = 所有 (query, doc) pair 经 tokenizer 编码（含特殊 token、应用 maxlen 截断）后的**最长**长度。超出 → `bad_request`，消息给出 `B`、`max_pair_len`、当前 `B×max_pair_len` 与上限。

> **为什么不是 Σ token**：批内按最长 padding，真实计算量是 `B × L` 而非长度之和。实测 31 条短文本 + 1 条 512 token（Σ 仅 884）→ `B×L = 32×512 = 16384` → **9.7 秒**，Σ 预算完全拦不住（2026-10-22 实测）。
>
> **有效条数上限随文本长度变化**：全 512 token 时最多 16 条；全 256 token 时最多 32 条；短文本（~12 token）时 32 条仅需 168ms。

**校验顺序（硬纪律 4 的窄例外，必须按此实现）**：
1. **懒加载前**（无需 tokenizer）：JSON 形状、query 长度、texts 条数 1..32、单条 1..2000 字符、空串下标、非字符串
2. **懒加载后**：引擎 → `max_pair_len` 预算校验（`spawn_blocking` 内，硬纪律 1）
3. permit → 推理

> 超预算请求会触发一次模型加载——这是为保持"预算准确"付出的代价，加载是一次性的（懒加载后常驻）。**错误优先级**：形状/长度类错误发生在引擎之前，必为 `bad_request`；`model_not_configured` / `capability_disabled` 先于**预算**错误（预算校验在引擎之后）。

成功响应（信封同全家）：

```json
{"status":"ok","data":{"queue_ms":0,"model":"model.onnx#<sha12>","pipeline":"jina-reranker-v2-base-multilingual|sigmoid|maxlen512",
 "count":2,"scores":[0.91,0.02],"ranking":[{"index":0,"score":0.91},{"index":1,"score":0.02}],"inference_ms":37}}
```

- `scores`：**输入序**，长度恒等于 `texts.len()`（handler 校验引擎返回长度，不符 → inference_failed）
- `ranking`：按 score 降序；**同分按 index 升序**（确定性 tie-break）；非有限分数（NaN/Inf）→ inference_failed
- 不单列 `sigmoid` 字段——pipeline 串已含 `|sigmoid|`

错误码零新增：bad_request / too_large / model_not_configured / capability_disabled / inference_failed / internal / overload（overload 唯一 retryable）；鉴权失败特例与全端点一致（200 信封无 code）。

### 6.2 新 env（全部缺省 = 现状）

| env | 默认 | 说明 |
|---|---|---|
| `ARGUS_TEXT_RERANKING_MODEL` / `_TOKENIZER` | 无 | 两件套，齐备才算 configured（无 DIM——输出是分数不是向量） |
| `ARGUS_TEXT_RERANKING_MAXLEN` | 512 | 截断上限（与 embedding 的 MAXLEN 独立：两族 reranker 支持更长上下文） |
| `ARGUS_TEXT_EMBEDDING_POOLING` | `mean` | `cls`/`mean`，透传 fork `PoolingKind` |
| `ARGUS_TEXT_EMBEDDING_MAXLEN` | 512 | 截断上限 |
| `ARGUS_TEXT_EMBEDDING_PIPELINE` / `ARGUS_TEXT_RERANKING_PIPELINE` | 表值 | 溯源串覆盖；**换模型族必设**，否则 pipeline 说谎。与 POOLING/MAXLEN 联动（改这两项也要改串） |
| `ARGUS_TEXT_RERANKING_POOL` | 回退 `ARGUS_ENGINE_POOL` | `CAPABILITY_KEYS` 循环自动生效（config.rs:241-257） |
| `ARGUS_TEXT_RERANKING_MAX_TEXTS` | 32 | 单请求 texts 条数上限，可调 1..=64（用户 2026-10-22 拍板默认 32；调大自担更长单请求占用，token 预算仍生效） |

**非法值处理**：`POOLING` 非 `cls`/`mean`、`MAXLEN` 非正整数 → **启动失败**。实现：`Config` 保留原始字符串，新增 `Config::validate(&self) -> Result<(), String>`，`main` 在 `from_env()` 后调用（**不在 `from_env` 内 panic**——既有测试大量直接调 `from_env`，且静默回退会让 pipeline 串与实际加工方式不符）。测试直接构造 `Config { text_embedding_pooling: Some("avg".into()), .. }` 调 `validate`，不依赖 env 锁。

**MAXLEN 上界提示**：模型自身 `max_position_embeddings` 不同（jina-reranker 1026 / bge-reranker 514），但 ONNX metadata 为空、fork 不读 `config.json`——**本期不做该 warn**（避免为一句提示新增 config.json 解析依赖）；README 写明推荐上限，超出由 tokenizer 截断兜底。

**默认 pipeline 表值**（`PIPELINE_TABLE` 新增行）：`text_reranking` = `jina-reranker-v2-base-multilingual|sigmoid|maxlen512`。

### 6.3 两对模型的标准 env 预设（README 落地）

- **jina 对**（双语，生产推荐）：现状 env + `TEXT_RERANKING_MODEL=.../jina-reranker-v2-base-multilingual/model_int8.onnx` + `_TOKENIZER`（int8 经 2026-10-22 评测：**排序指标与 fp32 无差**、快 1.5 倍、省 1GB 内存；分数不保证逐位一致）
- **bge 对**（纯中文）：`TEXT_EMBEDDING_*` 指 bge-small-zh-v1.5 + `POOLING=cls` + `DIM=512` + `PIPELINE="bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512"`；`TEXT_RERANKING_*` 指 bge-reranker-base + `PIPELINE="bge-reranker-base|sigmoid|maxlen512"`

`/models` 增 `text_reranking` 项（无 `dim` 字段；**未配置时也输出 pipeline 串**，与既有 16 项同规则）。既有 16 项在 env 不变时输出等值。

### 6.4 兼容性口径（精确表述）

- `/models`：解析后 JSON 的 `data.capabilities` **新增 `text_reranking` 一项**，既有 16 项逐字节等值；`tests/golden/models.json` 的 diff 仅新增项
- 既有 16 能力的推理响应、错误码、鉴权、backpressure 全部不变
- 新增端点 `/rerank` 无存量消费方

## 7. 错误处理与安全边界

- 校验顺序（硬纪律 4 的**窄例外**）：形状/长度/条数校验先于懒加载；**token 预算校验在懒加载之后**（tokenizer 在引擎内）——顺序见 §6.1
- 白名单门先于文件系统（ensure_enabled 既有语义）
- 构造期 fail loud：logits 缺失/末维≠1 → 拒载；运行期 shape/长度不符 → inference_failed
- **确定性契约（reranker 专属口径）**：同一请求（同一 texts 数组、同一顺序）重放**逐位确定**；但分数依赖整个批的 padding 形状——**texts 数组组成/顺序变化会导致分数漂移**（int8 实测 logits 变动 0.08-0.33，非 ulp 级；fp32 更小）。与 embedding 的逐条循环口径不同，文档与 README 如实写明
- **资源边界**：texts ≤32 且 `B × max_pair_len ≤ 8192`（实测：16×512 约 4.9s / +770MB 激活；32×512 被拒），最坏单请求 < 生产 shutdown 10s
- pool >1 时跨实例确定性未验证（生产 pool=1；README 注明）

## 8. 风险登记

| 风险 | 级别 | 缓解 | 验证 |
|---|---|---|---|
| 换族忘设 PIPELINE 覆盖 → 溯源串说谎 | 中 | README 预设表（整块复制）+ E2E 断言 bge 对返回 bge 串 + 启动日志打印 effective pipeline/pooling/maxlen | T9 |
| **改 POOLING/MAXLEN 忘改 PIPELINE** | 中 | 同上；README 把三个 env 写成联动组 | T9 |
| **最坏输入延迟/内存**（64×512 实测 17.7s/+4.7GB） | **高** | texts 上限 32 + `B × max_pair_len ≤ 8192`（Σ 指标可被 padding 绕过，实测 31短+1长 Σ=884 仍 9.7s）；§11 增加预算内最坏用例实测；生产 shutdown 10s 与 acquire 30s 预算校验 | T9/T10 |
| int8 精度 | 低（排序指标已实测无差） | 预设固定 int8；fp32 可 env 换文件；文档写明"排序无差、分数不逐位一致" | #22 评测 |
| 新 env 解析错误（POOLING=avg） | 低 | 启动失败（fail loud，`main` 校验） | T4 测试 |
| 对既有 16 能力回归 | 低 | 全部改动增量 + 默认值=现状 + golden diff 仅新增项 | T0/T7/T9 |
| 新能力键成 slots 保留字（历史 slots.toml 若用过该名 → 升级后拒启） | 低 | README 回滚/升级小节注明 | T8 |

## 9. 测试策略

fork：无模型自由函数单测（`encode_pairs` 形状/padding/**pad id≠0 词表**/截断 512、sigmoid 数学、`new_for_output` 输出名校验分支）+ 真模型门控自验（jina int8 与 bge fp32 各 3 断言：**固定 fixture 对**的正负判别、两输入不喂 token_type、输出形状 [B,1]）。
argus：oneshot 错误矩阵（≥10 例，照 embed.rs；**超预算用例归 T9 E2E**——它需要真实 tokenizer，`Config::default()` 只到 model_not_configured；预算判定抽纯函数 `budget_exceeded(b, max_pair_len)` 做单测）+ DTO 测试 + loader 三态（未配置/半套/白名单关）+ `model_paths_for`/`list_loaded`/provenance 接线测试 + models.rs 17 项 golden + pipeline override 生效单测。E2E：两对模型各起临时实例全验收（§11）。

## 10. 实施阶段

见 [plans/2026-10-22-text-reranking-and-model-swap-plan.md](../plans/2026-10-22-text-reranking-and-model-swap-plan.md)（F0 fork 增量 → F1 argus 能力/端点 → F2 文档/golden → F3 双对 E2E → F4 生产部署）。

## 11. 验收标准

1. cargo test 全绿（fork 与 argus）+ clippy 零警告（两仓库）
2. **jina 对临时实例（固定 fixture，阈值来自 2026-10-22 三模型实测标定）**：
   - 正对 `("孩子在沙滩堆城堡", "小女孩用沙子堆了一座城堡")` → sigmoid **> 0.6**（实测 int8 0.842 / fp32 0.829）
   - 负对 `("孩子在沙滩堆城堡", "季度财务报表数据汇总")` → sigmoid **< 0.1**（实测 0.028 / 0.023）
   - 且 `正对分 - 负对分 > 0.5`（实测间隔 0.814 / 0.806）
   - 同请求重放逐位相等（pool=1）
   - 错误矩阵全过；`pipeline` 串正确
3. bge 对临时实例：`POOLING=cls DIM=512` 下 /embed/text 正常（dim=512）+ /rerank 正常（同 2 的 fixture；bge 实测正 0.999 / 负 0.000，阈值同样适用）+ pipeline 串为 bge（证明 env 换族成立）
4. 默认 env 的实例 /models：**在与 golden 基线相同的图像模型 env 下**（非全默认），解析后 JSON 除新增 `text_reranking` 项外与基线等值
5. **预算内最坏输入实测**：`B×max_pair_len ≈ 8192`（如 16×512 或 32×256）单请求墙钟与 RSS 峰值记录在案，且 ≤ 生产 shutdown 预算；另记录**超预算拒绝**（如 32×512 → `B×L=16384`）的响应耗时（应快速返回 bad_request）
6. 生产部署 jina 对（int8）后：homepix 验证文档 V2-V6 复跑全过 + /rerank 生产冒烟

## 12. 回滚（三层，明确各自含义）

| 层 | 动作 | 含义 |
|---|---|---|
| fork 代码 | revert 单 commit | `new_for_output`/`encode_pairs`/`reranking.rs` 均为增量；`new` 委托改动由 T0-T3 既有测试守住 |
| argus 代码 | **不 revert**，随版本前进 | 类型改造（`ModelInfo.pipeline: String`、`pipeline_for` 方法）保留；`dim` 保持 `usize` 无 panic 路径；`encode_texts` 的 PAD_ID=0 不变 |
| argus 配置 | 新 env 全不设 | 既有 16 能力**可观测行为**与 `45e0bc9` 等价（由 golden + 既有 282 测试守住）；差异仅：`/models` 多 `text_reranking` 一项、`/rerank` 存在但返回 `model_not_configured` |
| 生产 | bundle 切回 `argus-0.1.0-c9ef409` | 整包回滚，含 env |

## 13. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-22 | 初版（批 D1 后文本族二期：reranking + 模型对 env 组合） |
| 2026-10-22 | 第一轮深度审查修正：texts 64→32 + token 预算（实测 64×512=17.7s/+4.7GB 推翻原 1.6s 估算）；pipeline 类型 `&'static str`→`String`（B2）；补 `new_for_output`（B1）；补 provenance/loaded/model_paths_for 接线（B3）；pad id 从 tokenizer 解析（B5）；新增 `TEXT_RERANKING_MAXLEN`；明确 UI 豁免；修正确定性口径与"逐字节不变"表述 |
| 2026-10-22 | 第二轮复核修正：预算指标 `Σ pair_len` → **`B × max_pair_len`**（实测 31短+1长 Σ=884 仍耗 9.7s，Σ 可被 padding 绕过）；预算校验位置明确为**懒加载后**（tokenizer 在引擎内，硬纪律 4 的窄例外，写明顺序与错误优先级）；`dim` 保持 `usize` 不引入 `expect` panic；pad_id 作用域限定 `encode_pairs`（`encode_texts` 保持 0）；验收 fixture 写死具体 pair 与阈值；最坏用例改为预算内；`Config::validate` 机制明确；回滚改三层表；补 `routes/mod.rs`、`max_pair_len` API、有效条数上限说明 |
