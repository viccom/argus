# rust-onnx-infer 文本 Embedding 引擎设计（批 D 前置）

> 状态：implemented（2026-10-21；fork commit `fdd7beb` + 审查修正 `4b4615e`，计划 T0-T8 全部销账）
> 日期：2026-10-21
> 基线：rust-onnx-infer @ `01f66db`（viccom fork master HEAD，与 argus `Cargo.lock` 锁定一致；upstream javpower master 静止于 `1c6ed67` = fork merge-base，2026-10-21 实查）；argus @ `29463df`
> 权威源：需求 `text-embedding-requirements.md`（homepix 侧 2026-10-21 成文，批 D1）；实施进度回填 [plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md](../plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md)；能力级进度待批 D 采纳后回填 `homepix-integration-plan.md`
> 取代关系：无

## 1. 背景与目标

### 1.1 背景

homepix 批 D1（`POST /embed/text`）要求中文文本→归一化向量。argus 现栈无法实现，三条路径全部堵死（均经代码核实，见 §2）：

1. `BaseOnnxEngine` 的推理入口只收 `Tensor<f32>`（`run_inference`/`run_named`），而 BERT 族模型的三输入在 ONNX 图中声明为 int64，ORT 类型检查直接拒绝 f32 喂入；
2. 原始 `session` 字段为 `pub(crate)`，argus 侧拿不到裸会话；
3. grounding_dino 的注释原话："混合 f32/int64 输入无法走 `run_named`"——它在 crate 内锁 session 直跑，这是既定出路。

因此文本引擎必须落在 fork 内，且为**纯增量**改动。

### 1.2 目标

- fork 新增 `TextEmbeddingEngine`：文本批/单条 → 定维 L2 归一化向量，确定性输出；
- 结构上为未来的 `TextRerankingEngine`（cross-encoder）与更多 embedding 模型（bge-m3/m3e 等）预留正交扩展位；
- 对既有 40+ 视觉引擎零触碰、零行为变化。

### 1.3 非目标（YAGNI 边界）

- 不改 argus 仓库任何代码（argus D1 集成另立计划）；
- 不实现 reranking（只留结构缝，见 §4.6）；
- 不引入任何新依赖（tokenizers/onig 已在 fork 依赖树，见 §2）；
- 不做 GPU/量化特化（int8 量化模型文件可作为性能备选直接加载，属模型选择不属代码）；
- 不发布 crates.io（版本号 bump 见 §12 未决项）。

## 2. 代码现状（基线 `01f66db`，路径相对 rust-onnx-infer 仓库根）

| 事实 | 证据 |
|---|---|
| base 推理入口 f32-only | `src/core/base.rs:436`（`run_inference(Tensor<f32>)`）、`:443`（`run_named(Vec<(&str, Tensor<f32>)>)`） |
| session 不可外取 | `src/core/base.rs:101`（`pub(crate) session: Mutex<Session>`） |
| BERT 分词 + int64 输入完整先例 | `src/engines/grounding_dino.rs:142-164`（`Tokenizer::from_file` + truncation/padding 配置）、`:192-203`（encode→i64 三路）、`:270-326`（`Tensor::from_array` + `ort::inputs!` 直跑 + 按名取输出带 dtype 校验） |
| embedding 输出侧同类先例 | `src/engines/reid.rs:116-141`（维度校验报错带 shape + L2 兜底）、`:225-233`（零向量安全 `l2_normalize`）、`:243-395`（纯逻辑单测 + 缺模型 `[SKIP]` 自验两层测试） |
| tokenizer 错误变体已存在 | `src/error.rs:26`（`VisionError::Tokenizer`） |
| 线程可按引擎自配 | `src/core/runtime_config.rs:34-43`（`intra_op_threads` 默认 4 / `inter_op_threads`，builder 传入） |
| trait 装配宏 | `src/core/engine.rs:76`（`impl_engine_forward!`） |
| 引擎注册面 | `src/engines/mod.rs`（42 个 `pub mod`）；根导出 `src/lib.rs:48-61`（reid 无 factory、直连构造的先例成立） |
| tokenizers 依赖现状 | `Cargo.toml:26`（`tokenizers 0.23`，default-features=false + onig + hf-hub；argus 的 Cargo.lock 已含 tokenizers 0.23.2 / onig 6.5.3，**零新增编译负担**） |
| 测试门控约定 | `tests/engines_smoke.rs`（`MODEL_DIR` 定位 + `MODEL_TESTS=1` 严格模式，无模型自动跳过）；模型不入库（`.gitignore` 含 `/models/` `/testmodels/`） |
| gate-1 工具现成 | `examples/model_probe.rs`（ONNX I/O shape dump，reid 文档示范用法） |
| README 门面 | `README.md` Highlights 自述 "**40+ vision engines**"；`Supported models` / `Module layout` 两表按引擎分行（有 `grounding_dino/` 先例行）；另有 `README.cn.md` 需同步 |

## 3. 决策与备选

### 3.1 选型：`engines/text/` 子模块 + `pub(crate)` 共享 runner

```
engines/text/
├── mod.rs        # pub use embedding::TextEmbeddingEngine;（未来 + reranking）
├── runner.rs     # TextSessionRunner（pub(crate)）：分词→张量→直跑→输出快照
└── embedding.rs  # TextEmbeddingEngine（pub）：池化 + L2 + dim
```

理由：Text Embedding 与 Text Reranking 共享全部输入侧机制（BERT 族分词、truncation/padding、int64 三输入、命名张量直跑），仅编码形状（单文本 vs 查询-文档对）与后处理（池化归一化 vs logit→sigmoid）不同——一个确定的现消费者加一个已上路线图的第二消费者，抽象非投机。所有部件均有 fork 内先例（§2），新引擎是已验证部件的拼装。

### 3.2 被否方案

| 备选 | 否决理由 |
|---|---|
| 泛化 `BaseOnnxEngine.run_*` 支持类型化张量 | 波及 40+ 引擎的共享核心，无新能力（grounding_dino 已示范直跑出路），风险收益倒挂 |
| `OnnxInferenceEngine` 泛化为 `InferenceEngine<Input, Output>` | 全生态破坏性重构，纯美学收益 |
| 单引擎 + mode 标志同时做 embed/rerank | 两个对外契约（dim/normalized vs 分数）塞一个 struct，API 说谎 |
| 读 sentence-transformers `config.json` 自动推断池化 | 隐式魔法，错配静默产出错误向量；显式参数 + gate-1 实测响亮得多 |

### 3.3 关键事实修正：bge 是 CLS 池化，不是 mean

FlagEmbedding 的 BGEEncoder 用 **CLSPooler + normalize**（m3e 系才是 mean pooling）。因此池化必须是构造参数而非写死（§4.3 `PoolingKind`），且 gate-1 以"能否复现官方相似度"判别选型——池化选错是最典型的静默劣化。

## 4. 架构与数据流

### 4.1 公开 API（fork 风格，构造签名对齐 grounding_dino）

```rust
// engines/text/embedding.rs —— 唯一公开导出
pub struct TextEmbeddingEngine {
    runner: TextSessionRunner,
    pooling: PoolingKind,
    normalize: bool,
}

pub enum PoolingKind { Cls, MaskedMean, PooledOutput }

impl TextEmbeddingEngine {
    /// bge 约定默认：Cls + normalize，max_length=512
    pub fn new(model_path, tokenizer_path, device) -> Result<Self>;
    pub fn with_options(model_path, tokenizer_path, device,
                        pooling: PoolingKind, normalize: bool,
                        max_length: usize, runtime_config: OnnxRuntimeConfig) -> Result<Self>;
    pub fn embed(&self, text: &str) -> Result<Vec<f32>>;            // 单条，自然长度
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>; // 批量，批内最长 padding
    /// 模型输出末维（构造期从 session 输出 shape 解析；解析不到静态维即构造失败，不返回 Option）
    pub fn dim(&self) -> usize;
    pub fn max_length(&self) -> usize;
}
```

### 4.2 `TextSessionRunner`（`pub(crate)`，不给下游）

持 `BaseOnnxEngine`（**仅作 session 容器 + 线程/设备配置**；`load_input_info` 对 `[B,L]` 二维输入跳过 NCHW 解析、留无害的 640×640 默认值，引擎不读它）+ `Tokenizer` + `max_length`。v1 方法面：

- `encode_batch(&[&str])` → 三路 i64 张量 `[B, L_max]`（truncation 至 max_length、padding 至批内最长，`encode` 自动加特殊 token）；
- `run(&tensors)` → 锁 session、`ort::inputs![input_ids => .., attention_mask => .., token_type_ids => ..]` 直跑（grounding_dino.rs:296 同款）；
- 按名取 f32 输出快照（dtype 校验，`snapshot_f32` 同款）。

**可测性关键设计**：分词与张量构造拆成**不依赖 `self.base` 的自由函数**（`encode_texts(tokenizer, texts, max_length) -> EncodedBatch`，`EncodedBatch { input_ids, attention_mask, token_type_ids, seq_len }`；`to_i64_tensors(&EncodedBatch) -> (Tensor<i64>, Tensor<i64>, Tensor<i64>)`）。理由：`BaseOnnxEngine` 构造必须加载真实 ONNX 文件（`session_factory.rs:196` 缺文件即 `ModelNotFound`），若分词逻辑长在 runner 上，T3 的"无模型单测"根本无法运行。自由函数纯逻辑、可用内存 WordPiece 词表直接测。

reranking 落地时**加一个** `encode_pairs` 方法，其余不动（§4.6）。v1 不预置（无消费者的代码是死代码）。

### 4.3 数据流

```
texts → tokenizer encode（truncate≤max_length, pad 批内最长, 自动特殊 token）
      → input_ids / attention_mask / token_type_ids  [B,L] i64
      → session.run（命名输入直跑）
      → last_hidden_state [B,L,H] 快照
      → PoolingKind：Cls（取 [:,0,:]）| MaskedMean（mask 加权平均，pad 位贡献 0）| PooledOutput（直接读池化输出）
      → L2 归一化（reid::l2_normalize 语义：零向量保持为零，幂等）
      → Vec<Vec<f32>>（每条长度 = dim()）
```

### 4.4 确定性口径（写死在模块文档）

- `embed(text)`：单条自然长度，**任意调用上下文下逐位确定**（homepix 查询路径恒走此 API）；
- `embed_batch(texts)`：padding 长度 = 批内最长，**同批重放逐位确定**；跨批组成变化可能引入末位 ulp 漂移（GEMM 分块随矩阵尺寸变化）——homepix 全库重算按 `source_hash` 整体覆写不做向量比对，无实害；若 argus 需要严格跨批逐位稳定，路由层循环调 `embed()`。
- 池内多实例加载同一模型文件 + 同一 `OnnxRuntimeConfig`，实例间结果一致。

### 4.5 线程与并发

- 线程经 `OnnxRuntimeConfig`（默认 intra=4）按引擎自配，不涉及全局 setter（与 argus B4 计划项解耦）；
- `Tokenizer` 与 `BaseOnnxEngine` 组合的 Send+Sync 已由 `GroundingDinoEngine` 实证（同字段组合实现了 `Send + Sync` 的 trait）。

### 4.6 未来扩展位

- **TextReranking**：`engines/text/reranking.rs` 新文件（~150 行）：runner 加 `encode_pairs`、后处理 logit→sigmoid，无池化/dim/归一化；
- **其他 embedding 模型**（bge-m3/m3e/GTE）：全部落在 `PoolingKind` + tokenizer.json 参数面内，零代码改动；
- 更远的 CLIP/SigLIP 文本塔同享张量管线，不为此设计但结构不堵死。

## 5. 契约与兼容性

**纯增量清单**：`engines/text/{mod,runner,embedding}.rs` 三个新文件 + `engines/mod.rs` 加一行 `pub mod text;`。

**零触碰清单**：`core/`（base/engine/factory/runtime_config/session_factory 全部）、既有 40+ 引擎、`imaging/`、`sahi/`、`model/`、`lib.rs` 根导出（reid 直连构造先例，不加 factory）。

**与 `OnnxInferenceEngine` trait 的关系**：不实现。该 trait 是图像向契约（`predict(&self, &Image)`）；grounding_dino 因属视觉任务桩实现为 Unsupported，纯文本引擎不是视觉任务，不参与该 trait。下游（argus）的 `EnginePool<E>` 只要求 `Send + Sync + 'static`，深检/预热由 argus 侧直调 `embed()`/`embed_batch()` 完成——此影响归 argus D1 集成计划。

**`pipeline` 串责任在 argus**（如 `bge-small-zh-v1.5|cls-pool-norm-v1`）；fork 引擎不感知。

**README 门面**：`Supported models` 加一行（Text embedding 类别，bge-small-zh-v1.5）、`Module layout` 模型目录表加行、Highlights 措辞从 "40+ vision engines" 调整为涵盖 text（如 "40+ engines"）；`README.cn.md` 同步。

## 6. 错误处理与安全边界

| 情形 | 映射 |
|---|---|
| tokenizer 加载/编码失败 | `VisionError::Tokenizer`（error.rs:26，argus 侧归 `inference_failed`） |
| 输出缺失/维度不符 | `VisionError::Inference`，报错带 shape（reid.rs:133 先例） |
| 空 texts / 空字符串边界 | `VisionError::InvalidArgument` |
| ORT 错误 | `From<ort::Error>` 既有转换 |

安全边界：**无网络**——只用 `Tokenizer::from_file` 本地路径（hf-hub feature 在依赖树但不使用其下载 API）；构造期日志只记模型路径/输出名/配置（grounding_dino 先例），**不记请求文本**（homepix 需求 §7：人物名与地点隐私）。

## 7. 风险登记

| 风险 | 级别 | 缓解 | 验证手段 |
|---|---|---|---|
| 池化选错（CLS vs Mean）静默劣化 | 高影响/低概率 | `PoolingKind` 显式参数；gate-1 以官方相似度复现判别 | python 参照向量余弦对比 |
| Xenova 导出形状非标（输出名/池化层有无） | 中/低 | gate-1 `model_probe` shape dump 前置；`PoolingKind::PooledOutput` 兜底 | dump 输出 + 自验断言 dim=512 |
| 批间 ulp 漂移破坏确定性预期 | 中/中 | §4.4 口径写死；单条 API 恒自然长度 | 自验两次逐位相等（单条+同批重放） |
| CPU p95 不达标（≤64 token ≤30ms） | 中/低 | python 探针在写码前实测（go/no-go）；intra_op 自配；int8 量化模型文件为备选（model hash 天然区分向量空间） | gate-1 探针 + argus E2E 复测 |
| fork push / cargo update 拖入无关变更 | 低/低 | upstream 静止 + fork master 零漂移已核（2026-10-21）；delta = 本设计单 commit，逐行审 | `git log 01f66db..<new>` |
| tokenizers 0.23 API 变动 | 低/低 | 同版本同 API 已在 grounding_dino 生产使用 | 编译即验 |

## 8. 测试策略（三层，全在 fork 仓库内）

1. **内存 fixture 单测（无模型、无文件）**：`tokenizers::models::wordpiece::WordPiece::builder().vocab(map).unk_token("[UNK]".into()).build()` 纯内存构造小词表（API 已核实存在：tokenizers 0.23.2 `models/wordpiece/mod.rs:71,78,157`）→ 测**自由函数** `encode_texts` / `to_i64_tensors`：truncation 上限、批内最长 padding、三路张量形状 `[B,L]`、attention_mask 正确性。**不构造 `BaseOnnxEngine`**（其构造需真实模型文件）；
2. **纯逻辑数学单测**：MaskedMean 忽略 pad 位（同内容不同 padding 长度结果一致）、Cls 取首 token、L2 零向量安全幂等（对齐 reid 既有测试）、维度不符报错含 shape；
3. **真模型自验（缺模型自动跳过，reid.rs:317 同款）**：dim=512、模长≈1、单条两次逐位相等、同批重放逐位相等、语义判别（"在海边玩"×"沙滩上堆城堡" 余弦 > ×"财务报表表格截图"）。模型路径解析**沿用 fork 冒烟测试的 `MODEL_DIR` 约定**（`tests/engines_smoke.rs:33`，默认 `models`），即模型放 `models/bge-small-zh-v1.5/` 或经 `MODEL_DIR` 指向的任何位置；`/models/` 与 `/testmodels/` 均已 gitignore，二者择一即可。

既有 `tests/engines_smoke.rs` 不受影响（新引擎无模型自动跳过）。

## 9. 实施阶段与门禁

任务拆分与验收见 [plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md](../plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md)。总门禁：fork `cargo test` 全绿（无模型环境）+ 有模型自验通过 + `cargo clippy --all-targets` 零警告 + argus `cargo update -p rust-onnx-infer` 后 `cargo test --jobs 6` / `cargo clippy --all-targets` 零回归。

## 10. 验收标准（可执行、可判定）

```bash
# fork 仓库（无模型环境）
cargo test --jobs 6            # 全绿（真模型自验自动 SKIP）
cargo clippy --all-targets --jobs 6   # 零警告
# fork 仓库（模型就位后：models/bge-small-zh-v1.5/ 或 MODEL_DIR 指向处）
cargo test text_embedding -- --nocapture   # 自验 5 断言全过（dim/norm/逐位×2/语义）
# argus 仓库（T7 零回归验证，验证后还原 Cargo.lock，正式 update 归 argus D1 计划）
cargo update -p rust-onnx-infer && cargo test --jobs 6 && cargo clippy --all-targets --jobs 6
```

## 11. 回滚

fork 侧：revert 单个增量 commit（纯新增文件 + mod 一行，无迁移）。argus 侧：还原 `Cargo.lock` 一行即回到 `01f66db`。两步互不牵连。

## 12. 未决项

- ~~模型工件获取渠道与批准~~（2026-10-21 已放行并下载，见计划 T1 证据）；
- ~~`PoolingKind` 默认值按 gate-1 结论定（预期 Cls）~~（**2026-10-21 gate-1 定论：Cls**——CLS/Mean 均过语义判别，CLS 对比度更高且为 FlagEmbedding 官方路径，实测见计划 T2 证据）；
- ~~int8 量化变体不进本批（fp32 达标即不引入）~~（2026-10-21 gate-1 实测 fp32 p95 最差 6.86ms，**int8 确认不引入**）；
- crates.io 版本 bump（0.2.0→0.3.0）为发布决策，git 依赖不感知，本批不动；
- argus D1 集成计划另立（等本计划 T7 完成解锁）。

## 13. 变更记录

- 2026-10-21：建立（draft）。基于 fork @ `01f66db` 深读结论与 homepix 批 D1 需求。
- 2026-10-21（审查修订）：①`dim()` 由 `Option<usize>` 改为 `usize`（bge 输出末维静态，解析不到即构造失败，fail loud）；②§4.2 增加"可测性关键设计"——分词/张量构造拆为不依赖 `self.base` 的自由函数，因 `BaseOnnxEngine` 构造必须加载真实模型（`session_factory.rs:196`）；③§8 测试策略改为自由函数单测 + `MODEL_DIR` 路径约定；④补 tokenizers WordPiece 内存构造 API 的核实行号（0.23.2 `models/wordpiece/mod.rs:71,78,157`）。
- 2026-10-21（gate-1 实测回填）：模型 Xenova/bge-small-zh-v1.5 fp32 已就位（计划 T1）。I/O 实测：三 int64 `[B,L]` 输入 + 单输出 `last_hidden_state [B,L,512]`，无池化输出分支；池化定论 **Cls**（CLS/Mean 均过语义判别，CLS 对比度更高且为官方路径）；fp32 p95 最差 6.86ms（预算 30ms），**int8 不引入**；python 侧同输入逐位相等实证。详见计划 T2 证据。
- 2026-10-21（实施完成，状态 draft→implemented）：fork commit `fdd7beb`（`engines/text/` 三文件 + `engines/mod.rs` 一行 + README 双语 + `examples/text_embed_bench.rs`，+987 行纯增量）已 push origin master；argus `cargo update` 后 258 测试全绿 + clippy 零警告（Cargo.lock 已还原，正式 update 归 argus D1 计划）。实施期两个新事实：①fork 在本机构建须带 `OPENSSL_LIB_DIR/OPENSSL_INCLUDE_DIR` env（旁路其 `.cargo/config.toml` 清空 pkg-config 的 macOS 约定，未改文件）；②引擎级实测（--release，端到端）p95 最差 9.00ms/61tok、batch64 402ms，且**本机 batch 慢于逐条循环**——argus D1 吞吐路径建议逐条 `embed()` 循环（兼得严格跨批确定性）。
