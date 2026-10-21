# rust-onnx-infer 文本 Embedding 引擎实施计划（批 D 前置）

> 状态：implemented（T0-T8 全部完成，2026-10-21；fork `fdd7beb`+审查修正 `4b4615e` 已 push，argus 零回归验证通过并还原）
> 日期：2026-10-21
> 基线：rust-onnx-infer @ `01f66db`（工作副本 `/root/appRun/argus/rust-onnx-infer`，origin=viccom fork，upstream=javpower）；argus @ `29463df`
> 权威源：设计 [designs/2026-10-21-rust-onnx-infer-text-embedding-design.md](../designs/2026-10-21-rust-onnx-infer-text-embedding-design.md)；需求 text-embedding-requirements.md（homepix 批 D1）
> 取代关系：无
>
> 执行域：**fork 仓库**。argus 仓库在本计划内只读（T7 的 `cargo update` 为零回归验证，验证后还原；正式 update 属 argus D1 集成计划首个任务）。

## 任务跟踪清单（执行时回填）

| ID | 阶段 | 任务 | 验收标准 | 状态 | 证据 |
|---|---|---|---|---|---|
| T0 | 准备 | fork 工作副本与基线核对 | clone 就绪、HEAD == argus 锁定、upstream 静止确认 | ✅ | 2026-10-21：clone 至 `/root/appRun/argus/rust-onnx-infer`，HEAD `01f66db` == argus Cargo.lock 锁定；upstream remote 已配，upstream master 停在 `1c6ed67` == merge-base（fetch + GitHub API 双核实） |
| T1 | 前置 | 模型工件获取（**批准门**） | 文件就位 + sha256 记录 | ✅ | 2026-10-21（用户放行后执行）：`models/bge-small-zh-v1.5/{model.onnx,tokenizer.json,config.json}`，来自 HF `Xenova/bge-small-zh-v1.5`。sha256：model.onnx `69a0b846f4f1…749a`、tokenizer.json `48cea5d44424…ae26`、config.json `d4193ead3a81…f42f`；大小与 HF API 逐字节一致（94,851,877/439,125/716）。config：BertModel，hidden 512，max_position 512 |
| T2 | 前置 | gate-1 模型前置验证（shape dump / 池化判别 / p95 探针） | 三项结论回填设计文档 §3.3/§12 | ✅ | 2026-10-21 实测（uv + onnxruntime/tokenizers/numpy，本机 ur5900）：①I/O：`input_ids/attention_mask/token_type_ids` 均 `[B,L]` int64，单输出 `last_hidden_state [B,L,512]` float32，**无池化输出分支**；②池化：CLS 与 MaskedMean 均过语义判别，CLS 对比度更高（cos 0.4899/0.1927 vs 0.4146/0.1823）且为 FlagEmbedding 官方路径 → **默认 Cls**；③p95：短句(6 tok) 2.23ms / 50 tok 5.00ms / 74 tok 6.86ms（预算 30ms，4~13 倍余量），batch64 251ms（预算 1s）——**fp32 达标，int8 不引入**；④python 同输入两次逐位相等（确定性实证）。参照向量存 `/tmp/bge_ref_vectors.json`（4 句 CLS 归一化）。探针脚本 `/tmp/argus-bge-probe.py` |
| T3 | F1 | `TextSessionRunner` TDD（内存 fixture） | 单测先红后绿 | ✅ | 2026-10-21（subagent 实施）：红 E0425×9 → 绿 5/5（截断/批内最长 padding/单条自然长度/[B,L] 张量形状/空批次 InvalidArgument）。自由函数 `encode_texts`/`to_i64_tensors` + `EncodedBatch`；手工 pad（tokenizer 构造后不可变，避免共享 padding 状态突变）；tokenizers API 编译期适配 2 处（全限定模块路径、vocab 收 `Into<AHashMap>`） |
| T4 | F1 | `TextEmbeddingEngine` TDD（池化/归一化/维度） | 单测先红后绿 | ✅ | 2026-10-21：红 E0425/E0432 → 绿 5/5（masked-mean 忽略 pad 位逐位相等/cls 取首 token/L2 零向量幂等/维度报错含 shape/PooledOutput 选名分支）。合计 10/10。偏差（合理）：`run` 增 `output_name` 参数使 PooledOutput 分支纯逻辑可测；空字符串在 embed/embed_batch API 边界拒（InvalidArgument），encode_texts 只拒空批次 |
| T5 | F1 | 真模型自验（`[SKIP]` 门控 5 断言） | 有模型时 5 断言全过 | ✅ | 2026-10-21 实测：dim=512、模长 1.0000、单条两次逐位相等（to_bits 位级）、批重放逐位相等、cos(海边玩,堆城堡)=0.4899 > cos(海边玩,财务报表)=0.1927——**与 gate-1 python 参照完全一致**；交叉验证内嵌参照前 8 分量差 <1e-3。11/11（runner+embedding 全部） |
| T6 | F2 | fork 全绿 + README/模块文档 | test/clippy 双绿 + 双语 README 两表加行 | ✅ | 2026-10-21：lib 108 passed/0 failed（基线 97+新增 11）；engines_smoke 25 过（无模型 SKIP 正常）；new_engines_smoke 10 failed 为**存量**（git stash 双向验证失败集合逐名一致）；clippy --all-targets 仅 sahi/sliced_predictor.rs:261 存量警告，engines/text/ 与全部改动零警告。README.md/README.cn.md 三处同步（Highlights 措辞、Supported models 表、模型目录表） |
| T7 | F2 | push + argus 零回归验证 | argus 双绿后 Cargo.lock 还原 | ✅ | 2026-10-21：fork 单 commit `fdd7beb`（7 文件 +987/-2，纯增量）push origin master（`01f66db..fdd7beb`，delta 逐行核对=本设计）；argus `cargo update -p rust-onnx-infer` → `#fdd7beb1`：`cargo test --jobs 6` **255 passed + 3 lifecycle 全绿 0 failed**、`cargo clippy --all-targets` 零警告；`git checkout Cargo.lock` 还原后 argus 工作区仅余本计划三份文档（零提交纪律达成） |
| T8 | F2 | 性能定版记录（可与 T7 并行） | p95 数字带时点回填 | ✅ | 2026-10-21 实测（examples/text_embed_bench.rs，ur5900，--release，默认 OnnxRuntimeConfig intra=4，端到端含分词+池化+归一化）：短句(6tok) p50 2.36/p95 2.61ms；中句(32tok) p50 4.77/p95 5.49ms；长句(61tok) p50 7.60/p95 **9.00ms**（预算 30ms，3.3 倍余量）；embed_batch×64=402ms（预算 1s）。**发现**：本机 batch(402ms) 慢于逐条 embed() 循环(323ms)——批内全 pad 到最长白付计算+线程竞争；argus D1 若以吞吐为目标应走逐条循环（同时获得严格跨批逐位确定性，与设计 §4.4 口径协同） |

## Task 0: 准备与基线核对（已完成，见上表）

## Task 1: 模型工件获取（批准门，阻塞 T2 及以后全部）

从 HuggingFace `Xenova/bge-small-zh-v1.5`（或等价 `onnx-community/bge-small-zh-v1.5-ONNX`）获取：

```
models/bge-small-zh-v1.5/          # 沿用 fork 冒烟测试的 MODEL_DIR 约定（tests/engines_smoke.rs:33）
├── model.onnx        # fp32 主选（~91MB）
├── tokenizer.json    # (~1MB)
└（可选后续）model_quantized.onnx  # int8 性能备选，本批不取
```

- 落位 fork `models/`（或同样被 gitignore 的 `testmodels/`，由 `MODEL_DIR` 指向），**不入库**；
- 按仓库先例（B2）模型获取需用户批准——**本任务等待批准后执行**；
- 记录两文件 sha256（进位 argus 侧 `model` 溯源串是 argus 计划的事，此处只留档计划表证据列）。

验收：`models/bge-small-zh-v1.5/` 两文件存在，sha256 已记录于本表证据列。

## Task 2: gate-1 模型前置验证（写任何引擎代码之前）

三件事，结论回填设计文档（§3.3 池化默认值、§12 未决项销账）：

1. **shape dump**：`cargo run --release --example model_probe -- models/bge-small-zh-v1.5/model.onnx`（fork 自带工具）。确认：输入名集合（预期 `input_ids`/`attention_mask`/`token_type_ids`）、输出名与形状（预期 `last_hidden_state [B,L,512]`，是否有池化输出）。
   ⚠️ **不要用 `--run=HxW`**：该选项按 NCHW 的第 2/3 轴填 H/W 并用 `vec![0f32; count]` 造全零输入（`examples/model_probe.rs:128-145`），对 `[B,L]` int64 文本模型既不适用也无意义。运行期形状由下一步 python 探针实测。
2. **池化判别 + 参照向量**：`uv run --with onnxruntime --with numpy python` 直跑模型（uv 0.10.10 在本机可用），对固定 4 句中文（含"在海边玩"/"沙滩上堆城堡"/"财务报表表格截图"）分别做 CLS 与 MaskedMean 池化，与 FlagEmbedding 官方实现（或 model card 示例相似度）比对——**能复现官方相似度结构的池化即正解**（预期 CLS）。产出的参照向量留作 T5/E2E 交叉验证基线。
3. **p95 探针**：同 python 环境在 ur5900 上实测单条（10/30/64 token）× 100 次的 p95。注意：python onnxruntime 与 ort rc.13 内嵌 ORT 版本不完全一致，结果只作 **go/no-go 信号**（≥50ms 触发预案：int8 模型文件 / 调 intra_op），定版数字以 T8/argus E2E 为准。

验收：三项结论（输出形状记录、池化选定 + 参照余弦表、p95 数字带时点）回填设计文档变更记录。

## Task 3: `TextSessionRunner` TDD（engines/text/runner.rs）

**可测性是本任务的头号约束**：`BaseOnnxEngine` 构造必须加载真实 ONNX（`core/session_factory.rs:196` 缺文件即 `ModelNotFound`），所以分词与张量构造必须是**不依赖 `self.base` 的自由函数**，单测才能无模型运行（设计 §4.2 可测性关键设计）。

```rust
// 自由函数（pub(crate)），可内存词表直测
struct EncodedBatch { input_ids: Vec<i64>, attention_mask: Vec<i64>, token_type_ids: Vec<i64>, seq_len: usize }
fn encode_texts(tokenizer: &Tokenizer, texts: &[&str], max_length: usize) -> Result<EncodedBatch>;
fn to_i64_tensors(b: &EncodedBatch) -> Result<(Tensor<i64>, Tensor<i64>, Tensor<i64>)>;
```

测试用**内存构造**的 WordPiece 小词表（`WordPiece::builder().vocab(map).unk_token("[UNK]".into()).build()`，API 已核实：tokenizers 0.23.2 `models/wordpiece/mod.rs:71,78,157`），零文件 fixture：

| 测试 | 断言 |
|---|---|
| `encode_should_truncate_to_max_length` | 超长文本被截到 max_length（含特殊 token 计数） |
| `encode_batch_should_pad_to_batch_longest` | 短句 mask 尾部为 0、ids 尾部为 pad id，L == 批内最长 |
| `encode_single_should_keep_natural_length` | 单条路径 L == 自身长度（确定性口径 §4.4 的锚） |
| `tensors_should_have_batch_seq_shape` | 三路张量形状 `[B, L]`，dtype i64 |
| `encode_empty_batch_should_error_or_noop` | 空批次契约明确（按设计 §6：`InvalidArgument`） |

实现要点（全部有先例）：`Tokenizer::from_file` + `with_truncation/with_padding`（grounding_dino.rs:142-164）；`Tensor::from_array((shape, i64数据))` + `ort::inputs!` 直跑（:281-303）；`BaseOnnxEngine::with_config` 只作 session 容器。**v1 不写 `encode_pairs`**（reranking 落地时加，无消费者不预置）。

## Task 4: `TextEmbeddingEngine` TDD（engines/text/embedding.rs）

| 测试 | 断言 |
|---|---|
| `masked_mean_should_ignore_padding` | 同内容不同 padding 长度 → 逐位相等 |
| `cls_pool_should_take_first_token` | 取 `[:,0,:]` |
| `l2_normalize_zero_safe_and_idempotent` | 对齐 reid.rs:264-272 既有断言语义 |
| `dim_mismatch_should_error_with_shape` | 报错含 shape（reid.rs:133 先例） |
| `pooled_output_kind_should_read_pooled_branch` | PoolingKind::PooledOutput 分支选池化输出名 |

`engines/mod.rs` 加 `pub mod text;`（一行）；`text/mod.rs` 只导出 `TextEmbeddingEngine`（runner 保持 `pub(crate)`）。

## Task 5: 真模型自验（缺模型自动跳过，reid.rs:317 同款）

模型路径**沿用 fork 冒烟测试的 `MODEL_DIR` 约定**（`tests/engines_smoke.rs:33`，默认 `models`，可用 `MODEL_DIR` 指到任意位置）。目录存在才跑，否则打印 `[SKIP]`：

```
assert dim == 512
assert 模长 ≈ 1（±1e-3）
assert 单条两次 embed() 逐位相等
assert 同批 embed_batch() 重放逐位相等
assert cos("在海边玩","沙滩上堆城堡") > cos("在海边玩","财务报表表格截图")
```

语义判别若失败 = 池化/预处理有错，回 T2 结论复查（gate-1 的参照向量在此交叉验证）。

## Task 6: fork 全绿 + 文档

```bash
cargo test --jobs 6                  # 无模型环境全绿（自验自动 SKIP）
cargo test text_embedding -- --nocapture   # 有模型环境 5 断言全过
cargo clippy --all-targets --jobs 6  # 零警告（fork 无 deny 配置，按纪律执行）
```

> 注：fork 的 `cargo test` 无 `--jobs 6` 之外的额外环境要求；argus 侧 `--jobs 6` 是 argus 仓库的约定，fork 沿用同一节奏仅为机器负载一致。

- `engines/text/mod.rs` 头注释按 grounding_dino 模块文档格式（模型准备/ONNX I/O/示例）；
- `README.md`：`Supported models` 表加 Text embedding 行、`Module layout` 模型目录表加 `bge-small-zh-v1.5/` 行、Highlights 措辞去 "vision" 硬限定；`README.cn.md` 同步。

## Task 7: push + argus 零回归验证

1. fork 单 commit 提交（中文 conventional，先例 `feat(sahi): …` 风格）：`feat(text): TextEmbeddingEngine——BERT 分词/int64 直跑/CLS·Mean 池化/L2 归一化（批 D1 前置）`；
2. `git push origin master`（delta 恰为本 commit，执行前 `git log 01f66db..HEAD` 复核）；
3. argus 工作区：`cargo update -p rust-onnx-infer` → `cargo test --jobs 6` 全绿 + `cargo clippy --all-targets --jobs 6` 零警告 → **`git checkout Cargo.lock` 还原**。理由：argus main 不在 D1 集成前提前前进依赖；正式 update 是 argus D1 计划首任务（几分钟内重复，无验证损失）。

验收：argus 双绿输出留档本表证据列；`git -C argus status` 干净。

## Task 8: 性能定版记录（可与 T7 并行）

有条件时用 fork `examples/engine_bench.rs` 扩一个文本基准（或最小 ad-hoc 二进制）复测 p95；无条件则标注"待 argus E2E 定版"。数字必须带时点。

## 门禁与提交纪律

- 完成定义：T3-T6 先红后绿全程、fork 双绿（无模型 + 有模型）、T7 argus 双绿零回归，缺一不可销账；
- fork commit 中文 conventional 前缀，引用批 D1 与设计文档路径；
- 模型文件绝不入库（`/models/` 与 `/testmodels/` 均已 gitignore）；
- argus 仓库本计划内零提交（文档三份属 argus，随批 D 采纳由维护者一并提交）。

## 风险与回滚

风险登记见设计文档 §7。回滚：fork revert 单 commit / argus 还原 Cargo.lock 一行，两步互不牵连（设计 §11）。

## 变更记录

- 2026-10-21：建立（draft），T0 已完成回填。T1 为批准门，等待模型工件获取批准后启动 T2。
- 2026-10-21（审查修订）：①T2 明确禁用 `model_probe --run=HxW`（该选项按 NCHW 第 2/3 轴填 H/W 且造 f32 全零输入，对 `[B,L]` int64 文本模型不适用，`examples/model_probe.rs:128-145`）；②T3 改为自由函数 + 内存 WordPiece 词表（原写法的单测需构造 `BaseOnnxEngine` 而它必须加载真实模型，无模型跑不通）；③T1/T5 模型路径统一为 fork 冒烟测试的 `MODEL_DIR` 约定（原 `testmodels/` 与 `tests/engines_smoke.rs:33` 的默认 `models` 两套并存）；④T3 增空批次契约测试；⑤T6 补 `--jobs 6` 说明。
- 2026-10-21（实施完成）：T0-T8 全部销账。subagent 按 TDD 实施 T3-T6（红→绿证据见各行）；T7 fork commit `fdd7beb` push 后 argus 全量 258 测试全绿 + clippy 零警告，Cargo.lock 已还原；T8 引擎级定版 p95 最差 9.00ms。实施期环境事实：fork 在本机构建须带 `OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include`（其 `.cargo/config.toml` 清空 pkg-config 的 macOS 约定在 Linux 挡住 openssl-sys，未改该文件、以 env 旁路）。待办移交：argus D1 集成计划（首任务=正式 `cargo update` + `/embed/text` 端点 + golden 同步 + 批 D 回填）。
- 2026-10-21（深度代码审查）：逐行审读 `fdd7beb` 全部 987 行并独立复验（测试 11/11、clippy 仅 sahi 存量 1 条、bench 复跑数字同量级复现且 batch-vs-循环现象复现）。**0 个正确性/安全缺陷**；2 项修正已落 fork `4b4615e` 并 push：①Send+Sync 编译期断言（引擎无 trait bound 强制，防未来依赖漂移静默破坏 Arc 共享）；②PooledOutput doc 与实际报错时机不符（构造期而非推理期）纠偏。审查确认的关键移交事项：**argus D1 的 HTTP 层必须前置校验空字符串**（fork 兜底 InvalidArgument 经 argus `From<VisionError>` 映射为 inference_failed，语义码不对——需求要求 bad_request）。
- 2026-10-21（**第二轮独立复验**，针对"fork 是 argus 基础"的可靠性要求）：①git 层：HEAD == origin/master == `4b4615e`，工作区干净，`01f66db..HEAD` 仅 7 文件（engines/text 三文件 + mod 一行 + README 双语 + bench），**零意外改动**；②**绕过引擎 API 的独立交叉验证**：用 ort + tokenizers 手工复现 CLS+L2 路径，与 `TextEmbeddingEngine::embed` 输出 **max|diff|=0、位级不一致 0/512**（分词 id 101/1762/3862/6804/4381/102 符合 BERT [CLS]...[SEP]）；③`PAD_ID=0` 硬编码经真实 tokenizer 词表验证（id 0 == `[PAD]`）；④并发：Arc 共享 8 线程 × 50 次与单线程基线**逐位一致**，4 路不同批组成 embed_batch 并发批内一致；⑤边界：1202 token 超长文本正确截断到 512 不报错、128 条批量正常（argus 上限）、空串拒绝为 InvalidArgument、单空格正常；⑥存量失败**独立确认**：worktree 检出基线 `01f66db` 跑 new_engines_smoke 同样 19 passed/10 failed（全为视觉引擎缺模型），非本改动；⑦argus 对 `4b4615e` 全量 255+3 测试全绿 + clippy 零警告，Cargo.lock 已还原至 `01f66db`。
