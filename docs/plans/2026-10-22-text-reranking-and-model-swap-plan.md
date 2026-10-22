# 实施计划：文本 reranking + 同家族模型对 env 组合

> 状态：进行中（2026-10-22；T0-T8 已销账，T9 E2E 与 T10 生产部署待做。批准口径：texts 上限默认 32、`ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调至 64；TDD 先红后绿，完成定义 = 双仓 test 全绿 + clippy 0 + 双对 E2E + golden 同步）
> 日期：2026-10-22
> 基线：argus @ `45e0bc9`；rust-onnx-infer @ `49b8585`
> 权威源：设计 [designs/2026-10-22-text-reranking-and-model-swap-design.md](../designs/2026-10-22-text-reranking-and-model-swap-design.md)（**已经两轮深度审查修订**）
> 执行域：fork 仓库（F0）→ argus 仓库（F1-F4）。fork 构建注意 `OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include` env 旁路

## 任务跟踪清单（执行时回填，不预填）

| ID | 阶段 | 任务 | 验收标准 | 状态 | 证据 |
|---|---|---|---|---|---|
| T0 | F0 | fork 基线全绿确认（cargo test + clippy，带 openssl env） | 双绿 | ✅ | 2026-10-22：lib 110 全绿；new_engines_smoke 10 失败为**存量** DART 组硬编码 /Volumes/macEx 路径（stash 对照确认基线一致，非本次范围）；clippy 仅 1 条存量工具链漂移警告 |
| T1 | F0 | **runner 改造**：(a) 构造期解析 pad id 存字段 `pad_id`（`<pad>`/`[PAD]`，回退 0），**作用域仅 `encode_pairs`**（`encode_texts` 保持 `PAD_ID=0`，jina embedding tokenizer 也是 `<pad>=1`，不动以免改既有输入张量）；(b) 新增 `new_for_output(..., output_name, expect_last_dim: Option<usize>)`，`new` 委托之（`dim` 保持 `usize`，不引入 `Option`/`expect`）；(c) `encode_pairs` 自由函数（pair 模板/批内最长 padding/截断/用 `pad_id`）；(d) `OUT_LOGITS` 常量。单测：pair 特殊 token 序、padding 对齐、**pad≠0 词表**、`new_for_output` 缺输出名报错、**既有 embedding 测试零回归** | 先红后绿 + 零回归 | ✅ | 2026-10-22（subagent）：红=E0425×2（resolve_pad_id/encode_pairs 未定义）→ 绿 115；绿=encode_pairs 测试 5 例（pair 模板/padding 对齐/**pad=11 fixture**/截断）；既有 embedding 零回归且真模型自验实际执行过（新委托构造器路径） |
| T2 | F0 | `reranking.rs`：`TextRerankingEngine::with_options`（`new_for_output("logits", Some(1))`）+ sigmoid 纯函数 + 运行期 `shape==[B,1]`/`data.len()==B` 校验 + Send/Sync const 断言 + `pub fn max_pair_len(&self, query, texts) -> Result<usize>`（供 HTTP 层预算校验）+ mod.rs 挂载；纯数学单测 | 先红后绿 | ✅ | 2026-10-22（subagent）：红=E0425×2 → 绿 119（sigmoid 三测 + logits_shape 校验）；真模型冒烟：jina int8 正 0.778/负 0.024、bge 0.999/0.000、同批重放逐位相等、max_pair_len=20 | |
| T3 | F0 | 真模型门控自验（MODEL_DIR 约定 [SKIP]）：jina int8 与 bge fp32 各 3 断言（固定 fixture 正负判别 / 两输入不喂 token_type / 输出 [B,1]），**分数用实现后的 pad_id 重新记录**；fork `README.md` **与 `README.cn.md`** 模型表补 reranker 两行；push fork + argus `cargo update -p rust-onnx-infer` + argus 全绿零回归 | 自验过 + argus 全绿 | ✅ | 2026-10-22：自验 2 测试过（jina 0.7779/0.0241 间隔 0.754；bge 0.9990/0.0000）+ [SKIP] 门控；README 双语各 +2 行（发现既有缺口：fork README 无 jina-embeddings 行，未顺手修）；fork push `3ca82a3`+`1800743`；argus `cargo update` → `#1800743c`，279+3 全绿 + clippy 0 |
| T4a | F1 | **config 纯字段**（不碰 CAPABILITY_KEYS）：`text_reranking_model/tokenizer`；`TEXT_RERANKING_MAXLEN`、`TEXT_EMBEDDING_POOLING/MAXLEN`、两个 PIPELINE override（**保留原始字符串**，空串=未设）；新增 `Config::validate() -> Result<(), String>`（非法 POOLING/MAXLEN 报错）；单测直接构造 Config 调 validate（不依赖 env 锁） | 先红后绿 | ✅ | 2026-10-22（`5bf8684`）：8 新字段（两件套/双 PIPELINE 覆盖/POOLING 原始串/双 MAXLEN/MAX_TEXTS，空串=未设）+ `Config::validate()`（POOLING 非 cls\|mean、MAXLEN<1、MAX_TEXTS 越界 1..=64 → Err 含实际值；from_env 不 panic）+ `main` fail-closed 调用；测试 282→291（+9：validate 合法/非法 7 + from_env 解析 2）先红后绿，clippy 0 |
| T4b | F1 | **机械接线（原子 commit）**：`CAPABILITY_KEYS` 17；`model_paths_for` + `configured_engines` 分支（两件套齐）；`PIPELINE_TABLE` 加 text_reranking 行；models.rs entries 数组 16→17 + 内联 golden 计数；`main` 调 `Config::validate()`。先写全部新断言（红）→ 一次性接线（绿） | 全绿（含既有） | ✅ | 2026-10-22（`b79d8ea`）：CAPABILITY_KEYS 16→17、model_paths_for（tokenizer 不进清单，ocr_dict 先例）/configured_engines（两件套齐）/PIPELINE_TABLE（`jina-reranker-v2-base-multilingual\|sigmoid\|maxlen512`）/models entries 16→17 + golden 同步；先改断言跑红（3 个计数/golden 测试 FAILED）→ 接线全绿 291（含既有 282），clippy 0 |
| T5 | F1 | engine：`text_reranking: OnceCell` 字段 + `new()` 初始化 + `list_loaded` 分支；loader（`pool_for("text_reranking")`、`TEXT_RERANKING_MAXLEN`、构造期校验）；warmup `\|e\| e.rerank("预热", &["预热文档"]).map(\|_\| ())`；deep_health 独立分支（照 engine.rs:354-367）；**`EngineManager::pipeline_for(&self, cap) -> String`（override 优先，空串视为未设）** + `ModelInfo.pipeline` 改 `String`（`slots.rs:221` 用 `.to_string()`）+ `model_info`/`models.rs:247` 改调方法；**自由函数改名 `pipeline_table_value`，`engine.rs:1340` 的既有测试同步改调**；text_embedding loader 接 POOLING/MAXLEN（**`TEXT_EMBEDDING_MAX_LENGTH` 常量保留为 MAXLEN 的默认值来源，避免 dead_code**）；启动日志打印 effective pipeline/pooling/maxlen。测试：effective pipeline（默认=表值/override 优先/空串=未设/非空） | loader/provenance 测试过 + 既有全绿 | ✅ | 2026-10-22（`4121883`）：text_reranking OnceCell/loader（pool_for/MAXLEN/构造期校验）/warmup/深检独立分支 + `ModelInfo.pipeline` String 化（`EngineManager::pipeline_for` 方法统一出口、自由函数改名 `pipeline_table_value`、slots.rs `.to_string()`、models.rs 改调方法）+ embedding loader 接 POOLING/MAXLEN（常量转默认值来源）+ 启动日志 effective 值；测试 291→301（+10：loader 三态 3 + effective pipeline 3 + model_info 覆盖 1 + pooling 2 + maxlen 1），clippy 0 |
| T6 | F1 | routes/rerank.rs：DTO + handler（**校验顺序见设计 §6.1**：懒加载前形状/长度/条数 → 引擎 → `max_pair_len` 预算（`pool.run(\|e\| e.max_pair_len(&q, &t))`，`EnginePool::run` 内部已 `spawn_blocking`）→ permit → 推理）+ 预算判定抽纯函数 `budget_exceeded(b, max_pair_len)` 做单测 + **`map_json_rejection` 从 embed.rs 提到 common.rs 复用** + `src/routes/mod.rs` 加 `pub mod rerank;` + health.rs 路由注册 + service_info 行 + ranking tie-break（同分 index 升序）/非有限分数/scores 长度校验 + oneshot 错误矩阵 ≥10 例（**超预算用例留 T9**，无模型测试套件构造不出） | 先红后绿 | ✅ | 2026-10-22（`3734411`）：routes/rerank.rs 484 行（DTO/handler，校验顺序=懒加载前形状→引擎→`max_pair_len` 预算→permit→推理；预算纯函数 `budget_exceeded` 3 例；ranking tie-break 同分 index 升序、非有限分数/scores 长度校验）+ `map_json_rejection` 提 common.rs 复用 + 路由注册/service_info 行；oneshot 错误矩阵 13 例；rerank 模块 21 测试，全量 301→322 全绿，clippy 0 |
| T7 | F2 | models.rs 17 项断言 + `tests/golden/models.json` 补 text_reranking 未配置项（含 pipeline 串）+ **`debug_assert_eq!` 比较键集合而非仅数量** + 硬编码 16 逐项清理：models.rs:239/253/313-318/404/554-558/580-581/627、README:55/68、engine.rs:376 | 全量绿 + golden diff 仅新增项 | ✅ | 2026-10-22（`904439f`）：golden 补 text_reranking 未配置项（既有 16 项逐字节不变）+ models.rs `debug_assert_eq!` 键集合断言（防拼错/漏项，审查 C10）+ 硬编码计数对齐（models.rs 注释、engine.rs 保留字注释、README 观测行 15→17）；**遗留**：README 槽名注释行"16 个能力键"未及清理，T8 补齐 17；322 全绿（2026-10-22 复跑） |
| T8 | F2 | README（/rerank 端点行 + env 表 + reranker 模型两行 + **两对预设整块**（含 POOLING/MAXLEN/PIPELINE 三 env 联动 + 有效条数上限随长度变化）+ slots 保留字 17 + 升级提示（历史 slots.toml 若用过 text_reranking 需改名））+ model-inventory #22/#23 状态 + **docs/e2e-and-models.md 补 reranker 踩坑**（`<pad>=1`、pair 模板、`logits [B,1]`、int8 padding 抖动）+ **AGENTS.md「/ui 卡片」措辞修订（文本能力豁免）** + homepix 计划表批 D 段 D2 + 本表/设计状态流转 + docs/README.md 索引 | 文档齐、索引登记 | ✅ | 2026-10-22（本提交）：README 四处（端点行/env 3 行/模型 2 行/「文本模型对预设」节）+ slots 保留字 16→17（T7 遗留项）+ inventory #22/#23"argus 已接入" + e2e 踩坑 3 行（接口三差异/int8 抖动/Σ 被 padding 绕过）+ AGENTS 三道门措辞 + homepix D2 行（D2+ 行摘除 rerank）+ design 状态 implemented + docs/README 索引两行流转 |
| T9 | F3 | E2E 双对（临时实例真模型）：**jina 对**（现状 env + reranker int8）：固定 fixture（设计 §11.2 的 pair 与阈值：正 >0.6 / 负 <0.1 / 间隔 >0.5）+ 同请求重放逐位相等（pool=1）+ 错误矩阵 4 例 + **超预算拒绝**（32×512）耗时 + pipeline 串；**bge 对**（`POOLING=cls DIM=512` + bge reranker）：/embed/text dim=512 + /rerank 同 fixture + **pipeline 串为 bge**；**预算内最坏输入**（16×512）实测墙钟/RSS；**/models** 在与 golden 相同图像 env 下 diff 仅新增项 | 设计 §11 验收 1-5 全过 | ⬜ | |
| T10 | F4 | 生产 bundle 部署 jina 对（int8 reranker 软链 + 两件套 env）+ 切 systemd + 生产验收（homepix V2-V6 复跑 + /rerank 冒烟 + **shutdown 10s / acquire 30s 预算校验**）+ model-inventory 状态回填。**双实例（:7042/:7043）分别部署与冒烟**（若 7043 仍在运行） | 生产全过 | ⬜ | |

## 完成定义

T1-T8 先红后绿 + 全量双绿 + 双对 E2E（T9）+ 生产部署验收（T10），缺一不可。过程纪律：fork 改动独立 commit + push；argus 改动 conventional commit；实测数字带时点回填本表；**状态列只在验证证据落地后回填，不预填**。

## 审查修正对照（两轮深度审查）

### 第一轮（6 阻断 + 12 遗漏）

| 审查项 | 修正落点 |
|---|---|
| B1 runner 构造器不可复用 | T1(b) `new_for_output`；设计 §5 |
| B2 `pipeline` 是 `&'static str` | T5 改 `String` + `EngineManager::pipeline_for`；设计 §4 |
| B3 provenance/loaded/model_paths_for 接线缺失 | T4b + T5；设计 §3 |
| B4 最坏输入 17.7s/+4.7GB | 设计 §4/§7/§8/§11 + T6 预算 + T9 实测 |
| B5 reranker pad=1 | T1(a)；设计 §3/§4 |
| B6 验收阈值不可判定 | 设计 §11.2 写死 pair 与阈值 + T9 |
| O1 UI 卡片缺失 | 设计 §2 豁免 + T8 修订 AGENTS |
| O2 e2e-and-models 未列入 | T8 |
| O3 fork README.cn.md | T3 |
| O4 默认 reranking pipeline 串未定义 | 设计 §6.2 |
| O5 ranking 排序规则 | 设计 §6.1 + T6 |
| O6 reranking 无 MAXLEN env | 设计 §4/§6.2 + T4a |
| O7 硬编码 16 清单 | T7 逐项 |
| O8 POOLING/MAXLEN 变更也需改 pipeline | 设计 §6.2/§8 |
| O9 生产双实例 | T10 |
| O10 slots 保留字升级提示 | 设计 §8 + T8 |
| O11 `map_json_rejection` 私有 | T6 |
| O12 状态矛盾 | 见下方"待审批"说明 |

### 第二轮复核（新发现 3 阻断 + 若干高/中）

| 审查项 | 修正落点 |
|---|---|
| C1 预算校验需 tokenizer，与"校验先于懒加载"矛盾 | 设计 §4/§6.1 明确校验顺序为"形状校验→引擎→预算→permit"；T6 照此实现；T2 增 `max_pair_len` API |
| C2 `Σ token` 可被 padding 绕过（31短+1长 Σ=884 仍 9.7s） | 设计 §4/§6.1 改 `B × max_pair_len ≤ 8192`（实测数据入表） |
| C3 §11.5/T9 最坏用例会被预算拒绝 | 设计 §11.5 改为预算内用例 + 单列超预算拒绝耗时 |
| C4 `main` 校验丢失非法值信息 | 设计 §6.2 `Config::validate()` 机制；T4a 单测直接构造 Config |
| C5 pad id 可能改 jina embedding 行为 | 设计 §5 明确作用域仅 `encode_pairs`；T1(a) |
| C6 B6 fixture/阈值仍不存在 | 设计 §11.2 写死三条断言与阈值 |
| C7 自由函数去向与 dead_code | T5 改名 `pipeline_table_value` + effective pipeline 测试 |
| C8 `dim: Option` 引入 expect panic | 设计 §5 保持 `usize`；T1(b) |
| C9 `routes/mod.rs` 未列 | T6 |
| C10 T7 注释清单不完整 | T7 补齐 models.rs:239/253/404 |
| C11 golden 对比 env 前提 | 设计 §11.4 + T9 写明"与 golden 相同图像 env" |
| C12 设计 draft vs 计划 approved | **本次修订后设计仍为 draft，等你审批；批准后一并改 approved**（不拖到 T8） |
| C13 MAXLEN 上界 | 设计 §6.2 loader warn；T5 |
| C14 override 可破坏 pipeline 互异 | T5 effective pipeline 测试含非空；互异仅对表值保证（override 是用户自担） |
| C15 T3"3 断言"与 §11.2"3 组"措辞 | 设计 §11.2 已写具体 pair，T3 为每模型 3 条断言 |

## 已拍板（2026-10-22 用户批准）

1. **texts 上限默认 32，`ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调（1..=64）**；预算 `B × max_pair_len ≤ 8192` 不变。
2. **预算校验位于懒加载之后**（tokenizer 在引擎内，超预算请求会触发一次模型加载）——已接受，文档注明即可（设计 §4/§6.1 已含）。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-22 | 初版（T0-T10，待执行） |
| 2026-10-22 | 第一轮审查后修订（T1 扩 runner、T4 并入接线、T5 增 provenance、T6 增预算、T7-T10 补遗漏） |
| 2026-10-22 | 第二轮复核后修订：T4 拆 a/b；T1 保持 `dim: usize`；T5 自由函数改名；T6 补 `routes/mod.rs` 与校验顺序；T7 补注释清单；T9 fixture 写实与用例修正；新增"待审批两点" |
