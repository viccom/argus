# argus 文本 Embedding 端点实施计划（批 D1）

> 状态：implemented（D0-D6 全部完成，2026-10-21；测试 258→282 全绿，E2E 6 项全过；生产 bundle 部署待做）
> 日期：2026-10-21
> 基线：argus @ `1b9adaa`（测试基线 255+3=258 全绿）；rust-onnx-infer 正式指向 `4b4615e`
> 权威源：设计 [designs/2026-10-21-argus-text-embedding-endpoint-design.md](../designs/2026-10-21-argus-text-embedding-endpoint-design.md)；需求 homepix 批 D1
> 取代关系：无
>
> 执行域：argus 仓库。TDD 先红后绿；完成定义 = cargo test 全绿 + clippy 零警告 + E2E 6 项 + golden 同步。

## 任务跟踪清单（执行时回填）

| ID | 阶段 | 任务 | 验收标准 | 状态 | 证据 |
|---|---|---|---|---|---|
| D0 | 准备 | 正式 `cargo update -p rust-onnx-infer`（→`4b4615e`）+ 基线全绿确认 | Cargo.lock 指向 `4b4615e`，258 全绿 | ✅ | 2026-10-21：`#01f66db7 → #4b4615e9`；255+3 全绿 + clippy 0 |
| D1 | F1 | config：16 键 + 三 env + 三件套判定 | config 测试先红后绿 | ✅ | 2026-10-21（subagent）：红=编译期 6×E0560 → 绿 31 passed。3 新测试（三 env 解析/半套不进 configured/白名单含新键）+ 断言 15→16 |
| D2 | F1 | engine：loader/池/预热/深检/PIPELINE_TABLE/provenance | loader 测试通过 | ✅ | 红=表覆盖测试 FAILED"缺 pipeline"+3×E0599 → 绿 27 passed。dim 加载期校验（不等即 Inference 拒载）、pool.warmup 短文本、深检独立 embed("ping") 分支 |
| D3 | F1 | routes/embed.rs：handler + oneshot 测试组（≥10 例） | oneshot 组先红后绿 | ✅ | 红=13 failed（路由未注册 404）→ 绿 16 passed：13 错误路径（空数组/129/128 边界/空串下标/2001 汉字且消息无原文/mode 非法/normalize=false/坏 JSON/非字符串/413→too_large/未配置/白名单关）+3 DTO 单测。路由注册提前到本任务（D3 转绿的必要条件） |
| D4 | F2 | models.rs 16 项+dim、health 深检分支、service_info、golden 重抓 | 全量测试绿 + golden 更新 | ✅ | 红=3 failed（断言先改 16）→ 绿 17 passed。golden 机制核实：tests/golden/*.json 是 E2E curl 基线（不被 cargo 测试消费），真正锁是 models.rs 内联 golden 测试（已 16 项）；golden/models.json 手补 text_embedding 未配置项；health-deep.json 零变化（未配置不含）。**全量 258→282（+24）** |
| D5 | F2 | README（端点/env/模型表）+ model-inventory + homepix 计划表批 D 段 + 本表/设计状态流转 | 文档齐、索引登记 | ✅ | 2026-10-21：README 四处（能力表/env 表两行/模型表/slots 保留字 15→16——slots.rs 用 CAPABILITY_KEYS 常量自动 16）；model-inventory §一 加 #19（如实标"生产 bundle 部署待做"）；homepix 计划表加 4b 批 D 段（D1 ✅ + D2+ 🚫）；设计 §3.2 纠偏（tokenizer 显式 env 对齐实现，引 ARGUS_OCR_DICT 先例） |
| D6 | F3 | E2E 验收 6 项（:7099 临时实例真模型） | 6 项证据回填 | ✅ | 2026-10-21 全过：①/models 四项齐（configured/enabled/pipeline/dim=512）②批量 2 条 count=2 len=512 模长 1.0000 溯源 model.onnx#69a0b846f4f1（与 T1 sha256 一致）③同输入两次 jq -S diff 空=逐位相等 ④cos(海边玩,堆城堡)=0.4899 > (海边玩,财务报表)=0.1927 与 gate-1 参照一致 ⑤空数组→bad_request、normalize=false→bad_request、并发 3 请求（fail-fast MAX_CONCURRENT=1）=1 成功+2 overload retryable:true、未配实例(:7098)→model_not_configured ⑥无 key 与 /health 逐字节同信封（"缺少 API key"）。另 HTTP 端到端 p95=3.6ms（10 次抽测，预算 30ms） |

## Task D0: 准备

`cargo update -p rust-onnx-infer`（此前零回归验证已两轮通过：`fdd7beb` 与 `4b4615e`，258 全绿 + clippy 0）。本次为正式锁定，随代码批次一并提交。

## Task D1: config

- `CAPABILITY_KEYS: [&str; 15]` → `[&str; 16]`，追加 `"text_embedding"`（数组长度即类型断言，编译期强制）；
- 新字段：`text_embedding_model: Option<PathBuf>`、`text_embedding_tokenizer: Option<PathBuf>`、`text_embedding_dim: Option<usize>`（env：`ARGUS_TEXT_EMBEDDING_MODEL` / `_TOKENIZER` / `_DIM`）；
- `model_paths_for("text_embedding")`：仅返回 onnx 路径（tokenizer 非 onnx，ocr_dict 先例不进清单）；configured 判定 = 三件套齐备；
- `configured_engines()` 加分支；`ARGUS_TEXT_EMBEDDING_POOL`/`_WARMUP_SIZE` 经既有 CAPABILITY_KEYS 循环自动生效（无需代码）；
- 测试：三 env 解析、半套配置（缺 tokenizer/dim）不进 configured_engines、白名单含新键、既有断言 15 处改 16。

## Task D2: engine

- `EngineManager` 增 `text_embedding: OnceCell<Arc<EnginePool<TextEmbeddingEngine>>>`（字段/初始化/list_loaded 各一行）；
- loader：`ensure_enabled` → 三件套校验（不齐 → ModelNotConfigured）→ `spawn_blocking` 构造（`TextEmbeddingEngine::new`，DeviceType 按配置）→ **dim 校验：`engine.dim() != cfg.dim` → Inference 拒载（fail loud，防模型替换）** → 池 → `pool.warmup(|e| e.embed("warmup").map(|_| ()))`（不走 warmup_engine helper——文本引擎无 OnnxInferPredict）；
- `PIPELINE_TABLE` 加 `("text_embedding", "bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512")`（表覆盖测试自动扩为 16 串互异）；
- `deep_health_check`：已加载时独立分支 `embed("ping")`（ok/degraded，名称 `text_embedding`；**不进 check_engine! 宏**）；
- 测试：未配置→ModelNotConfigured、白名单外→CapabilityDisabled（照既有 loader 测试范式）。

## Task D3: routes/embed.rs（照 classification.rs 范式）

DTO：`EmbedTextRequest { texts: Vec<String>, mode: Option<String>, normalize: Option<bool> }`（serde 默认宽松，逐项校验在 handler）；响应组装 `json!`（字段序对齐设计 §5）。

handler 顺序（硬纪律 4）：
1. `Result<Json<T>, JsonRejection>` → 解析失败 bad_request；**413 分支 → TooLarge**（`e.status()==PAYLOAD_TOO_LARGE`，map_field_err 同款判别法）；
2. `texts.is_empty() || len>128` → bad_request；逐条 `is_empty() || chars().count()>2000` → bad_request（消息 `"texts[i] 为空字符串/超 2000 字符上限"`，无原文）；
3. `mode` 非 {passage,query} → bad_request；`normalize==Some(false)` → bad_request（v1 不支持，显式报错）；
4. `engines.text_embedding()` → permit → **逐条 `pool.run(|e| e.embed(&text))` 循环**（计时 permit 之后起表，B5 纪律；embed 返回 Vec<f32>，无需再归一化——引擎构造已 normalize）；
5. 日志：`endpoint="embed_text", count, dim, inference_ms`——无文本。

oneshot 测试组（≥10）：缺 texts 字段（serde 拒）/ 空数组 / 129 条 / 空字符串元素 / 2001 字符元素 / mode 非法 / normalize=false / 坏 JSON / 未配置模型（code=model_not_configured）/ 白名单关（code=capability_disabled）/ 成功路径**用假引擎不可行（无 trait）——成功路径归 E2E**；DTO 序列化测试（dim/pipeline/embeddings 字段形状）。全部断言 `data.code` 与 `data.error` 关键词。

## Task D4: models / health / service_info / golden

- `models.rs`：entries 15→16（text_embedding：model+tokenizer+dim 三件套 configured；`models` 列表仅 onnx 文件名；**`dim: Option<usize>` 字段 skip_serializing_if None**——其他 15 项序列化逐字节不变）；pipeline 注入循环自动覆盖；
- `health.rs`：`service_info` endpoints 加 `POST /embed/text` 一行；build_router 加路由 `.route("/embed/text", post(routes::embed::embed_text))`；
- golden：**未配置实例**重抓 `models.json`（16 项，text_embedding configured=false）——零运行态字段，mask 约定下最稳；`health-deep.json` 不变（未加载不出现）；
- 全量 `cargo test` + `cargo clippy --all-targets` 双绿。

## Task D5: 文档

- README：端点表（/embed/text 一行含字段说明）、env 表（三个新 env + MAX_TEXT_CHARS=2000 常量说明 + normalize 仅 true）、模型表（bge-small-zh-v1.5 来源/Xenova 导出/指纹）、部署 bundle 说明（models/bge-small-zh-v1.5/ 两文件软链）；
- `docs/model-inventory.md`：回填 bge 行（E2E 后登记现役 §一，含 sha256 前 12 与实测数字时点）；
- `docs/homepix-integration-plan.md`：**加批 D 段**（D1 行指向需求文档与设计/计划链接，状态随回填）；
- `docs/README.md` 索引 + 本表/设计状态流转（draft→implemented）；
- AGENTS.md：架构地图 routes 行补 embed.rs（一行）。

## Task D6: E2E（验收清单 6 项）

`:7099` 临时实例（模型软链 `models/bge-small-zh-v1.5/`，env 三件套 + auth）：

1. `curl /models` → text_embedding 四项齐（configured/enabled/pipeline/dim=512）；
2. 批量 2 条 → count=2、每条 len==dim、模长≈1、pipeline 串正确；
3. 确定性：同输入两次响应 embeddings 逐位相等（jq 对比）；
4. 语义：cos(在海边玩,沙滩上堆城堡) ≈0.49 > cos(在海边玩,财务报表表格截图) ≈0.19（与 gate-1 参照一致）；
5. 错误形态：空数组→bad_request；未配实例→model_not_configured；并发压满→overload+retryable:true；
6. 鉴权：无 key 401 拒（与现有端点一致）。

另跑 p95 端到端抽测（HTTP 层，10 条查询）对齐预算口径。全部证据（命令+输出）回填本表。

## 门禁与提交纪律

- 完成定义：D1-D4 TDD 先红后绿 + 全量双绿 + golden 同步 + E2E 6 项，缺一不可；
- 提交：单批次 `feat(embed): POST /embed/text 文本 embedding 端点——批 D1（任务 D1）`（conventional 中文），含 Cargo.lock（`4b4615e`）与文档；模型文件不入库；
- 文档同步 D5 全链（README 是端点/env 唯一权威源）。

## 变更记录

- 2026-10-21：建立（draft）。承接 fork 计划（implemented）全部移交事项：空字符串 HTTP 前置校验、逐条循环策略、dim env 声明+加载期校验、/ui 豁免、latency_ms→queue_ms/inference_ms。
