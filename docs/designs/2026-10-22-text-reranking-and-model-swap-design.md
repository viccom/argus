# 设计：文本 reranking 能力 + 同家族模型对（embedding × reranker）env 级组合

> 状态：draft（待审批）
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
- 不做单实例同时运行两对模型（内存翻倍无场景；切换 = env + 重启）
- 不做 reranker 分数校准/阈值过滤（消费方自行取 top-N）
- 不改既有 15 个能力与 text_embedding 的默认行为（env 不设 = 逐字节现状）

## 3. 代码现状（锚点）

| 事实 | 位置（基线） |
|---|---|
| `CAPABILITY_KEYS: [&str; 16]` | src/config.rs:12 |
| 路由注册点（`.route("/embed/text", ...)`） | src/routes/health.rs:164 |
| embedding 加载器硬编码 MaskedMean + maxlen 512 | src/engine.rs loader（`with_options`） |
| `PIPELINE_TABLE` 硬编码单串（jina） | src/engine.rs:71 |
| fork：runner 有 `encode_texts`（自由函数可无模型测）、`has_token_type_input` 两/三输入分支 | rust-onnx-infer src/engines/text/runner.rs |
| fork：`PoolingKind{Cls,MaskedMean,PooledOutput}`、`with_options` | rust-onnx-infer src/engines/text/embedding.rs |
| 两族 reranker 输出名 `logits [B,1]`（gate-1 实测） | 模型目录 gate1 脚本与输出 |

## 4. 决策与备选

| 决策 | 备选 | 为什么 |
|---|---|---|
| 池化/maxlen 暴露为 env（`POOLING`/`MAXLEN`，默认 = 现状） | 保持硬编码 | bge 对需要 CLS；默认值保证零回归 |
| `PIPELINE` 串加 env 覆盖（不设 → 表值） | 按模型目录名自动推导 | 目录名经软链不稳定、格式漂移伤 golden；覆盖 env 最小且诚实（换族必设，README 写明） |
| reranker 输出恒 sigmoid | 加开关 | 两族官方约定一致，YAGNI |
| rerank 单查询批模式（query 固定 + texts 批量） | 逐对循环 | cross-encoder 一批一前向，批是效率的全部意义；确定性契约见 §7 |
| texts 上限 64（embed 是 128） | 128 | 重排 ~25ms/对（fp32 实测 120-150ms@8），64×25ms≈1.6s 是可接受的准入占用；128 会到 3s+ |
| `logits` 末维构造期校验 =1（fail loud） | 运行期发现 | 照 embedding dim 校验先例，防模型换错文件 |

## 5. 架构与数据流

```
POST /rerank {"query","texts"}
  → parse/校验（硬纪律 4：先于懒加载）
  → EngineManager::text_reranking()（OnceCell 懒加载，ensure_enabled 白名单门）
  → permit（B5 计时纪律）
  → pool.run(|e| e.rerank(query, texts))   // 一批一前向 → sigmoid → Vec<f32>（输入序）
  → 响应：scores（输入序）+ ranking（按分降序的 {index,score}）
```

fork 侧（纯增量，不动既有引擎）：

```
runner.rs: + encode_pairs 自由函数（tokenizer.encode(query, doc)，批内最长 padding，复用 PAD_ID）
           + OUT_LOGITS 常量；runner 复用既有 run(batch, output_name)
embedding.rs 旁新增 reranking.rs: TextRerankingEngine{runner}::with_options(model, tokenizer, device, max_length, ort_cfg)
           构造期：锁 session 校验 logits 声明末维 ==1；sigmoid 纯函数 + Send/Sync const 断言
```

## 6. 契约与兼容性

### 6.1 新端点 `POST /rerank`（JSON）

请求：`{"query":"...","texts":[".."]}`；query 1..2000 字符；texts 1..64 条、单条 1..2000 字符、空字符串拒绝（校验顺序与错误码照抄 embed.rs 范式）。

成功响应（信封同全家）：

```json
{"status":"ok","data":{"queue_ms":0,"model":"model.onnx#<sha12>","pipeline":"...|sigmoid|maxlen512",
 "sigmoid":true,"count":2,"scores":[0.91,0.02],"ranking":[{"index":0,"score":0.91},{"index":1,"score":0.02}],"inference_ms":37}}
```

错误码零新增：bad_request / too_large / model_not_configured / capability_disabled / inference_failed / internal / overload（overload 唯一 retryable）；鉴权失败特例与全端点一致（200 信封无 code）。

### 6.2 新 env（全部缺省 = 现状）

| env | 默认 | 说明 |
|---|---|---|
| `ARGUS_TEXT_RERANKING_MODEL` / `_TOKENIZER` | 无 | 两件套，齐备才算 configured（DIM 无对应物——输出是分数不是向量） |
| `ARGUS_TEXT_EMBEDDING_POOLING` | `mean` | `cls`/`mean`，透传 fork `PoolingKind` |
| `ARGUS_TEXT_EMBEDDING_MAXLEN` | 512 | 截断上限 |
| `ARGUS_TEXT_EMBEDDING_PIPELINE` / `ARGUS_TEXT_RERANKING_PIPELINE` | 表值 | 溯源串覆盖；**换模型族必设**，否则 pipeline 说谎 |
| `ARGUS_TEXT_RERANKING_POOL` | 回退 `ARGUS_ENGINE_POOL` | 既有循环自动生效 |

### 6.3 两对模型的标准 env 预设（README 落地）

- **jina 对**（双语，生产推荐）：现状 env + `TEXT_RERANKING_MODEL=.../jina-reranker-v2-base-multilingual/model_int8.onnx` + `_TOKENIZER`（int8 经 2026-10-22 评测与 fp32 精度无差、快 1.5 倍、省 1GB 内存）
- **bge 对**（纯中文）：`TEXT_EMBEDDING_*` 指 bge-small-zh-v1.5 + `POOLING=cls` + `DIM=512` + `PIPELINE="bge-small-zh-v1.5|cls-pool-norm-v1|maxlen512"`；`TEXT_RERANKING_*` 指 bge-reranker-base + `PIPELINE="bge-reranker-base|sigmoid|maxlen512"`

`/models` 增 `text_reranking` 项（无 `dim` 字段）；既有 16 项输出在 env 不变时逐字节不变（golden 锁）。

## 7. 错误处理与安全边界

- 校验先于懒加载（硬纪律 4）；错误消息不含 query/texts 原文（隐私，`texts[i]` 下标定位）
- 白名单门先于文件系统（ensure_enabled 既有语义）
- 构造期 fail loud：logits 缺失/末维≠1 → 拒载
- 确定性契约：同请求（同 batch 组成）逐位确定；不同批组成的跨批 ulp 漂移可能（GEMM 分块随形状变，与 embedding 同口径，文档如实写）

## 8. 风险登记

| 风险 | 级别 | 缓解 | 验证 |
|---|---|---|---|
| 换族忘设 PIPELINE 覆盖 → 溯源串说谎 | 中 | README 预设表 + E2E 断言 bge 对返回 bge 串 | T9 |
| int8 精度 | 低（已实测无差） | 预设固定 int8；fp32 可 env 换文件 | #22 评测 |
| 新 env 解析错误（如 POOLING=avg） | 低 | 非法值启动即 panic（fail loud，config 解析惯例） | T4 测试 |
| rerank 长文本内存峰值 | 低 | maxlen 512 截断 + texts≤64 上界 | T9 冒烟 |
| 对既有 16 能力回归 | 低 | 全部改动增量 + 默认值=现状 | T0/T3/T9 golden 逐字节对比 |

## 9. 测试策略

fork：无模型自由函数单测（encode_pairs 形状/padding/截断、sigmoid 数学）+ 真模型门控自验（jina int8 与 bge 各 3 断言：正对>负对、两输入不喂 token_type、batch 形状）。
argus：oneshot 错误矩阵（≥10 例，照 embed.rs）+ DTO 测试 + loader 三态（未配置/半套/白名单关）+ models.rs 17 项 golden。E2E：两对模型各起临时实例全验收（§11）。

## 10. 实施阶段

见 [plans/2026-10-22-text-reranking-and-model-swap-plan.md](../plans/2026-10-22-text-reranking-and-model-swap-plan.md)（F0 fork 增量 → F1 argus 能力/端点 → F2 文档/golden → F3 双对 E2E → F4 生产部署）。

## 11. 验收标准

1. cargo test 全绿（fork 与 argus）+ clippy 零警告（两仓库）
2. jina 对临时实例：/rerank 正对分>负对分（sigmoid 差 ≥0.3）、同请求逐位确定、错误矩阵全过、`pipeline` 串正确
3. bge 对临时实例：`POOLING=cls DIM=512` 下 /embed/text 正常（dim=512）+ /rerank 正常 + pipeline 串为 bge（证明 env 换族成立）
4. 默认 env 的实例 /models 输出与基线逐字节一致（除新增 text_reranking 项）
5. 生产部署 jina 对（int8）后：homepix 验证文档 V2-V6 复跑全过 + /rerank 生产冒烟

## 12. 回滚

fork 是纯增量文件，revert 单 commit；argus 侧 env 三组全不设 = 行为与 `45e0bc9` 一致；新端点无存量消费方。生产回滚 = bundle 切回 argus-0.1.0-c9ef409。

## 13. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-22 | 初版（批 D1 后文本族二期：reranking + 模型对 env 组合） |
