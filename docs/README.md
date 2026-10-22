# argus 文档索引

> 维护规则见 [conventions/documentation.md](conventions/documentation.md)。新增、改名、废弃文档必须在此登记。
> 进度状态以各计划表文内状态列为准；本索引只做导航。

## 规范

| 文档 | 用途 | 状态 |
|---|---|---|
| [conventions/documentation.md](conventions/documentation.md) | 文档目录、命名、文档头、必备章节、索引与写作纪律 | 生效（2026-10-21） |

## 设计

| 文档 | 用途 | 状态 |
|---|---|---|
| [designs/2026-10-21-engine-manager-dynamic-slots-design.md](designs/2026-10-21-engine-manager-dynamic-slots-design.md) | 检测族命名槽 + 热重载：`slots.toml`、可选 `model` 参数、动态溯源、静态零回归 | approved（2026-10-21）；**实施完成 2026-10-21**（计划 T0-T13 全回填，258 测试 + clippy 0 + E2E 验收 1-10 全过，已合并 main） |
| [designs/2026-10-21-rust-onnx-infer-text-embedding-design.md](designs/2026-10-21-rust-onnx-infer-text-embedding-design.md) | fork 新增文本 Embedding 引擎（`engines/text/` 子模块，批 D1 前置）：runner/池化/确定性口径，纯增量零触碰既有引擎 | implemented（2026-10-21；fork `fdd7beb`+审查修正 `4b4615e` 已 push，计划 T0-T8 全销账） |
| [designs/2026-10-21-argus-text-embedding-endpoint-design.md](designs/2026-10-21-argus-text-embedding-endpoint-design.md) | argus 批 D1 `POST /embed/text`：16 能力键、三件套配置、逐条循环、错误码映射、golden 同步 | implemented（2026-10-21；计划 D0-D6 全销账，测试 258→282，E2E 6 项过；生产已部署 c9ef409 bundle，2026-10-22 验收过） |
| [designs/2026-10-22-text-reranking-and-model-swap-design.md](designs/2026-10-22-text-reranking-and-model-swap-design.md) | 文本 reranking 能力 + 同家族模型对（embedding × reranker）env 级组合：`text_reranking` 17 键、`POST /rerank`、池化/maxlen/PIPELINE env 化、jina/bge 两对预设 | approved（2026-10-22，已开工） |

## 计划与进度（权威进度源）

| 文档 | 用途 | 状态 |
|---|---|---|
| [plans/2026-10-21-engine-manager-dynamic-slots-plan.md](plans/2026-10-21-engine-manager-dynamic-slots-plan.md) | 动态槽实施计划（13 任务 TDD 清单，对齐设计 F0-F4） | 已完成（2026-10-21；T0-T13 全部回填，合并 main） |
| [plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md](plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md) | fork 文本 Embedding 引擎实施计划（T0-T8；T1 模型获取为批准门） | 已完成（2026-10-21；T0-T8 全销账，fork `fdd7beb` push + argus 零回归验证通过） |
| [plans/2026-10-21-argus-text-embedding-endpoint-plan.md](plans/2026-10-21-argus-text-embedding-endpoint-plan.md) | argus 批 D1 端点实施计划（D0-D6；E2E 验收 6 项） | 已完成（2026-10-21；D0-D6 全销账） |
| [plans/2026-10-21-homepix-embed-text-verification-plan.md](plans/2026-10-21-homepix-embed-text-verification-plan.md) | 批 D1 交付的 homepix 侧验证计划（V1-V8 用例 + 硬条件映射 + 一键脚本 + 自起实例附录） | approved（2026-10-21，交付 homepix Coding Agent 执行） |
| [plans/2026-10-22-text-reranking-and-model-swap-plan.md](plans/2026-10-22-text-reranking-and-model-swap-plan.md) | reranking + 模型对组合实施计划（T0-T10：fork 增量 → argus 能力/端点 → 双对 E2E → 生产部署） | approved（2026-10-22，待执行） |
| [engine-exposure-plan.md](engine-exposure-plan.md) | A 类引擎暴露任务跟踪表（批 1/1.5/1.6/2-6） | 活跃 |
| [yolo-rs-absorption-plan.md](yolo-rs-absorption-plan.md) | yolo-rs 能力吸收与退役方案（阶段 0-6） | 进行中（阶段 0/1/2/3/5 完成；其余见文内） |
| [homepix-integration-plan.md](homepix-integration-plan.md) | Argus × homepix 集成改进表（A/B/C/D 批） | 进行中（A 完成；D1 完成（含生产部署，2026-10-22）；B2/B4、C1/C4/C5 未完成） |
| [lazy-loading-optimization.md](lazy-loading-optimization.md) | 按需加载与内存优化方案 | 阶段 1 完成；阶段 2 挂账 |

## 参考

| 文档 | 用途 | 状态 |
|---|---|---|
| [model-inventory.md](model-inventory.md) | **实测模型清单**（现役/已验证/排除 26 项，含指纹、来源、实测数据、排除原因） | 权威（随实测回填） |
| [model-download-candidates-20261021.md](model-download-candidates-20261021.md) | 互联网模型下载调研与两批实测记录（HF/魔搭渠道、A/B 数据、教训） | 参考（2026-10-21） |
| [e2e-and-models.md](e2e-and-models.md) | E2E 冒烟与模型接入踩坑/判别法 | 参考（随新模型回填） |
| [model-ecosystem-research.md](model-ecosystem-research.md) | 通用模型生态调研与吸纳评估 | 调研（2026-09，落地前需重新比选） |
