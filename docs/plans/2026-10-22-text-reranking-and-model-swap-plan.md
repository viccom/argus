# 实施计划：文本 reranking + 同家族模型对 env 组合

> 状态：approved（待执行；TDD 先红后绿，完成定义 = 双仓 test 全绿 + clippy 0 + 双对 E2E + golden 同步）
> 日期：2026-10-22
> 基线：argus @ `45e0bc9`；rust-onnx-infer @ `49b8585`
> 权威源：设计 [designs/2026-10-22-text-reranking-and-model-swap-design.md](../designs/2026-10-22-text-reranking-and-model-swap-design.md)
> 执行域：fork 仓库（F0）→ argus 仓库（F1-F4）。fork 构建注意 `OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include` env 旁路（.cargo/config.toml 的 macOS 约定在 Linux 挡 openssl-sys）

## 任务跟踪清单（执行时回填，不预填）

| ID | 阶段 | 任务 | 验收标准 | 状态 | 证据 |
|---|---|---|---|---|---|
| T0 | F0 | fork 基线全绿确认（cargo test + clippy，带 openssl env） | 双绿 | ⬜ | |
| T1 | F0 | runner：`encode_pairs` 自由函数 + `OUT_LOGITS` 常量（无模型单测：pair 模板 `<s>q</s></s>d</s>` / 批内最长 padding / 截断 512） | 先红后绿 | ⬜ | |
| T2 | F0 | `reranking.rs`：`TextRerankingEngine{runner}::with_options`（构造期 logits 末维==1 校验）+ sigmoid 纯函数 + Send/Sync const 断言；mod.rs 挂载；纯数学单测 | 先红后绿 | ⬜ | |
| T3 | F0 | 真模型门控自验（MODEL_DIR 约定 [SKIP] 模式）：jina int8 与 bge fp32 各 3 断言（正对>负对 / 两输入不喂 token_type / 输出形状 [B,1]）；fork README 模型表补 reranker 两行；push fork + argus `cargo update -p rust-onnx-infer` + argus 双绿零回归 | 自验过 + argus 全绿 | ⬜ | |
| T4 | F1 | argus config：17 键 + `text_reranking_model/tokenizer` 两件套 + `ARGUS_TEXT_EMBEDDING_POOLING/MAXLEN/PIPELINE_覆盖` 与 `ARGUS_TEXT_RERANKING_PIPELINE` 解析（非法值 fail loud）+ configured 判定（reranking 两件套齐） | config 测试先红后绿 | ⬜ | |
| T5 | F1 | engine：`text_reranking` loader（maxlen 512）/池/warmup（`rerank("预热",["预热文档"])`）/深检分支/PIPELINE_TABLE 两行 + env 覆盖读取；text_embedding loader 接 POOLING/MAXLEN env | loader 测试过 + 既有全绿 | ⬜ | |
| T6 | F1 | routes/rerank.rs：DTO + handler（校验链照 embed.rs：query 1..2000 字符、texts 1..64、错误无原文）+ health.rs 路由注册 + service_info 行 + oneshot 错误矩阵 ≥10 例 | 先红后绿 | ⬜ | |
| T7 | F2 | models.rs 17 项（reranking 无 dim）+ golden/models.json 补未配置项 + models 内联 golden 16→17 | 全量绿 + golden 同步 | ⬜ | |
| T8 | F2 | README（/rerank 端点行 + env 表 + 模型表 reranker 两行 + 两对模型预设小节）+ model-inventory #22/#23 状态 + homepix 计划表批 D 段 D2 + 本表/设计状态流转 + docs/README.md 索引 | 文档齐、索引登记 | ⬜ | |
| T9 | F3 | E2E 双对（临时实例真模型）：jina 对（现状 env + reranker int8：判别/确定性/错误矩阵/pipeline 串）+ bge 对（`POOLING=cls DIM=512` + bge reranker：/embed/text dim=512 + /rerank 过 + **pipeline 串为 bge**）+ 默认 env 实例 /models 与基线逐字节对比（除新增项） | 设计 §11 验收 1-4 全过 | ⬜ | |
| T10 | F4 | 生产 bundle 部署 jina 对（int8 reranker 软链 + 两件套 env）+ 切 systemd + 生产验收（homepix 验证脚本 V2-V6 复跑 + /rerank 冒烟）+ model-inventory 状态回填 | 生产全过 | ⬜ | |

## 完成定义

T1-T8 先红后绿 + 全量双绿 + 双对 E2E（T9）+ 生产部署验收（T10），缺一不可。过程纪律：fork 改动独立 commit + push；argus 改动 conventional commit；实测数字带时点回填本表；**状态列只在验证证据落地后回填，不预填**。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-22 | 初版（T0-T10 任务清单，待执行） |
