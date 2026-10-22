# Argus × homepix 集成改进计划 · 任务跟踪表

> 目的：把 Argus 从"工业安监服务"补齐成 homepix 家庭照片库的可靠视觉后端。
> 依据：2026-10-20 ur5900（8 线程 / 15GB / 无 GPU）实机部署与真实家庭照片探针，
> 数字与缺陷均为实测。原表由 homepix 侧 Coding Agent 成文，argus 侧逐条对照代码
> 复核后采纳（复核记录见变更记录），本表是该批改进的唯一进度事实源。

## 0. 全局约束（违反即返工）

1. **向后兼容优先**。生产已有消费方（safebot-go，/ppe /fire-smoke 等）。
   新增 multipart 字段一律可选，默认值复现改动前行为；新增响应字段可以，删改现有字段不行。
2. **门禁沿用仓库口径**：`cargo test --jobs 6` 全绿 + `cargo clippy --all-targets` 零警告；TDD 先红后绿。
3. 端点改动过 `engine-exposure-plan.md` 三道门的后两道（本批无新模型）。
4. **契约变更同步 homepix**：homepix `internal/argus/client.go` 只看 `data.error`、不写死维度——加字段不打断它，改语义会。

## 1. 可复现夹具（`/tmp/argus-test/`）

| 文件 | 内容 | 改动前实测 |
|---|---|---|
| `orient_raw.jpg` | 4032×3024，像素侧躺 | /face/detect → 2 脸 conf 0.72/0.64。**2026-10-20 复核：该文件实无 EXIF 段**（APP0 JFIF 后直达 DQT/SOF），无取向信号——任何解码器无法也不应旋转它；原表"orientation=6"描述与文件实态不符，推断制夹具时标签被剥离。真验证夹具为 `orient_exif6.jpg`（Pillow 自 orient_baked 逆时针转 90° + EXIF 274=6 构造），A1 修复后检出 38 脸与摆正基线一致 |
| `orient_baked.jpg` | 同一张摆正后 | 38 脸 conf 0.93 起；/face/recognize 38 向量 1475ms |
| `family3.jpg` | 三人合影 | 3 脸；recognize 53ms |
| `big15mp.jpg` | 海滩七人 | /detect 16 检出 |
| `cake_tight.jpg` | 蛋糕裁剪 | /classify → count=0（top-1<0.5 清空） |

相似度地真（homepix 逐张裁剪核对）：boy_id 为查询，同人 0.79/0.77/0.62，异人 0.47，其余 <0.42。B2 比选的参照。

## 2. 批 A —— P0

| ID | 任务 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|
| A1 | 解码时应用 EXIF Orientation（+C3 像素闸 `ARGUS_MAX_IMAGE_PIXELS` 顺同一 ImageReader 改造带入） | ✅ | `decode_image` 改走 ImageReader→into_decoder：orientation() 预读 + apply_orientation；像素闸在缓冲分配前经 dimensions() 预检。单测：8 取向对照（同 JPEG 字节基线规避有损差）+ 无 EXIF PNG 逐字节不变 + 像素闸边界/默认 0 不限 + config 解析。实现要点：NoTransforms 分支空操作=摄像头帧零开销；EXIF 损坏降级为无变换（debug 留痕）。C3 默认 0=不限（向后兼容），部署 env 设 6000 万 | 2026-10-20 |
| A2 | /classify 阈值可调 | ✅ | 引擎构造期 `set_confidence_threshold(0.0)`（license_plate 先例，不动 fork）；路由层 `apply_top1_gate` 按请求阈值复现历史怪癖：top-1 过线返回**全量** top-k 含低分尾巴、不过则空（4 单测锁定"top-1 门控 ≠ 逐项过滤"）；`confidence` ∈[0.0,1.0] 越界 400 | 2026-10-20 |
| A3 | 人脸门控 min_face_px/max_faces | ✅ | `gate_faces` 纯函数：短边过滤→置信度降序截前 N（稳定排序），默认全关零行为变更；detect 与 recognize 共用，recognize 的门控在逐脸 extract **之前**（成本就在每脸一次 rec 池推理）。6 门控单测（默认不重排/短边 80 边界含/截断/先滤后取） | 2026-10-20 |
| A4 | 响应回报 model/pipeline | ✅ | `EngineManager::model_info`：文件名#sha256 前 12 位（首查 spawn_blocking 后缓存；OCR/车牌双模型 + 拼接，半套配置 None）；PIPELINE_TABLE 15 能力管线标识，人脸识别/OCR/车牌串含前置检测模型（与 /models 同串防两处对照不一致）；15 个推理 handler 注入 data.model/pipeline；/models 每能力附 pipeline。5 单测（表覆盖/未配置 None/已知 sha/双模型/半套 None） | 2026-10-20 |
| A5 | 错误机器可读码（+C2 too_large 一并实现，避免枚举有码无源） | ✅ | AppError 增 code()/retryable()/TooLarge；信封 `data.code`+`data.retryable`，8 变体 8 码、仅 overload 可重试；body 超限经 axum MultipartError 413(LengthLimitError) 与解析错误区分（17 处映射集中 common::map_field_err），路由级测试 1MB 上限→too_large+提示 ARGUS_MAX_BODY_MB | 2026-10-20 |

## 3. 批 B —— P1

| ID | 任务 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|
| B1 | /face/recognize 补 landmarks + embed_dim | ✅ | FaceMatchDto.landmarks（5 点，与 /face/detect 同源同值）+ data.embed_dim；**crop=true 推迟到任务 32**（base64 图返回契约统一发明，避免两套形状）。DTO 单测 | 2026-10-20 |
| B2 | ArcFace/MobileFaceNet 可切换 | ⬜（条件启动） | 启动条件：A3+B1 落地后 SFace 余量仍不足（地真同人 0.62-0.79/异人 0.47 分界窄，先按 0.58 阈值带跑）。上游 `with_norm_mode(InsightFace)` 已支持 512 维，argus 侧改工厂+`ARGUS_FACE_REC_MODE` 半小时；模型文件获取需批准；换模型=历史向量全废（A4 已为其铺好失效判定） | |
| B3 | /face/recognize/batch | ✅ | 多 file ≤16/请求（超限 400 提示送派生图——与 20MB body 上限协同）；语义照抄 /detect/batch：逐图串行（**省 HTTP 往返非推理并行**，验收按此校准）、单图失败带 error 不炸整批、每图独立 permit、模型未配置=整批级错误；`recognize_faces` 共享函数消除单图/批量重复。3 路由测试 | 2026-10-20 |
| B4 | 暴露 ORT intra 线程数 | ⬜（等数据） | 需动 fork（runtime_config 无线程全局 setter，只有 GPU 显存有）≈15 行 + push + cargo update；对冲：B5 计数先行，压测数据决定值不值得 | |
| B5 | 过载可观测 | ✅ | InferencePermit RAII：in_flight 随 permit 存亡；acquire 打点 queued/rejected/acquire_timeout；全部推理响应加 queue_ms（与 inference_ms 分开）；/health 浅检增 backpressure 对象。4 单测（RAII 归零/fail-fast 只计 rejected/超时双计/浅检字段）。**C5 的 /metrics 是同一需求的 Prometheus 形态，后续在 B5 计数之上加文本暴露，别开两份** | 2026-10-20 |

## 4. 批 C —— P2

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| C1 | /models status "success"→"ok" | ⏸ | safebot-go 不在 argus 侧机器，字面量依赖需用户侧确认；homepix 客户端已确认不读 |
| C2 | 超限报对的错 | ✅ | 并入 A5 实现（见上） |
| C3 | 像素闸门 | ✅ | 并入 A1 实现（见上） |
| C4 | OpenAPI 契约文档 | ⬜ | README 表格已同步全部新字段；OpenAPI 全量文档插空做 |
| C5 | /metrics | ⬜ | 见 B5 备注，基于其计数加 Prometheus 文本暴露 |
| C6 | 不做 HEIC/AVIF | 🚫 | 明确不做：不引 libheif 破坏自包含单文件；homepix 送 JPEG 派生图 |

## 4b. 批 D —— 中文文本 embedding（homepix 新需求）

需求与验收口径：homepix 侧 `text-embedding-requirements.md`（2026-10-21 成文，未入库——硬条件以本表与 README 为准）。设计/计划：`docs/designs/2026-10-21-argus-text-embedding-endpoint-design.md` + `docs/plans/2026-10-21-argus-text-embedding-endpoint-plan.md`；引擎侧前置（fork `4b4615e`）见 `docs/plans/2026-10-21-rust-onnx-infer-text-embedding-plan.md`（implemented）。

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| D1 | `POST /embed/text`（批量文本 → 归一化向量） | ✅ | 2026-10-21 实施完成：第 16 能力键 `text_embedding`（三件套 env：MODEL/TOKENIZER/DIM，加载期 dim 校验 fail-loud）；逐条 `embed()` 循环（T8 实测快于批且严格跨批逐位确定）；空字符串/超限/mode 非法/normalize=false 全部 HTTP 前置校验→`bad_request`（fork 兜底会漏码成 inference_failed——已测试锁定）；错误码零新增。E2E 全 6 项过（:7099 真模型）：dim/模长/逐位确定/语义判别/overload retryable/鉴权一致。测试 258→282 全绿。**部署模型换型（2026-10-21 双语 A/B）**：bge-small-zh-v1.5（纯中文）→ **jina-embeddings-v2-base-zh**（中英双语，768 维 mean 池化，Apache-2.0）——英文判别间隔 0.43 vs 0.17、中文间隔 0.42 vs 0.30、p95 18.4ms（预算 30ms 内）；`pipeline=jina-embeddings-v2-base-zh\|mean-pool-norm-v1\|maxlen512`；fork 增两输入导出支持（`49b8585`）。**生产部署完成（2026-10-22）**：bundle argus-0.1.0-c9ef409（jina 模型软链入共享 models/，三件套 env DIM=768），:7042 验收验证脚本 V2-V6 全 24 项过（p95 20ms）+ detect/quality 旧能力回归过；model-inventory #19 已更新为生产在用 |
| D2 | `POST /rerank` 文本重排端点 + 同家族模型对 env 组合 | ✅ | 2026-10-22 实施：fork `1800743` + argus T4a-T7（17 键 `text_reranking`、`B×max_pair_len≤8192` 预算、POOLING/MAXLEN/PIPELINE env、两对模型预设见 README；测试 282→322 全绿；生产部署 T10 待做） |
| D2+ | 多语重排扩展 / 跨模态 | 🚫 | 需求 §8 原列"多语重排 / rerank / 跨模态"本期不做——其中 rerank 已作为 D2 实施（2026-10-22 二期设计批准，见上）；其余两项挂账；fork 引擎结构已留正交扩展位（`engines/text/runner.rs` 复用） |

## 5. 与既有计划的关系

- 本表与 `engine-exposure-plan.md`（能力广度）互不阻塞：那份扩新端点，这份补既有端点的契约可靠度。
- B1 的 crop=true 与该表任务 32（图像返回契约）合并执行。
- B4 若启动，走 rust-onnx-infer fork 流程（AGENTS.md 依赖说明）。

## 变更记录

- 2026-10-20（夜）：**部署后深度代码审查——5 项结论全部属实，1 项真 BUG 已修**（637a4f2）。核心发现：`/face/recognize`（单图+批量每图）的 `inference_ms` 把排队时间计入，违反本表 B5 行与 README 契约"queue_ms 与 inference_ms 分开"——`recognize_faces` 内取 permit，两处调用方却在外层起表；低负载 E2E（queue≈0）测不出。单 permit 竞争实测：修复前 queue_ms=426 / inference_ms=495（几乎相等，真实推理仅 ~70ms），同条件 /detect 对照 89/44 正常。修复：计时移入 permit 之后随返回值带出；新增回归测试 recognize_timing_should_exclude_queue_wait（占位 permit + 固定 400ms 排队窗），已验证旧代码上失败、新代码上通过——测试基线 209+1 → 210+1。同批清理：config.rs 文档注释错位归位、recognize_faces 内死计时、批量测试辅助 body/body2 双份构建残留、双空行。审查另 4 项（文档错位/死代码/测试残留/格式）均属清理级，无安全问题（gitleaks 扫 12 提交零泄漏）。
- 2026-10-20（夜）：**部署流程去硬编码（实发事故驱动）**（c4e44aa）。三处隐患：①run.sh 硬编码 cd 绝对路径——systemd unit 切新 bundle 后 run.sh 仍 cd 旧目录，实际运行旧二进制，靠 /proc/<pid>/exe 与版本串对不上才暴露；②argus.env 的 ORT_DYLIB_PATH/ARGUS_MODELS_DIR 指向两代前 bundle（9b5aba1）——旧 bundle 清理即断供；③build.rs 只监听 .git/HEAD——普通提交只改 refs/heads/<分支>，HEAD 文件内容不变，版本串停留旧 commit。修复：run.sh 改 `cd "$(dirname "$(readlink -f "$0")")"`；env 改相对路径 `./lib`/`./models`（软链随 bundle 走，字面量不随版本号变）；build.rs 经 `git rev-parse --git-dir` 解析绝对路径后监听 HEAD+refs/+packed-refs（提交后直接 cargo build 版本串即更新，已实测）。约定入 README「部署布局约定」。
- 2026-10-20（晚）：**E2E 全电池复验通过后部署生产**（ad-hoc :7099 debug 实例，逐项证据）：A1 决胜——orient_exif6 自动摆正 38 脸==摆正基线、无标签图回归不变；A2——cake_tight 默认 count=0 保持、confidence=0.05 得 5 条（top1 shower_curtain 0.461）、1.5 越界 bad_request；A3——min_face_px=80 得 14 脸全数短边≥81.2、max_faces=7 得 7 向量、不传参数 big15mp=10==基线；A5——25MB too_large+ARGUS_MAX_BODY_MB 提示、坏图 unsupported_image；B1——landmarks 5 点/脸、embed_dim=128（E2E 抓到单图响应 embed_dim 被 B5 重构误删，a4297b8 补回）；B3——3 图含 1 坏 → 2 成功 + 1 隔离错误；B5——backpressure 对象 + queue_ms 空闲为 0；**回归**——detect/face_detect/face_recognize 功能字段与生产基线逐位一致（仅 inference_ms 因 debug 构建慢 ~10 倍）。夹具发现见 §1（orient_raw 无 EXIF 段）。
- 2026-10-20：**批 A 全部 + B1/B3/B5 完成并部署**（A5+C2/A1+C3/A2/A3/A4/B1/B3/B5 共 8 项，测试基线 174+1 → 209+1 全绿，clippy 零警告）。原表（homepix 侧成文）经 argus 侧复核后采纳的修正：①A2 无需改上游（set_confidence_threshold 公开，license_plate 先例）；②C2 并入 A5（枚举有码无源问题）；③C3 并入 A1（同一 ImageReader 改造）；④B1 砍掉 crop=true 提前量（等任务 32 契约）；⑤B3 验收校准为"省往返非省推理"；⑥C3 默认 0=不限（向后兼容优先于文档建议的默认 6000 万，部署 env 显式设置）；⑦A1 复核销掉 image 0.25.10 API 可用性待确认项（apply_orientation/NoTransforms 空操作已核实）。
