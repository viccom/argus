# AGENTS.md — argus

工业视觉智能 HTTP 服务：Rust + axum 封装 rust-onnx-infer，把 `CAPABILITY_KEYS` 所列能力暴露为 REST API。
本文件是仓库入口地图（非手册）；端点/环境变量/模型来源的唯一权威源是 **README.md**，路线图与进度在 docs/ 两计划表。

## 常用命令

```bash
cargo test --jobs 6                   # 全量测试。不依赖模型文件，也无需 ORT_DYLIB_PATH（路由测试用 Config::default()，引擎从不加载）
cargo clippy --all-targets --jobs 6   # lint 门禁。Cargo.toml 已设 clippy all+perf=deny，任何警告即失败
cargo build --release --jobs 6        # 生产构建（LTO/strip）。运行需 ORT_DYLIB_PATH + models/ 模型文件
```

E2E 冒烟需运行中的实例与真实模型；模型不入库（.gitignore 含 /models /testdata .env* /deploy/lib），来源与踩坑判别法见 README 模型表 + docs/e2e-and-models.md。

部署走版本化 bundle（`argus-0.1.0-<sha>/`：二进制 + run.sh + argus.env，models/lib 用软链）。约定：run.sh 以自身所在目录为 cwd、env 内路径一律相对（`./lib` `./models`）、build.rs 监听 git refs（提交后直接构建版本串即更新）——发版只换 bundle 目录，不改任何字面路径。细节见 README「部署布局约定」。

## 架构地图

| 位置 | 职责 |
|---|---|
| src/config.rs | env 驱动配置；`CAPABILITY_KEYS` 为能力键唯一权威源（/models、EngineManager、白名单共用） |
| src/engine.rs | EngineManager（tokio::OnceCell 懒加载）+ EnginePool（信号量准入 + round-robin）+ 预热/深检 |
| src/routes/*.rs | 每能力一个路由文件；公共工具在 routes/common.rs；检测族共享 detection.rs::run_detect |
| src/error.rs / auth.rs / main.rs / build.rs | 错误信封 / API key 中间件（常数时间比较）/ 优雅停机 / git hash 版本注入 |
| web/index.html · deploy/docker · tests/lifecycle.rs | 内嵌控制台 / GPU 镜像 / 停机超时回归 |

## 硬纪律（每条对应代码锚点，违反即回归）

1. **CPU 密集必须 `spawn_blocking`**：图像解码（common.rs::decode_image）与推理（EnginePool::run）都不许占 async worker——大图解码百毫秒级，会拖死 /health 等轻端点。
2. **错误信封**：逻辑错误一律 HTTP 200 + `{"status":"error","data":{"error":...}}`（error.rs）。失败日志必须落在 `AppError::into_response`（TraceLayer 按状态码分级失效，对 200 信封不可见）；级别：BadRequest→debug、Overload→warn、其余→error。
3. **运行时参数只做后过滤，不改引擎**：引擎经 Arc 共享不可变；每请求 confidence 用"构造期低阈值 + HTTP 层过滤"（类内 NMS 保序，数学等价）。每请求 iou 明确不做。跨模块不变量用 `const _: () = assert!` 编译期锁定（engine.rs 顶部现例）。
4. **校验先于副作用**：参数校验/图像解码必须在触发引擎懒加载之前（handler 顺序 parse→decode→engine→permit→run；scene.rs 有测试锁定"无 file 报参数错而非模型未配置"）。白名单拒绝同样先于文件系统（EngineManager::ensure_enabled，不读路径不建 session）。
5. **命名单一权威源**：能力键以 `CAPABILITY_KEYS` 为准；按能力覆盖的 env 用能力键大写后缀，模型路径 env 沿用历史缩写（FACE_DET 等）——不引入第三套命名。
6. **multipart 未知字段必须消费**（读掉保持流一致），未知不等于可跳过字节。
7. **阈值下界比较在 f32 精度下进行**：floor 提升 f64 后 `0.10` 字面量会被判越界（detect_filter.rs 回归测试 `filter_should_accept_user_literal_at_floor`）。

## 新能力接入：三道门（进度事实源 docs/engine-exposure-plan.md）

1. **模型前置门**（写任何接入代码前）：magic bytes → shape dump → python onnxruntime 直跑看值域（防非标布局/归一化坐标坑）；
2. **TDD 门**：错误路径（缺 file/坏图/未配置/未启用）+ DTO 序列化测试先红后绿，不依赖模型文件；
3. **E2E 门**：curl 真实推理，坐标与 python 参照交叉验证 + /ui 卡片渲染。

路由范式照抄 routes/classification.rs（DTO+From / parse_*_multipart / handler / oneshot 测试组）；同族检测能力加 `DetectCap` 枚举复用 run_detect，不复制实现。

## 测试约定

- 路由测试：`build_router(test_state(), None)` + tower `ServiceExt::oneshot`，multipart 手工构造 1×1 PNG；
- 集成：tests/lifecycle.rs 守护"停机超时只从收到信号起算"（曾误写为从启动起算导致服务周期性自杀）；
- 完成定义：cargo test 全绿 + clippy 零警告 + 新端点 E2E 冒烟，三绿缺一不可提交。

## rust-onnx-infer 依赖流（git 依赖，Cargo.lock 锁 commit）

本地 `../rust-onnx-infer`（origin=viccom fork / upstream=javpower）改码 → push origin → 本仓库 `cargo update -p rust-onnx-infer`。fork 相对 upstream 的补丁（api-26、resize Convolution 抗混叠）及动机见 README 依赖说明，勿在 argus 侧绕过。

## 提交与文档纪律

- conventional commit 前缀 + 中文描述，批次任务号挂尾（如 `feat(scene): …（任务 22）`）；
- 注释/文档中文；实测数字必须带时点（如"2026-09-11 实测"），不写会腐烂的裸计数；
- 能力进度只回填 docs/ 三计划表（engine-exposure-plan / yolo-rs-absorption-plan / homepix-integration-plan），完成即附状态+验证证据+日期，不等记忆。
- 新文档按 `docs/conventions/documentation.md`：设计文档入 `docs/designs/YYYY-MM-DD-<topic>-design.md`，实施计划入 `docs/plans/`，并在 `docs/README.md` 索引登记。
