# EngineManager 动态化设计（检测族命名槽 + 热重载）

> 状态：approved
> 日期：2026-10-21
> 基线：argus `5b07e63`（2026-10-21）
> 权威源：`docs/engine-exposure-plan.md`（批 1.6 场景槽）、`docs/yolo-rs-absorption-plan.md`（阶段 2）
> 实施计划：待 writing-plans 产出 `docs/plans/2026-10-21-engine-manager-dynamic-slots-plan.md`

## 1. 背景与目标

### 1.1 问题

检测族新模型目前必须改代码发版：`EngineManager` 把 detection/ppe/fire_smoke 写成三个强类型 `OnceCell` 字段（`src/engine.rs:140-160`），模型路径来自 env。直接后果：

- `engine-exposure-plan.md` 批 1.6 的 6 个场景槽（YOLOE/crack/barcode/消防器材/行为违章/YOLO26 换代）全部被阻塞；
- PPE 自训（SH17 类别体系）换模型只能改 env + 重启，无法按实验并行多版本；
- 吸收计划的“目录即用/动态模型管理”（阶段 2）一直是唯一未完成的重构级项。

### 1.2 目标

1. 检测族新增模型从“改代码 + 发版”退化为“`slots.toml` 加一段 + 热重载”；
2. 三个检测族端点（`/detect`、`/ppe`、`/fire-smoke`）与 `/detect/batch` 支持可选 `model` 参数选择动态槽；
3. 静态 3 槽行为、契约、测试全部零回归；
4. 支持自定义类名（PPE 自训 SH17）、按槽池大小/预热尺寸；
5. 坏模型、配置错误、热重载与在途请求并发都有明确处理。

### 1.3 非目标（YAGNI 边界）

- 非检测族 12 能力（face/pose/depth/OCR…）不迁入动态管理；
- 不做空闲淘汰（`docs/lazy-loading-optimization.md` 阶段 2 的前置门未过，另行决策）；
- 不做手动 `POST /models/{slot}/unload` 端点；
- 不做每请求 confidence/iou 覆盖（沿用“构造期低阈值 + HTTP 后过滤”纪律，`AGENTS.md:32`）；
- 不做 `/ui` 槽选择器；
- 不监听模型文件本身的 mtime（见 §12 未决项）；
- 不做每槽独立 `/metrics` 计数（全局 B5 计数继续有效）。

## 2. 代码现状（基线 `5b07e63`）

| 事实 | 证据 |
|---|---|
| 15 个强类型 `OnceCell` 字段，对应 13 种引擎类型；检测族 3 槽共用 `DetectionEngine` | `src/engine.rs:140-160` |
| 池结构：`instances + Semaphore(池大小) + round-robin`；`run()` 在 permit 内 `spawn_blocking` | `src/engine.rs:730-748` |
| SAHI 需要具体类型：`predict_sliced_detection(engine: &dyn OnnxInferenceEngine<Output=Vec<Detection>>, single: &dyn SahiSinglePredictor<Detection>, ...)` | rust-onnx-infer `src/sahi/sliced_predictor.rs:39-45`；调用点 `src/routes/detection.rs:207-218` |
| 检测 DTO 的类名来自引擎 `get_label_name(class_id)`；上游 `BaseOnnxEngine::set_labels` 为 pub | rust-onnx-infer `src/core/base.rs:458`、`:287`；`src/routes/detection.rs:29-37` |
| `DetectCap` 三值枚举 + `key()` 提供能力键；`run_detect` 用 `cap.key()` 取溯源 | `src/routes/detection.rs:163-187`、`:252` |
| `/detect/batch` 已有 `model: Option<String>` 字段，但传入即拒（bad_request 错误信封）；溯源硬编码 `"detection"` | `src/routes/batch.rs:27-36`、`:60-64`、`:100` |
| `/models` 固定 15 项 `capabilities`；`loaded` 是单布尔 | `src/routes/models.rs:58`、`:190-203` |
| `/health` 的 `engines` 与 `engine_health[].name` 都是 `&'static str` | `src/routes/health.rs:57-90`；`src/engine.rs:94-100` |
| `EngineManager` 持有 `Config`（Clone）与 `DeviceType`；`AppState` 已 `#[derive(Clone)]` | `src/engine.rs:162-185`；`src/routes/health.rs:14-24` |
| 启动装配与优雅停机在 `main.rs`；可 clone `AppState` 后 spawn 后台任务 | `src/main.rs:44-89` |
| 能力键与按能力配置查询已就位 | `src/config.rs:8-12`、`:244-259` |

## 3. 决策与备选

### 3.1 范围：只做检测族（方案 A）

- 检测族 6 个待做场景槽 + PPE 自训 + YOLOE 全部是 `DetectionEngine` 同一输出类型，零类型擦除；
- 非检测族短期无多模型需求，迁入只会把回归面从 3 槽扩大到 15 能力。

**否决 B（能力槽 trait 全量）**：3-4 天、15 能力全量特征测试，收益不在近期。
**否决 C（全量 enum/Any 擦除）**：13 种引擎类型、match 分支扩散，过度设计。

### 3.2 注册方式：显式清单 `slots.toml`

**否决“目录即用”**：任何能写目录的进程都能改变 API 面；坏模型只在请求时暴露；与“拒绝非标模型”的既定纪律相悖（`docs/yolo-rs-absorption-plan.md` §三）。

### 3.3 路由：三端点 + batch 的可选 `model`

复用 `run_detect` 单一实现；`model` 缺省走原静态槽路径，行为逐字节不变。`model` 只接受**动态槽名**；静态能力键（15 个）保留，传静态名返回错误信封（HTTP 200 + status=error，`code="bad_request"`，提示用对应端点），避免跨端点语义混用。所有客户端错误的传输形态与既有契约一致（§5.2/§6：统一信封，不引入真 4xx）。

### 3.4 生效：watch 热重载

3s 轮询 `slots.toml` 的 mtime+size；解析成功才 diff 应用，失败保留 last-known-good。与 vl-gateway 的配置热重载纪律一致。

### 3.5 关键修正（相对初版设计）

| # | 初版 | 修正 | 原因 |
|---|---|---|---|
| C1 | 预置 3 槽迁入 registry | registry 纯增量，静态 3 槽保持 OnceCell | 消除最大回归面 |
| C2 | 动态槽进 `capabilities.detection.models[]` | 新增 `data.dynamic_slots` | 不改 safebot-go 读取的语义 |
| C3 | 动态槽进 `engine_health` | 新增 `dynamic_slot_health` | 避免 `&'static str` 类型扩散 |
| C4 | “路由层零改动” | 静态路径零改动，动态路径新增 | `cap.key()`/batch 溯源硬编码需按槽名解析 |
| C5 | 仅 `/detect` 加 model | 三端点 + batch 都加 | PPE 自训的真实入口是 `/ppe` |
| C6 | 仅构建信号量 | 信号量 + 30s 失败退避 | `get_or_try_init` 不缓存错误 |
| C7 | “路径校验照抄 yolo-rs” | 槽名正则 + 保留字 + 模型路径必须在 models_dir 内 | 槽名来自 URL，是外部输入 |
| C8 | 只监听 toml | v1 文档约定原子替换 + touch；模型 mtime 监听列后续 | 同路径原地换文件不触发 |

## 4. 架构与数据流

### 4.1 核心结构

```rust
struct SlotSpec {
    name: String,                       // ^[a-z][a-z0-9_]{0,31}$，不得是 CAPABILITY_KEYS
    model_path: PathBuf,                // canonicalize 后必须位于 ARGUS_MODELS_DIR 内
    class_names: Option<Vec<String>>,   // 可选；覆盖模型 metadata 的 labels
    pool_size: Option<usize>,           // 可选；默认 config.engine_pool，范围 1..=64
    warmup_size: Option<(u32, u32)>,    // 可选；默认 detection 的 warmup 配置
}

struct Slot<T> {
    spec: SlotSpec,
    pool: tokio::sync::OnceCell<Arc<EnginePool<T>>>,
    model_info: tokio::sync::OnceCell<ModelInfo>,   // 文件名#sha256 前 12 + pipeline
    last_error: std::sync::Mutex<Option<(String, std::time::Instant)>>,
    next_retry_at_ms: std::sync::atomic::AtomicU64, // 失败退避
}

struct SlotRegistry<T> {
    slots: std::sync::RwLock<HashMap<String, Arc<Slot<T>>>>,
    build_sem: Arc<tokio::sync::Semaphore>,          // 默认 1
    build_fn: Arc<dyn Fn(&SlotSpec, DeviceType) -> Result<EnginePool<T>, AppError> + Send + Sync>,
}
```

- **泛型 `T`**：生产 `SlotRegistry<DetectionEngine>`；单测用 `T=String` 注入假 builder，真实覆盖并发/退避而不依赖模型文件。
- **`Arc<Slot>` 整体替换**：热重载换槽 = registry 写入新的 `Arc<Slot>`；在途请求持旧 `Arc` 完成，旧池由其最后一个引用释放。无需每槽 `RwLock<Option<Arc>>`。
- **静态 3 槽不进 registry**：`EngineManager` 现有字段与方法不动；registry 只存 `slots.toml` 声明的动态槽。

### 4.2 请求解析

```
model 参数缺省
  └→ 原路径：cap.pool(&state) + provenance(cap.key())，零改动

model=动态槽名
  └→ registry.resolve(name)：
       read 取 Arc<Slot>（快路径，不持锁构建）
       pool 已初始化 → 直接用
       未初始化 → 检查退避窗口 → 持 build_sem → spawn_blocking 构建
                 → OnceCell 初始化 → 释放信号量
       → 槽级 provenance：ModelInfo { model: "文件名#hash12", pipeline: "detect-slot-v1" }

model=保留字（15 能力键）
  └→ 错误信封（HTTP 200 + status=error），data.code="bad_request"，data.error 提示改用对应端点

model=未注册槽名
  └→ 错误信封（HTTP 200 + status=error），data.code="unknown_model"，data.error 列出可用动态槽名（不泄露路径）
```

`/detect/batch` 在请求入口解析一次槽名，逐图复用同一池与同一溯源；单图失败仍按现有语义隔离，不因动态槽改变批处理错误模型。

构建参数：`DetectionEngine::for_yolo(path, device, DETECTION_CONFIDENCE_FLOOR, DETECTION_NMS_IOU)`；`pool_size` 个实例；构造后对每个实例 `base.set_labels(class_names)`（若提供）；预热沿用 `warmup_engine`。

### 4.3 热重载

```
每 ARGUS_SLOTS_WATCH_SECS（默认 3，0=关闭）轮询 slots.toml 的 mtime+size
  ├─ 文件不存在          → 动态槽清空（静态不受影响）
  ├─ 解析/校验失败        → 保留 last-known-good + error 日志（不部分应用）
  └─ 解析成功 → 按槽名 diff：
       新增 → 插入未加载 Slot
       删除 → 移除 Arc（在途持旧引用完成）
       变更（路径/类名/池/预热）→ 整体替换为未加载 Slot（下次请求按新配置懒加载）
```

启动时只解析与校验，不构建引擎；`slots.toml` 缺失等价于“无动态槽”（零回归）。

### 4.4 配置格式

```toml
# 默认路径：$ARGUS_MODELS_DIR/slots.toml；可用 ARGUS_SLOTS_CONFIG 指向仓库内文件
[slots.crack]
model = "yolov8n-crack.onnx"          # 相对 ARGUS_MODELS_DIR；绝对路径也必须落在该目录内
class_names = ["crack", "pothole"]     # 可选
pool_size = 1                          # 可选
warmup_size = [640, 640]               # 可选

[slots.ppe_sh17]
model = "ppe_sh17_v3.onnx"
# 示例（按实际自训模型的类别清单填写，不要求与 SH17 官方命名一一对应）
class_names = ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]
```

校验规则（启动 fail-closed；热重载失败保留旧配置）：

- 槽名：`^[a-z][a-z0-9_]{0,31}$`，且不得是 `CAPABILITY_KEYS` 中的 15 个能力键（保留字）；
- `model`：非空；解析/规范化后必须位于 `ARGUS_MODELS_DIR` 内（含符号链接解析）；
- `pool_size` ∈ [1, 64]；`warmup_size` 两维均 ∈ [32, 4096]；
- `class_names`：非空字符串数组；与模型 metadata 的 labels 数量不一致时 warn，不阻断（自训模型 metadata 可能缺失）；
- 未知字段：warn 并忽略（向前兼容）；重复槽名：解析失败。

### 4.5 观测与溯源

- 静态路径：`data.model`/`data.pipeline` 与现有完全一致；
- 动态槽：`data.model = "<文件名>#<sha256 前 12>"`，`data.pipeline = "detect-slot-v1"`；`model_info` 每槽缓存，热重载换槽时随之重建；
- `/models` 增 `data.dynamic_slots.{name}.{configured,loaded,model_file,pipeline}`；15 项 `capabilities` 逐字节不变；
- `/health` 浅检的 `engines` 保持静态；deep 增 `dynamic_engines` 与 `dynamic_slot_health`（`{name,status,error}`）；现有 `engine_health` 不变；
- B5 背压计数保持全局键（`in_flight/queued_total/rejected_total/acquire_timeout_total`），v1 不做每槽计数。

### 4.6 与既有配置的关系

- `ARGUS_ENABLED_CAPABILITIES` 仍按能力键过滤：白名单不含 `detection` 时，所有动态检测槽拒绝（`CapabilityDisabled`），与静态 detection 一致；
- `ARGUS_ENGINE_POOL` / `ARGUS_<CAP>_POOL` 继续作用于静态槽；动态槽默认取 `ARGUS_ENGINE_POOL`，槽级 `pool_size` 覆盖；
- 模型路径 env（`ARGUS_DETECTION_MODEL` 等）含义不变；
- `slots.toml` 是部署数据，默认落在 gitignore 的 `models/` 下；需要入库的配置用 `ARGUS_SLOTS_CONFIG` 指向仓库内路径。

## 5. 契约与兼容性

### 5.1 请求（全部为可选字段，缺省行为不变）

| 端点 | 新增字段 | 缺省 |
|---|---|---|
| `POST /detect` | multipart 文本 `model` | 静态 detection 槽 |
| `POST /ppe` | multipart 文本 `model` | 静态 ppe 槽 |
| `POST /fire-smoke` | multipart 文本 `model` | 静态 fire_smoke 槽 |
| `POST /detect/batch` | JSON `model` | 静态 detection 槽（原“传入即拒”解除） |

### 5.2 响应（只加不改）

- 推理响应：新增 `data.code="unknown_model"`（错误场景）；动态槽的 `data.model/pipeline` 取值见 §4.5；
- `/models`：新增 `data.dynamic_slots`；
- `/health?deep=true`：新增 `dynamic_engines`、`dynamic_slot_health`；
- 现有字段、信封形状、状态码语义不动。

### 5.3 消费方影响

- safebot-go：只读 `capabilities.detection.models` 与推理字段；新增字段不影响；`/detect` 无 `model` 时路径不变；
- homepix：`internal/argus/client.go` 只看 `data.error`；新增字段不打断；
- 门禁：F0 先抓 `/models`、`/health`、`/detect`（同一图）的解析后 JSON 作为 golden，F4 复比。

## 6. 错误处理与安全边界

| 场景 | 行为 |
|---|---|
| `slots.toml` 缺失 | 动态槽为空，服务正常（零回归） |
| 启动时解析/校验失败 | 拒绝启动（fail-closed），日志含槽名与原因 |
| 运行中重载失败 | 保留 last-known-good，error 日志，不部分应用 |
| 槽名非法/保留字/路径越界 | 启动拒绝；重载拒绝整份文件 |
| `model` 未注册 | HTTP 200 统一信封 + `status=error`，`data.code="unknown_model"`，`data.error` 列可用动态槽名 |
| `model` 是保留字（能力键） | HTTP 200 统一信封 + `status=error`，`data.code="bad_request"`，`data.error` 提示改用对应端点 |
| 模型文件缺失/损坏 | 首次请求构建失败，返回既有错误族（`ModelNotConfigured`/`Internal`），槽保持未加载；30s 内不重试，避免请求风暴 |
| 白名单不含 detection | `CapabilityDisabled`（与静态 detection 一致） |
| 热重载与在途请求 | 旧 `Arc<Slot>` 继续服务至请求结束；新请求用新槽 |
| 类名数量与模型不符 | warn 后仍注入；不阻断 |

安全边界：槽名来自 URL，必须正则+保留字校验；模型路径来自 toml，规范化后必须位于 `ARGUS_MODELS_DIR`（符号链接按真实路径判定）；错误信息不返回绝对路径。

## 7. 风险登记

| 级别 | 风险 | 缓解 | 验证 |
|---|---|---|---|
| P0 | 静态 3 槽回归 | registry 纯增量；`/detect` 无 model 走原路径 | F0 golden；F4 逐字段复比 |
| P0 | 热重载与在途请求竞态 | `Arc<Slot>` 整体替换 | 单测：构建中 reload，旧请求完成、新请求用新池 |
| P0 | 槽名/路径注入 | 正则+保留字+models_dir 约束 | 单测：`..`、绝对越界、保留字、非法名 |
| P1 | 坏模型请求风暴 | 30s 失败退避 + 构建信号量 | 单测：连续 N 请求只尝试 1 次构建 |
| P1 | 双实例内存翻倍 | 懒加载、替换不预载 | E2E RSS：reload 后仅新槽常驻；文档提示 7042/7043 各自加载 |
| P1 | 契约漂移 | 只加字段；现有字段不动 | F0/F4 golden；消费方 grep |
| P1 | watcher 生命周期 | watch 通道随优雅停机退出 | lifecycle 测试：SIGTERM 后无残留任务 |
| P2 | 同路径原地换模型不触发 | 文档约定原子替换 + touch `slots.toml` | 文档评审 |
| P2 | class_names 长度不符 | warn 不阻断 | 单测 |
| P2 | `slots.toml` 半写入 | 建议原子写（临时文件+rename）；解析失败保留旧配置 | 单测：截断文件 → 保留旧配置 |

## 8. 测试策略

1. **F0 特征测试（重构前）**：锁定静态 3 槽的懒加载、未配置错误、白名单先于文件系统、`pool_for` 回退、溯源缓存、`/models` 15 项、`/health` 形状；抓 golden。2. **F1 Registry 单测（`T=String` 假 builder）**：toml 解析/校验、保留字、重复槽、路径越界、add/remove/change diff、并发只建一份、失败退避、last-known-good。
3. **F2 路由测试**：`model` 缺省无变化；未注册（unknown_model）与保留字（bad_request）错误信封分流；动态槽推理路径（单元层可 mock 池）；provenance 按槽解析；batch `model` 不再拒绝。
4. **F3 watcher**：mtime+size 变更检测；新增/删除/变更/非法/截断；停机退出。
5. **F4 E2E（真实模型）**：用现有 `yolov8n.onnx` 注册别名槽 `alias-test`，走“加载→/detect?model=alias-test→溯源→touch slots.toml 改路径→热重载→删除→unknown_model 错误信封”；坏模型退避；静态端点与 golden 逐字段一致。
6. **门禁**：`cargo test --jobs 6` 全绿 + `cargo clippy --all-targets --jobs 6` 零警告（共享机纪律）。

## 9. 实施阶段与门禁

| 阶段 | 内容 | 门禁 | 估时 |
|---|---|---|---|
| F0 | 特征测试 + golden 基线 | 现有测试基线全绿；golden 入库 | 0.5d |
| F1 | `SlotRegistry<T>` 核心（解析/校验/diff/懒加载/信号量/退避） | 单测先红后绿；clippy 零警告 | 0.5d |
| F2 | 集成：EngineManager 字段、三端点+batch `model`、动态溯源、class_names | 路由测试；静态路径零变化 | 0.5d |
| F3 | watcher（mtime+size、last-known-good、三态 diff、停机退出） | watcher 单测 + lifecycle | 0.5d |
| F4 | E2E + 文档回填（README/AGENTS/三计划表 + `slots.example.toml`） | 全量测试 + clippy + golden 复比 + E2E | 0.5d |

降风险切分：F0→F2 即可交付“重启生效”版本；F3 热重载单独追加，代码结构不变。

## 10. 验收标准

1. `/detect`、`/ppe`、`/fire-smoke`、`/detect/batch` 不传 `model` 时，输出与 F0 golden 逐字段一致（排除推理耗时）；
2. `slots.toml` 新增槽后 ≤ 2 个轮询周期（默认 6s）内 `/detect?model=<name>` 可用，无需重启；
3. 删除槽后同一端点返回 `unknown_model` 错误信封（HTTP 200 + status=error），静态端点不受影响；
4. 同一槽的 N 个并发首请求只触发一次构建；
5. 坏模型连续请求只尝试一次构建/30s，服务不崩溃；
6. 槽名 `../x`、绝对路径越界、保留字、截断 toml 全部被拒绝且保留旧配置；
7. `/models` 的 15 项 `capabilities` 与现有 golden 完全一致；新增 `dynamic_slots` 正确反映 loaded；
8. `/health?deep=true` 的 `engines`/`engine_health` 与现有 golden 一致；新增字段正确；
9. SIGTERM 后 watcher 退出、无残留任务，优雅停机链不变；
10. `cargo test --jobs 6` 全绿、clippy 零警告。

## 11. 回滚

- 删除/重命名 `slots.toml` → 动态槽清空，静态服务不受影响；
- 回滚二进制到 `5b07e63` 及以前：新字段不被旧代码读取，`slots.toml` 被忽略；
- 无数据库/schema/迁移变更，无数据回滚。

## 12. 未决项与后续

| 项 | 决策 |
|---|---|
| 同路径原地替换模型文件 | v1 不监听；文档约定原子替换 + `touch slots.toml`。后续可加“槽模型文件 mtime”监听 |
| `/ui` 槽选择器 | 不做；如需要，后续在 multipart 表单加同名字段 |
| 非检测族动态化 | 后续按需逐族迁移，接口保持 `SlotRegistry<T>` 泛型形态 |
| 空闲淘汰 | 与懒加载阶段 2 合并，前置门（RSS 回落）未过前不做 |
| 每槽背压/延迟指标 | v1 全局计数；后续按槽扩展 B5 字段 |
| 默认 `/ppe` 模型热替换 | 通过 `/ppe?model=ppe_sh17` 使用新槽；把默认槽指向新模型仍需改 env + 重启（热重载不覆盖静态 env 槽） |
| 模型文件 mtime 监听 | 后续项，需处理大文件写入中的半成品（建议原子 rename） |

## 13. 变更记录

- 2026-10-21：初版设计获批。范围=检测族；注册=显式 `slots.toml`；路由=三端点+batch 可选 `model`；生效=watch 热重载。相对初稿采纳 8 项复核修正（C1-C8，见 §3.5）。
- 2026-10-21：执行期勘误（T4 代码审查发现）：§2/§3.3/§4.2/§5.1/§8/§10 原文零散写的"400"与 §5.2/§6 的"HTTP 200 统一信封"矛盾——统一修正为错误信封表述，以 §5.2/§6（与既有 `AppError` 契约一致）为准。无行为变化。
- 2026-10-21：执行期勘误（最终代码审查发现）："原子替换 + touch slots.toml 换模型"的表述对**已加载槽**有误导——touch 只触发重解析，规格未变的槽按 §4.3 diff 语义保留已加载池，仍用旧模型；touch 仅对未加载槽生效。README 与 slots.example.toml 已更正为"已加载槽换模型须改 spec（换路径）或重启"；模型文件 mtime 监听维持 §12 后续项定位。
