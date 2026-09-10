# yolo-rs 核心能力吸收与退役方案

> 目的:将 yolo-rs 的核心能力(除对外 API 契约)吸收进 vision-ext + rust-onnx-infer,最终归档 yolo-rs。
> 前提决策(2026-09-10 会话确认):yolo-rs **从未与 safe-bot 实际对接**(A100 仅为测试环境),"契约锁定"护城河不存在;safe-bot 适配走"直接改造消费方"(独立 api-gateway 仅作备选)。
> 本文是唯一实施事实源;进度同步到 `engine-exposure-plan.md` 批次表。

## 一、yolo-rs 核心能力与实现细节调研(来源:全源码 19 文件 + deploy/docs 实测记录)

### 1.1 推理层(`src/inference/`)

**Session 池(`session.rs`,292 行)**
- 结构:每个模型 `pool_size` 个 `Arc<StdMutex<Session>>` 实例 + `AtomicUsize` round-robin 游标;
- 锁策略:`std::sync::Mutex`,加锁发生在 `spawn_blocking` 阻塞线程内(推理是 CPU 密集同步调用),async 线程只调度——避免锁内 await;
- 线程预算:**`pool_size × intra_threads ≈ CPU 核数`**;实测结论(注释原文):高并发吞吐优先时 pool 应大于 intra——**C=8 时 pool=8/intra=2 吞吐比 pool=2/intra=8 高约 2 倍**;
- GraphOptimizationLevel::Level3;`intra_threads=0` 表示用 ORT 默认;
- 张量按所有权传入 `spawn_blocking` 闭包(`'static` 要求)。

**CUDA/显存(`build_cuda_ep` + 部署实测,核心中的核心)**
- **arena 只增不减**:A100 实测单 yolov8n 每 Session 约 2.9GB(pool=4 饱和时 11.4GB);不设上限时曾与 vLLM 共存膨胀至 **81GB 打死服务**(cuBLAS OOM 后永久失败);
- 必设 `gpu_mem_limit_mb`(`CUDA::default().with_memory_limit(mb*1024*1024)`);`0=不设上限`仅向后兼容 CPU;
- arena 策略 SameAsRequested/Heuristic 与默认对该负载吞吐无差异,取最简;
- **cuDNN 小版本是决定性因素**:基础镜像 CUDA 12.8.1(cuDNN 9.8)~500 QPS vs 12.4.1(cuDNN 9.1)~256 QPS,**差约 2 倍**(二分实验证实,勿回退);
- ORT 动态库固定 **1.26.0**:官方 1.27+ 用 CUDA 13.0 构建,与 GPU 包 CUDA 12.8 基底不匹配;1.26 vs 1.29 CPU 性能无差异(CNN 负载不吃 1.27+ 的 LLM 优化);
- EP 列表 `[CUDA, CPU]` 带 CPU 回退。

**启动预热(`warmup_session`)**
- 对池内**每个** Session 跑零张量(`Array4::zeros`)dummy 推理:消除引擎反序列化 + cuDNN autotune 懒加载风暴,否则首个并发请求集体卡顿;
- 失败**告警不阻断**(标记冷启动,首请求延迟高可接受);
- `load()` 同步执行(启动期无请求,无需 spawn_blocking)。

**深检(`warmup_check`→`/health?deep=true`)**
- 动机:arena OOM 后 cuBLAS **永久失败**,浅检(端口活)完全不可见——僵尸态;
- 实现:对池[0] 发 dummy 推理,失败即 degraded;未加载模型视为健康(无 session 可检);
- 仅在请求 `/health?deep=true` 时执行,不周期跑。

**arch 自动识别(`model_config.rs`)**
- `[_, C, N]` 且 `C>4 && N>C` → v8 族,`nc = C-4`;
- `[_, C, 6]` 且 `C>4` → v10 端到端(类别数 shape 不可推,回退 80 告警);v8 族误触发需 anchor 恰为 6,而 anchor 数由输入尺寸决定(640→8400,最小 32→21),现实中不可能——不设上限;
- 无法判断 → 回退 v8 族 + 80 类 + warn;
- **rust-onnx-infer 已有等价实现**(`detection.rs` 首推理 shape 识别传统/End2End + `Mutex<Option<bool>>` 缓存),此块零吸收成本。

**models.toml 覆盖**
- 优先级:toml 覆盖 > shape 自动检测 > 默认(COCO80/pixel/640);
- 字段:arch/num_classes/coord_format/class_names;
- 文件不存在/解析失败 → 空表 + warn 不阻断;锁中毒恢复数据(`PoisonError::into_inner`,HashMap 无不一致态);
- **coord_format(normalized)不吸收**(见 §三取舍)。

**预处理/后处理(`preprocess.rs`/`postprocess.rs`)**
- letterbox:等比缩放(bilinear)→ 居中 pad **114** → /255 → CHW;`ratio/dw/dh` 供坐标反算 `((x-dw)/ratio).clamp(0, 原图界)`;
- **NaN/Inf 防护**:异常输出直接丢弃——f32::NAN 穿透到 serde_json 序列化会 HTTP 500;
- v8 族:xywh→xyxy、逐 anchor argmax、类内 NMS(`iou > threshold` 抑制);v10:仅阈值/类别过滤无 NMS;
- rust-onnx-infer DetectionEngine 全套等价(letterbox 114/类内 NMS/End2End);且 resize 已修 Convolution 混叠(比 yolo-rs 的 image::Triangle 抗混叠路径更稳)。

**运行时参数(yolo-rs 有,rust-onnx-infer 无)**
- 每请求 `confidence`(multipart form)、全局 `YOLO_IOU`;
- rust-onnx-infer 阈值/NMS 均构造期固定(`set_confidence_threshold(&mut self)`),vision-ext 的 `Arc<EnginePool>` 不可变;
- **吸收方案**:检测引擎构造期用低阈值(0.10),HTTP 层收到每请求 confidence 后对结果**后过滤**(类内 NMS 保序:低阈值 NMS 结果是任意高阈值结果的超集,后过滤数学安全);iou 保持全局配置(v1 不做每请求 iou,工程取舍)。

### 1.2 模型管理(`models/manager.rs`)

- `RwLock<HashMap<name, Arc<SessionWrapper>>>` 懒加载缓存;读锁探查 → 锁外构建 → 写锁**双重检查**(并发同模型只建一份);
- **模型名校验(安全边界)**:拒绝空串/`..`/`/`/`\`/`\0`(name 直接拼路径,防遍历);先校验后碰文件系统;
- 目录即用:首次请求触发加载,文件不存在报错;`unload` 幂等(未加载返回 `unloaded=false`);进行中推理持 `Arc` 自然释放,无强制中断;
- `reload`(allow(dead_code)):重读 toml + 锁外加载 + 写锁替换。

### 1.3 服务层(`main.rs`/`routes/`)

- **背压语义(fail-fast)**:`try_acquire()` 耗尽即返 Overload,**不在信号量上排队**——生产建议 `max_concurrent = pool_size` 防 arena 雪崩;
  - ⚠️ vision-ext 现语义是 `acquire + 30s 超时`(排队)。**差异点**:内网低负载排队更友好;GPU 生产 fail-fast 防雪崩。吸收时做成可配(`acquire_timeout_ms=0` 即 fail-fast)。
- `/detect/batch`:**逐图 for 循环**(非张量 batch),每图独立背压 permit;契约源自 detector.py(`image_paths` 服务端本地路径);
- `save_result`:imageproc 红框标注 + uuid 命名落盘 result_dir(低优先,契约适配期再做);
- 慢请求日志:>1s 升级 warn(A100 热态单请求 ~18ms,超 1s 即异常);
- 双语类名(COCO 80 中英映射表)+ summary(person=0,vehicle=1/2/3/5/7):**适配层职责,不进 vision-ext 核心**(见 §三);
- request_id 三层:Set(最外)→ Propagate → Trace(span),推理日志自动携带——vision-ext 已同款;
- 优雅停机:SIGINT+SIGTERM,无超时上限(依赖 systemd TimeoutStopSec)。

## 二、差距矩阵(rust-onnx-infer / vision-ext 现状)

| # | yolo-rs 能力 | 现状 | 差距类型 |
|---|---|---|---|
| G1 | Session 池 + 线程预算 | vision-ext EnginePool(permit 排队+30s) | ✅ 已有;背压语义需可配(G6) |
| G2 | arch/End2End 自动识别 | rust-onnx-infer detection.rs | ✅ 已有 |
| G3 | letterbox/NMS/坐标反算/NaN 防护 | rust-onnx-infer | ✅ 已有(resize 更稳) |
| G4 | CUDA 显存上限 | `CUDA::default()` 无 with_memory_limit | ❌ **缺口 A**:~15 行 |
| G5 | 启动预热 + 深检 | 无 | ❌ **缺口 B**:各 ~30 行 |
| G6 | fail-fast 背压可配 | 仅排队式 | ⚠️ **缺口 C**:小改 |
| G7 | 动态模型管理(目录即用/unload/toml) | env 静态 + OnceCell | ❌ **缺口 D**(最大,~200 行) |
| G8 | 每请求 confidence | 构造期固定 | ⚠️ **缺口 E**:后过滤模式 |
| G9 | /batch 逐图批量 | 无端点 | ⚠️ **缺口 F**:常规 |
| G10 | ROI 多边形/target_classes 过滤 | 无 | ⚠️ **缺口 G**:常规(射线法 ~30 行) |
| G11 | save_result 标注落盘 | 无 | 契约适配期再做 |
| G12 | 双语类名/summary | 无 | 适配层职责,不进核心 |

## 三、明确不做的事(取舍,防范围蔓延)

1. **coord_format(normalized)支持不吸收**:rust-onnx-infer 的立场是拒绝非标模型(踩坑表已立规:一律 Ultralytics 官方导出);为 hf-mirror 变体模型加坐标格式开关,收益低且重蹈"模型野生态"覆辙。
2. **summary/双语类名不进核心**:这是 safe-bot 契约语义不是视觉服务语义。safe-bot 改造时由消费方按 detections 自算(COCO class 约定 5 行代码);真到必须服务端出,以"检测端点可选参数"形式追加,不污染核心。
3. **每请求 iou 不做**:NMS 已在引擎内完成,重做等于绕过引擎后处理;全局 iou 走配置。运行时 confidence 用后过滤(§1.3),数学安全(类内 NMS 保序)。
4. **真张量 batch(predict_batch_impl)不进 v1**:/batch 端点照 yolo-rs 逐图循环(每图独立背压/错误隔离);真 batch 是后续吞吐优化项,另立任务。

## 四、实施方案(阶段 0-6)

### 阶段 0:fork rust-onnx-infer → viccom(前置,消除单点)
- GitHub 建 `viccom/rust-onnx-infer`(含本地 2 补丁:api-26 + resize 卷积修复),origin 换 fork;
- vision-ext Cargo.toml 由 path 依赖换 git 依赖(或 path 指向固定 commit),README 依赖说明改写;
- **验证**:两仓库 CI 三绿;vision-ext 编译产物行为不变(bus.jpg 检出基线)。

### 阶段 1:GPU 硬化(缺口 A+B+C,~1 天)
- `OnnxRuntimeConfig` 加 `gpu_mem_limit_mb`(默认 0);`add_cuda_ep` 接 `with_memory_limit`;vision-ext env 透传 `VISION_EXT_GPU_MEM_LIMIT_MB`;
- 预热:EngineManager 引擎加载后对池内每实例跑 dummy 推理(DetectionEngine 用 640 零张量;各引擎用自身 input_size),失败 warn 不阻断;日志含每实例耗时;
- 深检:`/health?deep=true` 对已加载引擎发 dummy 推理,输出 `engine_health` 数组(ok/degraded);浅检不动;
- 背压可配:`VISION_EXT_ACQUIRE_TIMEOUT_MS`(默认 30000;**0 = try_acquire fail-fast**,生产 GPU 与 pool 对齐);
- TDD:config 解析/timeout=0 行为/dummy 预热失败不阻断;CPU 路径零回归(85 测试)。
- **关键细节必须带上**:arena 只增不减注释、2.9GB/Session 实测数、fail-fast 与 pool_size 对齐建议、1.26.0 ORT 匹配说明。

### 阶段 2:EngineManager 动态化(缺口 D,~1-2 天,与 P2 重构合并)
- 结构:`OnceCell<固定字段>` → `DashMap<String, Arc<EnginePool<DynEngine>>>`;name→引擎类型由**目录+能力注册表**解析(见下);
- 模型目录约定:`models/` 下按能力子目录(`detect/`、`classify/`、`pose/`…),文件名即模型名;`POST /models/{name}/unload` 幂等卸载(进行中推理持 Arc 自然释放);
- **路径校验照抄 yolo-rs**(拒绝 `..`/`/`/`\`/`\0`/空);
- `models.toml` 每模型覆盖(阶段 2 只做 `confidence_threshold`/`input_size`/`class_names` 三项,arch 自动识别已内建);
- 双重检查锁模式照抄 yolo-rs(读锁探查→锁外构建→写锁复检);
- 兼容:保留现有 env 显式路径配置作为"预置能力"(启动即注册),目录扫描作为"动态能力";/health 的 configured/loaded 语义随之扩展;
- 路由变化:`/detect` 支持可选 `model` 参数(缺省用预置);或 `/detect/{model}`(与现有 /detect 并存,REST 风格对齐批 1 模板);
- 风险:动核心结构,先补 EngineManager 现有行为的特征测试再重构;warden 托管下灰度。
- **这是唯一"重构"级阶段,放在批 1 收尾后。**

### 阶段 3:HTTP 参数层(缺口 E+F+G,~1 天)
- `/detect` 增参:`confidence`(后过滤)、`target_classes`、`roi`(归一化多边形,射线法判定中心点);`/classify` 等按需;
- `/detect/batch`:JSON `{image_paths[], model?, confidence?}`,逐图循环+独立 permit+错误隔离(单图失败该条目带 error,不炸整批);**注意**:image_paths 是服务端本地路径,内网文件语义,文档明示;
- save_result 暂缓(契约适配期)。

### 阶段 4:A100 对标(实测说话,~1 天)
- yhpc 部署:CUDA 12.8.1(cuDNN 9.8)容器基底(勿回退 9.1,差 2 倍);ORT 1.26.0;
- `gpu_mem_limit_mb = 模型数 × 单模型峰值`(yolov8n ~3GB/Session 起步);`acquire_timeout_ms=0` + `max_concurrent=pool_size` + `engine_pool` 与 `ORT 线程`预算对齐(pool×intra≈核);
- 压测基线:对齐 yolo-rs 的 500 QPS(bench 脚本可复用其 `scripts/bench_http.py` 思路);
- 深检/预热在 GPU 上的实测验证(冷启动首请求延迟对比)。

### 阶段 5:safe-bot 适配(消费方改造)
- detector.py/analysis/service.py 改指 vision-ext:双语类名(COCO 映射表搬过去)、summary 消费方自算;
- 若改造受阻 → 备选:独立 api-gateway(仅翻译契约,无推理);**默认走改造**。

### 阶段 6:yolo-rs 归档
- README 顶部标注 `superseded by vision-ext`,仓库转 archive;
- 前置条件:阶段 4 压测达标 + safe-bot 在 vision-ext 上运行 ≥1 个观察周期。

## 五、风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| EngineManager 重构引入回归 | 中 | 先特征测试后重构;warden 托管灰度;批 1 结束再做 |
| rust-onnx-infer CUDA 路径未经生产验证 | 中 | 阶段 4 实测背书;不达标不归档 yolo-rs |
| 每请求 confidence 后过滤与 SAHI 交互 | 低 | SAHI 路径同样低阈值构造+后过滤;E2E 用例覆盖 |
| fork 后上游 javpower 有更新 | 低 | fork 保留 upstream remote,按需 cherry-pick |

## 执行状态(新会话冷启动先读本节)

| 阶段 | 状态 | 验证证据 | 日期 |
|---|---|---|---|
| 0 fork rust-onnx-infer | ⬜ | | |
| 1 GPU 硬化(显存上限/预热/深检/背压可配) | ⬜ | | |
| 2 EngineManager 动态化(与 P2 重构合并,**批 1 收尾后才开始**) | ⬜ | | |
| 3 HTTP 参数层(confidence 后过滤/batch/ROI) | ⬜ | | |
| 4 A100 对标压测(yhpc,基线 500 QPS) | ⬜ | | |
| 5 safe-bot 消费方改造 | ⬜ | | |
| 6 yolo-rs 归档 | ⬜ | | |

- 每完成一阶段更新本表(状态+验证证据+日期),不等记忆;
- 并行线(能力广度)进度见 `engine-exposure-plan.md` 批次表,两表互链;
- 开工顺序建议:0→1 可立即开始;2 必须等批 1(任务 2-4)收尾。

## 变更记录

- 2026-09-10:调研成文(yolo-rs 19 文件全读);阶段 0-6 定稿;与 engine-exposure-plan.md 批次表互链。
