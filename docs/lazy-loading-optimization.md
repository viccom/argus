# vision-ext 按需加载优化方案

> 目的:治理按需加载(懒加载)模式的四类代价——冷启动击穿客户端超时、池大小线性放大成本、
> 加载无准入、内存只增不减。
> 状态:**阶段 1(止血)已实施并验证**(2026-09-11);阶段 2 待做。本文只定方案与验证口径,不含代码改动。
> 基线数据均为本机实测(16 核 / 31GB 共享 KVM,CPU 设备,ORT 1.26.0,pool=4)。

## 一、现状测量(基线)

### 1.1 冷启动成本

| 能力 | 池(4)加载 | 单实例预热 | 预热合计 | 首次请求最低耗时 |
|---|---|---|---|---|
| qr | 60ms | 15ms | 0.06s | ~0.1s |
| license_plate | 107ms | 20ms | 0.08s | ~0.2s |
| pose | 208ms | 60ms | 0.24s | ~0.4s |
| depth | 1.21–1.34s | 518ms | 2.07s | ~3.4s |
| human_parsing | 1.77s | 905ms | 3.62s | ~5.4s |
| **ocr** | 3.47–4.25s | **20.4s** | **81.5s** | **~85s** |

- 未测到的能力(classification/obb/face_detection/face_recognition/detection)其预热日志被 tail 截断,
  **不填猜测值**;实施前如需完整基线,按 §5 的方法补测。
- OCR 的 4 个实例预热耗时高度一致(20.42/20.36/20.32/20.36s),说明是确定性成本而非资源争抢。

### 1.2 内存成本(逐能力实测,重启后从空载逐个加载)

| 能力 | RSS 增量 | 权重体积 | 放大倍数 |
|---|---|---|---|
| classification | 0.05GB | 11MB | ~5× |
| qr | 0.03GB | 1MB | ~30× |
| face_detection | 0.09GB | 0.2MB | ~450× |
| license_plate | 0.10GB | 3.5MB | ~29× |
| obb | 0.18GB | 12MB | ~15× |
| pose | 0.18GB | 13MB | ~14× |
| detection | 0.22GB | 13MB | ~17× |
| face_recognition | 0.24GB | 37MB | ~6× |
| depth | 1.04GB | 95MB | ~11× |
| **ocr** | **4.88GB** | 165MB | ~30× |
| **human_parsing** | **5.38GB** | 105MB | ~51× |
| **合计** | **12.39GB** | 约 370MB | **~33×** |

结论:内存由 **ORT 每 session 的运行时开销**主导,与模型文件大小关系不大(受输入分辨率驱动的
激活张量占大头)。ocr + human_parsing 两项即占全量的 83%。

### 1.3 测量方法(可复现)

```bash
# 日志证据(加载耗时 / 每实例预热耗时)
grep -E '池\([0-9]+\) 加载完成|预热开始|实例 [0-9]+/[0-9]+ 预热' \
  ~/.local/state/warden/data/logs/vision-ext/$(date +%F).log

# 逐能力内存:重启后从空载逐个打端点,每次读 RSS
# 注意 PID 必须取 warden 报告的权威值(直接 pgrep 会取到过渡进程,曾导致一轮数据全为 0)
TOKEN=<warden token>
PID=$(curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8789/api/v1/services \
      | jq -r '.services[]|select(.name=="vision-ext")|.state.pid')
ps -o rss= -p "$PID"
```

## 二、问题清单

1. **首次请求承担全部冷启动,且被池大小线性放大**——`engine.rs` 各 loader 的
   `for _ in 0..pool_size` 是饿汉式:只来 1 个请求也要把 N 个实例全加载完并逐实例预热,
   故冷启动成本 = 加载 × N + 预热 × N。
2. **预热在 OCR 上是异常项**:单实例 20.4s,而 depth 0.52s / human_parsing 0.9s / pose 0.06s,
   差 20~300 倍。预热本意是消除首次抖动,在 OCR 上反而成了主要成本。
3. **击穿客户端超时并触发消费方假故障**:safe_bot `REQUEST_TIMEOUT = 60.0s`
   (`safe_bot_v0/web_app/backend/modules/ai/detector.py:157`)< OCR 冷启动 85s;
   而网关把 Timeout 归入连接类异常、累加 `consecutive_failures`,达 3 次摘除端点
   (30s 内健康检查会回池,可自愈)。后果:连续几次冷 OCR 调用会让消费方把**健康**的
   vision-ext 短暂标记为不健康。
4. **内存只增不减、无上限、无淘汰**:`OnceCell` 永不释放;无 unload 端点;
   `VISION_EXT_GPU_MEM_LIMIT_MB` 只管 CUDA,CPU 侧无内存上限。
5. **加载路径无准入控制、无超时**:`engines.X().await`(含加载+预热)在 `acquire_permit`
   **之前**(`src/routes/detection.rs:170-172`、`src/routes/analysis.rs:264-266` 等),
   因此背压信号量**管不到加载**;11 个并发首次请求 → 44 个 session 同时构建。
   且加载中的等待不受 `VISION_EXT_ACQUIRE_TIMEOUT_MS` 约束(许可在加载之后才申请),
   冷启动期间请求可无限期挂起。
6. **冷 / 缺 / 坏三态不可区分**:`/health?deep=true` 的 `check_engine!` 宏
   (`src/engine.rs:49-71`)对未加载引擎**直接跳过**(`if let Some(pool) = cell.get()`),
   故 `engine_health` 里没有 "cold" 状态;warden 仅做 TCP 探活,冷启动的服务在首请求
   被拖死之前看起来完全健康。
7. **模型替换必须重启**(已文档化的坑):`OnceCell` 缓存实例,换 `.onnx` 文件不生效——
   这是"无卸载 / 无重载"的另一面。

## 三、阶段 1:止血(env 级改动,不动 EngineManager 结构)

### 3.1 修 OCR 预热——测量驱动,不预设结论

成因目前**未确定**,有两个候选,只读证据无法分辨:

- **候选 A**:det 在 960×960 全零图上产出大量/大尺寸伪区域,导致后续 rec 批前向代价高。
  (dummy 确为全零图:`Image::new` 零填充,见 `rust-onnx-infer/src/imaging/image.rs:20-27`。
  注意 `OcrPipeline::recognize` 是 **det 后单次批前向**,不是逐框推理,见
  `rust-onnx-infer/src/engines/ocr.rs:759-766`。)
- **候选 B**:rec 首跑动态宽度触发 ORT 内核特化,属固有成本。

**对照实验(实施前先跑)**,三组配置各量**两个数**:

| 组 | 预热 dummy | 量什么 |
|---|---|---|
| i 现状 | 引擎 input_size(OCR=960×960) | 预热耗时 **与** 冷启动后首个真实请求耗时 |
| ii 小 dummy | 如 320×320 或更小 | 同上 |
| iii 关闭预热 | — | 同上 |

判据与分支:
- 若 ii 的**两个数**都显著低于 i → dummy 是主因,引入"按能力可覆盖的预热输入尺寸"
  (默认沿用引擎 input_size,OCR 用小图)。
- 若 ii 只是预热变快、而**首个真实请求**仍慢 → 小 dummy 温不到 rec 的动态宽度路径,
  修法不成立,退化为提供开关。
- 若 iii 的两个数与 i 相当 → 预热在 CPU 上本就无收益,提供
  `VISION_EXT_<CAP>_WARMUP=0/1` 关闭开关并在 README 记录代价。

> **口径要点**:必须同时量"首个真实请求耗时"。只量预热耗时会把"把成本挪到首请求"误判为优化。

### 3.2 单能力独立设池

- 新增 `VISION_EXT_<CAP>_POOL`,覆盖全局 `VISION_EXT_ENGINE_POOL`;未设置则回退全局。
- 新增 `Config::pool_for(capability) -> usize`;`src/engine.rs` 的 11 处 loader
  (`:168,196,224,250,278,310,340,374,400,428,454`)由 `self.config.engine_pool`
  改为 `self.config.pool_for("<capability>")`。
- 预期效果(对照 §1.2):`OCR / HUMAN_PARSING / DEPTH` 设为 1 后,全能力内存
  **12.4GB → 约 4GB**,OCR 预热 81.5s → 约 20s(叠加 §3.1 后应更低)。

#### 命名映射表(消除第三套命名)

现有两套名字:模型路径 env 用 `FACE_DET` / `FACE_REC` / `PLATE_DET`,而能力键
(`/models`、`list_loaded`、`configured_engines`)用 `face_detection` / `face_recognition` /
`license_plate`。**新增的 env 一律采用能力键**,不再引入第三套;与模型路径 env 的差异
在 README 中显式记录。

| 能力键(规范名) | 现有模型路径 env | 新增池 env | 白名单取值 |
|---|---|---|---|
| detection | `VISION_EXT_DETECTION_MODEL` | `VISION_EXT_DETECTION_POOL` | `detection` |
| classification | `VISION_EXT_CLASSIFICATION_MODEL` | `VISION_EXT_CLASSIFICATION_POOL` | `classification` |
| obb | `VISION_EXT_OBB_MODEL` | `VISION_EXT_OBB_POOL` | `obb` |
| face_detection | `VISION_EXT_FACE_DET_MODEL` | `VISION_EXT_FACE_DETECTION_POOL` | `face_detection` |
| face_recognition | `VISION_EXT_FACE_REC_MODEL` | `VISION_EXT_FACE_RECOGNITION_POOL` | `face_recognition` |
| ocr | `VISION_EXT_OCR_DET_MODEL` + `_REC_` + `_DICT` | `VISION_EXT_OCR_POOL` | `ocr` |
| license_plate | `VISION_EXT_PLATE_DET_MODEL` + `_REC_` | `VISION_EXT_LICENSE_PLATE_POOL` | `license_plate` |
| qr | `VISION_EXT_QR_MODEL` | `VISION_EXT_QR_POOL` | `qr` |
| pose | `VISION_EXT_POSE_MODEL` | `VISION_EXT_POSE_POOL` | `pose` |
| depth | `VISION_EXT_DEPTH_MODEL` | `VISION_EXT_DEPTH_POOL` | `depth` |
| human_parsing | `VISION_EXT_HUMAN_PARSING_MODEL` | `VISION_EXT_HUMAN_PARSING_POOL` | `human_parsing` |

### 3.3 能力白名单("配置中不定义的就不加载")

- 新增 `VISION_EXT_ENABLED_CAPABILITIES=detection,face_detection,...`(逗号分隔能力键)。
- **未设置 = 现有行为**(凡模型路径已配置即启用),保证零回归。
- 强制点在 11 个 loader 入口:不在白名单 → 返回明确错误(与"模型未配置"区分,便于排错),
  **不触碰文件系统、不创建 session**。
- `/models` 增 `enabled` 字段(增量字段;safe_bot 只读 `capabilities.detection.models`,
  不受影响)。`/health?deep=true` 的 `configured` 语义需与之一致,避免"配了却永远不会加载"
  的沉默状态。
- 动机:将来若做模型目录扫描(吸收计划阶段 2),白名单是防止把 13 个模型全拉起来的闸门。

## 四、阶段 2:空闲淘汰(需重构 `OnceCell`,与吸收计划阶段 2 合并)

### 4.1 前置门(不通过则不做)

先验证**丢弃引擎后 RSS 是否真的回落**。丢弃 `Arc` 只让 Rust 释放内存,glibc 分配器未必
把页归还 OS——若不降,本项收益存疑。对策三选一(接受 / `malloc_trim` / 更换分配器)需先定。

### 4.2 设计

- 11 个 `OnceCell<Arc<EnginePool<T>>>` 字段(`src/engine.rs:88-98`)改为
  `RwLock<Option<Arc<EnginePool<T>>>>` + 每能力 `last_used` 时间戳(成功取用即刷新)。
- `evict_idle(ttl) -> Vec<&'static str>`:仅当
  ①该池**全部许可空闲**(`available_permits() == capacity`,即无在途推理)、
  ②空闲时长 > TTL、③不在 pin 集合内 —— 才丢弃管理器持有的 `Arc`。
  在途请求各自持 `Arc`,淘汰不会中断它们;内存待最后一个 `Arc` 释放才回收。
- 后台回收任务由 `main.rs` 在 `VISION_EXT_IDLE_EVICT_SECS > 0` 时拉起(**默认 0 = 关闭**,
  保持现状)。
- **pin 与白名单语义分开**:白名单 = 可否加载;pin = 可否淘汰。当前可共用同一份列表,
  但语义必须分别文档化。
- 前置动作:先补齐现有行为的特征测试再重构(`list_loaded` / `deep_health_check` /
  `OnceCell::initialized` 语义),吸收计划已声明这是唯一"重构级"阶段。

## 五、验证口径

- 每阶段:`cargo clippy --all-targets --jobs 6` 零警告 + `cargo test --jobs 6` 全绿
  (当前基线 128 + 1 lifecycle)。
- **预热与内存类改动只能 E2E 验证**(单测不加载真实模型,无法覆盖预热耗时):用 §1.3 的方法
  ——PID 取 warden 权威值、逐能力 RSS 增量、日志预热耗时、冷启动首请求实测。
- 阶段 2 追加:静置 TTL 后 RSS 回落实测;淘汰后再次请求能重新加载成功;淘汰有日志留痕。

## 六、已知残留风险(不当作已解决)

- **加载仍无准入控制**:本次未纳入(实施时另行决定)。按需 + 淘汰下仍可能出现并发加载风暴,
  冷启动等待仍无超时,§2.5 的问题依旧存在。
- **冷 / 缺 / 坏三态不可区分**(§2.6)不在本次范围,深检仍会跳过未加载引擎。
- 若 §3.1 的测量表明 20.4s 是 CPU 上 OCR 的固有代价,则只提供开关并记录代价,
  **不强行优化**(避免造出"把成本挪走"的假优化)。
- 淘汰打开后,重冷启动会重复支付预热成本——这也是 §3.1 必须先做的原因。

## 七、执行状态

| 项 | 状态 | 验证证据 | 日期 |
|---|---|---|---|
| 基线测量(冷启动 + 内存) | ✅ | §1.1 / §1.2 表;日志 `~/.local/state/warden/data/logs/vision-ext/2026-09-11.log` | 2026-09-11 |
| 3.1 修 OCR 预热 | ✅ 已实施 | 三组对照(池 4,冷启动 POST /ocr 真实图):现状 960 dummy 预热 81.5s/首请求 91.3s;320 小 dummy 预热 1.8s/首请求 11.7s(首推理 6.44s 不变);关闭预热 首请求 10.0s(首推理 6.52s≈热态 6.55s)。**命中分支 1**:dummy 是主因。实现 `VISION_EXT_<CAP>_WARMUP_SIZE`(commit 1482e06);E2E:OCR=320x320 生效,预热 4×445ms,冷首请求 11.7s。附加发现:CPU 上预热对首次真实推理无可见收益(组 iii),保留预热仅消除潜在首次抖动,已记 README | 2026-09-11 |
| 3.2 单能力独立设池 | ✅ 已实施 | TDD 4 测试(回退/覆盖/能力隔离/非法值)。E2E(池覆盖 OCR/HP/DEPTH=1,其余池 4,全能力逐个加载):RSS 总增量 **12.39GB→3.91GB**(ocr +0.75GB、depth +0.33GB、human_parsing +1.74GB),OCR 冷首请求 11.7s→7.85s(pool=1 只加载 1 实例)。commit 846bd56 | 2026-09-11 |
| 3.3 能力白名单 | ✅ 已实施 | TDD 6 测试(loader 强制点用不存在路径验证不触文件系统)。E2E:白名单 `detection,ocr` 下 /depth 返回"能力未启用"(与未配置可区分)、/models `enabled=false`、/health deep `configured` 过滤为 `[detection,ocr]`、白名单内 /detect 正常;移除白名单后恢复默认全启用(零回归)。commit 14f63f0 | 2026-09-11 |
| 4 空闲淘汰 | ⬜ 前置门未过 | 需先验证 RSS 回落 | |

## 变更记录

- 2026-09-11:方案定稿(基于当日实测)。成因诊断经复核修正:OCR 预热**不是**"rec 逐框推理"
  (实际为单次批前向),真实成因待 §3.1 对照实验判定。
- 2026-09-11:阶段 1 实施完成。§3.1 对照实验判定成因:**候选 A 成立**(960 全零 dummy 让
  det 产出大量伪区域抬高 rec 批前向;320 小图预热 20.4s→0.44s/实例,首请求推理不变),
  采用"按能力可覆盖的预热输入尺寸"分支;附加实测:CPU 上预热对首次真实推理无可见收益
  (关闭预热首推理 ≈ 热态),未采用分支 3 的开关(判据"iii 与 i 相当"不成立)。
  §3.2 / §3.3 按 TDD 落地并 E2E 验证:全能力 RSS 增量 12.39GB→3.91GB(符合 §3.2 预期
  "约 4GB")。执行状态见 §7。
