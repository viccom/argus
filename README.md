# argus

工业视觉智能 HTTP 服务，基于 [rust-onnx-infer](https://github.com/viccom/rust-onnx-infer) 封装，提供 12 个图像推理 REST API。

## 能力

| 端点 | 说明 | 模型 |
|---|---|---|
| `POST /detect` | 目标检测（支持 SAHI 切片 `sahi=true`；请求级过滤 `confidence`/`target_classes`/`roi`，见下） | YOLO |
| `POST /detect/batch` | 逐图批量检测（JSON `image_paths` 为**服务端本地路径**；单图失败带 `error` 不炸整批，每图独立背压 permit；过滤参数同 `/detect`） | YOLO |
| `POST /classify` | 图像分类 Top-K（`top_k` 默认 5 上限 50；`confidence` 默认 0.5 为 **top-1 门控**：top-1 低于它返回空列表，过线返回全量 top-k 含低分尾巴——复现历史语义；显式传低值如 0.05 得到带各自 conf 的真实 top-k） | YOLOv8n-CLS |
| `POST /detect/obb` | 旋转框检测（自动识别传统 / End2End 布局） | YOLOv8n-OBB |
| `POST /segmentation` | 实例分割（响应 `data.segments[]`：检测字段 + `area_ratio` 掩码占比 + 64×64 `mask_preview` 下采样预览） | YOLOv8m-seg |
| `POST /ppe` | 工地 PPE 检测（10 类含 NO-Hardhat/NO-Safety Vest 等违章反类；参数语义同 `/detect`：SAHI/confidence/target_classes/roi） | PPE v8 10cls |
| `POST /fire-smoke` | 火焰/烟雾检测（fire/smoke 两类；参数语义同 `/detect`） | Fire-Smoke v8n |
| `POST /face/detect` | 人脸检测（bbox + 5 关键点；`min_face_px` bbox 短边下限 / `max_faces` 按置信度取前 N，默认均关） | YuNet |
| `POST /face/recognize` | 人脸识别（128 维嵌入 + 余弦相似度比对；matches 含 5 关键点 landmarks 与 data.embed_dim；门控参数同 /face/detect，过滤发生在提向量之前） | SFace |
| `POST /reid` | 行人重识别特征（multipart `file` + 可选 `boxes`=[[x1,y1,x2,y2],...] 像素框列表，缺省整图；逐框输出 512 维 L2 归一化 embedding，跨框相似度由消费方做余弦） | OSNet x1.0 |
| `POST /ocr` | OCR 文本检测 + 识别 | PP-OCRv5 |
| `POST /license-plate` | 车牌检测 + 识别（mnet 关键点矫正 + LPRNet） | mnet / LPRNet |
| `POST /qr` | 二维码检测 | WeChat QR |
| `POST /quality` | 图像质量评估（清晰度/亮度/对比度/噪声） | 无需模型 |
| `POST /pose` | 人体姿态（COCO 17 关键点） | YOLOv8n-Pose |
| `POST /depth` | 单目深度估计（min/max/mean + 64×64 预览） | Depth Anything V2 |
| `POST /human-parsing` | 人体解析（ATR 18 类部件占比） | SegFormer-B2 |
| `POST /face/recognize/batch` | 批量人脸识别（多 `file` 字段 ≤16/请求，共享 known_embeddings/threshold/门控参数；逐图串行，单图失败带 `error` 不炸整批，每图独立背压 permit；省 HTTP 往返而非推理并行） | SFace |
| `GET /health?deep=true` | 健康检查（浅检返回已加载引擎 + `backpressure` 对象：in_flight/queued_total/rejected_total/acquire_timeout_total；deep=true 额外返回已配置能力清单） | — |
| `GET /models` | 能力键控的模型清单（`configured`/`loaded`/`models`）。**注意与 yolo-rs 同名端点语义不同**：yolo-rs 是目录扫描（任意 *.onnx 即可用），本服务 models/ 混装异构模型，故按能力键控；消费方取 `data.capabilities.detection.models` 得目标检测可用模型名 | — |
| `GET /ui` | 内嵌 Web 控制台：浏览器拖拽图片、选择处理方式、可视化标注结果（页面免认证加载，API 调用仍需 key） | — |

所有模型按需懒加载（`tokio::OnceCell`），推理经 `spawn_blocking` + 信号量背压。

### 通用响应契约

- **推理响应**均含溯源与背压字段：`data.model`（`文件名#sha256前12位`，多模型端点 `+` 拼接）、`data.pipeline`（管线标识，换模型/预处理语义变化即变——向量失效判定键）、`data.queue_ms`（等待背压 permit 耗时，与 `inference_ms` 分开）。
- **解码边界统一应用 EXIF Orientation**（`image` crate 不隐式变换，实测 orientation=6 照片 2 脸 vs 摆正 38 脸）；bbox 坐标空间为摆正后图像。无 EXIF 的输入（摄像头帧/PNG）零开销不受影响。
- **错误信封**：逻辑错误 HTTP 200 + `{"status":"error","data":{"error":原文,"code":...,"retryable":...}}`。`code` ∈ `bad_request`/`too_large`（body 超 20MB 上限或图像超像素闸）/`unsupported_image`/`model_not_configured`/`capability_disabled`/`overload`/`inference_failed`/`internal`；仅 `overload` 可重试。

### 检测请求级过滤（`/detect` multipart 文本字段，`/detect/batch` JSON 同名字段）

| 参数 | 格式 | 语义 |
|---|---|---|
| `confidence` | `0.5`，范围 **[0.1, 1.0]**，默认 0.25 | 置信度阈值。引擎以 0.1 低阈值构造后按请求后过滤（类内 NMS 保序，与引擎直接以该阈值构造严格等价）；SAHI 路径在切片合并**前**过滤，避免低分碎片先放大高分框几何。低于 0.1 返回错误而非静默截断 |
| `target_classes` | JSON 整数数组 `[0, 2, 5]`（class_id，非空） | 仅保留这些类别 |
| `roi` | JSON 归一化多边形 `[[x,y],...]`，≥3 点，坐标 ∈ [0,1] | 仅保留 **bbox 中心点**落在多边形内的检测（射线法，支持凹多边形） |

`/detect/batch` 的 `model` 参数可选：命名动态槽（slots.toml 注册，见下节），缺省用预置 detection 槽。NMS IoU 全局固定 0.45（NMS 在引擎内完成，不做每请求覆盖）。

### 动态检测槽（slots.toml）

检测族新增模型从"改代码 + 发版"退化为"`slots.toml` 加一段 + 热重载"（2026-10-21 上线，吸收计划阶段 2）。示例见仓库根 [slots.example.toml](slots.example.toml)，校验规则见 `docs/designs/2026-10-21-engine-manager-dynamic-slots-design.md` §4.4。

```toml
[slots.crack]                       # 槽名：^[a-z][a-z0-9_]{0,31}$，不得与 15 个能力键重名
model = "yolov8n-crack.onnx"        # 相对 $ARGUS_MODELS_DIR；绝对路径也必须落在该目录内
class_names = ["crack", "pothole"]  # 可选：覆盖模型 metadata 类名（自训模型常缺 metadata）
pool_size = 1                       # 可选：缺省取 ARGUS_ENGINE_POOL，范围 1..=64
warmup_size = [640, 640]            # 可选：预热 dummy 尺寸，两维均 32..=4096
```

- **用法**：三个检测端点（`/detect`、`/ppe`、`/fire-smoke`）的 multipart 文本字段 `model=<槽名>`，batch 的 JSON 字段 `model`。缺省/空串走原静态路径，行为逐字节不变。响应溯源为 `data.model`（`文件名#sha256前12位`）+ `data.pipeline = "detect-slot-v1"`。
  ```bash
  curl -s -X POST localhost:7042/detect -H "X-API-Key: $KEY" -F "file=@bus.jpg" -F "model=crack" | jq '.data | {model, pipeline, count}'
  ```
- **错误码**：`model` 指向未注册/已删除槽 → `unknown_model`（信封内列可用槽名，不泄露路径）；`model` 传静态能力键（如 `ppe`）→ `bad_request`（提示改用对应端点，不跨端点转发）。启动时 `slots.toml` 非法 → **拒绝启动**（fail-closed）。
- **热重载**：watcher 轮询 `slots.toml` 指纹（mtime 纳秒 + size），默认 3s 周期，新增/修改/删除槽 ≤2 个周期生效；解析失败保留 last-known-good（不清空）。**换模型文件必须原子替换**（临时文件 + `mv`）后 `touch slots.toml`——同路径原地改写 slots.toml 不触发重载。
- **观测**：`/models` 的 `data.dynamic_slots.<name>.{configured,loaded,model_file,pipeline}`（15 项 `capabilities` 不变）；`/health?deep=true` 的 `dynamic_engines` + `dynamic_slot_health`（`engines`/`engine_health` 不变）。槽懒加载：首次请求才构建引擎，`loaded=false` 表示已注册未加载。
- **回滚**：删 `slots.toml`（或 `ARGUS_SLOTS_CONFIG` 指向不存在的路径）即清空动态槽，静态能力不受影响；二进制回滚到 2026-10-21 以前则 `slots.toml` 被忽略（新字段不被旧代码读取）。
- **生产提示**：`slots.toml` 默认落在 gitignore 的 `models/` 下（部署数据）。双实例（:7042/:7043）各自懒加载同一槽会双份内存——峰值由懒加载 + 替换不预载最小化，但同槽双加载仍须计入预算。**当前生产未激活**（无 slots.toml，零回归）；激活前需先在两个实例的 models 目录部署 slots.toml 并评估内存。

## 构建与运行

```bash
cargo build --release --jobs 6
```

运行时需 ORT 动态库（`load-dynamic`）与模型文件，环境变量见下。

> **E2E golden 复比约定**：`tests/golden/` 的静态端点基线复比**须在零推理时刻进行**（或 diff 前 mask 掉 `loaded`/`engines`/`engine_health` 等懒加载运行态字段）：懒加载后首次推理会翻转这些字段，产生与契约无关的假 diff（T12 对照实证，2026-10-21）。

版本标识由 build.rs 编译期探测注入（git hash + dirty 标记），见 `/health` 的 `version` 字段。build.rs 监听 git 目录的 `HEAD` + `refs/`：新提交后直接 `cargo build` 即更新版本串，无需手动 touch。
`SIGTERM`/`Ctrl+C` 触发优雅停机：等待在途请求完成后退出（`ARGUS_SHUTDOWN_TIMEOUT_SECS` 超时保底强退，默认 10s）。

## 配置（环境变量）

### 部署布局约定

`argus.env` 里的 `ORT_DYLIB_PATH` / `ARGUS_MODELS_DIR` 用**相对路径**（`./lib/...`、`./models`），由 `run.sh` 切到自身所在目录后解析：

- 发版只需替换 bundle 目录（二进制 + `run.sh` + `argus.env`，`models`/`lib` 用软链），**不改任何字面路径**；
- 直连 `./argus` 启动时须保证 cwd 为 bundle 目录，否则相对路径失效。

| 变量 | 默认 | 说明 |
|---|---|---|
| `ORT_DYLIB_PATH` | — | libonnxruntime.so 路径（必需；bundle 内为 `./lib/libonnxruntime.so.1.26.0`） |
| `ARGUS_PORT` | 7042 | 监听端口 |
| `ARGUS_DEVICE` | cpu | cpu / cuda |
| `ARGUS_AUTH_TOKEN` | 无 | 设置后启用 API key（`X-API-Key` 或 `Authorization: Bearer`） |
| `ARGUS_MAX_BODY_MB` | 20 | 请求体上限 |
| `ARGUS_MAX_IMAGE_PIXELS` | 0 | 解码前像素总数闸门（decoder.dimensions() 预检，防 20MB 内亿像素 PNG 解码炸弹；0=不限，部署建议 6000 万） |
| `ARGUS_MAX_CONCURRENT_INFERENCE` | 0 | 并发推理上限（0=不限） |
| `ARGUS_ENGINE_POOL` | 1 | 每引擎实例池大小。>1 时同模型多实例并行推理（多路客户端同模型并发不再串行）。**代价**：内存 = 模型大小 × 池大小 × 已加载引擎数；CPU 预算 = 池大小 × 4（ORT 线程）× 引擎数 ≈ 核数。与多实例部署可叠加（实例数 × 池大小统算） |
| `ARGUS_<CAP>_POOL` | 回退 `ARGUS_ENGINE_POOL` | 单能力独立设池，覆盖全局值（如 `ARGUS_OCR_POOL=1`）。`<CAP>` 为能力键大写，同 `WARMUP_SIZE`。内存治理杠杆：内存由 ORT 每 session 运行时开销主导（非模型文件大小），`OCR`/`HUMAN_PARSING`/`DEPTH` 设 1 即拿回大头（池 4 全能力 12.4GB，此三项占 ~10.3GB） |
| `ARGUS_ENABLED_CAPABILITIES` | 未设置 = 全部启用 | 能力白名单（逗号分隔能力键，如 `detection,ocr,face_detection`）。未设置 = 现有行为（凡模型路径已配置即启用），零回归。白名单外能力在 loader 入口即拒绝（返回"能力未启用"，与"模型未配置"可区分），**不触碰文件系统、不创建 session**；`/models` 的 `enabled` 字段与 `/health?deep=true` 的 `configured` 清单同步反映。动机：将来模型目录扫描落地时，这是防止把全部模型拉起来的闸门 |
| `ARGUS_<CAP>_WARMUP_SIZE` | 沿用引擎 input_size | 预热 dummy 输入尺寸覆盖，`WxH` 格式（如 `ARGUS_OCR_WARMUP_SIZE=320x320`）。`<CAP>` 为能力键大写：`DETECTION`/`PPE`/`FIRE_SMOKE`/`CLASSIFICATION`/`OBB`/`SEGMENTATION`/`FACE_DETECTION`/`FACE_RECOGNITION`/`REID`/`OCR`/`LICENSE_PLATE`/`QR`/`POSE`/`DEPTH`/`HUMAN_PARSING`（注意与模型路径 env 的 `FACE_DET`/`PLATE_DET` 缩写不同）。OCR 必设：960×960 全零 dummy 触发 det 大量伪区域 + rec 批前向，预热 20.4s/实例；320×320 实测 0.43s/实例且首个真实请求推理耗时不变（2026-09-11 实测，池 4：预热 81.5s→1.8s，冷启动首请求 91.3s→11.7s）。CPU 设备上预热对首次真实推理无可见收益（关闭预热实测首推理 6.5s ≈ 热态），保留仅消除潜在首次抖动 |
| `ARGUS_SHUTDOWN_TIMEOUT_SECS` | 10 | 优雅停机超时：SIGTERM 后等待在途请求的上限，超时强退。需覆盖最长推理（SAHI/depth ~1.5s，池化排队翻倍） |
| `ARGUS_GPU_MEM_LIMIT_MB` | 0 | CUDA EP 显存上限（MB）。>0 时设置 ORT arena `with_memory_limit`，防 arena 无限膨胀导致 cuBLAS OOM 后永久失败（僵尸态）。0 = 不限（ORT 默认） |
| `ARGUS_ACQUIRE_TIMEOUT_MS` | 30000 | 推理排队等待超时（ms）。全忙时请求等待引擎池 permit 的上限，超时返回 503。**0 = fail-fast**：不排队，全忙立即 503 |
| `ARGUS_SLOTS_CONFIG` | 未设置 = `$ARGUS_MODELS_DIR/slots.toml` | 检测族动态槽清单（slots.toml）显式路径覆盖。文件缺失 = 无动态槽（零回归）；文件非法 → 拒绝启动（fail-closed）。部署数据默认落 gitignore 的 models/，入库示例见仓库根 `slots.example.toml` |
| `ARGUS_SLOTS_WATCH_SECS` | 3 | slots.toml 热重载轮询周期（秒）。0 = 关闭 watcher（仅启动时加载一次）。新增/修改/删除槽 ≤2 个周期生效；同路径原地改写文件不触发（指纹 = mtime 纳秒 + size），换模型文件须原子替换后 touch slots.toml |
| `ARGUS_*_MODEL` | — | 各引擎模型路径（`DETECTION` / `PPE` / `FIRE_SMOKE` / `CLASSIFICATION` / `OBB` / `SEGMENTATION` / `FACE_DET` / `FACE_REC` / `REID` / `OCR_DET` / `OCR_REC` / `OCR_DICT` / `PLATE_DET` / `PLATE_REC` / `QR` / `POSE` / `DEPTH` / `HUMAN_PARSING`）。**命名差异**：模型路径 env 是历史缩写（`FACE_DET`/`FACE_REC`/`PLATE_DET`），新增的按能力 env（`<CAP>_POOL`/`<CAP>_WARMUP_SIZE` 等）一律用规范能力键大写（`FACE_DETECTION`/`FACE_RECOGNITION`/`LICENSE_PLATE`/`FIRE_SMOKE`），与 `/models` 的能力名一致，不引入第三套命名 |

## 模型下载

模型不入库。已验证的来源（均可直连或经 `hf-mirror.com` 镜像加速）：

| 文件 | 来源 |
|---|---|
| `yolov8n.onnx` | Ultralytics 导出：`YOLO('yolov8n.pt').export(format='onnx', imgsz=640)`。**勿用 Xenova/yolo-rs release 变体**——其检测头输出归一化坐标（0~1）且无类名元数据，与解码器（按 640 像素空间 letterbox 反算）不兼容，表现为 bbox 零宽/SAHI 全丢弃 |
| `yolov8n-cls.onnx` | Ultralytics 导出：`YOLO('yolov8n-cls.pt').export(format='onnx', opset=12, imgsz=224)`（.pt 取自 `ultralytics/assets` release v8.3.0）。ImageNet 1000 类标签嵌在 ONNX metadata（`names`），引擎自动读取；输出已 softmax |
| `yolov8n-obb.onnx` | Ultralytics 导出：`YOLO('yolov8n-obb.pt').export(format='onnx', imgsz=640)` |
| `yolov8n-pose.onnx` | [Xenova/yolov8n-pose](https://huggingface.co/Xenova/yolov8n-pose)（`onnx/model.onnx`） |
| `face_detection_yunet_2023mar.onnx` | [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` | [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface) |
| `ppocrv5_det.onnx` / `ppocrv5_rec.onnx` / `ppocrv5_dict.txt` | [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)（`detection/v5/det.onnx`、`languages/chinese/rec.onnx`、`languages/chinese/dict.txt`） |
| `wechat_qr_detect.onnx` | [opencv/opencv_contribution](https://huggingface.co/opencv/opencv_contribution) `wechat_qr/detect_2026april.onnx` **需图手术**：输入 384→416、按 416 重算 11 个 flatten shape 常量、裁掉 PriorBox×5 + DetectionOutput 尾层、输出改为 `mbox_loc [1,32448]` + `mbox_conf_flatten [1,16224]` |
| `mnet_plate.onnx` / `Final_LPRNet_model.onnx` | [hpc203/license-plate-detect-recoginition-opencv](https://github.com/hpc203/license-plate-detect-recoginition-opencv) raw 直链 |
| `depth_anything_v2_small.onnx` | [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)（`onnx/model.onnx`） |
| `segformer_b2_clothes.onnx` | [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)(`onnx/model.onnx`,110MB 全量) |
| `osnet_x1_0.onnx` | Qualcomm AI Hub 官方导出（[qualcomm/OSNet](https://huggingface.co/qualcomm/OSNet) 的 `release_assets.json` S3 直链，v0.62.2）。**zip 内为外置权重格式**（osnet.onnx + osnet.data），需 `onnx.load(load_external_data=True)` + `onnx.save` 内联为单文件；输入 `image [1,3,256,128]` 静态、输出 `embeddings [1,512]` 图内已 L2 |
| `yolov8m-seg.onnx` | 本地导出：`YOLO('yolov8m-seg.pt').export(format='onnx', imgsz=640, opset=12)`（.pt 取自 `ultralytics/assets` release v8.3.0）。COCO 80 类实例分割，`[1,116,8400]` + proto `[1,32,160,160]`。服务 `/segmentation` |
| `ppe_site_10cls.onnx` | [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) `best.pt` 本地导出（同上参数）。10 类：Hardhat/Mask/NO-Hardhat/NO-Mask/NO-Safety Vest/Person/Safety Cone/Safety Vest/machinery/vehicle（含**违章反类**）。服务 `/ppe`。qualcomm gear_guard_net 因三输出自定义头弃用 |
| `fire_smoke_2cls.onnx` | [mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8) `fire_smoke_yolov8.pt` 本地导出（同上参数）。fire/smoke 两类。正样本实测置信度中等（0.52~0.62，nano 级），生产采用前建议再比选模型。服务 `/fire-smoke` |

## 依赖说明

`rust-onnx-infer` 为 [viccom/rust-onnx-infer](https://github.com/viccom/rust-onnx-infer)（上游 [javpower/rust-onnx-infer](https://github.com/javpower/rust-onnx-infer) 的 fork），以 git 依赖引入（Cargo.toml，具体 commit 由 Cargo.lock 锁定）。相对上游有两处修改：

1. `ort` feature `api-27` → `api-26`。原因：ort 2.0.0-rc.13 的版本检查用 ORT 库的
   release minor 与 `ORT_API_VERSION` 比较，而 ORT 1.18 起 API version = release minor + 1，
   导致 ORT 1.26.0 动态库被误判 `BadVersion`；`api-26` 下 `ORT_API_VERSION=26` 与 1.26.0 匹配，
   且 ORT 库向后兼容 api-26 接口。
2. `imaging/resize.rs`：`Interpolation::Linear/Cubic` 由 `ResizeAlg::Interpolation`（固定核
   快速算法）改为 `ResizeAlg::Convolution`（核随缩放比拉伸）。原因：固定核在大比率缩小（>2x）
   时跳过核外源像素产生混叠——1080→224 缩小时分类置信度 0.58 崩至 0.38（等效 nearest）、
   detect 出混叠伪影框；修复后 detect 收敛为标准结果。附回归测试
   `linear_downscale_should_spread_impulse_energy`。

本地开发时 `../rust-onnx-infer` 仍为源仓库（`origin` = fork、`upstream` = javpower）；
改 rust-onnx-infer 后 `git push origin master`，再在本仓库 `cargo update -p rust-onnx-infer` 拉取新 commit。
上游更新按需 cherry-pick：`git fetch upstream && git cherry-pick <commit>`。

## License

MIT
