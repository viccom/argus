# vision-ext

工业视觉智能 HTTP 服务，基于 [rust-onnx-infer](https://github.com/viccom/rust-onnx-infer) 封装，提供 12 个图像推理 REST API。

## 能力

| 端点 | 说明 | 模型 |
|---|---|---|
| `POST /detect` | 目标检测（支持 SAHI 切片 `sahi=true`；请求级过滤 `confidence`/`target_classes`/`roi`，见下） | YOLO |
| `POST /detect/batch` | 逐图批量检测（JSON `image_paths` 为**服务端本地路径**；单图失败带 `error` 不炸整批，每图独立背压 permit；过滤参数同 `/detect`） | YOLO |
| `POST /classify` | 图像分类 Top-K（`top_k`，默认 5，上限 50） | YOLOv8n-CLS |
| `POST /detect/obb` | 旋转框检测（自动识别传统 / End2End 布局） | YOLOv8n-OBB |
| `POST /face/detect` | 人脸检测（bbox + 5 关键点） | YuNet |
| `POST /face/recognize` | 人脸识别（128 维嵌入 + 余弦相似度比对） | SFace |
| `POST /ocr` | OCR 文本检测 + 识别 | PP-OCRv5 |
| `POST /license-plate` | 车牌检测 + 识别（mnet 关键点矫正 + LPRNet） | mnet / LPRNet |
| `POST /qr` | 二维码检测 | WeChat QR |
| `POST /quality` | 图像质量评估（清晰度/亮度/对比度/噪声） | 无需模型 |
| `POST /pose` | 人体姿态（COCO 17 关键点） | YOLOv8n-Pose |
| `POST /depth` | 单目深度估计（min/max/mean + 64×64 预览） | Depth Anything V2 |
| `POST /human-parsing` | 人体解析（ATR 18 类部件占比） | SegFormer-B2 |
| `GET /health?deep=true` | 健康检查（浅检返回已加载引擎；deep=true 额外返回已配置能力清单） | — |
| `GET /models` | 能力键控的模型清单（`configured`/`loaded`/`models`）。**注意与 yolo-rs 同名端点语义不同**：yolo-rs 是目录扫描（任意 *.onnx 即可用），本服务 models/ 混装异构模型，故按能力键控；消费方取 `data.capabilities.detection.models` 得目标检测可用模型名 | — |
| `GET /ui` | 内嵌 Web 控制台：浏览器拖拽图片、选择处理方式、可视化标注结果（页面免认证加载，API 调用仍需 key） | — |

所有模型按需懒加载（`tokio::OnceCell`），推理经 `spawn_blocking` + 信号量背压。

### 检测请求级过滤（`/detect` multipart 文本字段，`/detect/batch` JSON 同名字段）

| 参数 | 格式 | 语义 |
|---|---|---|
| `confidence` | `0.5`，范围 **[0.1, 1.0]**，默认 0.25 | 置信度阈值。引擎以 0.1 低阈值构造后按请求后过滤（类内 NMS 保序，与引擎直接以该阈值构造严格等价）；SAHI 路径在切片合并**前**过滤，避免低分碎片先放大高分框几何。低于 0.1 返回错误而非静默截断 |
| `target_classes` | JSON 整数数组 `[0, 2, 5]`（class_id，非空） | 仅保留这些类别 |
| `roi` | JSON 归一化多边形 `[[x,y],...]`，≥3 点，坐标 ∈ [0,1] | 仅保留 **bbox 中心点**落在多边形内的检测（射线法，支持凹多边形） |

`/detect/batch` 的 `model` 参数在动态模型管理（吸收计划阶段 2）落地前传入即拒绝，不静默忽略。NMS IoU 全局固定 0.45（NMS 在引擎内完成，不做每请求覆盖）。

## 构建与运行

```bash
cargo build --release --jobs 6
```

运行时需 ORT 动态库（`load-dynamic`）与模型文件，环境变量见下。

版本标识由 build.rs 编译期探测注入（git hash + dirty 标记），见 `/health` 的 `version` 字段。
`SIGTERM`/`Ctrl+C` 触发优雅停机：等待在途请求完成后退出（`VISION_EXT_SHUTDOWN_TIMEOUT_SECS` 超时保底强退，默认 10s）。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ORT_DYLIB_PATH` | — | libonnxruntime.so 路径（必需） |
| `VISION_EXT_PORT` | 7042 | 监听端口 |
| `VISION_EXT_DEVICE` | cpu | cpu / cuda |
| `VISION_EXT_AUTH_TOKEN` | 无 | 设置后启用 API key（`X-API-Key` 或 `Authorization: Bearer`） |
| `VISION_EXT_MAX_BODY_MB` | 20 | 请求体上限 |
| `VISION_EXT_MAX_CONCURRENT_INFERENCE` | 0 | 并发推理上限（0=不限） |
| `VISION_EXT_ENGINE_POOL` | 1 | 每引擎实例池大小。>1 时同模型多实例并行推理（多路客户端同模型并发不再串行）。**代价**：内存 = 模型大小 × 池大小 × 已加载引擎数；CPU 预算 = 池大小 × 4（ORT 线程）× 引擎数 ≈ 核数。与多实例部署可叠加（实例数 × 池大小统算） |
| `VISION_EXT_<CAP>_WARMUP_SIZE` | 沿用引擎 input_size | 预热 dummy 输入尺寸覆盖，`WxH` 格式（如 `VISION_EXT_OCR_WARMUP_SIZE=320x320`）。`<CAP>` 为能力键大写：`DETECTION`/`CLASSIFICATION`/`OBB`/`FACE_DETECTION`/`FACE_RECOGNITION`/`OCR`/`LICENSE_PLATE`/`QR`/`POSE`/`DEPTH`/`HUMAN_PARSING`（注意与模型路径 env 的 `FACE_DET`/`PLATE_DET` 缩写不同）。OCR 必设：960×960 全零 dummy 触发 det 大量伪区域 + rec 批前向，预热 20.4s/实例；320×320 实测 0.43s/实例且首个真实请求推理耗时不变（2026-09-11 实测，池 4：预热 81.5s→1.8s，冷启动首请求 91.3s→11.7s）。CPU 设备上预热对首次真实推理无可见收益（关闭预热实测首推理 6.5s ≈ 热态），保留仅消除潜在首次抖动 |
| `VISION_EXT_SHUTDOWN_TIMEOUT_SECS` | 10 | 优雅停机超时：SIGTERM 后等待在途请求的上限，超时强退。需覆盖最长推理（SAHI/depth ~1.5s，池化排队翻倍） |
| `VISION_EXT_GPU_MEM_LIMIT_MB` | 0 | CUDA EP 显存上限（MB）。>0 时设置 ORT arena `with_memory_limit`，防 arena 无限膨胀导致 cuBLAS OOM 后永久失败（僵尸态）。0 = 不限（ORT 默认） |
| `VISION_EXT_ACQUIRE_TIMEOUT_MS` | 30000 | 推理排队等待超时（ms）。全忙时请求等待引擎池 permit 的上限，超时返回 503。**0 = fail-fast**：不排队，全忙立即 503 |
| `VISION_EXT_*_MODEL` | — | 各引擎模型路径（`DETECTION` / `CLASSIFICATION` / `OBB` / `FACE_DET` / `FACE_REC` / `OCR_DET` / `OCR_REC` / `OCR_DICT` / `PLATE_DET` / `PLATE_REC` / `QR` / `POSE` / `DEPTH` / `HUMAN_PARSING`） |

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
