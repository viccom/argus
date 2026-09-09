# vision-ext

工业视觉智能 HTTP 服务，基于 [rust-onnx-infer](https://github.com/javpower/rust-onnx-infer) 封装，提供 11 个图像推理 REST API。

## 能力

| 端点 | 说明 | 模型 |
|---|---|---|
| `POST /detect` | 目标检测（支持 SAHI 切片，`sahi=true`） | YOLO |
| `POST /detect/obb` | 旋转框检测（自动识别传统 / End2End 布局） | YOLOv8n-OBB |
| `POST /face/detect` | 人脸检测（bbox + 5 关键点） | YuNet |
| `POST /face/recognize` | 人脸识别（128 维嵌入 + 余弦相似度比对） | SFace |
| `POST /ocr` | OCR 文本检测 + 识别 | PP-OCRv5 |
| `POST /license-plate` | 车牌检测 + 识别（mnet 关键点矫正 + LPRNet） | mnet / LPRNet |
| `POST /qr` | 二维码检测 | WeChat QR |
| `POST /quality` | 图像质量评估（清晰度/亮度/对比度/噪声） | 无需模型 |
| `POST /pose` | 人体姿态（COCO 17 关键点） | YOLOv8n-Pose |
| `POST /depth` | 单目深度估计（min/max/mean + 64×64 预览） | Depth Anything V2 |
| `POST /human-parsing` | 人体解析（部件占比） | PP-HumanSeg |
| `GET /health?deep=true` | 健康检查（浅检返回已加载引擎；deep=true 额外返回已配置能力清单） | — |

所有模型按需懒加载（`tokio::OnceCell`），推理经 `spawn_blocking` + 信号量背压。

## 构建与运行

```bash
cargo build --release --jobs 6
```

运行时需 ORT 动态库（`load-dynamic`）与模型文件，环境变量见下。

版本标识由 build.rs 编译期探测注入（git hash + dirty 标记），见 `/health` 的 `version` 字段。
`SIGTERM`/`Ctrl+C` 触发优雅停机：等待在途请求完成后退出（60s 超时保底强退）。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ORT_DYLIB_PATH` | — | libonnxruntime.so 路径（必需） |
| `VISION_EXT_PORT` | 7042 | 监听端口 |
| `VISION_EXT_DEVICE` | cpu | cpu / cuda |
| `VISION_EXT_AUTH_TOKEN` | 无 | 设置后启用 API key（`X-API-Key` 或 `Authorization: Bearer`） |
| `VISION_EXT_MAX_BODY_MB` | 20 | 请求体上限 |
| `VISION_EXT_MAX_CONCURRENT_INFERENCE` | 0 | 并发推理上限（0=不限） |
| `VISION_EXT_*_MODEL` | — | 各引擎模型路径（`DETECTION` / `OBB` / `FACE_DET` / `FACE_REC` / `OCR_DET` / `OCR_REC` / `OCR_DICT` / `PLATE_DET` / `PLATE_REC` / `QR` / `POSE` / `DEPTH` / `HUMAN_PARSING`） |

## 模型下载

模型不入库。已验证的来源（均可直连或经 `hf-mirror.com` 镜像加速）：

| 文件 | 来源 |
|---|---|
| `yolov8n.onnx` | [yolo-rs release](https://github.com/klemen-002/yolo-rs/releases) 或 Ultralytics 导出 |
| `yolov8n-obb.onnx` | Ultralytics 导出：`YOLO('yolov8n-obb.pt').export(format='onnx', imgsz=640)` |
| `yolov8n-pose.onnx` | [Xenova/yolov8n-pose](https://huggingface.co/Xenova/yolov8n-pose)（`onnx/model.onnx`） |
| `face_detection_yunet_2023mar.onnx` | [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` | [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface) |
| `ppocrv5_det.onnx` / `ppocrv5_rec.onnx` / `ppocrv5_dict.txt` | [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)（`detection/v5/det.onnx`、`languages/chinese/rec.onnx`、`languages/chinese/dict.txt`） |
| `wechat_qr_detect.onnx` | [opencv/opencv_contribution](https://huggingface.co/opencv/opencv_contribution) `wechat_qr/detect_2026april.onnx` **需图手术**：输入 384→416、按 416 重算 11 个 flatten shape 常量、裁掉 PriorBox×5 + DetectionOutput 尾层、输出改为 `mbox_loc [1,32448]` + `mbox_conf_flatten [1,16224]` |
| `mnet_plate.onnx` / `Final_LPRNet_model.onnx` | [hpc203/license-plate-detect-recoginition-opencv](https://github.com/hpc203/license-plate-detect-recoginition-opencv) raw 直链 |
| `depth_anything_v2_small.onnx` | [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)（`onnx/model.onnx`） |
| `pp_humanseg_2023mar.onnx` | [opencv/human_segmentation_pphumanseg](https://huggingface.co/opencv/human_segmentation_pphumanseg) |

## 依赖说明

`rust-onnx-infer` 为本地 path 依赖（`../rust-onnx-infer`），本地相对上游有一处修改：
`ort` feature `api-27` → `api-26`。原因：ort 2.0.0-rc.13 的版本检查用 ORT 库的
release minor 与 `ORT_API_VERSION` 比较，而 ORT 1.18 起 API version = release minor + 1，
导致 ORT 1.26.0 动态库被误判 `BadVersion`；`api-26` 下 `ORT_API_VERSION=26` 与 1.26.0 匹配，
且 ORT 库向后兼容 api-26 接口。

## License

MIT
