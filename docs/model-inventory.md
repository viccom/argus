# argus 实测模型清单

> 收录范围：**在 argus 服务上实测过的每一个模型**（现役 + 验证过但未启用 + 验证不适用/被排除），逐个登记。
> 维护规则：新模型完成前置门/入槽/E2E 任一环节即回填一行；实测数字带时点；淘汰/排除件也必须留痕（防止重复踩坑）。
> 数据可信度分级：★ 生产在用 · ✅ 已验证可用（含动态槽实测）· ⚠️ 验证但受限/未启用 · ❌ 验证不适用（排除存档）。
> 配套文档：接入 SOP 与踩坑判别法见 `e2e-and-models.md`；候选调研见 `model-download-candidates-20261021.md`。

## 一、现役模型（静态能力槽，生产 :7042/:7043 在用）★

| # | 模型文件 | 能力/端点 | 大小 | sha256(前12) | 架构/类别 | 来源与导出方式 | 关键实测（时点） |
|---|---|---|---|---|---|---|---|
| 1 | yolov8n.onnx | detection `/detect` | 12.8MB | d6795a0eea3a | YOLOv8n，COCO 80 类 | Ultralytics 官方导出 imgsz=640。**禁用 Xenova 变体**（归一化坐标坑，见 §四） | 生产基线：bus.jpg count=5，top1 person 0.895，热态 57-61ms（2026-10-21） |
| 2 | ppe_site_10cls.onnx | ppe `/ppe` | 12.3MB | c345aca74d4a | YOLOv8 微调，10 类（Hardhat/Mask/NO-Hardhat/NO-Mask/NO-Safety Vest/Person/Safety Cone/Safety Vest/machinery/vehicle，含违章反类） | HF [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) best.pt 本地导出 | 工地实拍 8-25 检出，类别与 python 参照逐项一致（2026-09-21）；**击败 14 类比选件**（见 §三 #22） |
| 3 | fire_smoke_2cls.onnx | fire_smoke `/fire-smoke` | 12.3MB | 7df58b06861a | YOLOv8n 微调，fire/smoke 2 类 | HF [mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8) 本地导出 | 火灾正样本 fire 0.618 / bus.jpg 0 检出（2026-09-21）；置信度中等（0.52-0.62，nano 级），README 标注建议换代比选 |
| 4 | yolov8n-cls.onnx | classification `/classify` | 10.9MB | 3cb2bfb1b229 | YOLOv8n-cls，ImageNet 1000 类 | Ultralytics v8.3.0 官方导出 opset=12 imgsz=224，softmax 已烘焙 | bus.jpg top1 minibus 0.581（python 参照 0.583）（2026-09-10）；多主体场景图 top1<0.5 返回空为文档化语义非故障（2026-09 记录） |
| 5 | yolov8n-obb.onnx | obb `/detect/obb` | 12.5MB | a35496d54f28 | YOLOv8n-obb，DOTA 类 | Ultralytics 官方导出 | bus.jpg 0 检出为正确行为（无旋转目标），链路验证通过（2026-09-21） |
| 6 | yolov8m-seg.onnx | segmentation `/segmentation` | 109.4MB | 55c04dbb79b0 | YOLOv8m-seg，COCO 80 类实例分割（`[1,116,8400]`+proto） | Ultralytics v8.3.0 官方导出 opset=12 | bus.jpg 6 实例（bus 0.93 面积 31% + 4 person），掩码预览 63×64（2026-09-21） |
| 7 | face_detection_yunet_2023mar.onnx | face_detection `/face/detect` | 0.23MB | 8f2383e4dd3c | YuNet 5 关键点 | HF [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) 官方 | 生产在用（2026-09 接入）；orientation=6 照片摆正后 38 脸（EXIF 实测） |
| 8 | face_recognition_sface_2021dec.onnx | face_recognition `/face/recognize` | 38.7MB | 0ba9fbfa01b5 | SFace 128 维嵌入 | HF [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface) 官方 | 生产在用（2026-09 接入） |
| 9 | osnet_x1_0.onnx | reid `/reid` | 9.9MB | f3a28f0af6fb | OSNet x1.0，512 维 L2 归一化嵌入 | Qualcomm AI Hub v0.62.2 官方导出（**zip 外置权重需 onnx.save 内联**，见 README） | ppe_test.png 双人框 dim=512 L2=1.0、同人 cos=1.0/跨人 0.691（2026-09-21） |
| 10 | ppocrv5_det.onnx + ppocrv5_rec.onnx + ppocrv5_dict.txt | ocr `/ocr` | 88+84.5MB | 61824840edf6 / 26fa4f47060f | PP-OCRv5 检测+识别（中文） | HF [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx) | 生产在用（2026-09 接入）；OCR 预热 320 小图 0.43s/实例 vs 960 全零图 20.4s（2026-09-11 实测，warmup_size 起源） |
| 11 | mnet_plate.onnx + Final_LPRNet_model.onnx | license_plate `/license-plate` | 1.7+1.8MB | b4a208f937ad / 63753cfed58d | MNet 检测 + LPRNet 识别 | github [hpc203/license-plate-detect-recoginition-opencv](https://github.com/hpc203/license-plate-detect-recoginition-opencv) raw 直链 | 多车牌拼图实测：det 阈值 0.5→1 个牌、**0.15→6 个**（cheapai-1，2026-09 记录，构造期低阈值起源）；bus.jpg 0 检出为正确行为 |
| 12 | wechat_qr_detect.onnx | qr `/qr` | 1.0MB | e2c6bb7e207d | WeChat QR SSD 检测 | HF opencv/opencv_contribution **经图手术**（输入 384→416、重算 11 个 flatten 常量、裁 PriorBox/DetectionOutput 尾层，见 README） | 生产在用（2026-09 接入）；原版无法接入的图手术过程见 `e2e-and-models.md` §2.3 |
| 13 | yolov8n-pose.onnx | pose `/pose` | 13.5MB | 04f6d2416266 | YOLOv8n-pose，COCO 17 关键点 | HF [Xenova/yolov8n-pose](https://huggingface.co/Xenova/yolov8n-pose)（**Xenova 系中少数可用的例外**：pose 任务无坐标反算问题） | 摔倒检测链路验证（pose_rules 输入源）；已知边界：俯卧姿态不检出（2026-09-27） |
| 14 | depth_anything_v2_small.onnx | depth `/depth` | 99.1MB | afb6a5c28f3b | Depth Anything V2 Small | HF [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) | 生产在用（2026-09 接入）；单推理 ~1.5s 级（优雅停机超时取值依据之一） |
| 15 | segformer_b2_clothes.onnx | human_parsing `/human-parsing` | 110MB | a93a8dac171b | SegFormer-B2，ATR 18 类服饰部件 | HF [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) | 生产在用（2026-09 接入）；内存大户（与 OCR 合计占全量 83%，池=1 治理依据） |

## 二、已验证可用、未启用（动态槽候选，模型文件已存 models/）✅

| # | 模型文件 | 任务/类别 | 大小 | sha256(前12) | 来源 | 关键实测（2026-10-21，生产 :7042 热加载 A/B，bus.jpg） | 状态说明 |
|---|---|---|---|---|---|---|---|
| 16 | **yolov8s.onnx** | 检测，COCO 80 | 44.9MB | 211cb4e1f537 | github ultralytics/assets v8.4.0 + venv 导出 | count=5，top1 bus 0.920，两批热态均值 54-104ms（批次波动大，共享机负载）；官方参照一致 | **detection 默认升档首选**：精度高于 v8n；切默认需改 ARGUS_DETECTION_MODEL+重启 |
| 17 | **yolo11s.onnx** | 检测，COCO 80 | 38.1MB | cbdf4ed84a3a | **魔搭** `AI-ModelScope/YOLO11`（官方镜像）+ venv 导出 | count=5，top1 **bus 0.946**（三方最高；官方参照 0.937 同族），105-57ms | 精度王候选；来源证明魔搭官方镜像可信 |
| 18 | **yolo26n.onnx** | 检测，COCO 80，NMS-free e2e | 9.9MB | 4fd0c3d74b65 | github ultralytics/assets v8.4.0 + venv 导出 | count=5，top1 bus 0.908，**39ms（全场最快，同批内）** | 速度王候选；e2e 布局与传统同形，引擎兼容实证 |

> §二选型口径：**同批次内横向对比才有效**——两批次间延迟波动 40-90ms（共享机后台编译干扰）；精度看 top1 类别差异（三个模型对 bus.jpg 最优检出类别本身不同：v8n=person、v8s/11s/26n=bus），正式选型需按业务图集。
> 磁盘位置：`models/`（onnx 已就位，入槽只差 slots.toml 一段）；`.pt` 原件与导出环境在 `models-dl/`（326MB）。

## 三、验证过但受限/被排除（留痕防重复踩坑）

| # | 模型 | 大小 | sha256(前12) | 验证结论（时点） | 排除/受限原因 | 处置 |
|---|---|---|---|---|---|---|
| 19 | yoloe-26n-seg-pf.onnx | 15.7MB | 8100a38ff740 | 前置门通过：`[1,4621,8400]`+proto，值域正常（2026-10-21） | **4585 类 LVIS 词汇表 + segment 任务**——掩码输出应走 `/segmentation` 契约（上游 seg 引擎），检测族槽不适用；且 ONNX 导出为静态检测器（prompt 烘焙，换词汇表=重新导出，ultralytics issue #25530） | onnx+pt 存 models-dl/；走 seg 契约的评估挂后续 |
| 20 | yolov8s-cls.onnx | 25.5MB | 93b2bef743c8 | 前置门通过：`[1,1000]` softmax（2026-10-21） | 分类非检测族，无动态槽；仅作静态 classification 升档候选（n→s 精度 69→73.8） | onnx+pt 存 models-dl/；与 #4 同能力，切换需改 env+重启 |
| 21 | Xenova/yolo-rs yolov8n 变体 | — | — | 2026-09 判别：bbox 零宽、类名数字、SAHI count=0 | 检测头输出**归一化坐标**（cx∈[0,1]）且无类名元数据，与 640 像素空间反算解码器不兼容 | **纪律性禁入**：官方重导出替代；判别法=python 直跑看 cx 值域（踩坑表 #122） |
| 22 | jashwanthpeddisetty0712 yolov8m-ppe-14cls | — | — | 前置门过（`[1,18,8400]`）但实拍全面落败：工地图 Person 召回 0（现役 10 类同图 25 检出含 Person×9）、摔倒图零检出、.pt 原生推理复现（排除导出损失）、CPU 566ms 慢一倍（2026-09-27） | README mAP50=0.79 与实拍泛化严重不符 | **不换**，现役 ppe_site_10cls 保留；升级路线转自训 SH17 |
| 23 | melihuzunoglu yolov11-ppe（4-5 类） | — | — | 纸面评估淘汰（2026-09-27） | 类别面不足：无 NO-Vest 直接类、无机械车辆 | 未下载 |
| 24 | qualcomm gear_guard_net | — | — | 布局核验（2026-09 PPE 选型） | 三输出自定义头（boxes/scores/class_idx），非 v8 标准布局，DetectionEngine 不可复用 | 弃用；教训：接入前先 dump 输出 shape |
| 25 | wechat_qr_detect 原版（384 输入） | — | — | 布局核验（2026-09） | OpenCV 版头尾结构（PriorBox+DetectionOutput）无法直接接入 | **经图手术改造后启用**（→§一 #12），手术过程存 `e2e-and-models.md` §2.3 |
| 26 | damoyolo_tinynasL25_S_safetyhelmet.pt | 128MB | — | ckpt 结构核验：纯 state_dict（OrderedDict），无模型定义不可直接前向（2026-10-21，魔搭 iic 官方） | **非 Ultralytics 生态**（DAMO-YOLO 自研 tinynas 架构），需自家推理栈重建+导出，超出"即下即用"范围 | pt 存 models-dl/；安全帽场景已有 ppe_site_10cls 覆盖，无近期需求 |

## 四、横向维度速查

- **坐标空间纪律**：检测族一律 640 像素空间（Xenova 归一化坐标禁入）；pose 是 Xenova 系例外（无反算）。
- **多模型能力**：ocr（det+rec+dict 三件套）、license_plate（det+rec）缺一不可——溯源拼接 `+`，缺半套不报溯源（防误导向量失效判定）。
- **图手术先例**：wechat_qr（输入尺寸/常量重算/裁尾）——非标 OpenCV 头模型的可救路径，成本约半天。
- **外置权重**：Qualcomm AI Hub 导出需内联（onnx.load(load_external_data=True)+onnx.save）。
- **延迟参考**（CPU 16 核共享 KVM，n 档单实例，热态）：检测 40-60ms、cls 224 档 ~13ms、SAHI 翻倍、depth ~1.5s；共享机负载会使批次间漂移 40-90ms，**对比只认同批次**。
- **魔搭渠道**（2026-10-21 开通）：CLI=`yolo-rs/.venv/bin/modelscope download`；可信仓= `AI-ModelScope/*`（官方镜像）、`iic/*`（达摩院官方）；文件表 API=`/api/v1/models/<repo>/repo/files?Revision=master&Recursive=true`。
- **准入纪律**：任何新模型过三道前置门（大小对照 → magic 0x08 → shape dump + ORT 直跑值域）才可入槽；检测族入槽走 slots.toml（动态），非检测族走 env+重启（静态）。

## 变更记录

- 2026-10-21：首版。汇总现役 15（含 2 组多模型件）、已验证未启用 3、受限/排除 8，共 26 项；实测数据回源自 README 模型表、engine-exposure-plan 验证列、e2e-and-models 踩坑表、两批下载实测（b62257e/af823e0）。
