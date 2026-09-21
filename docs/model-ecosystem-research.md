# 通用模型生态调研与吸纳评估(2026-09)

> 目的:评估 argus 还可以吸纳哪些**通用且优良**的模型,前提是**当前架构不需要大改动**(小改进可接受),按企业应用场景价值排序。
> 方法:本地资产盘点(上游 rust-onnx-infer 42 引擎 vs argus 已暴露 15 能力,逐引擎核对文档)+ 全网检索 2026-09 模型生态(Ultralytics YOLO26/YOLOE-26、HF 微调模型、SigLIP2、垂直场景模型)。
> 检索时点:**2026-09-21**(模型生态月度迭代,落地前应重新比选当期最优)。
> 结论去向:有价值的项已编排进 **`engine-exposure-plan.md` 批次表**(批 1.6 与批 6),本文是该编排的调研依据;两文档互链。

## 一、准入准则与架构发现

**准入准则**(四条同时满足才入表):

1. **契约**:单图 HTTP(multipart 进),输出为框/掩码/关键点/embedding/直方图之一——现有 G1(检测模板)/G2(掩码 DTO)/G3(检测后属性两段式)可直接吃;
2. **算力**:CPU nano/small 档可跑(16 核共享 KVM,单请求百 ms 级);
3. **获取**:标准 Ultralytics 导出或公开 ONNX 直下(下载通道 hf-mirror/github + venv 导出工具链已于 2026-09-21 实测可用);
4. **改动**:复用现有三形态之一,或新增一个可控的小契约点(半天到两天)。

**关键架构发现**(改变候选格局):

- 上游 `segmentation.rs` **原生支持 YOLOE 与 YOLO26 的标准布局导出**(传统/End2End NMS-free 自动识别,YOLOE 经 `OnnxEngineFactory#createYOLOEEngine` 复用该引擎)。这意味着:
  - **开放词汇检测不需要新契约**——文本提示类别在导出时烘焙进图(prompt-free 导出),服务端按普通检测/分割模型对待;
  - **模型换代(YOLOv8→YOLO26)不需要改代码**——End2End 布局由引擎按输出 shape 自动识别。
- `/quality`(无参考 IQA)**已上线且为纯算法**(`QualityAssessor`,无 ONNX 模型)——此前批 1.5 任务 20 将其误列为"待接入",已修正。
- AI Hub 类模型(Qualcomm)多为**自有头布局**(如 gear_guard_net 三输出 boxes/scores/class_idx),不能按标准 YOLO 布局接入——教训:接入前先 dump 输出 shape(2026-09-21 PPE 选型时踩过)。

## 二、表 1|上游引擎已备、argus 未暴露(补暴露即得,约 0.5-1 天/项)

| 能力(建议端点) | 形态/复用 | 模型 | CPU 首请求(参照同类实测) | 价值场景 |
|---|---|---|---|---|
| face_attribute `/face/attribute` | G3 两段式(复用 face_rec 模板) | PINTO zoo #070 / OMZ IR(引擎双布局兼容,已查实) | 参照 face_rec 档(~1s) | 门禁/考勤补全年龄性别表情 |
| hand_keypoint `/hand` | G1(同 pose) | MediaPipe palm + 21pt(PINTO #033) | 参照 pose(~0.4s) | 手势前置/精细作业监控 |
| pose_rt `/pose` | G1(同 pose 契约) | RTMO | 待测 | 多人姿态换现代模型 |
| wholebody `/pose` 扩展 | G1,前端骨架扩展 | 133 关键点模型 | 待测 | 全身+手+脸关键点 |
| face_liveness `/face/liveness` | G3 两段式 | 引擎配套模型 | 待测 | 防照片/翻拍打卡作弊 |
| face_landmark106 | G3 | 精细关键点 | 待测 | 人脸对齐/疲劳检测前置 |
| semantic_segmentation `/semantic-seg` | G2(class map → parsing 式 DTO) | SegFormer-B0 ADE20K(城市场景) | 参照 human_parsing(~5s,建议池=1) | 场景理解/区域分割 |
| portrait_matting `/matting` | G2(alpha → FloatMask,零新契约) | MODNet 系 | 待测 | 人像抠图/工牌照处理 |
| birefnet `/matting` | G2 同上 | BiRefNet(大,标 GPU 备选) | 偏重 | 高质量显著性抠图 |
| table_recognition `/table` | G1,DTO 较重(HTML+单元格框) | PaddleOCR SLANet | 待测 | 工单/报表结构化 |
| **pose_rules**(零模型纯算法) | G1 纯算法 | 无需模型(吃 pose 输出) | 忽略不计 | **摔倒/举手/越界姿态事件**——与 pose 组合即巡检事件源,与 safe_bot 高处作业违章直接对齐 |

注:image_enhance / deblur / real_esrgan / style_transfer 引擎虽在上游,但输出是**图像**而非结构化数据,需要新增"图像返回"契约(见表 3),故归入表 3。

## 三、表 2|全新模型 + 现有模板,新能力槽即插(半天/项;阶段 2 动态化落地后退化纯配置)

| 能力(建议端点) | 复用 | 候选模型(2025H2-2026 生态检索) | 价值场景 | 备注 |
|---|---|---|---|---|
| **YOLOE 开放词汇** `/detect` 换模型 | 上游 segmentation 引擎原生支持(标准布局) | YOLOE-26 prompt-free 导出(类别烘焙进图,2026-01 arXiv 2602.00168) | **免训练自定义类别检测**(设备类型/物料/特定工器具)——对定制项目是杠杆 | 文本提示在导出时烘焙,服务端零新契约 |
| ppe 模型升级 `/ppe` 换模型 | DetectionEngine | 14 类版(含 NO-Goggles/NO-Gloves/Ladder/Fall,jashwanthpeddisetty0712 yolov8m)、YOLOv11 微调版(melihuzunoglu/ppe-detection) | 违章覆盖面扩展 | 14 类版 precision 0.70 偏低,需实测比选 |
| crack/pothole `/infra-defect` | DetectionEngine | cazzz307/yolov8-crack-detection(nano,2025-11);YOLOv8 道路病害系列(2026 Crack-YOLO 等)论文充分 | 基建巡检:裂缝/坑洞/井盖 | 壁面/道路/混凝土 |
| barcode `/barcode` | DetectionEngine | Piero2411/YOLOV8s-Barcode-Detection(barcode+QR);2026 EA-OBB 有旋转框变体(PMC12868771) | 仓储物流定位(解码由消费方 crop→zxing) | 只做定位不做解码,零新依赖 |
| fire equipment `/ppe` 或新槽 | DetectionEngine | 消防器材在位/缺陷检测(ETASR 有训练先例;killuminati1/construction-ppe-yolov8 含安全违章类) | 消防合规巡检,与 /fire-smoke 互补 | 具体模型落地时比选 |
| smoking/calling `/behavior` | DetectionEngine | HF 有 yolov8 吸烟/打电话微调版(方向存在,repo 未逐一验证) | 动火区吸烟、作业打电话违章 | 与 safe_bot 动火违章直接对齐 |
| 人流计数(务实版) | **零新代码** | 现有 `/detect` + `target_classes=[0]` 已支持 | 高峰人流/区域限员(低密度) | 高密度场景失真→见表 3 密度图 |
| YOLO26 全线换代 | 现有引擎(End2End 自动识别) | ultralytics 官方 v8→26 系列(2025-09 发布,NMS-free) | 全端点免 NMS 后处理、精度代际提升 | CPU 上 NMS 省得不多,优先级让位于场景项 |

## 四、表 3|小改动可接受(每个 1-2 天,新增一个可控契约点)

| 能力 | 新契约点 | 候选模型 | 价值场景 | 改动说明 |
|---|---|---|---|---|
| **图像 embedding `/embed`** | reid 契约变体(通用化) | CLIP ViT-B/32 / SigLIP2 base(onnx-community 有导出,optimum 一键导出;SigLIP2 为 2025-02 发布的当前主流) | **图像检索/去重/相似查找/零样本分类**(文本 embedding 离线预计算,消费方余弦)——企业通用性最强的新能力 | fork 加 ~50 行通用 embed 引擎(OSNet 专属预处理泛化),或导出时归一化烘焙进图;CPU 延迟待实测(B/32 档预期百 ms 级;so400m-512 过重不取) |
| crowd density `/crowd` | G2 掩码 DTO(复用 depth 的 stats+preview) | LCDnet 等轻量密度回归(ONNX 现成度需调研) | 高密度人群计数(person 检出数失真的场景) | 密度图→均值池化预览+积分计数 |
| 图像返回系端点 | JSON 内 base64 图(新契约,~半天) | image_enhance(低光/去雾/去噪)、real_esrgan、deblur(NAFNet) | 暗光增强作**检测前置**;画质修复 | 上游引擎全有;sr/deblur 重,按需池=1 |
| 仪表读数 `/meter` | 组合式(检测→pose 关键点→角度换算) | gauge-ai-reader 路线(YOLOv8-pose 关键点+角度数学,GitHub 开源;MDPI 2024/2025 多篇) | **无人巡厂指针表读数**——工业价值高 | 关键点走现有 pose 引擎,角度换算是 pose_rules 式零模型数学;需按表盘微调模型 |
| 图对匹配 `/match` | 双 file 字段 | dedode_g + lightglue / lomar(上游引擎已有) | 巡检比对/安装核验/视觉定位 | 契约极小但场景偏窄,低优先 |

## 五、明确不建议(现在)

| 类别 | 原因 |
|---|---|
| grounding_dino / dart / grounded_sam / sam / sam2 | 文本或点提示契约 + 文本编码器/大视觉塔,CPU 延迟与内存越线,交互式契约与 G1-G3 范式不匹配——开放词汇需求用表 2 YOLOE 烘焙导出替代 |
| action_recognition / gesture | 多帧序列输入,与单图 HTTP 契约根本冲突(维持批 5"等需求"结论) |
| 任何 VLM 类 | 属 safe_bot 侧职责(帧级语义违章),argus 定位是便宜/快/稳的结构化视觉信号源,分工边界不动 |

## 六、推荐落地顺序(Top 5)

1. **pose_rules + face_attribute + hand_keypoint**(表 1 白捡项:摔倒事件、人脸族补全,零模型获取风险);
2. **YOLOE 烘焙导出**(表 2:一个模型解锁"任意自定义类别检测",对后续每个定制项目都省一次训练);
3. **crack + ppe 升级**(表 2:直接对齐 safe_bot 违章目录的确定性场景);
4. **CLIP/SigLIP2 `/embed`**(表 3:通用性最强的新能力,检索/去重/零样本三合一);
5. **图像增强前置**(表 3:暗光/雾天检测质量兜底,工业现场刚需)。

## 七、未验证声明(显式)

- 表 2 的 HF 模型**均未过本地前置门**(下载→magic→shape→ORT 直跑),存在 Xenova 变体/归一化坐标等已知坑的可能(判别法见 `e2e-and-models.md` 踩坑表);14 类 PPE precision 0.70 需实测;smoking/calling 与消防器材是"方向确认、repo 未验证"。
- SigLIP2/CLIP 的 CPU 延迟与内存未实测(SigLIP2 归一化参数 mean/std=[0.5,0.5,0.5],与 CLIP 不同,导出/接入时注意);so400m-512 档明显过重,建议 B/32 或 base 起步。
- crowd density 的 ONNX 现成模型成熟度一般,LCDnet 论文轻量但需调研是否有维护良好的导出。
- YOLOE-26 兼容性判断来自上游代码注释("同 YOLO 对应格式"),实测需走前置门。
- 仪表读数的开源实现(gauge-ai-reader)以 keypoint 检测 + 角度换算为主,公开预训练权重能否直接用于本厂表盘需验证,大概率要微调。

## 变更记录

- 2026-09-21:首版。基于当日本会话调研成文(本地盘点 + 两轮全网检索);结论编排进 `engine-exposure-plan.md`(批 1.6 场景即插槽 + 批 6 小契约能力,任务 23-34),并修正该表任务 20 状态(/quality 已上线纯算法)。
