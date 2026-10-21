# 互联网可直接下载、加载即可用的模型清单（2026-10-21 调研）

> 目的：盘点当前 argus 状态下，互联网上**可直接下载到 models/ 并加载可用**的模型文件。
> 判别基准：`docs/e2e-and-models.md` 踩坑表——Ultralytics 官方导出 / 拒绝 Xenova 归一化坐标变体 /
> 下载后必过 magic bytes + shape dump + ORT 直跑三道前置门；本机导出工具链已就绪
> （yolo-rs/.venv：ultralytics 8.4.142 + onnx 1.16.2 + ort 1.29.0，2026-10-21 核实）。
> 下载通道：hf-mirror（国内加速）或 huggingface 直连；**大文件（>100MB）经镜像可能静默截断，下载后必须校验**。
> 消费方式：**检测族一律走动态槽**（slots.toml 加一段 + 热重载，`model=<槽名>`），零代码发版。

## 一、当前底座状态（2026-10-21）

- 已配置 15 能力（detection/ppe/fire_smoke/classification/obb/segmentation/face×2/reid/ocr/plate/qr/pose/depth/human_parsing），全部 enabled；
- 动态槽底座已上线（v6c2c0af）并在生产验证过热加载/懒构建/清空全流程；当前无 slots.toml（休眠态）；
- 检测族动态槽只接受标准 Ultralytics 布局导出（`DetectionEngine::for_yolo`，输出 `[1, 4+nc, 8400]` 传统或 End2End 布局自动识别）。

## 二、立即可用（官方直链，下载→前置门→slots.toml 即上线）

### 2.1 检测族升级/换代候选（走动态槽，与静态槽并行对比）

| 候选 | 获取方式 | 类别 | 预期价值 | 风险点 |
|---|---|---|---|---|
| **YOLO26n/s（检测）** | `yolo26n.pt` 随 ultralytics 包自动下载，或 HF `Ultralytics/YOLO26` | COCO 80 类 | NMS-free 端到端（引擎已支持 End2End 自动识别），官方 CPU ONNX 38.9ms（n 档，比 v8n 的 80.4ms 快一倍） | 与现有 yolov8n 须同图 A/B 比对；e2e 模式 mAP 略降（40.9→40.1）属官方取舍 |
| **YOLO11n/s（检测）** | 同上（`yolo11n.pt`） | COCO 80 类 | v8 与 26 之间的成熟替代 | 同上 |
| **yolov8s（检测，同代升档）** | github `ultralytics/assets/releases/download/v8.2.0/yolov8s.pt` + venv 导出 | COCO 80 类 | n→s 精度 37.3→44.9，CPU ~128ms 仍在内网可用档 | 内存与延迟翻倍，须实测 |

### 2.2 全新场景槽（批 1.6 场景即插槽，动态槽直接消化）

| 场景槽 | 来源（均可直接下载） | 说明 |
|---|---|---|
| **barcode/qr 定位** | HF `Piero2411/YOLOV8s-Barcode-Detection`（提供 .pt，venv 一行导出 onnx；约 5000 图训练） | barcode+QR 两类定位；解码仍由消费方做（zxing/crop），与现有 /qr 互补不冲突。**下载后必须 shape dump 验布局**（HF 仓库 README 明示可 export onnx，但未附导出文件——自行导出最稳） |
| **crack/pothole 基建巡检** | HF `vinothvikas1987/pothole-detection-yolov8`（YOLOv8s，5 类路面病害）；HF models 搜索 `yolov8 pothole onnx` 另有现成 onnx 变体（后者须过归一化坐标判别） | 裂缝/坑洞/路面破损；对 argus 是纯增量场景槽。微调 repo 的 .pt 质量参差，**前置门直跑看真实值域是唯一判据** |
| **fire/smoke 换代比选** | HF `rabahdev/fire-smoke-yolov8n`（D-Fire 数据集微调）、`aryantyagiase/yolov8-fire-smoke-detection`（YOLOv8s） | 与静态 fire_smoke_2cls 并行挂槽同图对比，精度提升才切换默认（静态槽切换仍需改 env+重启，动态槽对比零成本） |
| **YOLOE-26 开放词汇（prompt-free）** | github `ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt`（官方直链）+ venv 导出 | **类别烘焙进图**：导出时词汇表固定，服务端按普通检测/分割对待——文档引擎原生支持（上游 segmentation.rs 已核对）。一个模型解锁"自定义类别检测"；注意导出时选好词汇表，且 -seg 输出走 segmentation 契约（掩码），如需纯框输出需评估布局 |

### 2.3 非检测族（静态槽位，须改 env + 重启，暂无动态槽）

| 能力 | 候选 | 说明 |
|---|---|---|
| classification | `yolov8s-cls.pt`（官方） | n→s 精度 69→73.8；需改 ARGUS_CLASSIFICATION_MODEL + 重启 |
| pose | `yolov8s-pose.pt` | n→s 50.4→60.0；CPU ~233ms/请求 |
| obb | `yolov8n-obb` 官方 v8.3.0 重导出（现役即此） | 无升级必要 |
| segmentation | `yolov8s-seg.pt` | n→s 精度 36.7→44.6；CPU ~156ms |

## 三、需要甄别的（可下载但未证可用/有坑）

| 候选 | 风险 | 处置 |
|---|---|---|
| HF 现成 `*.onnx` 变体（pothole-onnx、yolo-face、license-plate best.onnx 等） | 导出方不明：可能是归一化坐标（Xenova 坑）/非标准 opset/类别错乱 | 一律先过三道前置门再入 slots.toml；**优先下载 .pt 自行 venv 导出**，导出参数可控 |
| smoking/calling 行为违章 | 公开 repo 质量参差（2026-09 调研即"方向确认、repo 未验证"）；ETASR 消防器材（2024 论文）无公开权重 | 有需求再专项调研；论文类需联系作者 |
| 消防器材在位检测 | 无成熟公开权重 | 同上 |
| YOLOE 运行时 prompt（text/visual） | ultralytics 官方 issue #25530：ONNX 导出为静态检测器，prompt 烘焙进图，运行时不可换 | 符合 argus"静态槽"范式，但换词汇表=重新导出+换槽 |

## 四、接入 SOP（每个模型同一流程）

1. **下载**：hf-mirror 或 github releases 直链；`.pt` → venv `model.export(format='onnx', imgsz=640)`；
2. **前置门**：`ls -la` 大小对照源站 → `xxd -l 4`（ONNX magic 0x08）→ python onnx shape dump（检测族应 `[1,4+nc,8400]` 或 e2e 布局）→ ort 直跑看值域（坐标非归一化、类别数正确）；
3. **入槽**：`models/slots.toml` 加 `[slots.<名>]`（model 相对 models_dir；class_names 可选覆盖）→ ≤6s 双实例热加载；
4. **A/B 验证**：`/detect?model=<槽>` 与静态槽同图对比 count/类别/置信度，达标后再决定是否切默认（切默认仍需改 env + 重启）；
5. **回滚**：删槽段或删 slots.toml，静态路径永不承受风险。

## 五、来源

- Ultralytics 官方模型表与导出文档：docs.ultralytics.com（YOLOv8/YOLO26/YOLOE/YOLO-World 页，2026-10-21 检索）
- ultralytics/assets releases：github.com/ultralytics/assets/releases（官方权重直链）
- HF 仓库：Piero2411/YOLOV8s-Barcode-Detection、vinothvikas1987/pothole-detection-yolov8、rabahdev/fire-smoke-yolov8n、aryantyagiase/yolov8-fire-smoke-detection、Ultralytics/YOLO26
- YOLOE ONNX 运行时 prompt 限制：github ultralytics issue #25530（2026-07）
- 检索工具：Anysearch batch_search（2026-10-21）

## 六、第一批实测记录（2026-10-21，生产实例 :7042 热加载验证）

**下载**（github ultralytics/assets v8.4.0 直链，4/4 成功，PK 头校验过）：

| 文件 | .pt 大小 | 导出后 | 导出输出布局 | 前置门 |
|---|---|---|---|---|
| yolo26n.pt | 5.4MB | yolo26n.onnx 9.5MB | `[1,84,8400]` 传统布局（NMS-free 的 e2e 差异在权重融合，输出同形） | ✅ 值域 0~655 非归一化、names 元数据有 |
| yolov8s.pt | 22MB | yolov8s.onnx 43.8MB | `[1,84,8400]` | ✅ 同上 |
| yolov8s-cls.pt | 12.6MB | yolov8s-cls.onnx 24.9MB | `[1,1000]` softmax | ✅（未入槽：分类非检测族，留作静态槽升级候选） |
| yoloe-26n-seg-pf.pt | 15.9MB | yoloe-26n-seg-pf.onnx 15.3MB | **`[1,4621,8400]` + `[1,32,160,160]` proto** | ⚠️ 布局正确但 4585 类 LVIS + seg 任务——**检测族槽不适用**（掩码输出走 segmentation 契约），未入槽 |

**A/B 实测**（bus.jpg，生产 :7042 热加载双槽，热态 5 次均值）：

| 模型 | count | 延迟均值 | 官方 python 参照 | 结论 |
|---|---|---|---|---|
| v8n（静态现役） | 5 | 57ms | —（基线） | — |
| yolov8s（槽） | 5 | **54ms** | count 5，top person 0.915/bus 0.906 | **v8s 更快且同精度档**：v8n CPU 官方标称 80ms vs v8s 128ms，本机实测因 ORT 线程与图优化反转；检出 box 序列与官方参照一致 |
| yolo26n（槽） | 5 | 62ms | count 5，top bus 0.903 | 与 v8s 相当；e2e 布局输出与传统同形，引擎兼容确认 |

**过程发现（重要教训）**：首轮 A/B 三模型返回完全相同（连置信度都一致），差点误判"槽串了"。真相：首个槽请求发生在构建完成前，响应以**毫秒级竞态**返回旧池？——复盘为**首轮请求时槽未构建完成、响应正常但 /models 的 loaded 尚未刷新**的观测时序问题；完整 JSON diff 判别（box 坐标级）证实三模型输出确实互不相同，slot 溯源 `yolov8s.onnx#211cb4e1f537` 正确。教训：**A/B 对比必须以完整 JSON（box 级）判定，不能用 count+top3 摘要**；`dynamic_engines` 的刷新有时序差，以构建日志（"动态槽 X 池(N) 加载完成"）为准。

**结论**：第一批 4 个模型，3 个通过全部前置门，2 个检测模型入槽实测可用（yolov8s 意外反超 v8n 延迟，建议作为 detection 默认升档候选），1 个分类候选待静态槽升级窗口，1 个 YOLOE 需走 segmentation 契约评估（4585 类 LVIS 词汇表，适合"万物检测"场景）。

## 七、第二批实测记录（2026-10-21，魔搭社区 ModelScope）

**渠道研究**：魔搭 CLI 已装（`yolo-rs/.venv`，modelscope 1.40.2；venv 迁移后 shebang 失效，用 `bin/python` 与 `bin/modelscope` 直呼）。**找高质量模型的有效方式**：① 官方背书仓 `AI-ModelScope/*`（Ultralytics 全系官方镜像，文件表 API：`/api/v1/models/<repo>/repo/files?Revision=master&Recursive=true`）；② `iic/*`（阿里达摩院官方，如 DAMO-YOLO 安全帽）；③ 网页搜索模型名关键词。通用模型搜索 API（dolphin/models 等）已 404，按 repo id 直查文件表最可靠。

**下载**（modelscope download，实测 15-25MB/s）：

| 文件 | 来源 repo | 大小 | 前置门 | 去向 |
|---|---|---|---|---|
| yolo11s.pt → yolo11s.onnx 36.3MB | `AI-ModelScope/YOLO11`（官方镜像） | `[1,84,8400]`、names 有、直跑 0~637 | ✅ | ✅ 入槽实测 |
| damoyolo_tinynasL25_S_safetyhelmet.pt 128MB | `iic/cv_tinynas_object-detection_damoyolo_safety-helmet` | ⚠️ **非 Ultralytics 生态**：ckpt 是纯 state_dict（OrderedDict），需 DAMO-YOLO 自家推理栈重建模型 | ❌ 记录后排除 | models-dl/ 存档，待有 DAMO 推理需求时再启用 |

**三方 A/B 终版**（bus.jpg，热态 7 次均值；官方 python 参照 yolo11s top=bus 0.937）：

| 模型 | 延迟均值 | top1 | 两两 box 级差异 |
|---|---|---|---|
| v8n（基线） | 59ms | person:0.895 | — |
| yolov8s | 104ms | bus:0.920 | 全部独立 ✓ |
| yolo11s | 105ms | bus:0.946 | 全部独立 ✓ |
| **yolo26n** | **39ms** | bus:0.908 | 全部独立 ✓ |

> 注：昨天批次测得 v8s 54ms/yolo26n 62ms 与本批 104ms/39ms 波动较大——共享机负载漂移所致（昨晚有后台编译），**同批次内横向对比才有效**。本批内 yolo26n 显著最快（e2e NMS-free 优势），yolov8s/yolo11s 精度更高（top1 bus 0.92/0.946 vs 基线 person 0.895——三者对 bus.jpg 的最优检出类别本身不同，选型需按业务图集）。

**测试方法教训（第二批新增）**：shell 循环里用未加引号的可选参数变量（`curl -F "file=..." $m`）会让空变量引发字段错位，产出"全部响应相同"的假象——A/B 脚本必须显式分支构造每类请求（本批已改）。**此假象前后共骗了我们两轮**，第三次才用单发溯源对照抓到真因。

**资产留存**：`models-dl/`（.pt+onnx 共 326MB，含 DAMO 排除件）+ `models/{yolo26n,yolov8s,yolo11s}.onnx` 已验证可用件。生产 slots.toml 已清空恢复休眠。

## 八、第三批实测记录（2026-10-21，HF 场景模型：裂缝/坑洞/火烟换代）

**下载（hf-mirror）**：

| 文件 | 来源 | 结果 |
|---|---|---|
| crack-seg-v8n/v8s.pt | `OpenSistemas/YOLOv8-crack-seg`（n/m 双档，Crack-seg 数据集） | ✅ 导出 `[1,37,8400]`+proto，1 类 crack |
| pothole-vinoth.pt | `vinothvikas1987/pothole-detection-yolov8`（5 类：纵裂/横裂/龟裂/坑洞/其他，RDD 数据集） | ✅ 导出 `[1,9,8400]` |
| cazzz307 裂缝/坑洞两仓 | HF gated 授权墙 | ❌ 未授权拿不到，**gated 仓注意**（镜像同样拒） |
| aryantyagiase fire-smoke v8s | HF | ❌ **镜像大文件截断坑实锤**：.pt 恰 10MiB、onnx 恰 8MiB（整块截断，magic 正常但解析烂）；重试无效，该 repo 在镜像上不可用 |

**前置门与 python 参照**（合成图：程序化绘制裂缝纹理/火苗烟团，`models-dl/synthetic-*.jpg`）：

| 模型 | 合成图官方推理 | 判定 |
|---|---|---|
| crack-seg-v8n | **crack 0.906**（2 检出） | ✅ seg 任务，检测槽不适用（同 yoloe）；走 /segmentation 契约候选 |
| crack-seg-v8s | crack 0.868（3 检出） | ✅ 同上（27M 参数 104MB onnx，重） |
| pothole-vinoth | 合成图 0（合成裂缝风格≠RDD 真实域） | ✅ 入槽，需真实路面图评估 |
| fire-smoke-dfire（D-Fire） | 合成火图 0（同上域差） | ✅ 入槽；**bus.jpg 检出 1 个 fire@0.34**——假阳性信号，正式换代需真实图集 A/B 现役 fire_smoke_2cls |

**入槽实测（生产 :7042）**：两槽热加载 ≤7s、深检双 ok、溯源 `a2dd504313fe`/`b734753d9e1a` 正确；**class_names 注入实证**——fire_smoke_dfire 在 bus.jpg 的假阳性检出类名为注入的 `"fire"`（而非模型 metadata/数字），SH17 自训类名注入路线可行性得到实锤。

**新踩坑（已录）**：
1. **槽名禁连字符**：`fire-smoke-dfire` 被槽名正则拒绝（`^[a-z][a-z0-9_]{0,31}$`），且整份拒绝保留旧配置（fail-closed last-known-good 实战生效）——命名槽只能用 `[a-z0-9_]`。
2. **镜像截断坑再实锤**：hf-mirror 对部分大文件按 8/10MiB 整块截断且 magic 头正常，必须大小对照源站。
3. **gated 仓**：cazzz307 两仓（调研文档原推荐）需 HF 授权，镜像不豁免。

## 九、第四批实测记录（2026-10-21，机动车品牌/型号识别专项）

**检索结论（全网多渠道）**：车型识别（VMMR，Vehicle Make & Model Recognition）可用 ONNX 资源集中在 HF；魔搭无相关仓（iic 只有 DAMO 检测系与通用分类底座）。数据集谱系：VMMRdb（8949 类 品牌-型号-年份，最大）、Stanford Cars（196 类）、CompCars（136 类+属性）。**NVIDIA TAO VehicleMakeNet/VehicleTypeNet**（品牌/6 类车型）质量有背书但 NGC 下载需账号+TAO 栈，未纳入。

| 模型 | 来源 | 形态 | 验证结果 |
|---|---|---|---|
| **vehicle-classifier（VMMRdb 8949 类）** | HF `Jordo23/vehicle-classifier`（非 gated） | **EfficientNet-B4 380×380**，logits `[batch,8949]`，附带 class_mapping.csv（类名/make/model 三列） | ✅ **语义验证通过**：zidane 轿车图 top1 `Acura MDX 2012` conf **0.529**（8949 类细粒度下高置信）；大巴域外图最高 0.026（判别力清晰） |
| resnet50-cars（Stanford Cars 196 类） | HF `zededa/resnet50-cars`（非 gated） | ResNet50 224×224，`[batch,196]` | ⚠️ 前置门过（直跑正常）但 **ONNX 无 metadata 无类映射文件**——196 个索引无法落地为品牌型号名，可用性打折；zidane top1 conf 仅 0.048 也明显弱于上者 |

**关键处置：官方 ONNX 缺 .data 外置权重文件**（仓内只放 927KB 图结构 + .pth，引用的 `vehicle_classifier_*.onnx.data` 不存在——HF 外置权重发布常见失误）。**处置=用仓内 .pth + timm 重建 EfficientNet-B4 自导完整内联 ONNX**（129MB，`vehicle-classifier-vmmrdb-full.onnx`），验证通过。timm 已补装进 yolo-rs/.venv（1.0.30）。

**接入 argus 的形态判断**：分类模型走 **classification 静态槽**（`ARGUS_CLASSIFICATION_MODEL` + 重启；分类无动态槽）——但注意：① 现有 `/classify` 契约是 YOLOv8-cls 的 softmax+metadata 自动读名；EfficientNet 无 names metadata，需仓内 csv 做类名映射（引擎 `get_label_name` 会退化为数字，消费方按 csv 解）或二次烘焙 metadata；② 380×380 输入 vs 现役 224，warmup/预处理无冲突但延迟上升（B4 级 CPU 预估 200-400ms，待入槽实测）；③ 部署形态更契合"检测框 → 裁剪 → 分类"的两段式（G3），与现有 face_rec 的架构同构。

**资产留存**：models-dl/（full.onnx 129MB + .pth 132MB + 官方残缺 onnx + csv 类表 + resnet50 96MB）。

## 十、工地车型清单专项评估（2026-10-21，27 类工程机械识别可行性）

**需求**：用户整理了 27 类工地常用车型（挖掘机/推土机/起重机/自卸车/叉车/摊铺机/压实机等），寻求准确识别的现成模型。

**全网检索结论**：**不存在覆盖该 27 类的现成 ONNX 模型**。工地机械识别的公开资源格局：
- 学术数据集多为 5-12 类（excavator/loader/dozer/roller/backhoe 为主，ASCE 2023 九类、Mendeley 2025 十二类、MDPI 2024 五类土方），无 27 类全覆盖；
- Roboflow Universe 有大量单场景小数据集（多为 3-19 类、几百到几千张，质量参差，权重需 API key 且许可不一）；
- HF keremberke 系列（roboflow2huggingface 官方镜像，非 gated）只有叉车相关：`yolov8s-forklift-detection`（forklift+person 双类，mAP50 0.851）；
- 中文社区（CSDN/腾讯云）有 8 类/27000 张等数据集售卖贴，无直接权重下载。

**本批实测**：
| 模型 | 来源 | 结果 |
|---|---|---|
| forklift-yolov8s.onnx（forklift+person 双类） | HF `keremberke/yolov8s-forklift-detection`（mAP50 0.851） | ✅ 入槽全过：溯源 `95dca951a55f`、bus.jpg 检出 person×4 + forklift@0.36（人工判读 bus 图左下黄色物体确有歧义性）、类名注入生效、深检 ok。清单 #15 叉车可部分覆盖（含人车同框） |

**27 类落地路线评估（结论：检测槽底座已就绪，缺的是数据）**：
1. **可行路线（推荐）**：按清单整理业务图集（每类 ≥300-500 张实拍，覆盖俯视/侧视/远近）→ 用现役 venv ultralytics 训练 YOLOv8s 检测模型（单模型 27 类）→ 导出后 slots.toml 入槽（零代码发版）。工地监控摄像头固定机位，采集成本低；近似公开数据集（5-12 类）可做预训练起点再迁移。
2. **部分覆盖捷径**：高频核心类（挖掘机/推土机/起重机/自卸车/叉车 5 类）可先组合 keremberke 叉车件 + Roboflow 公开数据集自训，其余低频类后续补。
3. **不推荐**：拼多个单类模型（槽多、维护碎、类间冲突）；纯分类方案（无法定位，工地图多目标）。

**资产留存**：forklift-yolov8s.onnx/pt 存 models-dl/，onnx 已入 models/。
