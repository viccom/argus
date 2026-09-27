# A 类引擎暴露计划·任务跟踪表

> 目的:将 rust-onnx-infer 的 A 类引擎(结构化输出、HTTP 友好)暴露为 argus 端点。
> 本表是唯一进度事实源,每完成一项即更新状态与验证证据,不等记忆。
> 创建:2026-09-10(基于 2026-09-09 会话的工作量评估)

## 状态图例

- ⬜ 待开始 · 🚧 进行中 · ✅ 完成(须附验证证据)· ⏸ 受阻/跳过(须附原因)

## 统一流程:每个引擎过三道门

1. **模型验证前置门**(写任何接入代码之前):下载 → magic bytes 校验(`xxd -l 4`,ONNX 应为 `0x08`)→ shape dump(python onnx)→ **python onnxruntime 直跑看输出值域**(detect 归一化坐标坑的制度化预防,判别法见 `e2e-and-models.md` 踩坑表);
2. **TDD 接入门**:错误路径(未配置/缺 file/坏图)+ DTO 序列化测试先红后绿(不依赖模型文件,秒级运行);
3. **E2E 冒烟门**:curl 真实推理 + 坐标交叉验证(与同类引擎输出对照)+ 前端卡片渲染验证 + 服务日志核对。

## 任务表

### 批 1:模板 + 高价值(目标 1.5 个工作日,含模型获取)

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 1 | classification 纵切片模板(后续引擎按此复制) | G1 同构 | yolov8n-cls.onnx(Ultralytics v8.3.0 .pt → onnx opset12/224 官方导出) | ✅ | 前置门:magic 0x08/shape [1,3,224,224]×[1,1000]/ORT 直跑 softmax sum=1.0/bus.jpg top1=minibus 0.576;TDD:config+engine+routes 3 轮先红后绿;E2E:curl /classify top1=minibus 0.581(与 python 参照 0.583 一致),top_k=2/错误路径全过;附带修复上游 resize 混叠 bug(见 2026-09-10 变更记录);测试 73→80 全绿 | 2026-09-10 |
| 2 | reid 行人重识别(OSNet 512 维) | G1,embedding 契约同 face_rec | Qualcomm AI Hub `qualcomm/OSNet` v0.62.2(外置权重需内联,见 README 模型表) | ✅ | 前置门:magic 0x08/签名与引擎文档逐项一致(`image [1,3,256,128]`→`embeddings [1,512]`)/ORT 直跑随机输入 L2=1.0000;TDD:路由 9 测试(缺 file/坏图/未配置/boxes 坏 JSON/坐标数错/退化框/clamp 行为/DTO)先红后绿,`cargo test --jobs 6` 154+1 全绿,clippy 零警告;E2E(ad-hoc :7099 debug 实例):ppe_test.png 双人框 count=2/dim=512/两 embedding L2=1.0000/278ms,同一人框模式 vs 整图模式 cos=1.0000(契约一致),跨人 cos=0.691(两人同装束,合理);错误路径 200 信封 `data.error` 提示 boxes 解析细节 | 2026-09-21 |
| 3 | segmentation 实例分割 | G2 mask,DTO 复用 depth 下采样先例 | `yolov8m-seg.onnx`(已入 models/) | ✅ | 前置门:标准布局 `[1,116,8400]`+proto `[1,32,160,160]`/bus.jpg 直跑 person/bus 值域正常(上批完成);TDD:路由 5 测试(缺 file/坏图/未配置/带掩码 DTO 序列化含 area_ratio=4/16 断言/无掩码 DTO),det 抽 `run_detect` 共享后 165+1 全绿、clippy 零警告;E2E(ad-hoc):bus.jpg 6 实例(bus 0.93/面积 31%,4 person,tie),掩码预览 63 列×64 行,`data.segments[]` 契约 | 2026-09-21 |
| 4 | face_attribute 年龄/性别/表情 | G3 组合式,复用 face_recognize 两段式模板 | 待获取 | ⏸ | 2026-09-27 挂起:年龄识别涉 PIPL 人脸敏感个人信息合规红线(safebot 侧判定,见 safebot-go docs/assistant/04 §3.9),表情/性别对安监价值低;待需求重估再启 | |

### 批 1.5:场景批(2026-09-21 插入,按企业价值排序)

> 背景:GPU(吸收计划阶段 4)暂缓;批 1 模型获取的网络阻塞已解除(2026-09-21 实测 hf-mirror/github 均 200,venv 具备 torch 2.14 + ultralytics 8.4 本地导出工具链)。按 safe_bot 隐患目录(10 类违章)对齐,PPE/火烟是高频刚需:两者**复用 DetectionEngine**(G1 模板,新能力槽位),无需等阶段 2 动态化。
> 排序依据:与 safe_bot `hazard_catalog.json` 首类"个人防护用品违章"对齐(PPE)、动火违章前置(火烟);IQA/增强/姿态规则为上游存量 CPU 友好资产,直击工地摄像头画质与跌倒场景。

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 18 | ppe_detect 安全帽/反光衣检测(新能力槽,复用 DetectionEngine) | G1 | 已选定:Hansung-Cho/yolov8-ppe-detection(10 类含违章反类),`ppe_site_10cls.onnx` 已入 models/;qualcomm gear_guard_net 为三输出自定义头(boxes/scores/class_idx,非 v8 布局),复用 DetectionEngine 不可行,弃用 | ✅ | 前置门(上批):标准布局 `[1,14,8400]`/真实工地照 8 检出;接入:`/ppe` 经 `DetectCap::Ppe` 复用 run_detect(参数校验先于引擎加载),E2E 8 检出与 python 参照类别逐项一致(Hardhat×2/NO-Mask×2/Person×2/Safety Vest×2),269ms(debug);5 个路由测试 | 2026-09-21 |
| 19 | fire_smoke_detect 火焰/烟雾检测(同上) | G1 | mfranzon/fire-smoke-yolov8(nano 级),`fire_smoke_2cls.onnx` 已入 models/ | ✅(生产采用前建议比选更强模型) | 接入:`/fire-smoke` 同上;E2E 正样本(房屋火灾)fire 0.618/负样本(bus.jpg)0 检出,1.6s(debug);5 个路由测试 | 2026-09-21 |
| 20 | image_quality 无参考画质评估(IQA) | G1 | 无需模型(纯算法) | ✅ | 状态核实(2026-09-21):`/quality` + `QualityAssessor` 纯算法评估器已上线(`routes/analysis.rs`,无 ONNX 模型),此前误列"待接入",修正 | 2026-09-21 |
| 21 | image_enhance 低光增强/去雾(检测前置) | G2 同尺寸 | 上游引擎已有;**前置依赖批 6 任务 32(图像返回契约)** | ⬜ | | |
| 22 | pose_rules 姿态几何规则(零模型,摔倒/举手) | G1 纯算法 | 无需模型(输入 pose 关键点);落地顺序第 1 位(研究 §六) | ⬜ | | |

### 批 1.6:场景即插槽(2026-09-21 研究新增,调研依据见 `model-ecosystem-research.md` 表 2)

> 每项 = 模型前置门 + 新能力槽(或换模型),约半天/个;阶段 2 动态化落地后新槽退化为"模型文件 + 配置"。模型**均未过前置门**,开工严格走三道门;踩坑判别法见 `e2e-and-models.md`。

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 23 | YOLOE 开放词汇检测(prompt-free 类别烘焙导出) | G1,复用 segmentation 引擎(上游原生支持标准布局) | YOLOE-26 ultralytics 官方导出 | ⬜ | | |
| 24 | crack/pothole 基建缺陷检测(新能力槽) | G1 | cazzz307/yolov8-crack-detection(nano)等,落地时比选 | ⬜ | | |
| 25 | ppe 模型升级比选(14 类含 NO-Goggles/Ladder / yolov11 版) | G1 换模型 | jashwanthpeddisetty0712(14 类)/ melihuzunoglu(v11) | ⬜ | | |
| 26 | barcode 条码定位(新能力槽,只定位不解码) | G1 | Piero2411/YOLOV8s-Barcode-Detection | ⬜ | | |
| 27 | fire equipment 消防器材检测(新能力槽,与 /fire-smoke 互补) | G1 | HF 微调版,落地时比选 | ⬜ | | |
| 28 | smoking/calling 行为违章检测(新能力槽) | G1 | HF 微调版(repo 未验证,先确认存在性) | ⬜ | | |
| 29 | YOLO26 全线换代评估(检测/分割/姿态 nano 档对照 v8 基线) | 换模型 | ultralytics 官方(End2End 引擎已支持) | ⬜ | | |

### 批 2:姿态/手部/人脸族

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 5 | hand_keypoint 手部 21 关键点 | G1,同 pose 模式 | 待获取 | ⬜ | | |
| 6 | pose_rt RTMO 多人姿态 | G1,同 /pose 契约 | 待获取 | ⬜ | | |
| 7 | wholebody 133 关键点 | G1,前端骨架连线需扩展 | 待获取 | ⬜ | | |
| 8 | face_liveness 活体检测 | G3 组合式 | 待获取 | ⬜ | | |

### 批 3:分割/文档族

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 9 | semantic_segmentation 150 类语义掩码 | G2,前端需图例 | 待获取 | ⬜ | | |
| 10 | portrait_matting 人像抠图 alpha | G2 | 待获取 | ⬜ | | |
| 11 | table_recognition 表格结构 | G1,DTO 较重 | 待获取 | ⬜ | | |

### 批 4(缓):低频

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 12 | birefnet 显著性抠图 | G2 | 待获取 | ⬜ | | |
| 13 | face_landmark106 精细关键点 | 待查 API 形态再定组 | 待获取 | ⬜ | | |

### 批 5(待契约):暂缓

| ID | 任务 | 阻塞点 | 状态 |
|---|---|---|---|
| 14 | action_recognition(ST-GCN 动作) | 输入为关键点序列,与单图 HTTP 契约不匹配,需先定义多帧上传格式 | ⬜(等需求) |
| 15 | gesture 静态手势 | 依赖 hand_keypoint 输出,同上 | ⬜(等需求) |

### 批 6:小契约能力(2026-09-21 研究新增,调研依据见 `model-ecosystem-research.md` 表 3)

> 每项新增一个可控契约点(1-2 天);开工前先读研究文档对应行的改动说明与未验证声明。

| ID | 任务 | 新契约点 | 模型/来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 30 | 图像 embedding `/embed`(检索/去重/零样本分类) | reid 契约通用化(fork 加 ~50 行通用 embed 引擎,或导出时归一化烘焙) | CLIP ViT-B/32 或 SigLIP2 base(B/32 档起步,so400m 不取) | ⬜ | | |
| 31 | crowd density `/crowd`(高密度人流) | G2 掩码 DTO 复用 depth 的 stats+preview | LCDnet 系(先调研 ONNX 现成度) | ⬜ | | |
| 32 | 图像返回契约(JSON 内 base64 图)+ image_enhance 接入(批 1.5 任务 21 的前置) | base64 图返回 | 上游 image_enhance 引擎已有 | ⬜ | | |
| 33 | 仪表读数 `/meter`(检测→pose 关键点→角度换算) | 组合式;角度数学照 pose_rules(任务 22) | gauge-ai-reader 路线,需按表盘微调 | ⬜ | | |
| 34 | 图对匹配 `/match`(低优先,巡检比对/视觉定位) | 双 file 字段 | dedode_g+lightglue / lomar(上游已有) | ⬜ | | |

### 收尾

| ID | 任务 | 状态 | 验证证据 |
|---|---|---|---|
| 16 | README 模型表 + 端点表更新;docs/e2e-and-models.md 回填新模型经验 | ⬜ | |
| 17 | 全量回归(clippy --all-targets 零警告 + cargo test 全绿 + 新端点 curl 冒烟)+ 提交推送 | ⬜ | |

## 风险与对冲

- **模型获取是最大不确定项**(0.5~2h/个,可能踩布局坑):严格走前置门;新坑实时回填 `e2e-and-models.md` 踩坑表;
- **单模型受阻不阻塞整批**:跳过换下一个,状态记 ⏸ 并注明原因;
- **形态分组依据**(2026-09-09 API 实测):G1=单模型单次推理;G2=输出 FloatMask 掩码;G3=需先检测再逐 ROI 推理;G4=序列输入(本计划外,暂缓)。

## 工作量口径

代码为确定性劳动(G1 约 30-60 分钟/个,G2/G3 约 1-1.5 小时/个);模型获取另计。
批 1 ≈ 1.5 个工作日(含模型);全部批次 ≈ 4-5 个工作日。

## 与 yolo-rs 吸收计划的关系

2026-09-10 决策:yolo-rs(未与 safe-bot 实际对接)停止演进,其核心能力由本服务吸收,详见 **`docs/yolo-rs-absorption-plan.md`**(调研细节 + 阶段 0-6 实施方案,唯一事实源)。两线并行节奏:批 1/2 引擎暴露(能力广度)先行,吸收计划阶段 1(GPU 硬化)可交错,阶段 2(EngineManager 动态化)在批 1 收尾后与本表 P2 结构重构合并执行。

## 落地优先级(2026-09-21 调研结论,详见研究文档 §六)

| 序 | 内容 | 涉及任务 | 理由 |
|---|---|---|---|
| 1 | pose_rules + hand_keypoint | 22 / 5 | 表 1 白捡项,零模型获取风险;摔倒事件直击 safe_bot 高处作业违章(face_attribute 已挂起,任务 4) |
| 2 | YOLOE 烘焙导出 | 23 | 一个模型解锁"任意自定义类别检测",后续每个定制项目省一次训练 |
| 3 | crack + ppe 升级 | 24 / 25 | 直接对齐 safe_bot 违章目录的确定性场景 |
| 4 | CLIP/SigLIP2 `/embed` | 30 | 通用性最强的新能力(检索/去重/零样本三合一) |
| 5 | 图像增强前置 | 32(连带 21) | 暗光/雾天检测质量兜底,工业现场刚需 |

其余批次(批 2 剩余/批 3/批 4/批 1.6 其余/批 6 其余)按获取难度与场景确认度插空执行;批 2 动态化(吸收计划)仍是所有"新能力槽"工作的架构级前置,宜尽早。

## 变更记录

- 2026-09-27:**状态过时部分校正**。任务 4 face_attribute 挂起(年龄=PIPL 合规红线,表情/性别价值低,落地优先级表同步移除);吸收计划侧阶段 0/3/5 标完成(fork 早已换 git 依赖、HTTP 参数层与 /detect/batch 已上线、safe-bot 适配由 safebot-go 原生完成并类别字典化)。另注:批 1.5 任务 18/19(ppe/fire-smoke)已于 2026-09-26 随新版二进制生产部署并激活(此前生产跑旧版无这些端点,系 argus.env 漏配四引擎变量,已用 warden 内联环境补齐);任务 19 的"比选更强模型"建议仍有效。
- 2026-09-21:**模型生态调研结论编排入表**(调研全文见 `model-ecosystem-research.md`,检索时点 2026-09-21):新增批 1.6(场景即插槽,任务 23-29:YOLOE 开放词汇/crack/ppe 升级/barcode/消防器材/行为违章/YOLO26 换代)与批 6(小契约能力,任务 30-34:/embed/密度图/图像返回契约/仪表读数/图对匹配);修正任务 20 状态(`/quality` 已上线纯算法,此前误列待接入);任务 21 标注对任务 32 的依赖;新增"落地优先级"节(Top5:白捡项→YOLOE→crack+ppe→/embed→图像增强)。架构发现:上游原生支持 YOLOE/YOLO26 标准布局,开放词汇与模型换代均零契约改动。
- 2026-09-21:插入批 1.5 场景批(PPE/火烟/IQA/增强/姿态规则),按企业价值排序。**方向修订**:GPU(吸收计划阶段 4)暂缓,主攻 CPU 线——批 1 解卡(网络已恢复)→ 场景批 → 跨帧跟踪 + 事件规则层(新工作流,另立方案)→ /metrics;阶段 2 动态化仍是唯一重构级合并点。同日完成上游 javpower 1c6ed67(OCR 旋转四边形)同步进 fork(01f66db)与 argus 依赖更新(ba3f83e),E2E 验证旋转文字识别恢复。
- 2026-09-21:**批 1 任务 3 与批 1.5 任务 18/19 全部接入完成**:`/segmentation`(G2,DTO 含 area_ratio + 64×64 掩码预览,复用抽出的 `downsample_grid` 共享函数)、`/ppe`、`/fire-smoke`(G1,`DetectCap` 枚举 + `run_detect` 共享实现,保留"参数校验先于引擎加载"语义,检测族构造抽 `load_detection_pool` 辅助);CAPABILITY_KEYS 12→15;/ui 增 4 卡(reid/实例分割/PPE/火烟,reid/seg 有专属渲染分支)。测试 154→165+1 全绿,clippy 零警告。E2E 全过(ad-hoc 实例):seg bus.jpg 6 实例带掩码、ppe 8 检出与 python 参照一致、fire-smoke 正/负样本正确。批 1 剩 face_attribute(任务 4,模型来源已查实待下载);批 1.5 剩 20/21/22(IQA/增强/姿态规则)。
- 2026-09-21:**批 1 解卡完成**:模型获取网络恢复(hf-mirror/github 实测 200,venv 具备 torch 2.14 + ultralytics 8.4 导出工具链);任务 2 reid 全三道门完成(端点 `/reid`,能力键 `reid`,CAPABILITY_KEYS 11→12);任务 3 segmentation 与批 1.5 任务 18/19 模型全部就绪入 `models/`(前置门过),接入待下批;qualcomm gear_guard_net 因三输出自定义头弃用(教训:AI Hub 模型多为自有头布局,接入前先 dump 输出 shape)。任务 4 face_attribute 模型来源已查实(README 引擎文档:PINTO zoo #070 + OMZ IR 两种布局引擎均兼容),待下载。
- 2026-09-10:建表,16+1 项任务定稿,批 1 范围确认(classification/reid/segmentation/face_attribute)。
- 2026-09-10:任务 1 classification 完成。模板沉淀:`routes/common.rs`(field_text/parse_field_text/decode_image/acquire_permit 公共工具,P2 挂账清偿)+ config/engine/routes 三层接入范式。**附带发现并修复 rust-onnx-infer 上游 resize bug**:`ResizeAlg::Interpolation` 固定核在大比率缩小(>2x)时跳过源像素(混叠,1080→224 分类置信度 0.58→0.38),改 `Convolution` 后 detect 结果也从混叠伪影 7 框收敛为标准 4 person+1 bus。回归测试 `linear_downscale_should_spread_impulse_energy`(旧算法红/新算法绿已对照验证)。上游修复未提交,待与下游一起定稿。
- 2026-09-10:yolo-rs 吸收计划定稿(见 `yolo-rs-absorption-plan.md`),两线并行节奏确认。
