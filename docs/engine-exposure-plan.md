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
| 4 | face_attribute 年龄/性别/表情 | G3 组合式,复用 face_recognize 两段式模板 | 待获取 | ⬜ | | |

### 批 1.5:场景批(2026-09-21 插入,按企业价值排序)

> 背景:GPU(吸收计划阶段 4)暂缓;批 1 模型获取的网络阻塞已解除(2026-09-21 实测 hf-mirror/github 均 200,venv 具备 torch 2.14 + ultralytics 8.4 本地导出工具链)。按 safe_bot 隐患目录(10 类违章)对齐,PPE/火烟是高频刚需:两者**复用 DetectionEngine**(G1 模板,新能力槽位),无需等阶段 2 动态化。
> 排序依据:与 safe_bot `hazard_catalog.json` 首类"个人防护用品违章"对齐(PPE)、动火违章前置(火烟);IQA/增强/姿态规则为上游存量 CPU 友好资产,直击工地摄像头画质与跌倒场景。

| ID | 任务 | 形态 | 模型来源 | 状态 | 验证证据 | 日期 |
|---|---|---|---|---|---|---|
| 18 | ppe_detect 安全帽/反光衣检测(新能力槽,复用 DetectionEngine) | G1 | 已选定:Hansung-Cho/yolov8-ppe-detection(10 类含违章反类),`ppe_site_10cls.onnx` 已入 models/;qualcomm gear_guard_net 为三输出自定义头(boxes/scores/class_idx,非 v8 布局),复用 DetectionEngine 不可行,弃用 | ✅ | 前置门(上批):标准布局 `[1,14,8400]`/真实工地照 8 检出;接入:`/ppe` 经 `DetectCap::Ppe` 复用 run_detect(参数校验先于引擎加载),E2E 8 检出与 python 参照类别逐项一致(Hardhat×2/NO-Mask×2/Person×2/Safety Vest×2),269ms(debug);5 个路由测试 | 2026-09-21 |
| 19 | fire_smoke_detect 火焰/烟雾检测(同上) | G1 | mfranzon/fire-smoke-yolov8(nano 级),`fire_smoke_2cls.onnx` 已入 models/ | ✅(生产采用前建议比选更强模型) | 接入:`/fire-smoke` 同上;E2E 正样本(房屋火灾)fire 0.618/负样本(bus.jpg)0 检出,1.6s(debug);5 个路由测试 | 2026-09-21 |
| 20 | image_quality 无参考画质评估(IQA) | G1 | 上游引擎已有,模型待查引擎文档 | ⬜ | | |
| 21 | image_enhance 低光增强/去雾(检测前置) | G2 同尺寸 | 上游引擎已有,模型待查引擎文档 | ⬜ | | |
| 22 | pose_rules 姿态几何规则(零模型,摔倒/举手) | G1 纯算法 | 无需模型(输入 pose 关键点) | ⬜ | | |

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

## 变更记录

- 2026-09-21:插入批 1.5 场景批(PPE/火烟/IQA/增强/姿态规则),按企业价值排序。**方向修订**:GPU(吸收计划阶段 4)暂缓,主攻 CPU 线——批 1 解卡(网络已恢复)→ 场景批 → 跨帧跟踪 + 事件规则层(新工作流,另立方案)→ /metrics;阶段 2 动态化仍是唯一重构级合并点。同日完成上游 javpower 1c6ed67(OCR 旋转四边形)同步进 fork(01f66db)与 argus 依赖更新(ba3f83e),E2E 验证旋转文字识别恢复。
- 2026-09-21:**批 1 任务 3 与批 1.5 任务 18/19 全部接入完成**:`/segmentation`(G2,DTO 含 area_ratio + 64×64 掩码预览,复用抽出的 `downsample_grid` 共享函数)、`/ppe`、`/fire-smoke`(G1,`DetectCap` 枚举 + `run_detect` 共享实现,保留"参数校验先于引擎加载"语义,检测族构造抽 `load_detection_pool` 辅助);CAPABILITY_KEYS 12→15;/ui 增 4 卡(reid/实例分割/PPE/火烟,reid/seg 有专属渲染分支)。测试 154→165+1 全绿,clippy 零警告。E2E 全过(ad-hoc 实例):seg bus.jpg 6 实例带掩码、ppe 8 检出与 python 参照一致、fire-smoke 正/负样本正确。批 1 剩 face_attribute(任务 4,模型来源已查实待下载);批 1.5 剩 20/21/22(IQA/增强/姿态规则)。
- 2026-09-21:**批 1 解卡完成**:模型获取网络恢复(hf-mirror/github 实测 200,venv 具备 torch 2.14 + ultralytics 8.4 导出工具链);任务 2 reid 全三道门完成(端点 `/reid`,能力键 `reid`,CAPABILITY_KEYS 11→12);任务 3 segmentation 与批 1.5 任务 18/19 模型全部就绪入 `models/`(前置门过),接入待下批;qualcomm gear_guard_net 因三输出自定义头弃用(教训:AI Hub 模型多为自有头布局,接入前先 dump 输出 shape)。任务 4 face_attribute 模型来源已查实(README 引擎文档:PINTO zoo #070 + OMZ IR 两种布局引擎均兼容),待下载。
- 2026-09-10:建表,16+1 项任务定稿,批 1 范围确认(classification/reid/segmentation/face_attribute)。
- 2026-09-10:任务 1 classification 完成。模板沉淀:`routes/common.rs`(field_text/parse_field_text/decode_image/acquire_permit 公共工具,P2 挂账清偿)+ config/engine/routes 三层接入范式。**附带发现并修复 rust-onnx-infer 上游 resize bug**:`ResizeAlg::Interpolation` 固定核在大比率缩小(>2x)时跳过源像素(混叠,1080→224 分类置信度 0.58→0.38),改 `Convolution` 后 detect 结果也从混叠伪影 7 框收敛为标准 4 person+1 bus。回归测试 `linear_downscale_should_spread_impulse_energy`(旧算法红/新算法绿已对照验证)。上游修复未提交,待与下游一起定稿。
- 2026-09-10:yolo-rs 吸收计划定稿(见 `yolo-rs-absorption-plan.md`),两线并行节奏确认。
