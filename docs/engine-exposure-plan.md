# A 类引擎暴露计划·任务跟踪表

> 目的:将 rust-onnx-infer 的 A 类引擎(结构化输出、HTTP 友好)暴露为 vision-ext 端点。
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
| 2 | reid 行人重识别(OSNet 512 维) | G1,embedding 契约同 face_rec | 待获取 | ⬜ | | |
| 3 | segmentation 实例分割 | G2 mask,DTO 复用 depth 下采样先例 | 待获取 | ⬜ | | |
| 4 | face_attribute 年龄/性别/表情 | G3 组合式,复用 face_recognize 两段式模板 | 待获取 | ⬜ | | |

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

- 2026-09-10:建表,16+1 项任务定稿,批 1 范围确认(classification/reid/segmentation/face_attribute)。
- 2026-09-10:任务 1 classification 完成。模板沉淀:`routes/common.rs`(field_text/parse_field_text/decode_image/acquire_permit 公共工具,P2 挂账清偿)+ config/engine/routes 三层接入范式。**附带发现并修复 rust-onnx-infer 上游 resize bug**:`ResizeAlg::Interpolation` 固定核在大比率缩小(>2x)时跳过源像素(混叠,1080→224 分类置信度 0.58→0.38),改 `Convolution` 后 detect 结果也从混叠伪影 7 框收敛为标准 4 person+1 bus。回归测试 `linear_downscale_should_spread_impulse_energy`(旧算法红/新算法绿已对照验证)。上游修复未提交,待与下游一起定稿。
- 2026-09-10:yolo-rs 吸收计划定稿(见 `yolo-rs-absorption-plan.md`),两线并行节奏确认。
