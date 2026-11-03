# 设计：argus 低风险低成本接入 PP-OCRv6 系列

> 状态：draft
> 日期：2026-11-03
> 基线：argus @ `c61b7f3`；rust-onnx-infer @ `1800743c`（`Cargo.lock` 钉定）
> 权威源：`README.md`（端点 / env / 模型来源）、`docs/model-inventory.md`（模型清单）、`docs/plans/2026-11-03-ppocrv6-support-plan.md`（实施计划）
> 取代关系：无

## 1. 背景与目标 / 非目标

### 1.1 背景

argus 的 `/ocr` 当前只配置并验证了 PP-OCRv5：`ppocrv5_det.onnx`（88 MB）+ `ppocrv5_rec.onnx`（84.5 MB）+ 6623 字字典，管线串 `ppocrv5-det960+rec-chn-v1`（`docs/model-inventory.md:21`、`src/engine.rs:65`）。底层 `OcrPipeline` 是通用的 PP-OCR「DBNet 检测 + CTC 识别」结构，模型路径由 `ARGUS_OCR_DET_MODEL` / `ARGUS_OCR_REC_MODEL` / `ARGUS_OCR_DICT` 指定（`src/config.rs:261-263`），并非编译期写死 v5。

PP-OCRv6 是 PaddleOCR 2026 年发布的第六代通用 OCR 模型族，官方文档明确三档：tiny 1.5M、small 7.7M、medium 34.5M 参数；tiny 支持 49 种语言，small/medium 支持 50 种；medium 官方口径检测 Hmean 86.2%、识别准确率 83.2%，相对 PP-OCRv5_server 分别 +4.6 / +5.1 个百分点；识别头训练期 CTC + NRTR 双头，**推理期只保留 CTCHead**（官方文档，见 §2.3）。这与 argus 的 CTC greedy 解码结构一致。

本设计的目标是：在**不改 `/ocr` 契约、不改 rust-onnx-infer、不新增推理运行时**的前提下，用配置化 profile 完整支持 tiny / small / medium 三档，并保持 v5 默认行为逐字节不变。

### 1.2 目标

| ID | 目标 | 可判定标准 |
|---|---|---|
| G1 | 三档 v6 全支持 | `ARGUS_OCR_PROFILE=ppocrv6-{tiny,small,medium}` 均可加载并完成 `/ocr` |
| G2 | v5 默认零回归 | 不设 profile 时，`cargo test`、`tests/golden/models.json`、`/ocr` 的 `pipeline` 与现状完全一致 |
| G3 | 契约不变 | `/ocr` 请求/响应 DTO、错误信封、`CAPABILITY_KEYS` 17 项均不变；仅 `data.pipeline` 随 profile 变化 |
| G4 | 低风险 | 默认关闭 v6；单实例 env 灰度；回滚只需改 env + 重启 |
| G5 | 低复杂度 | 不新增依赖、不改 fork、不改 `EnginePool` / 路由层；新增代码集中在一个 `ocr_profile` 模块 |

### 1.3 非目标

| ID | 非目标 | 原因 |
|---|---|---|
| N1 | 本阶段不追求与 PaddleOCR 官方逐字对齐 | argus 的 DB 后处理是简化版（BFS + 轴对齐 unclip），完整多边形后处理是独立增量，见 §12 |
| N2 | 不接入 VLM / RTen / Paddle Inference | 违反现有 ORT 单栈与低依赖约束 |
| N3 | 不实现 PaddleOCR 的文档方向分类、矫正、文本行方向模块 | argus `/ocr` 从来没有这些模块，不属于本次范围 |
| N4 | 不做动态槽 / 流式 / 热切换 | OCR 是单模型双文件能力，现有 env + 重启语义足够 |
| N5 | 不承诺 INT8 | 官方 ONNX 仓库当前以 FP32 为主；INT8 需单独验证 |
| N6 | 不改变 v5 的阈值、预热、后处理行为 | v5 是生产基线，任何变化都应以独立变更评估 |

## 2. 代码现状与事实

### 2.1 argus 现状（基线 `c61b7f3`）

| 事实 | 证据 |
|---|---|
| 能力键共 17 项，`ocr` 是其中之一 | `src/config.rs:8-13` |
| OCR 三件套 env：`ARGUS_OCR_DET_MODEL` / `ARGUS_OCR_REC_MODEL` / `ARGUS_OCR_DICT` | `src/config.rs:261-263` |
| OCR loader 调 `create_ocr_pipeline`，不传任何检测参数 | `src/engine.rs:706-735` |
| OCR 管线串固定在表里 | `src/engine.rs:65`：`ppocrv5-det960+rec-chn-v1` |
| `pipeline_for` 只对文本两能力支持 env 覆盖；OCR 直接取表值 | `src/engine.rs:263-272` |
| `/ocr` 响应字段：`queue_ms` / `model` / `pipeline` / `regions` / `count` / `text` / `inference_ms` | `src/routes/text.rs:113-145` |
| `regions[]` 字段：`quad` / `score` / `text` / `text_score` | `src/routes/text.rs:20-37` |
| `/models` 的每能力 `pipeline` 与推理响应同源 | `src/routes/models.rs:270-278` |
| golden 中 OCR 的 `pipeline` 被逐字比对 | `tests/golden/models.json:86-95` |
| v5 模型来源、大小、状态 | `docs/model-inventory.md:21` |

### 2.2 rust-onnx-infer 现状（钉定 `1800743c`）

在 `viccom/rust-onnx-infer@1800743c`（argus `Cargo.lock` 中的 revision）核查：

| 事实 | 证据 |
|---|---|
| `create_ocr_pipeline` 只是 `OcrPipeline::new` 的薄包装，不设任何参数 | `src/core/factory.rs:506-513` |
| 检测默认：960/max、thresh 0.3、box 0.5、unclip 2.0、max_candidates 1000、dilation false | `src/engines/ocr.rs:160-175` |
| 检测参数 setter 齐全 | `src/engines/ocr.rs:185-245`（thresh / box / unclip / limit side / limit type / max candidates） |
| 识别默认：输入 48×320、`use_space_char=true`、blank + dict + space | `src/engines/ocr.rs:518-528`、`:792-799` |
| 识别参数 setter 齐全 | `src/engines/ocr.rs:549-570` |
| 字典加载：逐行保留，仅去掉尾部空行 | `src/engines/ocr.rs:802-815` |
| 识别输出 C 与字典不一致时**只 warn，不报错**，越界索引被跳过 | `src/engines/ocr.rs:690-696` |
| `OcrPipeline` 两个字段 public，可直接构造 | `src/engines/ocr.rs:823-848` |
| 输出名在 session 创建时捕获，`run_inference` 取最后一个输出，不硬编码 `sigmoid_0.tmp_0` | `src/core/base.rs:165`、`:436` |

> 关键含义：argus 可以在**不修改 fork**的情况下，用 `OcrDetector::new` + `OcrRecognizer::new` + 公开 setter + `OcrPipeline { detector, recognizer }` 构造 v6 管线；v5 继续走 `OcrPipeline::new` 保证等价。
>
> 另一个关键含义：字典与模型 C 不匹配时当前不会 fail fast，而是 warn + 跳过越界索引，可能静默降级。v6 profile 必须增加 `expected_rec_classes` 预检（见 §6.3）。

### 2.3 官方 PP-OCRv6 事实

**官方文档**（`PaddlePaddle/PaddleOCR`，main 分支 2026-11-03 抓取，commit `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`）：

- 三档：tiny 1.5M、small 7.7M、medium 34.5M 参数；tiny 49 语言（不含日语），small/medium 50 语言。
- medium 官方口径：检测 Hmean 86.2%、识别准确率 83.2%，相对 PP-OCRv5_server +4.6 / +5.1；GPU 2.37×。
- 识别模块：CTCHead 用于推理，NRTRHead 仅训练期辅助，**推理期移除**。
- 统一骨干 PPLCNetV4、检测颈 RepLKFPN、识别颈 EncoderWithLightSVTR。
- 来源：<https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md>

**官方模型资产**（Hugging Face `PaddlePaddle`）：

| 档位 | 仓库 |
|---|---|
| tiny det | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx> |
| tiny rec | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx> |
| small det/rec | `PaddlePaddle/PP-OCRv6_small_det_onnx` / `..._small_rec_onnx` |
| medium det/rec | `PaddlePaddle/PP-OCRv6_medium_det_onnx` / `..._medium_rec_onnx` |

> 2026-11-03 核实：上述 tiny/small/medium 共 6 个 HF 仓库均返回 HTTP 200（仓库存在）。tiny 仓库实测包含 `inference.onnx` / `inference.yml` / `inference.json` / `README.md`，且已下载并实测；small/medium 的文件清单与 `inference.yml` 因镜像 429 未取，T1 必须逐档核对文件、det 参数、字典行数与 rec C，**不得按 tiny 复制**。

**tiny 档实测事实（2026-11-03）**：

| 项 | det | rec |
|---|---|---|
| 文件 | `inference.onnx`（1,780,590 B） | `inference.onnx`（4,462,639 B） |
| SHA256 | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| 输入 | `x` `[N,3,H,W]` 动态 | `x` `[N,3,48,W]` 动态 |
| 输出 | `fetch_name_0` `[N,1,H,W]`，sigmoid 概率图（实测 range 0–0.999996） | `fetch_name_0` `[N,T,6906]`，T=40（W=320 时实测） |
| opset | 14，标准算子 | 11，标准算子 |
| 字典 | — | `inference.yml` 的 `character_dict` 共 6904 项 |
| 类别数 | — | 6906 = 6904 + blank + space；argus 的 `blank + dict + space` 恰好 6906 |
| det 后处理（官方 yml） | thresh 0.2、box_thresh 0.4、max_candidates 3000、unclip_ratio 1.4、`DBPostProcess` | — |
| det 预处理（官方 yml） | BGR、ImageNet mean/std、/255、`DetResizeForTest: null` | — |
| rec 预处理（官方 yml） | — | `RecResizeImg` `[3,48,320]`、`CTCLabelDecode` |

**PaddleOCR 源码默认值**（main 分支 2026-11-03）：

| 事实 | 证据 |
|---|---|
| `DetResizeForTest` 无参数时默认 `limit_side_len=736, limit_type="min"` | `ppocr/data/imaug/operators.py:208-227` |
| `DBPostProcess` 默认 `thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=2.0, use_dilation=False, score_mode="fast"` | `ppocr/postprocess/db_postprocess.py:34-57` |
| `CTCLabelDecode` 默认 `use_space_char=False`，`add_special_char` 为 `["blank"] + dict` | `ppocr/postprocess/rec_postprocess.py:201-232` |
| v6 tiny det yml 未显式写 `DetResizeForTest` 参数，因此按源码默认 736/min | 上述两条合并结论 |
| v6 tiny rec 模型 C=6906，argus 的 `blank + dict + space` 实测解码正确 | 本表 + §附录 A PoC |

> 注意：官方 yml 的 `PostProcess` 未显式写 `use_space_char`，而模型 C=6906 表明存在“空格”类别；本设计以**模型 C 与 argus 字符表对齐**为准（6906 = 6904 + blank + space），不依赖 yml 默认值推断。small/medium 必须在 T1 逐档重新核对，不能假设与 tiny 相同。

### 2.4 与 argus 现状的关键差异

| 差异 | v5 / argus 现状 | v6 官方 | 影响 |
|---|---|---|---|
| det 缩放 | 960 / max | 736 / min | 小图/小字不放大时召回下降；必须按 profile 配置 |
| det 阈值 | 0.3 / 0.5 / 2.0 | 0.2 / 0.4 / 1.4 | 影响候选框数量与几何 |
| det max_candidates | 1000 | 3000 | 密集文本可能截断 |
| det 归一化 | ImageNet | ImageNet | 已一致，无需改 |
| rec 形状 | 48×320 | 48×320 | 已一致 |
| rec 解码 | blank + dict + space | CTC（blank + dict + space） | 已一致 |
| 后处理几何 | 8 连通域 BFS + 轴对齐 unclip | polygon contour + minAreaRect | 旋转/弯曲文本可能有差距，见 §12 |
| 字典/C 不匹配 | warn + 跳过越界索引 | — | 需增加 profile 预检 |
| 预热默认 | 960×960（`input_size`） | 建议 320×320 | 不覆盖会重演 20s 级预热 |

> 表中 v6 列来自 tiny 官方 yml + 实测；small/medium 的 det 参数、rec 形状、字典/C 在 T1 核对后更新，不预设。

### 2.5 资料索引（2026-11-03 抓取）

| 资料 | URL |
|---|---|
| PP-OCRv6 官方文档 | <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md> |
| `DetResizeForTest` 默认值 | <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/data/imaug/operators.py#L208-L227> |
| `DBPostProcess` 默认值 | <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py#L34-L57> |
| `CTCLabelDecode` / 字符表约定 | <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/rec_postprocess.py#L201-L232> |
| tiny det 官方 yml | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/blob/main/inference.yml> |
| tiny rec 官方 yml | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/blob/main/inference.yml> |
| tiny det ONNX | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/blob/main/inference.onnx> |
| tiny rec ONNX | <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/blob/main/inference.onnx> |

> 网络说明：本次通过 HF 镜像下载 tiny 资产（官方仓库同源）；small/medium 的 `inference.yml` 因镜像 429 未取，T1 改用官方站或 ModelScope，并在证据列记录下载渠道。

## 3. 决策与备选

| ID | 决策 | 理由 |
|---|---|---|
| D1 | 在 argus 内新增 `ocr_profile` 模块 + profile 预设，不改 rust-onnx-infer | fork 改动需 push + `cargo update` + 双仓测试，成本高；argus 侧使用公开 API 即可完成；若上游 public 字段/setter 变更，编译期会暴露而非静默 |
| D2 | v5 默认路径继续调用 `OcrPipeline::new`，v6 才走“构造 + setter”分支 | 保证 v5 行为逐字节不变，回归面最小 |
| D3 | 显式 `ARGUS_OCR_PROFILE`，不做文件名自动推断 | 文件名可任意；推断错误会静默算错参数 |
| D4 | 阶段 1 沿用 argus 现有 DB 后处理；多边形后处理作为阶段 2 可选增量 | 先拿到“可用”，再用基准数据决定是否投入；避免一次性高复杂度改造 |
| D5 | 不改 `CAPABILITY_KEYS`、不改 `/ocr` DTO、不加新端点 | 契约稳定；`/models` 的 17 项与 golden 不变 |
| D6 | 字典从官方 `inference.yml` 的 `character_dict` 提取，并校验 `len(dict)+2 == rec C` | 字典必须与模型同源；用模型 C 反向校验，避免抄错版本 |

**备选方案与拒绝理由**：

| 备选 | 结论 | 理由 |
|---|---|---|
| 在 rust-onnx-infer 加 `OcrPreset` + 新工厂 | 暂不采用 | 更干净，但需要 fork 改动、push、`cargo update`、双仓测试；可在后续 upstream 化时再做 |
| 引入 RTen 等第二套推理引擎 | 拒绝 | 第二套推理运行时，违背 argus 单 ORT 栈与低依赖约束 |
| 按模型文件名自动选 v5/v6 | 拒绝 | 文件名不受控，推断错误会静默用错参数 |
| 直接改用 PaddleOCR Python 服务 | 拒绝 | 外部运行时、部署与观测面扩大，不符合 argus 定位 |
| 先移植 polygon 后处理再接 v6 | 拒绝 | 延迟交付、风险高；先量化差距再决定 |
| 用 VLM 替代 PP-OCR | 拒绝 | 任务不同（读字/定位 vs 文档理解）；不与 `/ocr` 行级坐标契约兼容 |

## 4. 架构与数据流

### 4.1 组件关系

```
Config::from_env
  ├─ ARGUS_OCR_PROFILE ──► OcrProfile
  ├─ ARGUS_OCR_*_MODEL / DICT（不变）
  └─ 可选校准 env ───────► OcrPreset 覆盖
                 │
                 ▼
EngineManager::ocr()
  └─ build_ocr_pipeline(config, device)
       ├─ profile == ppocrv5  → OcrPipeline::new(...)           （现状路径）
       └─ profile == ppocrv6-* → OcrDetector::new + setters
                                OcrRecognizer::new + setters
                                OcrPipeline { detector, recognizer }
                 │
                 ▼
routes/text.rs::ocr → pool.run(|e| e.predict(&image)) → TextRegionDto[]
```

### 4.2 profile 与预设

| profile | det 缩放 | thresh / box / unclip | max_candidates | rec 形状 | 字典/C | 状态 |
|---|---|---|---|---|---|---|
| `ppocrv5`（默认） | 960 / max | 0.3 / 0.5 / 2.0 | 1000 | 48×320 | 6623 / 6625（现状） | 现状，不改 |
| `ppocrv6-tiny` | 736 / min | 0.2 / 0.4 / 1.4 | 3000 | 48×320 | 6904 / 6906（实测） | 预设已可由官方 yml 固化 |
| `ppocrv6-small` | T1 从官方 yml 提取 | T1 提取 | T1 提取 | T1 核对（预期 48×320） | T1 提取并校验 | 待 T1 |
| `ppocrv6-medium` | T1 从官方 yml 提取 | T1 提取 | T1 提取 | T1 核对（预期 48×320） | T1 提取并校验 | 待 T1 |

> small/medium 的参数与字典**不得按 tiny 复制**；T1 必须逐档读取 `inference.yml` 的 `PostProcess` / `PreProcess` / `character_dict`，并以 ONNX 输出 C 校验。
>
> 模型文件建议命名（实际路径仍由 `ARGUS_OCR_*` 决定）：`ppocrv6_<tier>_det.onnx` / `ppocrv6_<tier>_rec.onnx` / `ppocrv6_<tier>_dict.txt`。

### 4.3 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ARGUS_OCR_PROFILE` | `ppocrv5` | `ppocrv5` / `ppocrv6-tiny` / `ppocrv6-small` / `ppocrv6-medium`；非法值启动失败 |
| `ARGUS_OCR_DET_MODEL` | 无 | 现有，不变 |
| `ARGUS_OCR_REC_MODEL` | 无 | 现有，不变 |
| `ARGUS_OCR_DICT` | 无 | 现有，不变 |
| `ARGUS_OCR_DET_LIMIT_SIDE` | 按 profile | 可选；正整数 |
| `ARGUS_OCR_DET_LIMIT_TYPE` | 按 profile | 可选；`max` / `min` |
| `ARGUS_OCR_DET_THRESH` | 按 profile | 可选；0..1 |
| `ARGUS_OCR_DET_BOX_THRESH` | 按 profile | 可选；0..1 |
| `ARGUS_OCR_DET_UNCLIP` | 按 profile | 可选；> 0 |
| `ARGUS_OCR_DET_MAX_CANDIDATES` | 按 profile | 可选；≥ 1 |
| `ARGUS_OCR_REC_HEIGHT` | 按 profile（48） | 可选；≥ 1 |
| `ARGUS_OCR_REC_MAX_WIDTH` | 按 profile（320） | 可选；≥ 1 |
| `ARGUS_OCR_WARMUP_SIZE` | v6 默认 320×320；v5 维持现状 | 现有变量；v6 未显式设置时用 320×320 |

> 校准 env 只做覆盖，不改变 profile 默认值；`Config::validate()` 对非法值 fail loud，避免“以为生效实际回退”。

### 4.4 pipeline 串

- v5 且无覆盖：保持 `ppocrv5-det960+rec-chn-v1`，避免消费方缓存失效。
- v6 或存在覆盖：按有效参数生成，例如
  `ppocrv6-tiny-det736-min-b0.4-u1.4+rec-48x320-dict6904-v1`。
- 生成规则必须包含：profile、det limit/type、box_thresh、unclip、rec 形状、字典行数（raw；模型 C 另由 `expected_rec_classes` 预检保证）。参数或字典变化时串必须变化，供消费方做缓存/向量失效判定。
- `/ocr` 的 `data.pipeline` 与 `/models` 的 `pipeline` 继续同源（`EngineManager::pipeline_for`）。

### 4.5 预热

- v5：不设置 `ARGUS_OCR_WARMUP_SIZE` 时维持现状（`input_size` 960×960）。
- v6：未显式设置时默认 320×320。依据：现有 README 记录 OCR 960 全零图预热 20.4s/实例、320×320 为 0.43s/实例（2026-09-11 实测，`README.md:108`），v6 不应重演该成本。
- `ARGUS_OCR_WARMUP_SIZE` 显式设置时优先。

## 5. 契约与兼容性

| 面 | 变化 | 兼容性 |
|---|---|---|
| `/ocr` 请求 | 无 | 完全兼容 |
| `/ocr` 响应字段 | 无；仅 `data.pipeline` 在 v6 profile 下变化 | 加法/值变化，不破坏结构 |
| `/models` | 17 项不变；`models` 文件名随 env；`pipeline` 随 profile | 默认 v5 下逐字不变 |
| 错误信封 | 无 | 完全兼容 |
| `CAPABILITY_KEYS` | 无新增 | 17 项不变 |
| 环境变量 | 新增 `ARGUS_OCR_PROFILE` 与可选校准项 | 全部可选，默认不改变行为 |
| 模型目录 | 新增 v6 文件，v5 保留 | 可并存；不覆盖 v5 |
| 依赖 | 无新增 | `Cargo.toml` / `Cargo.lock` 不变 |

**默认行为保证**：不设置 `ARGUS_OCR_PROFILE` 时，`pipeline` 仍为 `ppocrv5-det960+rec-chn-v1`，`tests/golden/models.json:86-95` 逐字不变。

## 6. 错误处理与安全边界

### 6.1 配置错误

- 未知 `ARGUS_OCR_PROFILE` → `Config::validate()` 返回 Err，进程启动失败，消息含合法值。
- 校准 env 非法（非数字、越界、`limit_type` 不是 max/min）→ 启动失败，消息含变量名与实际值。
- 与现有 `Config::validate()` 的 fail-loud 风格一致（参考 `src/config.rs:351-381` 的 MAXLEN 处理）。

### 6.2 模型缺失

- 三件套任一缺失 → 现有 `ModelNotConfigured("ocr")` 行为不变（`src/engine.rs:710-715`）。
- v6 profile 不改变路径解析，仍由 `ARGUS_OCR_*` 指定；文档需明确“profile 与模型文件必须匹配”。

### 6.3 字典/C 不匹配

现状：`rust-onnx-infer` 在 `classes != char_list.len()` 时只 warn，越界索引被跳过（`src/engines/ocr.rs:690-696`）。这意味着错配可能**静默降级**。

本设计的缓解：

1. **profile 预检**：v6 预设携带 `expected_rec_classes`（tiny=6906；small/medium 由 T1 固化）。构造 `OcrRecognizer` 后，用公开的 `dict_len()` 校验 `dict_len + 2 == expected_rec_classes`；不匹配则返回明确错误，不进入服务。
2. **模型前置门**：T1 用 ONNX Runtime 直接读取 rec 输出最后一维，与 `dict_len+2` 对比并记录证据。
3. **E2E**：对每档跑固定图片，确认文本与参照一致。
4. **后续可选**：推动 rust-onnx-infer 把 warn 改为可配置的硬失败；本阶段不改 fork。

### 6.4 模型供应链

- 模型只从官方 `PaddlePaddle` Hugging Face 仓库获取；记录 SHA256 与文件字节数到 `docs/model-inventory.md`。
- 模型文件不入库（现有约定）；部署时放入 `$ARGUS_MODELS_DIR`，bundle 用软链。
- 不把 token、内网地址写入文档或示例。

## 7. 风险登记

| ID | 风险 | 级别 | 缓解 | 验证 |
|---|---|---|---|---|
| R1 | argus 简化 DB 后处理与 v6 官方多边形后处理有差距 | 中 | 阶段 1 先量化；阶段 2 再决定是否移植 | T6 的 CER/旋转子集对比 |
| R2 | small/medium 参数或字典与 tiny 不同 | 中 | 逐档从官方 yml 提取，不复制 tiny 预设 | T1 每档 IO/C/字典证据 |
| R3 | 字典/C 错配静默降级 | 中 | `expected_rec_classes` 预检 + 模型前置门 + E2E | T1/T3/T6 |
| R4 | 配置项膨胀/误配 | 低 | 单 profile + 少量覆盖；`validate()` fail loud | T2 单测 |
| R5 | v6 精度在某些场景低于 v5 | 中 | 默认 v5；按基准选择档位；保留回滚 | T6 数据集 A/B |
| R6 | 预热/内存回归 | 低 | v6 默认 320 预热；tiny/small 优先；OCR pool=1 | T6 RSS/启动耗时 |
| R7 | ORT 1.26 加载 v6 失败 | 低 | opset 11/14、纯标准算子；E2E 用 argus 同版本 ORT | T6 |
| R8 | pipeline 串变化导致消费方缓存失效 | 低 | v5 串保持不变；v6 串含全参数；文档说明 | T5/T6 |
| R9 | 模型许可/来源不合规 | 低 | 只用官方 PaddlePaddle 资产；逐档记录模型卡许可（tiny rec 模型卡为 Apache-2.0） | T1 |

## 8. 测试策略

### 8.1 单元测试（不依赖模型）

- `OcrProfile` 解析：合法四值、非法值、空串。
- `Config::validate()`：各校准 env 的合法/非法边界。
- `OcrPreset`：每档参数与 pipeline 串的字面量断言（防止“拿表值比表值”自证）。
- `build_ocr_pipeline` 的分支选择：v5 走 `OcrPipeline::new`；v6 走构造 + setter（可用 mock/fake 或参数断言）。
- `warmup_size_for("ocr")`：v5 默认 None；v6 默认 320×320；显式 env 优先。
- `pipeline_for("ocr")`：v5 默认旧串；v6 profile 串；覆盖参数后串变化。

### 8.2 模型前置门（T1）

每档执行并记录：

1. `sha256sum` 与字节数。
2. `onnxruntime` 打印输入/输出名、shape、dtype。
3. 确认 rec 输出最后一维 = `len(character_dict)+2`。
4. 用官方 `inference.yml` 的预处理参数跑 1–2 张公开测试图，确认文本合理。
5. 确认 opset 与算子均为标准 ONNX（无 contrib）。

### 8.3 E2E（T6）

- 起临时实例（非生产端口），v5 基线 + v6 三档分别跑 `/ocr`。
- 固定图片集：发票/收据、工业标签、截图、中英混排、旋转、空图/无文本。
- 指标：行级 CER/准确率、p50/p95 延迟、RSS、预热耗时。
- 断言：`status=ok`、`regions` 非空（有文本图）、`pipeline`/`model` 正确；空图不崩溃。
- **v5 默认回归**：不设 profile 时全量测试 + golden 逐字不变。

### 8.4 回滚演练（T7）

- 灰度实例设置 v6 → 验证 → 改回 `ppocrv5` + 重启 → 验证 `/ocr` 回到旧 pipeline 串与旧结果。
- 记录耗时与命令。

## 9. 实施阶段与门禁

详细任务拆解见 `docs/plans/2026-11-03-ppocrv6-support-plan.md`。阶段摘要：

| 阶段 | 内容 | 门禁 |
|---|---|---|
| T0 | 基线与 pinned API 核对 | `OcrPipeline` 公共字段/setter 在 `1800743c` 存在 |
| T1 | 三档模型前置门 | 每档 SHA256 / IO / 字典 / C 证据齐全 |
| T2 | config profile + 校验 | 单测先红后绿 |
| T3 | `ocr_profile` 预设 + builder | v5 路径等价；v6 参数断言 |
| T4 | engine 接线 + pipeline/warmup | 全量测试 + clippy 0 |
| T5 | 文档与索引 | README / inventory / index 更新 |
| T6 | E2E + 基准 | 验收标准 §10 全过 |
| T7 | 灰度 + 回滚演练 | :7043 A/B 通过，回滚可复现 |
| T8 | 可选 polygon 后处理 | 仅在 T6 显示实质差距时启动 |
| T9 | 生产切换 + 回填 | 双实例/单实例按实际；inventory 回填 |

## 10. 验收标准

1. 不设 `ARGUS_OCR_PROFILE` 时，`cargo test --jobs 6` 全绿、`cargo clippy --all-targets --jobs 6` 零警告、`tests/golden/models.json` 逐字不变。
2. `ppocrv6-tiny` / `small` / `medium` 三档均能加载并返回 `status=ok`；`data.pipeline` 以 `ppocrv6-<tier>` 开头；`data.model` 为 det+rec 文件名#sha。
3. 每档字典 `len(dict)+2` 与 rec 模型 C 一致；错配时预检返回明确错误。
4. 固定图片集上，v6 medium 的行级 CER 不劣于 v5；tiny/small 的精度/延迟/RSS 有实测数据与说明。
5. 空图、无文本图、旋转图不崩溃、不返回 500；行为与 v5 同语义（空 regions）；`/ui` 的 OCR 卡片在 v6 profile 下正常渲染（DTO 未变）。
6. 预热 < 1s（320×320）；启动日志包含 profile 与有效参数。
7. 回滚：改回 `ppocrv5` + 重启后，`/ocr` 的 `pipeline` 与结果回到基线。
8. 文档：`README.md` 模型/env 表、`docs/model-inventory.md`、`docs/e2e-and-models.md`（如有新踩坑）、本设计/计划状态、`docs/README.md` 索引全部更新。

## 11. 回滚

- **配置级回滚**（首选）：`ARGUS_OCR_PROFILE=ppocrv5`，模型路径指回 v5 三件套，重启实例。v5 文件保留在 `$ARGUS_MODELS_DIR`，不需要回滚二进制。
- **部署级回滚**：若新二进制本身异常，回退到上一个 bundle；v5 默认路径不受 profile 代码影响。
- **数据面**：OCR 无会话/无状态，无需数据迁移；`/ocr` 的 `pipeline` 串变化只影响消费方缓存，回滚后串恢复。
- **演练**：T7 必须在灰度实例做一次“v6 → v5”切换并记录。

## 12. 未决项 / 后续

| ID | 事项 | 处理 |
|---|---|---|
| O1 | small/medium 的官方 det 参数、字典行数、rec C | T1 逐档提取并固化 |
| O2 | polygon 后处理（contours + minAreaRect + unclip） | T6 量化差距后决定；若需要则独立设计/计划 |
| O3 | 字典 hash 进入溯源串 | 现有 `/models` 的 `model` 只含 det/rec（`ocr_dict` 先例），暂不改变；可用 pipeline 串的 `dict6904` 表达版本 |
| O4 | rust-onnx-infer 将 C 不匹配改为硬失败 | 本阶段不改 fork；profile 预检先兜底 |
| O5 | INT8 官方资产 | 官方发布后再评估 |
| O6 | `score_mode` / dilation 与官方完全对齐 | 属阶段 2 后处理工作的一部分 |
| O7 | 自动 profile 推断 | 不做；显式配置优先 |

## 13. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-03 | 初稿：基于 argus `c61b7f3`、rust-onnx-infer `1800743c`、PaddleOCR main `dab3fe3`、PP-OCRv6 tiny 实测资产编写 |

---

## 附录 A：前置验证记录（2026-11-03）

**A.1 模型文件**

```
$ stat -c '%n %s bytes' det.onnx rec.onnx
det.onnx 1780590 bytes
rec.onnx 4462639 bytes

$ sha256sum det.onnx rec.onnx
193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8  det.onnx
9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6  rec.onnx
```

**A.2 ONNX IO**

```
== det.onnx
  in  x [N, 3, H, W] tensor(float)
  out fetch_name_0 [N, 1, H, W] tensor(float)       # 实测值域 0 ~ 0.999996
== rec.onnx
  in  x [N, 3, 48, W] tensor(float)
  out fetch_name_0 [N, T, 6906] tensor(float)       # W=320 时 T=40
```

**A.3 字典**

- 官方 `inference.yml` 的 `PostProcess.character_dict`：6904 项，无空行、无多字 token。
- 6904 + blank + space = 6906，与 rec 输出 C 一致。

**A.4 参考管线 PoC（Python + ONNX Runtime 1.29，非 argus 进程）**

- 预处理：det 736/min、ImageNet；rec 48×320、`(x/255-0.5)/0.5`；CTC greedy；blank + dict + space。
- 输入：公开发票测试图。
- 输出节选：`INVOICE`、`Ad4tech Material LLC`、`67h, Martin street`、`Logo`、`Alexander road`、`576832`、`Mobile: +123456789`、`Email: ad4example@gmail.com`。
- 对照：改用 argus v5 参数（960/max、0.3/0.5/2.0）仍可运行，但出现 `Log`（应为 `Logo`）、`Mobile+123456789`（缺冒号）、`gmai.com`（漏字），说明参数适配是必要的。

> 该 PoC 只证明模型/管线同构与参数影响，不等同于 argus 进程内 E2E；后者在 T6 执行。
