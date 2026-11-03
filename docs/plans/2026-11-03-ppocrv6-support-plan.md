# 实施计划：argus 接入 PP-OCRv6 系列

> 状态：draft
> 日期：2026-11-03
> 基线：argus @ `c61b7f3`；rust-onnx-infer @ `1800743c`（`Cargo.lock` 钉定）
> 权威源：设计 [designs/2026-11-03-ppocrv6-support-design.md](../designs/2026-11-03-ppocrv6-support-design.md)
> 执行域：argus 单仓；不改 rust-onnx-infer，只使用其公开 API（`OcrDetector` / `OcrRecognizer` / `OcrPipeline` 及 setter）
> 说明：状态列只在验证证据落地后回填，不预填。

## 1. 执行原则

1. **v5 默认零回归**：不设置 `ARGUS_OCR_PROFILE` 时，走现有 `OcrPipeline::new` 路径，pipeline 串与 golden 逐字不变。
2. **v6 全部 opt-in**：只通过 `ARGUS_OCR_PROFILE` + 现有三件套 env 启用，不改 `CAPABILITY_KEYS`、不加端点、不改 DTO。
3. **不碰 fork**：v6 参数通过公开 setter 应用；`cargo update -p rust-onnx-infer` 不在本计划内。
4. **先红后绿**：每个代码任务先写失败测试，再实现。
5. **事实带证据**：模型参数、IO、字典、C 均以官方 `inference.yml` / ONNX 实测为准；不按 tiny 复制 small/medium。
6. **可回滚**：任何阶段都能通过 env 切回 v5 + 重启恢复。

## 2. 任务跟踪清单（执行时回填）

| ID | 阶段 | 任务 | 验收标准 | 状态 | 证据 |
|---|---|---|---|---|---|
| T0 | 基线 | 核对 `viccom/rust-onnx-infer@1800743c` 的 OCR 公开 API：`OcrPipeline` 字段 public、9 个 setter（6 det + 3 rec）存在、`dict_len()` 存在；记录 argus 当前测试基线 | 逐项 grep 有输出；`cargo test --jobs 6` 基线全绿 | ⬜ 待执行 | — |
| T1 | 模型门 | 逐档（tiny/small/medium）下载官方 `inference.onnx` + `inference.yml`；提取 `character_dict`；记录字节数/SHA256；用 ONNX Runtime 打印 IO；校验 `len(dict)+2 == rec C`；用官方预处理跑 1–2 张公开图；把每档 det 参数（limit side/type、thresh、box、unclip、max_candidates）与 `expected_rec_classes` 写入预设表 | 三档各一行证据齐全；tiny 已有 2026-11-03 记录，补 small/medium；HF 镜像 429 时改用官方站/ModelScope，并记录下载渠道 | ⬜ 待执行 | tiny 部分已见设计附录 A；small/medium 仓库存在性已核实（2026-11-03 API 200） |
| T2 | 配置 | `src/config.rs`：新增 `OcrProfile`（四值）解析、`ARGUS_OCR_PROFILE` + 可选校准 env、`Config::validate()` 校验；单测覆盖合法/非法/空串/边界 | 先红后绿；非法值启动失败消息含变量名与实际值 | ⬜ 待执行 | — |
| T3 | 预设 | 新增 `src/ocr_profile.rs`：`OcrPreset` 表、`build_ocr_pipeline()`；v5 走 `OcrPipeline::new`，v6 构造 + setter；v6 `expected_rec_classes` 预检；单测断言每档参数与 pipeline 串字面量 | v5 路径等价（代码审查 + 测试）；v6 参数断言全过 | ⬜ 待执行 | — |
| T4 | 接线 | `src/engine.rs`：OCR loader 改调 `build_ocr_pipeline`；`pipeline_for("ocr")` 按 profile/有效参数返回；`warmup_size_for("ocr")` v6 默认 320×320；启动日志打印 profile/参数/pipeline | 全量测试 + clippy 0；v5 golden 不变 | ⬜ 待执行 | — |
| T5 | 文档 | `README.md`（profile/env/模型表）、`docs/model-inventory.md`（三档来源/SHA256/状态）、`docs/e2e-and-models.md`（如出现新踩坑）、`docs/README.md` 索引 | 索引登记；权威源无重复正文 | ⬜ 待执行 | — |
| T6 | E2E | 临时实例（非生产端口）跑 v5 + v6 三档：固定图片集（发票/工业标签/截图/中英混排/旋转/空图）；记录 CER、p50/p95、RSS、预热；校验 `/ocr`、`/models`、`/health?deep=true`、`/ui` OCR 卡片 | 设计 §10 验收 1–6 全过；数据带 2026-11 时点 | ⬜ 待执行 | — |
| T7 | 灰度 | 备实例 `:7043` 配 v6（优先 medium 或 small），与 `:7042` v5 做 A/B；执行一次“v6 → v5”回滚演练；若 :7043 未运行，用临时端口等价执行 | A/B 通过；回滚命令与耗时记录 | ⬜ 待执行 | — |
| T8 | 可选 | 若 T6 显示旋转/弯曲/密集小字与官方参照差距超过阈值，启动 polygon 后处理设计与实施（独立增量，不在本计划内直接展开）。建议启动阈值（T6 前确认）：v6 medium 在旋转/弯曲子集相对官方参照 CER 高 ≥3 个百分点，或固定集上不优于 v5；否则记录“不做” | 决策记录：做/不做 + 数据依据 | ⬜ 待执行 | — |
| T9 | 上线 | 按实际实例数切换生产 profile；回填 `model-inventory` 状态；更新本表与设计状态；记录 bundle/版本 | 生产 `/ocr` 冒烟过；回滚路径可用 | ⬜ 待执行 | — |

## 3. 依赖与顺序

```
T0 ──► T1 ──► T2 ──► T3 ──► T4 ──► T5 ──► T6 ──► T7 ──► T9
                                      │
                                      └──► T8（仅差距超阈值）
```

- T1 是**批准门**：small/medium 的官方参数/字典/C 未取齐前，不开始 T3 的对应预设固化。
- T2/T3 可并行编写测试，但 T3 依赖 T1 的 tiny 参数（已有）与 T2 的 profile 类型。
- T6 必须在 argus 同版本 ORT（1.26）上执行，不能只用 Python 1.29 结果代替。
- T8 不阻塞 T9；若差距可接受，记录“不做”的依据即可。

## 4. 关键实现约束

### 4.1 v5 路径必须逐字节等价

- 不设置 `ARGUS_OCR_PROFILE` 时：
  - `build_ocr_pipeline` 必须调用 `OcrPipeline::new(det, rec, dict, device)`，与现状 `create_ocr_pipeline` 相同。
  - 不应用任何 v6 setter。
  - `pipeline_for("ocr")` 返回 `ppocrv5-det960+rec-chn-v1`。
  - `warmup_size_for("ocr")` 行为不变（未设置即 None）。
- 可用单元测试钉死字符串，禁止“拿表值比表值”。

### 4.2 v6 构造顺序

1. `OcrDetector::new(det, device)` → 应用 `set_limit_side_len` / `set_limit_type` / `set_det_thresh` / `set_box_thresh` / `set_unclip_ratio` / `set_max_candidates`。
2. `OcrRecognizer::new(rec, dict, device)` → 应用 `set_rec_image_height` / `set_rec_max_width` / `set_use_space_char(true)`。
3. `OcrPipeline { detector, recognizer }`。
4. 校验 `recognizer.dict_len() + 2 == preset.expected_rec_classes`；不匹配返回明确错误。
5. 记录启动日志：profile、det 参数、rec 形状、`dict_len`、pipeline。

### 4.3 校准 env 只做覆盖

- 预设是默认值；env 存在时覆盖。
- `Config::validate()` 必须拒绝非法值，不静默回退。
- pipeline 串按**有效参数**生成；覆盖后串必须变化。

### 4.4 字典与 C

- 字典从对应档的官方 `inference.yml` 提取，保存为 `ppocrv6_<tier>_dict.txt`（每行一项，保留原字符）。
- 校验：`len(lines) + 2 == rec_output_C`。
- tiny 已知：6904 + 2 = 6906。
- small/medium：T1 实测后填入预设，不假设。

### 4.5 预热

- v6 未设置 `ARGUS_OCR_WARMUP_SIZE` 时默认 `320x320`。
- v5 未设置时维持 `None`（引擎用 `input_size` 960×960），避免改变现状。
- 显式 env 优先于 profile 默认。

## 5. 验证命令（执行时记录真实输出）

```bash
# 基线
cd <argus 仓库根>
cargo test --jobs 6
cargo clippy --all-targets --jobs 6

# 模型文件
sha256sum <det.onnx> <rec.onnx>
stat -c '%n %s bytes' <det.onnx> <rec.onnx>

# ONNX IO（示例，使用可用的 onnxruntime 环境）
python -I - <<'PY'
import onnxruntime as ort
for p in ("<det.onnx>", "<rec.onnx>"):
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print(p)
    for i in s.get_inputs(): print(" in ", i.name, i.shape, i.type)
    for o in s.get_outputs(): print(" out", o.name, o.shape, o.type)
PY

# E2E（临时实例，端口按实际）
curl -s -F "file=@<image>" http://127.0.0.1:<port>/ocr | jq '{status, pipeline:.data.pipeline, count:.data.count, model:.data.model}'

# 回滚
# 改 ARGUS_OCR_PROFILE=ppocrv5 + v5 三件套路径 → 重启 → 重跑同一 curl
```

## 6. 验收标准（与设计 §10 对齐）

1. 默认 v5：`cargo test --jobs 6` 全绿、`cargo clippy --all-targets --jobs 6` 零警告、`tests/golden/models.json` 逐字不变。
2. 三档 v6 均可加载并返回 `status=ok`；`data.pipeline` 以 `ppocrv6-<tier>` 开头；`data.model` 为 det+rec 文件名#sha。
3. 三档字典 `len(dict)+2 == rec C`；错配预检返回明确错误。
4. v6 medium 在固定图片集上 CER 不劣于 v5；tiny/small 的精度/延迟/RSS 有实测数据。
5. 空图/无文本/旋转图不崩溃、不返回 500。
6. v6 预热 < 1s；启动日志含 profile 与有效参数。
7. 回滚演练通过：v6 → v5 后 `/ocr` 的 `pipeline` 与结果恢复基线。
8. 文档与索引更新完成。

## 7. 成本估算

| 阶段 | 内容 | 估算 |
|---|---|---|
| T0–T1 | 基线核对 + 三档模型门 | 0.5–1 人日（tiny 已完成，small/medium 需下载/校验） |
| T2–T4 | 配置/预设/接线 + 单测 | 1–1.5 人日 |
| T5 | 文档 | 0.25 人日 |
| T6 | E2E + 基准 | 0.5–1 人日 |
| T7/T9 | 灰度/上线 | 0.5 人日 |
| **合计** | 不含 T8 | **约 2.75–4.25 人日** |
| T8 | polygon 后处理（可选） | 3–5 人日，独立评估 |

## 8. 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| v6 精度低于 v5 | 默认 v5；T6 数据决定档位；保留 v5 文件 | env 切回 v5 |
| 字典/C 错配 | T1 校验 + profile 预检 + E2E | 切回 v5 |
| 配置误写 | `validate()` fail loud | 修正 env/重启 |
| 预热/内存回归 | v6 默认 320 预热；tiny/small 优先；OCR pool=1 | 调小 pool/切回 v5 |
| 消费方缓存受 pipeline 串影响 | v5 串不变；v6 串显式版本化；文档通知 | 切回 v5 串恢复 |

**回滚步骤**：

1. `ARGUS_OCR_PROFILE` 改回 `ppocrv5`（或删除）。
2. `ARGUS_OCR_DET_MODEL` / `ARGUS_OCR_REC_MODEL` / `ARGUS_OCR_DICT` 指回 v5 三件套。
3. 重启实例（warden restart 或临时实例 kill/拉起）。
4. 验证 `/ocr` 的 `pipeline` 为 `ppocrv5-det960+rec-chn-v1`，固定图片结果与基线一致。

## 9. 完成定义

- T0–T7、T9 全部销账（T8 可记录“不做”并附数据依据）。
- `cargo test --jobs 6` 全绿 + `cargo clippy --all-targets --jobs 6` 零警告。
- 三档 v6 E2E 证据 + v5 零回归证据 + 回滚演练记录。
- 文档权威源更新，索引登记，状态流转为 `implemented`（设计）与完成（本计划）。

## 10. 不做的事

- 不改 `rust-onnx-infer` / 不 push fork / 不 `cargo update`。
- 不新增能力键、不改 `/ocr` DTO、不加端点。
- 不引入 RTen / Paddle Inference / VLM / Python 服务。
- 不做动态槽、流式、热切换。
- 不在本阶段承诺 INT8 或与 PaddleOCR 官方逐字对齐。
