# homepix 验证计划：`POST /embed/text`（批 D1 交付验收）

> 状态：approved（交付 homepix Coding Agent 执行；执行后按 §7 回填结论）
> 日期：2026-10-21
> 基线：argus @ `95a2eaf`（本地 main）；rust-onnx-infer @ `49b8585`（fork master）；模型 jina-embeddings-v2-base-zh（docs/model-inventory.md #19）
> 权威源：端点 / env / 模型来源 = [README.md](../../README.md)；批 D 需求与进度 = [homepix-integration-plan.md](../homepix-integration-plan.md) §4b
> 取代关系：无

## 1. 目的与执行方式

本文档供 homepix 项目 Coding Agent 对 argus 批 D1 交付（`POST /embed/text` 文本 embedding 端点）做独立测试验证。所有用例可执行、判定标准明确：

- 核心用例 **V1–V6** 对任何已正确配置的 argus 实例可跑（§5.1 一键脚本，约 2 分钟）；
- **V7/V8** 为条件用例（实例配置了鉴权 / 专用低并发实例才可测）；
- §6 给出 homepix 批 D1 六条硬条件与用例的映射，逐条判定；
- 执行完把结果回填 §7 结论表（含脚本输出摘录）。

依赖：`curl` + `python3`（标准库即可，验证脚本不用第三方包）。

## 2. 被测能力事实卡

| 项 | 值 | 代码锚点（基线 95a2eaf） |
|---|---|---|
| 端点 | `POST /embed/text`，JSON body | src/routes/embed.rs |
| 能力键 | `text_embedding`（第 16 个能力键） | src/config.rs:12 `CAPABILITY_KEYS` |
| 模型 | jina-embeddings-v2-base-zh：JinaBERT，**中英双语（中文优先）**，Apache-2.0，641MB fp32 | docs/model-inventory.md #19 |
| 维度 | **768**（env 声明 + 加载期与引擎实际输出校验，不符即拒载） | src/engine.rs loader |
| 池化/归一化 | MaskedMean + L2 归一化（恒归一化，响应 `normalized:true`） | src/engine.rs:875 |
| pipeline 串 | `jina-embeddings-v2-base-zh\|mean-pool-norm-v1\|maxlen512` | src/engine.rs:71 `PIPELINE_TABLE` |
| 截断 | 512 token（含特殊 token；约 500 汉字以上截断） | src/engine.rs:41 |
| mode | `passage`（缺省）/ `query`，两值等价（jina v2 无需指令前缀，可始终不传） | src/routes/embed.rs |
| 确定性 | 同输入**逐位**确定；逐条推理循环实现，批组成不影响单条向量 | 计划表 D6（T8 决策） |
| 性能预算 | CPU 单条 p95 ≤ 30ms（fork 直跑实测 18.4ms，2026-10-21，ur5900） | docs/model-inventory.md #19 |

## 3. 环境前置

设 `BASE` 为目标实例地址（示例用 `http://127.0.0.1:7101`）。实例配了鉴权时再设 `KEY`。

### 3.1 就绪检查

```bash
BASE=http://127.0.0.1:7101
# 实例配了鉴权时必须带 key 头（/models 不在免鉴权名单，只有 / 与 /ui 免）
curl -s -H "authorization: Bearer $KEY" "$BASE/models" | jq '.data.capabilities.text_embedding'
```

预期（已配置实例；`loaded` 首次推理后才变 true，懒加载）：

```json
{
  "configured": true,
  "enabled": true,
  "loaded": false,
  "models": ["model"],
  "dim": 768,
  "pipeline": "jina-embeddings-v2-base-zh|mean-pool-norm-v1|maxlen512"
}
```

判定：`configured=true`、`dim=768`、`pipeline` 串逐字符一致。`configured=false` → 实例未配三件套 env，按附录 A 自起验证实例。

### 3.2 模型文件完整性（自起实例前）

| 文件 | sha256 前 12 |
|---|---|
| `model.onnx`（641MB） | `4b0e9fa6e5c7` |
| `tokenizer.json`（2MB） | `0046da43cc8c` |

```bash
cd /root/appRun/argus/rust-onnx-infer/models/jina-embeddings-v2-base-zh
sha256sum model.onnx tokenizer.json
```

指纹不符 = 模型文件被动过，停止验证并上报。注意：若被换成**同维度**的其他模型，加载期 dim 校验不会拦截——sha256 指纹才是唯一判据。

## 4. 契约

### 4.1 请求

```http
POST /embed/text
Content-Type: application/json
Authorization: Bearer <key>        # 或 X-API-Key: <key>；实例未设 ARGUS_AUTH_TOKEN 时免鉴权
```

```json
{"texts": ["文本1", "文本2"], "mode": "passage", "normalize": true}
```

| 字段 | 类型 | 约束（违反 → `bad_request`） |
|---|---|---|
| `texts` | string[] | 必填；1..128 条；单条 1..2000 字符（按 Unicode 码点计，非字节）；元素必须是字符串 |
| `mode` | string? | 仅 `passage` / `query`（两值等价，可不传） |
| `normalize` | bool? | 仅 `true` / 缺省；`false` 显式拒绝（引擎恒归一化） |

### 4.2 成功响应

```json
{
  "status": "ok",
  "data": {
    "queue_ms": 0,
    "model": "model.onnx#4b0e9fa6e5c7",
    "pipeline": "jina-embeddings-v2-base-zh|mean-pool-norm-v1|maxlen512",
    "dim": 768,
    "normalized": true,
    "count": 2,
    "embeddings": [[0.0123, ...], [...]],
    "inference_ms": 37
  }
}
```

| 字段 | 说明 |
|---|---|
| `queue_ms` | 准入排队耗时（空闲时 0） |
| `model` | `文件名#sha256前12` 溯源——应与 §3.2 指纹一致 |
| `dim` / `normalized` / `count` | 768 / 恒 true / `len(texts)` |
| `embeddings` | 与 `texts` **顺序一一对应**；每条 768 个 float，已 L2 归一化——**余弦相似度退化为点积** |
| `inference_ms` | 服务端逐条推理总耗时（准入之后起表，不含排队） |

### 4.3 错误响应

**HTTP 状态码恒 200**（信封语义，全端点一致）——**不能靠状态码判错，必须检查 body 的 `status` 字段**：

```json
{"status": "error", "data": {"error": "人类可读消息", "code": "bad_request", "retryable": false}}
```

`/embed/text` 可达错误码（**全部是既有 code，零新增**）：

| code | 触发 | retryable |
|---|---|---|
| `bad_request` | 参数校验失败 / 坏 JSON / 缺字段 | false |
| `too_large` | body 超限（`ARGUS_MAX_BODY_MB`，默认 20MB） | false |
| `model_not_configured` | 三件套 env 缺任一 | false |
| `capability_disabled` | `ARGUS_ENABLED_CAPABILITIES` 白名单未含 | false |
| `inference_failed` | 引擎推理失败 | false |
| `internal` | 引擎加载任务异常 | false |
| `overload` | 并发准入满 / 排队超时 | **true（唯一可重试）** |

特例——**鉴权失败**（仅实例设了 `ARGUS_AUTH_TOKEN` 时）：HTTP 200 + `{"status":"error","data":{"error":"缺少 API key…"}}`，**`data` 无 `code` 字段**（auth 中间件信封，与全端点一致）。消费方判错逻辑应以 `status=="error"` 为准，不要假设 `code` 恒存在。

### 4.4 集成注意事项

1. **pipeline 串是向量缓存失效键**：homepix 库内向量必须与 `pipeline` 串绑定存储；串变（换模型/池化/maxlen）= 旧向量全部失效，需全量重嵌。
2. 隐私：错误消息不回显输入文本（用 `texts[i]` 下标定位）；服务端日志同样不含输入文本。
3. 批上界 128 条 × 2000 字符/条；更大批量自行分批（逐条推理，拆批不影响向量）。
4. 冷启动：首次请求触发模型加载（641MB，数秒到十几秒）+ 池预热。批量入库前先发一条探测请求。
5. 性能判定须 **release 构建**实例（debug 慢约 10 倍，只验功能不判性能）。

## 5. 验证用例

### 5.1 一键脚本（V2–V6 核心）

保存为 `verify_embed_text.py` 后执行：`BASE=http://127.0.0.1:7101 KEY=<key> python3 verify_embed_text.py`（未配鉴权则不设 KEY）。仅标准库，无第三方依赖：

```python
#!/usr/bin/env python3
"""homepix 验证脚本：argus POST /embed/text 核心用例（V2-V6）。
退出码 0 = 全过；非 0 = 有失败项（见输出 FAIL 行）。"""
import json, math, os, sys, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:7101")
KEY = os.environ.get("KEY", "")
PIPELINE = "jina-embeddings-v2-base-zh|mean-pool-norm-v1|maxlen512"
failures = []

def post(payload):
    req = urllib.request.Request(
        BASE + "/embed/text", data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    if KEY:
        req.add_header("authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)

# ---- V2 基本功能（单条中文）----
d = post({"texts": ["一家人在海边度假的照片"]}).get("data", {})
emb = (d.get("embeddings") or [[]])[0]
norm = math.sqrt(sum(x * x for x in emb)) if emb else 0.0
check("V2 status=ok 且 count=1", bool(d) and d.get("count") == 1)
check("V2 dim=768（≤768 预算）", d.get("dim") == 768, f"dim={d.get('dim')}")
check("V2 向量长度=768", len(emb) == 768, f"len={len(emb)}")
check("V2 L2 模长≈1", abs(norm - 1.0) < 1e-3, f"|v|={norm:.6f}")
check("V2 normalized=true", d.get("normalized") is True)
check("V2 pipeline 串", d.get("pipeline") == PIPELINE, str(d.get("pipeline")))
check("V2 模型溯源", str(d.get("model", "")).startswith("model.onnx#4b0e9fa6e5c7"), str(d.get("model")))

# ---- V2b 批量（中英混合，顺序对应）----
texts = ["孩子们在沙滩上堆城堡", "playing on the beach", "财务报表"]
d = post({"texts": texts}).get("data", {})
check("V2b 批量 count=3", d.get("count") == 3 and len(d.get("embeddings", [])) == 3)
check("V2b 每条 768 维", all(len(v) == 768 for v in d.get("embeddings", [])))

# ---- V3 确定性（同输入两次，逐位相等）----
a = post({"texts": [texts[0], texts[2], texts[1]]})["data"]["embeddings"]
b = post({"texts": [texts[0], texts[2], texts[1]]})["data"]["embeddings"]
check("V3 同输入两次逐位确定", a == b)

# ---- V4 语义判别（双语；已归一化 → 点积即余弦）----
v = post({"texts": [
    "在海边玩",                    # 0 中文锚
    "沙滩上堆城堡",                # 1 中文正相关
    "财务报表表格截图",            # 2 中文负相关
    "playing by the seaside",      # 3 英文正相关
    "quarterly financial report",  # 4 英文负相关
]})["data"]["embeddings"]
cos = lambda i, j: sum(x * y for x, y in zip(v[i], v[j]))
zh_pos, zh_neg = cos(0, 1), cos(0, 2)
en_pos, en_neg = cos(0, 3), cos(0, 4)
check("V4 中文：正对 > 负对", zh_pos > zh_neg, f"{zh_pos:.4f} > {zh_neg:.4f}")
check("V4 跨语言：英文同义 > 中文负例", en_pos > en_neg, f"{en_pos:.4f} > {en_neg:.4f}")
print(f"INFO 判别间隔：中文 {zh_pos - zh_neg:.4f} / 跨语言 {en_pos - en_neg:.4f}"
      "（参考实测 2026-10-21：中文 0.42 / 跨语言 0.45，探针语句不同会有浮动）")

# ---- V5 性能：单条 p95 ≤ 30ms（服务端 inference_ms；预热 3 次后取 30 样本）----
for _ in range(3):
    post({"texts": ["预热"]})
samples = sorted(post({"texts": ["一只橘猫趴在窗台上晒太阳"]})["data"]["inference_ms"]
                 for _ in range(30))
p95 = samples[math.ceil(0.95 * len(samples)) - 1]
check("V5 单条 p95≤30ms（须 release 实例）", p95 <= 30,
      f"p95={p95}ms min={samples[0]} max={samples[-1]}")

# ---- V6 错误路径（信封 + code 精确匹配；消息不回显原文）----
cases = [
    ("缺 texts 字段", {"mode": "passage"}, "bad_request"),
    ("空数组", {"texts": []}, "bad_request"),
    ("空字符串元素", {"texts": ["正常文本", ""]}, "bad_request"),
    ("元素非字符串", {"texts": [123]}, "bad_request"),
    ("mode 非法", {"texts": ["正常"], "mode": "doc"}, "bad_request"),
    ("normalize=false", {"texts": ["正常"], "normalize": False}, "bad_request"),
]
for name, payload, want in cases:
    d = post(payload).get("data", {})
    check(f"V6 {name}→{want}", d.get("code") == want, str(d.get("code")))
d = post({"texts": [f"文本{i}" for i in range(129)]}).get("data", {})
check("V6 129 条→bad_request", d.get("code") == "bad_request")
d = post({"texts": ["汉" * 2001]}).get("data", {})
check("V6 2001 字→bad_request", d.get("code") == "bad_request")
check("V6 错误消息不含原文（隐私）", "汉" not in d.get("error", ""))
d = post({"texts": ["正常文本", ""]}).get("data", {})
check("V6 空串以下标定位", "texts[1]" in d.get("error", ""), str(d.get("error")))

print()
if failures:
    print(f"结果：{len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果：V2-V6 全部通过")
```

### 5.2 条件用例

**V7 鉴权**（仅实例设了 `ARGUS_AUTH_TOKEN`）：

```bash
# 无 key → 200 信封、data 无 code 字段
curl -s "$BASE/embed/text" -H 'content-type: application/json' -d '{"texts":["测试"]}'
# 期望：{"status":"error","data":{"error":"缺少 API key（Authorization: Bearer <key> 或 X-API-Key: <key>）"}}

# 错 key → 同信封、"无效的 API key"
curl -s "$BASE/embed/text" -H 'content-type: application/json' -H 'authorization: Bearer wrong' -d '{"texts":["测试"]}'

# 对 key：Bearer 与 X-API-Key 两种头均可
curl -s "$BASE/embed/text" -H 'content-type: application/json' -H "X-API-Key: $KEY" -d '{"texts":["测试"]}' | jq .status   # "ok"
```

**V8 过载与 retryable**（需专用实例，附录 A 命令追加两个 env：`ARGUS_MAX_CONCURRENT_INFERENCE=1 ARGUS_ACQUIRE_TIMEOUT_MS=0`——后者 0 = fail-fast 模式，拿不到许可立即拒绝；**默认 30000ms 排队模式下短文本请求会排队成功，复现不了 overload**）：

```bash
for i in 1 2 3; do
  curl -s "$BASE/embed/text" -H 'content-type: application/json' -H "authorization: Bearer $KEY" \
       -d '{"texts":["并发压测文本一号","并发压测文本二号","并发压测文本三号"]}' | jq -c '{status,code:.data.code,retryable:.data.retryable}' &
done; wait
```

判定：3 个响应 = 1 个 `{"status":"ok",...}` + 2 个 `{"status":"error","code":"overload","retryable":true}`。

## 6. 批 D1 硬条件 ↔ 用例映射

| D1 硬条件 | 用例 | argus 侧已有证据（2026-10-21） |
|---|---|---|
| `CAPABILITY_KEYS` 含 `text_embedding` | V1 | config.rs 16 键；cargo test 282 全绿 |
| 响应回报 `dim` 与 `pipeline` | V1 + V2 | `data.dim`/`data.pipeline` + `/models` 预检；单测锁定信封形状 |
| 错误码零新增 | V6 | 既有 code 集（§4.3）；embed.rs 13 条错误路径单测 |
| 同输入确定性 | V3 | E2E 两次请求 embeddings 逐位相等（jq diff 为空） |
| CPU 单条 p95 ≤ 30ms | V5 | fork 直跑 p95 18.4ms（2026-10-21，ur5900）；目标实例以 V5 实测为准 |
| dim ≤ 768 | V2 | 768 |

## 7. 结论回填表（homepix 执行后填写）

| 用例 | 结果（PASS/FAIL/SKIP） | 证据（输出摘录） | 执行人 / 日期 |
|---|---|---|---|
| V1 就绪与配置回报 | | | |
| V2/V2b 基本功能与批量 | | | |
| V3 确定性 | | | |
| V4 语义判别（双语） | | | |
| V5 性能 p95 | | | |
| V6 错误路径矩阵 | | | |
| V7 鉴权（条件） | | | |
| V8 过载与 retryable（条件） | | | |

## 8. 附录 A：自起验证实例（本机路径）

模型文件在 fork 仓库（生产 bundle 部署待做，2026-10-21 时点状态）：

```bash
cd /root/appRun/argus/argus
ORT_DYLIB_PATH=/root/appRun/argus/argus-0.1.0-c4e44aa/lib/libonnxruntime.so.1.26.0 \
ARGUS_PORT=7101 \
ARGUS_TEXT_EMBEDDING_MODEL=/root/appRun/argus/rust-onnx-infer/models/jina-embeddings-v2-base-zh/model.onnx \
ARGUS_TEXT_EMBEDDING_TOKENIZER=/root/appRun/argus/rust-onnx-infer/models/jina-embeddings-v2-base-zh/tokenizer.json \
ARGUS_TEXT_EMBEDDING_DIM=768 \
ARGUS_AUTH_TOKEN=测试用key占位 \
./target/release/argus
```

- 启动后探活（/health **不免鉴权**，须带 key）：`curl -s -H "authorization: Bearer $KEY" http://127.0.0.1:7101/health`；Ctrl-C 优雅停机。
- 只配 text_embedding 三件套（其余能力懒加载不触发）；V8 需追加 `ARGUS_MAX_CONCURRENT_INFERENCE=1 ARGUS_ACQUIRE_TIMEOUT_MS=0`。
- 可选 `ARGUS_TEXT_EMBEDDING_POOL=N` 调文本池大小（默认 1）。
- 免鉴权跑功能用例时不设 `ARGUS_AUTH_TOKEN`（V7 需设）。

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-21 | 初版（基线 95a2eaf：jina-embeddings-v2-base-zh 换型后契约；V1-V8 用例 + 一键脚本） |
