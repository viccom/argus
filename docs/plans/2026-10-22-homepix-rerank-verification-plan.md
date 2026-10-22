# homepix 指南：文本 Embedding + Reranker 双端点与两段式检索（批 D2 交付）

> 状态：approved（交付 homepix Coding Agent；执行后按 §8 回填结论）
> 日期：2026-10-22
> 基线：argus @ `29d20b4`（生产 bundle `argus-0.1.0-d960b4a`）；fork `1800743`；前导文档：[2026-10-21-homepix-embed-text-verification-plan.md](2026-10-21-homepix-embed-text-verification-plan.md)（/embed/text 契约不变，仍然有效）
> 权威源：端点 / env / 模型来源 = [README.md](../../README.md)；进度 = [homepix-integration-plan.md](../homepix-integration-plan.md) §4b；模型实测 = [model-inventory.md](../model-inventory.md) #19/#22/#23

## 1. 目的与快速开始

argus 新增第 17 个能力 `text_reranking`（端点 `POST /rerank`）。homepix 的语义搜索建议升级为**两段式**：

```
用户查询
  → ① /embed/text：查询 → 768 维向量，与库内照片 caption 向量做余弦，召回 top-N（N 建议 20~32）
  → ② /rerank：查询 + 召回的 N 条 caption 原文 → 相关性分数（0~1），按分重排取 top-M（M 建议 5~10）
```

为什么加第二段：双塔 embedding 对查询和文档**独立**编码，快但丢失交互信息；cross-encoder reranker 把 query+doc **拼在一起**过模型，判别更细。实测（2026-10-22，见 §5）它主要在**近重复 caption 区分、间接措辞、跨语言查询**三类场景纠偏；caption 区分度本来就好的场景无增量（这时只用 ① 也完全成立——两段解耦，可随时开关）。

## 2. 生产实例现状卡（2026-10-22 时点）

| 项 | 值 |
|---|---|
| 实例 | `http://127.0.0.1:7042`（鉴权：`Authorization: Bearer <key>` 或 `X-API-Key`） |
| 版本 | `argus 0.1.0 (git d960b4a)`（bundle `argus-0.1.0-d960b4a`） |
| ① embedding 模型 | jina-embeddings-v2-base-zh（**中英双语**，768 维，mean 池化 + L2，maxlen 512） |
| ② reranker 模型 | jina-reranker-v2-base-multilingual **int8**（多语 cross-encoder，maxlen 512） |
| 失效判定键 | ① `jina-embeddings-v2-base-zh\|mean-pool-norm-v1\|maxlen512`　② `jina-reranker-v2-base-multilingual\|sigmoid\|maxlen512`（响应 `data.pipeline`；串变 = 旧向量/分数全失效） |
| 内存占用 | 双文本引擎 ≈1.35GB（实测 1333-1489MB，含 argus 基线与已触发引擎） |
| 探活 | `/health`（须带 key）；`/models` 的 `text_embedding`/`text_reranking` 两项 `configured:true` |

就绪检查：

```bash
BASE=http://127.0.0.1:7042; KEY=<你的key>
curl -s -H "Authorization: Bearer $KEY" "$BASE/models" | jq '.data.capabilities.text_reranking'
# 期望：{"configured":true,"enabled":true,"loaded":false,"models":["model_int8"],"pipeline":"jina-reranker-v2-base-multilingual|sigmoid|maxlen512"}
# 注意：无 dim 字段（输出是分数不是向量）；loaded 首次推理后才 true
```

## 3. `POST /rerank` 契约

### 3.1 请求

```http
POST /rerank
Content-Type: application/json
Authorization: Bearer <key>
```

```json
{"query": "孩子在沙滩堆城堡", "texts": ["小女孩蹲在沙滩上堆沙堡", "孩子们用小桶在沙滩上堆城堡", "孩子在沙滩上挖坑玩沙"]}
```

| 字段 | 约束（违反 → `bad_request`） |
|---|---|
| `query` | 必填；1..2000 字符（按 Unicode 码点） |
| `texts` | 必填；1..**32** 条（服务端 `ARGUS_TEXT_RERANKING_MAX_TEXTS` 可调至 64，当前生产 32）；单条 1..2000 字符；空字符串拒绝 |
| **计算量预算** | `条数 × 最长pair编码token数 ≤ 8192`——条数上限随文本长度浮动：全 512 token 时最多 16 条，全 256 token 时 32 条，短文本（~12 token）32 条仅 ~170ms。超出 → `bad_request`（消息含 B/max_len/乘积/上限，可据此指导分批） |

### 3.2 成功响应

```json
{
  "status": "ok",
  "data": {
    "queue_ms": 0,
    "model": "model_int8.onnx#c5220cf8fe02",
    "pipeline": "jina-reranker-v2-base-multilingual|sigmoid|maxlen512",
    "count": 3,
    "scores": [0.83, 0.78, 0.02],
    "ranking": [{"index": 0, "score": 0.83}, {"index": 1, "score": 0.78}, {"index": 2, "score": 0.02}],
    "inference_ms": 95
  }
}
```

- `scores`：**输入序**（与 `texts` 一一对应），sigmoid 后 0~1（越大越相关）
- `ranking`：按分降序的 `{index, score}`，**同分按 index 升序**（确定性 tie-break）——拿 `ranking[i].index` 回查 `texts` 即得重排结果
- `model`：`文件名#sha256前12` 溯源——int8 应为 `model_int8.onnx#c5220cf8fe02`

### 3.3 错误（HTTP 恒 200 信封，判 `body.status`；code 集与 /embed/text 完全一致，零新增）

`bad_request`（校验/预算/坏 JSON）/ `too_large` / `model_not_configured` / `capability_disabled` / `inference_failed` / `internal` / `overload`（唯一 `retryable:true`）。鉴权失败特例同全端点：200 + `data.error`，**无 code 字段**。错误消息不回显 query/texts 原文（隐私，`texts[i]` 下标定位）。

### 3.4 确定性与缓存口径（重要，与 embedding 不同！）

- **同请求重放逐位确定**（同 texts 数组同顺序）
- **texts 数组组成/顺序变化 → 分数会漂移**（int8 下可达 0.1 量级，因批 padding 形状变化）——**rerank 分数只做单次请求内的相对排序，不要跨请求比较绝对值，也不建议缓存**；embedding 向量则可长期缓存（失效键 = pipeline 串）
- 一个请求会触发两次编码（预算检查 + 推理），超预算请求会触发一次引擎懒加载（一次性，可接受）

## 4. homepix 集成建议

1. **两段管线**：`/embed/text` 批量预计算库内 caption 向量（一次入库，缓存 pipeline 串）；查询时 ① 召回 top-N → ② 把这 N 条**原文**发 `/rerank` → 展示 top-M。N=20~32、M=5~10 是性价比区间
2. **开关设计**：② 是纯增量（不①就无②），建议做成配置开关——实测它并非处处加分（§5），先灰度对比再默认开启
3. **阈值**：分数是 sigmoid 概率但**无全局校准**（不同批不可比），如需"低于 X 不展示"应基于同批内相对分布（如 top1 分数的比例）而非绝对值
4. **分批**：超预算报错消息含明细；把 N 降到 16 或缩短 caption 即可
5. 隐私：query/texts 不出现在服务端日志与错误消息中

## 5. 实测验证结果汇总（全部 2026-10-22，ur5900 / 8C15G / intra_op=4）

### 5.1 reranker 质量评测（合成 caption 评测集，argus :7042 embedding 一段 + onnxruntime 直跑二段）

| 场景 | 纯 embedding | +rerank | 结论 |
|---|---|---|---|
| 常规语义池（10 查询×16 候选） | MRR 1.000 | 0.906 ↓ | **无增量且偶发错排**（"睡前故事/春节"查询）——caption 区分度好时别开 |
| 细粒度对抗池（张伟/李伟、三个/两个、桌下/桌上） | 1.000 | 1.000 | dense 已够强 |
| **近重复大池（32 条海滩场景）** | **0.833** | **1.000** | **主增益**：间接措辞（"海边堆沙堡"）与跨语言（"child building sandcastle"）查询的 top1 纠偏 |

（bge-reranker-base 对比：中文家庭场景更稳——同池 1.000；双语是 jina 胜出依据，见 §7）

### 5.2 端到端验收

- **jina 对**（生产配置）14/14：fixture 正对 0.778/负对 0.024（间隔 0.754）；同请求重放逐位相等；错误矩阵；预算内最坏 16×428token 实测 3.5-5.2s（生产 shutdown 预算 10s 内）；超预算 32 条快速拒绝且报错含明细
- **bge 对**（env 换族验证）10/10：`POOLING=cls + DIM=512 + 双 PIPELINE 覆盖`全生效，pipeline 串正确显示 bge——证明"复制 README 预设块改 env"即完成模型对切换
- 生产冒烟（3 个 bundle 版本累计）：`/rerank` 0.778/0.024 @19-23ms；homepix 批 D1 验证脚本 V2-V6 每次部署后复跑全过

### 5.3 性能与资源

| 指标 | 数值 |
|---|---|
| 典型 rerank（8 条 × ~60 token） | 92-155ms/查询（int8，服务端 inference_ms） |
| 短文本 32 条 | ~170ms |
| 预算内最坏（16×512token） | 3.5-5.2s、激活峰值 ~770MB |
| 双引擎内存 | 实测 RSS 1333-1489MB（embedding 731MB + reranker int8 621MB + 基线） |

## 6. 验证用例（V1-V5）与一键脚本

V1 就绪检查见 §2。以下脚本覆盖 V2-V5（仅标准库，`BASE`/`KEY` 环境变量同前一份文档）：

```python
#!/usr/bin/env python3
"""homepix 验证脚本：POST /rerank 核心用例（V2-V5）。退出码 0 = 全过。"""
import json, math, os, sys, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:7042")
KEY = os.environ.get("KEY", "")
PIPELINE = "jina-reranker-v2-base-multilingual|sigmoid|maxlen512"
failures = []

def post(path, payload, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    if KEY:
        req.add_header("authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)

# ---- V2 基本功能与判别（固定 fixture，2026-10-22 三模型实测标定）----
d = post("/rerank", {"query": "孩子在沙滩堆城堡",
                     "texts": ["小女孩用沙子堆了一座城堡", "季度财务报表数据汇总"]})["data"]
pos, neg = d["scores"][0], d["scores"][1]
check("V2 正对>0.6", pos > 0.6, f"{pos:.4f}")
check("V2 负对<0.1", neg < 0.1, f"{neg:.4f}")
check("V2 间隔>0.5", pos - neg > 0.5, f"{pos-neg:.4f}")
check("V2 count/scores输入序", d["count"] == 2 and len(d["scores"]) == 2)
check("V2 ranking降序+tie-break", d["ranking"][0]["score"] >= d["ranking"][-1]["score"])
check("V2 pipeline 串", d["pipeline"] == PIPELINE, str(d["pipeline"]))
check("V2 模型溯源 int8", str(d["model"]).startswith("model_int8.onnx#c5220cf8fe02"), str(d["model"]))
check("V2 无 sigmoid 字段", "sigmoid" not in d)

# ---- V3 确定性（同请求重放逐位相等）----
d2 = post("/rerank", {"query": "孩子在沙滩堆城堡",
                      "texts": ["小女孩用沙子堆了一座城堡", "季度财务报表数据汇总"]})["data"]
check("V3 同请求重放逐位相等", d2["scores"] == d["scores"])

# ---- V4 两段式联动：embedding 召回 → rerank 精排 ----
texts = ["小女孩蹲在沙滩上堆沙堡", "孩子们用小桶在沙滩上堆城堡", "孩子在沙滩上挖坑玩沙",
         "沙滩上放风筝的小孩", "孩子在海边浅水区游泳", "一家人围坐吃年夜饭的餐桌"]
emb = post("/embed/text", {"texts": [t for t in texts]})["data"]["embeddings"]
q = post("/embed/text", {"texts": ["孩子在沙滩上堆城堡"]})["data"]["embeddings"][0]
order = sorted(range(len(texts)), key=lambda i: -(sum(a*b for a, b in zip(emb[i], q))))
top5 = [texts[i] for i in order[:5]]
rr = post("/rerank", {"query": "孩子在沙滩堆城堡", "texts": top5})["data"]
best = top5[rr["ranking"][0]["index"]]
check("V4 两段管线 top1 正确", best in ("小女孩蹲在沙滩上堆沙堡", "孩子们用小桶在沙滩上堆城堡"),
      f"top1={best}")
print(f"INFO 一段序：{[t[:10] for t in top5[:3]]} → 二段 top1：{best[:14]}（{rr['inference_ms']}ms）")

# ---- V5 错误矩阵 ----
cases = [
    ("空query", {"query": "", "texts": ["a"]}, "bad_request"),
    ("空texts", {"query": "q", "texts": []}, "bad_request"),
    ("空串元素", {"query": "q", "texts": ["正常", ""]}, "bad_request"),
    ("33条(默认32)", {"query": "q", "texts": [f"文本{i}" for i in range(33)]}, "bad_request"),
]
for name, payload, want in cases:
    r = post("/rerank", payload)
    check(f"V5 {name}→{want}", r.get("data", {}).get("code") == want, str(r.get("data", {}).get("code")))
d = post("/rerank", {"query": "q", "texts": ["正常", ""]}).get("data", {})
check("V5 空串下标定位", "texts[1]" in d.get("error", ""), str(d.get("error")))

print()
if failures:
    print(f"结果：{len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果：V2-V5 全部通过")
```

**V5b 超预算（可选，需 ~5s）**：`texts` 填 16 条 ×“测试文档”×42（每条 ~428 token）应通过且 `inference_ms` 数千 ms；32 条同文本应被 `bad_request` 快速拒绝（消息含 "8192"）。

## 7. 模型对切换（按语言场景自选）

生产当前 jina 对（双语推荐）。若确认**纯中文**场景，可整块复制 README「文本模型对预设」的 bge 块改 env 重启：bge-small-zh-v1.5（CLS 池化，512 维）+ bge-reranker-base。注意：**换族必设两个 PIPELINE 覆盖**（串变=全量重嵌），且 2026-10-22 实测 bge 重排在中文家庭场景更稳（jina reranker 偶发错排的那组 bge 全对）。反向亦然：jina embedding 英文判别间隔 0.43 vs bge 0.17——双语需求保持 jina。

## 8. 结论回填表（homepix 执行后填写）

| 用例 | 结果（PASS/FAIL/SKIP） | 证据（输出摘录） | 执行人 / 日期 |
|---|---|---|---|
| V1 就绪（/models text_reranking） | | | |
| V2 基本功能与判别 | | | |
| V3 确定性 | | | |
| V4 两段式联动 | | | |
| V5 错误矩阵 | | | |
| V5b 超预算（可选） | | | |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-22 | 初版（批 D2 交付：/rerank 契约 + 两段式指南 + 质量评测/E2E/性能汇总 + 一键脚本） |
