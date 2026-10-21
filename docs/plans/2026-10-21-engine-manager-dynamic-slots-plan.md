# EngineManager 动态化（检测族命名槽 + 热重载）实施计划

> **执行方式**：逐任务执行（executing-plans 技能）；每任务按"测试先行 → 实现 → 门禁 → 提交"推进，完成即在下方跟踪清单回填状态与验证证据（时点必须写）。
> **执行状态**：✅ 全部完成（2026-10-21）——T0-T13 逐任务门禁通过并回填；worktree `feat/dynamic-slots` 已 `--no-ff` 合并回 main（`6ad4bd5`），worktree 与分支已清理；生产未激活（实例无 slots.toml，零回归）。
> **设计权威**：[../designs/2026-10-21-engine-manager-dynamic-slots-design.md](../designs/2026-10-21-engine-manager-dynamic-slots-design.md)（approved，基线 `5b07e63`）。本计划不重述设计论证，只给可执行步骤；与设计冲突处以设计为准并回改本计划。

**Goal:** 检测族新增模型从"改代码 + 发版"退化为"`slots.toml` 加一段 + 热重载"，静态 3 槽与现有契约零回归。

**Architecture:** 新增纯增量模块 `src/slots.rs`（泛型 `SlotRegistry<T>`：解析/校验/懒加载/失败退避/热重载 diff）；静态 15 个 `OnceCell` 字段一律不动（设计 C1）。三个检测端点与 `/detect/batch` 增加可选 `model` 参数，缺省走原路径。watcher 轮询 `slots.toml` 指纹（mtime 纳秒 + size），解析成功才 diff 应用，失败保留 last-known-good。

**Tech Stack:** Rust / axum 0.8 / tokio OnceCell+Semaphore / 新增直接依赖 `toml = "0.8"`。测试分四层：单测（`T=String` 假 builder 覆盖并发/退避/diff）、路由测试（oneshot）、lifecycle 进程级测试（`tests/lifecycle.rs`）、F4 真实模型 E2E + golden 逐字段复比。

**共享机纪律（每个涉及编译的步骤都适用）：**
- cargo 一律 `--jobs 6`（`cargo install` 也要显式传）；
- argus 编译/测试前 `source .env.build`（toolchain-root/PATH/libatomic/openssl）；
- 验证命令全量捕获输出（如 `2>&1 | grep -cE "warning|error"`），不用单行过滤交差；
- 门禁：`cargo test --jobs 6` 全绿 + `cargo clippy --all-targets --jobs 6` 零警告。

**环境事实（2026-10-21 核对）：**
- 仓库 `/home/ncpe/dongit/argus`，HEAD `5b07e63`，main 领先 origin/main 16 个提交（未推送，用户指令：暂不推代码服务器）；
- `.env.build` / `.env.runtime` / `testdata/` / `models/` 均 gitignore——**worktree 是全新检出，这些文件必须手工复制**；
- 完整 18 键模型 env 在 `~/.config/warden/argus.env`（仓库 `.env.runtime` 缺 4 个模型键，以 warden 版为准）；`models_dir` 是绝对路径 `/home/ncpe/dongit/argus/models`，worktree 无需复制模型；
- E2E 测试图 `testdata/bus.jpg`；生产实例 :7042/:7043 由 warden 托管，**本计划 E2E 一律用临时端口 :7049，不动生产**。

---

## 任务跟踪清单（执行时回填）

| # | 任务 | 内容 | 阶段门禁 | 状态 | 验证证据（时点） |
|---|---|---|---|---|---|
| T0 | 准备 | 文档批次提交 main + 建 worktree/分支 | 分支就绪、基线测试全绿 | ✅ | main `481ede6`（docs 批次）+ worktree `feat/dynamic-slots`；基线 `5b07e63`+docs 全绿（2026-10-21 上午，worktree 首次编译） |
| T1 | F0 特征测试 | 静态路径特征单测 + `/models` `/health` golden 锁定 | 现有测试基线全绿；golden 断言通过 | ✅ | `769b253`（12:21）：+6 特征/golden 单测，clippy 0 |
| T2 | F0 E2E golden | 真实模型实例抓 6 端点 golden 入库 | golden 6 文件入库且非空 | ✅ | `705591f`（12:40）：:7049 临时实例抓 6 文件，`yolov8n.onnx#<hash12>` 校验通过 |
| T3 | F1 解析校验 | `toml` 依赖 + `slots.rs` 解析/校验/路径边界 | 解析单测先红后绿 | ✅ | `e7182c4`（12:45）：8 个解析单测（含符号链接逃逸/取值域/截断拒绝） |
| T4 | F1 错误码与配置 | `UnknownModel` 错误码 + `ARGUS_SLOTS_*` | 单测 + clippy 零警告 | ✅ | `c7c1a7f`（13:03）+ 设计勘误 `a15b363`；config/error 单测全绿 |
| T5 | F1 注册表核心 | `Slot<T>`/`SlotRegistry<T>` 懒加载/并发单建/退避 | 并发/退避单测通过 | ✅ | `2e2c27a`（13:17）：8 并发首请求只构建 1 次；退避窗口内 0 重试、过期 1 次 |
| T6 | F1 热重载 diff | `apply_specs` diff + 文件指纹 + `reload_once` | diff/last-known-good 单测通过 | ✅ | `6bec15e`（13:47）：同规格重放不重建、变更换新槽旧 Arc 仍服务、坏文件保留 last-known-good、文件消失清空 |
| T7 | F2 引擎接线 | EngineManager 字段/build 闭包/动态槽解析与深检 | clippy 零警告；编译通过 | ✅ | `6bba5f8`（14:06）+ 设计勘误 `439b773`（保留字分流）；clippy 0 |
| T8 | F2 路由 model 参数 | 三端点 + batch 可选 `model`；静态路径零改动 | 路由测试通过；batch 旧拒绝测试按新语义重写 | ✅ | `3420f8a`（14:34）：detection/scene/batch 路由测试全绿（含空 model 回落静态） |
| T9 | F2 观测字段 | `/models` `dynamic_slots` + `/health` deep 动态字段 | 字段单测通过 | ✅ | `3059c26`（15:00）：全信封 golden 同步 `dynamic_slots:{}`（"只加不改"契约）；浅检无 dynamic 字段 |
| T10 | F3 watcher 装配 | main 启动 fail-closed + `watch_loop` | 手动冒烟：touch 生效 ≤2 周期 | ✅ | `03b09a2`（15:25）：watch=1s 冒烟，热加槽后 `/models` 出现两个槽 |
| T11 | F3 lifecycle | 非法 slots 拒启 + SIGTERM watcher 干净退出 | lifecycle 测试通过 | ✅ | `ee08e36`（17:02）：`--test lifecycle` 3 个测试全过（6.00s） |
| T12 | F4 E2E | 真实模型动态槽全流程 + golden 逐字段复比 | 设计 §10 验收 1-8 全过 | ✅ | 15:00-17:00 真实模型 :7049 全流程：溯源 `detect-slot-v1`、热增删 unknown_model、坏模型只构建 1 次；golden 6 端点复比 2 处 diff 经独立对照实验定性为懒加载时序偏差（零推理复比逐字节一致）；golden mask 修正 `c686489`。**验收 1-10 全过**（含 T11 覆盖的 9 与各任务门禁覆盖的 10） |
| T13 | 文档与收尾 | README/example/计划表回填 + 合并 main + 全量门禁 | 全量 test+clippy 绿；合并完成 | ✅ | 本任务：`slots.example.toml` + README 动态槽小节/env 两行/golden 约定 + 三计划表回填；worktree 最终门禁 `cargo test --jobs 6` 258 全绿（255 lib + 3 lifecycle）、clippy 0；合并 main 提交 `6ad4bd5`（--no-ff），main 复验 258 全绿（2026-10-21 晚） |

---

### Task 0: 准备——文档批次提交与 worktree

**Files:**
- Modify: 无代码；git 操作 + 复制 gitignore 文件
- Commit: `docs: 文档规范/动态槽设计/实施计划/索引（吸收阶段 2）`

**Step 1** 在主检出 `/home/ncpe/dongit/argus` 提交待入库文档（规范、设计、计划、索引、AGENTS.md 指针）：

```bash
cd /home/ncpe/dongit/argus
git add AGENTS.md docs/README.md docs/conventions/ docs/designs/ docs/plans/
git commit -m "docs: 文档规范/动态槽设计/实施计划/索引（吸收阶段 2）"
git log --oneline -1   # 确认提交
```

**Step 2** 本特性跨 ≥5 文件、≥3 提交，按全局纪律用 worktree 隔离：

```bash
cd /home/ncpe/dongit/argus
git worktree add ../argus-dynamic-slots -b feat/dynamic-slots
# gitignore 文件手工带过去（worktree 不含它们）
cp .env.build .env.runtime ../argus-dynamic-slots/
cp -r testdata ../argus-dynamic-slots/
ls ../argus-dynamic-slots/{.env.build,.env.runtime,testdata/bus.jpg}
```

**Step 3** 基线验证（worktree 全新 target，首次编译较久）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo test --jobs 6 2>&1 | tail -5
# 预期：全部通过、0 failed（基线 5b07e63 + 文档提交）
```

**Step 4** 无代码改动，不提交。门禁：分支 `feat/dynamic-slots` 就绪、基线全绿。

---

### Task 1: F0 特征测试——静态路径行为锁定（单测层）

重构前把静态 3 槽的关键行为钉死。只加测试，不改产品代码。

**Files:**
- Modify: `src/engine.rs`（tests 模块追加）
- Modify: `src/routes/models.rs`（tests 模块追加）
- Modify: `src/routes/health.rs`（tests 模块追加）

**Step 1** `src/engine.rs` tests 模块追加 ppe/fire_smoke 白名单先于文件系统的特征测试（镜像现有 detection 版本）：

```rust
    // ---- F0 特征测试：静态检测族 3 槽行为锁定（动态化重构前的安全网） ----

    #[tokio::test]
    async fn ppe_loader_should_return_capability_disabled_when_not_whitelisted() {
        let config = Config {
            ppe_model: Some("/nonexistent/ppe.onnx".into()),
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.ppe().await.err().expect("白名单外应返回错误");
        assert!(matches!(err, AppError::CapabilityDisabled(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn fire_smoke_loader_should_return_capability_disabled_when_not_whitelisted() {
        let config = Config {
            fire_smoke_model: Some("/nonexistent/fs.onnx".into()),
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.fire_smoke().await.err().expect("白名单外应返回错误");
        assert!(matches!(err, AppError::CapabilityDisabled(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn static_detect_slots_should_error_model_not_configured_by_default() {
        // 缺省配置下三槽都必须是 ModelNotConfigured（懒加载入口未破坏）
        let mgr = EngineManager::new(Config::default());
        for result in [mgr.detection().await, mgr.ppe().await, mgr.fire_smoke().await] {
            assert!(matches!(result, Err(AppError::ModelNotConfigured(_))), "实际: {result:?}");
        }
    }
```

**Step 2** `src/routes/models.rs` tests 模块追加全信封 golden（`/models` 无 version/uptime 字段，完全确定，可整体等值比较）：

```rust
    // ---- F0 golden：/models 全信封形状锁定（动态化只允许"加字段"） ----

    #[tokio::test]
    async fn models_envelope_with_default_config_should_be_exact_golden() {
        let app = crate::routes::health::build_router(test_state(Config::default()), None);
        let resp = app
            .oneshot(Request::get("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = body_json(resp).await;
        let mut expected = json!({"status": "success", "data": {"capabilities": {}}});
        for key in [
            "detection", "ppe", "fire_smoke", "classification", "obb", "segmentation",
            "face_detection", "face_recognition", "reid", "ocr", "license_plate", "qr",
            "pose", "depth", "human_parsing",
        ] {
            expected["data"]["capabilities"][key] = json!({
                "configured": false, "enabled": true, "loaded": false, "models": [],
            });
        }
        // pipeline 是每能力必带字段（A4），共 15 个非空互异串——只锁"detection"一个具体值防漂移
        expected["data"]["capabilities"]["detection"]["pipeline"] =
            json!("yolov8-det640-letterbox-v1");
        let caps = body["data"]["capabilities"].as_object().expect("capabilities 应为对象");
        let pipelines: Vec<&str> = caps
            .iter()
            .filter_map(|(k, v)| {
                let p = v.get("pipeline")?.as_str()?;
                assert!(!p.is_empty(), "{k} 的 pipeline 不得为空");
                Some(p)
            })
            .collect();
        let uniq: std::collections::HashSet<&str> = pipelines.iter().copied().collect();
        assert_eq!(pipelines.len(), 15, "应为 15 项能力");
        assert_eq!(uniq.len(), 15, "pipeline 必须互不相同");
        expected["data"]["capabilities"].as_object_mut().unwrap().iter_mut().for_each(|(k, v)| {
            if k != "detection" {
                v["pipeline"] = body["data"]["capabilities"][k]["pipeline"].clone();
            }
        });
        assert_eq!(body, expected, "/models 默认信封与 golden 不一致");
    }
```

> 注意：该测试同时锁 `capabilities` 恰 15 键且 pipeline 全非空互异（消费方失效判定依赖）。

**Step 3** `src/routes/health.rs` tests 模块追加浅检/深检形状 golden（volatile 字段替换哨兵后整体等值）：

```rust
    // ---- F0 golden：/health 形状锁定（动态化只允许 deep 增字段） ----

    #[tokio::test]
    async fn health_shallow_envelope_should_be_exact_golden() {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let mut body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        // volatile 字段替换哨兵后整体等值
        body["version"] = json!("X");
        body["uptime_s"] = json!(0);
        body["backpressure"] = json!({
            "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
            "rejected_total": 0, "acquire_timeout_total": 0,
        });
        assert_eq!(
            body,
            json!({
                "status": "healthy", "device": "cpu", "engines": [],
                "version": "X", "uptime_s": 0,
                "backpressure": {
                    "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
                    "rejected_total": 0, "acquire_timeout_total": 0,
                },
            }),
            "浅检信封与 golden 不一致"
        );
    }

    #[tokio::test]
    async fn health_deep_with_detection_configured_should_be_exact_golden() {
        let config = Config {
            detection_model: Some(std::path::PathBuf::from("/models/yolov8n.onnx")),
            ..Default::default()
        };
        let mgr = EngineManager::new(config.clone());
        let app = build_router(AppState::new(config, mgr), None);
        let resp = app
            .oneshot(Request::get("/health?deep=true").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let mut body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        body["version"] = json!("X");
        body["uptime_s"] = json!(0);
        body["backpressure"] = json!({
            "max_concurrent": 0, "in_flight": 0, "queued_total": 0,
            "rejected_total": 0, "acquire_timeout_total": 0,
        });
        assert_eq!(body["configured"], json!(["detection"]), "configured 应恰含 detection");
        assert_eq!(body["engine_health"], json!([]), "未加载引擎深检应为空数组");
    }
```

**Step 4** 运行（新测试应直接通过——特征测试是锁定现状，不是驱动新行为）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo test --jobs 6 2>&1 | tail -5        # 预期：全绿（新增 6 个测试通过）
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
```

**Step 5** 提交：

```bash
git add src/engine.rs src/routes/models.rs src/routes/health.rs
git commit -m "test: 静态检测族特征测试与 /models //health golden 锁定（吸收阶段 2 前置）"
```

---

### Task 2: F0 E2E golden 基线（真实模型实例）

**Files:**
- Create: `tests/golden/{models,health-deep,detect,detect-batch,ppe,fire-smoke}.json`
- Commit: `test(e2e): 静态检测族端点 golden 基线入库（F0）`

**Step 1** 构建并起临时实例（:7049，warden 完整 env，不碰生产）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo build --release --jobs 6 2>&1 | tail -2      # 预期 Finished
set -a; source ~/.config/warden/argus.env; set +a
ARGUS_PORT=7049 ./target/release/argus > /tmp/argus-golden.log 2>&1 &
ARGUS_PID=$!
sleep 3 && curl -s localhost:7049/health -H "X-API-Key: $ARGUS_AUTH_TOKEN" | jq -r .status   # 预期 healthy
```

**Step 2** 抓 golden。顺序固定：先 health-deep（背压计数全 0）与 models（loaded 全 false），再推理端点：

```bash
mkdir -p tests/golden
KEY=$ARGUS_AUTH_TOKEN
curl -s "localhost:7049/health?deep=true" -H "X-API-Key: $KEY" \
  | jq -S 'del(.version,.uptime_s) | .backpressure |= {}' > tests/golden/health-deep.json
curl -s "localhost:7049/models" -H "X-API-Key: $KEY" | jq -S . > tests/golden/models.json
curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" \
  | jq -S 'del(.data.inference_ms,.data.queue_ms)' > tests/golden/detect.json
curl -s -X POST localhost:7049/detect/batch -H "X-API-Key: $KEY" \
  -H "content-type: application/json" -d '{"image_paths":["testdata/bus.jpg"]}' \
  | jq -S 'del(.data.elapsed_ms,.data.results[]?.inference_ms)' > tests/golden/detect-batch.json
curl -s -X POST localhost:7049/ppe -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" \
  | jq -S 'del(.data.inference_ms,.data.queue_ms)' > tests/golden/ppe.json
curl -s -X POST localhost:7049/fire-smoke -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" \
  | jq -S 'del(.data.inference_ms,.data.queue_ms)' > tests/golden/fire-smoke.json
kill $ARGUS_PID
```

**Step 3** 校验 golden 有效（每个文件都应 `status=ok/success` 且模型串非空）：

```bash
for f in tests/golden/*.json; do
  jq -e '.status != "error"' "$f" > /dev/null || { echo "BAD $f"; exit 1; }
done
jq -r '.data.model' tests/golden/detect.json    # 预期 yolov8n.onnx#<12位hash>
wc -c tests/golden/*.json                        # 预期均非零
```

> 若后续 F4 复比出现 detection 浮点抖动（CPU 线程调度导致的末位差异），把 mask 升级为
> `(.data.detections[]?.confidence) |= (. * 1000 | round / 1000)` 并在两个时点用同一 mask 重抓——须在证据里注明。

**Step 4** 提交：

```bash
git add tests/golden/
git commit -m "test(e2e): 静态检测族端点 golden 基线入库（F0）"
```

---

### Task 3: F1 `slots.toml` 解析与校验

**Files:**
- Modify: `Cargo.toml`（依赖）
- Create: `src/slots.rs`
- Modify: `src/main.rs`（`mod slots;`）
- Commit: `feat(slots): slots.toml 解析与校验（槽名/保留字/路径边界/取值域）`

**Step 1** `Cargo.toml` `[dependencies]` 追加（`serde_json` 之后）：

```toml
toml = "0.8"
```

**Step 2** `src/main.rs` 模块声明追加一行：`mod slots;`

**Step 3** 先写失败测试——创建 `src/slots.rs`，只放测试所需最小骨架（类型签名 + `todo!()` 会拖累编译；直接写"测试 + 空实现编译不过"的形态：先写完整测试，实现函数先返回 `Err("未实现".into())` 的桩）：

```rust
//! 检测族动态槽：`slots.toml` 显式清单 + 懒加载 + 失败退避 + 热重载 diff。
//!
//! 设计权威：`docs/designs/2026-10-21-engine-manager-dynamic-slots-design.md`。
//! 静态 3 槽（detection/ppe/fire_smoke 的 OnceCell）不在本模块管理——registry
//! 纯增量（设计 C1），`slots.toml` 缺失时动态槽为空、行为零回归。

use std::path::{Path, PathBuf};

use crate::config::CAPABILITY_KEYS;

/// 动态槽管线标识（`data.pipeline`），与静态能力键的串不同族。
pub const SLOT_PIPELINE: &str = "detect-slot-v1";

/// 校验后的槽规格（热重载 diff 的比较单元）。
#[derive(Debug, Clone, PartialEq)]
pub struct SlotSpec {
    pub name: String,
    /// 已解析的绝对路径（符号链接已按真实路径判定，见 `resolve_within`）。
    pub model_path: PathBuf,
    pub class_names: Option<Vec<String>>,
    pub pool_size: Option<usize>,
    pub warmup_size: Option<(u32, u32)>,
}

/// 槽名规则：小写字母开头，仅小写字母/数字/下划线，长度 1..=32。
pub fn is_valid_slot_name(name: &str) -> bool {
    todo!()
}

/// 解析并校验 `slots.toml` 文本。任一槽非法 → 整份拒绝（不部分应用）。
pub fn parse_slots_toml(text: &str, models_dir: &Path) -> Result<Vec<SlotSpec>, String> {
    let _ = (text, models_dir);
    todo!()
}
```

同文件追加测试模块（后续任务继续在此模块累加）：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 每个测试独立的临时 models 目录（含一个真实存在的假模型文件）。
    fn tmp_models(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("argus-slots-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
        dir
    }

    #[test]
    fn valid_minimal_and_full_slots_should_parse() {
        let dir = tmp_models("parse-ok");
        let specs = parse_slots_toml(
            "[slots.crack]\nmodel = \"m.onnx\"\n",
            &dir,
        ).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "crack");
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join("m.onnx"));
        assert!(specs[0].class_names.is_none() && specs[0].pool_size.is_none());

        let specs = parse_slots_toml(
            "[slots.ppe_sh17]\nmodel = \"m.onnx\"\nclass_names = [\"Hardhat\", \"NO-Hardhat\"]\npool_size = 2\nwarmup_size = [640, 640]\n",
            &dir,
        ).unwrap();
        assert_eq!(specs[0].class_names, Some(vec!["Hardhat".into(), "NO-Hardhat".into()]));
        assert_eq!(specs[0].pool_size, Some(2));
        assert_eq!(specs[0].warmup_size, Some((640, 640)));
    }

    #[test]
    fn reserved_and_invalid_names_should_be_rejected() {
        let dir = tmp_models("parse-name");
        for bad in ["ppe", "detection", "Fire", "1x", "a-b", "a".repeat(33).as_str()] {
            let toml_text = format!("[slots.\"{bad}\"]\nmodel = \"m.onnx\"\n");
            assert!(parse_slots_toml(&toml_text, &dir).is_err(), "{bad} 不应通过");
        }
    }

    #[test]
    fn path_escape_should_be_rejected() {
        let dir = tmp_models("parse-escape");
        let outside = dir.parent().unwrap().join("outside.onnx");
        std::fs::write(&outside, b"x").unwrap();
        // 相对路径逃逸
        assert!(parse_slots_toml("[slots.a]\nmodel = \"../outside.onnx\"\n", &dir).is_err());
        // 绝对路径在目录外
        assert!(parse_slots_toml(
            &format!("[slots.a]\nmodel = \"{}\"\n", outside.display()), &dir,
        ).is_err());
        // 符号链接指向目录外（真实路径判定）
        #[cfg(unix)]
        {
            let link = dir.join("alias.onnx");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            assert!(parse_slots_toml("[slots.a]\nmodel = \"alias.onnx\"\n", &dir).is_err());
        }
        // 绝对路径在目录内 → 允许
        let inside = dir.canonicalize().unwrap().join("m.onnx");
        assert!(parse_slots_toml(&format!("[slots.a]\nmodel = \"{}\"\n", inside.display()), &dir).is_ok());
    }

    #[test]
    fn not_yet_existing_model_inside_dir_should_be_allowed() {
        // 槽允许先注册后放模型：文件缺失在首次请求构建时报错（设计 §6），路径约束仍须生效
        let dir = tmp_models("parse-missing");
        let specs = parse_slots_toml("[slots.future]\nmodel = \"not_yet.onnx\"\n", &dir).unwrap();
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join("not_yet.onnx"));
    }

    #[test]
    fn value_ranges_should_be_enforced() {
        let dir = tmp_models("parse-range");
        for bad in [
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 0\n",
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 65\n",
            "[slots.a]\nmodel = \"m.onnx\"\nwarmup_size = [16, 640]\n",
            "[slots.a]\nmodel = \"m.onnx\"\nwarmup_size = [640]\n",
            "[slots.a]\nmodel = \"m.onnx\"\nclass_names = []\n",
            "[slots.a]\nmodel = \"m.onnx\"\nclass_names = [\"\"]\n",
            "[slots.a]\nmodel = \"\"\n",
        ] {
            assert!(parse_slots_toml(bad, &dir).is_err(), "应拒绝: {bad}");
        }
    }

    #[test]
    fn truncated_or_duplicate_toml_should_be_rejected() {
        let dir = tmp_models("parse-bad");
        assert!(parse_slots_toml("[slots.crack\nmodel = ", &dir).is_err());
        assert!(parse_slots_toml(
            "[slots.a]\nmodel = \"m.onnx\"\n[slots.a]\nmodel = \"m.onnx\"\n", &dir,
        ).is_err(), "重复槽名应被 toml 解析拒绝");
    }

    #[test]
    fn unknown_field_should_be_ignored_and_empty_file_means_no_slots() {
        let dir = tmp_models("parse-fwd");
        assert!(parse_slots_toml("[slots.a]\nmodel = \"m.onnx\"\nextra = 1\n", &dir).is_ok(),
            "未知字段 warn 后忽略（向前兼容）");
        assert!(parse_slots_toml("", &dir).unwrap().is_empty());
        assert!(parse_slots_toml("[other]\nx = 1\n", &dir).unwrap().is_empty(), "无 [slots] 段 = 空清单");
    }
}
```

**Step 4** 运行确认失败（`todo!()` panic 或编译错误均算"红"）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo test --jobs 6 slots:: 2>&1 | tail -5    # 预期：FAILED / panics
```

**Step 5** 实现解析与校验（替换两个 `todo!()`）：

```rust
/// toml 原始槽（serde 反序列化；未知字段在表级 warn 后忽略）。
#[derive(Debug, serde::Deserialize)]
struct SlotRaw {
    model: String,
    class_names: Option<Vec<String>>,
    pool_size: Option<usize>,
    warmup_size: Option<Vec<u32>>,
}

/// 槽的已知字段（未知字段 warn 不阻断，向前兼容）。
const SLOT_FIELDS: [&str; 4] = ["model", "class_names", "pool_size", "warmup_size"];

pub fn is_valid_slot_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    match bytes.first() {
        Some(b'a'..=b'z') => {}
        _ => return false,
    }
    bytes.len() <= 32
        && bytes[1..].iter().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

pub fn parse_slots_toml(text: &str, models_dir: &Path) -> Result<Vec<SlotSpec>, String> {
    let value: toml::Value = text.parse().map_err(|e| format!("slots.toml 解析失败: {e}"))?;
    let empty = toml::Value::Table(Default::default());
    let slots_table = value
        .get("slots")
        .unwrap_or(&empty)
        .as_table()
        .ok_or_else(|| "slots 必须是 [slots.<name>] 表".to_string())?;

    let mut specs = Vec::with_capacity(slots_table.len());
    for (name, raw_value) in slots_table {
        if !is_valid_slot_name(name) {
            return Err(format!("槽名 {name:?} 非法（须 ^[a-z][a-z0-9_]{{0,31}}$）"));
        }
        if CAPABILITY_KEYS.contains(&name.as_str()) {
            return Err(format!("槽名 {name:?} 是能力键保留字，请换名"));
        }
        let raw_table = raw_value.as_table().ok_or_else(|| format!("槽 {name} 必须是表"))?;
        for key in raw_table.keys() {
            if !SLOT_FIELDS.contains(&key.as_str()) {
                tracing::warn!("slots.toml 槽 {name} 含未知字段 {key}（忽略，向前兼容）");
            }
        }
        let raw: SlotRaw = raw_value
            .clone()
            .try_into()
            .map_err(|e| format!("槽 {name} 字段无效: {e}"))?;
        if raw.model.trim().is_empty() {
            return Err(format!("槽 {name} 缺 model"));
        }
        let model_path = resolve_within(models_dir, raw.model.trim())
            .map_err(|e| format!("槽 {name} 的 model 路径越界: {e}"))?;
        if let Some(names) = &raw.class_names {
            if names.is_empty() || names.iter().any(|s| s.trim().is_empty()) {
                return Err(format!("槽 {name} 的 class_names 不能含空项"));
            }
        }
        if let Some(p) = raw.pool_size {
            if !(1..=64).contains(&p) {
                return Err(format!("槽 {name} 的 pool_size {p} 越界（1..=64）"));
            }
        }
        if let Some(size) = &raw.warmup_size {
            if size.len() != 2 || !size.iter().all(|v| (32..=4096).contains(v)) {
                return Err(format!("槽 {name} 的 warmup_size 须为 [w,h] 且各维 32..=4096"));
            }
        }
        specs.push(SlotSpec {
            name: name.clone(),
            model_path,
            class_names: raw.class_names,
            pool_size: raw.pool_size,
            warmup_size: raw.warmup_size.map(|v| (v[0], v[1])),
        });
    }
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(specs)
}

/// 解析槽模型路径并强制落在 models_dir 内（符号链接按真实路径判定）。
/// 相对路径基于 models_dir；绝对路径也必须落在该目录内。
fn resolve_within(models_dir: &Path, model: &str) -> Result<PathBuf, String> {
    let candidate = if Path::new(model).is_absolute() {
        PathBuf::from(model)
    } else {
        models_dir.join(model)
    };
    let dir_canon = std::fs::canonicalize(models_dir)
        .map_err(|e| format!("models_dir {} 不可用: {e}", models_dir.display()))?;
    let resolved = best_canonicalize(&candidate);
    if !resolved.starts_with(&dir_canon) {
        return Err(format!("模型路径必须在 models_dir({}) 内", dir_canon.display()));
    }
    Ok(resolved)
}

/// canonicalize 尽力而为版：文件可能尚未就位，逐级上溯取最深已存在祖先做真实
/// 解析、余下分量原样拼回（尾部不存在 ≠ 越界；越界由符号链接解析后的祖先判定）。
fn best_canonicalize(path: &Path) -> PathBuf {
    match std::fs::canonicalize(path) {
        Ok(p) => p,
        Err(_) => match path.parent() {
            Some(parent) => {
                let mut p = best_canonicalize(parent);
                if let Some(file) = path.file_name() {
                    p.push(file);
                }
                p
            }
            None => path.to_path_buf(),
        },
    }
}
```

**Step 6** 运行通过 + clippy：

```bash
cargo test --jobs 6 slots:: 2>&1 | tail -3      # 预期：全部通过
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
```

**Step 7** 提交：`git add Cargo.toml Cargo.lock src/slots.rs src/main.rs && git commit -m "feat(slots): slots.toml 解析与校验（槽名/保留字/路径边界/取值域）"`

---

### Task 4: F1 `unknown_model` 错误码与 `ARGUS_SLOTS_*` 配置

**Files:**
- Modify: `src/error.rs`
- Modify: `src/config.rs`
- Commit: `feat(slots): unknown_model 错误码与 ARGUS_SLOTS_* 配置`

**Step 1** `src/error.rs` tests 追加（先红——`UnknownModel` 尚不存在，编译失败即红）：

```rust
    #[test]
    fn unknown_model_should_have_stable_code_and_not_retryable() {
        let err = AppError::UnknownModel("nope（可用动态槽: a, b）".into());
        assert_eq!(err.code(), "unknown_model");
        assert!(!err.retryable());
    }
```

并把现有 `every_variant_should_have_distinct_stable_code` 的 samples 数组追加一项
`AppError::UnknownModel("".into()),`（锁"互异码表"契约覆盖新变体）。

**Step 2** `src/config.rs` tests 追加（沿用 `env_lock` 串行模式）：

```rust
    // ---- 动态槽配置（slots.toml 路径与热重载周期） ----

    #[test]
    fn slots_defaults_should_be_models_dir_and_3s_watch() {
        let config = Config::default();
        assert_eq!(config.slots_config, None);
        assert_eq!(config.slots_watch_secs, 3);
        assert_eq!(config.slots_path(), PathBuf::from("./models/slots.toml"));
    }

    #[test]
    fn from_env_should_parse_slots_overrides() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_SLOTS_CONFIG", "/srv/argus/slots.toml");
            env::set_var("ARGUS_SLOTS_WATCH_SECS", "5");
        }
        let config = Config::from_env();
        assert_eq!(config.slots_config.as_deref(), Some(std::path::Path::new("/srv/argus/slots.toml")));
        assert_eq!(config.slots_watch_secs, 5);
        assert_eq!(config.slots_path(), PathBuf::from("/srv/argus/slots.toml"));
        unsafe {
            env::remove_var("ARGUS_SLOTS_CONFIG");
            env::remove_var("ARGUS_SLOTS_WATCH_SECS");
        }
    }

    #[test]
    fn from_env_should_keep_zero_watch_to_disable_hot_reload() {
        let _guard = env_lock();
        unsafe { env::set_var("ARGUS_SLOTS_WATCH_SECS", "0"); }
        assert_eq!(Config::from_env().slots_watch_secs, 0, "0 必须原样保留（关闭 watcher）");
        unsafe { env::remove_var("ARGUS_SLOTS_WATCH_SECS"); }
    }
```

**Step 3** 运行确认红：

```bash
cargo test --jobs 6 error:: config:: 2>&1 | tail -3   # 预期：编译错误（UnknownModel/slots 字段不存在）
```

**Step 4** 实现。`src/error.rs`：枚举追加变体（`TooLarge` 之后）：

```rust
    /// `model` 参数指向未注册/已删除的动态槽。消息内嵌可用槽名清单，
    /// 不泄露文件系统路径（槽名来自 URL，是外部输入）。
    #[error("未知模型槽: {0}")]
    UnknownModel(String),
```

`code()` 追加 arm：`AppError::UnknownModel(_) => "unknown_model",`；
`into_response` 的 debug 日志 arm 改为
`AppError::BadRequest(_) | AppError::TooLarge(_) | AppError::UnknownModel(_) => tracing::debug!(...)`。

`src/config.rs`：`Config` 结构体追加两个字段（`enabled_capabilities` 之前）：

```rust
    /// slots.toml 显式路径覆盖（`ARGUS_SLOTS_CONFIG`）。None = 默认 `$ARGUS_MODELS_DIR/slots.toml`。
    pub slots_config: Option<PathBuf>,
    /// slots.toml 轮询周期秒（`ARGUS_SLOTS_WATCH_SECS`）。默认 3；0 = 关闭热重载（仅启动时加载）。
    pub slots_watch_secs: u64,
```

`Default` 追加 `slots_config: None, slots_watch_secs: 3,`；
`from_env` 在 `acquire_timeout_ms` 块后追加：

```rust
        if let Ok(v) = env::var("ARGUS_SLOTS_WATCH_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.slots_watch_secs = n;
            }
        }
        cfg.slots_config = env_path("ARGUS_SLOTS_CONFIG");
```

`impl Config` 追加 helper：

```rust
    /// slots.toml 有效路径：显式覆盖优先，否则 $ARGUS_MODELS_DIR/slots.toml。
    pub fn slots_path(&self) -> PathBuf {
        self.slots_config
            .clone()
            .unwrap_or_else(|| self.models_dir.join("slots.toml"))
    }
```

**Step 5** 运行 + clippy + 提交：

```bash
cargo test --jobs 6 error:: config:: 2>&1 | tail -3      # 预期全绿
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/error.rs src/config.rs
git commit -m "feat(slots): unknown_model 错误码与 ARGUS_SLOTS_* 配置"
```

---

### Task 5: F1 `Slot<T>` / `SlotRegistry<T>`——懒加载、并发单建、失败退避

**Files:**
- Modify: `src/slots.rs`（追加核心结构）
- Modify: `src/engine.rs`（`sha256_prefix_12` 改 `pub(crate)`，一行）
- Commit: `feat(slots): SlotRegistry 懒加载/并发单建/失败退避`

> 与设计 §4.1 草图的偏差（记录）：`Slot.last_error` 用 `Mutex<Option<String>>` 而非
> `Option<(String, Instant)>`——`Instant` 无消费点，退避时间已由 `next_retry_at_ms`
> （epoch 毫秒，AtomicU64）承载。语义不变。

**Step 1** `src/slots.rs` tests 追加（先写测试；`SlotRegistry` 尚不存在 → 编译红）：

```rust
    // ---- SlotRegistry 核心（T=String 假 builder，不依赖模型文件） ----

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use crate::engine::EnginePool;
    use crate::error::AppError;

    fn counting_builder() -> (Arc<AtomicUsize>, Arc<dyn Fn(&SlotSpec, rust_onnx_infer::DeviceType) -> Result<EnginePool<String>, AppError> + Send + Sync>) {
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let f = move |spec: &SlotSpec, _d| {
            c.fetch_add(1, Ordering::SeqCst);
            // 放大竞争窗口：并发首请求若未互斥，会明显多次构建
            std::thread::sleep(Duration::from_millis(50));
            Ok(EnginePool::new(vec![Arc::new(spec.name.clone())]))
        };
        (count, Arc::new(f))
    }

    fn one_spec(dir: &Path, name: &str) -> SlotSpec {
        parse_slots_toml(&format!("[slots.{name}]\nmodel = \"m.onnx\"\n"), dir)
            .unwrap()
            .pop()
            .unwrap()
    }

    #[tokio::test]
    async fn resolve_should_build_once_under_concurrency() {
        let dir = tmp_models("registry-concur");
        let (count, build) = counting_builder();
        let registry = Arc::new(SlotRegistry::new(
            rust_onnx_infer::DeviceType::from_name("cpu"), build,
        ));
        registry.apply_specs(vec![one_spec(&dir, "concur")]);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let r = Arc::clone(&registry);
            handles.push(tokio::spawn(async move { r.resolve("concur").await }));
        }
        for h in handles {
            h.await.unwrap().expect("应构建成功");
        }
        assert_eq!(count.load(Ordering::SeqCst), 1, "8 个并发首请求必须只构建一次");
    }

    #[tokio::test]
    async fn resolve_unknown_slot_should_return_unknown_model_error() {
        let (_c, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        let err = registry.resolve("nope").await.err().expect("未注册槽应报错");
        assert!(matches!(err, AppError::UnknownModel(msg) if msg.contains("无已注册动态槽")), "实际: {err:?}");
    }

    #[tokio::test]
    async fn failed_build_should_backoff_and_retry_after_window() {
        let dir = tmp_models("registry-backoff");
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let registry = SlotRegistry::new(
            rust_onnx_infer::DeviceType::from_name("cpu"),
            Arc::new(move |_: &SlotSpec, _d| {
                c.fetch_add(1, Ordering::SeqCst);
                Err(AppError::Internal("坏模型".into()))
            }),
        );
        // 测试用短退避（生产默认 30s，见 SLOT_RETRY_BACKOFF）
        let mut registry = registry;
        registry.set_retry_backoff_for_test(Duration::from_millis(80));
        registry.apply_specs(vec![one_spec(&dir, "bad")]);

        assert!(registry.resolve("bad").await.is_err(), "首次构建失败");
        let err = registry.resolve("bad").await.err().expect("退避窗口内应拒绝");
        assert!(err.to_string().contains("退避"), "应提示退避中: {err}");
        assert_eq!(count.load(Ordering::SeqCst), 1, "退避窗口内不得重试构建");

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(registry.resolve("bad").await.is_err(), "窗口过期后重试仍失败（模型还是坏的）");
        assert_eq!(count.load(Ordering::SeqCst), 2, "窗口过期后应重试一次");
    }

    #[tokio::test]
    async fn slot_model_info_should_report_filename_hash_and_slot_pipeline() {
        let dir = tmp_models("registry-prov");
        let (_c, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        registry.apply_specs(vec![one_spec(&dir, "prov")]);
        let slot = registry.slot("prov").unwrap();
        let mi = slot.model_info().await.expect("文件存在应可解析");
        assert!(mi.model.starts_with("m.onnx#") && mi.model.len() == "m.onnx#".len() + 12, "实际: {}", mi.model);
        assert_eq!(mi.pipeline, SLOT_PIPELINE);
    }
```

**Step 2** 运行确认红（`SlotRegistry` 未定义 → 编译失败）：

```bash
cargo test --jobs 6 slots:: 2>&1 | tail -3
```

**Step 3** 实现。`src/slots.rs` 头部 use 扩充：

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_onnx_infer::DeviceType;
use tokio::sync::OnceCell;

use crate::config::CAPABILITY_KEYS;
use crate::engine::{sha256_prefix_12, EnginePool, ModelInfo};
use crate::error::AppError;
```

追加核心结构：

```rust
/// 槽构建失败退避窗口（生产值）。
pub const SLOT_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// 单个动态槽：懒加载池 + 溯源缓存 + 失败退避状态。
pub struct Slot<T> {
    pub spec: SlotSpec,
    pool: OnceCell<Arc<EnginePool<T>>>,
    model_info_cache: OnceCell<ModelInfo>,
    last_error: std::sync::Mutex<Option<String>>,
    /// 失败退避截止（epoch 毫秒）；0 = 无退避（从未失败）。
    next_retry_at_ms: std::sync::atomic::AtomicU64,
}

impl<T> Slot<T> {
    fn new(spec: SlotSpec) -> Self {
        Self {
            spec,
            pool: OnceCell::new(),
            model_info_cache: OnceCell::new(),
            last_error: std::sync::Mutex::new(None),
            next_retry_at_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn loaded(&self) -> bool {
        self.pool.initialized()
    }

    /// 已加载池（未加载返回 None；深检与快路径用）。
    pub fn pool(&self) -> Option<Arc<EnginePool<T>>> {
        self.pool.get().cloned()
    }

    /// 槽级溯源：`文件名#sha256 前 12 位` + `detect-slot-v1`，首查后缓存。
    /// 文件不可读返回 None（响应 model/pipeline 为 null，与静态未配置能力同语义）。
    pub async fn model_info(&self) -> Option<ModelInfo> {
        if let Some(hit) = self.model_info_cache.get() {
            return Some(hit.clone());
        }
        let path = self.spec.model_path.clone();
        let computed = tokio::task::spawn_blocking(move || {
            let name = path.file_name()?.to_string_lossy().to_string();
            let hash = sha256_prefix_12(&path)?;
            Some(ModelInfo { model: format!("{name}#{hash}"), pipeline: SLOT_PIPELINE })
        })
        .await
        .ok()
        .flatten()?;
        let _ = self.model_info_cache.set(computed.clone());
        Some(computed)
    }
}

/// 动态槽注册表。生产实例是 `SlotRegistry<DetectionEngine>`（装配在
/// `EngineManager`，见 engine.rs）；单测注入假 builder（T=String）覆盖
/// 并发/退避/diff 行为，不依赖模型文件。
pub struct SlotRegistry<T> {
    slots: std::sync::RwLock<HashMap<String, Arc<Slot<T>>>>,
    /// 构建互斥（1 permit）：坏模型风暴时全局最多一个构建在跑。
    build_sem: Arc<tokio::sync::Semaphore>,
    device: DeviceType,
    build_fn: Arc<dyn Fn(&SlotSpec, DeviceType) -> Result<EnginePool<T>, AppError> + Send + Sync>,
    retry_backoff: Duration,
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl<T: Send + Sync + 'static> SlotRegistry<T> {
    pub fn new(
        device: DeviceType,
        build_fn: Arc<dyn Fn(&SlotSpec, DeviceType) -> Result<EnginePool<T>, AppError> + Send + Sync>,
    ) -> Self {
        Self {
            slots: std::sync::RwLock::new(HashMap::new()),
            build_sem: Arc::new(tokio::sync::Semaphore::new(1)),
            device,
            build_fn,
            retry_backoff: SLOT_RETRY_BACKOFF,
        }
    }

    /// 热重载 diff 应用（Task 6 实现，先占位保编译）。
    pub fn apply_specs(&self, _specs: Vec<SlotSpec>) {
        todo!()
    }

    /// 按名取槽（不触发构建）。
    pub fn slot(&self, name: &str) -> Option<Arc<Slot<T>>> {
        self.slots
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    /// 全部槽快照，按槽名排序（/models、深检用）。
    pub fn slots_snapshot(&self) -> Vec<Arc<Slot<T>>> {
        let mut v: Vec<_> = self
            .slots
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect();
        v.sort_by(|a, b| a.spec.name.cmp(&b.spec.name));
        v
    }

    /// 请求路径解析：快路径命中已加载池；未加载则退避检查 → 构建互斥 → 构建。
    pub async fn resolve(&self, name: &str) -> Result<(Arc<Slot<T>>, Arc<EnginePool<T>>), AppError> {
        let slot = self
            .slot(name)
            .ok_or_else(|| AppError::UnknownModel(self.unknown_model_message(name)))?;
        if let Some(pool) = slot.pool() {
            return Ok((slot, pool));
        }
        // 失败退避：窗口内不重试（防坏模型请求风暴）
        let now = now_epoch_ms();
        let until = slot.next_retry_at_ms.load(std::sync::atomic::Ordering::Relaxed);
        if now < until {
            let last = slot
                .last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
                .unwrap_or_else(|| "未知错误".into());
            return Err(AppError::Internal(format!(
                "动态槽 {} 此前加载失败（{last}），退避中，{}ms 后可重试",
                slot.spec.name,
                until - now
            )));
        }
        // 构建互斥：排队后复检——并发首请求只建一份
        let _build_permit = self
            .build_sem
            .acquire()
            .await
            .map_err(|_| AppError::Internal("槽构建信号量已关闭".into()))?;
        if let Some(pool) = slot.pool() {
            return Ok((slot, pool));
        }
        let spec = slot.spec.clone();
        let build_fn = Arc::clone(&self.build_fn);
        let device = self.device;
        let result = tokio::task::spawn_blocking(move || build_fn(&spec, device))
            .await
            .map_err(|e| AppError::Internal(format!("槽构建任务失败: {e}")))?;
        match result {
            Ok(pool) => {
                let pool = Arc::new(pool);
                let _ = slot.pool.set(Arc::clone(&pool));
                Ok((slot, pool))
            }
            Err(e) => {
                *slot.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e.to_string());
                slot.next_retry_at_ms.store(
                    now_epoch_ms() + self.retry_backoff.as_millis() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                tracing::error!(
                    "动态槽 {} 构建失败（退避 {:?}）: {e}",
                    slot.spec.name, self.retry_backoff
                );
                Err(e)
            }
        }
    }

    fn unknown_model_message(&self, name: &str) -> String {
        let names = self
            .slots_snapshot()
            .iter()
            .map(|s| s.spec.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        if names.is_empty() {
            format!("{name}（当前无已注册动态槽，请检查 slots.toml）")
        } else {
            format!("{name}（可用动态槽: {names}）")
        }
    }

    /// 单测专用：缩短退避窗口。
    #[cfg(test)]
    pub(crate) fn set_retry_backoff_for_test(&mut self, d: Duration) {
        self.retry_backoff = d;
    }
}
```

同时 `src/engine.rs:85` 的 `fn sha256_prefix_12` 改为 `pub(crate) fn sha256_prefix_12`。

注意：`slot.pool` 字段与 `Slot::pool()` 方法同名——Rust 允许（字段与方法不同命名空间），但 `slot.pool` 在同模块外不可见（私有），`slots.rs` 内 `Slot` 定义处使用无冲突。若 clippy 报 field-reassign 或风格问题，按提示微调。

**Step 4** 运行（此阶段 `apply_specs` 仍是 `todo!()`，但测试尚未触达热重载路径——`resolve`/`apply_specs` 占位不影响本组测试；若 `counting_builder` 测试触发了 `apply_specs` 的 `todo!()` panic，则先临时在测试里直接构造 `Slot` 插入——**不要**，正确做法是把 `apply_specs` 的真实现提前到本任务，见 Step 3 补充）：

> 补充：`apply_specs` 直接在本任务一并实现（diff 逻辑 10 行，与懒加载强耦合，拆开反而造桩）：

```rust
    /// 热重载 diff 应用：消失移除、新增插入、变更整体替换为未加载新 Slot；
    /// 未变槽原样保留（已加载池与在途请求不受影响）。
    pub fn apply_specs(&self, specs: Vec<SlotSpec>) {
        let mut slots = self.slots.write().unwrap_or_else(|e| e.into_inner());
        slots.retain(|name, _| specs.iter().any(|s| &s.name == name));
        for spec in specs {
            let unchanged = slots.get(&spec.name).map(|s| s.spec == spec).unwrap_or(false);
            if !unchanged {
                slots.insert(spec.name.clone(), Arc::new(Slot::new(spec)));
            }
        }
    }
```

（Task 6 只补测试与 `reload_once`/指纹，不再动这段。）

**Step 5** 运行 + clippy + 提交：

```bash
cargo test --jobs 6 slots:: 2>&1 | tail -3      # 预期全绿
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/slots.rs src/engine.rs
git commit -m "feat(slots): SlotRegistry 懒加载/并发单建/失败退避"
```

---

### Task 6: F1 热重载 diff 与文件指纹

**Files:**
- Modify: `src/slots.rs`
- Commit: `feat(slots): 热重载 diff 应用与文件指纹`

**Step 1** tests 追加：

```rust
    // ---- 热重载 diff 与 reload_once（last-known-good 语义） ----

    use crate::config::Config;

    #[tokio::test]
    async fn unchanged_spec_should_keep_loaded_pool() {
        let dir = tmp_models("diff-keep");
        let (count, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        let spec = one_spec(&dir, "keep");
        registry.apply_specs(vec![spec.clone()]);
        registry.resolve("keep").await.expect("首次加载");
        // 同规格重放：不替换槽、不重建池
        registry.apply_specs(vec![spec]);
        registry.resolve("keep").await.expect("重放后仍可用");
        assert_eq!(count.load(Ordering::SeqCst), 1, "同规格重放不得重建");
    }

    #[tokio::test]
    async fn changed_spec_should_replace_slot_and_old_arc_still_works() {
        let dir = tmp_models("diff-replace");
        let (_c, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        registry.apply_specs(vec![one_spec(&dir, "rep")]);
        let (old_slot, old_pool) = registry.resolve("rep").await.expect("v1 加载");

        let v2 = parse_slots_toml(
            &format!("[slots.rep]\nmodel = \"m.onnx\"\nclass_names = [\"a\"]\n"), &dir,
        ).unwrap().pop().unwrap();
        registry.apply_specs(vec![v2]);
        // 新槽未加载；旧 Arc 仍可用（在途请求语义）
        let new_slot = registry.slot("rep").unwrap();
        assert!(!new_slot.loaded(), "变更后应为未加载新槽");
        assert!(old_slot.loaded(), "旧槽引用视角仍是已加载");
        let v = old_pool.run(|e| Ok::<_, rust_onnx_infer::VisionError>(e.clone())).await.unwrap();
        assert_eq!(v, "rep", "旧池必须继续服务");
    }

    #[tokio::test]
    async fn vanished_spec_should_remove_slot() {
        let dir = tmp_models("diff-rm");
        let (_c, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        registry.apply_specs(vec![one_spec(&dir, "a"), one_spec(&dir, "b")]);
        registry.apply_specs(vec![one_spec(&dir, "b")]);
        assert!(registry.slot("a").is_none());
        assert!(registry.slot("b").is_some());
    }

    fn config_with_slots(dir: &Path, text: &str) -> (Config, PathBuf) {
        let slots_path = dir.join("slots.toml");
        std::fs::write(&slots_path, text).unwrap();
        let config = Config {
            models_dir: dir.to_path_buf(),
            ..Config::default()
        };
        (config, slots_path)
    }

    #[tokio::test]
    async fn reload_once_should_apply_change_reject_bad_and_clear_on_missing() {
        let dir = tmp_models("reload");
        let (_c, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        let (config, slots_path) = config_with_slots(&dir, "[slots.a]\nmodel = \"m.onnx\"\n");
        let mut fp = None;
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("a").is_some(), "首次应加载槽");

        // 坏内容：保留 last-known-good
        std::fs::write(&slots_path, "[slots.'X']\nmodel").unwrap();
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("a").is_some(), "解析失败必须保留旧槽");
        // 截断同样保留
        std::fs::write(&slots_path, "[slots.a\nmod").unwrap();
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("a").is_some());

        // 合法变更：换名生效
        std::fs::write(&slots_path, "[slots.b]\nmodel = \"m.onnx\"\n").unwrap();
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("a").is_none() && registry.slot("b").is_some());

        // 文件消失：清空动态槽
        std::fs::remove_file(&slots_path).unwrap();
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("b").is_none(), "文件消失应清空动态槽");
    }

    #[tokio::test]
    async fn reload_once_should_skip_when_fingerprint_unchanged() {
        let dir = tmp_models("reload-fp");
        let (count, build) = counting_builder();
        let registry = SlotRegistry::new(rust_onnx_infer::DeviceType::from_name("cpu"), build);
        let (config, _p) = config_with_slots(&dir, "[slots.a]\nmodel = \"m.onnx\"\n");
        let mut fp = None;
        reload_once(&registry, &config, &mut fp);
        assert!(registry.slot("a").is_some());
        // 指纹不变（未修改文件）：二次调用必须是 no-op（不读文件、不重放）
        let fp_before = fp;
        reload_once(&registry, &config, &mut fp);
        assert_eq!(fp, fp_before, "指纹不变不得重读");
        assert_eq!(count.load(Ordering::SeqCst), 0, "diff 未变不触发构建");
    }

    #[test]
    fn file_fingerprint_should_track_mtime_and_size() {
        let dir = tmp_models("fingerprint");
        let p = dir.join("probe.toml");
        std::fs::write(&p, b"one").unwrap();
        let f1 = file_fingerprint(&p).expect("存在文件应有指纹");
        std::fs::write(&p, b"one-longer").unwrap();
        let f2 = file_fingerprint(&p).expect("改写后应有指纹");
        assert_ne!(f1, f2, "size 变化必须改变指纹");
        assert!(file_fingerprint(&dir.join("nope.toml")).is_none(), "缺失文件无指纹");
    }
```

**Step 2** 运行确认红（`reload_once`/`file_fingerprint` 未定义）。

**Step 3** 实现（`src/slots.rs` 追加；`Config` 已在 use 中引入）：

```rust
/// 文件指纹：mtime 纳秒 + size。同名同 size 但 mtime 变化（touch）也触发重读。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileFingerprint {
    pub mtime_ns: u128,
    pub size: u64,
}

pub fn file_fingerprint(path: &Path) -> Option<FileFingerprint> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    Some(FileFingerprint {
        mtime_ns: mtime.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos(),
        size: meta.len(),
    })
}

/// 热重载单轮：指纹变化才读取解析；解析失败保留 last-known-good（不部分应用，
/// 但指纹仍前进——避免每周期对同一份坏文件重复解析刷屏）；文件消失清空动态槽。
pub fn reload_once<T: Send + Sync + 'static>(
    registry: &SlotRegistry<T>,
    config: &crate::config::Config,
    last_fp: &mut Option<FileFingerprint>,
) {
    let path = config.slots_path();
    let fp = file_fingerprint(&path);
    if fp == *last_fp {
        return;
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => match parse_slots_toml(&text, &config.models_dir) {
            Ok(specs) => {
                tracing::info!("slots.toml 重载成功: {} 个动态槽", specs.len());
                registry.apply_specs(specs);
                *last_fp = fp;
            }
            Err(e) => {
                tracing::error!("slots.toml 重载失败(保留旧配置): {e}");
                *last_fp = fp;
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!("slots.toml 已移除: 清空动态槽(静态能力不受影响)");
            registry.apply_specs(Vec::new());
            *last_fp = None;
        }
        Err(e) => {
            tracing::error!("slots.toml 读取失败(保留旧配置): {e}");
            *last_fp = fp;
        }
    }
}
```

（测试 use 里 `use crate::config::Config;` 与文件级 use 合并即可。）

**Step 4** 运行 + clippy + 提交：

```bash
cargo test --jobs 6 slots:: 2>&1 | tail -3
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/slots.rs && git commit -m "feat(slots): 热重载 diff 应用与文件指纹"
```

> F1 门禁自查：解析/校验/退避/并发/diff/last-known-good 单测全绿、clippy 零警告。对照设计 §8.2 完成度：✅。

---

### Task 7: F2 EngineManager 接线（装配、动态槽解析、深检）

**Files:**
- Modify: `src/engine.rs`
- Commit: `feat(slots): EngineManager 动态槽装配与动态深检`

**Step 1** tests 追加（`src/engine.rs` tests 模块）——先红：

```rust
    // ---- 动态槽接线（F2） ----

    use crate::slots::{parse_slots_toml, SLOT_PIPELINE};

    fn registry_with_slot(dir: &std::path::Path, name: &str) -> EngineManager {
        let config = Config {
            models_dir: dir.to_path_buf(),
            ..Config::default()
        };
        let mgr = EngineManager::new(config.clone());
        std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
        mgr.slots.apply_specs(
            parse_slots_toml(&format!("[slots.{name}]\nmodel = \"m.onnx\"\n"), dir).unwrap(),
        );
        mgr
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("argus-em-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn resolve_dynamic_slot_should_reject_reserved_names() {
        let mgr = EngineManager::new(Config::default());
        for key in ["detection", "ppe", "fire_smoke"] {
            let err = mgr.resolve_dynamic_slot(key).await.err().expect("保留字应拒绝");
            assert!(matches!(err, AppError::BadRequest(_)), "{key} 实际: {err:?}");
            assert!(err.to_string().contains("端点"), "应提示改用端点: {err}");
        }
    }

    #[tokio::test]
    async fn resolve_dynamic_slot_should_return_unknown_model_when_unregistered() {
        let mgr = EngineManager::new(Config::default());
        let err = mgr.resolve_dynamic_slot("nope").await.err().expect("未注册应拒绝");
        assert!(matches!(err, AppError::UnknownModel(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn resolve_dynamic_slot_should_gate_on_detection_whitelist() {
        // 白名单不含 detection：所有动态检测槽拒绝（设计 §4.6），即使槽已注册
        let dir = tmp_dir("wl");
        let config = Config {
            models_dir: dir.clone(),
            enabled_capabilities: Some(vec!["ppe".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
        mgr.slots.apply_specs(parse_slots_toml("[slots.x]\nmodel = \"m.onnx\"\n", &dir).unwrap());
        let err = mgr.resolve_dynamic_slot("x").await.err().expect("应被白名单拒绝");
        assert!(matches!(err, AppError::CapabilityDisabled(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn dynamic_loaded_and_deep_health_should_reflect_registry() {
        let dir = tmp_dir("deep");
        let mgr = registry_with_slot(&dir, "probe");
        assert_eq!(mgr.dynamic_loaded(), Vec::<String>::new(), "未构建应为空");
        let health = mgr.dynamic_slot_health().await;
        assert!(health.is_empty(), "未加载槽不进深检");
        // 槽级溯源走 registry（不构建引擎也能取）
        let slot = mgr.slots.slot("probe").unwrap();
        let mi = slot.model_info().await.unwrap();
        assert_eq!(mi.pipeline, SLOT_PIPELINE);
    }
```

**Step 2** 运行确认红（`mgr.slots` / `resolve_dynamic_slot` 等未定义）。

**Step 3** 实现。`src/engine.rs`：

头部追加 use：`use crate::slots::{Slot, SlotRegistry, SlotSpec};`

`EngineManager` 结构体追加字段（`model_registry` 之后）：

```rust
    /// 检测族动态槽（slots.toml）。静态 3 槽保持 OnceCell——registry 纯增量（设计 C1）。
    pub(crate) slots: SlotRegistry<DetectionEngine>,
```

`EngineManager::new` 装配（`EngineManager {` 之前构造 registry，字段列表追加 `slots,`）：

```rust
    pub fn new(config: Config) -> Self {
        let device = DeviceType::from_name(&config.device);
        let slots = Self::build_slot_registry(device, config.engine_pool);
        EngineManager {
            config,
            device,
            detection: tokio::sync::OnceCell::new(),
            /* …15 个 OnceCell 原样… */
            model_registry: std::sync::RwLock::new(std::collections::HashMap::new()),
            slots,
        }
    }

    /// 生产 build 闭包：与静态检测槽同参数（低阈值 + 全局 NMS IoU），
    /// class_names 注入（数量不符仅 warn），预热失败 warn 不阻断（设计 §4.2/§4.4）。
    fn build_slot_registry(device: DeviceType, default_pool: usize) -> SlotRegistry<DetectionEngine> {
        let build = move |spec: &SlotSpec, device: DeviceType| -> Result<EnginePool<DetectionEngine>, AppError> {
            let pool_size = spec.pool_size.unwrap_or(default_pool).max(1);
            let mut engines = Vec::with_capacity(pool_size);
            for _ in 0..pool_size {
                let mut engine = DetectionEngine::for_yolo(
                    &spec.model_path, device, DETECTION_CONFIDENCE_FLOOR, DETECTION_NMS_IOU,
                )?;
                if let Some(names) = &spec.class_names {
                    // 自训模型常缺 metadata：数量不符仅 warn，以 class_names 为准
                    if let Some(model_labels) = engine.base.labels() {
                        if model_labels.len() != names.len() {
                            tracing::warn!(
                                "槽 {} class_names({}) 与模型 labels({}) 数量不一致，以 class_names 为准",
                                spec.name, names.len(), model_labels.len()
                            );
                        }
                    }
                    engine.base.set_labels(names.clone());
                }
                engines.push(Arc::new(engine));
            }
            let pool = EnginePool::new(engines);
            // 预热：每实例 dummy 推理，失败 warn 不阻断。本闭包运行于 spawn_blocking，
            // 同步执行与静态槽 EnginePool::warmup 的阻塞路径语义等价
            let (w, h) = spec.warmup_size.unwrap_or((640, 640));
            for (idx, engine) in pool.instances.iter().enumerate() {
                match engine.predict(&dummy_image(w as i32, h as i32)).map(|_| ()) {
                    Ok(()) => tracing::info!("槽 {} 实例 {}/{} 预热完成", spec.name, idx + 1, pool_size),
                    Err(e) => tracing::warn!("槽 {} 实例 {}/{} 预热失败(不阻断): {e}", spec.name, idx + 1, pool_size),
                }
            }
            Ok(pool)
        };
        SlotRegistry::new(device, Arc::new(build))
    }
```

`impl EngineManager` 追加（`deep_health_check` 之后）：

```rust
    /// 检测族动态槽解析（三端点与 /detect/batch 的 `model` 参数入口）。
    /// 保留字（15 能力键）→ BadRequest 提示改用对应端点；白名单按 detection
    /// 门控（设计 §4.6）；未注册槽 → UnknownModel（列出可用槽名，不泄露路径）。
    pub async fn resolve_dynamic_slot(
        &self,
        name: &str,
    ) -> Result<(Arc<Slot<DetectionEngine>>, Arc<EnginePool<DetectionEngine>>), AppError> {
        if crate::config::CAPABILITY_KEYS.contains(&name) {
            return Err(AppError::BadRequest(format!(
                "model={name} 是静态能力键：请使用对应端点，或去掉 model 走该端点预置槽"
            )));
        }
        self.ensure_enabled("detection")?;
        self.slots.resolve(name).await
    }

    /// 已加载动态槽名（/health deep 的 dynamic_engines，字典序）。
    pub fn dynamic_loaded(&self) -> Vec<String> {
        self.slots
            .slots_snapshot()
            .into_iter()
            .filter(|s| s.loaded())
            .map(|s| s.spec.name)
            .collect()
    }

    /// 动态槽深检：对每个已加载槽发 dummy 推理（与 check_engine! 同语义，
    /// 未加载槽跳过）。槽名是 String，与静态 engine_health 的 &'static str 分列
    /// 两个字段（设计 C3）。
    pub async fn dynamic_slot_health(&self) -> Vec<crate::slots::DynamicSlotHealth> {
        let mut results = Vec::new();
        for slot in self.slots.slots_snapshot() {
            let Some(pool) = slot.pool() else { continue };
            let status = pool
                .run(|e| e.predict(&dummy_image(640, 640)).map(|_| ()))
                .await;
            results.push(crate::slots::DynamicSlotHealth {
                name: slot.spec.name.clone(),
                status: if status.is_ok() { "ok" } else { "degraded" },
                error: status.err().map(|e| e.to_string()),
            });
        }
        results
    }
```

`src/slots.rs` 追加深检 DTO（`SLOT_PIPELINE` 常量旁）：

```rust
/// 动态槽深检条目（/health?deep=true 的 dynamic_slot_health）。
#[derive(Debug, serde::Serialize)]
pub struct DynamicSlotHealth {
    pub name: String,
    pub status: &'static str, // "ok" | "degraded"
    pub error: Option<String>,
}
```

**Step 4** 运行 + clippy + 提交：

```bash
cargo test --jobs 6 engine:: 2>&1 | tail -3
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/engine.rs src/slots.rs && git commit -m "feat(slots): EngineManager 动态槽装配与动态深检"
```

---

### Task 8: F2 路由 `model` 参数（三端点 + batch）

**Files:**
- Modify: `src/routes/detection.rs`
- Modify: `src/routes/batch.rs`
- Commit: `feat(detect): 三端点与 /detect/batch 可选 model 参数（动态槽）`

**Step 1** 路由测试先写。

`src/routes/detection.rs` tests 追加：

```rust
    // ---- 动态槽 model 参数（F2） ----

    #[tokio::test]
    async fn detect_with_unknown_model_returns_unknown_model_envelope() {
        let text = post_detect(multipart_png_with_fields(&[("model", "nope")])).await;
        assert!(text.contains("\"code\":\"unknown_model\""), "{text}");
        assert!(text.contains("无已注册动态槽"), "{text}");
    }

    #[tokio::test]
    async fn detect_with_reserved_model_name_returns_bad_request_hint() {
        let text = post_detect(multipart_png_with_fields(&[("model", "ppe")])).await;
        assert!(text.contains("静态能力键"), "{text}");
        assert!(text.contains("端点"), "{text}");
    }

    #[tokio::test]
    async fn detect_with_empty_model_field_should_be_treated_as_absent() {
        // 前端表单常发空串：视为未传，走静态路径（未配模型 → 模型未配置错误）
        let text = post_detect(multipart_png_with_fields(&[("model", "")])).await;
        assert!(text.contains("未配置") || text.contains("model"), "空 model 应走静态路径: {text}");
        assert!(!text.contains("unknown_model"), "空串不是 unknown_model: {text}");
    }
```

`src/routes/batch.rs` 测试**改造**：删除 `batch_with_model_param_is_rejected_not_ignored`（旧语义），替换为：

```rust
    #[tokio::test]
    async fn batch_with_model_unknown_slot_returns_unknown_model() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": "nope"}"#).await;
        assert!(text.contains("\"code\":\"unknown_model\""), "{text}");
    }

    #[tokio::test]
    async fn batch_with_reserved_model_returns_bad_request_hint() {
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": "detection"}"#).await;
        assert!(text.contains("静态能力键") && text.contains("端点"), "{text}");
    }

    #[tokio::test]
    async fn batch_with_empty_model_should_fall_back_to_static() {
        // 空 model = 未传：走静态 detection（未配模型 → 整批"模型未配置"）
        let (_, text) = post_batch(r#"{"image_paths": ["/tmp/a.jpg"], "model": ""}"#).await;
        assert!(text.contains("未配置") || text.contains("model"), "{text}");
        assert!(!text.contains("unknown_model"), "{text}");
    }
```

**Step 2** 运行确认红（`model` 字段被忽略 → 断言失败）。

**Step 3** 实现。

`src/routes/detection.rs`：

- `DetectParams` 追加字段 `model: Option<String>,`；`parse_detect_multipart` 的局部变量与 match 追加：

```rust
            "model" => {
                let text = field_text(field).await?;
                if !text.trim().is_empty() {
                    model = Some(text.trim().to_string());
                }
            }
```

- `run_detect` 中 `let pool = cap.pool(&state).await?;` 替换为：

```rust
    /// 池来源：静态能力槽（缺省）或动态命名槽（model 参数）。
    /// 静态路径逐字节不变（设计 C4：只加动态分支）。
    enum DetectSource {
        Static(&'static str),
        Dynamic(std::sync::Arc<crate::slots::Slot<rust_onnx_infer::engines::detection::DetectionEngine>>),
    }

    let (pool, source) = match params.model.as_deref() {
        None => (cap.pool(&state).await?, DetectSource::Static(cap.key())),
        Some(name) => {
            let (slot, pool) = state.engines.resolve_dynamic_slot(name).await?;
            (pool, DetectSource::Dynamic(slot))
        }
    };
```

（`DetectSource` 放在 `run_detect` 外的模块层。）

- 末尾溯源替换：

```rust
    let (model, pipeline) = match &source {
        DetectSource::Static(key) => state.engines.provenance(key).await,
        DetectSource::Dynamic(slot) => match slot.model_info().await {
            Some(m) => (serde_json::json!(m.model), serde_json::json!(m.pipeline)),
            None => (serde_json::Value::Null, serde_json::Value::Null),
        },
    };
```

`src/routes/batch.rs`：

- 模块头注释第 9 行（"`model` 参数待动态模型管理…"）改为
  `//! - `model` 可选：命名动态槽（slots.toml），缺省用预置 detection 槽；`
- `BatchRequest.model` 注释改为 `/// 可选动态槽名（slots.toml 注册）；缺省用预置 detection 槽`
- 删除"传入即拒绝"块，`image_paths` 空检查之后、`roi/filter` 解析之后替换池获取为：

```rust
    // 模型未配置是整批级错误（无引擎可用），不做逐图隔离；动态槽在入口解析一次，
    // 逐图复用同一池与同一溯源（单图失败仍按现有语义隔离）
    let model = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let (pool, dyn_slot) = match model {
        None => (state.engines.detection().await?, None),
        Some(name) => {
            let (slot, pool) = state.engines.resolve_dynamic_slot(name).await?;
            (pool, Some(slot))
        }
    };
```

- 响应溯源替换：

```rust
    let (model, pipeline) = match &dyn_slot {
        Some(slot) => match slot.model_info().await {
            Some(m) => (json!(m.model), json!(m.pipeline)),
            None => (Value::Null, Value::Null),
        },
        None => state.engines.provenance("detection").await,
    };
```

**Step 4** 运行（detection + batch 全部，确认旧测试无回归）+ clippy + 提交：

```bash
cargo test --jobs 6 routes::detection routes::batch 2>&1 | tail -3
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/routes/detection.rs src/routes/batch.rs
git commit -m "feat(detect): 三端点与 /detect/batch 可选 model 参数（动态槽）"
```

---

### Task 9: F2 观测字段——`/models` `dynamic_slots` 与 `/health` deep 动态字段

**Files:**
- Modify: `src/routes/models.rs`、`src/routes/health.rs`
- Commit: `feat(models): /models dynamic_slots 与 /health dynamic 深检字段`

**Step 1** 测试先写。

`src/routes/models.rs` tests 追加：

```rust
    // ---- 动态槽观测字段（F2，设计 C2：只加不改） ----

    fn state_with_slot(tag: &str) -> AppState {
        let dir = std::env::temp_dir().join(format!("argus-models-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
        let config = Config {
            models_dir: dir.clone(),
            ..Config::default()
        };
        let mgr = EngineManager::new(config.clone());
        mgr.slots.apply_specs(
            crate::slots::parse_slots_toml("[slots.crack]\nmodel = \"m.onnx\"\n", &dir).unwrap(),
        );
        AppState::new(config, mgr)
    }

    #[tokio::test]
    async fn models_should_always_carry_dynamic_slots_map() {
        // 无槽时也应输出空对象：字段形状稳定（只加不改的"加"）
        let app = crate::routes::health::build_router(test_state(Config::default()), None);
        let resp = app.oneshot(Request::get("/models").body(Body::empty()).unwrap()).await.unwrap();
        let body = body_json(resp).await;
        assert_eq!(body["data"]["dynamic_slots"], serde_json::json!({}), "{body}");
    }

    #[tokio::test]
    async fn models_should_report_slot_status_and_keep_capabilities_untouched() {
        let app = crate::routes::health::build_router(state_with_slot("dyn"), None);
        let resp = app.oneshot(Request::get("/models").body(Body::empty()).unwrap()).await.unwrap();
        let body = body_json(resp).await;
        let slot = &body["data"]["dynamic_slots"]["crack"];
        assert_eq!(slot["configured"], true, "{body}");
        assert_eq!(slot["loaded"], false, "未构建应 loaded=false: {body}");
        assert_eq!(slot["model_file"], "m.onnx", "只报文件名不报路径: {body}");
        assert_eq!(slot["pipeline"], "detect-slot-v1", "{body}");
        // 15 项 capabilities 逐字节不受影响（golden 已锁形状，此处再锁数量）
        assert_eq!(body["data"]["capabilities"].as_object().unwrap().len(), 15, "{body}");
    }
```

`src/routes/health.rs` tests 追加：

```rust
    // ---- 动态槽 deep 字段（F2，设计 C3：engines/engine_health 不变） ----

    #[tokio::test]
    async fn health_deep_should_carry_dynamic_fields_and_shallow_should_not() {
        let app = build_router(test_state(), None);
        let resp = app
            .clone()
            .oneshot(Request::get("/health?deep=true").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body["dynamic_engines"], serde_json::json!([]), "无槽应为空数组");
        assert_eq!(body["dynamic_slot_health"], serde_json::json!([]), "{body}");

        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = body_text(resp).await;
        assert!(!text.contains("dynamic_engines"), "浅检不得新增字段: {text}");
    }
```

**Step 2** 运行确认红。

**Step 3** 实现。

`src/routes/models.rs` `list_models` 末尾返回值改造：

```rust
    // 动态槽清单（设计 C2：独立 dynamic_slots 键，capabilities 的 15 项不动——
    // safebot-go 只读 capabilities.detection.models，语义零变化）
    let mut dynamic = Map::new();
    for slot in state.engines.slots.slots_snapshot() {
        dynamic.insert(
            slot.spec.name.clone(),
            json!({
                "configured": true,
                "loaded": slot.loaded(),
                "model_file": slot
                    .spec
                    .model_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                "pipeline": crate::slots::SLOT_PIPELINE,
            }),
        );
    }
    Json(json!({ "status": "success", "data": { "capabilities": caps, "dynamic_slots": dynamic } }))
```

`src/routes/health.rs` deep 分支追加（`engine_health` 之后）：

```rust
        body["dynamic_engines"] = json!(state.engines.dynamic_loaded());
        body["dynamic_slot_health"] = json!(state.engines.dynamic_slot_health().await);
```

`service_info` 端点描述同步（`/detect` 与 `/detect/batch` 两行加 `model?`——`/ppe` `/fire-smoke` 本就不在清单里，不扩）：

```rust
            "POST /detect": "目标检测（multipart: file/sahi/slice_w/slice_h/overlap/confidence?/target_classes?/roi?/model?[动态槽名]）",
            "POST /detect/batch": "批量检测（JSON: image_paths[服务端本地路径]/confidence?/target_classes?/roi?/model?[动态槽名]，逐图独立背压与错误隔离）",
```

**Step 4** 运行（含 Task 1 的 golden——`/models` 全信封 golden 会因新增 `dynamic_slots` 键**失败**，这是预期内的契约演进：修改 golden 的 expected，追加 `"dynamic_slots": {}`，并在提交信息里注明）：

```bash
cargo test --jobs 6 routes::models routes::health 2>&1 | tail -3
cargo test --jobs 6 2>&1 | tail -3    # 全量：golden 更新后应全绿
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
git add src/routes/models.rs src/routes/health.rs
git commit -m "feat(models): /models dynamic_slots 与 /health dynamic 深检字段（golden 随'只加不改'契约同步）"
```

> F2 门禁自查：路由测试全绿；静态路径无 `model` 时行为与 F0 特征测试一致；batch 旧"传入即拒"测试已被新语义测试替换。对照设计 §8.3：✅。

---

### Task 10: F3 watcher 装配（main 启动 fail-closed + watch_loop）

**Files:**
- Modify: `src/slots.rs`（`watch_loop`）
- Modify: `src/main.rs`
- Commit: `feat(slots): watcher 热重载装配与启动 fail-closed`

**Step 1** `src/slots.rs` 追加（无需单测——循环体即 `reload_once`，已被 T6 覆盖；生命周期由 T11 进程级测试覆盖）：

```rust
/// 启动时解析 slots.toml（fail-closed：错误上抛，main 拒绝启动）。
/// 文件缺失 → 空槽清单（零回归，设计 §4.3）。
pub fn load_specs(config: &crate::config::Config) -> Result<Vec<SlotSpec>, String> {
    let path = config.slots_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            parse_slots_toml(&text, &config.models_dir).map_err(|e| format!("{} 无效: {e}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("读取 {} 失败: {e}", path.display())),
    }
}

/// watcher 主循环（main 中 spawn；slots_watch_secs=0 不启动）。
/// 纯内存应用无清理需求：进程退出时随 runtime 终止（设计风险表 P1——
/// lifecycle 测试在 Task 11 锁定）。
pub async fn watch_loop(
    engines: std::sync::Arc<crate::engine::EngineManager>,
    config: std::sync::Arc<crate::config::Config>,
) {
    let interval = Duration::from_secs(config.slots_watch_secs.max(1));
    let mut last_fp = file_fingerprint(&config.slots_path());
    tracing::info!(
        "slots.toml 热重载已启动(周期 {interval:?}, 路径 {})",
        config.slots_path().display()
    );
    loop {
        tokio::time::sleep(interval).await;
        reload_once(&engines.slots, &config, &mut last_fp);
    }
}
```

**Step 2** `src/main.rs`：模块声明改 `mod slots;`（若 T3 已加则跳过）；`AppState::new` 之后、`build_router` 之前插入：

```rust
    // 动态槽：启动解析 fail-closed（配置错误拒绝启动）；文件缺失等价空槽
    let slot_specs = slots::load_specs(&config).map_err(|e| format!("slots 配置无效: {e}"))?;
    if !slot_specs.is_empty() {
        tracing::info!(
            "动态槽已注册: {:?}",
            slot_specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
        );
    }
    engines.slots.apply_specs(slot_specs);

    let state = AppState::new(config.clone(), engines);
    if state.config.slots_watch_secs > 0 {
        tokio::spawn(slots::watch_loop(
            std::sync::Arc::clone(&state.engines),
            std::sync::Arc::clone(&state.config),
        ));
    }
```

（原 `let state = AppState::new(config, engines);` 行合并进上段。）

**Step 3** 编译 + 全量测试 + 手动冒烟（验证 ≤2 周期生效）：

```bash
cargo build --jobs 6 2>&1 | tail -2 && cargo test --jobs 6 2>&1 | tail -3
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0

# 冒烟：watch=1s 的临时实例
mkdir -p /tmp/argus-slot-smoke && cd /home/ncpe/dongit/argus-dynamic-slots
set -a; source ../argus/.env.build; set +a
set -a; source ~/.config/warden/argus.env; set +a
printf '[slots.alias_smoke]\nmodel = "yolov8n.onnx"\n' > /tmp/argus-slot-smoke/slots.toml
ARGUS_PORT=7049 ARGUS_SLOTS_CONFIG=/tmp/argus-slot-smoke/slots.toml ARGUS_SLOTS_WATCH_SECS=1 \
  ./target/debug/argus > /tmp/argus-slot-smoke/serve.log 2>&1 &
SPID=$!; sleep 3
# 热加一个槽 → ≤2s 后 /models 应出现
printf '[slots.alias_smoke]\nmodel = "yolov8n.onnx"\n[slots.second]\nmodel = "yolov8n.onnx"\n' > /tmp/argus-slot-smoke/slots.toml
sleep 3
curl -s localhost:7049/models -H "X-API-Key: $ARGUS_AUTH_TOKEN" | jq '.data.dynamic_slots | keys'
# 预期输出: ["alias_smoke","second"]
kill $SPID; rm -rf /tmp/argus-slot-smoke
```

**Step 4** 提交：

```bash
git add src/slots.rs src/main.rs && git commit -m "feat(slots): watcher 热重载装配与启动 fail-closed"
```

---

### Task 11: F3 lifecycle 进程级测试

**Files:**
- Modify: `tests/lifecycle.rs`
- Commit: `test(lifecycle): slots 启动拒绝与 SIGTERM watcher 干净退出`

**Step 1** `tests/lifecycle.rs` 追加：

```rust
// ---- 动态槽生命周期（F3） ----
// 自包含 models_dir（假模型文件即可：启动只注册不构建，不发推理请求）。
// 不依赖仓库相对路径 ./models——worktree/CI 检出没有该目录。

#[cfg(unix)]
fn free_port() -> u16 {
    33000 + (std::process::id() % 10000) as u16
}

/// 自包含临时 models_dir + slots.toml（真实存在于 temp，非仓库相对路径）。
#[cfg(unix)]
fn slot_env(tag: &str, slots_text: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("argus-slots-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
    let slots = dir.join("slots.toml");
    std::fs::write(&slots, slots_text).unwrap();
    slots
}

/// 非法 slots.toml（槽名大写）必须拒绝启动（fail-closed，设计 §6）。
#[cfg(unix)]
#[test]
fn invalid_slots_toml_should_fail_startup() {
    let slots = slot_env("lc1", "[slots.BadName]\nmodel = \"m.onnx\"\n");
    let dir = slots.parent().unwrap().to_path_buf();
    let status = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", free_port().to_string())
        .env("ARGUS_MODELS_DIR", &dir)
        .env("ARGUS_SLOTS_CONFIG", &slots)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("启动 argus 测试实例");
    assert!(!status.success(), "非法 slots.toml 必须拒绝启动: {status:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// watcher 活跃时 SIGTERM：进程须在超时上限内干净退出（runtime 终止后台任务，
/// 无残留；优雅停机链不变——设计风险表 P1 / 验收 9）。
#[cfg(unix)]
#[test]
fn sigterm_with_active_watcher_should_exit_cleanly() {
    let slots = slot_env("lc2", "[slots.a]\nmodel = \"m.onnx\"\n");
    let dir = slots.parent().unwrap().to_path_buf();
    let mut child = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", free_port().to_string())
        .env("ARGUS_MODELS_DIR", &dir)
        .env("ARGUS_SLOTS_CONFIG", &slots)
        .env("ARGUS_SLOTS_WATCH_SECS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 argus 测试实例");
    std::thread::sleep(Duration::from_secs(2)); // watcher 至少跑一轮
    Command::new("kill").arg(child.id().to_string()).status().unwrap(); // SIGTERM

    let deadline = std::time::Instant::now() + Duration::from_secs(15); // > shutdown_timeout 默认 10s
    let mut exited = None;
    while std::time::Instant::now() < deadline {
        match child.try_wait().expect("查询进程状态") {
            Some(status) => {
                exited = Some(status);
                break;
            }
            None => std::thread::sleep(Duration::from_millis(200)),
        }
    }
    let status = exited.unwrap_or_else(|| {
        let _ = child.kill();
        panic!("SIGTERM 后 15s 未退出（watcher 阻塞停机）");
    });
    assert_eq!(
        status.code(),
        Some(0),
        "优雅停机应 exit 0（被信号杀则 code=None）: {status:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
```

**Step 2** 运行（`CARGO_BIN_EXE_argus` 需先 `cargo build --jobs 6 --tests`；测试较慢，单独跑）：

```bash
source .env.build && cargo test --jobs 6 --test lifecycle 2>&1 | tail -5   # 预期 3 个测试全过
git add tests/lifecycle.rs && git commit -m "test(lifecycle): slots 启动拒绝与 SIGTERM watcher 干净退出"
```

> F3 门禁自查：watcher 单测（T6）+ lifecycle（本任务）全绿。对照设计 §8.4：✅。

---

### Task 12: F4 E2E——真实模型动态槽全流程 + golden 逐字段复比

**Files:** 无代码改动预期（发现缺陷则 fix commit 并回补单测）。产物是本文件跟踪清单的证据列与（如需）golden 更新说明。

**Step 1** 起验证实例（watch=1s，release 构建）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo build --release --jobs 6 2>&1 | tail -2
set -a; source ~/.config/warden/argus.env; set +a
mkdir -p /tmp/argus-f4
printf '[slots.alias_test]\nmodel = "yolov8n.onnx"\n' > /tmp/argus-f4/slots.toml
ARGUS_PORT=7049 ARGUS_SLOTS_CONFIG=/tmp/argus-f4/slots.toml ARGUS_SLOTS_WATCH_SECS=1 \
  ./target/release/argus > /tmp/argus-f4/serve.log 2>&1 &
F4PID=$!; sleep 3; KEY=$ARGUS_AUTH_TOKEN
```

**Step 2** 验收 2：动态槽推理 + 溯源（`detect-slot-v1`）：

```bash
curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" \
  -F "file=@testdata/bus.jpg" -F "model=alias_test" | jq '{status, model: .data.model, pipeline: .data.pipeline, count: .data.count}'
# 预期：status=ok；model 含 "yolov8n.onnx#<hash12>"；pipeline=="detect-slot-v1"；count 与静态 /detect 一致（同模型同图）
```

**Step 3** 验收 7/8：`/models` `dynamic_slots.alias_test.loaded==true`；`/health?deep=true` 两个新字段且 `engine_health` 不含动态槽：

```bash
curl -s localhost:7049/models -H "X-API-Key: $KEY" | jq '.data.dynamic_slots'
curl -s "localhost:7049/health?deep=true" -H "X-API-Key: $KEY" | jq '{dynamic_engines, dynamic_slot_health}'
```

**Step 4** 验收 3（热重载删除）+ 验收 2（热加）：

```bash
printf '' > /tmp/argus-f4/slots.toml; sleep 3     # 删除全部槽
curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" \
  -F "file=@testdata/bus.jpg" -F "model=alias_test" | jq '{status, code: .data.code}'
# 预期：status=error 且 code=="unknown_model"
printf '[slots.alias_test]\nmodel = "yolov8n.onnx"\nclass_names = ["p0","p1"]\n' > /tmp/argus-f4/slots.toml
sleep 3
curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" \
  -F "file=@testdata/bus.jpg" -F "model=alias_test" | jq -r .status   # 预期 ok（重新可用，warn 类名数量不符在日志）
```

**Step 5** 验收 5（坏模型退避）：

```bash
# 在真实 models 目录临时放垃圾模型（结束后清理；models/ 是 gitignore 部署数据目录）
echo garbage > /home/ncpe/dongit/argus/models/__f4_bad.onnx
printf '[slots.bad]\nmodel = "__f4_bad.onnx"\n' > /tmp/argus-f4/slots.toml; sleep 3
for i in 1 2 3; do
  curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" -F "model=bad" | jq -r '.data.error' | head -c 80; echo
done
# 预期：第 1 次为加载/推理失败；第 2、3 次为"退避中"提示
grep -c "槽 bad 构建失败" /tmp/argus-f4/serve.log   # 预期 1（只构建一次）
rm /home/ncpe/dongit/argus/models/__f4_bad.onnx      # 清理
```

**Step 6** 验收 1（golden 逐字段复比——静态路径零回归）：

```bash
mkdir -p /tmp/argus-f4/after
# 与 Task 2 完全相同的 mask 与顺序；models 的 mask 剔除新增 dynamic_slots（golden 锁的是静态面）；
# health 的 backpressure 已在两个时点统一清空（T2 的 mask 已含 .backpressure |= {}）
curl -s "localhost:7049/health?deep=true" -H "X-API-Key: $KEY" | jq -S 'del(.version,.uptime_s,.dynamic_engines,.dynamic_slot_health) | .backpressure |= {}' > /tmp/argus-f4/after/health-deep.json
curl -s "localhost:7049/models" -H "X-API-Key: $KEY" | jq -S 'del(.data.dynamic_slots)' > /tmp/argus-f4/after/models.json
curl -s -X POST localhost:7049/detect -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" | jq -S 'del(.data.inference_ms,.data.queue_ms)' > /tmp/argus-f4/after/detect.json
curl -s -X POST localhost:7049/detect/batch -H "X-API-Key: $KEY" -H "content-type: application/json" \
  -d '{"image_paths":["testdata/bus.jpg"]}' | jq -S 'del(.data.elapsed_ms,.data.results[]?.inference_ms)' > /tmp/argus-f4/after/detect-batch.json
curl -s -X POST localhost:7049/ppe -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" | jq -S 'del(.data.inference_ms,.data.queue_ms)' > /tmp/argus-f4/after/ppe.json
curl -s -X POST localhost:7049/fire-smoke -H "X-API-Key: $KEY" -F "file=@testdata/bus.jpg" | jq -S 'del(.data.inference_ms,.data.queue_ms)' > /tmp/argus-f4/after/fire-smoke.json
for f in health-deep models detect detect-batch ppe fire-smoke; do
  diff tests/golden/$f.json /tmp/argus-f4/after/$f.json && echo "GOLDEN OK: $f"
done
kill $F4PID; rm -rf /tmp/argus-f4
```

**Step 7** 验收 4/6（并发单建、注入拒绝）已由单测锁定（T5/T3），此处核对清单即可。全量门禁：

```bash
cargo test --jobs 6 2>&1 | tail -3
cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 预期 0
```

把每步实际输出（jq 结果、diff 结果、grep 计数、时间）记入跟踪清单证据列（含时点）。

---

### Task 13: F4 文档回填与合并收尾

**Files:**
- Create: `slots.example.toml`（仓库根，gitignore 之外）
- Modify: `README.md`、`docs/yolo-rs-absorption-plan.md`、`docs/engine-exposure-plan.md`（批 1.6 备注）、`docs/README.md`（状态）、`docs/plans/`（本文件清单回填）
- Commit: `docs(slots): 动态槽使用说明回填（README/示例/计划表）`；合并提交

**Step 1** `slots.example.toml`：

```toml
# 检测族动态槽示例。复制到 $ARGUS_MODELS_DIR/slots.toml（或任意路径 + ARGUS_SLOTS_CONFIG 指向它）。
# 字段与校验规则见 docs/designs/2026-10-21-engine-manager-dynamic-slots-design.md §4.4。
# 换模型文件的约定：原子替换（临时文件 + mv）后 touch 本文件——同路径原地改写不触发重载。

[slots.crack]
model = "yolov8n-crack.onnx"          # 相对 ARGUS_MODELS_DIR；绝对路径也必须落在该目录内
class_names = ["crack", "pothole"]     # 可选：覆盖模型 metadata 的类名（自训模型常缺 metadata）
pool_size = 1                          # 可选：缺省取 ARGUS_ENGINE_POOL，范围 1..=64
warmup_size = [640, 640]               # 可选：预热 dummy 尺寸，两维均 32..=4096

[slots.ppe_sh17]
model = "ppe_sh17_v3.onnx"
class_names = ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]
```

**Step 2** `README.md`：在配置/env 表加 `ARGUS_SLOTS_CONFIG`、`ARGUS_SLOTS_WATCH_SECS` 两行；新增「动态检测槽（slots.toml）」小节（格式示例、`model` 参数用法、错误码 `unknown_model`、热重载语义与原子替换约定、回滚方式=删文件/回滚二进制）。实测数字带时点。

**Step 3** 计划表回填：
- `docs/yolo-rs-absorption-plan.md` 阶段 2 状态改 ✅ 完成，附验证证据（测试计数、E2E 时点）；
- `docs/engine-exposure-plan.md` 批 1.6 加一行备注「动态槽底座已就绪（2026-10-XX，见 designs/…-dynamic-slots-design.md），场景槽接入退化为配置项」；
- `docs/README.md` 设计条目状态与计划状态同步。

**Step 4** 本计划跟踪清单全部回填（状态 + 证据 + 时点）。

**Step 5** 合并回 main（在主检出操作）：

```bash
cd /home/ncpe/dongit/argus-dynamic-slots && source .env.build
cargo test --jobs 6 2>&1 | tail -3 && cargo clippy --all-targets --jobs 6 2>&1 | grep -cE "^(warning|error)"   # 0
cd /home/ncpe/dongit/argus
git merge --no-ff feat/dynamic-slots -m "merge: EngineManager 动态化（检测族命名槽 + 热重载，吸收阶段 2）"
cargo test --jobs 6 2>&1 | tail -3    # main 上复验
git worktree remove ../argus-dynamic-slots && git branch -d feat/dynamic-slots
```

**Step 6** 汇报（按全局纪律）：改动摘要 + 验证证据（实际命令与真实输出）+ 风险/未覆盖点（如 golden 的 backpressure mask 处理、双实例 7042/7043 各自加载的内存提示）。**不推送远程**（用户指令：暂不推代码服务器；GitHub 镜像同步亦待指令）。生产升级（`install` 到 `~/.local/bin` + warden restart）不在本计划内，由用户决定窗口。

---

## 已知取舍与未覆盖点（执行者须知）

1. **`model` 传静态能力键返回 BadRequest 而非跨端点转发**（设计 §3.3）：`/ppe?model=detection` 不会路由到 detection——语义混用是契约坑，宁可报错提示。
2. **同路径原地替换模型文件不触发重载**（设计 C8/§12）：README 与 example 已写"原子替换 + touch"。模型文件 mtime 监听是显式未决项。
3. **watcher 的 mtime 粒度**：同秒同 size 的两次写入理论上可漏检（纳秒 mtime 已极大缓解）；重载失败后指纹仍前进，坏文件修复自身不触发重载——**必须再改一次文件**（touch）才会重试。
4. **双实例内存**：7042/7043 各自懒加载同一槽会双份内存（设计 P1）——懒加载 + 替换不预载已把峰值最小化，README 提示。
5. **golden 的 backpressure/浮点抖动**：T2 的 mask 已剔除已知易变字段；若 diff 仍报差异，按 T12 Step 6 注记的扩展 mask 处理并在证据中注明，不得静默改 golden。
6. **`/ppe` `/fire-smoke` 不在 `service_info` 端点清单**是既有现状，本计划不扩（精准修改）。

## 验收对照（设计 §10 十条 → 计划落点）

| # | 设计验收 | 落点 |
|---|---|---|
| 1 | 无 model 时输出与 golden 逐字段一致 | T1 golden 断言 + T12 Step 6 diff |
| 2 | 新槽 ≤2 周期可用 | T10 冒烟 + T12 Step 2/4 |
| 3 | 删槽 unknown_model 错误信封，静态不受影响 | T12 Step 4 |
| 4 | 并发首请求只建一次 | T5 `resolve_should_build_once_under_concurrency` |
| 5 | 坏模型 30s 退避不崩溃 | T5 backoff 单测 + T12 Step 5 |
| 6 | 注入/截断全部拒绝且保留旧配置 | T3 解析单测 + T6 reload 单测 |
| 7 | /models 15 项不变 + dynamic_slots 正确 | T1/T9 golden + 单测 |
| 8 | /health engines/engine_health 不变 + 新字段 | T1/T9 golden + 单测 |
| 9 | SIGTERM watcher 退出无残留 | T11 lifecycle |
| 10 | test 全绿 + clippy 零警告 | 每任务门禁 + T12/T13 全量 |

## 变更记录

- 2026-10-21：初版（writing-plans 产出）。13 任务对齐设计 F0-F4 五阶段与 C1-C8 修正；任务粒度按"测试先行 → 实现 → 门禁 → 提交"。
- 2026-10-21（执行期）：`a15b363`（T4 审查）统一 unknown_model/保留字错误为信封表述并回改设计；`439b773`（T7 审查）保留字与未注册错误码分流勘误回改设计；`c686489`（T9 审查）T12 health golden 掩码剔除 dynamic 新字段，防验收假失败。
- 2026-10-21（收尾，T13）：`slots.example.toml` 入库；README 增 `ARGUS_SLOTS_CONFIG`/`ARGUS_SLOTS_WATCH_SECS` 两行、「动态检测槽（slots.toml）」小节与 E2E golden 复比约定；吸收计划阶段 2 状态 ⬜→✅（含与初稿形态差异注记）；engine-exposure 批 1.6 加"动态槽底座已就绪"备注；docs/README 索引状态同步；本清单 T0-T13 全部回填。worktree 最终门禁：`cargo test --jobs 6` 258 全绿（255 lib + 3 lifecycle，2026-10-21 晚）、`cargo clippy --all-targets --jobs 6` 0 警告。合并：`git merge --no-ff feat/dynamic-slots` → `6ad4bd5`（main），main 复验 258 全绿 + clippy 0；worktree 与分支清理完成。**未推送远程**（用户指令）。
