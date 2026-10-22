//! 检测族动态槽：`slots.toml` 显式清单 + 懒加载 + 失败退避 + 热重载 diff。
//!
//! 设计权威：`docs/designs/2026-10-21-engine-manager-dynamic-slots-design.md`。
//! 静态 3 槽（detection/ppe/fire_smoke 的 OnceCell）不在本模块管理——registry
//! 纯增量（设计 C1），`slots.toml` 缺失时动态槽为空、行为零回归。

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_onnx_infer::DeviceType;
use tokio::sync::OnceCell;

use crate::config::{CAPABILITY_KEYS, Config};
use crate::engine::{sha256_prefix_12, EnginePool, ModelInfo};
use crate::error::AppError;

/// 动态槽管线标识（`data.pipeline`），与静态能力键的串不同族。
pub const SLOT_PIPELINE: &str = "detect-slot-v1";

/// 动态槽深检条目（/health?deep=true 的 dynamic_slot_health）。
#[derive(Debug, serde::Serialize)]
pub struct DynamicSlotHealth {
    pub name: String,
    pub status: &'static str, // "ok" | "degraded"
    pub error: Option<String>,
}

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

/// 槽名规则：小写字母开头，仅小写字母/数字/下划线，长度 1..=32。
pub fn is_valid_slot_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    match bytes.first() {
        Some(b'a'..=b'z') => {}
        _ => return false,
    }
    bytes.len() <= 32
        && bytes[1..].iter().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

/// 解析并校验 `slots.toml` 文本。任一槽非法 → 整份拒绝（不部分应用）。
pub fn parse_slots_toml(text: &str, models_dir: &Path) -> Result<Vec<SlotSpec>, String> {
    let value: toml::Value = text.parse().map_err(|e| format!("slots.toml 解析失败: {e}"))?;
    let empty = toml::Value::Table(Default::default());
    let slots_table = value
        .get("slots")
        .unwrap_or(&empty)
        .as_table()
        .ok_or_else(|| "slots 必须是 [slots.<name>] 表".to_string())?;
    if slots_table.is_empty() {
        return Ok(Vec::new());
    }
    // models_dir 每份清单只 canonicalize 一次；空清单零文件系统调用（零回归）。
    let dir_canon = std::fs::canonicalize(models_dir)
        .map_err(|e| format!("models_dir {} 不可用: {e}", models_dir.display()))?;

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
        let model = raw.model.trim();
        if model.is_empty() {
            return Err(format!("槽 {name} 缺 model"));
        }
        if Path::new(model).components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(format!("槽 {name} 的 model 路径不允许包含 .. 分量"));
        }
        // 纯内存校验先于文件系统调用（先廉后贵）
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
        let model_path = resolve_within(&dir_canon, model)
            .map_err(|e| format!("槽 {name} 的 model 路径无效: {e}"))?;
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
/// 相对路径基于传入的已规范化目录（dir_canon）；绝对路径也必须落在该目录内。
fn resolve_within(dir_canon: &Path, model: &str) -> Result<PathBuf, String> {
    let candidate = if Path::new(model).is_absolute() {
        PathBuf::from(model)
    } else {
        dir_canon.join(model)
    };
    let resolved = best_canonicalize(&candidate);
    if !resolved.starts_with(dir_canon) {
        return Err(format!("模型路径必须在 models_dir({}) 内", dir_canon.display()));
    }
    Ok(resolved)
}

/// canonicalize 尽力而为版：模型文件可能尚未就位，自底向上找最深已存在祖先做
/// 真实解析，余下分量逆序拼回（尾部不存在 ≠ 越界；符号链接按祖先的真实路径
/// 判定）。迭代实现：入口虽已拒绝 `..` 分量，上万层的深路径仍会打爆递归栈
/// （实测 exit 134 abort），违反热重载"失败保旧配置"契约。
fn best_canonicalize(path: &Path) -> PathBuf {
    let mut current = path;
    let mut tails: Vec<std::ffi::OsString> = Vec::new();
    while let Some(parent) = current.parent() {
        match std::fs::canonicalize(current) {
            Ok(resolved) => {
                let mut full = resolved;
                for t in tails.iter().rev() {
                    full.push(t);
                }
                return full;
            }
            Err(_) => match current.file_name() {
                Some(file) => {
                    tails.push(file.to_os_string());
                    current = parent;
                }
                // 无文件名分量可收集（如根/空路径），放弃上溯按原样返回
                None => break,
            },
        }
    }
    path.to_path_buf()
}

/// 槽构建失败退避窗口（生产值）。
pub const SLOT_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// 单个动态槽：懒加载池 + 溯源缓存 + 失败退避状态。
pub struct Slot<T> {
    pub spec: SlotSpec,
    pool: OnceCell<Arc<EnginePool<T>>>,
    model_info_cache: OnceCell<ModelInfo>,
    /// 上次构建失败信息与退避截止（同一把锁，单调时钟——免疫墙钟跳变；
    /// 此前用 epoch 毫秒 AtomicU64，时钟回拨会拉长退避窗口）。
    failure: std::sync::Mutex<Option<(String, std::time::Instant)>>,
}

impl<T> Slot<T> {
    fn new(spec: SlotSpec) -> Self {
        Self {
            spec,
            pool: OnceCell::new(),
            model_info_cache: OnceCell::new(),
            failure: std::sync::Mutex::new(None),
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
            Some(ModelInfo { model: format!("{name}#{hash}"), pipeline: SLOT_PIPELINE.to_string() })
        })
        .await
        .ok()
        .flatten()?;
        let _ = self.model_info_cache.set(computed.clone());
        Some(computed)
    }
}

/// 槽构建函数：接收规格与设备，产出引擎池。EngineManager 注入真实构建
/// （T=DetectionEngine），单测注入假 builder（T=String），不依赖模型文件。
pub type SlotBuildFn<T> =
    Arc<dyn Fn(&SlotSpec, DeviceType) -> Result<EnginePool<T>, AppError> + Send + Sync>;

/// 动态槽注册表。生产实例是 `SlotRegistry<DetectionEngine>`（装配在
/// `EngineManager`，见 engine.rs）；单测注入假 builder（T=String）覆盖
/// 并发/退避/diff 行为，不依赖模型文件。
pub struct SlotRegistry<T> {
    slots: std::sync::RwLock<HashMap<String, Arc<Slot<T>>>>,
    /// 构建互斥（1 permit）：坏模型风暴时全局最多一个构建在跑。
    build_sem: Arc<tokio::sync::Semaphore>,
    device: DeviceType,
    build_fn: SlotBuildFn<T>,
    retry_backoff: Duration,
}

impl<T: Send + Sync + 'static> SlotRegistry<T> {
    pub fn new(device: DeviceType, build_fn: SlotBuildFn<T>) -> Self {
        Self {
            slots: std::sync::RwLock::new(HashMap::new()),
            build_sem: Arc::new(tokio::sync::Semaphore::new(1)),
            device,
            build_fn,
            retry_backoff: SLOT_RETRY_BACKOFF,
        }
    }

    /// 热重载 diff 应用：消失移除、新增插入、变更整体替换为未加载新 Slot；
    /// 未变槽原样保留（已加载池与在途请求不受影响）。
    /// （T5 即实现真逻辑——T6 只补 diff 测试，不再动这段。）
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
        Self::check_backoff(&slot)?;
        // 构建互斥：排队后复检——并发首请求只建一份
        let _build_permit = self
            .build_sem
            .acquire()
            .await
            .map_err(|_| AppError::Internal("槽构建信号量已关闭".into()))?;
        // 排队期间首个失败者可能已设置退避：复检后拒绝重复失败构建
        //（全局 1-permit 下，排队者串行失败重建会阻塞其它合法槽的首建）
        Self::check_backoff(&slot)?;
        if let Some(pool) = slot.pool() {
            return Ok((slot, pool));
        }
        let spec = slot.spec.clone();
        let build_fn = Arc::clone(&self.build_fn);
        let device = self.device;
        // panic payload 提取为消息后直接 drop（不 resume_unwind——本层要把构建 panic
        // 转成错误信封并计退避，而非把它重新抛进请求链路）
        let result = tokio::task::spawn_blocking(move || build_fn(&spec, device))
            .await
            .map_err(|e| match e.try_into_panic() {
                Ok(payload) => {
                    let msg = payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "非字符串 panic".to_string());
                    AppError::Internal(format!("槽构建 panic: {msg}"))
                }
                // 非 panic 的 JoinError（任务被取消/runtime 关闭）不冠 panic 前缀
                Err(_) => AppError::Internal("槽构建任务未执行（runtime 关闭）".into()),
            });
        match result {
            Ok(Ok(pool)) => {
                let pool = Arc::new(pool);
                // set 必须先于断言执行：debug_assert! 在 release 下不求值参数，
                // 若内联 set 则 release 永远不缓存池（行为回退为每次重建）
                let set_ok = slot.pool.set(Arc::clone(&pool)).is_ok();
                debug_assert!(set_ok, "set 不败依赖 build_sem=1 permit");
                Ok((slot, pool))
            }
            // 构建失败与构建 panic 同一退避记账：panic 不设退避会形成坏 builder 风暴
            Ok(Err(e)) | Err(e) => {
                *slot.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((e.to_string(), std::time::Instant::now() + self.retry_backoff));
                tracing::error!(
                    "动态槽 {} 构建失败（退避 {:?}）: {e}",
                    slot.spec.name, self.retry_backoff
                );
                Err(e)
            }
        }
    }

    /// 失败退避检查：窗口内拒绝并报告上次错误与剩余等待毫秒数。
    /// resolve 两处调用（acquire 前后各一次），acquire 后的复检确保已排队
    /// 请求不重复触发失败构建。
    fn check_backoff(slot: &Slot<T>) -> Result<(), AppError> {
        let failure = slot.failure.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((last, deadline)) = failure.as_ref() {
            if let Some(wait) = deadline.checked_duration_since(std::time::Instant::now()) {
                return Err(AppError::Internal(format!(
                    "动态槽 {} 此前加载失败（{last}），退避中，{}ms 后可重试",
                    slot.spec.name,
                    wait.as_millis()
                )));
            }
        }
        Ok(())
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
    config: &Config,
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
        },
        Err(e) => {
            tracing::error!("slots.toml 读取失败(保留旧配置): {e}");
            *last_fp = fp;
        }
    }
}

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
    // 初始指纹必须为 None 而非读取当前文件：main 的 load_specs 与本行之间存在
    // 窗口，文件恰在窗口内变更时，"当前指纹"已等于新版本而 registry 仍是旧版，
    // 之后每轮 fp==last_fp 恒成立、新版本永不生效。None 保证首轮必重读并
    // apply（同规格 diff 幂等，未变槽不重建），代价仅启动后一次冗余解析。
    let mut last_fp: Option<FileFingerprint> = None;
    tracing::info!(
        "slots.toml 热重载已启动(周期 {interval:?}, 路径 {})",
        config.slots_path().display()
    );
    loop {
        tokio::time::sleep(interval).await;
        reload_once(&engines.slots, &config, &mut last_fp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::engine::EnginePool;
    use crate::error::AppError;

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
    fn boundary_values_should_be_accepted() {
        let dir = tmp_models("parse-bound");
        let specs = parse_slots_toml(
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 1\nwarmup_size = [32, 32]\n", &dir,
        ).unwrap();
        assert_eq!(specs[0].pool_size, Some(1));
        assert_eq!(specs[0].warmup_size, Some((32, 32)));

        let specs = parse_slots_toml(
            "[slots.b]\nmodel = \"m.onnx\"\npool_size = 64\nwarmup_size = [4096, 4096]\n", &dir,
        ).unwrap();
        assert_eq!(specs[0].pool_size, Some(64));
        assert_eq!(specs[0].warmup_size, Some((4096, 4096)));
    }

    #[test]
    fn whitespace_only_model_should_be_rejected_as_missing() {
        let dir = tmp_models("parse-ws");
        let err = parse_slots_toml("[slots.a]\nmodel = \"   \"\n", &dir).unwrap_err();
        assert!(err.contains("缺 model"), "实际错误: {err}");
    }

    #[test]
    fn parent_dir_components_should_be_rejected() {
        let dir = tmp_models("parse-dotdot");
        for bad in ["../x.onnx", "a/../m.onnx", "m/..", ".."] {
            let toml_text = format!("[slots.a]\nmodel = \"{bad}\"\n");
            let err = parse_slots_toml(&toml_text, &dir).unwrap_err();
            assert!(err.contains(".."), "{bad} 应因含 .. 分量被拒，实际: {err}");
        }
    }

    #[test]
    fn deep_legal_path_should_not_crash() {
        // 10000 层合法分量（无 ..）：迭代实现不得栈溢出（递归版会 abort，
        // 违反热重载"失败保旧配置"契约），路径仍须落在 models_dir 内
        let dir = tmp_models("parse-deep");
        let deep = format!("{}m.onnx", "a/".repeat(10_000));
        let specs = parse_slots_toml(
            &format!("[slots.a]\nmodel = \"{deep}\"\n"), &dir,
        ).unwrap();
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join(&deep));
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

    // ---- SlotRegistry 核心（T=String 假 builder，不依赖模型文件） ----

    fn counting_builder() -> (Arc<AtomicUsize>, SlotBuildFn<String>) {
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
        assert!(matches!(err, AppError::UnknownModel(ref msg) if msg.contains("无已注册动态槽")), "实际: {err:?}");
    }

    #[tokio::test]
    async fn failed_build_should_backoff_and_retry_after_window() {
        let dir = tmp_models("registry-backoff");
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let registry = SlotRegistry::new(
            rust_onnx_infer::DeviceType::from_name("cpu"),
            Arc::new(move |_: &SlotSpec, _d| -> Result<EnginePool<String>, AppError> {
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
    async fn panicking_builder_should_report_panic_and_backoff() {
        // panic 路径与普通构建失败同一退避记账：不设退避的坏 builder 会形成重复
        // panic 风暴；payload 消息进错误信封（不丢失），不向请求链路 resume_unwind
        let dir = tmp_models("registry-panic");
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let mut registry = SlotRegistry::new(
            rust_onnx_infer::DeviceType::from_name("cpu"),
            Arc::new(move |_: &SlotSpec, _d| -> Result<EnginePool<String>, AppError> {
                c.fetch_add(1, Ordering::SeqCst);
                panic!("boom");
            }),
        );
        // 拉长退避窗口：测试在二次请求后即返回，不真等窗口过期
        registry.set_retry_backoff_for_test(Duration::from_secs(10));
        registry.apply_specs(vec![one_spec(&dir, "panic")]);

        let err = registry.resolve("panic").await.err().expect("构建 panic 应转为错误");
        assert!(
            err.to_string().contains("槽构建 panic: boom"),
            "错误应携带提取的 panic payload（防回归回 JoinError Display 拼接）: {err}"
        );
        let err = registry.resolve("panic").await.err().expect("退避窗口内应拒绝");
        assert!(err.to_string().contains("退避"), "应提示退避中: {err}");
        assert_eq!(count.load(Ordering::SeqCst), 1, "退避窗口内不得重试构建");
    }

    #[tokio::test]
    async fn concurrent_failures_should_build_once_and_backoff_the_rest() {
        // 回归：acquire 后的退避复检——8 个并发请求打到坏槽，只有首个真的
        // 构建，其余 7 个在排队等到 permit 后被退避拒绝（否则 1-permit 下
        // 排队者串行失败重建，还会阻塞其它合法槽的首建）
        let dir = tmp_models("registry-fail-storm");
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        let mut registry = SlotRegistry::new(
            rust_onnx_infer::DeviceType::from_name("cpu"),
            Arc::new(move |_: &SlotSpec, _d| -> Result<EnginePool<String>, AppError> {
                c.fetch_add(1, Ordering::SeqCst);
                // 放大竞争窗口：让其余请求充分排队后再失败设置退避
                std::thread::sleep(Duration::from_millis(50));
                Err(AppError::Internal("坏模型".into()))
            }),
        );
        // 拉长退避窗口（10s），避免与 50ms 构建延迟产生时序竞争；测试在
        // 全部句柄完成后即返回，不会真等满窗口
        registry.set_retry_backoff_for_test(Duration::from_secs(10));
        registry.apply_specs(vec![one_spec(&dir, "storm")]);
        let registry = Arc::new(registry);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let r = Arc::clone(&registry);
            handles.push(tokio::spawn(async move { r.resolve("storm").await }));
        }
        let mut backoff_errs = 0;
        for h in handles {
            let err = h.await.unwrap().err().expect("坏槽必须全部失败");
            if err.to_string().contains("退避") {
                backoff_errs += 1;
            }
        }
        assert_eq!(count.load(Ordering::SeqCst), 1, "8 个并发失败请求必须只构建一次");
        assert_eq!(backoff_errs, 7, "其余 7 个应得到退避错误，实际 {backoff_errs} 个");
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

    // ---- 热重载 diff 与 reload_once（last-known-good 语义） ----

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
            "[slots.rep]\nmodel = \"m.onnx\"\nclass_names = [\"a\"]\n", &dir,
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
        // mtime 维度：等长重写内容，size 不变、mtime 显式推进 → 指纹必须变化。
        // 显式 set_times 而非依赖两次 write 的自然间隔：Linux inode mtime 取自
        // 内核 coarse clock（jiffy 级，非纳秒精度），同刻度内两次 write 的 mtime
        // 相同，实测会 flake
        std::fs::write(&p, b"two").unwrap();
        set_mtime_ns(&p, f1.mtime_ns + 1).unwrap();
        let f2 = file_fingerprint(&p).expect("等长重写后应有指纹");
        assert_ne!(f1, f2, "同 size 不同 mtime 必须改变指纹");
        assert_eq!(f1.size, f2.size, "等长重写 size 应不变");
        std::fs::write(&p, b"one-longer").unwrap();
        let f3 = file_fingerprint(&p).expect("改写后应有指纹");
        assert_ne!(f2, f3, "size 变化必须改变指纹");
        assert!(file_fingerprint(&dir.join("nope.toml")).is_none(), "缺失文件无指纹");
    }

    /// 测试助手：显式设置 mtime（纳秒，epoch 起），规避 coarse clock 粒度。
    fn set_mtime_ns(path: &Path, mtime_ns: u128) -> std::io::Result<()> {
        use std::time::{Duration, UNIX_EPOCH};
        let f = std::fs::File::options().write(true).open(path)?;
        let t = UNIX_EPOCH + Duration::from_nanos(mtime_ns as u64);
        f.set_times(std::fs::FileTimes::new().set_modified(t))
    }
}
