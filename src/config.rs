use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

/// 能力键全集（规范名，与 `/models`、`EngineManager::list_loaded`、`configured_engines`
/// 一致）。按能力覆盖的 env（预热尺寸、池大小等）一律以能力键大写为后缀，
/// 不引入第三套命名；与模型路径 env（`FACE_DET` 等）的差异在 README 记录。
pub const CAPABILITY_KEYS: [&str; 12] = [
    "detection", "classification", "obb", "face_detection", "face_recognition", "reid",
    "ocr", "license_plate", "qr", "pose", "depth", "human_parsing",
];

/// 服务配置。所有字段均可通过环境变量覆盖（前缀 ARGUS_）。
#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub models_dir: PathBuf,
    pub device: String,
    pub auth_token: Option<String>,
    pub max_body_mb: usize,
    pub max_concurrent_inference: usize,
    /// 每个引擎的实例池大小(同模型多实例并行推理)。默认 1 = 单实例串行,行为与池化前一致。
    /// 注意:对全部已加载引擎生效——内存代价为 模型大小 × 池大小 × 已加载引擎数,
    /// 且 池大小 × 每实例 ORT 线程数(默认 4)应 ≈ 可用核数。
    pub engine_pool: usize,
    /// 优雅停机超时(秒):SIGTERM 后等待在途请求的上限,超时强制退出。
    /// 需覆盖最长推理(SAHI/depth 单次 ~1.5s,池化排队可翻倍),默认 10s;
    /// 过大(如旧值 60)在浏览器 keep-alive 不关闭时白白拉长停机窗口。
    pub shutdown_timeout_secs: u64,
    /// GPU 显存上限(MB)。0 = 不设上限(仅 CPU / 向后兼容)。
    /// CUDA arena 只增不减:A100 实测单 yolov8n 每 Session ~2.9GB,pool=4 饱和时
    /// 11.4GB;不设上限曾与 vLLM 共存膨胀至 81GB 打死服务(cuBLAS OOM 后永久失败)。
    /// 生产 GPU 必设:模型数 × 单模型峰值(yolov8n ~3GB/Session 起步)。
    pub gpu_mem_limit_mb: usize,
    /// 推理背压排队超时(毫秒)。默认 30000(排队式,内网低负载更友好);
    /// **0 = fail-fast**(try_acquire,耗尽即拒,生产 GPU 防 arena 雪崩,
    /// 与 max_concurrent=pool_size 对齐)。
    pub acquire_timeout_ms: u64,
    /// 检测模型路径（YOLO/RT-DETR/RF-DETR）。None 则 /detect 返回未配置错误。
    pub detection_model: Option<PathBuf>,
    /// 图像分类模型路径（YOLO-cls 系列）。
    pub classification_model: Option<PathBuf>,
    /// OBB 模型路径。
    pub obb_model: Option<PathBuf>,
    /// 人脸检测模型路径（YuNet）。
    pub face_det_model: Option<PathBuf>,
    /// 人脸识别模型路径（SFace/ArcFace）。
    pub face_rec_model: Option<PathBuf>,
    /// 行人重识别模型路径（OSNet x1.0,512 维 embedding）。
    pub reid_model: Option<PathBuf>,
    /// OCR 检测模型路径。
    pub ocr_det_model: Option<PathBuf>,
    /// OCR 识别模型路径。
    pub ocr_rec_model: Option<PathBuf>,
    /// OCR 字典路径。
    pub ocr_dict: Option<PathBuf>,
    /// 车牌检测模型路径。
    pub plate_det_model: Option<PathBuf>,
    /// 车牌识别模型路径。
    pub plate_rec_model: Option<PathBuf>,
    /// 二维码检测模型路径。
    pub qr_model: Option<PathBuf>,
    /// 姿态模型路径。
    pub pose_model: Option<PathBuf>,
    /// 深度估计模型路径。
    pub depth_model: Option<PathBuf>,
    /// 人体解析模型路径。
    pub human_parsing_model: Option<PathBuf>,
    /// 每能力预热 dummy 输入尺寸覆盖（`ARGUS_<CAP>_WARMUP_SIZE=WxH`）。
    /// 未设置的能力沿用引擎 `input_size`。背景：OCR 预热在引擎默认的 960×960
    /// 全零 dummy 上为 20.4s/实例（det 对大块零图产出大量伪区域 + rec 批前向），
    /// 320×320 小图实测 0.43s/实例，首个真实请求推理耗时不变（2026-09-11 实测）。
    pub warmup_sizes: HashMap<String, (u32, u32)>,
    /// 每能力实例池大小覆盖（`ARGUS_<CAP>_POOL`），未设置回退全局 `engine_pool`。
    /// 背景：内存被 ORT 每 session 运行时开销主导，ocr + human_parsing 两项占全量
    /// 83%（12.4GB 中 ~10.3GB）；把这两项与 depth 设 1 即可拿回大部分内存。
    pub pools: HashMap<String, usize>,
    /// 能力白名单（`ARGUS_ENABLED_CAPABILITIES`，逗号分隔能力键）。
    /// `None` = 未设置 = 现有行为（凡模型路径已配置即启用），零回归。
    /// 白名单外的能力在 loader 入口即被拒绝：不触碰文件系统、不创建 session。
    pub enabled_capabilities: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 7042,
            models_dir: PathBuf::from("./models"),
            device: "cpu".to_string(),
            auth_token: None,
            max_body_mb: 20,
            max_concurrent_inference: 0,
            engine_pool: 1,
            shutdown_timeout_secs: 10,
            gpu_mem_limit_mb: 0,
            acquire_timeout_ms: 30000,
            detection_model: None,
            classification_model: None,
            obb_model: None,
            face_det_model: None,
            face_rec_model: None,
            reid_model: None,
            ocr_det_model: None,
            ocr_rec_model: None,
            ocr_dict: None,
            plate_det_model: None,
            plate_rec_model: None,
            qr_model: None,
            pose_model: None,
            depth_model: None,
            human_parsing_model: None,
            warmup_sizes: HashMap::new(),
            pools: HashMap::new(),
            enabled_capabilities: None,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = env::var("ARGUS_PORT") {
            if let Ok(port) = v.parse() {
                cfg.port = port;
            }
        }
        if let Ok(v) = env::var("ARGUS_MODELS_DIR") {
            cfg.models_dir = PathBuf::from(v);
        }
        if let Ok(v) = env::var("ARGUS_DEVICE") {
            cfg.device = v.to_lowercase();
        }
        if let Ok(v) = env::var("ARGUS_AUTH_TOKEN") {
            if !v.is_empty() {
                cfg.auth_token = Some(v);
            }
        }
        if let Ok(v) = env::var("ARGUS_MAX_BODY_MB") {
            if let Ok(n) = v.parse() {
                cfg.max_body_mb = n;
            }
        }
        if let Ok(v) = env::var("ARGUS_MAX_CONCURRENT_INFERENCE") {
            if let Ok(n) = v.parse() {
                cfg.max_concurrent_inference = n;
            }
        }
        if let Ok(v) = env::var("ARGUS_ENGINE_POOL") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.engine_pool = n.max(1);
            }
        }
        if let Ok(v) = env::var("ARGUS_SHUTDOWN_TIMEOUT_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.shutdown_timeout_secs = n.max(1);
            }
        }
        if let Ok(v) = env::var("ARGUS_GPU_MEM_LIMIT_MB") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.gpu_mem_limit_mb = n;
            }
        }
        if let Ok(v) = env::var("ARGUS_ACQUIRE_TIMEOUT_MS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.acquire_timeout_ms = n;
            }
        }

        cfg.detection_model = env_path("ARGUS_DETECTION_MODEL");
        cfg.classification_model = env_path("ARGUS_CLASSIFICATION_MODEL");
        cfg.obb_model = env_path("ARGUS_OBB_MODEL");
        cfg.face_det_model = env_path("ARGUS_FACE_DET_MODEL");
        cfg.face_rec_model = env_path("ARGUS_FACE_REC_MODEL");
        cfg.reid_model = env_path("ARGUS_REID_MODEL");
        cfg.ocr_det_model = env_path("ARGUS_OCR_DET_MODEL");
        cfg.ocr_rec_model = env_path("ARGUS_OCR_REC_MODEL");
        cfg.ocr_dict = env_path("ARGUS_OCR_DICT");
        cfg.plate_det_model = env_path("ARGUS_PLATE_DET_MODEL");
        cfg.plate_rec_model = env_path("ARGUS_PLATE_REC_MODEL");
        cfg.qr_model = env_path("ARGUS_QR_MODEL");
        cfg.pose_model = env_path("ARGUS_POSE_MODEL");
        cfg.depth_model = env_path("ARGUS_DEPTH_MODEL");
        cfg.human_parsing_model = env_path("ARGUS_HUMAN_PARSING_MODEL");

        for cap in CAPABILITY_KEYS {
            if let Some(size) = env::var(format!("ARGUS_{}_WARMUP_SIZE", cap.to_uppercase()))
                .ok()
                .and_then(|v| parse_wxh(&v))
            {
                cfg.warmup_sizes.insert(cap.to_string(), size);
            }
            // 池 0 会构造空实例池（EnginePool 构造 panic），非数字解析失败——
            // 两者都视为未设置，回退全局
            if let Some(n) = env::var(format!("ARGUS_{}_POOL", cap.to_uppercase()))
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n >= 1)
            {
                cfg.pools.insert(cap.to_string(), n);
            }
        }

        let whitelist: Option<Vec<String>> = env::var("ARGUS_ENABLED_CAPABILITIES")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|k| !k.is_empty() && CAPABILITY_KEYS.contains(k))
                    .map(str::to_string)
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty());
        cfg.enabled_capabilities = whitelist;

        cfg
    }

    /// 能力是否在白名单内；未设置白名单 = 全部允许（现有行为）。
    pub fn capability_enabled(&self, capability: &str) -> bool {
        match &self.enabled_capabilities {
            None => true,
            Some(list) => list.iter().any(|k| k == capability),
        }
    }

    /// 预热 dummy 输入尺寸覆盖；`None` = 沿用引擎 `input_size`（默认行为）。
    pub fn warmup_size_for(&self, capability: &str) -> Option<(u32, u32)> {
        self.warmup_sizes.get(capability).copied()
    }

    /// 能力级实例池大小；未设置（或非法）回退全局 `engine_pool`。
    pub fn pool_for(&self, capability: &str) -> usize {
        self.pools.get(capability).copied().unwrap_or(self.engine_pool)
    }

    /// 列出已配置模型路径的引擎能力名（deep health 与启动日志用）。
    /// 命名与 `EngineManager::list_loaded` 一致，便于对照"配置了未加载"状态。
    pub fn configured_engines(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.detection_model.is_some() { v.push("detection"); }
        if self.classification_model.is_some() { v.push("classification"); }
        if self.obb_model.is_some() { v.push("obb"); }
        if self.face_det_model.is_some() { v.push("face_detection"); }
        if self.face_rec_model.is_some() { v.push("face_recognition"); }
        if self.reid_model.is_some() { v.push("reid"); }
        if self.ocr_det_model.is_some() { v.push("ocr"); }
        if self.plate_det_model.is_some() { v.push("license_plate"); }
        if self.qr_model.is_some() { v.push("qr"); }
        if self.pose_model.is_some() { v.push("pose"); }
        if self.depth_model.is_some() { v.push("depth"); }
        if self.human_parsing_model.is_some() { v.push("human_parsing"); }
        v
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var(key).ok().filter(|s| !s.is_empty()).map(PathBuf::from)
}

/// 解析 `WxH` 形式的尺寸（如 `320x320`）；非法值返回 None（回退引擎默认）。
fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// env 是进程全局状态：并行测试写同一变量会互相覆盖（曾致 auth_token 断言
    /// 间歇失败），所有写 env 的测试必须先持有此锁串行执行。
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn default_config_should_have_sensible_values() {
        let config = Config::default();
        assert_eq!(config.port, 7042);
        assert_eq!(config.models_dir, PathBuf::from("./models"));
        assert_eq!(config.device, "cpu");
        assert!(config.auth_token.is_none());
        assert_eq!(config.max_body_mb, 20);
        assert_eq!(config.max_concurrent_inference, 0);
        assert_eq!(config.engine_pool, 1);
        assert_eq!(config.shutdown_timeout_secs, 10);
        assert!(config.detection_model.is_none());
        assert!(config.face_det_model.is_none());
        assert!(config.ocr_det_model.is_none());
    }

    #[test]
    fn from_env_should_override_shutdown_timeout() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_SHUTDOWN_TIMEOUT_SECS", "30");
        }
        let config = Config::from_env();
        assert_eq!(config.shutdown_timeout_secs, 30);
        unsafe {
            env::remove_var("ARGUS_SHUTDOWN_TIMEOUT_SECS");
        }
    }

    #[test]
    fn default_config_should_have_zero_gpu_mem_limit() {
        let config = Config::default();
        assert_eq!(config.gpu_mem_limit_mb, 0, "默认不设显存上限(CPU/向后兼容)");
    }

    #[test]
    fn default_config_should_have_30s_acquire_timeout() {
        let config = Config::default();
        assert_eq!(config.acquire_timeout_ms, 30000, "默认排队式 30s");
    }

    #[test]
    fn from_env_should_override_gpu_mem_limit() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_GPU_MEM_LIMIT_MB", "4096");
        }
        let config = Config::from_env();
        assert_eq!(config.gpu_mem_limit_mb, 4096);
        unsafe {
            env::remove_var("ARGUS_GPU_MEM_LIMIT_MB");
        }
    }

    #[test]
    fn from_env_should_override_acquire_timeout() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ACQUIRE_TIMEOUT_MS", "5000");
        }
        let config = Config::from_env();
        assert_eq!(config.acquire_timeout_ms, 5000);
        unsafe {
            env::remove_var("ARGUS_ACQUIRE_TIMEOUT_MS");
        }
    }

    #[test]
    fn from_env_should_allow_zero_acquire_timeout_for_fail_fast() {
        // 0 = try_acquire fail-fast(生产 GPU 防 arena 雪崩)
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ACQUIRE_TIMEOUT_MS", "0");
        }
        let config = Config::from_env();
        assert_eq!(config.acquire_timeout_ms, 0, "0 必须原样保留(fail-fast 语义)");
        unsafe {
            env::remove_var("ARGUS_ACQUIRE_TIMEOUT_MS");
        }
    }

    #[test]
    fn from_env_should_override_defaults() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_PORT", "8080");
            env::set_var("ARGUS_DEVICE", "CUDA");
            env::set_var("ARGUS_AUTH_TOKEN", "secret-key-123");
            env::set_var("ARGUS_DETECTION_MODEL", "/models/yolo.onnx");
        }
        let config = Config::from_env();
        assert_eq!(config.port, 8080);
        assert_eq!(config.device, "cuda");
        assert_eq!(config.auth_token.as_deref(), Some("secret-key-123"));
        assert_eq!(
            config.detection_model.as_deref(),
            Some(std::path::Path::new("/models/yolo.onnx"))
        );
        unsafe {
            env::remove_var("ARGUS_PORT");
            env::remove_var("ARGUS_DEVICE");
            env::remove_var("ARGUS_AUTH_TOKEN");
            env::remove_var("ARGUS_DETECTION_MODEL");
        }
    }

    #[test]
    fn from_env_should_parse_reid_model() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_REID_MODEL", "/models/osnet_x1_0.onnx");
        }
        let config = Config::from_env();
        assert_eq!(
            config.reid_model.as_deref(),
            Some(std::path::Path::new("/models/osnet_x1_0.onnx"))
        );
        assert!(config.configured_engines().contains(&"reid"));
        unsafe {
            env::remove_var("ARGUS_REID_MODEL");
        }
    }

    #[test]
    fn from_env_should_parse_classification_model() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_CLASSIFICATION_MODEL", "/models/yolov8n-cls.onnx");
        }
        let config = Config::from_env();
        assert_eq!(
            config.classification_model.as_deref(),
            Some(std::path::Path::new("/models/yolov8n-cls.onnx"))
        );
        assert!(config.configured_engines().contains(&"classification"));
        unsafe {
            env::remove_var("ARGUS_CLASSIFICATION_MODEL");
        }
    }

    #[test]
    fn from_env_should_ignore_empty_auth_token() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_AUTH_TOKEN", "");
        }
        let config = Config::from_env();
        assert!(config.auth_token.is_none());
        unsafe {
            env::remove_var("ARGUS_AUTH_TOKEN");
        }
    }

    // ---- 预热输入尺寸（§3.1：按能力可覆盖，dummy 尺寸是 OCR 预热 80s 的主因） ----

    #[test]
    fn from_env_should_parse_ocr_warmup_size() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_OCR_WARMUP_SIZE", "320x320");
        }
        let config = Config::from_env();
        assert_eq!(
            config.warmup_size_for("ocr"),
            Some((320, 320)),
            "OCR 预热应用小图 dummy，避免 960 全零图触发大量伪区域"
        );
        unsafe {
            env::remove_var("ARGUS_OCR_WARMUP_SIZE");
        }
    }

    #[test]
    fn warmup_size_should_default_to_engine_input_size_when_unset() {
        // 断言依赖"该 env 未设置",必须持锁,否则与并行 set env 的测试竞争(实测 flaky)
        let _guard = env_lock();
        let config = Config::from_env();
        assert_eq!(
            config.warmup_size_for("ocr"),
            None,
            "未设置时预热沿用引擎 input_size，行为与改动前一致"
        );
    }

    #[test]
    fn warmup_size_should_be_per_capability() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_OCR_WARMUP_SIZE", "320x320");
        }
        let config = Config::from_env();
        assert_eq!(
            config.warmup_size_for("depth"),
            None,
            "OCR 的覆盖不得泄漏到其他能力"
        );
        unsafe {
            env::remove_var("ARGUS_OCR_WARMUP_SIZE");
        }
    }

    #[test]
    fn from_env_should_ignore_malformed_warmup_size() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_OCR_WARMUP_SIZE", "abc");
        }
        let config = Config::from_env();
        assert_eq!(config.warmup_size_for("ocr"), None, "非法值应回退默认");
        unsafe {
            env::remove_var("ARGUS_OCR_WARMUP_SIZE");
        }
    }

    // ---- 单能力独立设池（§3.2：ARGUS_<CAP>_POOL 覆盖全局 ENGINE_POOL） ----

    #[test]
    fn pool_for_should_fall_back_to_global_when_per_cap_unset() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENGINE_POOL", "4");
        }
        let config = Config::from_env();
        assert_eq!(
            config.pool_for("ocr"),
            4,
            "未设置能力级池时必须回退全局 engine_pool（零回归默认）"
        );
        unsafe {
            env::remove_var("ARGUS_ENGINE_POOL");
        }
    }

    #[test]
    fn per_cap_pool_should_override_global_pool() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENGINE_POOL", "4");
            env::set_var("ARGUS_OCR_POOL", "1");
        }
        let config = Config::from_env();
        assert_eq!(
            config.pool_for("ocr"), 1,
            "OCR/HUMAN_PARSING/DEPTH 池=1 是内存止血的关键杠杆（12.4GB→约 4GB）"
        );
        unsafe {
            env::remove_var("ARGUS_ENGINE_POOL");
            env::remove_var("ARGUS_OCR_POOL");
        }
    }

    #[test]
    fn per_cap_pool_should_not_leak_to_other_capabilities() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENGINE_POOL", "4");
            env::set_var("ARGUS_OCR_POOL", "1");
        }
        let config = Config::from_env();
        assert_eq!(config.pool_for("depth"), 4, "OCR 的覆盖不得影响 depth");
        unsafe {
            env::remove_var("ARGUS_ENGINE_POOL");
            env::remove_var("ARGUS_OCR_POOL");
        }
    }

    #[test]
    fn per_cap_pool_of_zero_or_garbage_should_fall_back_to_global() {
        // 池 0 会构造空实例池（EnginePool 构造 panic），必须挡在解析层回退全局
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENGINE_POOL", "2");
            env::set_var("ARGUS_DEPTH_POOL", "0");
            env::set_var("ARGUS_HUMAN_PARSING_POOL", "x");
        }
        let config = Config::from_env();
        assert_eq!(config.pool_for("depth"), 2, "0 非法，回退全局");
        assert_eq!(config.pool_for("human_parsing"), 2, "非数字非法，回退全局");
        unsafe {
            env::remove_var("ARGUS_ENGINE_POOL");
            env::remove_var("ARGUS_DEPTH_POOL");
            env::remove_var("ARGUS_HUMAN_PARSING_POOL");
        }
    }

    // ---- 能力白名单（§3.3：ARGUS_ENABLED_CAPABILITIES） ----

    #[test]
    fn capability_enabled_should_default_to_all_when_unset() {
        // 断言依赖"白名单 env 未设置"，必须持锁防并行污染
        let _guard = env_lock();
        let config = Config::from_env();
        assert!(
            config.capability_enabled("detection") && config.capability_enabled("ocr"),
            "未设置白名单 = 现有行为（凡模型路径已配置即启用），零回归默认"
        );
    }

    #[test]
    fn from_env_should_parse_enabled_capabilities() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENABLED_CAPABILITIES", "detection,ocr");
        }
        let config = Config::from_env();
        assert!(config.capability_enabled("detection"));
        assert!(config.capability_enabled("ocr"));
        assert!(
            !config.capability_enabled("depth"),
            "白名单外能力必须被拒绝加载"
        );
        unsafe {
            env::remove_var("ARGUS_ENABLED_CAPABILITIES");
        }
    }

    #[test]
    fn enabled_capabilities_should_trim_spaces_and_drop_unknown_keys() {
        let _guard = env_lock();
        unsafe {
            env::set_var("ARGUS_ENABLED_CAPABILITIES", " ocr , bogus_cap ");
        }
        let config = Config::from_env();
        assert!(
            config.capability_enabled("ocr"),
            "空格应被容忍（手写 env 常见）"
        );
        assert!(
            !config.capability_enabled("bogus_cap"),
            "未知键不是合法能力名，不得意外生效"
        );
        unsafe {
            env::remove_var("ARGUS_ENABLED_CAPABILITIES");
        }
    }
}
