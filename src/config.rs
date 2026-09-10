use std::env;
use std::path::PathBuf;

/// 服务配置。所有字段均可通过环境变量覆盖（前缀 VISION_EXT_）。
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
    /// 检测模型路径（YOLO/RT-DETR/RF-DETR）。None 则 /detect 返回未配置错误。
    pub detection_model: Option<PathBuf>,
    /// OBB 模型路径。
    pub obb_model: Option<PathBuf>,
    /// 人脸检测模型路径（YuNet）。
    pub face_det_model: Option<PathBuf>,
    /// 人脸识别模型路径（SFace/ArcFace）。
    pub face_rec_model: Option<PathBuf>,
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
            detection_model: None,
            obb_model: None,
            shutdown_timeout_secs: 10,
            face_det_model: None,
            face_rec_model: None,
            ocr_det_model: None,
            ocr_rec_model: None,
            ocr_dict: None,
            plate_det_model: None,
            plate_rec_model: None,
            qr_model: None,
            pose_model: None,
            depth_model: None,
            human_parsing_model: None,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = env::var("VISION_EXT_PORT") {
            if let Ok(port) = v.parse() {
                cfg.port = port;
            }
        }
        if let Ok(v) = env::var("VISION_EXT_MODELS_DIR") {
            cfg.models_dir = PathBuf::from(v);
        }
        if let Ok(v) = env::var("VISION_EXT_DEVICE") {
            cfg.device = v.to_lowercase();
        }
        if let Ok(v) = env::var("VISION_EXT_AUTH_TOKEN") {
            if !v.is_empty() {
                cfg.auth_token = Some(v);
            }
        }
        if let Ok(v) = env::var("VISION_EXT_MAX_BODY_MB") {
            if let Ok(n) = v.parse() {
                cfg.max_body_mb = n;
            }
        }
        if let Ok(v) = env::var("VISION_EXT_MAX_CONCURRENT_INFERENCE") {
            if let Ok(n) = v.parse() {
                cfg.max_concurrent_inference = n;
            }
        }
        if let Ok(v) = env::var("VISION_EXT_ENGINE_POOL") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.engine_pool = n.max(1);
            }
        }

        cfg.detection_model = env_path("VISION_EXT_DETECTION_MODEL");
        cfg.obb_model = env_path("VISION_EXT_OBB_MODEL");
        if let Ok(v) = env::var("VISION_EXT_SHUTDOWN_TIMEOUT_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.shutdown_timeout_secs = n.max(1);
            }
        }
        cfg.face_det_model = env_path("VISION_EXT_FACE_DET_MODEL");
        cfg.face_rec_model = env_path("VISION_EXT_FACE_REC_MODEL");
        cfg.ocr_det_model = env_path("VISION_EXT_OCR_DET_MODEL");
        cfg.ocr_rec_model = env_path("VISION_EXT_OCR_REC_MODEL");
        cfg.ocr_dict = env_path("VISION_EXT_OCR_DICT");
        cfg.plate_det_model = env_path("VISION_EXT_PLATE_DET_MODEL");
        cfg.plate_rec_model = env_path("VISION_EXT_PLATE_REC_MODEL");
        cfg.qr_model = env_path("VISION_EXT_QR_MODEL");
        cfg.pose_model = env_path("VISION_EXT_POSE_MODEL");
        cfg.depth_model = env_path("VISION_EXT_DEPTH_MODEL");
        cfg.human_parsing_model = env_path("VISION_EXT_HUMAN_PARSING_MODEL");

        cfg
    }

    /// 列出已配置模型路径的引擎能力名（deep health 与启动日志用）。
    /// 命名与 `EngineManager::list_loaded` 一致，便于对照"配置了未加载"状态。
    pub fn configured_engines(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.detection_model.is_some() { v.push("detection"); }
        if self.obb_model.is_some() { v.push("obb"); }
        if self.face_det_model.is_some() { v.push("face_detection"); }
        if self.face_rec_model.is_some() { v.push("face_recognition"); }
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
        assert!(config.detection_model.is_none());
        assert!(config.face_det_model.is_none());
        assert!(config.ocr_det_model.is_none());
    }

        assert_eq!(config.engine_pool, 1);
        assert_eq!(config.shutdown_timeout_secs, 10);
    #[test]
    fn from_env_should_override_defaults() {
        let _guard = env_lock();
        unsafe {
            env::set_var("VISION_EXT_PORT", "8080");
    #[test]
    fn from_env_should_override_shutdown_timeout() {
        let _guard = env_lock();
        unsafe {
            env::set_var("VISION_EXT_SHUTDOWN_TIMEOUT_SECS", "30");
        }
        let config = Config::from_env();
        assert_eq!(config.shutdown_timeout_secs, 30);
        unsafe {
            env::remove_var("VISION_EXT_SHUTDOWN_TIMEOUT_SECS");
        }
    }

            env::set_var("VISION_EXT_DEVICE", "CUDA");
            env::set_var("VISION_EXT_AUTH_TOKEN", "secret-key-123");
            env::set_var("VISION_EXT_DETECTION_MODEL", "/models/yolo.onnx");
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
            env::remove_var("VISION_EXT_PORT");
            env::remove_var("VISION_EXT_DEVICE");
            env::remove_var("VISION_EXT_AUTH_TOKEN");
            env::remove_var("VISION_EXT_DETECTION_MODEL");
        }
    }

    #[test]
    fn from_env_should_ignore_empty_auth_token() {
        let _guard = env_lock();
        unsafe {
            env::set_var("VISION_EXT_AUTH_TOKEN", "");
        }
        let config = Config::from_env();
        assert!(config.auth_token.is_none());
        unsafe {
            env::remove_var("VISION_EXT_AUTH_TOKEN");
        }
    }
}
