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
    /// 图像质量模型路径（FIQA）。None 时仅用传统算法。
    pub quality_model: Option<PathBuf>,
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
            detection_model: None,
            obb_model: None,
            face_det_model: None,
            face_rec_model: None,
            ocr_det_model: None,
            ocr_rec_model: None,
            ocr_dict: None,
            plate_det_model: None,
            plate_rec_model: None,
            qr_model: None,
            quality_model: None,
            pose_model: None,
            depth_model: None,
            human_parsing_model: None,
        }
    }
}

#[allow(dead_code)]
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

        cfg.detection_model = env_path("VISION_EXT_DETECTION_MODEL");
        cfg.obb_model = env_path("VISION_EXT_OBB_MODEL");
        cfg.face_det_model = env_path("VISION_EXT_FACE_DET_MODEL");
        cfg.face_rec_model = env_path("VISION_EXT_FACE_REC_MODEL");
        cfg.ocr_det_model = env_path("VISION_EXT_OCR_DET_MODEL");
        cfg.ocr_rec_model = env_path("VISION_EXT_OCR_REC_MODEL");
        cfg.ocr_dict = env_path("VISION_EXT_OCR_DICT");
        cfg.plate_det_model = env_path("VISION_EXT_PLATE_DET_MODEL");
        cfg.plate_rec_model = env_path("VISION_EXT_PLATE_REC_MODEL");
        cfg.qr_model = env_path("VISION_EXT_QR_MODEL");
        cfg.quality_model = env_path("VISION_EXT_QUALITY_MODEL");
        cfg.pose_model = env_path("VISION_EXT_POSE_MODEL");
        cfg.depth_model = env_path("VISION_EXT_DEPTH_MODEL");
        cfg.human_parsing_model = env_path("VISION_EXT_HUMAN_PARSING_MODEL");

        cfg
    }

    /// 从 models_dir 解析模型文件名（加 .onnx 后缀）。
    pub fn model_path(&self, name: &str) -> PathBuf {
        self.models_dir.join(format!("{name}.onnx"))
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var(key).ok().filter(|s| !s.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn model_path_should_join_models_dir_with_onnx_extension() {
        let config = Config {
            models_dir: PathBuf::from("/data/models"),
            ..Default::default()
        };
        assert_eq!(
            config.model_path("yolov8n"),
            PathBuf::from("/data/models/yolov8n.onnx")
        );
    }

    #[test]
    fn from_env_should_override_defaults() {
        unsafe {
            env::set_var("VISION_EXT_PORT", "8080");
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
