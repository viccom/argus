use std::sync::Arc;

use rust_onnx_infer::{
    create_depth_engine, create_face_detection_engine, create_face_recognition_engine,
    create_ocr_pipeline, create_pose_engine, DeviceType,
};
use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::engines::face_detection::FaceDetectResult;
use rust_onnx_infer::engines::face_recognition::FaceRecognitionEngine;
use rust_onnx_infer::engines::human_parsing::HumanParsingEngine;
use rust_onnx_infer::engines::image_quality::ImageQualityEngine;
use rust_onnx_infer::engines::license_plate::LicensePlateEngine;
use rust_onnx_infer::engines::obb_detection::ObbDetectionEngine;
use rust_onnx_infer::engines::ocr::OcrPipeline;
use rust_onnx_infer::engines::qr_detector::QrDetector;
use rust_onnx_infer::imaging::FloatMask;
use rust_onnx_infer::model::PoseResult;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::config::Config;
use crate::error::AppError;

/// trait object 类型别名（缩短签名）。
type DynFaceDet = Box<dyn OnnxInferenceEngine<Output = Vec<FaceDetectResult>>>;
type DynPose = Box<dyn OnnxInferenceEngine<Output = Vec<PoseResult>>>;
type DynDepth = Box<dyn OnnxInferenceEngine<Output = FloatMask>>;

/// 引擎管理器：按能力类型懒加载 rust-onnx-infer 引擎，Arc 共享。
///
/// rust-onnx-infer 的 `Session` 包在 `Mutex` 里（单实例串行推理），
/// 因此每种引擎只创建一个实例，并发请求排队等待。
///
/// detection 存储为具体类型 `DetectionEngine`（非 trait object），
/// 因为 SAHI 切片推理需要 `&dyn SahiSinglePredictor`——该 trait 仅具体类型实现。
#[allow(dead_code)]
pub struct EngineManager {
    config: Config,
    device: DeviceType,
    detection: tokio::sync::OnceCell<Arc<DetectionEngine>>,
    obb: tokio::sync::OnceCell<Arc<ObbDetectionEngine>>,
    face_det: tokio::sync::OnceCell<Arc<DynFaceDet>>,
    face_rec: tokio::sync::OnceCell<Arc<FaceRecognitionEngine>>,
    ocr: tokio::sync::OnceCell<Arc<OcrPipeline>>,
    plate: tokio::sync::OnceCell<Arc<LicensePlateEngine>>,
    qr: tokio::sync::OnceCell<Arc<QrDetector>>,
    quality: tokio::sync::OnceCell<Arc<ImageQualityEngine>>,
    pose: tokio::sync::OnceCell<Arc<DynPose>>,
    depth: tokio::sync::OnceCell<Arc<DynDepth>>,
    human_parsing: tokio::sync::OnceCell<Arc<HumanParsingEngine>>,
}

#[allow(dead_code)]
impl EngineManager {
    pub fn new(config: Config) -> Self {
        let device = DeviceType::from_name(&config.device);
        EngineManager {
            config,
            device,
            detection: tokio::sync::OnceCell::new(),
            obb: tokio::sync::OnceCell::new(),
            face_det: tokio::sync::OnceCell::new(),
            face_rec: tokio::sync::OnceCell::new(),
            ocr: tokio::sync::OnceCell::new(),
            plate: tokio::sync::OnceCell::new(),
            qr: tokio::sync::OnceCell::new(),
            quality: tokio::sync::OnceCell::new(),
            pose: tokio::sync::OnceCell::new(),
            depth: tokio::sync::OnceCell::new(),
            human_parsing: tokio::sync::OnceCell::new(),
        }
    }

    pub fn device_type(&self) -> DeviceType {
        self.device
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// 列出已加载的引擎名。
    pub fn list_loaded(&self) -> Vec<&'static str> {
        let mut loaded = Vec::new();
        if self.detection.initialized() { loaded.push("detection"); }
        if self.obb.initialized() { loaded.push("obb"); }
        if self.face_det.initialized() { loaded.push("face_detection"); }
        if self.face_rec.initialized() { loaded.push("face_recognition"); }
        if self.ocr.initialized() { loaded.push("ocr"); }
        if self.plate.initialized() { loaded.push("license_plate"); }
        if self.qr.initialized() { loaded.push("qr"); }
        if self.quality.initialized() { loaded.push("quality"); }
        if self.pose.initialized() { loaded.push("pose"); }
        if self.depth.initialized() { loaded.push("depth"); }
        if self.human_parsing.initialized() { loaded.push("human_parsing"); }
        loaded
    }

    // ---- 各引擎懒加载入口 ----

    pub async fn detection(&self) -> Result<Arc<DetectionEngine>, AppError> {
        self.detection
            .get_or_try_init(|| async {
                let path = self.config.detection_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("detection".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    DetectionEngine::for_yolo(&path, device, 0.25, 0.45)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn obb(&self) -> Result<Arc<ObbDetectionEngine>, AppError> {
        self.obb
            .get_or_try_init(|| async {
                let path = self.config.obb_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("obb".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || ObbDetectionEngine::new(&path, device))
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn face_detection(&self) -> Result<Arc<DynFaceDet>, AppError> {
        self.face_det
            .get_or_try_init(|| async {
                let path = self.config.face_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_detection".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    create_face_detection_engine(&path, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn face_recognition(&self) -> Result<Arc<FaceRecognitionEngine>, AppError> {
        self.face_rec
            .get_or_try_init(|| async {
                let path = self.config.face_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_recognition".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    create_face_recognition_engine(&path, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn ocr(&self) -> Result<Arc<OcrPipeline>, AppError> {
        self.ocr
            .get_or_try_init(|| async {
                let det = self.config.ocr_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let rec = self.config.ocr_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let dict = self.config.ocr_dict.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    create_ocr_pipeline(&det, &rec, &dict, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn license_plate(&self) -> Result<Arc<LicensePlateEngine>, AppError> {
        self.plate
            .get_or_try_init(|| async {
                let det = self.config.plate_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let rec = self.config.plate_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    LicensePlateEngine::new(&det, &rec, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn qr(&self) -> Result<Arc<QrDetector>, AppError> {
        self.qr
            .get_or_try_init(|| async {
                let path = self.config.qr_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("qr".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || QrDetector::new(&path, device))
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn quality(&self) -> Result<Arc<ImageQualityEngine>, AppError> {
        self.quality
            .get_or_try_init(|| async {
                let path = self.config.quality_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("quality".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    ImageQualityEngine::new(&path, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn pose(&self) -> Result<Arc<DynPose>, AppError> {
        self.pose
            .get_or_try_init(|| async {
                let path = self.config.pose_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("pose".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    create_pose_engine(&path, device, 0.25)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn depth(&self) -> Result<Arc<DynDepth>, AppError> {
        self.depth
            .get_or_try_init(|| async {
                let path = self.config.depth_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("depth".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || create_depth_engine(&path, device))
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }

    pub async fn human_parsing(&self) -> Result<Arc<HumanParsingEngine>, AppError> {
        self.human_parsing
            .get_or_try_init(|| async {
                let path = self.config.human_parsing_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("human_parsing".into()))?;
                let device = self.device;
                let engine = tokio::task::spawn_blocking(move || {
                    HumanParsingEngine::new(&path, device)
                })
                .await
                .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                Ok(Arc::new(engine))
            })
            .await
            .cloned()
    }
}

/// 在 spawn_blocking 中执行阻塞推理。
pub async fn run_blocking<F, T>(f: F) -> Result<T, AppError>
where
    F: FnOnce() -> rust_onnx_infer::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::Internal(format!("推理任务失败: {e}")))?
        .map_err(AppError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn device_type_from_config_cpu() {
        let config = Config {
            device: "cpu".into(),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        assert_eq!(mgr.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn device_type_from_config_cuda() {
        let config = Config {
            device: "cuda".into(),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        assert_eq!(mgr.device_type(), DeviceType::Cuda);
    }

    #[test]
    fn device_type_from_config_unknown_falls_back_to_cpu() {
        let config = Config {
            device: "tpu".into(),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        assert_eq!(mgr.device_type(), DeviceType::Cpu);
    }

    #[tokio::test]
    async fn get_detection_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.detection().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn get_face_det_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.face_detection().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn get_ocr_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.ocr().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn list_loaded_should_be_empty_initially() {
        let mgr = EngineManager::new(Config::default());
        assert!(mgr.list_loaded().is_empty());
    }
}
