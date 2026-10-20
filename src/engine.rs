use std::sync::Arc;

use rust_onnx_infer::{
    create_depth_engine, create_face_detection_engine, create_face_recognition_engine,
    create_ocr_pipeline, create_pose_engine, DeviceType,
};
use rust_onnx_infer::engines::classification::ClassificationEngine;
use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::engines::face_detection::FaceDetectResult;
use rust_onnx_infer::engines::face_recognition::FaceRecognitionEngine;
use rust_onnx_infer::engines::human_parsing::HumanParsingEngine;
use rust_onnx_infer::engines::license_plate::LicensePlateEngine;
use rust_onnx_infer::engines::obb_detection::ObbDetectionEngine;
use rust_onnx_infer::engines::ocr::OcrPipeline;
use rust_onnx_infer::engines::qr_detector::QrDetector;
use rust_onnx_infer::engines::reid::ReidEngine;
use rust_onnx_infer::engines::segmentation::SegmentationEngine;
use rust_onnx_infer::imaging::{FloatMask, Rect};
use rust_onnx_infer::model::PoseResult;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::config::Config;
use crate::error::AppError;

/// 检测引擎构造阈值(下限)。引擎是 Arc 共享不可变,每请求 `confidence` 在 HTTP 层
/// 后过滤(类内 NMS 保序:低阈值结果是任意高阈值结果的超集,后过滤严格等价)。
///
/// **必须 ≥ `SahiConfig::LOW_MODEL_CONFIDENCE`(0.1)**,否则 SAHI postprocess 自动切换
/// NMS/IOU 改变合并语义(有测试锁定)。
pub const DETECTION_CONFIDENCE_FLOOR: f32 = 0.10;
/// 检测 NMS IoU 阈值(全局,不做每请求覆盖——NMS 在引擎内完成,重做等于绕过引擎后处理)。
pub const DETECTION_NMS_IOU: f32 = 0.45;
/// 分割引擎构造阈值(Ultralytics 默认 conf 0.25)。
pub const SEGMENTATION_CONFIDENCE: f32 = 0.25;
/// 分割 NMS IoU 阈值(Ultralytics 默认 0.7;过小会得到比官方更多的重叠检出)。
pub const SEGMENTATION_NMS_IOU: f32 = 0.7;

// 编译期锁定:下限低于 LOW_MODEL_CONFIDENCE 会让 SAHI 路径行为静默漂移
const _: () = assert!(
    DETECTION_CONFIDENCE_FLOOR >= rust_onnx_infer::sahi::config::SahiConfig::LOW_MODEL_CONFIDENCE,
    "DETECTION_CONFIDENCE_FLOOR 必须 >= SahiConfig::LOW_MODEL_CONFIDENCE"
);

/// 深检结果:单个引擎的健康状态。
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineHealth {
    pub name: &'static str,
    /// "ok" = dummy 推理成功;"degraded" = 失败(arena OOM 后 cuBLAS 永久失败等)
    pub status: &'static str,
    pub error: Option<String>,
}

/// 深检宏:对已加载引擎的池实例 0 发 dummy 推理,结果推入 results。
/// 未加载引擎跳过(OnceCell::get 返回 None)。
macro_rules! check_engine {
    ($cell:expr, $name:literal, $ty:ty, $results:expr) => {
        if let Some(pool) = $cell.get() {
            let pool: &Arc<EnginePool<$ty>> = pool;
            let (w, h) = pool.instances.first()
                .map(|e| OnnxInferPredict::input_size(e.as_ref()))
                .unwrap_or((640, 640));
            let img = dummy_image(w, h);
            let pool = Arc::clone(pool);
            let status = tokio::task::spawn_blocking(move || {
                pool.instances[0].predict(&img).map(|_| ())
            })
            .await
            .map_err(|e| rust_onnx_infer::VisionError::inference(format!("深检任务失败: {e}")))
            .and_then(|r| r);
            $results.push(EngineHealth {
                name: $name,
                status: if status.is_ok() { "ok" } else { "degraded" },
                error: status.err().map(|e| e.to_string()),
            });
        }
    };
}

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
pub struct EngineManager {
    config: Config,
    device: DeviceType,
    detection: tokio::sync::OnceCell<Arc<EnginePool<DetectionEngine>>>,
    ppe: tokio::sync::OnceCell<Arc<EnginePool<DetectionEngine>>>,
    fire_smoke: tokio::sync::OnceCell<Arc<EnginePool<DetectionEngine>>>,
    classification: tokio::sync::OnceCell<Arc<EnginePool<ClassificationEngine>>>,
    obb: tokio::sync::OnceCell<Arc<EnginePool<ObbDetectionEngine>>>,
    segmentation: tokio::sync::OnceCell<Arc<EnginePool<SegmentationEngine>>>,
    face_det: tokio::sync::OnceCell<Arc<EnginePool<DynFaceDet>>>,
    face_rec: tokio::sync::OnceCell<Arc<EnginePool<FaceRecognitionEngine>>>,
    reid: tokio::sync::OnceCell<Arc<EnginePool<ReidEngine>>>,
    ocr: tokio::sync::OnceCell<Arc<EnginePool<OcrPipeline>>>,
    plate: tokio::sync::OnceCell<Arc<EnginePool<LicensePlateEngine>>>,
    qr: tokio::sync::OnceCell<Arc<EnginePool<QrDetector>>>,
    pose: tokio::sync::OnceCell<Arc<EnginePool<DynPose>>>,
    depth: tokio::sync::OnceCell<Arc<EnginePool<DynDepth>>>,
    human_parsing: tokio::sync::OnceCell<Arc<EnginePool<HumanParsingEngine>>>,
}

impl EngineManager {
    pub fn new(config: Config) -> Self {
        let device = DeviceType::from_name(&config.device);
        EngineManager {
            config,
            device,
            detection: tokio::sync::OnceCell::new(),
            ppe: tokio::sync::OnceCell::new(),
            fire_smoke: tokio::sync::OnceCell::new(),
            classification: tokio::sync::OnceCell::new(),
            obb: tokio::sync::OnceCell::new(),
            segmentation: tokio::sync::OnceCell::new(),
            face_det: tokio::sync::OnceCell::new(),
            face_rec: tokio::sync::OnceCell::new(),
            reid: tokio::sync::OnceCell::new(),
            ocr: tokio::sync::OnceCell::new(),
            plate: tokio::sync::OnceCell::new(),
            qr: tokio::sync::OnceCell::new(),
            pose: tokio::sync::OnceCell::new(),
            depth: tokio::sync::OnceCell::new(),
            human_parsing: tokio::sync::OnceCell::new(),
        }
    }

    /// 白名单强制点(§3.3):不在 `ARGUS_ENABLED_CAPABILITIES` 白名单的能力,
    /// 在读取模型路径/触碰文件系统之前即拒绝,与"模型未配置"区分。
    fn ensure_enabled(&self, capability: &str) -> Result<(), AppError> {
        if self.config.capability_enabled(capability) {
            Ok(())
        } else {
            Err(AppError::CapabilityDisabled(capability.into()))
        }
    }

    /// 列出已加载的引擎名。
    pub fn list_loaded(&self) -> Vec<&'static str> {
        let mut loaded = Vec::new();
        if self.detection.initialized() { loaded.push("detection"); }
        if self.ppe.initialized() { loaded.push("ppe"); }
        if self.fire_smoke.initialized() { loaded.push("fire_smoke"); }
        if self.classification.initialized() { loaded.push("classification"); }
        if self.obb.initialized() { loaded.push("obb"); }
        if self.segmentation.initialized() { loaded.push("segmentation"); }
        if self.face_det.initialized() { loaded.push("face_detection"); }
        if self.face_rec.initialized() { loaded.push("face_recognition"); }
        if self.reid.initialized() { loaded.push("reid"); }
        if self.ocr.initialized() { loaded.push("ocr"); }
        if self.plate.initialized() { loaded.push("license_plate"); }
        if self.qr.initialized() { loaded.push("qr"); }
        if self.pose.initialized() { loaded.push("pose"); }
        if self.depth.initialized() { loaded.push("depth"); }
        if self.human_parsing.initialized() { loaded.push("human_parsing"); }
        loaded
    }

    /// 深检:对每个已加载引擎的池实例 0 发 dummy 推理,返回健康状态数组。
    ///
    /// 动机:arena OOM 后 cuBLAS **永久失败**,浅检(端口活)完全不可见——僵尸态。
    /// 仅在请求 `/health?deep=true` 时执行,不周期跑。未加载引擎不检查(无 session 可检)。
    pub async fn deep_health_check(&self) -> Vec<EngineHealth> {
        let mut results = Vec::new();
        check_engine!(self.detection, "detection", DetectionEngine, results);
        check_engine!(self.ppe, "ppe", DetectionEngine, results);
        check_engine!(self.fire_smoke, "fire_smoke", DetectionEngine, results);
        check_engine!(self.classification, "classification", ClassificationEngine, results);
        check_engine!(self.obb, "obb", ObbDetectionEngine, results);
        check_engine!(self.segmentation, "segmentation", SegmentationEngine, results);
        check_engine!(self.face_det, "face_detection", DynFaceDet, results);
        check_engine!(self.face_rec, "face_recognition", FaceRecognitionEngine, results);
        check_engine!(self.reid, "reid", ReidEngine, results);
        check_engine!(self.ocr, "ocr", OcrPipeline, results);
        check_engine!(self.plate, "license_plate", LicensePlateEngine, results);
        check_engine!(self.qr, "qr", QrDetector, results);
        check_engine!(self.pose, "pose", DynPose, results);
        check_engine!(self.depth, "depth", DynDepth, results);
        check_engine!(self.human_parsing, "human_parsing", HumanParsingEngine, results);
        results
    }

    // ---- 各引擎懒加载入口 ----

    pub async fn detection(&self) -> Result<Arc<EnginePool<DetectionEngine>>, AppError> {
        self.detection
            .get_or_try_init(|| async {
                self.ensure_enabled("detection")?;
                let path = self.config.detection_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("detection".into()))?;
                self.load_detection_pool("detection", path).await
            })
            .await
            .cloned()
    }

    pub async fn ppe(&self) -> Result<Arc<EnginePool<DetectionEngine>>, AppError> {
        self.ppe
            .get_or_try_init(|| async {
                self.ensure_enabled("ppe")?;
                let path = self.config.ppe_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ppe".into()))?;
                self.load_detection_pool("ppe", path).await
            })
            .await
            .cloned()
    }

    pub async fn fire_smoke(&self) -> Result<Arc<EnginePool<DetectionEngine>>, AppError> {
        self.fire_smoke
            .get_or_try_init(|| async {
                self.ensure_enabled("fire_smoke")?;
                let path = self.config.fire_smoke_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("fire_smoke".into()))?;
                self.load_detection_pool("fire_smoke", path).await
            })
            .await
            .cloned()
    }

    /// 检测族共用加载(ppe/fire_smoke 复用 DetectionEngine,构造参数与预热语义一致)。
    async fn load_detection_pool(
        &self,
        capability: &'static str,
        path: std::path::PathBuf,
    ) -> Result<Arc<EnginePool<DetectionEngine>>, AppError> {
        let model_display = path.display().to_string();
        let t0 = std::time::Instant::now();
        let device = self.device;
        let pool_size = self.config.pool_for(capability);
        let mut engines = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let p = path.clone();
            let engine = tokio::task::spawn_blocking(move || {
                DetectionEngine::for_yolo(&p, device, DETECTION_CONFIDENCE_FLOOR, DETECTION_NMS_IOU)
            })
            .await
            .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
            engines.push(Arc::new(engine));
        }
        tracing::info!("引擎 {capability} 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
        let pool = Arc::new(EnginePool::new(engines));
        warmup_engine(capability, &pool, self.config.warmup_size_for(capability), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
        Ok(pool)
    }

    pub async fn classification(&self) -> Result<Arc<EnginePool<ClassificationEngine>>, AppError> {
        self.classification
            .get_or_try_init(|| async {
                self.ensure_enabled("classification")?;
                let path = self.config.classification_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("classification".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("classification");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        // 引擎阈值降为 0（softmax 输出 ≥0 恒过）：top-k 列表不再被
                        // base 默认 0.5 的 top-1 门控清空（家庭照片实测恒 count=0）。
                        // 门控语义上移到路由层按请求实现（apply_top1_gate），
                        // 与检测族"构造期低阈值+HTTP 层后过滤"同一纪律。
                        let mut e = ClassificationEngine::for_yolo(&p, device)?;
                        e.base.set_confidence_threshold(0.0);
                        Ok::<_, rust_onnx_infer::VisionError>(e)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 classification 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("classification", &pool, self.config.warmup_size_for("classification"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn obb(&self) -> Result<Arc<EnginePool<ObbDetectionEngine>>, AppError> {
        self.obb
            .get_or_try_init(|| async {
                self.ensure_enabled("obb")?;
                let path = self.config.obb_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("obb".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("obb");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || ObbDetectionEngine::new(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 obb 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("obb", &pool, self.config.warmup_size_for("obb"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn segmentation(&self) -> Result<Arc<EnginePool<SegmentationEngine>>, AppError> {
        self.segmentation
            .get_or_try_init(|| async {
                self.ensure_enabled("segmentation")?;
                let path = self.config.segmentation_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("segmentation".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("segmentation");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        SegmentationEngine::for_yolo(&p, device, SEGMENTATION_CONFIDENCE, SEGMENTATION_NMS_IOU)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 segmentation 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("segmentation", &pool, self.config.warmup_size_for("segmentation"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn face_detection(&self) -> Result<Arc<EnginePool<DynFaceDet>>, AppError> {
        self.face_det
            .get_or_try_init(|| async {
                self.ensure_enabled("face_detection")?;
                let path = self.config.face_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_detection".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("face_detection");
                let mut engines: Vec<Arc<DynFaceDet>> = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        create_face_detection_engine(&p, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 face_detection 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("face_detection", &pool, self.config.warmup_size_for("face_detection"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn face_recognition(&self) -> Result<Arc<EnginePool<FaceRecognitionEngine>>, AppError> {
        self.face_rec
            .get_or_try_init(|| async {
                self.ensure_enabled("face_recognition")?;
                let path = self.config.face_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_recognition".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("face_recognition");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        create_face_recognition_engine(&p, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 face_recognition 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("face_recognition", &pool, self.config.warmup_size_for("face_recognition"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn reid(&self) -> Result<Arc<EnginePool<ReidEngine>>, AppError> {
        self.reid
            .get_or_try_init(|| async {
                self.ensure_enabled("reid")?;
                let path = self.config.reid_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("reid".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("reid");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        ReidEngine::new(&p, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 reid 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("reid", &pool, self.config.warmup_size_for("reid"), |e, w, h| {
                    e.embed(&dummy_image(w, h), Rect::new(0, 0, w, h)).map(|_| ())
                })
                .await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn ocr(&self) -> Result<Arc<EnginePool<OcrPipeline>>, AppError> {
        self.ocr
            .get_or_try_init(|| async {
                self.ensure_enabled("ocr")?;
                let det = self.config.ocr_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let rec = self.config.ocr_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let dict = self.config.ocr_dict.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let model_display = format!("{} + {}", det.display(), rec.display());
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("ocr");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let (d, r, dc) = (det.clone(), rec.clone(), dict.clone());
                    let engine = tokio::task::spawn_blocking(move || {
                        create_ocr_pipeline(&d, &r, &dc, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 ocr 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("ocr", &pool, self.config.warmup_size_for("ocr"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn license_plate(&self) -> Result<Arc<EnginePool<LicensePlateEngine>>, AppError> {
        self.plate
            .get_or_try_init(|| async {
                self.ensure_enabled("license_plate")?;
                let det = self.config.plate_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let rec = self.config.plate_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let model_display = format!("{} + {}", det.display(), rec.display());
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("license_plate");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let (d, r) = (det.clone(), rec.clone());
                    let engine = tokio::task::spawn_blocking(move || {
                        // 车牌检测阈值 0.15(引擎默认 0.5 过严):多车牌拼图/远景
                        // 小牌 det 分常低于 0.5(chepai-1 实测 8 牌:0.5→1 个,
                        // 0.15→6 个,余下为极小/模糊牌属模型能力边界)。LPRNet
                        // 另有 rec 分,垃圾候选总 score 低可辨识
                        let mut e = LicensePlateEngine::new(&d, &r, device)?;
                        e.base.set_confidence_threshold(0.15);
                        Ok::<_, rust_onnx_infer::VisionError>(e)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 license_plate 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("license_plate", &pool, self.config.warmup_size_for("license_plate"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn qr(&self) -> Result<Arc<EnginePool<QrDetector>>, AppError> {
        self.qr
            .get_or_try_init(|| async {
                self.ensure_enabled("qr")?;
                let path = self.config.qr_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("qr".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("qr");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || QrDetector::new(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 qr 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("qr", &pool, self.config.warmup_size_for("qr"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn pose(&self) -> Result<Arc<EnginePool<DynPose>>, AppError> {
        self.pose
            .get_or_try_init(|| async {
                self.ensure_enabled("pose")?;
                let path = self.config.pose_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("pose".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("pose");
                let mut engines: Vec<Arc<DynPose>> = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        create_pose_engine(&p, device, 0.25)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 pose 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("pose", &pool, self.config.warmup_size_for("pose"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn depth(&self) -> Result<Arc<EnginePool<DynDepth>>, AppError> {
        self.depth
            .get_or_try_init(|| async {
                self.ensure_enabled("depth")?;
                let path = self.config.depth_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("depth".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("depth");
                let mut engines: Vec<Arc<DynDepth>> = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || create_depth_engine(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 depth 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("depth", &pool, self.config.warmup_size_for("depth"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    pub async fn human_parsing(&self) -> Result<Arc<EnginePool<HumanParsingEngine>>, AppError> {
        self.human_parsing
            .get_or_try_init(|| async {
                self.ensure_enabled("human_parsing")?;
                let path = self.config.human_parsing_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("human_parsing".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("human_parsing");
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        HumanParsingEngine::new(&p, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 human_parsing 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                let pool = Arc::new(EnginePool::new(engines));
                warmup_engine("human_parsing", &pool, self.config.warmup_size_for("human_parsing"), |e, w, h| e.predict(&dummy_image(w, h)).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }
}

/// 引擎池：同一模型的 N 个可互换实例 + 信号量准入 + round-robin 分发。
///
/// 背景：rust-onnx-infer 的 `Session` 锁在引擎实例内部,单实例对并发请求串行;
/// 每实例独立加载模型,天然可并行。池大小 1(默认)时行为与单实例完全一致。
///
/// 设计(与 yolo-rs Session 池同款语义,已在 A100 生产验证):
/// - `Semaphore` 控制同时在途的推理数(= 池大小),permit 是 RAII,cancel/panic 安全;
/// - round-robin 均摊请求,同实例偶发冲突时由引擎内部锁串行——锁等待发生在
///   `spawn_blocking` 的阻塞线程内,不影响 async runtime;
/// - 注意资源预算:池大小 × 每实例 ORT 线程数(默认 4)应 ≈ 可用核数,
///   内存代价为模型大小 × 池大小(对全部已加载引擎生效)。
pub struct EnginePool<E> {
    instances: Vec<Arc<E>>,
    permits: Arc<tokio::sync::Semaphore>,
    next: std::sync::atomic::AtomicUsize,
}

impl<E> EnginePool<E>
where
    E: Send + Sync + 'static,
{
    pub fn new(instances: Vec<Arc<E>>) -> Self {
        // run() 以实例数取模分发,空池会除零——在构造边界显式失败
        assert!(!instances.is_empty(), "EnginePool 至少需要一个实例");
        EnginePool {
            permits: Arc::new(tokio::sync::Semaphore::new(instances.len())),
            instances,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 取一个实例执行闭包(闭包收到 `&E`)。排队上限 = 池大小,全忙时在信号量上等待。
    pub async fn run<T, F>(&self, f: F) -> Result<T, AppError>
    where
        F: FnOnce(&E) -> rust_onnx_infer::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| AppError::Internal("引擎池已关闭".into()))?;
        let idx = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.instances.len();
        let engine = Arc::clone(&self.instances[idx]);
        tokio::task::spawn_blocking(move || f(&engine))
            .await
            .map_err(|e| AppError::Internal(format!("推理任务失败: {e}")))?
            .map_err(AppError::from)
    }

    /// 启动预热:对池内**每个**实例跑一次 dummy 推理。
    ///
    /// 消除引擎反序列化 + cuDNN autotune 懒加载风暴,否则首个并发请求集体卡顿。
    /// **失败告警不阻断**(标记冷启动,首请求延迟高可接受)——与 yolo-rs 同款语义。
    ///
    /// `dummy_fn` 接收 `&E`,返回 `Result<()>`;各引擎用自身 input_size 构造零张量。
    pub async fn warmup<F>(&self, dummy_fn: F)
    where
        F: Fn(&E) -> rust_onnx_infer::Result<()> + Send + Sync + 'static,
    {
        let dummy_fn = Arc::new(dummy_fn);
        for (idx, engine) in self.instances.iter().enumerate() {
            let engine = Arc::clone(engine);
            let dummy_fn = Arc::clone(&dummy_fn);
            let t0 = std::time::Instant::now();
            let result = tokio::task::spawn_blocking(move || dummy_fn(&engine))
                .await
                .map_err(|e| rust_onnx_infer::VisionError::inference(format!("预热任务失败: {e}")))
                .and_then(|r| r);
            match &result {
                Ok(()) => tracing::info!(
                    "引擎池实例 {}/{} 预热完成 ({:?})",
                    idx + 1,
                    self.instances.len(),
                    t0.elapsed()
                ),
                Err(e) => tracing::warn!(
                    "引擎池实例 {}/{} 预热失败(不阻断,首请求延迟可能偏高): {e}",
                    idx + 1,
                    self.instances.len()
                ),
            }
        }
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

/// 构造全零 dummy 图像(预热用)。
fn dummy_image(w: i32, h: i32) -> rust_onnx_infer::Image {
    // input_size 可能返回 -1(动态维度),回退 640
    let w = if w > 0 { w as usize } else { 640 };
    let h = if h > 0 { h as usize } else { 640 };
    rust_onnx_infer::Image::new(w, h, 3)
}

/// 引擎预热 helper:构造零张量,对池内每实例跑 dummy 推理。
///
/// `override_size` 覆盖 dummy 尺寸(来自 `ARGUS_<CAP>_WARMUP_SIZE`),None 沿用
/// 引擎 `input_size`。OCR 必须用小图:960 全零 dummy 实测 20.4s/实例(det 大量伪区域
/// + rec 批前向),320 小图 0.43s/实例,首个真实请求推理不变。
///
/// `predict_fn` 接收 `&E` + `(width, height)`,返回 `Result<()>`。
/// 失败 warn 不阻断(与 yolo-rs 同款语义)。
async fn warmup_engine<E, F>(
    name: &str,
    pool: &EnginePool<E>,
    override_size: Option<(u32, u32)>,
    predict_fn: F,
) where
    E: OnnxInferPredict + Send + Sync + 'static,
    F: Fn(&E, i32, i32) -> rust_onnx_infer::Result<()> + Send + Sync + 'static,
{
    // 覆盖优先;否则从实例 0 取 input_size(各实例同模型,size 一致)
    let (w, h) = override_size
        .map(|(w, h)| (w as i32, h as i32))
        .or_else(|| pool.instances.first().map(|e| e.input_size()))
        .unwrap_or((640, 640));
    tracing::info!("引擎 {name} 预热开始 (dummy {w}x{h})");
    pool.warmup(move |e| predict_fn(e, w, h)).await;
}

/// 引擎预热的 predict 能力 trait(具体类型和 trait object 都可实现)。
trait OnnxInferPredict {
    fn input_size(&self) -> (i32, i32);
}

// 为所有用到预热的引擎类型实现 trait(只暴露 input_size,predict 在闭包里直接调)
impl OnnxInferPredict for DetectionEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for ClassificationEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for ObbDetectionEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for FaceRecognitionEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for ReidEngine {
    // OSNet x1.0 静态输入 128x256(W,H),与模型文件绑定,不取自 session
    fn input_size(&self) -> (i32, i32) { (128, 256) }
}
impl OnnxInferPredict for SegmentationEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for LicensePlateEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for QrDetector {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for HumanParsingEngine {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
impl OnnxInferPredict for OcrPipeline {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self) }
}
// trait object 类型:Box<dyn OnnxInferenceEngine<Output=...>> 通过 deref 调用 trait 方法
impl OnnxInferPredict for DynFaceDet {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self.as_ref()) }
}
impl OnnxInferPredict for DynPose {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self.as_ref()) }
}
impl OnnxInferPredict for DynDepth {
    fn input_size(&self) -> (i32, i32) { rust_onnx_infer::OnnxInferenceEngine::input_size(self.as_ref()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn get_detection_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.detection().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn get_classification_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.classification().await;
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

    // ---- EnginePool 并发行为 ----

    /// 记录 n 个并发 run 中"同时处于闭包内"的峰值数。
    async fn concurrent_peak(pool: std::sync::Arc<EnginePool<usize>>, n: usize) -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let peak = std::sync::Arc::new(AtomicUsize::new(0));
        let current = std::sync::Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..n {
            let pool = std::sync::Arc::clone(&pool);
            let peak = std::sync::Arc::clone(&peak);
            let current = std::sync::Arc::clone(&current);
            handles.push(tokio::spawn(async move {
                let _ = pool
                    .run(move |_e| {
                        let now = current.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(60));
                        current.fetch_sub(1, Ordering::SeqCst);
                        Ok::<(), rust_onnx_infer::VisionError>(())
                    })
                    .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        peak.load(Ordering::SeqCst)
    }

    #[tokio::test]
    async fn pool_of_1_should_serialize_concurrent_runs() {
        let pool = std::sync::Arc::new(EnginePool::new(vec![std::sync::Arc::new(0usize)]));
        let peak = concurrent_peak(std::sync::Arc::clone(&pool), 4).await;
        assert_eq!(peak, 1, "池大小 1 时并发请求必须串行,实际并发峰值 {peak}");
    }

    #[tokio::test]
    async fn pool_of_2_should_allow_two_parallel_runs() {
        let pool = std::sync::Arc::new(EnginePool::new(vec![
            std::sync::Arc::new(0usize),
            std::sync::Arc::new(1usize),
        ]));
        let peak = concurrent_peak(std::sync::Arc::clone(&pool), 4).await;
        assert_eq!(peak, 2, "池大小 2 时应允许 2 路并行,实际并发峰值 {peak}");
    }

    #[tokio::test]
    async fn pool_run_should_return_closure_value_and_instance_visible() {
        // round-robin 应把请求分发到不同实例(2 实例 2 请求,读到的引擎值集合完整)
        let pool = EnginePool::new(vec![
            std::sync::Arc::new(10usize),
            std::sync::Arc::new(20usize),
        ]);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2 {
            let v = pool.run(|e| Ok::<usize, rust_onnx_infer::VisionError>(*e)).await.unwrap();
            seen.insert(v);
        }
        assert_eq!(seen, [10, 20].into_iter().collect(), "两实例应各被命中一次");
    }

    #[tokio::test]
    async fn pool_run_should_map_closure_error_to_inference() {
        let pool = EnginePool::new(vec![std::sync::Arc::new(0usize)]);
        let err = pool
            .run(|_e| Err::<usize, _>(rust_onnx_infer::VisionError::image("boom")))
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::Inference(_)), "实际: {err:?}");
    }

    #[test]
    #[should_panic(expected = "EnginePool 至少需要一个实例")]
    fn empty_pool_should_panic_with_clear_message() {
        // 空池防御:run() 的取模运算在 len=0 时除零,必须在构造边界显式失败
        let _ = EnginePool::<usize>::new(vec![]);
    }

    // ---- 能力白名单(§3.3:强制点在 loader 入口,先于一切模型路径/文件系统访问) ----

    #[tokio::test]
    async fn loader_should_return_capability_disabled_when_not_whitelisted() {
        // 模型路径指向不存在的文件:若错误是 CapabilityDisabled 则证明加载在
        // 触碰文件系统之前就被拒绝;若返回加载失败/未配置,说明强制点放错了位置
        let config = Config {
            detection_model: Some("/nonexistent/yolov8n.onnx".into()),
            enabled_capabilities: Some(vec!["ocr".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.detection().await.err().expect("白名单外应返回错误");
        assert!(
            matches!(err, AppError::CapabilityDisabled(_)),
            "实际: {err:?}"
        );
    }

    #[tokio::test]
    async fn loader_should_proceed_when_capability_whitelisted() {
        // 白名单内 + 未配置模型路径 = ModelNotConfigured(而非白名单错误),
        // 证明白名单放行后走原有路径
        let config = Config {
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.detection().await.err().expect("未配置模型应返回错误");
        assert!(
            matches!(err, AppError::ModelNotConfigured(_)),
            "实际: {err:?}"
        );
    }
}
