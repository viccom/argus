use std::sync::Arc;

use rust_onnx_infer::{
    create_depth_engine, create_face_detection_engine, create_face_recognition_engine,
    create_ocr_pipeline, create_pose_engine, DeviceType,
};
use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::engines::face_detection::FaceDetectResult;
use rust_onnx_infer::engines::face_recognition::FaceRecognitionEngine;
use rust_onnx_infer::engines::human_parsing::HumanParsingEngine;
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
pub struct EngineManager {
    config: Config,
    device: DeviceType,
    detection: tokio::sync::OnceCell<Arc<EnginePool<DetectionEngine>>>,
    obb: tokio::sync::OnceCell<Arc<EnginePool<ObbDetectionEngine>>>,
    face_det: tokio::sync::OnceCell<Arc<EnginePool<DynFaceDet>>>,
    face_rec: tokio::sync::OnceCell<Arc<EnginePool<FaceRecognitionEngine>>>,
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
            obb: tokio::sync::OnceCell::new(),
            face_det: tokio::sync::OnceCell::new(),
            face_rec: tokio::sync::OnceCell::new(),
            ocr: tokio::sync::OnceCell::new(),
            plate: tokio::sync::OnceCell::new(),
            qr: tokio::sync::OnceCell::new(),
            pose: tokio::sync::OnceCell::new(),
            depth: tokio::sync::OnceCell::new(),
            human_parsing: tokio::sync::OnceCell::new(),
        }
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
        if self.pose.initialized() { loaded.push("pose"); }
        if self.depth.initialized() { loaded.push("depth"); }
        if self.human_parsing.initialized() { loaded.push("human_parsing"); }
        loaded
    }

    // ---- 各引擎懒加载入口 ----

    pub async fn detection(&self) -> Result<Arc<EnginePool<DetectionEngine>>, AppError> {
        self.detection
            .get_or_try_init(|| async {
                let path = self.config.detection_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("detection".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || {
                        DetectionEngine::for_yolo(&p, device, 0.25, 0.45)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 detection 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn obb(&self) -> Result<Arc<EnginePool<ObbDetectionEngine>>, AppError> {
        self.obb
            .get_or_try_init(|| async {
                let path = self.config.obb_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("obb".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || ObbDetectionEngine::new(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 obb 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn face_detection(&self) -> Result<Arc<EnginePool<DynFaceDet>>, AppError> {
        self.face_det
            .get_or_try_init(|| async {
                let path = self.config.face_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_detection".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
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
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn face_recognition(&self) -> Result<Arc<EnginePool<FaceRecognitionEngine>>, AppError> {
        self.face_rec
            .get_or_try_init(|| async {
                let path = self.config.face_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("face_recognition".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
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
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn ocr(&self) -> Result<Arc<EnginePool<OcrPipeline>>, AppError> {
        self.ocr
            .get_or_try_init(|| async {
                let det = self.config.ocr_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let rec = self.config.ocr_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let dict = self.config.ocr_dict.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("ocr".into()))?;
                let model_display = format!("{} + {}", det.display(), rec.display());
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
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
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn license_plate(&self) -> Result<Arc<EnginePool<LicensePlateEngine>>, AppError> {
        self.plate
            .get_or_try_init(|| async {
                let det = self.config.plate_det_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let rec = self.config.plate_rec_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("license_plate".into()))?;
                let model_display = format!("{} + {}", det.display(), rec.display());
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let (d, r) = (det.clone(), rec.clone());
                    let engine = tokio::task::spawn_blocking(move || {
                        LicensePlateEngine::new(&d, &r, device)
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 license_plate 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn qr(&self) -> Result<Arc<EnginePool<QrDetector>>, AppError> {
        self.qr
            .get_or_try_init(|| async {
                let path = self.config.qr_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("qr".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || QrDetector::new(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 qr 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn pose(&self) -> Result<Arc<EnginePool<DynPose>>, AppError> {
        self.pose
            .get_or_try_init(|| async {
                let path = self.config.pose_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("pose".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
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
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn depth(&self) -> Result<Arc<EnginePool<DynDepth>>, AppError> {
        self.depth
            .get_or_try_init(|| async {
                let path = self.config.depth_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("depth".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
                let mut engines: Vec<Arc<DynDepth>> = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let p = path.clone();
                    let engine = tokio::task::spawn_blocking(move || create_depth_engine(&p, device))
                        .await
                        .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                tracing::info!("引擎 depth 池({pool_size}) 加载完成: {model_display} ({:?})", t0.elapsed());
                Ok(Arc::new(EnginePool::new(engines)))
            })
            .await
            .cloned()
    }

    pub async fn human_parsing(&self) -> Result<Arc<EnginePool<HumanParsingEngine>>, AppError> {
        self.human_parsing
            .get_or_try_init(|| async {
                let path = self.config.human_parsing_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("human_parsing".into()))?;
                let model_display = path.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.engine_pool;
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
                Ok(Arc::new(EnginePool::new(engines)))
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
        let n = instances.len().max(1);
        EnginePool {
            instances,
            permits: Arc::new(tokio::sync::Semaphore::new(n)),
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn size(&self) -> usize {
        self.instances.len()
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
}
