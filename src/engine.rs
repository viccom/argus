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
use rust_onnx_infer::engines::text::{PoolingKind, TextEmbeddingEngine, TextRerankingEngine};
use rust_onnx_infer::imaging::{FloatMask, Rect};
use rust_onnx_infer::model::PoseResult;
use rust_onnx_infer::OnnxInferenceEngine;

use crate::config::Config;
use crate::error::AppError;
use crate::slots::{Slot, SlotRegistry, SlotSpec};

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
/// 文本 Embedding 截断上限(含特殊 token)。jina-v2 支持到 8192,homepix 文档列
/// 最长约几百字符,512 充裕;变更属预处理语义变化,须同步改 PIPELINE_TABLE 串。
const TEXT_EMBEDDING_MAX_LENGTH: usize = 512;

// 编译期锁定:下限低于 LOW_MODEL_CONFIDENCE 会让 SAHI 路径行为静默漂移
const _: () = assert!(
    DETECTION_CONFIDENCE_FLOOR >= rust_onnx_infer::sahi::config::SahiConfig::LOW_MODEL_CONFIDENCE,
    "DETECTION_CONFIDENCE_FLOOR 必须 >= SahiConfig::LOW_MODEL_CONFIDENCE"
);

// ---- A4 溯源：管线标识表 ----

/// 每能力管线标识（`data.pipeline`）。内容语义：模型家族 + 关键预处理/输出形态。
/// 契约是"换模型或预处理语义变化时必改串"，供消费方（homepix）做向量失效判定；
/// 文件级失效用 `data.model`（文件名#sha256 前 12 位）。人脸识别/OCR/车牌的串
/// 已含其前置检测模型（端点响应与 /models 报同一串，避免两处对照不一致）。
pub const PIPELINE_TABLE: &[(&str, &str)] = &[
    ("detection", "yolov8-det640-letterbox-v1"),
    ("ppe", "yolov8-ppe10cls-det640-letterbox-v1"),
    ("fire_smoke", "yolov8-firesmoke2cls-det640-letterbox-v1"),
    ("classification", "yolov8-cls224-softmax-v1"),
    ("obb", "yolov8-obb640-v1"),
    ("segmentation", "yolov8-seg640-mask-proto-v1"),
    ("face_detection", "yunet-det5pt-v1"),
    ("face_recognition", "yunet+sface112-align5-emb128-v1"),
    ("reid", "osnet-256x128-emb512-v1"),
    ("ocr", "ppocrv5-det960+rec-chn-v1"),
    ("license_plate", "mnet-det+lprnet-rec-v1"),
    ("qr", "wechatqr-ssd-det416-v1"),
    ("pose", "yolov8-pose17-coco-v1"),
    ("depth", "depth-anything-v2-small-v1"),
    ("human_parsing", "segformer-b2-atr18-v1"),
    ("text_embedding", "jina-embeddings-v2-base-zh|mean-pool-norm-v1|maxlen512"),
    ("text_reranking", "jina-reranker-v2-base-multilingual|sigmoid|maxlen512"),
];

/// 能力键 → 管线表值。仅表值查询/测试用；生产出口（/models、推理响应溯源）
/// 必须走 [`EngineManager::pipeline_for`]——env override 只在 manager 方法生效。
/// 未知能力返回空串（调用方以空串为"无溯源"信号）。
pub fn pipeline_table_value(capability: &str) -> &'static str {
    PIPELINE_TABLE
        .iter()
        .find(|(cap, _)| *cap == capability)
        .map(|(_, p)| *p)
        .unwrap_or("")
}

/// 溯源信息：`model` 为"文件名#sha256 前 12 位"（多模型端点用 + 拼接）。
/// `pipeline` 为 String：文本族可经 `ARGUS_*_PIPELINE` 覆盖（运行时 env 串，
/// 装不进 `&'static str`——设计 §4 审查 B2）。
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub model: String,
    pub pipeline: String,
}

/// 计算模型文件的 sha256 前 12 位（hex）。文件不存在返回 None。
pub(crate) fn sha256_prefix_12(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).ok()?;
    let digest = Sha256::digest(&bytes);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Some(hex[..12].to_string())
}

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
    text_embedding: tokio::sync::OnceCell<Arc<EnginePool<TextEmbeddingEngine>>>,
    text_reranking: tokio::sync::OnceCell<Arc<EnginePool<TextRerankingEngine>>>,
    /// A4 溯源缓存：能力键 → ModelInfo（model 串含文件 hash，首查后缓存）。
    model_registry: std::sync::RwLock<std::collections::HashMap<String, ModelInfo>>,
    /// 检测族动态槽（slots.toml）。静态 3 槽保持 OnceCell——registry 纯增量（设计 C1）。
    pub(crate) slots: SlotRegistry<DetectionEngine>,
}

impl EngineManager {
    pub fn new(config: Config) -> Self {
        let device = DeviceType::from_name(&config.device);
        let slots = Self::build_slot_registry(device, config.engine_pool);
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
            text_embedding: tokio::sync::OnceCell::new(),
            text_reranking: tokio::sync::OnceCell::new(),
            model_registry: std::sync::RwLock::new(std::collections::HashMap::new()),
            slots,
        }
    }

    /// 生产 build 闭包：与静态检测槽同参数（低阈值 + 全局 NMS IoU），
    /// class_names 注入（数量不符仅 warn），预热失败 warn 不阻断（设计 §4.2/§4.4）。
    fn build_slot_registry(device: DeviceType, default_pool: usize) -> SlotRegistry<DetectionEngine> {
        let build = move |spec: &SlotSpec, device: DeviceType| -> Result<EnginePool<DetectionEngine>, AppError> {
            let t0 = std::time::Instant::now();
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
            // 加载完成日志先于预热：与静态 loader 的"引擎 {capability} 池(N) 加载完成"
            // 位置一致，耗时不含预热；预热的每实例日志已带各自耗时
            tracing::info!(
                "动态槽 {} 池({pool_size}) 加载完成: {} ({:?})",
                spec.name, spec.model_path.display(), t0.elapsed()
            );
            // 预热：每实例 dummy 推理，失败 warn 不阻断。本闭包运行于 spawn_blocking，
            // 同步执行与静态槽 EnginePool::warmup 的阻塞路径语义等价。
            // 预热亦占用全局构建互斥，期间其它槽首建排队（设计 C6 资源上界的代价）
            let (w, h) = spec.warmup_size.unwrap_or((640, 640));
            for (idx, engine) in pool.instances.iter().enumerate() {
                let t_warm = std::time::Instant::now();
                match engine.predict(&dummy_image(w as i32, h as i32)).map(|_| ()) {
                    Ok(()) => tracing::info!(
                        "槽 {} 实例 {}/{} 预热完成 ({:?})",
                        spec.name, idx + 1, pool_size, t_warm.elapsed()
                    ),
                    Err(e) => tracing::warn!("槽 {} 实例 {}/{} 预热失败(不阻断): {e}", spec.name, idx + 1, pool_size),
                }
            }
            Ok(pool)
        };
        SlotRegistry::new(device, Arc::new(build))
    }

    // ---- A4 溯源 ----

    /// 能力 → 生效 pipeline 串（String）。文本两能力支持 env 覆盖
    /// （`ARGUS_TEXT_EMBEDDING_PIPELINE` / `ARGUS_TEXT_RERANKING_PIPELINE`，
    /// 换模型族必设，否则 pipeline 说谎——设计 §6.2）；None/空串 = 未设回退表值，
    /// 其余能力恒表值。/models 与推理响应都经此出口，保证两处同源。
    pub fn pipeline_for(&self, capability: &str) -> String {
        let override_pipeline = match capability {
            "text_embedding" => self.config.text_embedding_pipeline.as_deref(),
            "text_reranking" => self.config.text_reranking_pipeline.as_deref(),
            _ => None,
        };
        match override_pipeline.filter(|s| !s.is_empty()) {
            Some(s) => s.to_string(),
            None => pipeline_table_value(capability).to_string(),
        }
    }

    /// 能力溯源：`model` = 文件名#sha256 前 12 位（多模型能力用 + 拼接），
    /// `pipeline` = 生效管线标识（表值或 env 覆盖，见 [`Self::pipeline_for`]）。
    ///
    /// 首次调用时读文件计算 hash（spawn_blocking，百 MB 级 ~0.3s 一次性成本），
    /// 之后走缓存。能力未配置（或多模型能力只配一半）返回 None。
    pub async fn model_info(&self, capability: &str) -> Option<ModelInfo> {
        if let Some(hit) = self.model_registry.read().unwrap_or_else(|e| e.into_inner()).get(capability) {
            return Some(hit.clone());
        }
        let paths = self.config.model_paths_for(capability);
        let pipeline = self.pipeline_for(capability);
        if paths.is_empty() || pipeline.is_empty() {
            return None;
        }
        let parts = tokio::task::spawn_blocking(move || {
            paths
                .iter()
                .map(|p| {
                    let name = p.file_name()?.to_string_lossy().to_string();
                    let hash = sha256_prefix_12(p)?;
                    Some(format!("{name}#{hash}"))
                })
                .collect::<Option<Vec<_>>>()
        })
        .await
        .ok()
        .flatten()?;
        let info = ModelInfo {
            model: parts.join("+"),
            pipeline,
        };
        self.model_registry
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(capability.to_string(), info.clone());
        Some(info)
    }

    /// 溯源 JSON 字段对（data.model / data.pipeline），供推理响应注入。
    /// 未配置能力为 null——消费方以非空 model 串作为结果/向量的失效键。
    pub async fn provenance(&self, capability: &str) -> (serde_json::Value, serde_json::Value) {
        match self.model_info(capability).await {
            Some(m) => (serde_json::json!(m.model), serde_json::json!(m.pipeline)),
            None => (serde_json::Value::Null, serde_json::Value::Null),
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
        if self.text_embedding.initialized() { loaded.push("text_embedding"); }
        if self.text_reranking.initialized() { loaded.push("text_reranking"); }
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
        // 文本引擎独立分支：无图像 predict 契约（check_engine! 宏要求 OnnxInferPredict），
        // 用短文本 embed 替代 dummy 图像推理，语义与宏一致（实例 0、ok/degraded）
        if let Some(pool) = self.text_embedding.get() {
            let pool = Arc::clone(pool);
            let status = tokio::task::spawn_blocking(move || {
                pool.instances[0].embed("ping").map(|_| ())
            })
            .await
            .map_err(|e| rust_onnx_infer::VisionError::inference(format!("深检任务失败: {e}")))
            .and_then(|r| r);
            results.push(EngineHealth {
                name: "text_embedding",
                status: if status.is_ok() { "ok" } else { "degraded" },
                error: status.err().map(|e| e.to_string()),
            });
        }
        // reranker 同理：短 pair rerank 替代 dummy 图像推理（实例 0、ok/degraded）
        if let Some(pool) = self.text_reranking.get() {
            let pool = Arc::clone(pool);
            let status = tokio::task::spawn_blocking(move || {
                pool.instances[0].rerank("ping", &["ping"]).map(|_| ())
            })
            .await
            .map_err(|e| rust_onnx_infer::VisionError::inference(format!("深检任务失败: {e}")))
            .and_then(|r| r);
            results.push(EngineHealth {
                name: "text_reranking",
                status: if status.is_ok() { "ok" } else { "degraded" },
                error: status.err().map(|e| e.to_string()),
            });
        }
        results
    }

    // ---- 动态槽（F2：装配、请求路径解析、观测） ----
    // resolve_dynamic_slot 由路由 model 参数消费（detection/batch）；
    // dynamic_loaded/dynamic_slot_health 由 /health deep 与 /models 消费（T9 接线）。

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
            .map(|s| s.spec.name.clone())
            .collect()
    }

    /// 动态槽深检：对每个已加载槽发 dummy 推理（与 check_engine! 同语义，
    /// 未加载槽跳过）。槽名是 String，与静态 engine_health 的 &'static str 分列
    /// 两个字段（设计 C3）。
    pub async fn dynamic_slot_health(&self) -> Vec<crate::slots::DynamicSlotHealth> {
        let mut results = Vec::new();
        for slot in self.slots.slots_snapshot() {
            let Some(pool) = slot.pool() else { continue };
            // 深检 dummy 尺寸与预热对齐（spec.warmup_size，缺省 640）
            let (w, h) = slot.spec.warmup_size.unwrap_or((640, 640));
            let status = pool
                .run(move |e| e.predict(&dummy_image(w as i32, h as i32)).map(|_| ()))
                .await;
            results.push(crate::slots::DynamicSlotHealth {
                name: slot.spec.name.clone(),
                status: if status.is_ok() { "ok" } else { "degraded" },
                error: status.err().map(|e| e.to_string()),
            });
        }
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

    /// 文本 Embedding 引擎懒加载入口（批 D1）。
    ///
    /// 与图像引擎的差异：
    /// - 三件套校验（model+tokenizer+dim）先于文件系统访问；
    /// - **dim 校验**：`engine.dim()` 与 `ARGUS_TEXT_EMBEDDING_DIM` 声明不符即
    ///   Inference 拒载（fail loud，防模型文件被悄悄替换成不同维度）；
    /// - 预热不走 `warmup_engine` helper（其依赖图像 `OnnxInferPredict` trait），
    ///   直接 `pool.warmup` 短文本 embed（reid loader 同款闭包写法）。
    pub async fn text_embedding(&self) -> Result<Arc<EnginePool<TextEmbeddingEngine>>, AppError> {
        self.text_embedding
            .get_or_try_init(|| async {
                self.ensure_enabled("text_embedding")?;
                let model = self.config.text_embedding_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("text_embedding".into()))?;
                let tokenizer = self.config.text_embedding_tokenizer.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("text_embedding".into()))?;
                let expect_dim = self.config.text_embedding_dim
                    .ok_or_else(|| AppError::ModelNotConfigured("text_embedding".into()))?;
                let model_display = model.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("text_embedding");
                // POOLING/MAXLEN env 接线（默认 = 现状：MaskedMean + 512，零回归）；
                // 默认对应 jina 官方约定（mean pooling + normalize，
                // 1_Pooling/config.json 的 mean_tokens=true）；bge 对需 POOLING=cls
                let pooling = pooling_kind_for(&self.config);
                let maxlen = maxlen_for(&self.config);
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let (m, t) = (model.clone(), tokenizer.clone());
                    let engine = tokio::task::spawn_blocking(move || {
                        TextEmbeddingEngine::with_options(
                            &m,
                            &t,
                            device,
                            pooling,
                            true,
                            maxlen,
                            rust_onnx_infer::OnnxRuntimeConfig::defaults(),
                        )
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                // 各实例同模型，取实例 0 的维度与 env 声明比对
                let actual_dim = engines.first().map(|e| e.dim()).unwrap_or(0);
                if actual_dim != expect_dim {
                    return Err(AppError::Inference(format!(
                        "text_embedding 模型维度不符：引擎输出 {actual_dim} 维，ARGUS_TEXT_EMBEDDING_DIM 声明 {expect_dim} 维（拒载，防模型文件被替换）"
                    )));
                }
                // 生效参数全量入日志（含 effective pipeline）：换族忘设 PIPELINE/
                // POOLING 时排错依据（设计 §8 风险登记）
                tracing::info!(
                    "引擎 text_embedding 池({pool_size}) 加载完成: {model_display} dim={actual_dim} pooling={pooling:?} maxlen={maxlen} pipeline={} ({:?})",
                    self.pipeline_for("text_embedding"),
                    t0.elapsed()
                );
                let pool = Arc::new(EnginePool::new(engines));
                pool.warmup(|e| e.embed("warmup").map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }

    /// 文本 Reranking 引擎懒加载入口（T5，设计 §5 数据流 ②）。
    ///
    /// 与 text_embedding loader 的差异：两件套校验（model+tokenizer，无 dim——
    /// 输出是分数不是向量）；maxlen 独立 env（`ARGUS_TEXT_RERANKING_MAXLEN`，
    /// 两族 reranker 支持更长上下文，与 embedding 取值本就该独立）。
    /// 池化不适用（cross-encoder 单 logit + sigmoid，fork 引擎内固定）。
    /// 生产消费方在 T6 路由接线后就位，届时移除 allow（增量交付中间态）。
    #[allow(dead_code)]
    pub async fn text_reranking(&self) -> Result<Arc<EnginePool<TextRerankingEngine>>, AppError> {
        self.text_reranking
            .get_or_try_init(|| async {
                self.ensure_enabled("text_reranking")?;
                let model = self.config.text_reranking_model.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("text_reranking".into()))?;
                let tokenizer = self.config.text_reranking_tokenizer.clone()
                    .ok_or_else(|| AppError::ModelNotConfigured("text_reranking".into()))?;
                let model_display = model.display().to_string();
                let t0 = std::time::Instant::now();
                let device = self.device;
                let pool_size = self.config.pool_for("text_reranking");
                let maxlen = self.config.text_reranking_maxlen.unwrap_or(512);
                let mut engines = Vec::with_capacity(pool_size);
                for _ in 0..pool_size {
                    let (m, t) = (model.clone(), tokenizer.clone());
                    let engine = tokio::task::spawn_blocking(move || {
                        // 构造期校验 logits 输出存在且声明末维 == 1（fail loud，
                        // 防换错模型文件——fork reranking.rs）
                        TextRerankingEngine::with_options(
                            &m,
                            &t,
                            device,
                            maxlen,
                            rust_onnx_infer::OnnxRuntimeConfig::defaults(),
                        )
                    })
                    .await
                    .map_err(|e| AppError::Internal(format!("引擎加载任务失败: {e}")))??;
                    engines.push(Arc::new(engine));
                }
                // 各实例同构造参数，取实例 0 的 max_length 回显（截断上限生效凭据）
                let max_length = engines.first().map(|e| e.max_length()).unwrap_or(maxlen);
                // 生效参数入日志（含 effective pipeline）：换族忘设 PIPELINE 时排错依据
                tracing::info!(
                    "引擎 text_reranking 池({pool_size}) 加载完成: {model_display} maxlen={max_length} pipeline={} ({:?})",
                    self.pipeline_for("text_reranking"),
                    t0.elapsed()
                );
                let pool = Arc::new(EnginePool::new(engines));
                pool.warmup(|e| e.rerank("预热", &["预热文档"]).map(|_| ())).await;
                Ok(pool)
            })
            .await
            .cloned()
    }
}

// ---- 文本 Embedding 构造参数接线（env → fork 引擎参数，纯函数供单测） ----

/// `ARGUS_TEXT_EMBEDDING_POOLING` → fork `PoolingKind`。
/// 未设/空串/`mean` → MaskedMean（jina 现状，零回归）；`cls` → Cls（bge 对）。
/// 非法值由 `Config::validate` 在启动期拦截（fail loud），此处 match 不到的
/// 分支仅防直连构造绕过 validate——兜底 MaskedMean + warn，不 panic。
fn pooling_kind_for(config: &Config) -> PoolingKind {
    match config.text_embedding_pooling.as_deref().filter(|s| !s.is_empty()) {
        Some("cls") => PoolingKind::Cls,
        Some("mean") | None => PoolingKind::MaskedMean,
        Some(other) => {
            tracing::warn!(
                "ARGUS_TEXT_EMBEDDING_POOLING 非法值 \"{other}\"，兜底 mean（启动期 validate 应已拦截）"
            );
            PoolingKind::MaskedMean
        }
    }
}

/// `ARGUS_TEXT_EMBEDDING_MAXLEN` → 截断上限；未设回退常量默认 512
/// （常量保留作默认值来源，变更属预处理语义变化须同步改 PIPELINE_TABLE 串）。
fn maxlen_for(config: &Config) -> usize {
    config.text_embedding_maxlen.unwrap_or(TEXT_EMBEDDING_MAX_LENGTH)
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
    async fn get_text_embedding_engine_should_error_when_not_configured() {
        let config = Config::default();
        let mgr = EngineManager::new(config);
        let result = mgr.text_embedding().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn text_embedding_with_partial_config_should_error_not_configured() {
        // 三件套（model+tokenizer+dim）缺任一必须在触碰文件系统之前拒绝，
        // 不得把半套配置漏成加载失败（inference_failed）
        let config = Config {
            text_embedding_model: Some("/nonexistent/model.onnx".into()),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.text_embedding().await.err().expect("半套配置应返回错误");
        assert!(matches!(err, AppError::ModelNotConfigured(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn text_embedding_loader_should_return_capability_disabled_when_not_whitelisted() {
        // 三件套齐备但白名单外：CapabilityDisabled 证明拒绝先于文件系统访问
        let config = Config {
            text_embedding_model: Some("/nonexistent/model.onnx".into()),
            text_embedding_tokenizer: Some("/nonexistent/tokenizer.json".into()),
            text_embedding_dim: Some(512),
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.text_embedding().await.err().expect("白名单外应返回错误");
        assert!(matches!(err, AppError::CapabilityDisabled(_)), "实际: {err:?}");
    }

    // ---- text_reranking loader 三态（T5，两件套 model+tokenizer） ----

    #[tokio::test]
    async fn get_text_reranking_engine_should_error_when_not_configured() {
        let mgr = EngineManager::new(Config::default());
        let result = mgr.text_reranking().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(AppError::ModelNotConfigured(_))));
    }

    #[tokio::test]
    async fn text_reranking_with_partial_config_should_error_not_configured() {
        // 两件套（model+tokenizer）缺任一必须在触碰文件系统之前拒绝，
        // 不得把半套配置漏成加载失败（inference_failed）
        let config = Config {
            text_reranking_model: Some("/nonexistent/model.onnx".into()),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.text_reranking().await.err().expect("半套配置应返回错误");
        assert!(matches!(err, AppError::ModelNotConfigured(_)), "实际: {err:?}");
    }

    #[tokio::test]
    async fn text_reranking_loader_should_return_capability_disabled_when_not_whitelisted() {
        // 两件套齐备但白名单外：CapabilityDisabled 证明拒绝先于文件系统访问
        let config = Config {
            text_reranking_model: Some("/nonexistent/model.onnx".into()),
            text_reranking_tokenizer: Some("/nonexistent/tokenizer.json".into()),
            enabled_capabilities: Some(vec!["detection".to_string()]),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let err = mgr.text_reranking().await.err().expect("白名单外应返回错误");
        assert!(matches!(err, AppError::CapabilityDisabled(_)), "实际: {err:?}");
    }

    // ---- effective pipeline 出口（env override，/models 与推理响应同源） ----

    #[test]
    fn pipeline_for_should_return_table_value_by_default() {
        // 不设 override：全部能力 = 表值（既有 16 能力 + 文本两项默认串）
        let mgr = EngineManager::new(Config::default());
        assert_eq!(mgr.pipeline_for("detection"), "yolov8-det640-letterbox-v1");
        assert_eq!(
            mgr.pipeline_for("text_embedding"),
            pipeline_table_value("text_embedding")
        );
        assert_eq!(
            mgr.pipeline_for("text_reranking"),
            pipeline_table_value("text_reranking")
        );
        assert_eq!(mgr.pipeline_for("nope"), "", "未知能力空串（表值语义）");
    }

    #[test]
    fn pipeline_for_should_prefer_nonempty_override() {
        // 换模型族必设 PIPELINE 覆盖（设计 §6.2）：非空 override 优先于表值
        let config = Config {
            text_embedding_pipeline: Some("custom|v2".into()),
            text_reranking_pipeline: Some("bge-reranker-base|sigmoid|maxlen512".into()),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        assert_eq!(mgr.pipeline_for("text_embedding"), "custom|v2");
        assert_eq!(
            mgr.pipeline_for("text_reranking"),
            "bge-reranker-base|sigmoid|maxlen512"
        );
        // override 只对声明的能力生效，不串扰其他能力
        assert_eq!(mgr.pipeline_for("detection"), "yolov8-det640-letterbox-v1");
    }

    #[test]
    fn pipeline_for_should_treat_empty_override_as_unset() {
        // 空串 = 未设（env 解析约定）：回退表值，不得输出空 pipeline 串
        let config = Config {
            text_embedding_pipeline: Some(String::new()),
            text_reranking_pipeline: Some(String::new()),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        assert_eq!(
            mgr.pipeline_for("text_embedding"),
            pipeline_table_value("text_embedding")
        );
        assert_eq!(
            mgr.pipeline_for("text_reranking"),
            pipeline_table_value("text_reranking")
        );
    }

    #[tokio::test]
    async fn model_info_should_reflect_pipeline_override() {
        // 溯源出口（model_info/provenance）走 manager.pipeline_for：
        // override 须传导到 /models 与推理响应，两处同源（设计 §4）
        let dir = std::env::temp_dir().join("argus-test-provenance-rerank");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("fake_reranker.onnx");
        std::fs::write(&model, b"hello").unwrap();
        let config = Config {
            text_reranking_model: Some(model),
            text_reranking_pipeline: Some("bge-reranker-base|sigmoid|maxlen512".into()),
            ..Config::default()
        };
        let mgr = EngineManager::new(config);
        let mi = mgr.model_info("text_reranking").await.expect("已配置应可解析");
        assert_eq!(mi.pipeline, "bge-reranker-base|sigmoid|maxlen512");
    }

    // ---- POOLING/MAXLEN env 接线（纯函数，loader 消费） ----

    #[test]
    fn pooling_kind_for_should_map_env_to_pooling_kind() {
        use rust_onnx_infer::engines::text::PoolingKind;
        // 未设/空串/mean → MaskedMean（jina 现状零回归）；cls → Cls（bge 对）
        assert_eq!(pooling_kind_for(&Config::default()), PoolingKind::MaskedMean);
        assert_eq!(
            pooling_kind_for(&Config {
                text_embedding_pooling: Some(String::new()),
                ..Config::default()
            }),
            PoolingKind::MaskedMean,
        );
        assert_eq!(
            pooling_kind_for(&Config {
                text_embedding_pooling: Some("mean".into()),
                ..Config::default()
            }),
            PoolingKind::MaskedMean,
        );
        assert_eq!(
            pooling_kind_for(&Config {
                text_embedding_pooling: Some("cls".into()),
                ..Config::default()
            }),
            PoolingKind::Cls,
        );
    }

    #[test]
    fn pooling_kind_for_should_fall_back_to_mean_on_illegal_value() {
        use rust_onnx_infer::engines::text::PoolingKind;
        // 非法值由 Config::validate 在启动期拦截（fail loud）；此分支只防
        // 直连构造绕过 validate 的路径——兜底 mean 而非 panic/静默选错池化
        assert_eq!(
            pooling_kind_for(&Config {
                text_embedding_pooling: Some("avg".into()),
                ..Config::default()
            }),
            PoolingKind::MaskedMean,
        );
    }

    #[test]
    fn maxlen_for_should_use_env_or_const_default() {
        // 常量保留作默认值来源（512），env 显式值优先
        assert_eq!(maxlen_for(&Config::default()), TEXT_EMBEDDING_MAX_LENGTH);
        assert_eq!(
            maxlen_for(&Config {
                text_embedding_maxlen: Some(256),
                ..Config::default()
            }),
            256,
        );
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

    // ---- F0 特征测试：静态检测族 3 槽行为锁定（动态化重构前的安全网） ----

    #[tokio::test]
    async fn ppe_loader_should_return_capability_disabled_when_not_whitelisted() {
        // 模型路径指向不存在的文件:若错误是 CapabilityDisabled 则证明加载在
        // 触碰文件系统之前就被拒绝;若返回加载失败/未配置,说明强制点放错了位置
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
        // 模型路径指向不存在的文件:若错误是 CapabilityDisabled 则证明加载在
        // 触碰文件系统之前就被拒绝;若返回加载失败/未配置,说明强制点放错了位置
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
        // 缺省配置下三槽都必须是 ModelNotConfigured（懒加载入口未破坏）。
        // 注：EnginePool 未实现 Debug，取 err 后对 AppError（derive(Debug)）格式化
        let mgr = EngineManager::new(Config::default());
        for result in [mgr.detection().await, mgr.ppe().await, mgr.fire_smoke().await] {
            let err = result.err().expect("缺省配置应返回错误");
            assert!(matches!(err, AppError::ModelNotConfigured(_)), "实际: {err:?}");
        }
    }

    // ---- A4 溯源单测 ----

    #[test]
    fn pipeline_table_should_cover_all_capabilities_distinctly() {
        // 码表性质：每能力一条非空 pipeline，且互不相同（撞串 = 失效判定失真）
        let mut seen = std::collections::HashSet::new();
        for cap in crate::config::CAPABILITY_KEYS {
            let p = pipeline_table_value(cap);
            assert!(!p.is_empty(), "{cap} 缺 pipeline");
            assert!(seen.insert(p), "{cap} 的 pipeline 与其他能力撞串: {p}");
        }
    }

    #[tokio::test]
    async fn model_info_should_be_none_when_capability_unconfigured() {
        let mgr = EngineManager::new(crate::config::Config::default());
        assert!(mgr.model_info("detection").await.is_none());
    }

    #[tokio::test]
    async fn model_info_should_report_filename_and_sha_prefix() {
        // 已知内容（b"hello"）的 sha256 = 2cf24dba5fb0a30e...，前 12 位固定
        let dir = std::env::temp_dir().join("argus-test-provenance");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("fake_model.onnx");
        std::fs::write(&model, b"hello").unwrap();
        let config = crate::config::Config {
            detection_model: Some(model.clone()),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        let mi = mgr.model_info("detection").await.expect("已配置应可解析");
        assert!(mi.model.starts_with("fake_model.onnx#"), "实际: {}", mi.model);
        assert!(mi.model.ends_with("2cf24dba5fb0"), "sha256 前 12 位不符: {}", mi.model);
        assert!(mi.model.len() == "fake_model.onnx#".len() + 12);

        // 二次调用走缓存（同值）
        let mi2 = mgr.model_info("detection").await.unwrap();
        assert_eq!(mi.model, mi2.model);
    }

    #[tokio::test]
    async fn model_info_for_ocr_should_join_det_and_rec() {
        let dir = std::env::temp_dir().join("argus-test-provenance2");
        std::fs::create_dir_all(&dir).unwrap();
        let det = dir.join("ppocrv5_det.onnx");
        let rec = dir.join("ppocrv5_rec.onnx");
        std::fs::write(&det, b"det-bytes").unwrap();
        std::fs::write(&rec, b"rec-bytes").unwrap();
        let config = crate::config::Config {
            ocr_det_model: Some(det),
            ocr_rec_model: Some(rec),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        let mi = mgr.model_info("ocr").await.expect("双模型都配置才可解析");
        assert_eq!(mi.model.matches('#').count(), 2, "OCR 双模型应拼接: {}", mi.model);
        assert!(mi.model.contains("ppocrv5_det.onnx") && mi.model.contains("ppocrv5_rec.onnx"));
    }

    #[tokio::test]
    async fn model_info_for_ocr_should_be_none_when_only_half_configured() {
        let dir = std::env::temp_dir().join("argus-test-provenance3");
        std::fs::create_dir_all(&dir).unwrap();
        let det = dir.join("only_det.onnx");
        std::fs::write(&det, b"x").unwrap();
        let config = crate::config::Config {
            ocr_det_model: Some(det),
            ..Default::default()
        };
        let mgr = EngineManager::new(config);
        assert!(mgr.model_info("ocr").await.is_none(), "缺 rec 模型不应报告半套溯源");
    }

    // ---- 动态槽接线（F2） ----

    use crate::slots::{parse_slots_toml, SLOT_PIPELINE};

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("argus-em-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

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
}
