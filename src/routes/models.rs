//! `GET /models` — 能力键控的模型清单（配置态 / 加载态 / 模型文件名）。
//!
//! **与 yolo-rs 的 `/models` 语义不同，勿照抄**：yolo-rs 的 models 目录专供检测
//! 模型，"目录扫描任意 *.onnx 即可用"成立；argus 的 models/ 混装检测、分类、
//! OCR、分割等异构模型，naive 目录扫描会把 segformer/ppocr 之类全报成"可互换的
//! 检测模型"——对消费方说谎。故按**能力**键控输出，每项给出该能力实际使用的
//! 模型文件（文件名去扩展名）。
//!
//! 数据全部来自 `Config` 的模型路径字段（配置态）与 `EngineManager::list_loaded`
//! （加载态），只读聚合，不触碰引擎加载逻辑——与吸收计划阶段 2（EngineManager
//! 动态化）无耦合，后者落地后可自然扩展目录扫描出的动态条目。

use std::path::PathBuf;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::routes::health::AppState;

/// 单能力清单。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapabilityModels {
    /// 该能力所需的全部模型路径均已配置（缺任一即 false，引擎无法加载）。
    pub configured: bool,
    /// 在 `ARGUS_ENABLED_CAPABILITIES` 白名单内（未设置白名单 = 恒 true）。
    /// 白名单外的能力在 loader 入口即被拒绝，永远不会加载。
    pub enabled: bool,
    /// 引擎实例已创建（`OnceCell` 已初始化）。
    ///
    /// 注意：`get_or_try_init` 失败时不写入 cell，加载失败后仍为 false——
    /// 下次请求会重新尝试加载。判"推理可用"须配合 `/health?deep=true` 的
    /// `engine_health`。
    pub loaded: bool,
    /// 模型文件名去扩展名（不含目录）。多段式管线（OCR/车牌）含 det + rec，
    /// 字典等非模型文件不计入。
    pub models: Vec<String>,
    /// 文本 Embedding 向量维度（`ARGUS_TEXT_EMBEDDING_DIM` 声明值，loader 加载期
    /// 已校验与引擎一致）。仅 text_embedding 项出现，其余能力不序列化该字段。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim: Option<usize>,
}

/// 组装单个能力项：`model_paths` 中已配置的路径取文件名去扩展名。
/// `dim` 仅文本 Embedding 携带（其余能力传 None，序列化时跳过）。
fn cap(
    model_paths: &[&Option<PathBuf>],
    configured: bool,
    enabled: bool,
    loaded: bool,
    dim: Option<usize>,
) -> CapabilityModels {
    CapabilityModels {
        configured,
        enabled,
        loaded,
        models: model_paths
            .iter()
            .filter_map(|p| p.as_ref())
            .filter_map(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .collect(),
        dim,
    }
}

/// 汇总全部能力的模型清单（键 = 能力名，与 `EngineManager::list_loaded`、
/// `Config::configured_engines` 的命名一致）。
fn capability_models(config: &Config, loaded: &[&str]) -> Map<String, Value> {
    let on = |name: &str| loaded.contains(&name);
    let entries: [(&str, CapabilityModels); 17] = [
        (
            "detection",
            cap(
                &[&config.detection_model],
                config.detection_model.is_some(),
                config.capability_enabled("detection"), on("detection"),
                None,
            ),
        ),
        (
            "ppe",
            cap(
                &[&config.ppe_model],
                config.ppe_model.is_some(),
                config.capability_enabled("ppe"), on("ppe"),
                None,
            ),
        ),
        (
            "fire_smoke",
            cap(
                &[&config.fire_smoke_model],
                config.fire_smoke_model.is_some(),
                config.capability_enabled("fire_smoke"), on("fire_smoke"),
                None,
            ),
        ),
        (
            "classification",
            cap(
                &[&config.classification_model],
                config.classification_model.is_some(),
                config.capability_enabled("classification"), on("classification"),
                None,
            ),
        ),
        (
            "obb",
            cap(
                &[&config.obb_model],
                config.obb_model.is_some(),
                config.capability_enabled("obb"), on("obb"),
                None,
            ),
        ),
        (
            "segmentation",
            cap(
                &[&config.segmentation_model],
                config.segmentation_model.is_some(),
                config.capability_enabled("segmentation"), on("segmentation"),
                None,
            ),
        ),
        (
            "face_detection",
            cap(
                &[&config.face_det_model],
                config.face_det_model.is_some(),
                config.capability_enabled("face_detection"), on("face_detection"),
                None,
            ),
        ),
        (
            "face_recognition",
            cap(
                &[&config.face_rec_model],
                config.face_rec_model.is_some(),
                config.capability_enabled("face_recognition"), on("face_recognition"),
                None,
            ),
        ),
        (
            "reid",
            cap(
                &[&config.reid_model],
                config.reid_model.is_some(),
                config.capability_enabled("reid"), on("reid"),
                None,
            ),
        ),
        (
            "ocr",
            // dict 是字典不是模型：计入 configured，不出现在 models
            cap(
                &[&config.ocr_det_model, &config.ocr_rec_model],
                config.ocr_det_model.is_some()
                    && config.ocr_rec_model.is_some()
                    && config.ocr_dict.is_some(),
                config.capability_enabled("ocr"), on("ocr"),
                None,
            ),
        ),
        (
            "license_plate",
            cap(
                &[&config.plate_det_model, &config.plate_rec_model],
                config.plate_det_model.is_some() && config.plate_rec_model.is_some(),
                config.capability_enabled("license_plate"), on("license_plate"),
                None,
            ),
        ),
        (
            "qr",
            cap(
                &[&config.qr_model],
                config.qr_model.is_some(),
                config.capability_enabled("qr"), on("qr"),
                None,
            ),
        ),
        (
            "pose",
            cap(
                &[&config.pose_model],
                config.pose_model.is_some(),
                config.capability_enabled("pose"), on("pose"),
                None,
            ),
        ),
        (
            "depth",
            cap(
                &[&config.depth_model],
                config.depth_model.is_some(),
                config.capability_enabled("depth"), on("depth"),
                None,
            ),
        ),
        (
            "human_parsing",
            cap(
                &[&config.human_parsing_model],
                config.human_parsing_model.is_some(),
                config.capability_enabled("human_parsing"), on("human_parsing"),
                None,
            ),
        ),
        (
            "text_embedding",
            // tokenizer/dim 非 onnx 文件：计入三件套 configured 判定，
            // 不出现在 models（ocr_dict 先例）；dim 是声明值（/models 刻意不触碰 session）
            cap(
                &[&config.text_embedding_model],
                config.text_embedding_model.is_some()
                    && config.text_embedding_tokenizer.is_some()
                    && config.text_embedding_dim.is_some(),
                config.capability_enabled("text_embedding"), on("text_embedding"),
                config.text_embedding_dim,
            ),
        ),
        (
            "text_reranking",
            // 两件套（model+tokenizer）齐备即 configured，无 dim——输出是分数不是
            // 向量；tokenizer 非 onnx 文件不进 models 清单（ocr_dict 先例）
            cap(
                &[&config.text_reranking_model],
                config.text_reranking_model.is_some()
                    && config.text_reranking_tokenizer.is_some(),
                config.capability_enabled("text_reranking"), on("text_reranking"),
                None,
            ),
        ),
    ];
    entries
        .into_iter()
        .map(|(name, cm)| {
            let value = serde_json::to_value(cm).expect("CapabilityModels 序列化不可能失败");
            (name.to_string(), value)
        })
        .collect()
}

/// GET /models — 返回 `{"status":"success","data":{"capabilities":{能力名:{...}},
/// "dynamic_slots":{槽名:{...}}}}`。
///
/// 消费方（如 safe_bot）取 `capabilities.detection.models` 即该端点可用于目标
/// 检测的模型名清单；`dynamic_slots` 是 slots.toml 动态槽的配置态/加载态
/// （设计 C2：独立键，capabilities 17 项形状不受影响）。
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let loaded = state.engines.list_loaded();
    let mut caps = capability_models(&state.config, &loaded);
    // A4：每能力附 pipeline 标识（与推理响应的 data.pipeline 同串），
    // 消费方可预先比对——换模型后哪些结果需要失效。
    // 走 manager 出口（env override 只在 manager.pipeline_for 生效），
    // 与 provenance 同源；自由表值函数 pipeline_table_value 不感知 override。
    for (name, cap_value) in caps.iter_mut() {
        if let serde_json::Value::Object(map) = cap_value {
            let pipeline = state.engines.pipeline_for(name);
            if !pipeline.is_empty() {
                map.insert("pipeline".into(), serde_json::json!(pipeline));
            }
        }
    }
    // 动态槽清单（设计 C2：独立 dynamic_slots 键，capabilities 的 17 项不动——
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineManager;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn paths(pairs: &[(&str, Option<&str>)]) -> Config {
        let mut config = Config::default();
        for (key, value) in pairs {
            let p = value.map(PathBuf::from);
            match *key {
                "detection" => config.detection_model = p,
                "ppe" => config.ppe_model = p,
                "fire_smoke" => config.fire_smoke_model = p,
                "classification" => config.classification_model = p,
                "reid" => config.reid_model = p,
                "segmentation" => config.segmentation_model = p,
                "ocr_det" => config.ocr_det_model = p,
                "ocr_rec" => config.ocr_rec_model = p,
                "ocr_dict" => config.ocr_dict = p,
                "plate_det" => config.plate_det_model = p,
                "plate_rec" => config.plate_rec_model = p,
                other => panic!("未知字段 {other}"),
            }
        }
        config
    }

    fn cap_of(caps: &Map<String, Value>, name: &str) -> CapabilityModels {
        serde_json::from_value(caps[name].clone()).expect("能力项结构固定")
    }

    #[test]
    fn empty_config_should_list_all_capabilities_unconfigured() {
        let caps = capability_models(&Config::default(), &[]);
        assert_eq!(
            caps.len(),
            17,
            "应有 17 项能力: {:?}",
            caps.keys().collect::<Vec<_>>()
        );
        for (name, value) in &caps {
            let cm = cap_of(&caps, name);
            assert!(!cm.configured, "{name} 无配置应为 configured=false");
            assert!(!cm.loaded, "{name} 未加载应为 loaded=false");
            assert!(cm.models.is_empty(), "{name} 无配置不应有模型: {value}");
        }
    }

    #[test]
    fn detection_configured_should_expose_model_stem_without_extension() {
        let config = paths(&[("detection", Some("/models/yolov8n.onnx"))]);
        let cm = cap_of(&capability_models(&config, &[]), "detection");
        assert!(cm.configured);
        assert_eq!(cm.models, ["yolov8n"], "应为去扩展名的文件名（不含目录）");
        assert!(!cm.loaded, "未加载态由 EngineManager 决定，此处应为 false");
    }

    #[test]
    fn reid_configured_should_expose_model_stem() {
        let config = paths(&[("reid", Some("/models/osnet_x1_0.onnx"))]);
        let cm = cap_of(&capability_models(&config, &[]), "reid");
        assert!(cm.configured);
        assert_eq!(cm.models, ["osnet_x1_0"]);
        assert!(config.configured_engines().contains(&"reid"));
    }

    #[test]
    fn ocr_should_list_det_and_rec_but_exclude_dict() {
        let config = paths(&[
            ("ocr_det", Some("/models/ppocrv5_det.onnx")),
            ("ocr_rec", Some("/models/ppocrv5_rec.onnx")),
            ("ocr_dict", Some("/models/ppocrv5_dict.txt")),
        ]);
        let cm = cap_of(&capability_models(&config, &[]), "ocr");
        assert!(cm.configured, "三段齐备才算已配置");
        assert_eq!(
            cm.models,
            ["ppocrv5_det", "ppocrv5_rec"],
            "字典不是模型，不得出现在 models 里"
        );
    }

    #[test]
    fn ocr_without_dict_should_be_unconfigured() {
        // 缺字典时引擎加载必失败，configured 必须为 false 而非"看似就绪"
        let config = paths(&[
            ("ocr_det", Some("/models/ppocrv5_det.onnx")),
            ("ocr_rec", Some("/models/ppocrv5_rec.onnx")),
        ]);
        let cm = cap_of(&capability_models(&config, &[]), "ocr");
        assert!(!cm.configured, "缺 dict 应为未配置");
    }

    // ---- 文本 Embedding（批 D1：三件套判定 + dim 字段） ----

    #[test]
    fn text_embedding_should_require_full_triple() {
        // 三件套（model+tokenizer+dim）缺任一即 configured=false：引擎加载必失败，
        // 报"已配置"会误导排错（ocr 半套同款语义）
        let only_model = Config {
            text_embedding_model: Some(PathBuf::from("/models/bge/model.onnx")),
            ..Config::default()
        };
        let cm = cap_of(&capability_models(&only_model, &[]), "text_embedding");
        assert!(!cm.configured, "半套配置不得报已配置");

        let full = Config {
            text_embedding_model: Some(PathBuf::from("/models/bge/model.onnx")),
            text_embedding_tokenizer: Some(PathBuf::from("/models/bge/tokenizer.json")),
            text_embedding_dim: Some(512),
            ..Config::default()
        };
        let caps = capability_models(&full, &[]);
        let cm = cap_of(&caps, "text_embedding");
        assert!(cm.configured, "三件套齐备才算已配置");
        assert_eq!(
            cm.models, ["model"],
            "tokenizer 非 onnx 文件，不得出现在 models 清单"
        );
        assert_eq!(cm.dim, Some(512));
        assert!(full.configured_engines().contains(&"text_embedding"));
    }

    #[test]
    fn text_embedding_dim_should_be_skipped_when_unset() {
        // 未配置（dim=None）时序列化不出现 dim 字段——其余 15 项形状逐字节不变
        let caps = capability_models(&Config::default(), &[]);
        let raw = serde_json::to_value(&caps["text_embedding"]).unwrap();
        assert!(raw.get("dim").is_none(), "None 必须跳过序列化: {raw}");
    }

    #[test]
    fn license_plate_should_require_both_det_and_rec() {
        let only_det = paths(&[("plate_det", Some("/models/mnet_plate.onnx"))]);
        assert!(!cap_of(&capability_models(&only_det, &[]), "license_plate").configured);

        let both = paths(&[
            ("plate_det", Some("/models/mnet_plate.onnx")),
            ("plate_rec", Some("/models/Final_LPRNet_model.onnx")),
        ]);
        let cm = cap_of(&capability_models(&both, &[]), "license_plate");
        assert!(cm.configured);
        assert_eq!(cm.models, ["mnet_plate", "Final_LPRNet_model"]);
    }

    #[test]
    fn loaded_should_reflect_engine_manager_state() {
        let config = paths(&[("detection", Some("/models/yolov8n.onnx"))]);
        let caps = capability_models(&config, &["detection"]);
        assert!(
            cap_of(&caps, "detection").loaded,
            "已加载能力应为 loaded=true"
        );
        assert!(
            !cap_of(&caps, "classification").loaded,
            "未加载能力应为 false"
        );
    }

    // ---- enabled 字段(§3.3 能力白名单) ----

    #[test]
    fn enabled_should_default_true_when_whitelist_unset() {
        let config = paths(&[("detection", Some("/models/yolov8n.onnx"))]);
        let cm = cap_of(&capability_models(&config, &[]), "detection");
        assert!(
            cm.enabled,
            "未设置白名单时全部能力 enabled=true（零回归默认）"
        );
    }

    #[test]
    fn enabled_should_reflect_whitelist_exclusion() {
        let mut config = paths(&[("detection", Some("/models/yolov8n.onnx"))]);
        config.enabled_capabilities = Some(vec!["ocr".into()]);
        let caps = capability_models(&config, &[]);
        assert!(
            !cap_of(&caps, "detection").enabled,
            "白名单外能力 enabled=false（即使模型路径已配置）"
        );
        assert!(
            cap_of(&caps, "ocr").enabled,
            "白名单内能力 enabled=true（即使模型未配置）"
        );
    }

    // ---- 路由层 ----

    fn test_state(config: Config) -> AppState {
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn list_models_route_should_return_capability_map() {
        let config = paths(&[("detection", Some("/models/yolov8n.onnx"))]);
        let app = crate::routes::health::build_router(test_state(config), None);
        let resp = app
            .oneshot(Request::get("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["status"], "success");
        assert_eq!(
            body["data"]["capabilities"]["detection"]["models"][0], "yolov8n",
            "消费方按 capabilities.detection.models 取可用清单: {body}"
        );
        assert_eq!(
            body["data"]["capabilities"]["detection"]["configured"],
            true
        );
    }

    #[tokio::test]
    async fn list_models_should_require_auth_when_token_configured() {
        // /models 泄露服务模型布局，不在白名单内（仅 / 与 /ui 豁免）
        let app = crate::routes::health::build_router(
            test_state(Config::default()),
            Some(Arc::new("secret".into())),
        );
        let resp = app
            .oneshot(Request::get("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let text = String::from_utf8_lossy(
            &axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .to_string();
        assert!(text.contains("缺少"), "缺少 key 应被拒: {text}");
    }

    #[tokio::test]
    async fn models_should_report_pipeline_per_capability() {
        // A4：/models 的 pipeline 与推理响应 data.pipeline 同串，消费方预检失效用
        let config = Config {
            detection_model: Some(std::path::PathBuf::from("/models/yolov8n.onnx")),
            ..Default::default()
        };
        let mgr = EngineManager::new(config.clone());
        let app = crate::routes::health::build_router(AppState::new(config, mgr), None);
        let resp = app
            .oneshot(Request::get("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes).to_string();
        assert!(
            text.contains("\"pipeline\":\"yolov8-det640-letterbox-v1\""),
            "detection 应带 pipeline: {text}"
        );
    }

    // ---- F0 golden：/models 全信封形状锁定（动态化只允许"加字段"） ----

    #[tokio::test]
    async fn models_envelope_with_default_config_should_be_exact_golden() {
        let app = crate::routes::health::build_router(test_state(Config::default()), None);
        let resp = app
            .oneshot(Request::get("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = body_json(resp).await;
        let mut expected = json!({"status": "success", "data": {"capabilities": {}, "dynamic_slots": {}}});
        for key in [
            "detection", "ppe", "fire_smoke", "classification", "obb", "segmentation",
            "face_detection", "face_recognition", "reid", "ocr", "license_plate", "qr",
            "pose", "depth", "human_parsing", "text_embedding", "text_reranking",
        ] {
            expected["data"]["capabilities"][key] = json!({
                "configured": false, "enabled": true, "loaded": false, "models": [],
            });
        }
        // 检测族三槽钉死（本次重构变更面），其余 12 个弱锁（防无关能力合法变更致脆）
        expected["data"]["capabilities"]["detection"]["pipeline"] =
            json!("yolov8-det640-letterbox-v1");
        expected["data"]["capabilities"]["ppe"]["pipeline"] =
            json!("yolov8-ppe10cls-det640-letterbox-v1");
        expected["data"]["capabilities"]["fire_smoke"]["pipeline"] =
            json!("yolov8-firesmoke2cls-det640-letterbox-v1");
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
        assert_eq!(pipelines.len(), 17, "应为 17 项能力");
        assert_eq!(uniq.len(), 17, "pipeline 必须互不相同");
        expected["data"]["capabilities"].as_object_mut().unwrap().iter_mut().for_each(|(k, v)| {
            if !matches!(k.as_str(), "detection" | "ppe" | "fire_smoke") {
                v["pipeline"] = body["data"]["capabilities"][k]["pipeline"].clone();
            }
        });
        assert_eq!(body, expected, "/models 默认信封与 golden 不一致");
    }

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
        // 17 项 capabilities 逐字节不受影响（golden 已锁形状，此处再锁数量）
        assert_eq!(body["data"]["capabilities"].as_object().unwrap().len(), 17, "{body}");
    }
}
