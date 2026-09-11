//! `GET /models` — 能力键控的模型清单（配置态 / 加载态 / 模型文件名）。
//!
//! **与 yolo-rs 的 `/models` 语义不同，勿照抄**：yolo-rs 的 models 目录专供检测
//! 模型，"目录扫描任意 *.onnx 即可用"成立；vision-ext 的 models/ 混装检测、分类、
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
    /// 引擎实例已创建（`OnceCell` 已初始化）。
    ///
    /// 注意：`OnceCell` 会缓存失败实例，加载失败时此处也为 true——判"推理可用"
    /// 须配合 `/health?deep=true` 的 `engine_health`。
    pub loaded: bool,
    /// 模型文件名去扩展名（不含目录）。多段式管线（OCR/车牌）含 det + rec，
    /// 字典等非模型文件不计入。
    pub models: Vec<String>,
}

/// 组装单个能力项：`model_paths` 中已配置的路径取文件名去扩展名。
fn cap(model_paths: &[&Option<PathBuf>], configured: bool, loaded: bool) -> CapabilityModels {
    CapabilityModels {
        configured,
        loaded,
        models: model_paths
            .iter()
            .filter_map(|p| p.as_ref())
            .filter_map(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .collect(),
    }
}

/// 汇总全部能力的模型清单（键 = 能力名，与 `EngineManager::list_loaded`、
/// `Config::configured_engines` 的命名一致）。
fn capability_models(config: &Config, loaded: &[&str]) -> Map<String, Value> {
    let on = |name: &str| loaded.contains(&name);
    let entries: [(&str, CapabilityModels); 11] = [
        (
            "detection",
            cap(
                &[&config.detection_model],
                config.detection_model.is_some(),
                on("detection"),
            ),
        ),
        (
            "classification",
            cap(
                &[&config.classification_model],
                config.classification_model.is_some(),
                on("classification"),
            ),
        ),
        (
            "obb",
            cap(&[&config.obb_model], config.obb_model.is_some(), on("obb")),
        ),
        (
            "face_detection",
            cap(
                &[&config.face_det_model],
                config.face_det_model.is_some(),
                on("face_detection"),
            ),
        ),
        (
            "face_recognition",
            cap(
                &[&config.face_rec_model],
                config.face_rec_model.is_some(),
                on("face_recognition"),
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
                on("ocr"),
            ),
        ),
        (
            "license_plate",
            cap(
                &[&config.plate_det_model, &config.plate_rec_model],
                config.plate_det_model.is_some() && config.plate_rec_model.is_some(),
                on("license_plate"),
            ),
        ),
        (
            "qr",
            cap(&[&config.qr_model], config.qr_model.is_some(), on("qr")),
        ),
        (
            "pose",
            cap(
                &[&config.pose_model],
                config.pose_model.is_some(),
                on("pose"),
            ),
        ),
        (
            "depth",
            cap(
                &[&config.depth_model],
                config.depth_model.is_some(),
                on("depth"),
            ),
        ),
        (
            "human_parsing",
            cap(
                &[&config.human_parsing_model],
                config.human_parsing_model.is_some(),
                on("human_parsing"),
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

/// GET /models — 返回 `{"status":"success","data":{"capabilities":{能力名:{...}}}}`。
///
/// 消费方（如 safe_bot）取 `capabilities.detection.models` 即该端点可用于目标
/// 检测的模型名清单。
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let loaded = state.engines.list_loaded();
    let caps = capability_models(&state.config, &loaded);
    Json(json!({ "status": "success", "data": { "capabilities": caps } }))
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
                "classification" => config.classification_model = p,
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
            11,
            "应有 11 项能力: {:?}",
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
}
