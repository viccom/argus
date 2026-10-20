//! 姿态几何规则路由:`POST /pose-rules`(摔倒/举手判定)。
//!
//! 任务 22(批 1.5):零模型纯算法——输入 COCO 17 关键点(内部先跑 `/pose`
//! 同款推理),几何规则投票出事件。设计依据:`docs/model-ecosystem-research.md`
//! 表 1 白捡项,摔倒事件直击 safe_bot 高处作业违章(高坠占施工事故 52-59%)。
//!
//! 规则语义(数值阈值均可单测复现):
//! - **fall(摔倒)** 三票制,≥2 票命中,confidence = 命中票数 / 3:
//!   1. 躯干向量与竖直方向夹角 > 50°(肩中点→髋中点);
//!   2. bbox 宽高比 > 1.15(站立人高>宽,躺倒反转);
//!   3. 躯干纵向投影 / bbox 高 < 0.40(躺平时躯干横向);
//! - **raise_hand(举手)**:任一侧腕点高于同侧肩点 0.06×bbox 高以上,且腕/肩
//!   关键点分均 > 0.3;confidence = (肩.y - 腕.y) / bbox 高。
//! - **skip_low_quality**:肩/髋 4 关键点 score>0.3 的不足 3 个——有人但
//!   无法判定,显式输出让消费方不误读"无事件=无异常"。

use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};

use rust_onnx_infer::model::{coco, PoseResult};

use crate::error::AppError;
use crate::routes::common::{acquire_permit, decode_image};
use crate::routes::health::AppState;

// COCO 17 关键点索引
const L_SHOULDER: usize = 5;
const R_SHOULDER: usize = 6;
const L_WRIST: usize = 9;
const R_WRIST: usize = 10;
const L_HIP: usize = 11;
const R_HIP: usize = 12;

/// 关键点分低于此值不参与判定
const KP_MIN_SCORE: f64 = 0.3;

#[derive(Serialize, Clone, Debug)]
pub struct PoseRuleEvent {
    pub person: usize,
    pub rule: &'static str,
    pub confidence: f64,
    pub evidence: String,
}

/// 单人关键点 (x, y, score) × 17。
pub type Keypoints = [(f64, f64, f64); 17];

/// evaluate_pose 对单人关键点跑几何规则,返回命中事件(可空:正常站立)。
/// 低质量关键点返回 skip_low_quality 单事件。
pub fn evaluate_pose(kps: &Keypoints, bbox: [f64; 4], person: usize) -> Vec<PoseRuleEvent> {
    // 关键点质量门:肩/髋 4 点中 score>阈值 至少 3 个,否则显式跳过
    let torso_kps = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP];
    let valid = torso_kps.iter().filter(|&&i| kps[i].2 > KP_MIN_SCORE).count();
    if valid < 3 {
        return vec![PoseRuleEvent {
            person,
            rule: "skip_low_quality",
            confidence: 0.0,
            evidence: format!("torso_keypoints_valid={valid}/4"),
        }];
    }

    let mid_sh = midpoint(&kps[L_SHOULDER], &kps[R_SHOULDER]);
    let mid_hip = midpoint(&kps[L_HIP], &kps[R_HIP]);
    let (bx1, by1, bx2, by2) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let bw = (bx2 - bx1).max(1e-6);
    let bh = (by2 - by1).max(1e-6);

    // ---- fall:三票制 ----
    let dx = (mid_hip.0 - mid_sh.0).abs();
    let dy = (mid_hip.1 - mid_sh.1).abs();
    let torso_angle = dx.atan2(dy).to_degrees(); // 肩→髋竖直时 0°,躺平 90°
    let bbox_ar = bw / bh;
    let torso_span_ratio = (mid_sh.1 - mid_hip.1).abs() / bh;
    let mut votes = 0;
    if torso_angle > 50.0 {
        votes += 1;
    }
    if bbox_ar > 1.15 {
        votes += 1;
    }
    if torso_span_ratio < 0.40 {
        votes += 1;
    }
    let mut events = Vec::new();
    if votes >= 2 {
        events.push(PoseRuleEvent {
            person,
            rule: "fall",
            confidence: votes as f64 / 3.0,
            evidence: format!(
                "torso_angle={torso_angle:.0}deg bbox_ar={bbox_ar:.2} torso_span={torso_span_ratio:.2} votes={votes}/3"
            ),
        });
    }

    // ---- raise_hand:腕高于同侧肩 0.06×bbox 高以上 ----
    for (wrist, shoulder, side) in [(L_WRIST, L_SHOULDER, "left"), (R_WRIST, R_SHOULDER, "right")] {
        let (w, s) = (&kps[wrist], &kps[shoulder]);
        if w.2 > KP_MIN_SCORE && s.2 > KP_MIN_SCORE && s.1 - w.1 > 0.06 * bh {
            events.push(PoseRuleEvent {
                person,
                rule: "raise_hand",
                confidence: (s.1 - w.1) / bh,
                evidence: format!("{side}_wrist_above_shoulder_by={:.2}bh", (s.1 - w.1) / bh),
            });
            break; // 一人一次举手事件
        }
    }

    events
}

fn midpoint(a: &(f64, f64, f64), b: &(f64, f64, f64)) -> (f64, f64) {
    ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
}

// ---- 端点 ----

#[derive(Serialize)]
struct PoseWithEventsDto {
    confidence: f64,
    bbox: [f64; 4],
    keypoints: Vec<KeypointDto>,
    events: Vec<PoseRuleEvent>,
}

#[derive(Serialize)]
struct KeypointDto {
    x: f64,
    y: f64,
    score: f64,
    name: &'static str,
}

/// POST /pose-rules — 姿态推理 + 摔倒/举手规则判定。
pub async fn pose_rules(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    let params = parse_file_multipart(multipart).await?;
    let image = decode_image(params.file, state.config.max_image_pixels).await?;

    let pool = state.engines.pose().await?;
    let permit = acquire_permit(&state).await?;

    let t0 = std::time::Instant::now();
    let results = pool.run(move |e| e.predict(&image)).await?;
    let elapsed = t0.elapsed().as_millis();

    let dtos: Vec<PoseWithEventsDto> = results
        .iter()
        .enumerate()
        .map(|(i, p)| pose_with_events(p, i))
        .collect();
    let events_total: usize = dtos.iter().map(|d| d.events.len()).sum();
    let count = dtos.len();
    tracing::info!(
        endpoint = "pose_rules",
        count,
        events = events_total,
        inference_ms = elapsed as u64,
        "推理+规则完成"
    );

    let (model, _pipeline) = state.engines.provenance("pose").await;
    Ok(Json(json!({
        "status": "ok",
        "data": {
            "queue_ms": permit.queue_ms,
            "model": model,
            "pipeline": "yolov8-pose17-coco-rules-v1",
            "poses": dtos,
            "count": count,
            "events": events_total,
            "inference_ms": elapsed,
        }
    })))
}

struct FileParams {
    file: Vec<u8>,
}

async fn parse_file_multipart(mut multipart: Multipart) -> Result<FileParams, AppError> {
    let mut file: Option<Vec<u8>> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| crate::routes::common::map_field_err(e, "multipart 解析失败"))?
    {
        match field.name() {
            Some("file") => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("file 读取失败: {e}")))?;
                if bytes.is_empty() {
                    return Err(AppError::BadRequest("file 字段为空".into()));
                }
                file = Some(bytes.to_vec());
            }
            _ => {
                // 未知字段忽略(与其他端点一致的前向兼容语义)
                let _ = field.bytes().await;
            }
        }
    }
    Ok(FileParams {
        file: file.ok_or_else(|| AppError::BadRequest("缺少 file 字段".into()))?,
    })
}

fn pose_with_events(p: &PoseResult, person: usize) -> PoseWithEventsDto {
    let d = &p.detection;
    let mut kps: Keypoints = [(0.0, 0.0, 0.0); 17];
    for (i, k) in p.keypoints.iter().take(17).enumerate() {
        kps[i] = (k.x as f64, k.y as f64, k.score as f64);
    }
    let events = evaluate_pose(&kps, [d.x1(), d.y1(), d.x2(), d.y2()], person);
    let keypoints = p
        .keypoints
        .iter()
        .enumerate()
        .map(|(i, k)| KeypointDto {
            x: k.x as f64,
            y: k.y as f64,
            score: k.score as f64,
            name: coco::NAMES.get(i).copied().unwrap_or("?"),
        })
        .collect();
    PoseWithEventsDto {
        confidence: d.confidence(),
        bbox: [d.x1(), d.y1(), d.x2(), d.y2()],
        keypoints,
        events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造 17 个关键点(全部 score=1),坐标按传参覆盖。
    fn kps(over: &[(usize, f64, f64)]) -> Keypoints {
        let mut k = [(100.0, 100.0, 1.0); 17];
        for &(i, x, y) in over {
            k[i] = (x, y, 1.0);
        }
        k
    }

    fn rules_of(events: &[PoseRuleEvent]) -> Vec<&str> {
        events.iter().map(|e| e.rule).collect()
    }

    // ---- fall:三票制 ----

    // ---- handler 错误路径(照 scene.rs 测试模式,不依赖模型文件) ----

    use crate::config::Config;
    use crate::engine::EngineManager;
    use crate::routes::health::{build_router, AppState};
    use axum::body::Body;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let config = Config::default();
        let mgr = EngineManager::new(config.clone());
        AppState::new(config, mgr)
    }

    async fn body_text(resp: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn multipart_png_with_fields(fields: &[(&str, &str)]) -> Vec<u8> {
        let img = image::DynamicImage::new_rgb8(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let img_bytes = buf.into_inner();
        let boundary = "----testboundary";
        let mut full = Vec::new();
        for (name, value) in fields {
            full.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
        }
        full.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n").as_bytes());
        full.extend_from_slice(&img_bytes);
        full.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        full
    }

    fn multipart_no_file() -> Vec<u8> {
        b"------testboundary--\r\n".to_vec()
    }

    async fn post(path: &str, body: Vec<u8>) -> String {
        let app = build_router(test_state(), None);
        let resp = app
            .oneshot(
                axum::http::Request::post(path)
                    .header("content-type", "multipart/form-data; boundary=----testboundary")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        body_text(resp).await
    }

    #[tokio::test]
    async fn pose_rules_without_file_returns_error_envelope() {
        let text = post("/pose-rules", multipart_no_file()).await;
        assert!(text.contains("缺少 file"), "{text}");
    }

    #[tokio::test]
    async fn pose_rules_when_model_not_configured_returns_error_envelope() {
        // 1x1 png 过解码,但 pose 引擎未配置(configured=false)→ 统一信封
        let text = post("/pose-rules", multipart_png_with_fields(&[])).await;
        assert!(text.contains("\"status\":\"error\"") || text.contains("未配置"), "{text}");
    }

    #[test]
    fn standing_person_produces_no_events() {
        // 站立:肩(100,80)/(110,80),髋(102,170)/(108,170),窄高 bbox
        let k = kps(&[(L_SHOULDER, 100.0, 80.0), (R_SHOULDER, 110.0, 80.0),
                      (L_HIP, 102.0, 170.0), (R_HIP, 108.0, 170.0)]);
        let ev = evaluate_pose(&k, [90.0, 60.0, 130.0, 200.0], 0);
        assert!(ev.is_empty(), "站立不应有事件: {ev:?}");
    }

    #[test]
    fn lying_person_triggers_fall_with_full_votes() {
        // 躺倒:肩横放(50,100)/(150,100),髋(200,105)/(300,105),宽扁 bbox
        let k = kps(&[(L_SHOULDER, 50.0, 100.0), (R_SHOULDER, 150.0, 100.0),
                      (L_HIP, 200.0, 105.0), (R_HIP, 300.0, 105.0)]);
        let ev = evaluate_pose(&k, [40.0, 80.0, 320.0, 140.0], 0);
        assert_eq!(rules_of(&ev), vec!["fall"]);
        assert!((ev[0].confidence - 1.0).abs() < 1e-9, "三票全中应 1.0: {}", ev[0].confidence);
        assert!(ev[0].evidence.contains("torso_angle"), "证据应含躯干角度: {}", ev[0].evidence);
    }

    #[test]
    fn bent_person_single_vote_no_fall() {
        // 弯腰 60°:躯干倾斜但窄高 bbox、躯干纵向占比仍高——仅 1 票,不触发
        let k = kps(&[(L_SHOULDER, 100.0, 80.0), (R_SHOULDER, 110.0, 80.0),
                      (L_HIP, 152.0, 148.0), (R_HIP, 158.0, 158.0)]);
        let ev = evaluate_pose(&k, [88.0, 60.0, 175.0, 210.0], 0);
        assert!(ev.is_empty(), "单票不应触发 fall: {ev:?}");
    }

    #[test]
    fn low_quality_keypoints_skip_instead_of_silent_pass() {
        // 肩/髋 4 点中 3 点 score=0.1 → skip_low_quality(不静默当无异常)
        let mut k = kps(&[(L_SHOULDER, 100.0, 80.0), (R_SHOULDER, 110.0, 80.0),
                          (L_HIP, 102.0, 170.0), (R_HIP, 108.0, 170.0)]);
        k[R_SHOULDER].2 = 0.1;
        k[L_HIP].2 = 0.1;
        k[R_HIP].2 = 0.1;
        let ev = evaluate_pose(&k, [90.0, 60.0, 130.0, 200.0], 2);
        assert_eq!(rules_of(&ev), vec!["skip_low_quality"]);
        assert_eq!(ev[0].person, 2, "person 序号应透传");
    }

    // ---- raise_hand ----

    #[test]
    fn raised_wrist_triggers_raise_hand() {
        // 左腕(95,40) 高于左肩(100,80),bbox 高 140 → 举手
        let k = kps(&[(L_SHOULDER, 100.0, 80.0), (R_SHOULDER, 110.0, 80.0),
                      (L_WRIST, 95.0, 40.0),
                      (L_HIP, 102.0, 170.0), (R_HIP, 108.0, 170.0)]);
        let ev = evaluate_pose(&k, [90.0, 30.0, 130.0, 200.0], 0);
        assert_eq!(rules_of(&ev), vec!["raise_hand"]);
        assert!(ev[0].confidence > 0.0);
    }

    #[test]
    fn low_score_wrist_does_not_trigger_raise_hand() {
        let mut k = kps(&[(L_SHOULDER, 100.0, 80.0), (R_SHOULDER, 110.0, 80.0),
                          (L_WRIST, 95.0, 40.0),
                          (L_HIP, 102.0, 170.0), (R_HIP, 108.0, 170.0)]);
        k[L_WRIST].2 = 0.2; // 低于 KP_MIN_SCORE
        let ev = evaluate_pose(&k, [90.0, 30.0, 130.0, 200.0], 0);
        assert!(ev.is_empty(), "低分腕点不应判举手: {ev:?}");
    }

    #[test]
    fn fall_and_raise_hand_can_coexist() {
        // 躺倒 + 一腕高于肩(倒地挣扎姿势)——两事件并存
        let k = kps(&[(L_SHOULDER, 50.0, 100.0), (R_SHOULDER, 150.0, 100.0),
                      (L_WRIST, 30.0, 60.0),
                      (L_HIP, 200.0, 105.0), (R_HIP, 300.0, 105.0)]);
        let ev = evaluate_pose(&k, [20.0, 40.0, 320.0, 140.0], 0);
        let rules = rules_of(&ev);
        assert!(rules.contains(&"fall") && rules.contains(&"raise_hand"), "{rules:?}");
    }
}
