//! 检测结果的请求级后过滤：`confidence` / `target_classes` / `roi`。
//!
//! 引擎以 [`DETECTION_CONFIDENCE_FLOOR`] 低阈值构造（`Arc` 共享不可变，无法按请求重建），
//! 每请求阈值在 HTTP 层后过滤：类内 NMS 保序，低阈值结果是任意高阈值结果的超集，
//! 后过滤与"引擎直接以该阈值构造"严格等价。SAHI 路径由 `SahiConfig::min_confidence`
//! 在合并前过滤（合并后再滤会让低分碎片先放大高分框几何）。
//!
//! `/detect`（multipart）与 `/detect/batch`（JSON）共用同一套语义。

use rust_onnx_infer::model::Detection;

use crate::engine::DETECTION_CONFIDENCE_FLOOR;
use crate::error::AppError;

/// 请求未传 `confidence` 时的默认值——与引入后过滤前的引擎构造阈值一致，默认行为零变化。
pub const DEFAULT_CONFIDENCE: f64 = 0.25;

/// 归一化多边形（坐标 ∈ [0,1]，相对图像宽高），≥3 点。
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon(Vec<(f64, f64)>);

impl Polygon {
    /// 从点列构造并校验：≥3 点、坐标有限且在 [0,1]。
    pub fn from_points(points: Vec<[f64; 2]>) -> Result<Self, AppError> {
        if points.len() < 3 {
            return Err(AppError::BadRequest(format!(
                "roi 至少需要 3 个点，实际 {}",
                points.len()
            )));
        }
        for (i, [x, y]) in points.iter().enumerate() {
            let in_unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
            if !in_unit(*x) || !in_unit(*y) {
                return Err(AppError::BadRequest(format!(
                    "roi 第 {i} 点 [{x}, {y}] 越界：坐标须为归一化值 [0,1]"
                )));
            }
        }
        Ok(Polygon(points.into_iter().map(|[x, y]| (x, y)).collect()))
    }

    /// 解析 JSON 文本 `[[x,y],[x,y],...]`（multipart 文本字段用）。
    pub fn parse(text: &str) -> Result<Self, AppError> {
        let points: Vec<[f64; 2]> = serde_json::from_str(text).map_err(|e| {
            AppError::BadRequest(format!(
                "roi 须为 JSON 二维点数组 [[x,y],...]（归一化坐标）：{e}"
            ))
        })?;
        Self::from_points(points)
    }

    /// 射线法判定点是否在多边形内（边界上的点归属由射线奇偶性决定，不作特殊处理）。
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let pts = &self.0;
        let n = pts.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = pts[i];
            let (xj, yj) = pts[j];
            // (yi > y) != (yj > y) 保证 yi != yj，除法安全
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

/// 解析 `target_classes` JSON 文本 `[0, 2, 5]`（class_id 数组，非空）。
pub fn parse_target_classes(text: &str) -> Result<Vec<i32>, AppError> {
    let ids: Vec<i32> = serde_json::from_str(text).map_err(|e| {
        AppError::BadRequest(format!("target_classes 须为 JSON 整数数组 [0,1,2]：{e}"))
    })?;
    validate_target_classes(&ids)?;
    Ok(ids)
}

fn validate_target_classes(ids: &[i32]) -> Result<(), AppError> {
    if ids.is_empty() {
        return Err(AppError::BadRequest(
            "target_classes 不能为空数组（不过滤请省略该字段）".into(),
        ));
    }
    Ok(())
}

/// 请求级过滤参数（已校验）。
#[derive(Debug, Clone)]
pub struct DetectFilter {
    /// 有效置信度阈值，∈ [DETECTION_CONFIDENCE_FLOOR, 1.0]
    pub confidence: f64,
    pub target_classes: Option<Vec<i32>>,
    pub roi: Option<Polygon>,
}

impl DetectFilter {
    /// 构造并校验：`confidence` 缺省 [`DEFAULT_CONFIDENCE`]，须 ∈ [floor, 1.0]；
    /// `target_classes` 非空。`roi` 在 [`Polygon`] 构造时已校验。
    pub fn new(
        confidence: Option<f64>,
        target_classes: Option<Vec<i32>>,
        roi: Option<Polygon>,
    ) -> Result<Self, AppError> {
        let floor = DETECTION_CONFIDENCE_FLOOR;
        let confidence = confidence.unwrap_or(DEFAULT_CONFIDENCE);
        // 下限比较在 f32 精度下进行:floor 提升到 f64 会变成 0.10000000149…,
        // 用户输入的字面量 0.1 反而被判越界(引擎本身就是 f32 阈值)
        if !confidence.is_finite() || (confidence as f32) < floor || confidence > 1.0 {
            return Err(AppError::BadRequest(format!(
                "confidence 须在 [{floor}, 1.0] 内（引擎构造阈值 {floor} 为下限），实际 {confidence}"
            )));
        }
        if let Some(ids) = &target_classes {
            validate_target_classes(ids)?;
        }
        Ok(DetectFilter {
            confidence,
            target_classes,
            roi,
        })
    }

    /// 应用过滤。`roi` 以 bbox 中心点归一化坐标判定（与 yolo-rs 同款语义）。
    pub fn apply(&self, mut dets: Vec<Detection>, img_w: usize, img_h: usize) -> Vec<Detection> {
        let (w, h) = (img_w.max(1) as f64, img_h.max(1) as f64);
        dets.retain(|d| {
            if d.confidence() < self.confidence {
                return false;
            }
            if let Some(ids) = &self.target_classes {
                if !ids.contains(&d.class_id) {
                    return false;
                }
            }
            if let Some(roi) = &self.roi {
                let cx = (d.x1() + d.x2()) / 2.0 / w;
                let cy = (d.y1() + d.y2()) / 2.0 / h;
                if !roi.contains(cx, cy) {
                    return false;
                }
            }
            true
        });
        dets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(class_id: i32, x1: f64, y1: f64, x2: f64, y2: f64, conf: f64) -> Detection {
        Detection::new("c", class_id, x1, y1, x2, y2, conf)
    }

    fn unit_square() -> Polygon {
        Polygon::from_points(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).unwrap()
    }

    // ---- Polygon ----

    #[test]
    fn polygon_parse_should_accept_json_pairs() {
        let p = Polygon::parse("[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]").unwrap();
        assert_eq!(p.0.len(), 4);
        assert_eq!(p.0[2], (0.9, 0.9));
    }

    #[test]
    fn polygon_should_reject_fewer_than_three_points() {
        let err = Polygon::parse("[[0,0],[1,1]]").unwrap_err();
        assert!(err.to_string().contains("3 个点"), "{err}");
    }

    #[test]
    fn polygon_should_reject_out_of_unit_coords() {
        let err = Polygon::parse("[[0,0],[1.5,0],[1,1]]").unwrap_err();
        assert!(err.to_string().contains("越界"), "{err}");
        let err = Polygon::parse("[[0,0],[-0.1,0],[1,1]]").unwrap_err();
        assert!(err.to_string().contains("越界"), "{err}");
    }

    #[test]
    fn polygon_should_reject_non_json_and_wrong_shape() {
        assert!(Polygon::parse("0.1,0.1,0.9,0.9").is_err());
        assert!(Polygon::parse("[[0.1,0.1,0.2],[0.9,0.1],[0.9,0.9]]").is_err());
        assert!(Polygon::parse("{\"polygon\":[]}").is_err());
    }

    #[test]
    fn polygon_contains_should_use_ray_casting() {
        let sq = Polygon::parse("[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]").unwrap();
        assert!(sq.contains(0.5, 0.5));
        assert!(!sq.contains(0.1, 0.5));
        assert!(!sq.contains(0.5, 0.9));
        assert!(!sq.contains(0.9, 0.9));
    }

    #[test]
    fn polygon_contains_should_handle_concave_shape() {
        // L 形：右上角缺口 (0.5..1, 0..0.5) 不在多边形内
        let l = Polygon::parse("[[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1]]").unwrap();
        assert!(l.contains(0.25, 0.25), "左上实体块");
        assert!(l.contains(0.75, 0.75), "右下实体块");
        assert!(!l.contains(0.75, 0.25), "右上缺口应在外");
    }

    // ---- target_classes ----

    #[test]
    fn target_classes_should_parse_json_ints() {
        assert_eq!(parse_target_classes("[0, 2, 5]").unwrap(), vec![0, 2, 5]);
    }

    #[test]
    fn target_classes_should_reject_empty_and_malformed() {
        assert!(parse_target_classes("[]").unwrap_err().to_string().contains("不能为空"));
        assert!(parse_target_classes("0,2,5").is_err());
        assert!(parse_target_classes("[\"person\"]").is_err());
    }

    // ---- DetectFilter::new ----

    #[test]
    fn filter_default_confidence_should_match_legacy_engine_threshold() {
        // 未传 confidence 时必须等于旧引擎构造阈值 0.25，否则默认 /detect 行为漂移
        let f = DetectFilter::new(None, None, None).unwrap();
        assert_eq!(f.confidence, 0.25);
        assert_eq!(DEFAULT_CONFIDENCE, 0.25);
    }

    #[test]
    fn filter_should_reject_confidence_below_engine_floor() {
        let err = DetectFilter::new(Some(0.05), None, None).unwrap_err();
        assert!(err.to_string().contains("0.1"), "应提示下限: {err}");
    }

    #[test]
    fn filter_should_reject_confidence_above_one_or_nan() {
        assert!(DetectFilter::new(Some(1.5), None, None).is_err());
        assert!(DetectFilter::new(Some(f64::NAN), None, None).is_err());
    }

    #[test]
    fn filter_should_accept_floor_and_one_inclusive() {
        assert!(DetectFilter::new(Some(DETECTION_CONFIDENCE_FLOOR as f64), None, None).is_ok());
        assert!(DetectFilter::new(Some(1.0), None, None).is_ok());
    }

    #[test]
    fn filter_should_accept_user_literal_at_floor() {
        // E2E 踩坑:floor(f32) as f64 = 0.10000000149…,用户敲的 "0.1"/"0.10" 解析为 f64 0.1
        // 比它小,曾被判越界。比较须在 f32 精度下进行
        let f = DetectFilter::new(Some(0.1), None, None).unwrap();
        assert!((f.confidence - 0.1).abs() < 1e-12);
        let err = DetectFilter::new(Some(0.05), None, None).unwrap_err().to_string();
        assert!(err.contains("[0.1, 1.0]"), "错误信息应显示整洁的 0.1 而非 f32 提升噪声: {err}");
    }

    #[test]
    fn filter_should_reject_empty_target_classes() {
        assert!(DetectFilter::new(None, Some(vec![]), None).is_err());
    }

    // ---- DetectFilter::apply ----

    #[test]
    fn apply_should_drop_below_confidence() {
        let f = DetectFilter::new(Some(0.5), None, None).unwrap();
        let out = f.apply(
            vec![det(0, 0.0, 0.0, 10.0, 10.0, 0.9), det(0, 0.0, 0.0, 10.0, 10.0, 0.49)],
            100,
            100,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].confidence(), 0.9);
    }

    #[test]
    fn apply_should_keep_only_target_classes() {
        let f = DetectFilter::new(None, Some(vec![2, 7]), None).unwrap();
        let out = f.apply(
            vec![det(0, 0.0, 0.0, 10.0, 10.0, 0.9), det(2, 0.0, 0.0, 10.0, 10.0, 0.9), det(7, 0.0, 0.0, 10.0, 10.0, 0.9)],
            100,
            100,
        );
        assert_eq!(out.iter().map(|d| d.class_id).collect::<Vec<_>>(), vec![2, 7]);
    }

    #[test]
    fn apply_should_filter_by_bbox_center_in_roi() {
        // 图 200x100，ROI 左半 [0,0.5]；框 A 中心 (50,50)→(0.25,0.5) 在内；框 B 中心 (150,50)→(0.75,0.5) 在外；
        // 框 C 横跨中线但中心 (110,50)→(0.55,0.5) 在外——按中心而非交集判定
        let roi = Polygon::parse("[[0,0],[0.5,0],[0.5,1],[0,1]]").unwrap();
        let f = DetectFilter::new(None, None, Some(roi)).unwrap();
        let out = f.apply(
            vec![
                det(0, 40.0, 40.0, 60.0, 60.0, 0.9),
                det(1, 140.0, 40.0, 160.0, 60.0, 0.9),
                det(2, 80.0, 40.0, 140.0, 60.0, 0.9),
            ],
            200,
            100,
        );
        assert_eq!(out.iter().map(|d| d.class_id).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn apply_with_unit_square_roi_should_keep_all_inside_image() {
        let f = DetectFilter::new(None, None, Some(unit_square())).unwrap();
        let out = f.apply(vec![det(0, 0.0, 0.0, 10.0, 10.0, 0.9)], 100, 100);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn apply_should_combine_all_three_filters() {
        let roi = Polygon::parse("[[0,0],[0.5,0],[0.5,1],[0,1]]").unwrap();
        let f = DetectFilter::new(Some(0.5), Some(vec![0]), Some(roi)).unwrap();
        let out = f.apply(
            vec![
                det(0, 40.0, 40.0, 60.0, 60.0, 0.9),  // 通过
                det(0, 40.0, 40.0, 60.0, 60.0, 0.3),  // 分低
                det(1, 40.0, 40.0, 60.0, 60.0, 0.9),  // 类别不符
                det(0, 140.0, 40.0, 160.0, 60.0, 0.9), // ROI 外
            ],
            200,
            100,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].confidence(), 0.9);
        assert_eq!(out[0].class_id, 0);
    }

    #[test]
    fn apply_should_not_divide_by_zero_on_degenerate_image() {
        let f = DetectFilter::new(None, None, Some(unit_square())).unwrap();
        let out = f.apply(vec![det(0, 0.0, 0.0, 1.0, 1.0, 0.9)], 0, 0);
        assert_eq!(out.len(), 1);
    }
}
