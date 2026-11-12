//! 进程内指标导出(D3,obs-platform 05-metrics-spec §4.5)。
//!
//! 设计约束:
//! - **开关语义(R3.5)**:`OTEL_EXPORTER_OTLP_ENDPOINT` 空 = 不初始化导出器,
//!   全部记录方法 no-op,行为与改造前一致(与 safebot-go otelx 同语义);
//! - **计数同源**:背压/槽指标读既有原子状态(`InferenceMetrics`、`SlotRegistry`
//!   的 reload 计数/Slot.build_errors)只读映射,不开第二份计数;
//! - **fanout 红线(08 §2)**:OTLP/HTTP protobuf、Sum/Histogram 默认 cumulative、
//!   直方图显式桶 + unit 恒定(桶布局见 05 §3,违者 fanout 整批 400);
//! - 资源属性按 05 §2:service.name=argus、instance=argus@<port>、
//!   service.version=CARGO_PKG_VERSION **恒定**(version 参与序列身份,不用 git hash)。
//!
//! 本模块是 argus 唯一 import opentelemetry 的地方;域代码只经 `OtelMetrics`
//! 的方法与 `AppState` 交互(disabled 时零开销)。

use std::sync::Arc;
use std::time::Duration;

use opentelemetry::metrics::{AsyncInstrument, Counter, Histogram, MeterProvider};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{MetricExporter, Protocol, WithExportConfig};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::Resource;

use crate::engine::EngineManager;
use crate::routes::common::InferenceMetrics;

/// 推理/检测池直方图桶(05 §3 预定义集合,变更须按新序列处理并登记)。
const INFERENCE_BUCKET_BOUNDS: [f64; 10] = [
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];
/// HTTP 入站直方图桶(05 §3)。
const HTTP_BUCKET_BOUNDS: [f64; 11] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];
/// SDK periodic reader 周期(05 §1:30s)。
const EXPORT_INTERVAL: Duration = Duration::from_secs(30);

/// 同步仪器集(disabled 时不存在)。
struct Instruments {
    inference_requests: Counter<u64>,
    inference_duration: Histogram<f64>,
    inference_queue: Histogram<f64>,
    inference_errors: Counter<u64>,
    http_requests: Counter<u64>,
    http_duration: Histogram<f64>,
}

/// 进程内指标句柄:Clone 廉价(instrument/provider 内部是 Arc);disabled 时全 no-op。
#[derive(Clone, Default)]
pub struct OtelMetrics {
    inner: Option<Arc<Instruments>>,
    provider: Option<SdkMeterProvider>,
}

impl OtelMetrics {
    /// 关闭态:所有记录方法零开销(R3.5 回滚 = env 置空 + 重启)。
    pub fn disabled() -> Self {
        Self::default()
    }

    /// 初始化真实导出链:OTLP/HTTP protobuf → `<endpoint>/v1/metrics`(collector :4318)。
    /// 导出器构建失败(配置级错误)降级为关闭并落日志,不阻断启动。
    pub fn init(
        endpoint: &str,
        port: u16,
        metrics: Arc<InferenceMetrics>,
        engines: Arc<EngineManager>,
    ) -> Self {
        // 踩坑(opentelemetry-otlp 0.33):程序化 with_endpoint 是**整段 URI**,
        // 不自动拼信号路径(仅 OTEL_EXPORTER_OTLP_ENDPOINT 环境变量路径才拼
        // "/v1/metrics",exporter/http/mod.rs resolve_http_endpoint)——只传基址
        // 会 POST 到 "/" 被 collector 404。此处显式补全信号路径。
        let url = if endpoint.ends_with("/v1/metrics") {
            endpoint.to_string()
        } else {
            format!("{}/v1/metrics", endpoint.trim_end_matches('/'))
        };
        let exporter = match MetricExporter::builder()
            .with_http()
            .with_endpoint(url)
            .with_protocol(Protocol::HttpBinary)
            .with_timeout(Duration::from_secs(10))
            .build()
        {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("OTLP 指标导出器构建失败,指标导出关闭: {e}");
                return Self::disabled();
            }
        };
        let reader = PeriodicReader::builder(exporter)
            .with_interval(EXPORT_INTERVAL)
            .build();
        let otel = Self::init_with_reader(reader, port, metrics, engines);
        tracing::info!("OTLP 指标导出已启用: endpoint={endpoint} instance=argus@{port}");
        otel
    }

    /// 停机终期冲刷(periodic reader shutdown = 最终一次 collect+export)。
    /// 放在 axum drain 之后调用,与 safebot-go 的"OTel flush 收尾"语义一致。
    pub fn shutdown(&self) {
        if let Some(provider) = &self.provider {
            if let Err(e) = provider.shutdown() {
                tracing::warn!("指标终期冲刷失败: {e}");
            }
        }
    }

    /// 推理成功完成点(19 处,与完成日志同源取数):量 + 延迟 + 排队等待。
    pub fn observe_inference(
        &self,
        endpoint: &str,
        model: &str,
        queue_ms: u128,
        inference_ms: u128,
    ) {
        if let Some(i) = &self.inner {
            let attrs = [
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("model", model.to_string()),
            ];
            i.inference_requests.add(1, &attrs);
            i.inference_duration.record(inference_ms as f64, &attrs);
            i.inference_queue
                .record(queue_ms as f64, &[attrs[0].clone()]);
        }
    }

    /// 批量端点(detect_batch/face_recognize_batch/quality)口径:整批 elapsed,
    /// 逐图 permit 在内部、无单一排队值,不记 queue_ms(05 §4.5 batch 注记)。
    pub fn observe_inference_batch(&self, endpoint: &str, model: &str, elapsed_ms: u128) {
        if let Some(i) = &self.inner {
            let attrs = [
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("model", model.to_string()),
            ];
            i.inference_requests.add(1, &attrs);
            i.inference_duration.record(elapsed_ms as f64, &attrs);
        }
    }

    /// 推理失败:错误信封单点(into_response 扩展标记)由 HTTP 中间件关联 endpoint 落账。
    pub fn record_inference_error(&self, endpoint: &str, error_kind: &str) {
        if let Some(i) = &self.inner {
            i.inference_errors.add(
                1,
                &[
                    KeyValue::new("endpoint", endpoint.to_string()),
                    KeyValue::new("error_kind", error_kind.to_string()),
                ],
            );
        }
    }

    /// HTTP 入站 RED(最外层中间件,MatchedPath 有值才记录;404 不计)。
    pub fn observe_http(&self, endpoint: &str, duration_ms: u128) {
        if let Some(i) = &self.inner {
            let attrs = [KeyValue::new("endpoint", endpoint.to_string())];
            i.http_requests.add(1, &attrs);
            i.http_duration.record(duration_ms as f64, &attrs);
        }
    }

    /// 组装 provider + 仪器 + 回调(真实导出链与单测 InMemory 导出器共用;
    /// 两者都经 PeriodicReader 承载,测试用 force_flush 触发采集)。
    fn init_with_reader(
        reader: PeriodicReader<impl opentelemetry_sdk::metrics::exporter::PushMetricExporter>,
        port: u16,
        metrics: Arc<InferenceMetrics>,
        engines: Arc<EngineManager>,
    ) -> Self {
        let resource = Resource::builder()
            .with_service_name("argus")
            .with_attributes([
                KeyValue::new("service.namespace", "safebot"),
                KeyValue::new("service.instance.id", format!("argus@{port}")),
                // 恒定不用 git hash:version 参与 fanout 序列身份(05 §4.5 注记)
                KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                KeyValue::new("deployment.environment.name", "local-prod"),
            ])
            .build();
        let provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(reader)
            .build();
        let meter = provider.meter("argus/otel_metrics");

        let instruments = Instruments {
            inference_requests: meter.u64_counter("argus.inference.requests_total").build(),
            inference_duration: meter
                .f64_histogram("argus.inference.duration_ms")
                .with_unit("ms")
                .with_boundaries(INFERENCE_BUCKET_BOUNDS.to_vec())
                .build(),
            inference_queue: meter
                .f64_histogram("argus.inference.queue_ms")
                .with_unit("ms")
                .with_boundaries(INFERENCE_BUCKET_BOUNDS.to_vec())
                .build(),
            inference_errors: meter.u64_counter("argus.inference.errors_total").build(),
            http_requests: meter
                .u64_counter("argus.http.server.request.requests_total")
                .build(),
            http_duration: meter
                .f64_histogram("argus.http.server.request.duration_ms")
                .with_unit("ms")
                .with_boundaries(HTTP_BUCKET_BOUNDS.to_vec())
                .build(),
        };

        // ---- ObservableGauge/Counter:只读映射既有原子状态(计数同源) ----

        let in_flight = Arc::clone(&metrics);
        let _ = meter
            .u64_observable_gauge("argus.inference.in_flight")
            .with_callback(move |g: &dyn AsyncInstrument<u64>| {
                g.observe(
                    in_flight
                        .in_flight
                        .load(std::sync::atomic::Ordering::Relaxed) as u64,
                    &[],
                );
            })
            .build();

        // rejected{reason}:两个拒绝分支与既有原子同点累加,这里按不变量拆分——
        // acquire_timeout_total ⊆ rejected_total(排队超时分支两者同增;fail-fast
        // 模式 acquire_timeout 恒 0),故 fail_fast = rejected_total - acquire_timeout_total。
        let rejected = Arc::clone(&metrics);
        let _ = meter
            .u64_observable_counter("argus.backpressure.rejected_total")
            .with_callback(move |c: &dyn AsyncInstrument<u64>| {
                use std::sync::atomic::Ordering::Relaxed;
                let total = rejected.rejected_total.load(Relaxed);
                let timeout = rejected.acquire_timeout_total.load(Relaxed);
                c.observe(
                    total.saturating_sub(timeout),
                    &[KeyValue::new("reason", "fail_fast")],
                );
                c.observe(timeout, &[KeyValue::new("reason", "queue_timeout")]);
            })
            .build();

        let slots_reg = Arc::clone(&engines);
        let _ = meter
            .u64_observable_counter("argus.slots.reload_total")
            .with_callback(move |c: &dyn AsyncInstrument<u64>| {
                use std::sync::atomic::Ordering::Relaxed;
                let s = &slots_reg.slots.reload;
                c.observe(
                    s.outcome_ok.load(Relaxed),
                    &[KeyValue::new("outcome", "ok")],
                );
                c.observe(
                    s.outcome_parse_error.load(Relaxed),
                    &[KeyValue::new("outcome", "parse_error")],
                );
                c.observe(
                    s.outcome_removed.load(Relaxed),
                    &[KeyValue::new("outcome", "removed")],
                );
                c.observe(
                    s.outcome_read_error.load(Relaxed),
                    &[KeyValue::new("outcome", "read_error")],
                );
            })
            .build();

        let slots_loaded = Arc::clone(&engines);
        let _ = meter
            .u64_observable_gauge("argus.slots.loaded")
            .with_callback(move |g: &dyn AsyncInstrument<u64>| {
                for slot in slots_loaded.slots.slots_snapshot() {
                    g.observe(
                        u64::from(slot.loaded()),
                        &[KeyValue::new("slot", slot.spec.name.clone())],
                    );
                }
            })
            .build();

        let slots_broken = Arc::clone(&engines);
        let _ = meter
            .u64_observable_counter("argus.slots.build_errors_total")
            .with_callback(move |c: &dyn AsyncInstrument<u64>| {
                use std::sync::atomic::Ordering::Relaxed;
                for slot in slots_broken.slots.slots_snapshot() {
                    c.observe(
                        slot.build_errors.load(Relaxed),
                        &[KeyValue::new("slot", slot.spec.name.clone())],
                    );
                }
            })
            .build();

        let slots_last_ok = Arc::clone(&engines);
        let _ = meter
            .f64_observable_gauge("argus.slots.last_ok_age_s")
            .with_callback(move |g: &dyn AsyncInstrument<f64>| {
                let last_ok = slots_last_ok
                    .slots
                    .reload
                    .last_ok
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(t) = *last_ok {
                    g.observe(t.elapsed().as_secs_f64(), &[]);
                }
            })
            .build();

        Self {
            inner: Some(Arc::new(instruments)),
            provider: Some(provider),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_sdk::metrics::data::{
        AggregatedMetrics, Metric, MetricData, ResourceMetrics,
    };
    use opentelemetry_sdk::metrics::InMemoryMetricExporter;

    /// 用内存导出器构建(单测断言用,07-tdd §3.2 的 ManualReader 等价物;
    /// InMemory 也经 PeriodicReader 承载,force_flush 触发采集)。
    fn init_test() -> (OtelMetrics, InMemoryMetricExporter) {
        let config = crate::config::Config::default();
        let engines = Arc::new(crate::engine::EngineManager::new(config));
        let metrics = Arc::new(InferenceMetrics::default());
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let otel = OtelMetrics::init_with_reader(reader, 7042, metrics, engines);
        (otel, exporter)
    }

    fn flush(otel: &OtelMetrics, exporter: &InMemoryMetricExporter) -> Vec<ResourceMetrics> {
        otel.provider.as_ref().unwrap().force_flush().unwrap();
        exporter.get_finished_metrics().unwrap()
    }

    fn find_metric<'a>(finished: &'a [ResourceMetrics], name: &str) -> &'a Metric {
        let mut existing = Vec::new();
        for rm in finished {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == name {
                        return m;
                    }
                    existing.push(m.name().to_string());
                }
            }
        }
        panic!("未找到指标 {name},现有: {existing:?}");
    }

    fn attr_value<'a>(mut attrs: impl Iterator<Item = &'a KeyValue>, key: &str) -> Option<String> {
        attrs
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.as_str().to_string())
    }

    /// 断言 metric 是 u64 Sum,返回其数据点列表 (attrs, value)。
    fn sum_points(m: &Metric) -> Vec<(Vec<KeyValue>, u64)> {
        let AggregatedMetrics::U64(data) = m.data() else {
            panic!("{} 应为 u64 聚合", m.name());
        };
        let MetricData::Sum(sum) = data else {
            panic!("{} 应为 Sum", m.name());
        };
        sum.data_points()
            .map(|dp| {
                let attrs: Vec<KeyValue> = dp.attributes().cloned().collect();
                (attrs, dp.value())
            })
            .collect()
    }

    /// 直方图数据点摘要:(attrs, count, bounds, bucket_counts)。
    type HistPoint = (Vec<KeyValue>, u64, Vec<f64>, Vec<u64>);

    /// 断言 metric 是 f64 Histogram,返回其数据点 (attrs, count, bounds, bucket_counts)。
    fn histogram_points(m: &Metric) -> Vec<HistPoint> {
        let AggregatedMetrics::F64(data) = m.data() else {
            panic!("{} 应为 f64 聚合", m.name());
        };
        let MetricData::Histogram(h) = data else {
            panic!("{} 应为 Histogram", m.name());
        };
        h.data_points()
            .map(|dp| {
                let attrs: Vec<KeyValue> = dp.attributes().cloned().collect();
                (
                    attrs,
                    dp.count(),
                    dp.bounds().collect(),
                    dp.bucket_counts().collect(),
                )
            })
            .collect()
    }

    #[test]
    fn disabled_should_be_silent_noop() {
        let otel = OtelMetrics::disabled();
        // 全方法可安全调用且不 panic(无导出器可查)
        otel.observe_inference("detect", "m#abc", 0, 5);
        otel.observe_inference_batch("detect_batch", "m#abc", 10);
        otel.record_inference_error("detect", "overload");
        otel.observe_http("/detect", 7);
        otel.shutdown();
    }

    #[test]
    fn observe_inference_should_record_requests_duration_queue() {
        let (otel, exporter) = init_test();
        otel.observe_inference("detect", "yolo.onnx#abc123def456", 12, 30);
        let finished = flush(&otel, &exporter);

        let req = find_metric(&finished, "argus.inference.requests_total");
        let pts = sum_points(req);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].1, 1);
        assert_eq!(
            attr_value(pts[0].0.iter(), "endpoint").as_deref(),
            Some("detect")
        );
        assert_eq!(
            attr_value(pts[0].0.iter(), "model").as_deref(),
            Some("yolo.onnx#abc123def456")
        );

        let hd = find_metric(&finished, "argus.inference.duration_ms");
        assert_eq!(hd.unit(), "ms", "unit 恒定参与 fanout 序列身份");
        let hpts = histogram_points(hd);
        assert_eq!(hpts.len(), 1);
        assert_eq!(hpts[0].1, 1);
        assert_eq!(hpts[0].2, INFERENCE_BUCKET_BOUNDS.to_vec());
        // 30ms 落 [25,50) 桶 = 第 3 个 bucket_counts 分量(0..10/10..25/25..50)
        assert_eq!(hpts[0].3[2], 1, "30ms 应落 [25,50) 桶: {:?}", hpts[0].3);

        // queue_ms 只带 endpoint 维度
        let q = find_metric(&finished, "argus.inference.queue_ms");
        let qpts = histogram_points(q);
        assert_eq!(qpts.len(), 1);
        assert_eq!(qpts[0].1, 1);
        assert_eq!(
            attr_value(qpts[0].0.iter(), "model"),
            None,
            "queue_ms 不带 model 维度"
        );
    }

    #[test]
    fn record_inference_error_should_carry_kind() {
        let (otel, exporter) = init_test();
        otel.record_inference_error("ocr", "inference_failed");
        let finished = flush(&otel, &exporter);

        let pts = sum_points(find_metric(&finished, "argus.inference.errors_total"));
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].1, 1);
        assert_eq!(
            attr_value(pts[0].0.iter(), "error_kind").as_deref(),
            Some("inference_failed")
        );
        assert_eq!(
            attr_value(pts[0].0.iter(), "endpoint").as_deref(),
            Some("ocr")
        );
    }

    #[test]
    fn rejected_should_split_by_reason_from_existing_atomics() {
        let config = crate::config::Config::default();
        let engines = Arc::new(crate::engine::EngineManager::new(config));
        let metrics = Arc::new(InferenceMetrics::default());
        // 模拟既有原子:2 次 fail-fast + 1 次排队超时(超时分支两计数同增)
        use std::sync::atomic::Ordering::Relaxed;
        metrics.rejected_total.store(3, Relaxed);
        metrics.acquire_timeout_total.store(1, Relaxed);
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let otel = OtelMetrics::init_with_reader(reader, 7042, metrics, engines);
        let finished = flush(&otel, &exporter);

        let pts = sum_points(find_metric(&finished, "argus.backpressure.rejected_total"));
        let by_reason: std::collections::HashMap<String, u64> = pts
            .iter()
            .map(|(attrs, v)| (attr_value(attrs.iter(), "reason").unwrap(), *v))
            .collect();
        assert_eq!(
            by_reason.get("fail_fast"),
            Some(&2),
            "fail_fast = rejected_total - acquire_timeout_total: {by_reason:?}"
        );
        assert_eq!(by_reason.get("queue_timeout"), Some(&1));
    }

    #[test]
    fn resource_should_carry_stable_identity_and_http_red() {
        let (otel, exporter) = init_test();
        otel.observe_http("/detect", 3);
        let finished = flush(&otel, &exporter);

        let rm = finished.first().expect("应有资源指标");
        let res = rm.resource();
        let get = |k: &str| {
            res.get(&opentelemetry::Key::new(k.to_string()))
                .map(|v| v.to_string())
                .unwrap_or_default()
        };
        assert_eq!(get("service.name"), "argus");
        assert_eq!(get("service.instance.id"), "argus@7042");
        assert_eq!(get("service.version"), env!("CARGO_PKG_VERSION"));
        assert_eq!(get("deployment.environment.name"), "local-prod");

        let pts = sum_points(find_metric(
            &finished,
            "argus.http.server.request.requests_total",
        ));
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].1, 1);
        assert_eq!(
            attr_value(pts[0].0.iter(), "endpoint").as_deref(),
            Some("/detect")
        );
    }

    #[test]
    fn reload_and_slot_gauges_should_map_registry_state() {
        let config = crate::config::Config::default();
        let engines = Arc::new(crate::engine::EngineManager::new(config));
        use std::sync::atomic::Ordering::Relaxed;
        engines.slots.reload.outcome_ok.store(2, Relaxed);
        engines.slots.reload.outcome_parse_error.store(1, Relaxed);
        let metrics = Arc::new(InferenceMetrics::default());
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let otel = OtelMetrics::init_with_reader(reader, 7042, metrics, engines);
        let finished = flush(&otel, &exporter);

        let pts = sum_points(find_metric(&finished, "argus.slots.reload_total"));
        let by_outcome: std::collections::HashMap<String, u64> = pts
            .iter()
            .map(|(attrs, v)| (attr_value(attrs.iter(), "outcome").unwrap(), *v))
            .collect();
        assert_eq!(by_outcome.get("ok"), Some(&2));
        assert_eq!(by_outcome.get("parse_error"), Some(&1));
        assert_eq!(
            by_outcome.get("removed"),
            Some(&0),
            "四结局恒观测(0 也上报,series 身份稳定)"
        );

        let age = find_metric(&finished, "argus.slots.last_ok_age_s");
        let AggregatedMetrics::F64(data) = age.data() else {
            panic!("应为 f64");
        };
        let MetricData::Gauge(g) = data else {
            panic!("应为 Gauge");
        };
        let dp = g.data_points().next().expect("last_ok 初始化即有值");
        assert!(dp.value() >= 0.0);
    }

    /// 带内存指标的真实 router(中间件经 AppState.otel 记录)。
    async fn router_with_metrics() -> (OtelMetrics, InMemoryMetricExporter, axum::Router) {
        let config = crate::config::Config::default();
        let engines = Arc::new(crate::engine::EngineManager::new(config));
        let metrics = Arc::new(InferenceMetrics::default());
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let otel = OtelMetrics::init_with_reader(reader, 7042, metrics, engines);
        let mut state = crate::routes::health::AppState::new(
            crate::config::Config::default(),
            crate::engine::EngineManager::new(crate::config::Config::default()),
        );
        state.otel = otel.clone();
        let router = crate::routes::health::build_router(state, None);
        (otel, exporter, router)
    }

    #[tokio::test]
    async fn http_middleware_should_record_red_and_join_error_marker() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let (otel, exporter, app) = router_with_metrics().await;

        // 成功路径:命中路由(无 auth_state,鉴权层直通)
        let resp = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // 失败路径:/detect/batch 非法 JSON → AppError 信封(恒 200)+ ErrorMetric 扩展
        let resp = app
            .clone()
            .oneshot(
                Request::post("/detect/batch")
                    .header("content-type", "application/json")
                    .body(Body::from("not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "argus 错误信封恒 200");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("\"error\""), "应回错误信封: {text}");

        // 404:未命中路由无 MatchedPath,不应计入
        let resp = app
            .oneshot(Request::get("/__nope__").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        let finished = flush(&otel, &exporter);

        let pts = sum_points(find_metric(
            &finished,
            "argus.http.server.request.requests_total",
        ));
        let endpoints: Vec<String> = pts
            .iter()
            .map(|(attrs, _)| attr_value(attrs.iter(), "endpoint").unwrap())
            .collect();
        assert!(endpoints.contains(&"/health".to_string()), "{endpoints:?}");
        assert!(
            endpoints.contains(&"/detect/batch".to_string()),
            "{endpoints:?}"
        );
        assert!(
            !endpoints.contains(&"/__nope__".to_string()),
            "404 不计: {endpoints:?}"
        );

        // 错误标记关联:错误信封经最外层中间件落 endpoint × error_kind
        let err_pts = sum_points(find_metric(&finished, "argus.inference.errors_total"));
        assert_eq!(err_pts.len(), 1, "仅 /detect/batch 一处错误: {err_pts:?}");
        assert_eq!(
            attr_value(err_pts[0].0.iter(), "endpoint").as_deref(),
            Some("/detect/batch")
        );
        assert_eq!(
            attr_value(err_pts[0].0.iter(), "error_kind").as_deref(),
            Some("bad_request")
        );

        let hd = histogram_points(find_metric(
            &finished,
            "argus.http.server.request.duration_ms",
        ));
        assert_eq!(hd.len(), 2, "/health 与 /detect/batch 各一点: {hd:?}");
        assert_eq!(hd[0].2, HTTP_BUCKET_BOUNDS.to_vec(), "HTTP 入站桶(05 §3)");
    }
}
