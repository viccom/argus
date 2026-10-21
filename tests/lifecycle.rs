//! 服务生命周期集成测试:进程在无停机信号时不得自行退出。
//!
//! 回归背景:main 曾用 `timeout(shutdown_timeout, server)` 包住整个 server,
//! 超时从**进程启动**起算——本意"停机后给 N 秒收尾"被写成"进程总寿命 N 秒",
//! 服务每 N 秒无信号自杀(exit 0),长期被误判为"被外部周期性终止"。
//!
//! 同文件另覆盖动态槽 F3 生命周期:slots.toml 非法拒绝启动(fail-closed)、
//! watcher 活跃时 SIGTERM 干净退出。

use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn server_should_survive_beyond_shutdown_timeout_without_signal() {
    // 端口随机避并行冲突;无模型配置即不加载引擎,/health 仍可用
    let port = 32000 + (std::process::id() % 10000) as u16;
    let mut child = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", port.to_string())
        // 缩短超时基数:旧 bug 下 3s 即自杀,等待 2 倍(6s)足以区分存活/自退
        .env("ARGUS_SHUTDOWN_TIMEOUT_SECS", "3")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 argus 测试实例");

    std::thread::sleep(Duration::from_secs(6));
    let exited = child.try_wait().expect("查询进程状态");
    assert!(
        exited.is_none(),
        "服务在无停机信号时自行退出(寿命 bug 回归): {exited:?}"
    );

    // 清理(测试不覆盖优雅停机细节,该路径由 warden 托管 E2E 验证)
    let _ = child.kill();
    let _ = child.wait();
}

// ---- 动态槽生命周期（F3） ----
// 自包含 models_dir（解析不依赖模型文件存在：启动只注册不构建）。
// 不依赖仓库相对路径 ./models——worktree/CI 检出没有该目录。

/// 临时探测端口（OS 从 ephemeral 段分配，bind 即释放；两测试各取不会相撞，
/// 与上方回归测试的 `32000 + pid%10000` 取样区间也不重叠）。
#[cfg(unix)]
fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("探测空闲端口");
    l.local_addr().expect("端口地址").port()
}

/// 自包含临时 models_dir + slots.toml（真实存在于 temp，非仓库相对路径）。
/// m.onnx 只是让 models_dir 非空；启动解析不看它（懒加载，首次请求才构建）。
#[cfg(unix)]
fn slot_env(tag: &str, slots_text: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("argus-slots-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
    let slots = dir.join("slots.toml");
    std::fs::write(&slots, slots_text).unwrap();
    slots
}

/// 非法 slots.toml（槽名大写）必须拒绝启动（fail-closed，设计 §6）。
#[cfg(unix)]
#[test]
fn invalid_slots_toml_should_fail_startup() {
    let slots = slot_env("lc1", "[slots.BadName]\nmodel = \"m.onnx\"\n");
    let dir = slots.parent().unwrap().to_path_buf();
    // 无超时风险:fail-closed 在 TcpListener bind 之前立即 Err 返回，
    // status() 阻塞等待必退出（真挂死则由 cargo test 自身超时兜底）。
    let status = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", free_port().to_string())
        .env("ARGUS_MODELS_DIR", &dir)
        .env("ARGUS_SLOTS_CONFIG", &slots)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("启动 argus 测试实例");
    assert!(!status.success(), "非法 slots.toml 必须拒绝启动: {status:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// watcher 活跃时 SIGTERM：进程须在超时上限内干净退出（runtime 终止后台任务，
/// 无残留；优雅停机链不变——设计风险表 P1 / 验收 9）。
#[cfg(unix)]
#[test]
fn sigterm_with_active_watcher_should_exit_cleanly() {
    let slots = slot_env("lc2", "[slots.a]\nmodel = \"m.onnx\"\n");
    let dir = slots.parent().unwrap().to_path_buf();
    let mut child = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", free_port().to_string())
        .env("ARGUS_MODELS_DIR", &dir)
        .env("ARGUS_SLOTS_CONFIG", &slots)
        .env("ARGUS_SLOTS_WATCH_SECS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 argus 测试实例");
    std::thread::sleep(Duration::from_secs(2)); // watcher 至少跑一轮
    Command::new("kill").arg(child.id().to_string()).status().unwrap(); // SIGTERM

    let deadline = std::time::Instant::now() + Duration::from_secs(15); // > shutdown_timeout 默认 10s
    let mut exited = None;
    while std::time::Instant::now() < deadline {
        match child.try_wait().expect("查询进程状态") {
            Some(status) => {
                exited = Some(status);
                break;
            }
            None => std::thread::sleep(Duration::from_millis(200)),
        }
    }
    let status = match exited {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let reaped = child.wait().expect("回收测试实例");
            panic!("SIGTERM 后 15s 未退出（watcher 阻塞停机）: {reaped:?}");
        }
    };
    assert_eq!(
        status.code(),
        Some(0),
        "优雅停机应 exit 0（被信号杀则 code=None）: {status:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
