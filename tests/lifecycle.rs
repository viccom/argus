//! 服务生命周期集成测试:进程在无停机信号时不得自行退出。
//!
//! 回归背景:main 曾用 `timeout(shutdown_timeout, server)` 包住整个 server,
//! 超时从**进程启动**起算——本意"停机后给 N 秒收尾"被写成"进程总寿命 N 秒",
//! 服务每 N 秒无信号自杀(exit 0),长期被误判为"被外部周期性终止"。

use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn server_should_survive_beyond_shutdown_timeout_without_signal() {
    // 端口随机避并行冲突;无模型配置即不加载引擎,/health 仍可用
    let port = 32000 + (std::process::id() % 10000) as u16;
    let mut child = Command::new(env!("CARGO_BIN_EXE_vision-ext"))
        .env("VISION_EXT_PORT", port.to_string())
        // 缩短超时基数:旧 bug 下 3s 即自杀,等待 2 倍(6s)足以区分存活/自退
        .env("VISION_EXT_SHUTDOWN_TIMEOUT_SECS", "3")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 vision-ext 测试实例");

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
