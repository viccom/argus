//! 日志格式回归测试(04-log-spec §6.2 A-1/A-2/A-3)。
//!
//! 覆盖三件事:
//!   A-1 stdout 每行必须是 JSON(可被 serde_json::parse),且含 level/target/timestamp 字段;
//!   A-2 探活路径(/health、/)的请求完成日志不得以 INFO 出现(降噪);
//!   A-3 非探活路径(/models)的请求完成日志仍须以 INFO 出现(降噪不得把日志整体掐掉)。
//!
//! 手法:起真实进程(与 04 §6.2 A-3 "启动服务抓 stdout" 一致),读 stdout 逐行断言。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const AUTH_TOKEN: &str = "test-log-format-key";

/// 起一个 argus 实例并后台收集 stdout 行;返回 (子进程, stdout 行缓冲)。
fn spawn_argus(port: u16) -> (Child, Arc<Mutex<Vec<String>>>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_argus"))
        .env("ARGUS_PORT", port.to_string())
        .env("ARGUS_AUTH_TOKEN", AUTH_TOKEN)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("启动 argus 测试实例");

    let stdout = child.stdout.take().expect("取 stdout 管道");
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&lines);
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            sink.lock().expect("stdout 缓冲锁").push(line);
        }
    });
    (child, lines)
}

/// 裸 HTTP/1.1 GET(不引额外依赖)。
fn http_get(port: u16, path: &str) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("连接 argus");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("设读超时");
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {AUTH_TOKEN}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).expect("写请求");
    let mut sink = Vec::new();
    let _ = stream.read_to_end(&mut sink);
}

fn wait_listening(port: u16) {
    for _ in 0..100 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("argus 未在 10s 内监听 :{port}");
}

#[test]
fn stdout_lines_are_json_and_probe_noise_is_debug_only() {
    let port = 33000 + (std::process::id() % 5000) as u16;
    let (mut child, lines) = spawn_argus(port);
    wait_listening(port);

    // 先打一次探活(应被降噪),再打一次业务路径(应保留 INFO)
    http_get(port, "/health");
    std::thread::sleep(Duration::from_millis(600));
    http_get(port, "/models");
    std::thread::sleep(Duration::from_millis(600));

    let _ = child.kill();
    let _ = child.wait();

    let snapshot = lines.lock().expect("stdout 缓冲锁").clone();
    assert!(
        !snapshot.is_empty(),
        "argus 没有任何 stdout 输出——测试前提不成立"
    );

    // A-1:每行必须是 JSON,且含 level/target/timestamp
    let mut parsed: Vec<serde_json::Value> = Vec::new();
    for (i, line) in snapshot.iter().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("第 {i} 行不是合法 JSON({e}): {line}"));
        for key in ["level", "target", "timestamp"] {
            assert!(
                v.get(key).is_some(),
                "第 {i} 行缺 {key} 字段: {line}"
            );
        }
        parsed.push(v);
    }

    // A-2:探活路径不得出现 INFO 级别的"请求完成"
    let probe_info = parsed.iter().any(|v| {
        v["level"] == "INFO" && v.to_string().contains("/health")
    });
    assert!(
        !probe_info,
        "探活 /health 仍产生 INFO 日志(降噪未生效): {:?}",
        parsed
            .iter()
            .filter(|v| v["level"] == "INFO" && v.to_string().contains("/health"))
            .collect::<Vec<_>>()
    );

    // A-3:非探活路径仍须有 INFO 的"请求完成"
    let business_info = parsed.iter().any(|v| {
        v["level"] == "INFO" && v.to_string().contains("/models")
    });
    assert!(
        business_info,
        "业务路径 /models 缺 INFO 请求完成日志(降噪过头): {:?}",
        parsed
    );
}
