//! 编译期探测 git 构建标识，注入版本字符串。
//!
//! - 无 git 环境（源码包/CI 浅克隆异常）时回退 "unknown"，不阻断构建
//! - `rerun-if-changed=.git/HEAD`：新提交后重新生成版本（hash 变化触发重编译）
//! - 容器构建上下文不含 .git：经 VISION_EXT_BUILD_GIT_HASH / _GIT_DIRTY 环境变量
//!   注入（docker build --build-arg GIT_HASH=...），yolo-rs 同款约定

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-env-changed=VISION_EXT_BUILD_GIT_HASH");
    println!("cargo:rerun-if-env-changed=VISION_EXT_BUILD_GIT_DIRTY");

    let hash = env_override("VISION_EXT_BUILD_GIT_HASH")
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // 工作区有未提交改动（含 untracked）时标记 dirty；容器构建经 env 注入
    let dirty = match env_override("VISION_EXT_BUILD_GIT_DIRTY") {
        Some(v) => v == "true",
        None => Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false),
    };

    println!("cargo:rustc-env=VISION_EXT_GIT_HASH={hash}");
    println!("cargo:rustc-env=VISION_EXT_DIRTY={}", if dirty { "dirty" } else { "" });
}

/// 非空环境变量覆盖（空字符串视为未设置）。
fn env_override(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}
