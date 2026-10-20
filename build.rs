//! 编译期探测 git 构建标识，注入版本字符串。
//!
//! - 无 git 环境（源码包/CI 浅克隆异常）时回退 "unknown"，不阻断构建
//! - 监听 HEAD 与 refs/：普通提交只改 `refs/heads/<分支>`，HEAD 文件本身不变，
//!   只监听 .git/HEAD 会让版本串停留在旧 commit。路径经 `git rev-parse --git-dir`
//!   解析，构建目录不等于仓库根目录（cargo 在任意 cwd 调用）时同样有效
//! - 容器构建上下文不含 .git：经 ARGUS_BUILD_GIT_HASH / _GIT_DIRTY 环境变量
//!   注入（docker build --build-arg GIT_HASH=...），yolo-rs 同款约定

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ARGUS_BUILD_GIT_HASH");
    println!("cargo:rerun-if-env-changed=ARGUS_BUILD_GIT_DIRTY");

    // 监听 git 元数据：无 git 时静默跳过（回退 unknown）
    if let Some(git_dir) = git_dir() {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("refs").display());
        // 打包仓库（packed-refs）时分支引用不在 refs/ 下
        println!("cargo:rerun-if-changed={}", git_dir.join("packed-refs").display());
    }

    let hash = env_override("ARGUS_BUILD_GIT_HASH")
        .or_else(|| git_output(&["rev-parse", "--short", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());

    // 工作区有未提交改动（含 untracked）时标记 dirty；容器构建经 env 注入
    let dirty = match env_override("ARGUS_BUILD_GIT_DIRTY") {
        Some(v) => v == "true",
        None => Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false),
    };

    println!("cargo:rustc-env=ARGUS_GIT_HASH={hash}");
    println!("cargo:rustc-env=ARGUS_DIRTY={}", if dirty { "dirty" } else { "" });
}

/// 仓库 git 目录的绝对路径（无 git 仓库时为 None）。
fn git_dir() -> Option<PathBuf> {
    let raw = git_output(&["rev-parse", "--git-dir"])?;
    let path = PathBuf::from(raw);
    // rev-parse 相对构建 cwd 解析；转成绝对路径供 cargo 监听
    Some(if path.is_absolute() {
        path
    } else {
        std::env::current_dir().ok()?.join(path)
    })
}

/// 执行 git 子命令，成功且非空时返回去除首尾空白的 stdout。
fn git_output(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// 非空环境变量覆盖（空字符串视为未设置）。
fn env_override(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}
