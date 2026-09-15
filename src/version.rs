/// 编译期版本标识。
///
/// git hash 与 dirty 状态由 build.rs 在编译期探测注入（无 git 环境时为 "unknown"），
/// 使部署实例可通过 `/health`、`/` 与启动/停机日志精确定位到构建来源。
pub fn version_string() -> String {
    let dirty = env!("ARGUS_DIRTY");
    let dirty_mark = if dirty.is_empty() { "" } else { "+dirty" };
    format!(
        "{} {} (git {}{})",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("ARGUS_GIT_HASH"),
        dirty_mark
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_should_contain_pkg_version_and_git_info() {
        let v = version_string();
        assert!(v.contains(env!("CARGO_PKG_VERSION")), "应含包版本: {v}");
        assert!(v.contains("git"), "应含 git 标识: {v}");
        assert!(
            v.contains(env!("ARGUS_GIT_HASH")),
            "应含编译期注入的 git hash: {v}"
        );
    }

    #[test]
    fn version_string_should_mark_dirty_build() {
        // dirty 标记与编译期状态一致：非空时必须出现 +dirty 后缀
        if !env!("ARGUS_DIRTY").is_empty() {
            assert!(version_string().contains("+dirty"));
        }
    }
}
