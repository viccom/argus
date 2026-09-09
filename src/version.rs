/// 编译期版本标识。
pub fn version_string() -> String {
    format!(
        "vision-ext {} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_NAME")
    )
}
