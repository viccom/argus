//! 检测族动态槽：`slots.toml` 显式清单 + 懒加载 + 失败退避 + 热重载 diff。
//!
//! 设计权威：`docs/designs/2026-10-21-engine-manager-dynamic-slots-design.md`。
//! 静态 3 槽（detection/ppe/fire_smoke 的 OnceCell）不在本模块管理——registry
//! 纯增量（设计 C1），`slots.toml` 缺失时动态槽为空、行为零回归。

// 本模块的生产消费者在后续任务接入（T5 注册表 / T6 热重载），当前仅测试引用，
// bin 目标下 dead_code 属增量交付的已知中间态——接入后移除本 allow。
#![allow(dead_code)]

use std::path::{Component, Path, PathBuf};

use crate::config::CAPABILITY_KEYS;

/// 动态槽管线标识（`data.pipeline`），与静态能力键的串不同族。
pub const SLOT_PIPELINE: &str = "detect-slot-v1";

/// 校验后的槽规格（热重载 diff 的比较单元）。
#[derive(Debug, Clone, PartialEq)]
pub struct SlotSpec {
    pub name: String,
    /// 已解析的绝对路径（符号链接已按真实路径判定，见 `resolve_within`）。
    pub model_path: PathBuf,
    pub class_names: Option<Vec<String>>,
    pub pool_size: Option<usize>,
    pub warmup_size: Option<(u32, u32)>,
}

/// toml 原始槽（serde 反序列化；未知字段在表级 warn 后忽略）。
#[derive(Debug, serde::Deserialize)]
struct SlotRaw {
    model: String,
    class_names: Option<Vec<String>>,
    pool_size: Option<usize>,
    warmup_size: Option<Vec<u32>>,
}

/// 槽的已知字段（未知字段 warn 不阻断，向前兼容）。
const SLOT_FIELDS: [&str; 4] = ["model", "class_names", "pool_size", "warmup_size"];

/// 槽名规则：小写字母开头，仅小写字母/数字/下划线，长度 1..=32。
pub fn is_valid_slot_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    match bytes.first() {
        Some(b'a'..=b'z') => {}
        _ => return false,
    }
    bytes.len() <= 32
        && bytes[1..].iter().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

/// 解析并校验 `slots.toml` 文本。任一槽非法 → 整份拒绝（不部分应用）。
pub fn parse_slots_toml(text: &str, models_dir: &Path) -> Result<Vec<SlotSpec>, String> {
    let value: toml::Value = text.parse().map_err(|e| format!("slots.toml 解析失败: {e}"))?;
    let empty = toml::Value::Table(Default::default());
    let slots_table = value
        .get("slots")
        .unwrap_or(&empty)
        .as_table()
        .ok_or_else(|| "slots 必须是 [slots.<name>] 表".to_string())?;
    if slots_table.is_empty() {
        return Ok(Vec::new());
    }
    // models_dir 每份清单只 canonicalize 一次；空清单零文件系统调用（零回归）。
    let dir_canon = std::fs::canonicalize(models_dir)
        .map_err(|e| format!("models_dir {} 不可用: {e}", models_dir.display()))?;

    let mut specs = Vec::with_capacity(slots_table.len());
    for (name, raw_value) in slots_table {
        if !is_valid_slot_name(name) {
            return Err(format!("槽名 {name:?} 非法（须 ^[a-z][a-z0-9_]{{0,31}}$）"));
        }
        if CAPABILITY_KEYS.contains(&name.as_str()) {
            return Err(format!("槽名 {name:?} 是能力键保留字，请换名"));
        }
        let raw_table = raw_value.as_table().ok_or_else(|| format!("槽 {name} 必须是表"))?;
        for key in raw_table.keys() {
            if !SLOT_FIELDS.contains(&key.as_str()) {
                tracing::warn!("slots.toml 槽 {name} 含未知字段 {key}（忽略，向前兼容）");
            }
        }
        let raw: SlotRaw = raw_value
            .clone()
            .try_into()
            .map_err(|e| format!("槽 {name} 字段无效: {e}"))?;
        let model = raw.model.trim();
        if model.is_empty() {
            return Err(format!("槽 {name} 缺 model"));
        }
        if Path::new(model).components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(format!("槽 {name} 的 model 路径不允许包含 .. 分量"));
        }
        // 纯内存校验先于文件系统调用（先廉后贵）
        if let Some(names) = &raw.class_names {
            if names.is_empty() || names.iter().any(|s| s.trim().is_empty()) {
                return Err(format!("槽 {name} 的 class_names 不能含空项"));
            }
        }
        if let Some(p) = raw.pool_size {
            if !(1..=64).contains(&p) {
                return Err(format!("槽 {name} 的 pool_size {p} 越界（1..=64）"));
            }
        }
        if let Some(size) = &raw.warmup_size {
            if size.len() != 2 || !size.iter().all(|v| (32..=4096).contains(v)) {
                return Err(format!("槽 {name} 的 warmup_size 须为 [w,h] 且各维 32..=4096"));
            }
        }
        let model_path = resolve_within(&dir_canon, model)
            .map_err(|e| format!("槽 {name} 的 model 路径无效: {e}"))?;
        specs.push(SlotSpec {
            name: name.clone(),
            model_path,
            class_names: raw.class_names,
            pool_size: raw.pool_size,
            warmup_size: raw.warmup_size.map(|v| (v[0], v[1])),
        });
    }
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(specs)
}

/// 解析槽模型路径并强制落在 models_dir 内（符号链接按真实路径判定）。
/// 相对路径基于传入的已规范化目录（dir_canon）；绝对路径也必须落在该目录内。
fn resolve_within(dir_canon: &Path, model: &str) -> Result<PathBuf, String> {
    let candidate = if Path::new(model).is_absolute() {
        PathBuf::from(model)
    } else {
        dir_canon.join(model)
    };
    let resolved = best_canonicalize(&candidate);
    if !resolved.starts_with(dir_canon) {
        return Err(format!("模型路径必须在 models_dir({}) 内", dir_canon.display()));
    }
    Ok(resolved)
}

/// canonicalize 尽力而为版：模型文件可能尚未就位，自底向上找最深已存在祖先做
/// 真实解析，余下分量逆序拼回（尾部不存在 ≠ 越界；符号链接按祖先的真实路径
/// 判定）。迭代实现：入口虽已拒绝 `..` 分量，上万层的深路径仍会打爆递归栈
/// （实测 exit 134 abort），违反热重载"失败保旧配置"契约。
fn best_canonicalize(path: &Path) -> PathBuf {
    let mut current = path;
    let mut tails: Vec<std::ffi::OsString> = Vec::new();
    while let Some(parent) = current.parent() {
        match std::fs::canonicalize(current) {
            Ok(resolved) => {
                let mut full = resolved;
                for t in tails.iter().rev() {
                    full.push(t);
                }
                return full;
            }
            Err(_) => match current.file_name() {
                Some(file) => {
                    tails.push(file.to_os_string());
                    current = parent;
                }
                // 无文件名分量可收集（如根/空路径），放弃上溯按原样返回
                None => break,
            },
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个测试独立的临时 models 目录（含一个真实存在的假模型文件）。
    fn tmp_models(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("argus-slots-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.onnx"), b"fake").unwrap();
        dir
    }

    #[test]
    fn valid_minimal_and_full_slots_should_parse() {
        let dir = tmp_models("parse-ok");
        let specs = parse_slots_toml(
            "[slots.crack]\nmodel = \"m.onnx\"\n",
            &dir,
        ).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "crack");
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join("m.onnx"));
        assert!(specs[0].class_names.is_none() && specs[0].pool_size.is_none());

        let specs = parse_slots_toml(
            "[slots.ppe_sh17]\nmodel = \"m.onnx\"\nclass_names = [\"Hardhat\", \"NO-Hardhat\"]\npool_size = 2\nwarmup_size = [640, 640]\n",
            &dir,
        ).unwrap();
        assert_eq!(specs[0].class_names, Some(vec!["Hardhat".into(), "NO-Hardhat".into()]));
        assert_eq!(specs[0].pool_size, Some(2));
        assert_eq!(specs[0].warmup_size, Some((640, 640)));
    }

    #[test]
    fn reserved_and_invalid_names_should_be_rejected() {
        let dir = tmp_models("parse-name");
        for bad in ["ppe", "detection", "Fire", "1x", "a-b", "a".repeat(33).as_str()] {
            let toml_text = format!("[slots.\"{bad}\"]\nmodel = \"m.onnx\"\n");
            assert!(parse_slots_toml(&toml_text, &dir).is_err(), "{bad} 不应通过");
        }
    }

    #[test]
    fn path_escape_should_be_rejected() {
        let dir = tmp_models("parse-escape");
        let outside = dir.parent().unwrap().join("outside.onnx");
        std::fs::write(&outside, b"x").unwrap();
        // 相对路径逃逸
        assert!(parse_slots_toml("[slots.a]\nmodel = \"../outside.onnx\"\n", &dir).is_err());
        // 绝对路径在目录外
        assert!(parse_slots_toml(
            &format!("[slots.a]\nmodel = \"{}\"\n", outside.display()), &dir,
        ).is_err());
        // 符号链接指向目录外（真实路径判定）
        #[cfg(unix)]
        {
            let link = dir.join("alias.onnx");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            assert!(parse_slots_toml("[slots.a]\nmodel = \"alias.onnx\"\n", &dir).is_err());
        }
        // 绝对路径在目录内 → 允许
        let inside = dir.canonicalize().unwrap().join("m.onnx");
        assert!(parse_slots_toml(&format!("[slots.a]\nmodel = \"{}\"\n", inside.display()), &dir).is_ok());
    }

    #[test]
    fn not_yet_existing_model_inside_dir_should_be_allowed() {
        // 槽允许先注册后放模型：文件缺失在首次请求构建时报错（设计 §6），路径约束仍须生效
        let dir = tmp_models("parse-missing");
        let specs = parse_slots_toml("[slots.future]\nmodel = \"not_yet.onnx\"\n", &dir).unwrap();
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join("not_yet.onnx"));
    }

    #[test]
    fn value_ranges_should_be_enforced() {
        let dir = tmp_models("parse-range");
        for bad in [
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 0\n",
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 65\n",
            "[slots.a]\nmodel = \"m.onnx\"\nwarmup_size = [16, 640]\n",
            "[slots.a]\nmodel = \"m.onnx\"\nwarmup_size = [640]\n",
            "[slots.a]\nmodel = \"m.onnx\"\nclass_names = []\n",
            "[slots.a]\nmodel = \"m.onnx\"\nclass_names = [\"\"]\n",
            "[slots.a]\nmodel = \"\"\n",
        ] {
            assert!(parse_slots_toml(bad, &dir).is_err(), "应拒绝: {bad}");
        }
    }

    #[test]
    fn boundary_values_should_be_accepted() {
        let dir = tmp_models("parse-bound");
        let specs = parse_slots_toml(
            "[slots.a]\nmodel = \"m.onnx\"\npool_size = 1\nwarmup_size = [32, 32]\n", &dir,
        ).unwrap();
        assert_eq!(specs[0].pool_size, Some(1));
        assert_eq!(specs[0].warmup_size, Some((32, 32)));

        let specs = parse_slots_toml(
            "[slots.b]\nmodel = \"m.onnx\"\npool_size = 64\nwarmup_size = [4096, 4096]\n", &dir,
        ).unwrap();
        assert_eq!(specs[0].pool_size, Some(64));
        assert_eq!(specs[0].warmup_size, Some((4096, 4096)));
    }

    #[test]
    fn whitespace_only_model_should_be_rejected_as_missing() {
        let dir = tmp_models("parse-ws");
        let err = parse_slots_toml("[slots.a]\nmodel = \"   \"\n", &dir).unwrap_err();
        assert!(err.contains("缺 model"), "实际错误: {err}");
    }

    #[test]
    fn parent_dir_components_should_be_rejected() {
        let dir = tmp_models("parse-dotdot");
        for bad in ["../x.onnx", "a/../m.onnx", "m/..", ".."] {
            let toml_text = format!("[slots.a]\nmodel = \"{bad}\"\n");
            let err = parse_slots_toml(&toml_text, &dir).unwrap_err();
            assert!(err.contains(".."), "{bad} 应因含 .. 分量被拒，实际: {err}");
        }
    }

    #[test]
    fn deep_legal_path_should_not_crash() {
        // 10000 层合法分量（无 ..）：迭代实现不得栈溢出（递归版会 abort，
        // 违反热重载"失败保旧配置"契约），路径仍须落在 models_dir 内
        let dir = tmp_models("parse-deep");
        let deep = format!("{}m.onnx", "a/".repeat(10_000));
        let specs = parse_slots_toml(
            &format!("[slots.a]\nmodel = \"{deep}\"\n"), &dir,
        ).unwrap();
        assert_eq!(specs[0].model_path, dir.canonicalize().unwrap().join(&deep));
    }

    #[test]
    fn truncated_or_duplicate_toml_should_be_rejected() {
        let dir = tmp_models("parse-bad");
        assert!(parse_slots_toml("[slots.crack\nmodel = ", &dir).is_err());
        assert!(parse_slots_toml(
            "[slots.a]\nmodel = \"m.onnx\"\n[slots.a]\nmodel = \"m.onnx\"\n", &dir,
        ).is_err(), "重复槽名应被 toml 解析拒绝");
    }

    #[test]
    fn unknown_field_should_be_ignored_and_empty_file_means_no_slots() {
        let dir = tmp_models("parse-fwd");
        assert!(parse_slots_toml("[slots.a]\nmodel = \"m.onnx\"\nextra = 1\n", &dir).is_ok(),
            "未知字段 warn 后忽略（向前兼容）");
        assert!(parse_slots_toml("", &dir).unwrap().is_empty());
        assert!(parse_slots_toml("[other]\nx = 1\n", &dir).unwrap().is_empty(), "无 [slots] 段 = 空清单");
    }
}
