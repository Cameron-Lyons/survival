//! Test-time scan of `src/` that fails when a `#[pyclass]` or `#[pyfunction]`
//! declared anywhere in the crate is not registered by a file under
//! `src/api/python/`, or when two declarations share a Python name.
//!
//! The scan is textual so it runs in every feature configuration, including
//! `--no-default-features`, where the PyO3 attributes are no-ops. It
//! recognises the three registration forms used by the binding files:
//! `m.add_class::<Name>()`, `wrap_pyfunction!(name, m)` and the
//! `register_classes!`/`register_functions!` lists.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Index just past the raw string literal whose `r` is at `start`, if one
/// starts there (`r"..."`, `r#"..."#`, ...).
fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    let mut hashes = 0;
    while bytes.get(i) == Some(&b'#') {
        hashes += 1;
        i += 1;
    }
    if bytes.get(i) != Some(&b'"') {
        return None;
    }
    i += 1;
    while i < bytes.len() {
        if bytes[i] == b'"' && bytes[i + 1..].iter().take(hashes).all(|b| *b == b'#') {
            return Some(i + 1 + hashes);
        }
        i += 1;
    }
    None
}

/// Index just past the char literal starting at `start`, if one starts there
/// (`'"'`, `'\\''`, `'\\u{7f}'`); lifetimes and labels are left alone.
fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    let close = if bytes.get(start + 1) == Some(&b'\\') {
        start + 2 + bytes[start + 2..].iter().position(|b| *b == b'\'')?
    } else {
        start + 2
    };
    (bytes.get(close) == Some(&b'\'')).then_some(close + 1)
}

/// Removes `//` and `/* */` comments and blanks the contents of string and
/// char literals, so a `#[pyclass]` quoted in a comment or literal is not a
/// declaration.
fn strip_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        let at_word_start =
            i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if byte == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i = (i + 1).min(bytes.len());
            out.push_str("\"\"");
        } else if let Some(end) = (byte == b'r' && at_word_start)
            .then(|| raw_string_end(bytes, i))
            .flatten()
        {
            out.push_str("\"\"");
            i = end;
        } else if let Some(end) = (byte == b'\'')
            .then(|| char_literal_end(bytes, i))
            .flatten()
        {
            out.push_str("' '");
            i = end;
        } else if byte == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if byte == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 1;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            out.push(' ');
        } else {
            let ch = source[i..].chars().next().expect("in bounds");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Index just past the group that starts at `open` (which must be `[` or `(`).
fn skip_group(source: &str, open: usize) -> usize {
    let bytes = source.as_bytes();
    let (open_byte, close_byte) = match bytes[open] {
        b'[' => (b'[', b']'),
        b'(' => (b'(', b')'),
        other => panic!("not a group opener: {}", other as char),
    };
    let mut depth = 0;
    let mut i = open;
    while i < bytes.len() {
        if bytes[i] == open_byte {
            depth += 1;
        } else if bytes[i] == close_byte {
            depth -= 1;
            if depth == 0 {
                return i + 1;
            }
        }
        i += 1;
    }
    panic!("unbalanced group starting at byte {open}");
}

fn skip_whitespace(source: &str, mut i: usize) -> usize {
    let bytes = source.as_bytes();
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn ident_at(source: &str, i: usize) -> &str {
    let end = source[i..]
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
        .map_or(source.len(), |offset| i + offset);
    &source[i..end]
}

/// Names of the items that carry `#[<attribute>...]` and are introduced by one
/// of `keywords` (after any further attributes and a visibility). Macro
/// templates (`$name`) are skipped.
fn declared(source: &str, attribute: &str, keywords: &[&str]) -> Vec<String> {
    let marker = format!("#[{attribute}");
    let mut names = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = source[search_from..].find(&marker) {
        let at = search_from + offset;
        let mut i = skip_group(source, at + 1);
        search_from = i;
        loop {
            i = skip_whitespace(source, i);
            if source[i..].starts_with("#[") {
                i = skip_group(source, i + 1);
            } else {
                break;
            }
        }
        if source[i..].starts_with("pub") {
            i += 3;
            i = skip_whitespace(source, i);
            if source[i..].starts_with('(') {
                i = skip_group(source, i);
                i = skip_whitespace(source, i);
            }
        }
        let keyword = ident_at(source, i);
        if !keywords.contains(&keyword) {
            continue;
        }
        i = skip_whitespace(source, i + keyword.len());
        let name = ident_at(source, i);
        if !name.is_empty() && !name.starts_with('$') {
            names.push(name.to_string());
        }
    }
    names
}

fn last_segment(path: &str) -> String {
    path.trim()
        .rsplit("::")
        .next()
        .unwrap_or_default()
        .trim_start_matches("r#")
        .to_string()
}

/// Names registered by `add_class::<..>`, `wrap_pyfunction!(..)` and the
/// `register_classes!`/`register_functions!` lists.
fn registered(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (marker, terminator) in [("add_class::<", '>'), ("wrap_pyfunction!(", ',')] {
        let mut search_from = 0;
        while let Some(offset) = source[search_from..].find(marker) {
            let start = search_from + offset + marker.len();
            let end = start
                + source[start..]
                    .find(terminator)
                    .expect("terminated registration");
            names.push(last_segment(&source[start..end]));
            search_from = end;
        }
    }
    for marker in ["register_classes!(", "register_functions!("] {
        let mut search_from = 0;
        while let Some(offset) = source[search_from..].find(marker) {
            let open = search_from + offset + marker.len() - 1;
            let close = skip_group(source, open);
            let body = &source[open + 1..close - 1];
            names.extend(
                body.split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty() && *item != "m")
                    .map(last_segment),
            );
            search_from = close;
        }
    }
    names
}

#[test]
fn every_pyclass_and_pyfunction_is_registered_exactly_once() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);

    let mut declarations: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut registrations: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let source =
            strip_comments_and_literals(&fs::read_to_string(path).expect("readable source"));
        let relative = path
            .strip_prefix(&src)
            .expect("under src")
            .display()
            .to_string();
        let mut names = declared(&source, "pyclass", &["struct", "enum"]);
        names.extend(declared(&source, "pyfunction", &["fn"]));
        for name in names {
            declarations.entry(name).or_default().push(relative.clone());
        }
        if relative.starts_with("api/python") {
            for name in registered(&source) {
                *registrations.entry(name).or_default() += 1;
            }
        }
    }
    assert!(
        declarations.len() > 500 && registrations.len() > 500,
        "scan found only {} declarations and {} registrations; the scanner is broken",
        declarations.len(),
        registrations.len()
    );

    let mut problems = Vec::new();
    for (name, files) in &declarations {
        if files.len() > 1 {
            problems.push(format!(
                "{name} is declared more than once: {}",
                files.join(", ")
            ));
        }
        match registrations.get(name) {
            None => problems.push(format!("{name} ({}) is never registered", files[0])),
            Some(1) => {}
            Some(count) => problems.push(format!("{name} is registered {count} times")),
        }
    }
    assert!(
        problems.is_empty(),
        "binding registration problems:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn scanner_handles_attributes_visibility_and_macros() {
    let source = r###"
        /// `#[pyclass]` in a doc comment is ignored.
        #[pyclass(from_py_object)]
        #[derive(Debug, Clone)]
        pub struct Visible;
        #[pyclass] pub(crate) enum Hidden { A }
        #[pyfunction(name = "exported")]
        #[pyo3(signature = (x, y=None))]
        fn internal_name(x: f64) {}
        macro_rules! m { ($name:ident) => { #[pyclass] pub struct $name; } }
        // #[pyfunction] fn commented() {}
        const S: &str = "#[pyfunction] fn in_string() {}";
        const R: &str = r##"#[pyclass] "quoted" struct InRaw;"##;
        const C: char = '"'; #[pyclass] struct AfterChar;
        fn lifetimes<'a>(x: &'a str) -> &'a str { x }
    "###;
    let source = strip_comments_and_literals(source);
    assert_eq!(
        declared(&source, "pyclass", &["struct", "enum"]),
        ["Visible", "Hidden", "AfterChar"]
    );
    assert_eq!(declared(&source, "pyfunction", &["fn"]), ["internal_name"]);

    let registration = r#"
        m.add_class::<Visible>()?;
        m.add_function(wrap_pyfunction!(crate::a::internal_name, m)?)?;
        register_classes!(m, Hidden, b::Other,);
        register_functions!(
            m,
            load_aml,
            load_lung
        );
    "#;
    assert_eq!(
        registered(registration),
        [
            "Visible",
            "internal_name",
            "Hidden",
            "Other",
            "load_aml",
            "load_lung"
        ]
    );
}
