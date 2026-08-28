use std::fs;
use std::path::{Path, PathBuf};

fn json_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("tableau resource directory must be readable") {
        let path = entry.expect("resource entry must be readable").path();
        if path.is_dir() {
            json_files(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            files.push(path);
        }
    }
}

fn scalar_arrays_are_single_line(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (index, &byte) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'[' => stack.push((index, false)),
            b'{' => {
                if let Some((_, nested)) = stack.last_mut() {
                    *nested = true;
                }
            }
            b']' => {
                let Some((start, nested)) = stack.pop() else {
                    return false;
                };
                if !nested && source[start..=index].contains('\n') {
                    return false;
                }
                if let Some((_, parent_nested)) = stack.last_mut() {
                    *parent_nested = true;
                }
            }
            _ => {}
        }
    }

    stack.is_empty() && !in_string
}

#[test]
fn compile_time_tableau_resources_are_valid_and_fractured() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tableau/resources");
    let mut files = Vec::new();
    json_files(&root, &mut files);
    files.sort();

    assert!(
        !files.is_empty(),
        "tableau resources must remain discoverable"
    );
    for path in files {
        let source = fs::read_to_string(&path).expect("tableau resource must be UTF-8");
        serde_json::from_str::<serde_json::Value>(&source)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert!(
            !source.contains("schema_version"),
            "{} retains an obsolete schema version",
            path.display()
        );
        assert!(
            scalar_arrays_are_single_line(&source),
            "{} must keep scalar arrays and matrix rows on one line",
            path.display()
        );
        assert!(
            source.ends_with('\n'),
            "{} needs a final newline",
            path.display()
        );
    }
}
