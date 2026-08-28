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

#[test]
fn compile_time_tableau_resources_are_valid() {
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
    }
}
