//! Conversions through the command line, against the fixtures that `fixtures/make_fixtures.py` writes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn convert(from: &str, to: &str, input: &Path, output: &Path) {
    let result = Command::new(env!("CARGO_BIN_EXE_dataset-converter"))
        .args(["--from", from, "--to", to])
        .arg(input)
        .arg(output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{from} to {to} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// Every visible file under `root`, by its path relative to `root`.
fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut folders = vec![root.to_owned()];
    while let Some(folder) = folders.pop() {
        for item in fs::read_dir(folder).unwrap() {
            let path = item.unwrap().path();
            if path.file_name().unwrap().to_string_lossy().starts_with('.') {
                continue;
            }
            if path.is_dir() {
                folders.push(path);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_owned(),
                    fs::read(&path).unwrap(),
                );
            }
        }
    }
    files
}

#[test]
fn yolo_round_trip_gives_the_same_files() {
    let out = tempfile::tempdir().unwrap();
    let parquet = out.path().join("parquet");
    convert("yolo", "parquet", &fixture("yolo"), &parquet);
    convert("parquet", "yolo", &parquet, &out.path().join("yolo"));
    assert_eq!(files(&out.path().join("yolo")), files(&fixture("yolo")));
}

#[test]
fn reads_parquet_that_datasets_wrote() {
    let out = tempfile::tempdir().unwrap();
    convert("parquet", "yolo", &fixture("hf"), &out.path().join("yolo"));
    assert_eq!(files(&out.path().join("yolo")), files(&fixture("yolo")));
}
