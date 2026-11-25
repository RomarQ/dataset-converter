//! The Ultralytics YOLO layout.
//!
//! A dataset is a folder with a `data.yaml` such as:
//!
//! ```yaml
//! train: images/train
//! val: images/val
//! names:
//!   0: ball
//!   1: target
//! ```
//!
//! Each of the keys `train`, `val` and `test` names the folder of a split's pictures, relative to the folder
//! of `data.yaml`; a `path` key is not read. `names` maps each class index to its name, or lists the names.
//!
//! The boxes of `images/<split>/<name>.jpg` are in `labels/<split>/<name>.txt`, one box a line as
//! `class cx cy w h`: the class index, then the box's centre and size in fractions of the picture. A picture
//! with no label file has no boxes.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use saphyr::{LoadableYamlNode, SafelyIndex, Yaml};

use crate::{Bbox, Frame, Object, Split};

/// The picture file extensions read: Ultralytics' `IMG_FORMATS` whose formats the `imagesize` crate sizes.
const IMAGE_EXTENSIONS: [&str; 7] = ["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

/// A YOLO dataset on disk.
#[derive(Debug)]
pub struct Dataset {
    names: Vec<String>,
    images: BTreeMap<Split, PathBuf>,
}

/// A picture of a split, with its label file.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The picture file.
    pub image: PathBuf,
    /// The label file, or `None` for a picture with no boxes.
    pub label: Option<PathBuf>,
    /// The size of the picture file in bytes.
    pub size: u64,
}

impl Dataset {
    /// Reads the `data.yaml` in `root`.
    ///
    /// # Errors
    ///
    /// Fails when `data.yaml` cannot be read, names no split, or has no class names.
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("data.yaml");
        let text =
            fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display()))?;
        parse_data_yaml(root, &text).with_context(|| format!("cannot use {}", path.display()))
    }

    /// The class names, by class index.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The splits that `data.yaml` names.
    pub fn splits(&self) -> impl Iterator<Item = Split> + '_ {
        self.images.keys().copied()
    }

    /// The pictures of `split`, sorted by file name.
    ///
    /// Files without a picture extension, and hidden files, are skipped.
    ///
    /// # Errors
    ///
    /// Fails when a folder cannot be read, when two pictures share a label file, such as `a.jpg` and `a.png`,
    /// or when a label file has no picture, since its boxes would be lost.
    pub fn entries(&self, split: Split) -> Result<Vec<Entry>> {
        let Some(images) = self.images.get(&split) else {
            return Ok(Vec::new());
        };
        let labels = label_dir(images)?;
        let mut by_stem = BTreeMap::new();
        for item in
            fs::read_dir(images).with_context(|| format!("cannot read {}", images.display()))?
        {
            let image = item?.path();
            if !is_picture(&image) {
                continue;
            }
            let stem = file_stem(&image)?;
            let label = labels.join(format!("{stem}.txt"));
            let size = fs::metadata(&image)
                .with_context(|| format!("cannot read {}", image.display()))?
                .len();
            let entry = Entry {
                label: label.is_file().then_some(label),
                image,
                size,
            };
            if let Some(other) = by_stem.insert(stem.clone(), entry) {
                bail!(
                    "{} and another picture share the label file {stem}.txt",
                    other.image.display()
                );
            }
        }
        if labels.is_dir() {
            for item in fs::read_dir(&labels)
                .with_context(|| format!("cannot read {}", labels.display()))?
            {
                let label = item?.path();
                if label.extension().is_some_and(|ext| ext == "txt") {
                    ensure!(
                        by_stem.contains_key(&file_stem(&label)?),
                        "{} has no picture in {}",
                        label.display(),
                        images.display()
                    );
                }
            }
        }
        let mut entries: Vec<Entry> = by_stem.into_values().collect();
        entries.sort_by(|a, b| a.image.file_name().cmp(&b.image.file_name()));
        Ok(entries)
    }

    /// Reads a picture and its boxes.
    ///
    /// # Errors
    ///
    /// Fails when a file cannot be read, when the picture's size cannot be read from its header, or when a
    /// label line is not `class cx cy w h` with a class that has a name.
    pub fn read(&self, entry: &Entry) -> Result<Frame> {
        let bytes = fs::read(&entry.image)
            .with_context(|| format!("cannot read {}", entry.image.display()))?;
        let size = imagesize::blob_size(&bytes)
            .map_err(|err| anyhow!("cannot read the size of {}: {err}", entry.image.display()))?;
        let width = u32::try_from(size.width)?;
        let height = u32::try_from(size.height)?;
        let mut objects = Vec::new();
        if let Some(label) = &entry.label {
            let text = fs::read_to_string(label)
                .with_context(|| format!("cannot read {}", label.display()))?;
            for (number, line) in text.lines().enumerate() {
                if !line.trim().is_empty() {
                    let object = parse_line(line, width, height, self.names.len());
                    objects.push(
                        object.with_context(|| format!("{}:{}", label.display(), number + 1))?,
                    );
                }
            }
        }
        let file_name = entry
            .image
            .file_name()
            .and_then(|name| name.to_str())
            .context("the picture name is not UTF-8")?;
        Ok(Frame {
            file_name: file_name.to_owned(),
            bytes,
            width,
            height,
            objects,
        })
    }
}

/// Writes a dataset in the YOLO layout.
#[derive(Debug)]
pub struct Writer {
    root: PathBuf,
    names: Vec<String>,
    splits: BTreeSet<Split>,
}

impl Writer {
    /// Starts a dataset in `root`, which must be missing or empty.
    ///
    /// # Errors
    ///
    /// Fails when `root` is not empty or cannot be created.
    pub fn create(root: &Path, names: Vec<String>) -> Result<Self> {
        crate::create_empty_dir(root)?;
        Ok(Self {
            root: root.to_owned(),
            names,
            splits: BTreeSet::new(),
        })
    }

    /// Writes the picture of `frame` and its label file.
    ///
    /// # Errors
    ///
    /// Fails when the picture's name has a folder part, when the split already has a picture or label file of
    /// that name, when a box's class has no name, or when a file cannot be written.
    pub fn write(&mut self, split: Split, frame: &Frame) -> Result<()> {
        let name = plain_file_name(&frame.file_name)?;
        let images = self.root.join("images").join(split.yolo_name());
        let labels = self.root.join("labels").join(split.yolo_name());
        if self.splits.insert(split) {
            fs::create_dir_all(&images)
                .with_context(|| format!("cannot create {}", images.display()))?;
            fs::create_dir_all(&labels)
                .with_context(|| format!("cannot create {}", labels.display()))?;
        }
        ensure!(
            frame.objects.is_empty() || (frame.width > 0 && frame.height > 0),
            "{name} has boxes but no picture size"
        );
        let mut text = String::new();
        for object in &frame.objects {
            ensure!(
                object.class < self.names.len(),
                "{name} has a box of class {}, which has no name",
                object.class
            );
            text.push_str(&format_line(object, frame.width, frame.height));
            text.push('\n');
        }
        write_new(&images.join(name), &frame.bytes)?;
        write_new(
            &labels.join(format!("{}.txt", file_stem(Path::new(name))?)),
            text.as_bytes(),
        )
    }

    /// Writes `data.yaml`.
    ///
    /// # Errors
    ///
    /// Fails when `data.yaml` cannot be written.
    pub fn finish(self) -> Result<()> {
        let mut text = String::new();
        for split in &self.splits {
            text.push_str(&format!("{0}: images/{0}\n", split.yolo_name()));
        }
        text.push_str("names:\n");
        for (index, name) in self.names.iter().enumerate() {
            text.push_str(&format!("  {index}: {}\n", yaml_scalar(name)));
        }
        write_new(&self.root.join("data.yaml"), text.as_bytes())
    }
}

/// Reads a label line, `class cx cy w h` in fractions of a `width` by `height` picture, as a box in pixels.
fn parse_line(line: &str, width: u32, height: u32, classes: usize) -> Result<Object> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let &[class, cx, cy, w, h] = fields.as_slice() else {
        bail!("expected `class cx cy w h`, found {line:?}");
    };
    let class: usize = class
        .parse()
        .with_context(|| format!("{class:?} is not a class index"))?;
    ensure!(class < classes, "class {class} has no name in data.yaml");
    let value = |text: &str| {
        text.parse::<f64>()
            .with_context(|| format!("{text:?} is not a number"))
    };
    let (cx, cy, w, h) = (value(cx)?, value(cy)?, value(w)?, value(h)?);
    let (width, height) = (f64::from(width), f64::from(height));
    let bbox = Bbox {
        x: (cx - w / 2.0) * width,
        y: (cy - h / 2.0) * height,
        width: w * width,
        height: h * height,
    };
    Ok(Object { class, bbox })
}

/// Writes a box as a label line, without the line end.
fn format_line(object: &Object, width: u32, height: u32) -> String {
    let (width, height) = (f64::from(width), f64::from(height));
    let Bbox {
        x,
        y,
        width: w,
        height: h,
    } = object.bbox;
    let [cx, cy, w, h] = [
        (x + w / 2.0) / width,
        (y + h / 2.0) / height,
        w / width,
        h / height,
    ]
    .map(|v| if v == 0.0 { 0.0 } else { v });
    format!("{} {cx:.6} {cy:.6} {w:.6} {h:.6}", object.class)
}

fn parse_data_yaml(root: &Path, text: &str) -> Result<Dataset> {
    let docs = Yaml::load_from_str(text)?;
    let doc = docs.first().context("the file is empty")?;
    let names: Vec<String> = match doc.get("names") {
        Some(Yaml::Mapping(map)) => {
            let mut by_index = BTreeMap::new();
            for (key, value) in map {
                let index = key
                    .as_integer()
                    .context("a key of `names` is not a class index")?;
                by_index.insert(index, scalar_text(value)?);
            }
            ensure!(
                (0..).zip(by_index.keys()).all(|(n, index)| n == *index),
                "the class indexes in `names` do not count up from 0"
            );
            by_index.into_values().collect()
        }
        Some(Yaml::Sequence(list)) => list.iter().map(scalar_text).collect::<Result<_>>()?,
        _ => bail!("there is no `names` mapping or list"),
    };
    if let Some(count) = doc.get("nc").and_then(Yaml::as_integer) {
        ensure!(
            usize::try_from(count) == Ok(names.len()),
            "`nc` is {count}, but `names` has {} classes",
            names.len()
        );
    }
    let mut images = BTreeMap::new();
    for split in Split::ALL {
        let Some(value) = doc.get(split.yolo_name()) else {
            continue;
        };
        let folder = value
            .as_str()
            .with_context(|| format!("`{}` is not one folder path", split.yolo_name()))?;
        ensure!(
            !folder.ends_with(".txt"),
            "`{}` is a list file; only a folder of pictures is read",
            split.yolo_name()
        );
        images.insert(split, root.join(folder));
    }
    ensure!(
        !images.is_empty(),
        "there is no `train`, `val` or `test` key"
    );
    Ok(Dataset { names, images })
}

fn scalar_text(value: &Yaml) -> Result<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    value
        .as_integer()
        .map(|number| number.to_string())
        .context("a class name is not text")
}

/// The label folder of a picture folder: Ultralytics replaces the last `images` part of the path with `labels`.
fn label_dir(images: &Path) -> Result<PathBuf> {
    let parts: Vec<Component> = images.components().collect();
    let index = parts
        .iter()
        .rposition(|part| part.as_os_str() == "images")
        .with_context(|| {
            format!(
                "{} has no `images` folder in its path, so its labels cannot be found",
                images.display()
            )
        })?;
    let mut labels = PathBuf::new();
    for (position, part) in parts.iter().enumerate() {
        labels.push(if position == index {
            OsStr::new("labels")
        } else {
            part.as_os_str()
        });
    }
    Ok(labels)
}

fn is_picture(path: &Path) -> bool {
    let visible = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| !name.starts_with('.'));
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);
    visible
        && path.is_file()
        && extension.is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.as_str()))
}

fn file_stem(path: &Path) -> Result<String> {
    let stem = path.file_stem().and_then(|stem| stem.to_str());
    stem.map(str::to_owned)
        .with_context(|| format!("{} has no UTF-8 file name", path.display()))
}

/// `name` when it is a file name alone. A picture name read from a Parquet file must not reach outside its folder.
fn plain_file_name(name: &str) -> Result<&str> {
    let mut parts = Path::new(name).components();
    let plain = matches!((parts.next(), parts.next()), (Some(Component::Normal(part)), None) if part == name);
    ensure!(
        plain && !name.contains('\\'),
        "the picture name {name:?} is not a plain file name"
    );
    Ok(name)
}

/// `text` as a YAML scalar that PyYAML, which Ultralytics uses, reads back as the same string.
fn yaml_scalar(text: &str) -> String {
    let plain = text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && !text.ends_with(' ')
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.'))
        && !matches!(
            text.to_ascii_lowercase().as_str(),
            "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "y" | "n"
        );
    if plain {
        text.to_owned()
    } else {
        serde_json::Value::from(text).to_string()
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file =
        File::create_new(path).with_context(|| format!("cannot create {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_class_with_no_name() {
        let error = parse_line("2 0.5 0.5 0.1 0.1", 640, 360, 2).unwrap_err();
        assert_eq!(error.to_string(), "class 2 has no name in data.yaml");
    }

    #[test]
    fn rejects_picture_names_that_leave_the_folder() {
        for name in ["../a.jpg", "sub/a.jpg", "/a.jpg", "..", "", "a\\b.jpg"] {
            assert!(plain_file_name(name).is_err(), "{name:?} passed");
        }
        assert_eq!(
            plain_file_name("-clip_f00001.jpg").unwrap(),
            "-clip_f00001.jpg"
        );
    }
}
