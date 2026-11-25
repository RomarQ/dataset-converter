//! Converts object-detection datasets between the Ultralytics YOLO layout and Hugging Face Parquet.
//!
//! The YOLO side is a folder with `data.yaml`, `images/<split>/` and `labels/<split>/`, one box a line as
//! `class cx cy w h` in fractions of the picture (see [`yolo`]). The Parquet side is the layout that the
//! `datasets` library writes and the Hugging Face Hub reads: `data/<split>-NNNNN-of-NNNNN.parquet`, with each
//! picture's file bytes and its boxes in pixels (see [`hf`]).
//!
//! A conversion copies each picture's bytes as they are. The YOLO side always gets a label file for each
//! picture, with each value in six decimals and one LF line end a box. So a round trip from YOLO to Parquet
//! and back gives the same files when the labels already have that form, and the same boxes to six decimals
//! when they do not.
//!
//! ```no_run
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! for summary in dataset_converter::yolo_to_parquet(Path::new("yolo-dataset"), Path::new("parquet-dataset"))? {
//!     println!("{summary}");
//! }
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};

pub mod hf;
pub mod yolo;

/// One picture of a dataset and the boxes on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// The picture's file name, with no folder part.
    pub file_name: String,
    /// The picture file's bytes, as they are on disk.
    pub bytes: Vec<u8>,
    /// The picture's width in pixels.
    pub width: u32,
    /// The picture's height in pixels.
    pub height: u32,
    /// The boxes on the picture.
    pub objects: Vec<Object>,
}

/// A box with its class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Object {
    /// The class, as an index into the dataset's class names.
    pub class: usize,
    /// Where the box is.
    pub bbox: Bbox,
}

/// A box in pixels, as its top-left corner and its size: the COCO convention that Hugging Face datasets use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    /// The left edge.
    pub x: f64,
    /// The top edge.
    pub y: f64,
    /// The width.
    pub width: f64,
    /// The height.
    pub height: f64,
}

/// A split of a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    /// The frames a model trains on.
    Train,
    /// The frames that measure a model while it trains.
    Validation,
    /// The frames held back for the final measurement.
    Test,
}

impl Split {
    /// Every split, in the order a dataset lists them.
    pub const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];

    /// The split's key in `data.yaml` and its folder name in the YOLO layout.
    #[must_use]
    pub fn yolo_name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "val",
            Self::Test => "test",
        }
    }

    /// The split's name on the Hugging Face Hub.
    #[must_use]
    pub fn hf_name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

/// What a conversion wrote for one split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// The split.
    pub split: Split,
    /// The pictures written.
    pub frames: usize,
    /// The boxes written.
    pub boxes: usize,
    /// The Parquet files the split takes.
    pub files: usize,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let files = if self.files == 1 { "file" } else { "files" };
        write!(
            f,
            "{}: {} frames, {} boxes, {} Parquet {files}",
            self.split.hf_name(),
            self.frames,
            self.boxes,
            self.files
        )
    }
}

/// Converts the YOLO dataset in `yolo_root` to Parquet files in `parquet_root`.
///
/// `yolo_root` holds `data.yaml`. `parquet_root` must be missing or empty; the files go to its `data/` folder.
/// A split with no pictures gets no file.
///
/// # Errors
///
/// Fails when `data.yaml` or a label file does not follow the YOLO layout, when a label file has no
/// picture, when a picture's size cannot be read, or when a file cannot be read or written.
pub fn yolo_to_parquet(yolo_root: &Path, parquet_root: &Path) -> Result<Vec<Summary>> {
    let dataset = yolo::Dataset::open(yolo_root)?;
    create_empty_dir(parquet_root)?;
    let mut summaries = Vec::new();
    for split in dataset.splits() {
        let entries = dataset.entries(split)?;
        let bytes = entries.iter().map(|entry| entry.size).sum();
        let ranges = hf::file_ranges(entries.len(), bytes);
        let files = ranges.len();
        let mut summary = Summary {
            split,
            frames: 0,
            boxes: 0,
            files,
        };
        for (index, range) in ranges.into_iter().enumerate() {
            let path = hf::file_path(parquet_root, split, index, files);
            let frames = entries[range]
                .iter()
                .map(|entry| dataset.read(entry))
                .inspect(|frame| {
                    if let Ok(frame) = frame {
                        summary.frames += 1;
                        summary.boxes += frame.objects.len();
                    }
                });
            hf::write(&path, dataset.names(), frames)
                .with_context(|| format!("cannot write {}", path.display()))?;
        }
        summaries.push(summary);
    }
    Ok(summaries)
}

/// Converts the Parquet files in `parquet_root/data/` to a YOLO dataset in `yolo_root`.
///
/// `yolo_root` must be missing or empty. Every file must carry the same class names.
///
/// # Errors
///
/// Fails when a file name does not follow `<split>-NNNNN-of-NNNNN.parquet`, when a file does not have the
/// columns and class names that [`hf`] describes, when a picture's file name has a folder part or repeats
/// in its split, or when a file cannot be read or written.
pub fn parquet_to_yolo(parquet_root: &Path, yolo_root: &Path) -> Result<Vec<Summary>> {
    let files = hf::files(parquet_root)?;
    let first = files
        .values()
        .flatten()
        .next()
        .context("no Parquet files found")?;
    let names = hf::names(first)?;
    let mut writer = yolo::Writer::create(yolo_root, names.clone())?;
    let mut summaries = Vec::new();
    for (split, paths) in &files {
        let mut summary = Summary {
            split: *split,
            frames: 0,
            boxes: 0,
            files: paths.len(),
        };
        for path in paths {
            let context = || format!("cannot convert {}", path.display());
            ensure!(
                hf::names(path).with_context(context)? == names,
                "{} has other class names than {}",
                path.display(),
                first.display()
            );
            for batch in hf::read(path).with_context(context)? {
                for frame in batch.with_context(context)? {
                    writer.write(*split, &frame)?;
                    summary.frames += 1;
                    summary.boxes += frame.objects.len();
                }
            }
        }
        summaries.push(summary);
    }
    writer.finish()?;
    Ok(summaries)
}

pub(crate) fn create_empty_dir(path: &Path) -> Result<()> {
    if path.exists() {
        let mut entries =
            fs::read_dir(path).with_context(|| format!("cannot read {}", path.display()))?;
        ensure!(entries.next().is_none(), "{} is not empty", path.display());
    }
    fs::create_dir_all(path).with_context(|| format!("cannot create {}", path.display()))
}
