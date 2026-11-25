//! The Hugging Face Parquet layout, as the `datasets` library writes it.
//!
//! A dataset is a folder with `data/<split>-NNNNN-of-NNNNN.parquet`, where `<split>` is `train`,
//! `validation` or `test`. Each row is one picture:
//!
//! | Column | Arrow type | `datasets` feature |
//! | --- | --- | --- |
//! | `image` | `struct<bytes: binary, path: string>` | `Image()` |
//! | `width`, `height` | `int32` | `Value("int32")` |
//! | `objects` | `struct<bbox: list<fixed_size_list<double>[4]>, category: list<int64>>` | `{"bbox": List(List(Value("float64"), length=4)), "category": List(ClassLabel(names))}` |
//!
//! `image.bytes` holds the picture file as it is, and `image.path` its file name. Each box is
//! `[x, y, width, height]` in pixels, from the picture's top-left corner. The `huggingface` key of the file
//! metadata holds these features as JSON, which is how `datasets` and the Hub know the `Image` column and
//! the class names. Every field is nullable: `datasets` ignores the stored features of a field that differs
//! from them, nullability included.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use arrow_array::cast::AsArray as _;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, BinaryArray, FixedSizeListArray, Float64Array, Int32Array, Int64Array, ListArray,
    RecordBatch, StringArray, StructArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, Encoding};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use serde_json::json;

use crate::{Bbox, Frame, Object, Split};

/// The rows of a row group: `datasets` writes image datasets in batches of 100 rows, and the Hub docs
/// recommend row groups of that size for pictures.
const ROWS_PER_GROUP: usize = 100;

/// The default `max_shard_size` of `datasets`' `push_to_hub`, 500 MB.
const FILE_BYTES: u64 = 500_000_000;

/// The file metadata key that holds the `datasets` features.
const FEATURES_KEY: &str = "huggingface";

/// The rows of each file, for a split of `rows` frames with `bytes` of pictures.
///
/// As `push_to_hub` does, the split takes `bytes / 500 MB + 1` files, rounded down, and the rows spread
/// over them in order, in shares that differ by one row at most. A file can hold more than 500 MB when the
/// pictures differ in size. A split with no rows takes no file.
#[must_use]
pub fn file_ranges(rows: usize, bytes: u64) -> Vec<Range<usize>> {
    let count = usize::try_from(bytes / FILE_BYTES + 1).map_or(rows, |count| count.min(rows));
    if count == 0 {
        return Vec::new();
    }
    let (share, rest) = (rows / count, rows % count);
    (0..count)
        .map(|index| {
            let start = index * share + index.min(rest);
            start..start + share + usize::from(index < rest)
        })
        .collect()
}

/// The path of a split's file `index` of `count` under `root`.
#[must_use]
pub fn file_path(root: &Path, split: Split, index: usize, count: usize) -> PathBuf {
    root.join("data").join(format!(
        "{}-{index:05}-of-{count:05}.parquet",
        split.hf_name()
    ))
}

/// Writes `frames` to a new Parquet file at `path`.
///
/// # Errors
///
/// Fails when `path` exists, when a frame fails to read, when a box's class has no name, when a picture is
/// wider or taller than `i32::MAX`, or when the file cannot be written.
pub fn write(
    path: &Path,
    names: &[String],
    frames: impl IntoIterator<Item = Result<Frame>>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let schema = Arc::new(schema(names));
    let file =
        File::create_new(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties(&schema)))?;
    let mut batch = Vec::with_capacity(ROWS_PER_GROUP);
    for frame in frames {
        let frame = frame?;
        let unnamed = frame
            .objects
            .iter()
            .find(|object| object.class >= names.len());
        ensure!(
            unnamed.is_none(),
            "{} has a box of a class with no name",
            frame.file_name
        );
        batch.push(frame);
        if batch.len() == ROWS_PER_GROUP {
            writer.write(&record_batch(&schema, &batch)?)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        writer.write(&record_batch(&schema, &batch)?)?;
    }
    writer.close()?;
    Ok(())
}

/// The Parquet files under `root/data/`, by split, in file order.
///
/// # Errors
///
/// Fails when `root/data/` cannot be read, when a `.parquet` file name does not follow
/// `<split>-NNNNN-of-NNNNN.parquet`, or when a split's files are not all there.
pub fn files(root: &Path) -> Result<BTreeMap<Split, Vec<PathBuf>>> {
    let dir = root.join("data");
    let mut found: BTreeMap<Split, (usize, BTreeMap<usize, PathBuf>)> = BTreeMap::new();
    for item in fs::read_dir(&dir).with_context(|| format!("cannot read {}", dir.display()))? {
        let path = item?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".parquet") {
            continue;
        }
        let (split, index, count) = parse_file_name(name).with_context(|| {
            format!(
                "{} does not follow <split>-NNNNN-of-NNNNN.parquet",
                path.display()
            )
        })?;
        let (expected, paths) = found.entry(split).or_insert((count, BTreeMap::new()));
        ensure!(
            *expected == count,
            "the {} files do not agree on their count",
            split.hf_name()
        );
        paths.insert(index, path);
    }
    let mut files = BTreeMap::new();
    for (split, (count, paths)) in found {
        ensure!(
            paths.keys().copied().eq(0..count),
            "the {} split has files missing: {count} expected",
            split.hf_name()
        );
        files.insert(split, paths.into_values().collect());
    }
    Ok(files)
}

/// The class names that a file's `huggingface` metadata gives `objects.category`.
///
/// # Errors
///
/// Fails when the file cannot be read, or when its metadata has no `ClassLabel` names for `objects.category`.
pub fn names(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let json = builder
        .schema()
        .metadata()
        .get(FEATURES_KEY)
        .context("the file has no `huggingface` metadata")?;
    let features: serde_json::Value =
        serde_json::from_str(json).context("the `huggingface` metadata is not JSON")?;
    let label = &features["info"]["features"]["objects"]["category"]["feature"];
    ensure!(
        label["_type"] == "ClassLabel",
        "the features do not give `objects.category` as a List of ClassLabel"
    );
    let names = label["names"]
        .as_array()
        .context("the ClassLabel of `objects.category` has no names")?;
    names
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_owned)
                .context("a class name is not text")
        })
        .collect()
}

/// Reads the frames of a Parquet file, one batch of up to 100 frames at a time.
///
/// # Errors
///
/// Fails when the file cannot be opened. Each batch fails when a column is missing, has another type, or
/// has a null value, or when a row has more boxes than categories or the other way round.
pub fn read(path: &Path) -> Result<impl Iterator<Item = Result<Vec<Frame>>>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(ROWS_PER_GROUP)
        .build()?;
    Ok(reader.map(|batch| frames(&batch?)))
}

fn frames(batch: &RecordBatch) -> Result<Vec<Frame>> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .with_context(|| format!("there is no `{name}` column"))
    };
    let image = column("image")?
        .as_struct_opt()
        .context("`image` is not a struct")?;
    let bytes = member(image, "image", "bytes")?
        .as_binary_opt::<i32>()
        .context("`image.bytes` is not binary")?;
    let paths = member(image, "image", "path")?
        .as_string_opt::<i32>()
        .context("`image.path` is not a string")?;
    let widths = column("width")?
        .as_primitive_opt::<Int32Type>()
        .context("`width` is not int32")?;
    let heights = column("height")?
        .as_primitive_opt::<Int32Type>()
        .context("`height` is not int32")?;
    let objects = column("objects")?
        .as_struct_opt()
        .context("`objects` is not a struct")?;
    let bbox = member(objects, "objects", "bbox")?
        .as_list_opt::<i32>()
        .context("`objects.bbox` is not a list")?;
    let category = member(objects, "objects", "category")?
        .as_list_opt::<i32>()
        .context("`objects.category` is not a list")?;
    let boxes = bbox
        .values()
        .as_fixed_size_list_opt()
        .context("a box is not a fixed-size list")?;
    ensure!(
        boxes.value_length() == 4,
        "a box has {} values, not 4",
        boxes.value_length()
    );
    let coords = boxes
        .values()
        .as_primitive_opt::<Float64Type>()
        .context("box values are not float64")?;
    let classes = category
        .values()
        .as_primitive_opt::<Int64Type>()
        .context("categories are not int64")?;
    let arrays: [(&str, &dyn Array); 11] = [
        ("image", image),
        ("image.bytes", bytes),
        ("image.path", paths),
        ("width", widths),
        ("height", heights),
        ("objects", objects),
        ("objects.bbox", bbox),
        ("objects.category", category),
        ("a box", boxes),
        ("a box value", coords),
        ("a category", classes),
    ];
    if let Some((name, _)) = arrays.iter().find(|(_, array)| array.null_count() > 0) {
        bail!("{name} has a null value");
    }
    let (box_offsets, class_offsets) = (bbox.value_offsets(), category.value_offsets());
    let mut frames = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let file_name = paths.value(row);
        let box_range = usize::try_from(box_offsets[row])?..usize::try_from(box_offsets[row + 1])?;
        let class_range =
            usize::try_from(class_offsets[row])?..usize::try_from(class_offsets[row + 1])?;
        ensure!(
            box_range.len() == class_range.len(),
            "{file_name} has {} boxes but {} categories",
            box_range.len(),
            class_range.len()
        );
        let mut row_objects = Vec::with_capacity(box_range.len());
        for (index, class) in box_range.zip(class_range) {
            let start = boxes.value_offset_at(index);
            let &[x, y, width, height] = &coords.values()[start..start + 4] else {
                unreachable!("a fixed-size list of 4 holds 4 values");
            };
            let class = usize::try_from(classes.value(class))
                .with_context(|| format!("{file_name} has a negative category"))?;
            row_objects.push(Object {
                class,
                bbox: Bbox {
                    x,
                    y,
                    width,
                    height,
                },
            });
        }
        frames.push(Frame {
            file_name: file_name.to_owned(),
            bytes: bytes.value(row).to_vec(),
            width: u32::try_from(widths.value(row))
                .with_context(|| format!("{file_name} has a negative width"))?,
            height: u32::try_from(heights.value(row))
                .with_context(|| format!("{file_name} has a negative height"))?,
            objects: row_objects,
        });
    }
    Ok(frames)
}

fn member<'a>(array: &'a StructArray, parent: &str, name: &str) -> Result<&'a Arc<dyn Array>> {
    array
        .column_by_name(name)
        .with_context(|| format!("`{parent}` has no `{name}` field"))
}

fn parse_file_name(name: &str) -> Option<(Split, usize, usize)> {
    let (rest, count) = name.strip_suffix(".parquet")?.rsplit_once("-of-")?;
    let (split, index) = rest.rsplit_once('-')?;
    let split = Split::ALL
        .into_iter()
        .find(|candidate| candidate.hf_name() == split)?;
    let (index, count) = (index.parse().ok()?, count.parse().ok()?);
    (index < count).then_some((split, index, count))
}

fn record_batch(schema: &SchemaRef, frames: &[Frame]) -> Result<RecordBatch> {
    let image = StructArray::try_new(
        image_fields(),
        vec![
            Arc::new(BinaryArray::from_iter_values(
                frames.iter().map(|frame| &frame.bytes),
            )),
            Arc::new(StringArray::from_iter_values(
                frames.iter().map(|frame| &frame.file_name),
            )),
        ],
        None,
    )?;
    let size = |pick: fn(&Frame) -> u32| -> Result<Int32Array> {
        let sizes = frames
            .iter()
            .map(|frame| i32::try_from(pick(frame)).context("a picture is too large"));
        Ok(Int32Array::from(sizes.collect::<Result<Vec<_>>>()?))
    };
    let objects = || frames.iter().flat_map(|frame| &frame.objects);
    let coords = Float64Array::from_iter_values(objects().flat_map(|object| {
        [
            object.bbox.x,
            object.bbox.y,
            object.bbox.width,
            object.bbox.height,
        ]
    }));
    let classes = Int64Array::from(
        objects()
            .map(|object| i64::try_from(object.class))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let offsets = OffsetBuffer::<i32>::from_lengths(frames.iter().map(|frame| frame.objects.len()));
    let boxes = FixedSizeListArray::try_new(element(DataType::Float64), 4, Arc::new(coords), None)?;
    let bbox = ListArray::try_new(element(box_type()), offsets.clone(), Arc::new(boxes), None)?;
    let category = ListArray::try_new(element(DataType::Int64), offsets, Arc::new(classes), None)?;
    let objects = StructArray::try_new(
        object_fields(),
        vec![Arc::new(bbox), Arc::new(category)],
        None,
    )?;
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(image),
        Arc::new(size(|frame| frame.width)?),
        Arc::new(size(|frame| frame.height)?),
        Arc::new(objects),
    ];
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

fn schema(names: &[String]) -> Schema {
    let fields = vec![
        Field::new("image", DataType::Struct(image_fields()), true),
        Field::new("width", DataType::Int32, true),
        Field::new("height", DataType::Int32, true),
        Field::new("objects", DataType::Struct(object_fields()), true),
    ];
    Schema::new_with_metadata(
        fields,
        HashMap::from([(FEATURES_KEY.to_owned(), features(names).to_string())]),
    )
}

/// The features JSON, in the form and key order that `datasets` writes.
fn features(names: &[String]) -> serde_json::Value {
    json!({"info": {"features": {
        "image": {"_type": "Image"},
        "width": {"dtype": "int32", "_type": "Value"},
        "height": {"dtype": "int32", "_type": "Value"},
        "objects": {
            "bbox": {"feature": {"feature": {"dtype": "float64", "_type": "Value"}, "length": 4, "_type": "List"}, "_type": "List"},
            "category": {"feature": {"names": names, "_type": "ClassLabel"}, "_type": "List"},
        },
    }}})
}

fn properties(schema: &Schema) -> WriterProperties {
    let features = schema.metadata()[FEATURES_KEY].clone();
    let mut builder = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
        .set_compression(Compression::SNAPPY)
        // parquet-rs keeps schema metadata inside `ARROW:schema`; `datasets` also writes the key at file level.
        .set_key_value_metadata(Some(vec![KeyValue::new(FEATURES_KEY.to_owned(), features)]));
    // Picture files are compressed already, so `datasets` stores their bytes and names plain and uncompressed.
    for leaf in ["bytes", "path"] {
        let column = ColumnPath::from(vec!["image".to_owned(), leaf.to_owned()]);
        builder = builder
            .set_column_compression(column.clone(), Compression::UNCOMPRESSED)
            .set_column_dictionary_enabled(column.clone(), false)
            .set_column_encoding(column.clone(), Encoding::PLAIN)
            .set_column_statistics_enabled(column, EnabledStatistics::None);
    }
    builder.build()
}

fn image_fields() -> Fields {
    Fields::from(vec![
        Field::new("bytes", DataType::Binary, true),
        Field::new("path", DataType::Utf8, true),
    ])
}

fn object_fields() -> Fields {
    Fields::from(vec![
        Field::new("bbox", DataType::List(element(box_type())), true),
        Field::new("category", DataType::List(element(DataType::Int64)), true),
    ])
}

fn box_type() -> DataType {
    DataType::FixedSizeList(element(DataType::Float64), 4)
}

/// A list item field, named `element` as in the Parquet spec for lists.
fn element(data_type: DataType) -> FieldRef {
    Arc::new(Field::new("element", data_type, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_cover_every_row_once_in_even_shares() {
        for (rows, bytes) in [
            (0, 0),
            (1, 0),
            (5, 3 * FILE_BYTES),
            (3, 9 * FILE_BYTES),
            (3608, 475_590_583),
            (3608, 2_400_000_000),
        ] {
            let ranges = file_ranges(rows, bytes);
            let flat: Vec<usize> = ranges.iter().cloned().flatten().collect();
            assert_eq!(
                flat,
                (0..rows).collect::<Vec<_>>(),
                "{rows} rows, {bytes} bytes"
            );
            let (least, most) = (
                ranges.iter().map(Range::len).min(),
                ranges.iter().map(Range::len).max(),
            );
            assert!(
                least.is_none_or(|least| least > 0 && most.unwrap() - least <= 1),
                "{ranges:?}"
            );
        }
        // push_to_hub: int(1.5 GB / 500 MB) + 1 files.
        assert_eq!(file_ranges(5, 3 * FILE_BYTES), [0..2, 2..3, 3..4, 4..5]);
    }
}
