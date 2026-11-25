# dataset-converter

Converts object-detection datasets between the Ultralytics YOLO layout and the Parquet layout that the Hugging Face `datasets` library writes and the Hub reads.

A conversion copies each picture's file bytes as they are. A conversion to YOLO writes a label file for each picture, with each value in six decimals, single spaces and one LF line end a box. So a round trip from YOLO to Parquet and back gives the same files when the labels already have that form, and the same boxes to six decimals when they do not.

## Install

```bash
cargo install --path .
```

## Use

```bash
dataset-converter --from yolo --to parquet path/to/yolo-dataset path/to/parquet-dataset
```

```bash
dataset-converter --from parquet --to yolo path/to/parquet-dataset path/to/yolo-dataset
```

`--from` names the format of the input folder, and `--to` the format to write. The formats are `yolo` and `parquet`. The output folder must be missing or empty. A conversion prints the frames, boxes and Parquet files of each split.

## YOLO layout

The YOLO dataset is the folder that holds `data.yaml`:

```yaml
train: images/train
val: images/val
names:
  0: ball
  1: target
```

- `train`, `val` and `test` each name one folder of pictures, relative to the folder of `data.yaml`. The tool does not read a `path` key or a list file of pictures.
- `names` maps each class index to its name, or lists the names.
- The boxes of `images/<split>/<name>.jpg` are in `labels/<split>/<name>.txt`, one box a line as `class cx cy w h`: the class index, then the centre and size of the box in fractions of the picture. A picture with no label file has no boxes.
- The pictures can be BMP, JPEG, PNG, TIFF or WebP files. Only JPEG and PNG pictures are tested. Hidden files and other files are skipped.

A conversion to YOLO writes the splits to `images/train`, `images/val` and `images/test`, a label file for every picture (empty for a picture with no boxes), and a `data.yaml` with those folders and the class names.

## Parquet layout

The Parquet dataset is a folder with `data/<split>-NNNNN-of-NNNNN.parquet`, where `<split>` is `train`, `validation` or `test`. The YOLO split `val` is the Parquet split `validation`. Each row is one picture:

| Column | Arrow type | `datasets` feature |
| --- | --- | --- |
| `image` | `struct<bytes: binary, path: string>` | `Image()` |
| `width`, `height` | `int32` | `Value("int32")` |
| `objects` | `struct<bbox: list<fixed_size_list<double>[4]>, category: list<int64>>` | `{"bbox": List(List(Value("float64"), length=4)), "category": List(ClassLabel(names))}` |

- `image.bytes` holds the picture file, and `image.path` its file name.
- Each box is `[x, y, width, height]` in pixels from the top-left corner of the picture, the COCO convention of the Hugging Face object detection guide.
- The `huggingface` key of the file metadata holds the features, so `datasets` loads `image` as pictures and `objects.category` with its class names.
- A split takes one file for each full 500 MB of pictures, plus one, as `datasets`' `push_to_hub` counts them with its default shard size. The frames spread over the files in order, in equal shares, so a file can hold more than 500 MB when the pictures differ in size.
- Each file has row groups of 100 rows.

A conversion from Parquet reads this layout only, as this tool or `datasets` writes it.

## Limits

- Only axis-aligned boxes. Segmentation, keypoints and oriented boxes are not read.
- The picture size comes from the file header. An EXIF orientation is not applied.
- A conversion to YOLO writes each label value with six decimals.

## Development

```bash
cargo fmt --check
```

```bash
cargo clippy --all-targets -- -D warnings
```

```bash
cargo test
```

The tests use the fixtures in `tests/fixtures/`. `yolo/` is a small YOLO dataset, and `hf/` holds the same frames as Parquet files that `datasets` itself wrote. To write them again:

```bash
uv run --with datasets --with pillow tests/fixtures/make_fixtures.py
```
