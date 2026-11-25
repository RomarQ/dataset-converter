"""Writes the test fixtures: a small YOLO dataset in yolo/, and the same frames as Parquet in hf/.

    uv run --with datasets --with pillow tests/fixtures/make_fixtures.py

The pictures are plain colour fields. The `datasets` library itself writes the Parquet files, and this script
converts the boxes to pixels on its own, so a test that reads hf/ checks the tool against `datasets`.
"""

import io
import shutil
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

HERE = Path(__file__).parent
NAMES = ["ball", "target"]
# split folder, file name, size, colour, label lines
FRAMES = [
    ("train", "a.jpg", (64, 32), (200, 40, 40), ["0 0.500000 0.500000 0.500000 0.500000", "1 0.250000 0.125000 0.100000 0.200000"]),
    ("train", "b.png", (32, 32), (40, 200, 40), []),
    ("val", "c.jpg", (48, 64), (40, 40, 200), ["1 0.312500 0.718750 0.125000 0.093750"]),
]
HF_SPLITS = {"train": "train", "val": "validation"}


def picture(name: str, size: tuple, colour: tuple) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", size, colour).save(out, "PNG" if name.endswith(".png") else "JPEG")
    return out.getvalue()


def pixels(line: str, width: int, height: int) -> tuple:
    """A YOLO line as its class and a COCO box: top-left corner and size in pixels."""
    cls, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    return int(cls), [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]


for folder in ("yolo", "hf"):
    shutil.rmtree(HERE / folder, ignore_errors=True)
yolo = HERE / "yolo"
(yolo / "data.yaml").parent.mkdir(parents=True)
(yolo / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: ball\n  1: target\n")
rows = {}
for split, name, size, colour, lines in FRAMES:
    data = picture(name, size, colour)
    for kind, file, content in (("images", name, data), ("labels", Path(name).stem + ".txt", "".join(f"{line}\n" for line in lines).encode())):
        path = yolo / kind / split / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    boxes = [pixels(line, *size) for line in lines]
    row = rows.setdefault(HF_SPLITS[split], {"image": [], "width": [], "height": [], "objects": []})
    row["image"].append({"bytes": data, "path": name})
    row["width"].append(size[0])
    row["height"].append(size[1])
    row["objects"].append({"bbox": [box for _, box in boxes], "category": [cls for cls, _ in boxes]})

features = Features(
    {
        "image": Image(),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence({"bbox": Sequence(Value("float64"), length=4), "category": ClassLabel(names=NAMES)}),
    }
)
for split, columns in rows.items():
    path = HERE / "hf" / "data" / f"{split}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(columns, features=features).to_parquet(str(path))
