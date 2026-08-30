"""Validate a YOLO dataset's image/label pairs and class IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def check_split(root: Path, split: str, class_count: int) -> list[str]:
    errors: list[str] = []
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    if not image_dir.exists():
        return [f"{split}: missing {image_dir}"]
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"{split}: missing label for {image_path.name}")
            continue
        try:
            width, height = Image.open(image_path).size
            if width <= 0 or height <= 0:
                errors.append(f"{split}: invalid image {image_path.name}")
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if len(values) != 5:
                    errors.append(f"{label_path}:{line_no}: expected 5 values")
                    continue
                class_id = int(values[0])
                coords = [float(v) for v in values[1:]]
                if not 0 <= class_id < class_count:
                    errors.append(f"{label_path}:{line_no}: class id {class_id} outside 0..{class_count - 1}")
                if any(value < 0 or value > 1 for value in coords):
                    errors.append(f"{label_path}:{line_no}: coordinates must be normalized to 0..1")
        except (OSError, ValueError) as error:
            errors.append(f"{image_path}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dataset")
    parser.add_argument("--classes", type=int, default=31)
    args = parser.parse_args()
    errors = [error for split in ("train", "val", "test") for error in check_split(Path(args.root), split, args.classes)]
    if errors:
        print("数据集校验失败：")
        print("\n".join(errors))
        raise SystemExit(1)
    print("数据集校验通过")


if __name__ == "__main__":
    main()
