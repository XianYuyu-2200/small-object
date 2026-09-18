#!/usr/bin/env python3
"""Create brightness, darkness, and blur variants for a YOLO dataset.

Only photometric changes are applied, so detection/segmentation labels remain
valid without editing. By default the train split is augmented and all splits
are copied into a standalone output dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METHODS = ("bright", "dark", "blur")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="亮度、暗度和模糊数据增强（标签保持不变）")
    parser.add_argument("--input", type=Path, required=True, help="原始 YOLO 数据集根目录")
    parser.add_argument("--output", type=Path, required=True, help="新的增强数据集目录")
    parser.add_argument(
        "--augment-splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train",),
        help="需要增强的 split，默认只增强 train",
    )
    parser.add_argument(
        "--copy-splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
        help="复制到新数据集的 split，默认全部复制",
    )
    parser.add_argument("--variants-per-method", type=int, default=1, help="每种方式生成几套变体，默认 1")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="增强图片 JPEG 质量，默认 95")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--limit", type=int, default=0, help="每个 split 最多处理多少张原图，0 表示全部")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出目录")
    return parser.parse_args()


def safe_prepare_output(input_root: Path, output_root: Path, overwrite: bool) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise RuntimeError("输出目录不能等于输入目录，也不能放在原始数据集内部")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output_root}；如需重建请加 --overwrite")
        if output_root == Path(output_root.anchor):
            raise RuntimeError(f"拒绝删除磁盘根目录：{output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_split(source_root: Path, output_root: Path, split: str, limit: int = 0) -> int:
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"缺少 split 目录：{image_dir} 或 {label_dir}")
    images = image_files(image_dir)
    if limit > 0:
        images = images[:limit]
    label_names = {f"{image.stem}.txt" for image in images}
    labels = sorted(label_dir / name for name in label_names if (label_dir / name).is_file())
    output_image_dir = output_root / split / "images"
    output_label_dir = output_root / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        copy_file(image, output_image_dir / image.name)
        label = label_dir / f"{image.stem}.txt"
        if label.is_file():
            copy_file(label, output_label_dir / label.name)
    for label in labels:
        output_label = output_label_dir / label.name
        if not output_label.exists():
            copy_file(label, output_label)
    return len(images)


def apply_gamma(image: np.ndarray, gamma: float, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    adjusted = cv2.LUT(image, table)
    if alpha != 1.0 or beta != 0.0:
        adjusted = cv2.convertScaleAbs(adjusted, alpha=alpha, beta=beta)
    return adjusted


def augment_image(image: np.ndarray, method: str, rng: random.Random) -> np.ndarray:
    if method == "bright":
        return apply_gamma(image, rng.uniform(0.72, 0.88), alpha=rng.uniform(1.00, 1.08), beta=rng.uniform(4.0, 14.0))
    if method == "dark":
        return apply_gamma(image, rng.uniform(1.14, 1.34), alpha=rng.uniform(0.88, 0.98), beta=rng.uniform(-8.0, -2.0))
    if method == "blur":
        kernel = rng.choice((5, 7, 9))
        sigma = rng.uniform(0.8, 1.8)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    raise ValueError(f"未知增强方式：{method}")


def augment_split(
    source_root: Path,
    output_root: Path,
    split: str,
    variants_per_method: int,
    jpeg_quality: int,
    limit: int,
    rng: random.Random,
) -> tuple[int, int]:
    source_image_dir = source_root / split / "images"
    source_label_dir = source_root / split / "labels"
    output_image_dir = output_root / split / "images"
    output_label_dir = output_root / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    images = image_files(source_image_dir)
    if limit > 0:
        images = images[:limit]
    generated = 0
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        label_path = source_label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少标签：{label_path}")
        for method in METHODS:
            for variant_index in range(1, variants_per_method + 1):
                suffix = f"__{method}_{variant_index:02d}"
                output_image = output_image_dir / f"{image_path.stem}{suffix}.jpg"
                output_label = output_label_dir / f"{image_path.stem}{suffix}.txt"
                augmented = augment_image(image, method, rng)
                ok = cv2.imwrite(str(output_image), augmented, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    raise RuntimeError(f"无法写入增强图片：{output_image}")
                copy_file(label_path, output_label)
                generated += 1
        if index % 50 == 0 or index == len(images):
            print(f"[{split}] {index}/{len(images)}，已生成 {generated} 张增强图")
    return len(images), generated


def write_dataset_yaml(source_root: Path, output_root: Path) -> None:
    source_yaml = source_root / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"缺少 data.yaml：{source_yaml}")
    raw = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    if "names" not in raw:
        raise ValueError("data.yaml 缺少 names")
    data = {
        "path": output_root.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": int(raw.get("nc", len(raw["names"]))),
        "names": raw["names"],
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.variants_per_method < 1:
        raise ValueError("--variants-per-method 必须大于等于 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 1 到 100 之间")
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not (input_root / "data.yaml").is_file():
        raise FileNotFoundError(f"输入目录不是有效 YOLO 数据集：{input_root}")
    safe_prepare_output(input_root, output_root, args.overwrite)

    rng = random.Random(args.seed)
    for split in args.copy_splits:
        copied = copy_split(input_root, output_root, split, args.limit if split in args.augment_splits else 0)
        print(f"[{split}] 已复制原图 {copied} 张")
    for split in args.augment_splits:
        originals, generated = augment_split(
            input_root,
            output_root,
            split,
            args.variants_per_method,
            args.jpeg_quality,
            args.limit,
            rng,
        )
        print(f"[{split}] 增强完成：原图 {originals} 张，新增 {generated} 张")
    write_dataset_yaml(input_root, output_root)
    print(f"增强数据集已生成：{output_root}")


if __name__ == "__main__":
    main()
