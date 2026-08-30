"""Train a YOLO detector after dataset and label review."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def make_dataset_config(dataset_path: str, classes_path: str, output_path: str) -> str:
    """Combine the single class registry with Ultralytics' required dataset YAML."""
    dataset = yaml.safe_load(Path(dataset_path).read_text(encoding="utf-8"))
    classes = yaml.safe_load(Path(classes_path).read_text(encoding="utf-8"))
    dataset["names"] = classes["names"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="config/dataset.yaml")
    parser.add_argument("--classes", default="config/classes.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请先执行 pip install -r requirements.txt") from error
    model = YOLO(args.model)
    data = make_dataset_config(args.data, args.classes, "runs/generated_dataset.yaml")
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, device=args.device, project=args.project)


if __name__ == "__main__":
    main()
