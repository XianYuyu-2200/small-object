"""Run YOLO inference and conservative planar-size triage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from swallow_yolo.calibration import load_calibration
from swallow_yolo.geometry import estimate_size_mm
from swallow_yolo.risk import RiskConfig, RiskInput, classify_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="图片、视频或摄像头编号")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", default="data/calibration/calibration.json")
    parser.add_argument("--risk-rules", default="config/risk_rules.yaml")
    parser.add_argument("--classes", default="config/classes.yaml")
    parser.add_argument("--output", default="runs/inference")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请先执行 pip install -r requirements.txt") from error

    calibration = load_calibration(args.calibration)
    rules = yaml.safe_load(Path(args.risk_rules).read_text(encoding="utf-8"))
    class_data = yaml.safe_load(Path(args.classes).read_text(encoding="utf-8"))
    names = {int(key): value for key, value in class_data["names"].items()}
    config = RiskConfig(
        ingestible_max_mm=rules.get("ingestible_max_mm"),
        buffer_mm=float(rules.get("buffer_mm", 0)),
        min_confidence=float(rules.get("min_confidence", 0.7)),
        manual_review_labels=frozenset(rules.get("manual_review_labels", [])),
    )
    model = YOLO(args.model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source) if isinstance(source, int) or Path(str(source)).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else None
    if capture is not None and not capture.isOpened():
        raise RuntimeError(f"无法打开输入源：{args.source}")
    image = None if capture else cv2.imread(str(source))
    if capture is None and image is None:
        raise RuntimeError(f"无法读取图片：{args.source}")
    while True:
        ok, raw_frame = capture.read() if capture is not None else (image is not None, image)
        if not ok:
            break
        if not calibration.validate_frame(raw_frame):
            raise RuntimeError("输入分辨率与 calibration.json 不匹配")
        frame = calibration.undistort(raw_frame)
        result = model.predict(source=frame, verbose=False)[0]
        records = []
        for box, confidence, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            width_mm, height_mm = estimate_size_mm(tuple(box), calibration.table_homography_px_to_mm)
            label = names.get(int(class_id), f"class_{int(class_id):02d}")
            triage = classify_risk(config, RiskInput(label, float(confidence), max(width_mm, height_mm), min(width_mm, height_mm)))
            records.append({"label": label, "confidence": float(confidence), "width_mm": width_mm, "height_mm": height_mm, "risk": triage.category, "reason": triage.reason})
        annotated = result.plot()
        cv2.imwrite(str(output / "latest.jpg"), annotated)
        (output / "latest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(records, ensure_ascii=False))
        if capture is None:
            break
        cv2.imshow("inference", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if capture is not None:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
