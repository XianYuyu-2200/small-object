"""Run YOLO inference and map each detected class to a true/false decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from swallow_yolo.mindvision import MindVisionCamera
from swallow_yolo.camera_profile import resolve_camera_profile
from swallow_yolo.swallowability import classify_swallowability, load_swallowability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="图片、视频或摄像头编号")
    parser.add_argument("--backend", choices=("opencv", "mindvision"))
    parser.add_argument("--sdk-path", help="迈德威视 SDK 根目录；mindvision 后端必填")
    parser.add_argument("--resolution-index", type=int, help="迈德威视 SDK 预设分辨率索引")
    parser.add_argument("--frame-speed-index", type=int, help="迈德威视帧速档位；2 对应 High")
    parser.add_argument("--exposure-us", type=float, help="手动曝光（微秒）；提供该值会关闭自动曝光")
    parser.add_argument("--gain-x", type=float, help="模拟增益倍数")
    parser.add_argument("--profile", default="config/camera_profile.yaml", help="相机默认参数文件")
    parser.add_argument("--model-max-edge", type=int, default=1280, help="模型输入最长边")
    parser.add_argument("--model", required=True)
    parser.add_argument("--swallowability", default="config/swallowability.yaml", help="类别到 true/false 的吞咽属性配置")
    parser.add_argument("--classes", help="可选外部类别映射；默认使用 best.pt 内置类别")
    parser.add_argument("--output", default="runs/inference")
    args = parser.parse_args()
    profile = yaml.safe_load(Path(args.profile).read_text(encoding="utf-8")) if Path(args.profile).exists() else {}
    camera_values = resolve_camera_profile(profile, vars(args))
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请先执行 pip install -r requirements.txt") from error

    swallowability, min_confidence = load_swallowability(args.swallowability)
    model = YOLO(args.model)
    names = {int(key): value for key, value in model.names.items()} if isinstance(model.names, dict) else dict(enumerate(model.names))
    if args.classes:
        class_data = yaml.safe_load(Path(args.classes).read_text(encoding="utf-8"))
        names = {int(key): value for key, value in class_data["names"].items()}
    missing_labels = sorted(set(names.values()) - set(swallowability))
    if missing_labels:
        print(f"提示：以下类别未配置吞咽属性，将输出无法判断：{', '.join(missing_labels)}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = int(args.source) if str(args.source).isdigit() else args.source
    mindvision = None
    if camera_values["backend"] == "mindvision":
        if not camera_values["sdk_path"]:
            raise SystemExit("使用 --backend mindvision 时必须提供 --sdk-path G:\\mindvision")
        mindvision = MindVisionCamera(camera_values["sdk_path"], int(args.source), camera_values["resolution_index"], camera_values["frame_speed_index"], camera_values["exposure_us"], camera_values["gain_x"])
        print(f"已连接迈德威视相机：{mindvision.device_name}")
    source_path = Path(str(source))
    image_files = sorted(p for p in source_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) if source_path.is_dir() else []
    capture = None if mindvision else (cv2.VideoCapture(source) if isinstance(source, int) or source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else None)
    if capture is not None and not capture.isOpened():
        raise RuntimeError(f"无法打开输入源：{args.source}")
    image = None if (capture or mindvision or image_files) else cv2.imread(str(source))
    if capture is None and mindvision is None and not image_files and image is None:
        raise RuntimeError(f"无法读取图片：{args.source}")
    file_index = 0
    while True:
        if capture is not None:
            ok, raw_frame = capture.read()
        elif mindvision is not None:
            ok, raw_frame = True, mindvision.read()
        elif image_files:
            ok = file_index < len(image_files)
            raw_frame = cv2.imread(str(image_files[file_index])) if ok else None
            file_index += 1
        else:
            ok, raw_frame = image is not None, image
        if not ok:
            break
        frame = raw_frame
        scale = min(1.0, args.model_max_edge / max(frame.shape[:2]))
        model_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        result = model.predict(source=model_frame, verbose=False)[0]
        records = []
        for index, (box, confidence, class_id) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())):
            label = names.get(int(class_id), f"class_{int(class_id):02d}")
            decision = classify_swallowability(swallowability, label, float(confidence), min_confidence)
            records.append({"label": label, "can_swallow": decision.can_swallow, "swallowability": decision.text})
        annotated = result.plot()
        for record, box in zip(records, result.boxes.xyxy.tolist()):
            x1, y1, _, _ = (int(value) for value in box)
            text = f"{record['swallowability']} ({record['label']})"
            cv2.putText(annotated, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / "latest.jpg"), annotated)
        (output / "latest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(records, ensure_ascii=False))
        if capture is None and mindvision is None:
            break
        cv2.imshow("inference", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if capture is not None:
        capture.release()
        cv2.destroyAllWindows()
    if mindvision is not None:
        mindvision.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
