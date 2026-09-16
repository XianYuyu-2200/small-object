"""Interactively capture the empty-table background and one object image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swallow_yolo.camera_profile import resolve_camera_profile
from swallow_yolo.mindvision import MindVisionCamera


def make_preview(frame, preview_width: int):
    height, width = frame.shape[:2]
    scale = min(1.0, float(preview_width) / max(height, width))
    if scale >= 1.0:
        return frame.copy()
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def write_jpeg(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"无法写入图片：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="采集空台背景图和目标物图。B 保存背景，O 保存目标物，Q 退出。")
    parser.add_argument("--camera-profile", type=Path, default=PROJECT_ROOT / "config" / "camera_profile.yaml")
    parser.add_argument("--background", type=Path, default=PROJECT_ROOT / "runs" / "measurement" / "background.jpg")
    parser.add_argument("--object", type=Path, default=PROJECT_ROOT / "runs" / "measurement" / "object_sample.jpg")
    parser.add_argument("--preview-width", type=int, default=1280)
    args = parser.parse_args()

    profile = yaml.safe_load(args.camera_profile.read_text(encoding="utf-8")) if args.camera_profile.exists() else {}
    values = resolve_camera_profile(profile or {}, {"preview_width": args.preview_width})
    args.background.parent.mkdir(parents=True, exist_ok=True)
    args.object.parent.mkdir(parents=True, exist_ok=True)

    status = "READY: B=background, O=object, Q=quit"
    background_saved = args.background.exists()
    object_saved = args.object.exists()

    try:
        with MindVisionCamera(
            values["sdk_path"],
            values["camera"],
            values["resolution_index"],
            values["frame_speed_index"],
            values["exposure_us"],
            values["gain_x"],
        ) as camera:
            window = "Measurement capture"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                frame = camera.read()
                preview = make_preview(frame, args.preview_width)
                cv2.putText(preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    f"Background: {'ready' if background_saved else 'missing'} | Object: {'ready' if object_saved else 'missing'}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(preview, "B background | O object | Q quit", (20, preview.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    break

                if key in (ord("b"), ord("B")):
                    write_jpeg(args.background, frame)
                    background_saved = True
                    status = f"SAVED BACKGROUND: {args.background.name}"

                if key in (ord("o"), ord("O")):
                    write_jpeg(args.object, frame)
                    object_saved = True
                    status = f"SAVED OBJECT: {args.object.name}"
    finally:
        cv2.destroyAllWindows()

    print(f"背景图：{args.background} ({'已保存' if background_saved else '未保存'})")
    print(f"目标物图：{args.object} ({'已保存' if object_saved else '未保存'})")


if __name__ == "__main__":
    main()
