"""Interactively capture chessboard calibration photos from the MindVision camera."""

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


def parse_pattern(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pattern 必须形如 7x10")
    try:
        columns, rows = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("pattern 必须包含两个整数，例如 7x10") from error
    if columns < 2 or rows < 2:
        raise argparse.ArgumentTypeError("pattern 的行列数都必须大于 1")
    return columns, rows


def next_calibration_index(output_dir: Path) -> int:
    indices = []
    for path in output_dir.glob("calib_*.jpg"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def make_preview(frame, preview_width: int):
    height, width = frame.shape[:2]
    scale = min(1.0, float(preview_width) / max(height, width))
    if scale >= 1.0:
        return frame.copy()
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def detect_chessboard(frame, pattern: tuple[int, int], detection_width: int) -> bool:
    sample = make_preview(frame, detection_width)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, _ = cv2.findChessboardCorners(gray, pattern, flags)
    return bool(found)


def write_jpeg(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"无法写入图片：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="采集棋盘格标定照片。空格保存标定图，T 保存台面参考图，Q 退出。")
    parser.add_argument("--camera-profile", type=Path, default=PROJECT_ROOT / "config" / "camera_profile.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "calibration" / "chessboard_new")
    parser.add_argument("--table-reference", type=Path, default=PROJECT_ROOT / "data" / "calibration" / "table_reference_new.jpg")
    parser.add_argument("--pattern", type=parse_pattern, default=(7, 10), help="棋盘格内角点，默认 7x10")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--detection-width", type=int, default=2600)
    args = parser.parse_args()

    profile = yaml.safe_load(args.camera_profile.read_text(encoding="utf-8")) if args.camera_profile.exists() else {}
    values = resolve_camera_profile(profile or {}, {"preview_width": args.preview_width})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_reference.parent.mkdir(parents=True, exist_ok=True)

    next_index = next_calibration_index(args.output_dir)
    status = "READY: SPACE=calibration image, T=table reference, Q=quit"
    saved_calibration = 0

    try:
        with MindVisionCamera(
            values["sdk_path"],
            values["camera"],
            values["resolution_index"],
            values["frame_speed_index"],
            values["exposure_us"],
            values["gain_x"],
        ) as camera:
            window = "Calibration capture"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                frame = camera.read()
                preview = make_preview(frame, args.preview_width)
                cv2.putText(preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(preview, f"Saved: {saved_calibration}  Next: calib_{next_index:02d}.jpg", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(preview, "SPACE save | T table reference | Q quit", (20, preview.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    break

                if key == 32:
                    if detect_chessboard(frame, args.pattern, args.detection_width):
                        path = args.output_dir / f"calib_{next_index:02d}.jpg"
                        write_jpeg(path, frame)
                        next_index += 1
                        saved_calibration += 1
                        status = f"SAVED: {path.name}"
                    else:
                        status = "NOT SAVED: chessboard not detected"

                if key in (ord("t"), ord("T")):
                    if detect_chessboard(frame, args.pattern, args.detection_width):
                        write_jpeg(args.table_reference, frame)
                        status = f"SAVED TABLE REFERENCE: {args.table_reference.name}"
                    else:
                        status = "NOT SAVED: table-reference chessboard not detected"
    finally:
        cv2.destroyAllWindows()

    print(f"已保存标定图：{saved_calibration} 张")
    print(f"标定图目录：{args.output_dir}")
    print(f"台面参考图：{args.table_reference}")


if __name__ == "__main__":
    main()
