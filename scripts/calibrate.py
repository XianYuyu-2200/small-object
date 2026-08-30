"""Create a reusable fixed-camera calibration file before the demonstration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def find_corners(paths: list[Path], columns: int, rows: int, square_mm: float):
    pattern = (columns, rows)
    object_points = np.zeros((columns * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    object_sets, image_sets, size = [], [], None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_sets.append(object_points)
        image_sets.append(refined)
        size = (gray.shape[1], gray.shape[0])
    if size is None or len(object_sets) < 10:
        raise RuntimeError("至少需要 10 张成功识别棋盘格的图片；请检查 --pattern、清晰度和棋盘格可见性")
    return object_sets, image_sets, size


def main() -> None:
    parser = argparse.ArgumentParser(description="赛前棋盘格标定，并以一张台面参考图建立像素到毫米映射。")
    parser.add_argument("--images", required=True, help="棋盘格标定图片目录")
    parser.add_argument("--pattern", default="9x6", help="棋盘格内角点列x行，例如 9x6")
    parser.add_argument("--square-mm", type=float, required=True, help="棋盘格单格边长（毫米）")
    parser.add_argument("--table-reference", required=True, help="平放在最终台面的棋盘格参考图片")
    parser.add_argument("--output", default="data/calibration/calibration.json")
    parser.add_argument("--camera-model", default="MV-SUA2000C-TV1-C")
    args = parser.parse_args()
    columns, rows = (int(value) for value in args.pattern.lower().split("x"))
    images = [path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    object_sets, image_sets, image_size = find_corners(images, columns, rows, args.square_mm)
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(object_sets, image_sets, image_size, None, None)

    reference = cv2.imread(args.table_reference)
    if reference is None:
        raise RuntimeError("无法读取 --table-reference")
    undistorted = cv2.undistort(reference, camera_matrix, distortion)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (columns, rows))
    if not found:
        raise RuntimeError("台面参考图未识别到棋盘格")
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)).reshape(-1, 2)
    table_points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32) * args.square_mm
    homography, mask = cv2.findHomography(refined, table_points, cv2.RANSAC)
    if homography is None or int(mask.sum()) < 4:
        raise RuntimeError("无法生成台面单应矩阵")
    result = {
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "table_homography_px_to_mm": homography.tolist(),
        "reprojection_error_px": float(rms),
        "metadata": {
            "camera_model": args.camera_model,
            "pattern": args.pattern,
            "square_mm": args.square_mm,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "Valid only for unchanged camera, lens focus, resolution, mount and table plane.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output}；重投影 RMS 误差：{rms:.3f} px")


if __name__ == "__main__":
    main()
