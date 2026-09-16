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


def estimate_coplanar_homography(object_points: np.ndarray, image_sets: list[np.ndarray], camera_matrix: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, int]:
    """Build one image-to-millimetre map from board views on the same table plane.

    Each photo may contain the board at a different table position. The board's
    local origin is therefore not shared; only the plane normal and plane
    distance are shared. A metric coordinate frame is anchored arbitrarily at
    the closest point of the recovered table plane.
    """
    normals: list[np.ndarray] = []
    distances: list[float] = []
    first_rotation: np.ndarray | None = None
    for corners in image_sets:
        ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        normal = rotation[:, 2].astype(float)
        normal /= np.linalg.norm(normal)
        distance = float(np.dot(normal, tvec.reshape(3)))
        if not normals:
            first_rotation = rotation
        elif float(np.dot(normal, normals[0])) < 0:
            normal = -normal
            distance = -distance
        if distance < 0:
            normal = -normal
            distance = -distance
        normals.append(normal)
        distances.append(distance)
    if len(normals) < 3 or first_rotation is None:
        raise RuntimeError("无法从棋盘格视图估计台面平面")
    normal = np.mean(np.asarray(normals), axis=0)
    normal /= np.linalg.norm(normal)
    distance = float(np.median(distances))
    e1 = first_rotation[:, 0].astype(float)
    e1 -= normal * float(np.dot(normal, e1))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    origin = normal * distance
    plane_to_camera = np.column_stack((e1, e2, origin))
    homography = np.linalg.inv(camera_matrix @ plane_to_camera)
    homography /= homography[2, 2]
    return homography, len(normals)


def main() -> None:
    parser = argparse.ArgumentParser(description="赛前棋盘格标定，并以一张台面参考图建立像素到毫米映射。")
    parser.add_argument("--images", required=True, help="棋盘格标定图片目录")
    parser.add_argument("--pattern", default="9x6", help="棋盘格内角点列x行，例如 9x6")
    parser.add_argument("--square-mm", type=float, required=True, help="棋盘格单格边长（毫米）")
    parser.add_argument("--table-reference", help="可选：平放在最终台面的棋盘格参考图片")
    parser.add_argument("--camera-height-mm", type=float, default=328.0, help="相机光心到台面的实测距离（毫米）；写入 metadata，无 table-reference 时用于生成理想映射")
    parser.add_argument("--output", default="data/calibration/calibration.json")
    parser.add_argument("--camera-model", default="MV-SUA2000C-TV1-C")
    args = parser.parse_args()
    columns, rows = (int(value) for value in args.pattern.lower().split("x"))
    images = [path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    object_sets, image_sets, image_size = find_corners(images, columns, rows, args.square_mm)
    rms, camera_matrix, distortion, rotations, _ = cv2.calibrateCamera(object_sets, image_sets, image_size, None, None)
    tilts = [float(np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rotations]
    if max(tilts) < 15.0:
        print(
            f"警告：所有棋盘格照片相对像面的倾角都不足 15°（最大 {max(tilts):.1f}°）。"
            "此时焦距与相机距离无法被可靠分离，camera_matrix 的 fx/fy 可能明显偏离真值；"
            "台面单应矩阵仍然有效，但视差修正必须使用实测的 camera_height_mm（见 config/object_heights.yaml）。"
            "建议补拍若干张棋盘格倾斜 20°–45° 的照片后重新标定。"
        )

    if args.table_reference:
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
        mapping_note = "homography estimated from table-reference"
    else:
        homography, plane_views = estimate_coplanar_homography(object_sets[0], image_sets, camera_matrix, distortion)
        mapping_note = f"metric table-plane mapping estimated from {plane_views} coplanar chessboard views"
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
            "camera_height_mm": args.camera_height_mm,
            "chessboard_tilt_deg_max": round(max(tilts), 2),
            "mapping_note": mapping_note,
            "note": "Valid only for unchanged camera, lens focus, resolution, mount and table plane.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output}；重投影 RMS 误差：{rms:.3f} px")


if __name__ == "__main__":
    main()
