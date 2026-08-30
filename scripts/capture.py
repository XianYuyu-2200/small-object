"""Capture labelled raw images from a USB/OpenCV camera for later annotation."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from swallow_yolo.mindvision import MindVisionCamera
from swallow_yolo.frame_rate import FrameRateMonitor
from swallow_yolo.display import fit_for_display


def main() -> None:
    parser = argparse.ArgumentParser(description="按空格键采集单类别原始图片；Esc 退出。")
    parser.add_argument("--label", required=True, help="与 config/classes.yaml 对应的稳定类别名")
    parser.add_argument("--backend", choices=("opencv", "mindvision"), default="opencv")
    parser.add_argument("--camera", type=int, default=0, help="所选后端中的设备编号")
    parser.add_argument("--sdk-path", help="迈德威视 SDK 根目录，例如 G:\\mindvision；仅 mindvision 后端需要")
    parser.add_argument("--resolution-index", type=int, help="迈德威视 SDK 预设分辨率索引；先用 diagnose_mindvision.py 查看")
    parser.add_argument("--frame-speed-index", type=int, help="迈德威视帧速档位；2 对应 High")
    parser.add_argument("--exposure-us", type=float, help="手动曝光（微秒）；提供该值会关闭自动曝光")
    parser.add_argument("--preview-width", type=int, default=1280, help="预览窗口最大宽度；不影响保存原图")
    parser.add_argument("--output", default="data/raw", help="原始图片目录")
    parser.add_argument("--width", type=int, help="采集宽度；须与标定/比赛分辨率一致")
    parser.add_argument("--height", type=int, help="采集高度；须与标定/比赛分辨率一致")
    args = parser.parse_args()

    output = Path(args.output) / args.label
    output.mkdir(parents=True, exist_ok=True)
    mindvision = None
    cap = None
    monitor = FrameRateMonitor(time.monotonic())
    if args.backend == "mindvision":
        if not args.sdk_path:
            raise SystemExit("使用 --backend mindvision 时必须提供 --sdk-path G:\\mindvision")
        mindvision = MindVisionCamera(args.sdk_path, args.camera, args.resolution_index, args.frame_speed_index, args.exposure_us)
        print(f"已连接迈德威视相机：{mindvision.device_name}")
        if args.width or args.height:
            print("注意：MindVision 后端当前按相机现有分辨率采集；请在 MVDCP2/SDK 中固定分辨率，再执行采集。")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 OpenCV 相机设备 {args.camera}")
    print("空格：保存；Esc：退出。请覆盖位置、旋转、光照、遮挡与多物件场景。")
    while True:
        frame = mindvision.read() if mindvision else None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("相机读取失败")
        monitor.record(time.monotonic())
        preview = fit_for_display(frame, args.preview_width)
        cv2.putText(preview, f"{frame.shape[1]}x{frame.shape[0]}  FPS {monitor.fps:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            target = output / name
            cv2.imwrite(str(target), frame)
            print(target)
    if cap is not None:
        cap.release()
    if mindvision is not None:
        mindvision.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
