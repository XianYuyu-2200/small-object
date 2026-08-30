"""Capture labelled raw images from a USB/OpenCV camera for later annotation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="按空格键采集单类别原始图片；Esc 退出。")
    parser.add_argument("--label", required=True, help="与 config/classes.yaml 对应的稳定类别名")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 视频设备编号")
    parser.add_argument("--output", default="data/raw", help="原始图片目录")
    parser.add_argument("--width", type=int, help="采集宽度；须与标定/比赛分辨率一致")
    parser.add_argument("--height", type=int, help="采集高度；须与标定/比赛分辨率一致")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机设备 {args.camera}；请检查迈德威视驱动/SDK 或 OpenCV 设备编号")
    output = Path(args.output) / args.label
    output.mkdir(parents=True, exist_ok=True)
    print("空格：保存；Esc：退出。请覆盖位置、旋转、光照、遮挡与多物件场景。")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("相机读取失败")
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            target = output / name
            cv2.imwrite(str(target), frame)
            print(target)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
