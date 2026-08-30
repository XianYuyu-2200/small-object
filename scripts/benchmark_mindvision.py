"""Benchmark MindVision SDK retrieval with no display, file I/O, or YOLO."""

from __future__ import annotations

import argparse
import json
import time

from swallow_yolo.mindvision import MindVisionCamera
from swallow_yolo.timing import summarize_durations


def main() -> None:
    parser = argparse.ArgumentParser(description="仅测迈德威视 SDK 取帧+图像处理耗时；不显示、不保存、不运行 YOLO。")
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--resolution-index", type=int, required=True)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames 至少为 2")
    with MindVisionCamera(args.sdk_path, args.camera, args.resolution_index) as camera:
        diagnostics = camera.diagnostics()
        durations = []
        for _ in range(args.frames):
            started = time.perf_counter()
            frame = camera.read(timeout_ms=5000)
            durations.append(time.perf_counter() - started)
        print(json.dumps({"camera": camera.device_name, "actual_resolution": diagnostics["resolution"], "requested_resolution_index": args.resolution_index, "frame_shape": list(frame.shape), "sdk_read_and_process": summarize_durations(durations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
