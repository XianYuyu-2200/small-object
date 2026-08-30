"""Print current MindVision capture settings without changing them."""

from __future__ import annotations

import argparse
import json

from swallow_yolo.mindvision import MindVisionCamera


def main() -> None:
    parser = argparse.ArgumentParser(description="只读输出迈德威视相机的分辨率、曝光、触发和缩放能力。")
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    with MindVisionCamera(args.sdk_path, args.camera) as camera:
        print(f"camera: {camera.device_name}")
        print(json.dumps(camera.diagnostics(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
