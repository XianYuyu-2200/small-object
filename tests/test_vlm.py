import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from swallow_yolo.vlm import analyze_frame, classify_size_level, encode_frame_jpeg, load_vlm_config, parse_vlm_content


def test_load_vlm_config_prefers_environment_key(tmp_path):
    path = tmp_path / "vlm.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "endpoint": "https://example.test/v1/chat/completions",
                "model": "vision-model",
                "api_key_env": "TEST_VLM_KEY",
                "api_key": "config-key",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_vlm_config(path, {"TEST_VLM_KEY": "env-key"})
    assert config.api_key == "env-key"
    assert config.model == "vision-model"
    assert config.use_system_proxy is False


def test_parse_vlm_content_accepts_markdown_and_alias():
    content = """```json
    {"object_name":"红色小珠子","decision":"能吞咽","confidence":"82%","reasons":["体积较小"]}
    ```"""
    result = parse_vlm_content(content)
    assert result.object_name == "红色小珠子"
    assert result.decision == "容易吞咽"
    assert result.confidence == 0.82
    assert result.reasons == ("体积较小",)
    assert "uncertainty" not in result.to_dict()


def test_encode_frame_jpeg_resizes_long_edge():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    encoded = encode_frame_jpeg(frame, max_image_edge=320, jpeg_quality=80)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (160, 320)


def test_analyze_frame_sends_openai_compatible_image_message(tmp_path):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "object_name": "小型红色物件",
                                        "decision": "不容易吞咽",
                                        "confidence": 0.9,
                                        "reasons": ["相对桌面尺寸较大"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

    class FakeOpener:
        def __init__(self):
            self.request = None
            self.timeout = None

        def __call__(self, request, timeout):
            self.request = request
            self.timeout = timeout
            return FakeResponse()

    config_path = tmp_path / "vlm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "endpoint": "https://example.test/v1/chat/completions",
                "model": "vision-model",
                "api_key": "secret",
                "timeout_seconds": 12,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = load_vlm_config(config_path, {})
    opener = FakeOpener()
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    result = analyze_frame(frame, config, opener=opener, measurement={"length_mm": 47.6, "width_mm": 14.7})

    assert result.object_name == "小型红色物件"
    assert result.decision == "可能吞咽"
    assert opener.timeout == 12
    assert opener.request.headers["Authorization"] == "Bearer secret"
    payload = json.loads(opener.request.data.decode("utf-8"))
    image_part = payload["messages"][1]["content"][1]
    prompt_part = payload["messages"][1]["content"][0]
    assert "47.60 mm × 14.70 mm" in prompt_part["text"]
    assert "不要重新估计尺寸" in prompt_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 700


def test_classify_size_level_boundaries():
    assert classify_size_level(8, 8)[0] == "极易吞咽"
    assert classify_size_level(18, 15)[0] == "容易吞咽"
    assert classify_size_level(25, 22)[0] == "可能吞咽"
    assert classify_size_level(35, 30)[0] == "不容易吞咽"
    assert classify_size_level(45, 40)[0] == "无法吞咽"
