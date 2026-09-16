"""OpenAI-compatible multimodal client used by the competition console."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, getproxies

import cv2
import numpy as np
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_SYSTEM_PROMPT = """你是一个实验台固定俯视相机演示中的视觉分析助手。
你需要先给出画面中央目标物的简短名称，再判断它是否容易被吞咽。
如果系统提供了尺寸测量值，必须直接采用该数值，不要自行估算。
吞咽等级只能使用：无法吞咽、不容易吞咽、可能吞咽、容易吞咽、极易吞咽。
如果无法确认物体名称，object_name 填写“未识别”，不要编造。
不要给出医疗建议，也不要编造不可见信息。"""

DEFAULT_USER_PROMPT = """请先简要识别当前画面中央的目标物，再分析它是否容易被吞咽。
严格只返回一个 JSON 对象，不要使用 Markdown 代码块，不要添加 JSON 之外的文字。
JSON 字段必须为：
{
  "object_name": "目标的简短名称；无法确认时填未识别",
  "decision": "无法吞咽 | 不容易吞咽 | 可能吞咽 | 容易吞咽 | 极易吞咽",
  "confidence": 0.0,
  "reasons": ["最多三条简短依据"]
}
confidence 必须在 0 到 1 之间。"""


class VlmError(RuntimeError):
    """Raised when the multimodal API cannot return a usable result."""


@dataclass(frozen=True)
class VlmConfig:
    endpoint: str
    model: str
    api_key: str
    timeout_seconds: float
    max_image_edge: int
    jpeg_quality: int
    system_prompt: str
    user_prompt: str
    use_system_proxy: bool
    extra_headers: Mapping[str, str]


@dataclass(frozen=True)
class VlmResult:
    object_name: str
    decision: str
    confidence: float | None
    reasons: tuple[str, ...]
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_name": self.object_name,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


def load_vlm_config(path: str | Path, environ: Mapping[str, str] | None = None) -> VlmConfig:
    """Load a provider-neutral OpenAI-compatible chat-completions config."""

    path = Path(path)
    if not path.exists():
        raise VlmError(f"缺少多模态 API 配置文件：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise VlmError(f"{path} 的根节点必须是 YAML 对象")

    environ = os.environ if environ is None else environ
    endpoint = str(data.get("endpoint", "")).strip()
    model = str(data.get("model", "")).strip()
    api_key_env = str(data.get("api_key_env", "")).strip()
    env_api_key = environ.get(api_key_env, "").strip() if api_key_env else ""
    api_key = env_api_key or str(data.get("api_key", "")).strip()

    if not endpoint:
        raise VlmError("请在 config/vlm.yaml 中配置 endpoint")
    if not endpoint.lower().startswith(("http://", "https://")):
        raise VlmError("endpoint 必须是完整的 http:// 或 https:// 地址")
    if not model:
        raise VlmError("请在 config/vlm.yaml 中配置 model")

    try:
        timeout_seconds = float(data.get("timeout_seconds", 90.0))
        max_image_edge = int(data.get("max_image_edge", 1280))
        jpeg_quality = int(data.get("jpeg_quality", 88))
    except (TypeError, ValueError) as error:
        raise VlmError("timeout_seconds、max_image_edge 和 jpeg_quality 必须是数字") from error

    if timeout_seconds <= 0:
        raise VlmError("timeout_seconds 必须大于 0")
    if max_image_edge < 320:
        raise VlmError("max_image_edge 不能小于 320")
    if not 1 <= jpeg_quality <= 100:
        raise VlmError("jpeg_quality 必须在 1 到 100 之间")

    raw_headers = data.get("extra_headers") or {}
    if not isinstance(raw_headers, dict):
        raise VlmError("extra_headers 必须是 YAML 对象")
    extra_headers = {str(key): str(value) for key, value in raw_headers.items()}

    use_system_proxy = data.get("use_system_proxy", False)
    if not isinstance(use_system_proxy, bool):
        raise VlmError("use_system_proxy 必须是 true 或 false")

    return VlmConfig(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_image_edge=max_image_edge,
        jpeg_quality=jpeg_quality,
        system_prompt=str(data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip(),
        user_prompt=str(data.get("user_prompt") or DEFAULT_USER_PROMPT).strip(),
        use_system_proxy=use_system_proxy,
        extra_headers=extra_headers,
    )


def _post_json(config: VlmConfig, headers: Mapping[str, str], payload: Mapping[str, Any]) -> bytes:
    """POST JSON with a direct connection by default and retry transient failures."""

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    if config.use_system_proxy:
        session.trust_env = True
        session.proxies.update(getproxies())
    else:
        session.trust_env = False
        session.proxies = {}

    try:
        response = session.post(
            config.endpoint,
            headers=dict(headers),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=(15, config.timeout_seconds),
        )
        response_bytes = response.content
        status_code = response.status_code
    except requests.Timeout as error:
        raise VlmError("请求超时，请检查网络后重试") from error
    except requests.RequestException as error:
        raise VlmError(f"网络连接中断：{error}") from error
    finally:
        session.close()

    if status_code >= 400:
        detail = response_bytes.decode("utf-8", errors="replace")[:1000]
        suffix = f"：{detail}" if detail else ""
        raise VlmError(f"请求失败，HTTP {status_code}{suffix}")
    return response_bytes


def encode_frame_jpeg(frame: np.ndarray, max_image_edge: int = 1280, jpeg_quality: int = 88) -> bytes:
    """Resize a BGR frame and encode it as JPEG for API transport."""

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise VlmError("没有可分析的相机画面")
    if frame.ndim not in (2, 3):
        raise VlmError("相机画面格式无效")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise VlmError("相机画面尺寸无效")
    scale = min(1.0, float(max_image_edge) / max(height, width))
    prepared = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    ok, encoded = cv2.imencode(".jpg", prepared, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise VlmError("相机画面 JPEG 编码失败")
    return encoded.tobytes()


def _extract_message_content(response_data: Mapping[str, Any]) -> Any:
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VlmError("API 响应中没有 choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise VlmError("API 响应的 choices[0] 格式无效")
    message = first.get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise VlmError("API 响应中没有 message.content")
    content = message["content"]
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, Mapping) and item.get("type") in ("text", "output_text"):
                texts.append(str(item.get("text", "")))
        content = "\n".join(part for part in texts if part)
    if not isinstance(content, str) or not content.strip():
        raise VlmError("API 返回的 message.content 为空")
    return content


def _parse_confidence(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return min(1.0, max(0.0, confidence))


SIZE_LEVEL_GUIDE = """尺寸分级规则（必须严格执行）：
有效尺寸 = min(长边, 短边) + 0.35 × (max(长边, 短边) - min(长边, 短边))。
有效尺寸 < 12 mm：极易吞咽；
12 mm ≤ 有效尺寸 < 20 mm：容易吞咽；
20 mm ≤ 有效尺寸 < 30 mm：可能吞咽；
30 mm ≤ 有效尺寸 < 40 mm：不容易吞咽；
有效尺寸 ≥ 40 mm：无法吞咽。"""


def classify_size_level(length_mm: float, width_mm: float) -> tuple[str, float]:
    """Return the deterministic five-level grade and effective planar size."""

    length = float(length_mm)
    width = float(width_mm)
    if length <= 0 or width <= 0:
        raise VlmError("目标尺寸必须大于 0")
    shorter, longer = sorted((length, width))
    effective_mm = shorter + 0.35 * (longer - shorter)
    if effective_mm < 12.0:
        return "极易吞咽", effective_mm
    if effective_mm < 20.0:
        return "容易吞咽", effective_mm
    if effective_mm < 30.0:
        return "可能吞咽", effective_mm
    if effective_mm < 40.0:
        return "不容易吞咽", effective_mm
    return "无法吞咽", effective_mm


def _normalize_decision(value: Any) -> str:
    decision = str(value or "").strip()
    aliases = {
        "无法吞咽": "无法吞咽",
        "不能吞咽": "无法吞咽",
        "很难吞咽": "无法吞咽",
        "不容易吞咽": "不容易吞咽",
        "不易吞咽": "不容易吞咽",
        "风险低": "不容易吞咽",
        "可能吞咽": "可能吞咽",
        "可能": "可能吞咽",
        "无法判断": "可能吞咽",
        "无法确定": "可能吞咽",
        "不确定": "可能吞咽",
        "容易吞咽": "容易吞咽",
        "易吞咽": "容易吞咽",
        "能吞咽": "容易吞咽",
        "可以吞咽": "容易吞咽",
        "风险高": "容易吞咽",
        "极易吞咽": "极易吞咽",
        "非常容易吞咽": "极易吞咽",
        "极易": "极易吞咽",
    }
    normalized = aliases.get(decision)
    if normalized is None:
        raise VlmError(f"API 返回了无法识别的 decision：{decision!r}")
    return normalized


def parse_vlm_content(content: Any) -> VlmResult:
    """Parse model output, tolerating accidental Markdown fences."""

    if isinstance(content, Mapping):
        parsed: Any = dict(content)
    else:
        text = str(content).strip().lstrip("\ufeff")
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise VlmError("API 返回内容中没有 JSON 对象")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise VlmError(f"API 返回的 JSON 无法解析：{error}") from error

    if not isinstance(parsed, Mapping):
        raise VlmError("API 返回的 JSON 根节点必须是对象")
    if isinstance(parsed.get("result"), Mapping):
        parsed = parsed["result"]

    if "decision" not in parsed:
        raise VlmError("API 返回 JSON 缺少 decision 字段")
    decision = _normalize_decision(parsed.get("decision"))
    object_name = str(parsed.get("object_name") or parsed.get("name") or "未识别").strip()
    object_name = object_name.splitlines()[0].strip() if object_name else "未识别"
    if object_name in {"不确定", "未知", "无法识别", "无法确认"}:
        object_name = "未识别"
    object_name = object_name[:18]

    raw_reasons = parsed.get("reasons", [])
    if isinstance(raw_reasons, str):
        reasons = (raw_reasons.strip(),) if raw_reasons.strip() else ()
    elif isinstance(raw_reasons, list):
        reasons = tuple(str(item).strip() for item in raw_reasons if str(item).strip())
    else:
        reasons = ()

    return VlmResult(
        object_name=object_name,
        decision=decision,
        confidence=_parse_confidence(parsed.get("confidence")),
        reasons=reasons[:5],
        raw=dict(parsed),
    )


def analyze_frame(
    frame: np.ndarray,
    config: VlmConfig,
    opener: Callable[..., Any] | None = None,
    measurement: Mapping[str, Any] | None = None,
) -> VlmResult:
    """Send one frame to an OpenAI-compatible multimodal endpoint."""

    user_prompt = config.user_prompt
    if measurement:
        try:
            length_mm = float(measurement["length_mm"])
            width_mm = float(measurement["width_mm"])
        except (KeyError, TypeError, ValueError) as error:
            raise VlmError("尺寸测量结果格式无效") from error
        level, effective_mm = classify_size_level(length_mm, width_mm)
        user_prompt += (
            "\n\n" + SIZE_LEVEL_GUIDE +
            "\nOpenCV 已测得目标在台面上的平面最大外接尺寸约为 "
            f"{length_mm:.2f} mm × {width_mm:.2f} mm。"
            f"对应有效尺寸为 {effective_mm:.2f} mm，尺寸等级为“{level}”。"
            "该测量值来自已标定相机，请直接采用，不要重新估计尺寸。"
            "decision 必须使用上述尺寸等级。"
        )

    image_bytes = encode_frame_jpeg(frame, config.max_image_edge, config.jpeg_quality)
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 700,
        # Seed 2.x defaults to a long reasoning pass for some photos. The
        # competition flow only needs a short structured answer, so disable it.
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    headers.update(config.extra_headers)

    if opener is None:
        response_bytes = _post_json(config, headers, payload)
    else:
        request = Request(
            config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with opener(request, timeout=config.timeout_seconds) as response:
                response_bytes = response.read()
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                detail = ""
            suffix = f"：{detail}" if detail else ""
            raise VlmError(f"请求失败，HTTP {error.code}{suffix}") from error
        except URLError as error:
            raise VlmError(f"网络连接中断：{error.reason}") from error
        except TimeoutError as error:
            raise VlmError("请求超时，请检查网络后重试") from error

    try:
        response_data = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VlmError("API 返回的不是有效 UTF-8 JSON") from error
    if not isinstance(response_data, Mapping):
        raise VlmError("API 响应根节点必须是 JSON 对象")
    result = parse_vlm_content(_extract_message_content(response_data))
    if measurement:
        required_level, effective_mm = classify_size_level(length_mm, width_mm)
        if result.decision != required_level:
            reasons = (f"按标定有效尺寸 {effective_mm:.2f} mm 划分为“{required_level}”",) + result.reasons[:2]
            result = replace(result, decision=required_level, reasons=reasons)
    return result
