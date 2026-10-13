"""Image helpers for VLM requests: encode numpy/PIL/bytes/path to JPEG data URLs, estimate token cost.

ROS image messages are converted without cv_bridge (`array_from_image_msg` is duck-typed on the message fields),
so this module has no ROS dependency and can be tested on any machine. Needs Pillow (preferred) or OpenCV.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image as _PIL
except ImportError:          # pragma: no cover - exercised only where Pillow is missing
    _PIL = None

# Qwen3-VL: 16 px patches merged 2x2 -> one visual token per 32x32 px (plus a few specials). Measured live, see docs/reports/agent.md.
PIXELS_PER_TOKEN = 32 * 32
IMAGE_TOKEN_OVERHEAD = 4


def estimate_image_tokens(width: int, height: int, min_tokens: int = 4, max_tokens: int = 1280) -> int:
    t = math.ceil(width / 32) * math.ceil(height / 32)
    return min(max(t, min_tokens), max_tokens) + IMAGE_TOKEN_OVERHEAD


def array_from_image_msg(msg: Any) -> np.ndarray:
    """sensor_msgs/Image (duck-typed: height, width, encoding, step, data) -> HxWx3 uint8 RGB (or HxW for mono)."""
    h, w, enc = int(msg.height), int(msg.width), str(msg.encoding).lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8) if not isinstance(msg.data, (bytes, bytearray)) else np.frombuffer(msg.data, dtype=np.uint8)
    step = int(getattr(msg, "step", 0)) or None
    if enc in ("rgb8", "bgr8"):
        step = step or w * 3
        arr = buf[: h * step].reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return arr[:, :, ::-1].copy() if enc == "bgr8" else arr.copy()
    if enc in ("rgba8", "bgra8"):
        step = step or w * 4
        arr = buf[: h * step].reshape(h, step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        return arr[:, :, ::-1].copy() if enc == "bgra8" else arr.copy()
    if enc in ("mono8", "8uc1"):
        step = step or w
        return buf[: h * step].reshape(h, step)[:, :w].copy()
    if enc in ("mono16", "16uc1"):
        step = step or w * 2
        a16 = np.frombuffer(bytes(buf[: h * step]), dtype="<u2").reshape(h, step // 2)[:, :w]
        return (a16 / 256).astype(np.uint8)
    raise ValueError(f"unsupported image encoding {enc!r}")


def _to_pil(img: Any):
    if _PIL is None:
        raise RuntimeError("Pillow is required for image encoding (pip install pillow)")
    if isinstance(img, _PIL.Image):
        return img
    if isinstance(img, (str, Path)):
        return _PIL.open(img)
    if isinstance(img, (bytes, bytearray)):
        return _PIL.open(io.BytesIO(bytes(img)))
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (arr * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return _PIL.fromarray(arr)
    raise TypeError(f"cannot encode image of type {type(img).__name__}")


def encode_image(img: Any, max_side: int | None = 512, quality: int = 80, fmt: str = "JPEG") -> bytes:
    """Encode to JPEG (or PNG) bytes, downscaling so the longest side is at most `max_side` (keeps aspect)."""
    pil = _to_pil(img)
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    if max_side and max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))), _PIL.BILINEAR)
    out = io.BytesIO()
    if fmt.upper() == "PNG":
        pil.save(out, format="PNG")
    else:
        pil.convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()


def image_size(data: bytes) -> tuple[int, int]:
    return _to_pil(data).size


def to_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_part(img: Any, max_side: int | None = 512, quality: int = 80) -> dict:
    """An OpenAI chat content part holding the image as a data URL."""
    data = img if isinstance(img, (bytes, bytearray)) and max_side is None else encode_image(img, max_side, quality)
    return {"type": "image_url", "image_url": {"url": to_data_url(bytes(data))}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def user_message(text: str, images: list | None = None, max_side: int | None = 512, quality: int = 80) -> dict:
    """Chat message with text and any number of images (images first, as in the NaVILA/Qwen prompts)."""
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [image_part(i, max_side, quality) for i in images] + [text_part(text)]}


def strip_images(messages: list[dict], placeholder: str = "[image]") -> list[dict]:
    """Copy of `messages` with image data replaced by a short placeholder (for logs and fake servers)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    parts.append({"type": "image_url", "image_url": {"url": f"{placeholder} ({len(url)} chars)"}})
                else:
                    parts.append(p)
            m = dict(m, content=parts)
        out.append(m)
    return out


def count_images(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n
