"""Debug overlay for /perception/detections_image (cv2 + numpy, no ROS imports)."""

from __future__ import annotations

import cv2
import numpy as np


def color_for(label: str) -> tuple:
    h = sum(ord(c) * (i + 7) for i, c in enumerate(label)) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(bgr: np.ndarray, items: list[dict], alpha: float = 0.4) -> np.ndarray:
    """items: {label, score, box (x1,y1,x2,y2 in bgr pixels), mask (bool at bgr size or None), text (extra string)}."""
    out = bgr.copy()
    layer = bgr.copy()
    for it in items:
        if it.get('mask') is not None:
            layer[it['mask']] = color_for(it['label'])
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    for it in items:
        col = color_for(it['label'])
        x1, y1, x2, y2 = (int(v) for v in it['box'])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        text = f'{it["label"]} {it["score"]:.2f} {it.get("text", "")}'.strip()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y0 = max(y1, th + 4)
        cv2.rectangle(out, (x1, y0 - th - 4), (x1 + tw + 2, y0), col, -1)
        cv2.putText(out, text, (x1 + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out
