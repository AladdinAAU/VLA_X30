#!/usr/bin/env python3
"""Smoke test of the OpenAI-compatible VLM server: text prompt, image prompt (PIL-generated), tokens/s.
  proof_vlm.py [--base-url http://vlm:8000/v1] [--model qwen3-vl]
"""
import argparse
import base64
import io
import json
import time

import requests
from PIL import Image, ImageDraw


def chat(base, model, content, max_tokens=64):
    t0 = time.time()
    r = requests.post(f"{base}/chat/completions", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                      "messages": [{"role": "user", "content": content}]}, timeout=180)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    return j, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://vlm:8000/v1")
    ap.add_argument("--model", default="qwen3-vl")
    a = ap.parse_args()
    print("models:", [m["id"] for m in requests.get(f"{a.base_url}/models", timeout=10).json()["data"]])

    j, dt = chat(a.base_url, a.model, "Reply with one short sentence: what is the capital of Denmark?")
    u = j["usage"]
    print(f"TEXT  {dt:.2f}s prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} -> {j['choices'][0]['message']['content']!r}")
    print("      server timings:", {k: round(v, 1) for k, v in j.get("timings", {}).items() if k in ("prompt_per_second", "predicted_per_second")})

    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 120, 300, 360], fill=(220, 30, 30))
    d.ellipse([380, 160, 560, 340], fill=(30, 60, 220))
    buf = io.BytesIO(); img.save(buf, "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
               {"type": "text", "text": "Describe the shapes and their colors in this image in one sentence."}]
    for i in range(2):
        j, dt = chat(a.base_url, a.model, content, max_tokens=80)
        u = j["usage"]; tm = j.get("timings", {})
        print(f"IMAGE#{i} {dt:.2f}s prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} "
              f"prompt/s={tm.get('prompt_per_second', 0):.0f} gen tok/s={tm.get('predicted_per_second', 0):.1f} -> {j['choices'][0]['message']['content']!r}")


if __name__ == "__main__":
    main()
