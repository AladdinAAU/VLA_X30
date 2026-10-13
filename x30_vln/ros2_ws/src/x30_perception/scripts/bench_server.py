#!/usr/bin/env python3
"""Latency benchmark of a running perception server.

  python bench_server.py --url http://localhost:8100 --config ../../../../configs/perception.yaml

Sends the two test photos (test/assets: bus.jpg, zidane.jpg, resized to the camera size) with the configured vocabulary at
several input sizes and reports client round-trip and server-side timing (median / p95), plus first-use costs of a new
task phrase. Needs only numpy, opencv, requests, yaml.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

ASSETS = Path(__file__).resolve().parents[1] / 'test' / 'assets'


def b64(img, q=90):
    return base64.b64encode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


def post(url, body):
    t = time.perf_counter()
    r = requests.post(url + '/segment', json=body, timeout=60)
    r.raise_for_status()
    return (time.perf_counter() - t) * 1000.0, r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8100')
    ap.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[4] / 'configs' / 'perception.yaml')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    vocab = yaml.safe_load(a.config.read_text())['vocabulary']
    imgs = {name: cv2.resize(cv2.imread(str(ASSETS / f'{name}.jpg')), (640, 480)) for name in ('bus', 'zidane')}
    out = {'health': requests.get(a.url + '/health', timeout=10).json(), 'phrases': len(vocab), 'runs': []}
    for imgsz in (640, 960, 1280):
        for name, img in imgs.items():
            body = {'image': b64(img), 'vocab': vocab, 'conf': 0.1, 'imgsz': imgsz}
            post(a.url, body)  # warm-up
            rtt, srv, inf, nd = [], [], [], 0
            for _ in range(a.n):
                t, j = post(a.url, body)
                rtt.append(t)
                srv.append(j['ms'])
                inf.append(j['timing']['infer_ms'])
                nd = len(j['instances'])
            row = {
                'imgsz': imgsz,
                'image': name,
                'detections': nd,
                'rtt_ms_median': float(np.median(rtt)),
                'rtt_ms_p95': float(np.percentile(rtt, 95)),
                'server_ms_median': float(np.median(srv)),
                'infer_ms_median': float(np.median(inf)),
                'response_kB': len(json.dumps(j)) / 1024.0,
            }
            out['runs'].append(row)
            print({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()})
    # task phrases: first use of a new phrase and of a changed vocabulary
    body = {'image': b64(imgs['bus']), 'vocab': vocab + ['red double-decker bus'], 'conf': 0.1, 'imgsz': 640}
    t_new, j_new = post(a.url, body)
    t_again, j_again = post(a.url, body)
    body['vocab'] = vocab
    t_back, j_back = post(a.url, body)
    out['new_phrase'] = {
        'first_ms': t_new,
        'vocab_ms': j_new['timing']['vocab_ms'],
        'repeat_ms': t_again,
        'back_to_base_ms': t_back,
        'back_vocab_ms': j_back['timing']['vocab_ms'],
    }
    # region pass (enlarged crop of the image centre)
    body = {
        'image': b64(imgs['bus']),
        'vocab': vocab,
        'conf': 0.1,
        'imgsz': 640,
        'regions': [[0.25, 0.3, 0.75, 0.7]],
        'region_imgsz': 1280,
    }
    post(a.url, body)
    out['region_pass_rtt_ms'] = float(np.median([post(a.url, body)[0] for _ in range(a.n)]))
    out['health_after'] = requests.get(a.url + '/health', timeout=10).json()
    print(json.dumps({k: v for k, v in out.items() if k != 'runs'}, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
