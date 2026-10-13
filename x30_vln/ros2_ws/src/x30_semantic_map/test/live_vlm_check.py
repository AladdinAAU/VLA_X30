#!/usr/bin/env python3
"""Live check of the annotator against a running OpenAI-compatible VLM (the llama.cpp Qwen3-VL service, port 8000).

Uses real renderings of simulated indoor scenes (AWS RoboMaker bookstore and hospital worlds, in datasets/external/aws)
cropped by hand with known labels. Measures per-call latency, label verification (true label vs deliberately wrong
label), caption keyword agreement, colour agreement, room naming and `find` ranking on a map built from the crops.
Never starts or stops a server. Writes JSON to --out.

    python test/live_vlm_check.py --url http://localhost:8000/v1 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / 'x30_common')]

from x30_common.vlm_client import VLMClient  # noqa: E402

from x30_semantic_map import annotator as ann  # noqa: E402
from x30_semantic_map import attributes as attr  # noqa: E402
from x30_semantic_map import rooms  # noqa: E402
from x30_semantic_map.config import load_config  # noqa: E402
from x30_semantic_map.datatypes import Detection, Frame  # noqa: E402
from x30_semantic_map.semantic_map import SemanticMap  # noqa: E402

ROOT = HERE.parents[3]
IMG = ROOT / 'datasets/external/aws'
BOOK = IMG / 'aws-robomaker-bookstore-world/docs/images/gazebo_01.png'
HOSP = IMG / 'aws-robomaker-hospital-world/docs/images/hospital_world.jpg'

# name, image, box (x1, y1, x2, y2), detector label, caption keywords (any), plausible colours (any), wrong label
CROPS = [
    ('stool_a', BOOK, (1150, 565, 1235, 680), 'stool', ['stool', 'chair', 'seat'], ['white', 'beige', 'gray', 'brown', 'black'], 'refrigerator'),
    ('stool_b', BOOK, (1050, 705, 1135, 820), 'stool', ['stool', 'chair', 'seat'], ['white', 'beige', 'gray', 'brown', 'black'], 'person'),
    ('stool_c', BOOK, (955, 860, 1040, 985), 'stool', ['stool', 'chair', 'seat'], ['white', 'beige', 'gray', 'brown', 'black'], 'couch'),
    ('bookshelf', BOOK, (575, 110, 745, 390), 'bookshelf', ['shelf', 'book', 'display', 'cabinet'], [], 'toilet'),
    ('wall_clock', BOOK, (100, 260, 160, 325), 'clock', ['clock'], ['yellow', 'green', 'orange', 'beige', 'blue'], 'bottle'),
    ('monitor_desk', BOOK, (115, 650, 200, 740), 'monitor', ['monitor', 'computer', 'screen', 'display', 'imac'], ['gray', 'white', 'black'], 'plant'),
    ('person_nurse', HOSP, (270, 125, 335, 250), 'person', ['person', 'woman', 'nurse', 'man', 'people', 'figure'], ['blue'], 'chair'),
    ('person_purple', HOSP, (610, 110, 662, 285), 'person', ['person', 'man', 'woman', 'people', 'figure'], ['purple', 'pink', 'blue'], 'table'),
    ('wheelchair', HOSP, (60, 148, 142, 232), 'wheelchair', ['wheelchair', 'chair', 'person', 'seat'], [], 'tv'),
    ('vending', HOSP, (518, 108, 578, 176), 'vending machine', ['vending', 'machine', 'refrigerator', 'cooler', 'dispenser', 'fridge'], ['white', 'gray'], 'bed'),
    ('elevator', HOSP, (298, 490, 386, 612), 'elevator', ['elevator', 'lift', 'door'], ['gray', 'white', 'beige'], 'sofa'),
    ('fridge', HOSP, (178, 735, 236, 812), 'refrigerator', ['refrigerator', 'fridge', 'cabinet', 'door'], ['gray', 'black', 'white'], 'bed'),
    ('armchair', HOSP, (60, 782, 142, 852), 'armchair', ['chair', 'armchair', 'sofa', 'couch', 'seat'], ['gray', 'white', 'beige'], 'refrigerator'),
    ('hospital_bed', HOSP, (0, 545, 92, 642), 'bed', ['bed', 'hospital', 'curtain', 'patient'], [], 'plant'),
    ('picture', HOSP, (428, 96, 464, 126), 'picture', ['picture', 'painting', 'frame', 'sunset', 'landscape', 'photo', 'screen'], [], 'chair'),
]
ROOM_CASES = [
    ('kitchen', {'refrigerator': 1, 'sink': 1, 'microwave': 1, 'table': 1, 'chair': 4}, ['kitchen', 'dining room']),
    ('bathroom', {'toilet': 1, 'sink': 1, 'shower': 1}, ['bathroom']),
    ('office', {'desk': 4, 'monitor': 4, 'office chair': 4, 'printer': 1}, ['office', 'lab', 'meeting room']),
    ('bedroom', {'bed': 2, 'nightstand': 2, 'wardrobe': 1}, ['bedroom']),
    ('living room', {'couch': 1, 'tv': 1, 'coffee table': 1, 'armchair': 2}, ['living room']),
]
FIND_CASES = [
    ('the vending machine', 'vending'), ('the person in the purple shirt', 'person_purple'),
    ('something to sit on', ('stool_a', 'stool_b', 'stool_c', 'armchair', 'wheelchair')), ('the wall clock', 'wall_clock'),
    ('where can I get a cold drink', 'vending'), ('the elevator', 'elevator'),
]


def crop_jpeg(path: Path, box) -> bytes:
    img = cv2.imread(str(path))
    x1, y1, x2, y2 = box
    pad = int(0.08 * max(x2 - x1, y2 - y1))
    c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    return attr.encode_jpeg(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), 90)


def build_map(cfg) -> tuple[SemanticMap, dict[str, int]]:
    m = SemanticMap(cfg)
    ids: dict[str, int] = {}
    for k, (name, path, box, label, *_rest) in enumerate(CROPS):
        jpg = crop_jpeg(path, box)
        pos = np.array([1.0 + 1.5 * (k % 5), 1.0 + 2.0 * (k // 5), 0.5])
        for i in range(8):
            m.process_frame(Frame(float(i) * 0.3, np.array([0.0, 0.0, 0.5]), [Detection(
                label, 0.7, pos, np.array([0.4, 0.4, 0.8]), np.array([1.0, 0, 0]), False, 100, crop=jpg,
                box_xyxy=(0, 0, box[2] - box[0], box[3] - box[1]))]))
        ids[name] = max(m.objects)
    return m, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=os.environ.get('X30_MAPPER_URL', 'http://localhost:8000/v1'))
    ap.add_argument('--model', default=os.environ.get('X30_MAPPER_MODEL', 'qwen3-vl'))
    ap.add_argument('--out', default='live_vlm_check.json')
    args = ap.parse_args()
    cfg = load_config()
    cfg.vlm.min_interval_s = 0.0
    cfg.vlm.max_calls_per_min = 10 ** 6
    cfg.vlm.timeout_s = 60.0
    client = VLMClient(args.url, args.model, max_tokens=cfg.vlm.max_tokens, timeout_s=60, max_retries=1, backoff_s=0.5)
    if not client.health():
        print('NO VLM at', args.url)
        return 2
    out: dict = {'url': args.url, 'model': args.model, 'n_crops': len(CROPS)}

    # ---- (a) caption + attribute + label check per crop, true label and wrong label
    prompts = ann.Prompts()
    a = ann.Annotator(SemanticMap(cfg), cfg.vlm, client)
    rows = []
    for name, path, box, label, kws, colours, wrong in CROPS:
        jpg = crop_jpeg(path, box)
        row = {'crop': name, 'label': label, 'wrong_label': wrong}
        for tag, lab in (('true', label), ('wrong', wrong)):
            p = prompts.render('caption_object', label=lab, size='medium', room_hint='')
            t0 = time.time()
            ans = a.call('caption', 'caption', p, [jpg], f'live:{name}:{tag}')
            row[f'{tag}_latency_s'] = round(time.time() - t0, 2)
            row[tag] = ans
        rows.append(row)
        print(name, '|', (row['true'] or {}).get('caption'), '|', (row['true'] or {}).get('colour'), '| wrong->',
              (row['wrong'] or {}).get('label_matches'), flush=True)
    ok_true = [r for r in rows if r['true']]
    ok_wrong = [r for r in rows if r['wrong']]
    out['caption'] = {
        'answered': f'{len(ok_true)}/{len(rows)}',
        'label_confirmed_when_true': sum(r['true']['label_matches'] for r in ok_true) / max(len(ok_true), 1),
        'label_rejected_when_wrong': sum(not r['wrong']['label_matches'] for r in ok_wrong) / max(len(ok_wrong), 1),
        'caption_keyword_hit': sum(any(k in (r['true']['caption'] + ' ' + r['true']['label']).lower()
                                       for k in CROPS[i][4]) for i, r in enumerate(rows) if r['true']) / max(len(ok_true), 1),
        'colour_plausible': (lambda c: sum(c) / max(len(c), 1))(
            [rows[i]['true']['colour'] in CROPS[i][5] for i in range(len(rows)) if rows[i]['true'] and CROPS[i][5]]),
        'colour_cases': sum(1 for i in range(len(rows)) if rows[i]['true'] and CROPS[i][5]),
        'colour_from_pixels_agrees_with_vlm': None,
    }
    px = [(attr.dominant_colour(crop_jpeg(CROPS[i][1], CROPS[i][2])), rows[i]['true']['colour']) for i in range(len(rows))
          if rows[i]['true']]
    out['caption']['colour_from_pixels_agrees_with_vlm'] = sum(p == v for p, v in px) / max(len(px), 1)
    lat = [r[f'{t}_latency_s'] for r in rows for t in ('true', 'wrong')]
    out['caption_latency_s'] = {'n': len(lat), 'mean': round(statistics.mean(lat), 2), 'median': round(statistics.median(lat), 2),
                                'max': round(max(lat), 2)}
    out['caption_rows'] = rows

    # ---- (b) room naming: text only, and with a crop montage
    room_rows = []
    for truth, objs, accept in ROOM_CASES:
        text = ', '.join(f'{n} {k}' if n > 1 else k for k, n in objs.items())
        p = prompts.render('name_region', area_m2='14', shape='roughly compact', objects=text,
                           room_types=', '.join(cfg.vlm.room_types), image_hint='')
        t0 = time.time()
        ans = a.call('room', 'room', p, [], f'live:room:{truth}')
        room_rows.append({'truth': truth, 'objects': text, 'answer': ans, 'ok': bool(ans and ans['room_type'] in accept),
                          'latency_s': round(time.time() - t0, 2)})
    mont = ann.montage([crop_jpeg(CROPS[i][1], CROPS[i][2]) for i in (6, 7, 9, 5)])
    p = prompts.render('name_region', area_m2='40', shape='roughly compact', objects='4 person, vending machine, monitor',
                       room_types=', '.join(cfg.vlm.room_types), image_hint='The image shows crops of some of these objects.')
    ans = a.call('room', 'room', p, [mont], 'live:room:lobby')
    room_rows.append({'truth': 'lobby/hall (reception area)', 'objects': 'person x4, vending machine, monitor (montage)',
                      'answer': ans, 'ok': bool(ans and ans['room_type'] in ('lobby', 'hall', 'office', 'corridor'))})
    out['rooms'] = {'accuracy': f"{sum(r['ok'] for r in room_rows)}/{len(room_rows)}", 'rows': room_rows}

    # ---- (c) find on a map built from the crops, captioned by the live VLM
    m, ids = build_map(cfg)
    a2 = ann.Annotator(m, cfg.vlm, client)
    t0 = time.time()
    n_calls = 0
    for _ in range(60):
        k = a2.step()
        n_calls += k
        if k == 0 and a2._next_job() is None:
            break
    out['annotation_pass'] = {'objects': len(m.objects), 'captioned': sum(1 for o in m.objects.values() if o.caption),
                              'calls': n_calls, 'seconds': round(time.time() - t0, 1)}
    id_name = {v: k for k, v in ids.items()}
    find_rows = []
    for text, truth in FIND_CASES:
        truths = {truth} if isinstance(truth, str) else set(truth)
        t0 = time.time()
        res = a2.find(text, (0.0, 0.0, 0.0), limit=3, timeout_s=40.0)
        top = [id_name.get(c['id'], '?') for c in res['matches']]
        heur = [id_name.get(c['id'], '?') for c in
                __import__('x30_semantic_map.finder', fromlist=['rank']).rank(m, text, (0, 0, 0), 3)['matches']]
        find_rows.append({'request': text, 'method': res['method'], 'top3': top, 'top1_ok': bool(top and top[0] in truths),
                          'heuristic_top3': heur, 'heuristic_top1_ok': bool(heur and heur[0] in truths),
                          'vlm_calls': res['vlm_calls'], 'latency_s': round(time.time() - t0, 2)})
        print(find_rows[-1], flush=True)
    out['find'] = {'top1_vlm': f"{sum(r['top1_ok'] for r in find_rows)}/{len(find_rows)}",
                   'top1_heuristic_only': f"{sum(r['heuristic_top1_ok'] for r in find_rows)}/{len(find_rows)}", 'rows': find_rows}
    out['call_log'] = {'caption_pass': a.calls.summary(), 'map_pass': a2.calls.summary()}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('caption', 'caption_latency_s', 'annotation_pass')}, indent=1))
    print('rooms', out['rooms']['accuracy'], 'find', out['find']['top1_vlm'], 'heuristic only', out['find']['top1_heuristic_only'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
