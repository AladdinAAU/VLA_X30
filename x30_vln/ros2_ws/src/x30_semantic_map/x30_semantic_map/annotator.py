"""Asynchronous model-based annotator of the semantic map (no ROS imports).

A background worker calls the *mapper VLM* (an OpenAI-compatible endpoint from configs/models.yaml, through
x30_common.VLMClient) to
  (a) caption confirmed objects from their best crop and extract attributes (colour, material, state), and check the label,
  (b) name rooms/regions from their objects (plus a montage of crops),
  (c) judge doubtful pairs of instances (same physical object or not),
  (d) answer `find` requests by ranking candidate instances, optionally re-checking the top ones with their crops.
It never blocks mapping: jobs are built from short snapshots taken under the map lock, the call happens outside the lock,
and results are applied under the lock afterwards. Calls are rate and budget limited, cached by (task, object, view
hash), retried, validated, and logged. When the endpoint is down a circuit breaker stops the calls and the map keeps
working without captions; `find` then answers from the model-free ranking of finder.py.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable

import cv2
import numpy as np

from . import attributes as attr
from . import finder, rooms
from .config import VlmCfg
from .semantic_map import MapObject, SemanticMap

try:                                               # the shared client of the framework
    from x30_common.jsonrepair import loads_lenient
    from x30_common.vlm_client import VLMClient, VLMError, VLMHTTPError
except ImportError:                                # pragma: no cover - only without the x30_common package
    VLMClient = None                               # type: ignore[assignment,misc]
    VLMError = VLMHTTPError = Exception            # type: ignore[assignment,misc]
    loads_lenient = None                           # type: ignore[assignment]

PROMPT_DIR = Path(__file__).resolve().parent / 'prompts'
log = logging.getLogger('x30_semantic_map.annotator')

# JSON schemas used for constrained decoding (response_format json_schema) and for validation hints
SCHEMAS: dict[str, dict[str, Any]] = {
    'caption': {'type': 'object', 'properties': {
        'caption': {'type': 'string'}, 'label_matches': {'type': 'boolean'}, 'label': {'type': 'string'},
        'colour': {'type': 'string'}, 'material': {'type': 'string'}, 'state': {'type': 'string'}},
        'required': ['caption', 'label_matches']},
    'room': {'type': 'object', 'properties': {
        'room_type': {'type': 'string'}, 'confidence': {'type': 'number'}, 'reason': {'type': 'string'}},
        'required': ['room_type']},
    'disambiguate': {'type': 'object', 'properties': {
        'same_object': {'type': 'boolean'}, 'confidence': {'type': 'number'}}, 'required': ['same_object']},
    'find': {'type': 'object', 'properties': {
        'ranking': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'id': {'type': 'integer'}, 'score': {'type': 'number'}}, 'required': ['id', 'score']}},
        'reason': {'type': 'string'}}, 'required': ['ranking']},
    'verify': {'type': 'object', 'properties': {
        'match': {'type': 'boolean'}, 'confidence': {'type': 'number'}}, 'required': ['match']},
}


# ----------------------------------------------------------------------------------------------- validation
class InvalidAnswer(ValueError):
    """The model answered, but not with the expected structure."""


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ('true', 'yes', 'y', '1'):
        return True
    if isinstance(v, str) and v.strip().lower() in ('false', 'no', 'n', '0'):
        return False
    raise InvalidAnswer(f'not a boolean: {v!r}')


def _as_float01(v: Any, default: float = 0.5) -> float:
    if v is None or v == '':
        return default
    try:
        return float(min(max(float(v), 0.0), 1.0))
    except (TypeError, ValueError) as ex:
        raise InvalidAnswer(f'not a number: {v!r}') from ex


def _word(v: Any, max_words: int = 2) -> str:
    if not isinstance(v, str):
        return ''
    w = ' '.join(v.lower().strip().strip('.').split()[:max_words])
    return '' if w in ('none', 'n/a', 'unknown', 'null', 'not applicable') else w


def _obj(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvalidAnswer(f'expected a JSON object, got {type(data).__name__}')
    return data


def validate_caption(data: Any) -> dict[str, Any]:
    d = _obj(data)
    cap = d.get('caption')
    if not isinstance(cap, str) or not cap.strip():
        raise InvalidAnswer('missing caption')
    return {'caption': cap.strip()[:200], 'label_matches': _as_bool(d.get('label_matches', True)),
            'label': _word(d.get('label')), 'colour': _word(d.get('colour'), 1), 'material': _word(d.get('material'), 1),
            'state': _word(d.get('state'), 1)}


def validate_room(data: Any, room_types: list[str]) -> dict[str, Any]:
    d = _obj(data)
    rt = d.get('room_type')
    if not isinstance(rt, str) or not rt.strip():
        raise InvalidAnswer('missing room_type')
    rt = rt.lower().strip()
    allowed = [r.lower() for r in room_types]
    if rt not in allowed:
        match = [a for a in allowed if a in rt or rt in a]
        if not match:
            raise InvalidAnswer(f'room_type {rt!r} not in the list')
        rt = match[0]
    return {'room_type': rt, 'confidence': _as_float01(d.get('confidence')), 'reason': str(d.get('reason', ''))[:120]}


def validate_same(data: Any) -> dict[str, Any]:
    d = _obj(data)
    if 'same_object' not in d:
        raise InvalidAnswer('missing same_object')
    return {'same_object': _as_bool(d['same_object']), 'confidence': _as_float01(d.get('confidence'))}


def validate_ranking(data: Any, valid_ids: set[int]) -> dict[str, Any]:
    d = _obj(data)
    rk = d.get('ranking')
    if not isinstance(rk, list):
        raise InvalidAnswer('missing ranking list')
    out, seen = [], set()
    for item in rk:
        if not isinstance(item, dict):
            continue
        try:
            oid = int(item.get('id'))
        except (TypeError, ValueError):
            continue
        if oid in valid_ids and oid not in seen:           # ids the model invented are dropped
            seen.add(oid)
            out.append({'id': oid, 'score': _as_float01(item.get('score'), 0.5)})
    out.sort(key=lambda r: -r['score'])
    return {'ranking': out, 'reason': str(d.get('reason', ''))[:160]}


def validate_verify(data: Any) -> dict[str, Any]:
    d = _obj(data)
    if 'match' not in d:
        raise InvalidAnswer('missing match')
    return {'match': _as_bool(d['match']), 'confidence': _as_float01(d.get('confidence'))}


# ----------------------------------------------------------------------------------------------- limits, log
class Budget:
    """Rate limits and totals. Thread-safe; time comes from `clock`."""

    def __init__(self, cfg: VlmCfg, clock: Callable[[], float]):
        self.cfg, self.clock = cfg, clock
        self.lock = threading.Lock()
        self.calls: collections.deque[float] = collections.deque()
        self.last = -1e9
        self.total_calls = 0
        self.total_completion_tokens = 0

    def check(self, urgent: bool = False) -> tuple[bool, str]:
        now = self.clock()
        with self.lock:
            while self.calls and now - self.calls[0] > 60.0:
                self.calls.popleft()
            if self.total_calls >= self.cfg.max_calls_total:
                return False, 'call budget exhausted'
            if self.total_completion_tokens >= self.cfg.max_completion_tokens_total:
                return False, 'token budget exhausted'
            if not urgent and len(self.calls) >= self.cfg.max_calls_per_min:   # user-facing find calls skip the rate
                return False, 'rate limit (calls per minute)'                   # limits but not the totals
            if not urgent and now - self.last < self.cfg.min_interval_s:
                return False, 'rate limit (interval)'
            return True, ''

    def record(self, completion_tokens: int) -> None:
        now = self.clock()
        with self.lock:
            self.calls.append(now)
            self.last = now
            self.total_calls += 1
            self.total_completion_tokens += int(completion_tokens)


class CallLog:
    """Every VLM call (prompt hash, latency, tokens) in memory and, if a path is set, as JSON lines."""

    def __init__(self, path: str = '', keep: int = 2000):
        self.path = Path(path) if path else None
        self.records: collections.deque[dict[str, Any]] = collections.deque(maxlen=keep)
        self.lock = threading.Lock()

    def add(self, rec: dict[str, Any]) -> None:
        with self.lock:
            self.records.append(rec)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(rec, sort_keys=True) + '\n')
                except OSError as ex:
                    log.warning('cannot write the VLM call log: %s', ex)

    def summary(self) -> dict[str, Any]:
        with self.lock:
            recs = list(self.records)
        real = [r for r in recs if not r.get('cached')]
        ok = [r for r in real if r.get('ok')]
        lat = sorted(r['latency_ms'] for r in ok)

        def pct(p: float) -> float:
            return float(lat[min(len(lat) - 1, int(p * len(lat)))]) if lat else 0.0
        by_task: dict[str, int] = {}
        for r in real:
            by_task[r['task']] = by_task.get(r['task'], 0) + 1
        return {'calls': len(real), 'ok': len(ok), 'failed': len(real) - len(ok),
                'cache_hits': len(recs) - len(real), 'by_task': by_task,
                'latency_ms_mean': float(np.mean(lat)) if lat else 0.0, 'latency_ms_p50': pct(0.5),
                'latency_ms_p95': pct(0.95), 'prompt_tokens': sum(r.get('prompt_tokens', 0) for r in real),
                'completion_tokens': sum(r.get('completion_tokens', 0) for r in real)}


# ----------------------------------------------------------------------------------------------- prompts
class Prompts:
    def __init__(self, directory: Path = PROMPT_DIR):
        self.dir = directory
        self._cache: dict[str, str] = {}

    def text(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = (self.dir / f'{name}.txt').read_text(encoding='utf-8').strip()
        return self._cache[name]

    def render(self, task: str, **values: Any) -> tuple[str, str]:
        user = Template(self.text(f'{task}.user')).safe_substitute({k: str(v) for k, v in values.items()})
        return self.text(f'{task}.system'), user


def montage(jpegs: list[bytes], tile: int = 160, cols: int = 2) -> bytes:
    """Grid of crops (each resized into a tile x tile cell) as one JPEG."""
    tiles = []
    for b in jpegs:
        rgb = attr.decode_jpeg(b)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        s = tile / max(h, w)
        small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        cell = np.full((tile, tile, 3), 235, np.uint8)
        cell[:small.shape[0], :small.shape[1]] = small
        tiles.append(cell)
    if not tiles:
        return b''
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 235, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return attr.encode_jpeg(np.vstack(rows), 85)


# ----------------------------------------------------------------------------------------------- annotator
@dataclass
class Job:
    task: str                       # caption | room | disambiguate
    key: str                        # cache key
    target: Any                     # object id, region id or (id_a, id_b)
    images: list[bytes] = field(default_factory=list)
    prompt: tuple[str, str] = ('', '')
    context: dict[str, Any] = field(default_factory=dict)


class Annotator:
    def __init__(self, smap: SemanticMap, cfg: VlmCfg | None = None, client: Any = None,
                 prompts: Prompts | None = None, clock: Callable[[], float] = time.time,
                 pose_fn: Callable[[], tuple[float, float, float] | None] | None = None):
        self.smap = smap
        self.cfg = cfg or smap.cfg.vlm
        self.client = client
        self.prompts = prompts or Prompts()
        self.clock = clock
        self.pose_fn = pose_fn or (lambda: None)
        self.budget = Budget(self.cfg, clock)
        self.calls = CallLog(self.cfg.log_path)
        self.cache: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self.failures: dict[str, int] = {}
        self.region_last: dict[int, float] = {}
        self.consec_fail = 0
        self.down_until = 0.0
        self.backoff = self.cfg.backoff_s
        self.last_error = ''
        self.response_format = self.cfg.response_format
        self.slot = threading.Semaphore(1)                  # one call at a time (one GPU, llama.cpp --parallel 1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.counters = {'captions': 0, 'rooms': 0, 'disambiguations': 0, 'finds': 0, 'merged': 0, 'relabelled': 0,
                         'demoted': 0, 'invalid': 0}

    # ------------------------------------------------------------------------------------ lifecycle
    def start(self, poll_s: float = 0.5) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(poll_s):
                try:
                    self.step()
                except Exception:                           # the worker must never die
                    log.exception('annotator step failed')
        self._thread = threading.Thread(target=loop, name='semantic-annotator', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self.client is not None and self.clock() >= self.down_until)

    def status(self) -> dict[str, Any]:
        return {'enabled': self.cfg.enabled, 'available': self.available, 'down_until': self.down_until,
                'consecutive_failures': self.consec_fail, 'last_error': self.last_error,
                'budget_calls': self.budget.total_calls, 'budget_completion_tokens': self.budget.total_completion_tokens,
                'cache_size': len(self.cache), 'counters': dict(self.counters), 'log': self.calls.summary()}

    # ------------------------------------------------------------------------------------ scheduling
    def step(self) -> int:
        """One scheduling pass: run at most one background job. Returns the number of VLM calls made (0 or 1+)."""
        if not self.available:
            return 0
        ok, _ = self.budget.check()
        if not ok:
            return 0
        job = self._next_job()
        if job is None:
            return 0
        if job.key in self.cache:                           # same task on the same view: reuse the answer
            self._apply(job, self.cache[job.key])
            self.calls.add({'t': self.clock(), 'task': job.task, 'key': job.key, 'cached': True, 'ok': True})
            return 0
        if not self.slot.acquire(blocking=False):
            return 0
        try:
            answer = self._run(job)
        finally:
            self.slot.release()
        if answer is not None:
            self._remember(job.key, answer)
            self._apply(job, answer)
        else:
            self.failures[job.key] = self.failures.get(job.key, 0) + 1
        return 1

    def _next_job(self) -> Job | None:
        c = self.cfg
        with self.smap.lock:
            if c.caption_objects:
                job = self._caption_job()
                if job:
                    return job
            if c.name_rooms:
                job = self._room_job()
                if job:
                    return job
            if c.disambiguate:
                job = self._pair_job()
                if job:
                    return job
        return None

    def _caption_job(self) -> Job | None:
        c = self.cfg
        pose = self.pose_fn()
        cand = []
        for o in self.smap.visible_objects():
            if o.bearing_only or not o.views or o.n_obs < c.caption_min_obs or o.captions_done >= c.max_captions_per_object:
                continue
            view = o.views[0]
            fresh = o.captions_done == 0
            better = o.caption_quality > 0 and view.quality >= c.recaption_quality_gain * o.caption_quality
            key = f'caption:{o.id}:{view.phash}:{self.smap.label(o)}'
            if not (fresh or better) or self.failures.get(key, 0) >= 2:
                continue
            dist = 0.0 if pose is None else float(np.hypot(o.position[0] - pose[0], o.position[1] - pose[1]))
            cand.append((0 if fresh else 1, dist, o.id, o, view, key))
        if not cand:
            return None
        _, _, _, o, view, key = min(cand, key=lambda x: x[:3])
        room = self.smap.room_name_of(o)
        sys_p, user_p = self.prompts.render(
            'caption_object', label=self.smap.label(o), size=attr.size_class(o.size, self.smap.cfg.size),
            room_hint=f', in the {room}' if room else '')
        return Job('caption', key, o.id, [view.jpeg], (sys_p, user_p), {'quality': view.quality})

    def _room_job(self) -> Job | None:
        now = self.clock()
        by_region = rooms.objects_by_region(self.smap)
        for rid, region in sorted(self.smap.regions.items()):
            members = by_region.get(rid, [])
            if len(members) < 2 or now - self.region_last.get(rid, -1e9) < 30.0:
                continue
            sig = rooms.label_signature(self.smap, members)
            if region.vlm_name_key == sig:
                continue
            key = f'room:{rid}:{sig}'
            if self.failures.get(key, 0) >= 2:
                continue
            counts = collections.Counter(self.smap.label(o) for o in members)
            objects = ', '.join(f'{n} {k}' if n > 1 else k for k, n in counts.most_common(12))
            shape = ('long and narrow' if region.width_m < 2.2 and region.length_m > 2.5 * max(region.width_m, 0.1)
                     else 'roughly compact')
            crops = [o.views[0].jpeg for o in members if o.views][:4]
            img = montage(crops) if crops else b''
            sys_p, user_p = self.prompts.render(
                'name_region', area_m2=f'{region.area_m2:.0f}', shape=shape, objects=objects,
                room_types=', '.join(self.cfg.room_types),
                image_hint='The image shows crops of some of these objects.' if img else '')
            return Job('room', key, rid, [img] if img else [], (sys_p, user_p), {'sig': sig})
        return None

    def _pair_job(self) -> Job | None:
        for a_id, b_id, dist in self.smap.doubtful_pairs(3):
            key = f'pair:{a_id}:{b_id}:{self.smap.objects[a_id].views[0].phash}:{self.smap.objects[b_id].views[0].phash}'
            if self.failures.get(key, 0) >= 2:
                continue
            a, b = self.smap.objects[a_id], self.smap.objects[b_id]
            sys_p, user_p = self.prompts.render('disambiguate', id_a=a_id, id_b=b_id, label=self.smap.label(a),
                                                dist=f'{dist:.2f}')
            return Job('disambiguate', key, (a_id, b_id), [a.views[0].jpeg, b.views[0].jpeg], (sys_p, user_p))
        return None

    # ------------------------------------------------------------------------------------ calling
    def _validate(self, task: str, data: Any, extra: Any = None) -> dict[str, Any]:
        if task == 'caption':
            return validate_caption(data)
        if task == 'room':
            return validate_room(data, self.cfg.room_types)
        if task == 'disambiguate':
            return validate_same(data)
        if task == 'find':
            return validate_ranking(data, extra)
        if task == 'verify':
            return validate_verify(data)
        raise ValueError(task)

    def _messages(self, prompt: tuple[str, str], images: list[bytes]) -> list[dict]:
        from x30_common.images import image_part
        content: list[dict] = [image_part(i, self.cfg.max_image_px, self.cfg.jpeg_quality) for i in images if i]
        content.append({'type': 'text', 'text': prompt[1]})
        return [{'role': 'system', 'content': prompt[0]}, {'role': 'user', 'content': content}]

    def _extra_body(self, schema_name: str) -> dict[str, Any]:
        if self.response_format == 'json_object':
            return {'response_format': {'type': 'json_object'}}
        if self.response_format == 'json_schema':
            return {'response_format': {'type': 'json_schema',
                                        'json_schema': {'name': schema_name, 'schema': SCHEMAS[schema_name]}}}
        return {}

    def call(self, task: str, schema: str, prompt: tuple[str, str], images: list[bytes], key: str, extra: Any = None,
             timeout_s: float | None = None, urgent: bool = False) -> dict[str, Any] | None:
        """One validated, logged VLM call (with one repair retry on an invalid answer). None on any failure."""
        ok, why = self.budget.check(urgent)
        if not ok or not self.available:
            self.last_error = why or 'endpoint unavailable'
            return None
        messages = self._messages(prompt, images)
        prompt_hash = hashlib.sha1((prompt[0] + prompt[1] + ''.join(hashlib.sha1(i).hexdigest() for i in images))
                                   .encode()).hexdigest()[:12]
        attempts = 1 + max(0, self.cfg.retries)
        for attempt in range(attempts):
            t0 = time.time()
            rec: dict[str, Any] = {'t': self.clock(), 'task': task, 'key': key, 'prompt_hash': prompt_hash,
                                   'n_images': len(images), 'attempt': attempt, 'cached': False}
            try:
                res = self.client.chat(messages, max_tokens=self.cfg.max_tokens,
                                       temperature=self.cfg.temperature if attempt == 0 else 0.0,
                                       timeout_s=timeout_s or self.cfg.timeout_s, extra=self._extra_body(schema))
            except VLMHTTPError as ex:
                rec.update(ok=False, error=str(ex)[:200], latency_ms=round((time.time() - t0) * 1000))
                self.calls.add(rec)
                if getattr(ex, 'status', 0) == 400 and self.response_format != 'none':
                    log.warning('endpoint rejected response_format=%s; continuing without it', self.response_format)
                    self.response_format = 'none'
                    continue
                self._failed(str(ex))
                return None
            except (VLMError, OSError) as ex:
                rec.update(ok=False, error=f'{type(ex).__name__}: {ex}'[:200], latency_ms=round((time.time() - t0) * 1000))
                self.calls.add(rec)
                self._failed(rec['error'])
                return None
            rec.update(latency_ms=round((time.time() - t0) * 1000), prompt_tokens=res.prompt_tokens,
                       completion_tokens=res.completion_tokens)
            self.budget.record(res.completion_tokens)
            try:
                data, _ = loads_lenient(res.content)
                answer = self._validate(task, data, extra)
            except (ValueError, InvalidAnswer) as ex:
                rec.update(ok=False, error=f'invalid answer: {ex}'[:200])
                self.calls.add(rec)
                self.counters['invalid'] += 1
                continue
            rec.update(ok=True)
            self.calls.add(rec)
            self.consec_fail = 0
            self.backoff = self.cfg.backoff_s
            return answer
        self.last_error = 'invalid answers'
        return None

    def _failed(self, msg: str) -> None:
        self.last_error = msg
        self.consec_fail += 1
        if self.consec_fail >= self.cfg.fail_threshold:
            self.down_until = self.clock() + self.backoff
            log.warning('mapper VLM unavailable (%s); pausing annotation for %.0f s', msg, self.backoff)
            self.backoff = min(self.backoff * 2.0, self.cfg.backoff_max_s)

    def _run(self, job: Job) -> dict[str, Any] | None:
        schema = {'caption': 'caption', 'room': 'room', 'disambiguate': 'disambiguate'}[job.task]
        return self.call(job.task, schema, job.prompt, job.images, job.key)

    def _remember(self, key: str, answer: dict[str, Any]) -> None:
        self.cache[key] = answer
        self.cache.move_to_end(key)
        while len(self.cache) > self.cfg.cache_size:
            self.cache.popitem(last=False)

    # ------------------------------------------------------------------------------------ applying results
    def _apply(self, job: Job, ans: dict[str, Any]) -> None:
        with self.smap.lock:
            if job.task == 'caption':
                self._apply_caption(job, ans)
            elif job.task == 'room':
                self._apply_room(job, ans)
            elif job.task == 'disambiguate':
                self._apply_pair(job, ans)

    def _apply_caption(self, job: Job, ans: dict[str, Any]) -> None:
        o = self.smap.objects.get(job.target)
        if o is None:
            return
        self.counters['captions'] += 1
        o.captions_done += 1
        o.caption_quality = float(job.context.get('quality', 0.0))
        if ans['caption'] != 'unclear':
            o.caption = ans['caption']
        for k in ('colour', 'material', 'state'):
            if ans[k]:
                o.vlm_attrs[k] = ans[k]
        if ans['label_matches']:
            o.label_ok_votes += 1
            return
        o.label_bad_votes += 1
        if o.label_bad_votes >= self.cfg.rejected_label_votes and o.label_bad_votes > o.label_ok_votes:
            new = ans['label']
            if new and new in self.smap.vocab.known_labels() and self.smap.vocab.compat(new, self.smap.label(o)) < 2:
                self.smap.relabel_object(o.id, new)
                o.label_bad_votes = o.label_ok_votes = 0
                self.counters['relabelled'] += 1
            else:
                self.smap.reject_object(o.id, 'vlm: label does not match the crop')
                self.counters['demoted'] += 1

    def _apply_room(self, job: Job, ans: dict[str, Any]) -> None:
        region = self.smap.regions.get(job.target)
        if region is None:
            return
        self.counters['rooms'] += 1
        self.region_last[region.id] = self.clock()
        region.vlm_name_key = job.context['sig']
        if ans['confidence'] >= 0.5:
            region.name, region.name_source = ans['room_type'], 'vlm'

    def _apply_pair(self, job: Job, ans: dict[str, Any]) -> None:
        a_id, b_id = job.target
        a, b = self.smap.objects.get(a_id), self.smap.objects.get(b_id)
        if a is None or b is None:
            return
        self.counters['disambiguations'] += 1
        if ans['same_object'] and ans['confidence'] >= 0.6:
            self.smap.merge_objects(min(a_id, b_id), max(a_id, b_id))
            self.counters['merged'] += 1
        elif not ans['same_object']:
            a.distinct_from.add(b_id)
            b.distinct_from.add(a_id)

    # ------------------------------------------------------------------------------------ find
    def find(self, text: str, pose: tuple[float, float, float] | None = None, limit: int = 5,
             timeout_s: float | None = None) -> dict[str, Any]:
        """Rank map objects for a text request. VLM re-ranking with optional crop re-checks inside `timeout_s`;
        the model-free ranking is returned (method 'heuristic') whenever the VLM cannot answer."""
        t_start = time.time()
        timeout_s = self.cfg.find_timeout_s if timeout_s is None else timeout_s
        deadline = t_start + timeout_s
        base = finder.rank(self.smap, text, pose, limit=max(limit, self.cfg.find_max_candidates))
        base['vlm_calls'] = 0
        out = dict(base)
        if not self.available:
            out.update(method='heuristic', vlm='unavailable', matches=base['matches'][:limit])
            return out
        self.counters['finds'] += 1
        with self.smap.lock:
            cands = list(base['matches'][:self.cfg.find_max_candidates])
            if not cands:                                    # nothing matched by words: let the VLM read all objects
                cands = [self.smap.describe(o, pose) for o in
                         self.smap.visible_objects()[:self.cfg.find_max_candidates]]
            rows = [self._candidate_row(c) for c in cands]
        if not cands:
            out.update(method='heuristic', matches=[])
            return out
        pose_hint = '' if pose is None else (f'Robot at x={pose[0]:.1f}, y={pose[1]:.1f}, heading '
                                             f'{np.degrees(pose[2]) % 360:.0f} deg; bearings are + to the left.')
        prompt = self.prompts.render('find_rank', text=text, pose_hint=pose_hint, candidates='\n'.join(rows))
        left = max(1.0, deadline - time.time())
        if not self.slot.acquire(timeout=left):
            out.update(method='heuristic', vlm='busy', matches=base['matches'][:limit])
            return out
        try:
            valid = {c['id'] for c in cands}
            key = 'find:' + hashlib.sha1((text + ','.join(map(str, sorted(valid)))).encode()).hexdigest()[:12]
            ans = self.call('find', 'find', prompt, [], key, extra=valid, timeout_s=max(1.0, deadline - time.time()),
                            urgent=True)
            out['vlm_calls'] = 1
            if ans is None:
                out.update(method='heuristic', vlm=self.last_error, matches=base['matches'][:limit])
                return out
            by_id = {c['id']: c for c in cands}
            scored = []
            for r in ans['ranking']:
                c = dict(by_id[r['id']])
                c.update(vlm_score=r['score'], score=round(0.7 * r['score'] + 0.3 * c.get('score', 0.0), 3))
                scored.append(c)
            matches = sorted(scored, key=lambda c: -c['score'])
            # crop re-check of the best candidates
            for c in matches[:self.cfg.find_recheck_top_k]:
                if time.time() > deadline - 1.5:
                    break
                o = self.smap.objects.get(c['id'])
                if o is None or not o.views:
                    continue
                p2 = self.prompts.render('verify_candidate', text=text, id=o.id, label=self.smap.label(o))
                v = self.call('verify', 'verify', p2, [o.views[0].jpeg], f'verify:{o.id}:{o.views[0].phash}:{text}',
                              timeout_s=max(1.0, deadline - time.time()), urgent=True)
                out['vlm_calls'] += 1
                if v is None:
                    break
                c['crop_check'] = v
                c['score'] = round(c['score'] * (1.0 if v['match'] else 0.35), 3) if not v['match'] else \
                    round(max(c['score'], 0.5 + 0.5 * v['confidence']), 3)
            matches.sort(key=lambda c: (-c['score'], c['id']))
            rest = [c for c in base['matches'] if c['id'] not in {m['id'] for m in matches}]
            out.update(method='vlm', matches=(matches + [dict(c, score=round(c['score'] * 0.2, 3)) for c in rest])[:limit],
                       reason=ans['reason'])
            return out
        finally:
            self.slot.release()
            out['latency_ms'] = round((time.time() - t_start) * 1000)

    def _candidate_row(self, c: dict[str, Any]) -> str:
        attrs = ', '.join(c.get('attributes', [])) or '-'
        rng = f"{c['range_m']} m, {c['bearing_deg']:+.0f} deg" if 'range_m' in c else '-'
        rel = ''
        try:
            from .relations import object_relations
            r = object_relations(self.smap, c['id'], self.pose_fn(), limit=3)
            rel = '; '.join(f"{'/'.join(x['relations']) or 'near'} #{x['id']} {x['label']}" for x in r['relations'][:3])
        except KeyError:
            pass
        return f"{c['id']} | {c['label']} | {attrs} | {c.get('caption') or '-'} | {c.get('room') or '-'} | {rng} | {rel or '-'}"
