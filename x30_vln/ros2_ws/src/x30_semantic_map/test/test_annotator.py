"""Annotator tests against a fake OpenAI-compatible server (no GPU, no ROS)."""
import json
import time

import numpy as np
import pytest
from fake_vlm import FakeVLM
from helpers import det, feed, make_map

from x30_common.vlm_client import VLMClient
from x30_semantic_map import annotator as ann
from x30_semantic_map import rooms
from x30_semantic_map import synthetic as syn
from x30_semantic_map.config import load_config
from x30_semantic_map.semantic_map import SemanticMap


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture()
def fake():
    f = FakeVLM().start()
    yield f
    f.stop()


def scenario_map(frames=330, seed=0):
    sc = syn.generate(syn.ScenarioConfig(seed=seed))
    cfg = load_config()
    m = SemanticMap(cfg)
    m.los_fn = lambda a, b: sc.grid.line_of_sight(a, b, ignore_end_m=0.45)
    for f in sc.frames[:frames]:
        m.process_frame(f)
    return m, sc


def make_annotator(m, fake, clock=None, **cfg):
    for k, v in cfg.items():
        setattr(m.cfg.vlm, k, v)
    m.cfg.vlm.min_interval_s = cfg.get('min_interval_s', 0.0)
    client = VLMClient(fake.url, 'fake', max_tokens=100, timeout_s=5, max_retries=0, backoff_s=0.01)
    return ann.Annotator(m, m.cfg.vlm, client, clock=clock or Clock())


def run_all(a, limit=200):
    n = 0
    for _ in range(limit):
        k = a.step()
        n += k
        if k == 0 and not a._next_job():
            break
    return n


def test_validators_accept_good_and_reject_bad_answers():
    assert ann.validate_caption({'caption': ' A red chair. ', 'label_matches': 'yes', 'colour': 'Red.', 'material': 'none'}) \
        == {'caption': 'A red chair.', 'label_matches': True, 'label': '', 'colour': 'red', 'material': '', 'state': ''}
    for bad in ({}, {'caption': ''}, [], {'caption': 'x', 'label_matches': 'maybe'}):
        with pytest.raises(ann.InvalidAnswer):
            ann.validate_caption(bad)
    rooms_ok = ['kitchen', 'living room']
    assert ann.validate_room({'room_type': 'Living Room', 'confidence': 2}, rooms_ok)['confidence'] == 1.0
    assert ann.validate_room({'room_type': 'the kitchen area'}, rooms_ok)['room_type'] == 'kitchen'
    with pytest.raises(ann.InvalidAnswer):
        ann.validate_room({'room_type': 'spaceship'}, rooms_ok)
    r = ann.validate_ranking({'ranking': [{'id': 3, 'score': 0.4}, {'id': 99, 'score': 1}, {'id': '5', 'score': 0.9},
                                          {'id': 3, 'score': 0.9}, 'junk']}, {3, 5})
    assert [x['id'] for x in r['ranking']] == [5, 3]
    with pytest.raises(ann.InvalidAnswer):
        ann.validate_ranking({'ranking': 'none'}, {1})
    with pytest.raises(ann.InvalidAnswer):
        ann.validate_same({'confidence': 1})


def test_prompts_live_in_files_and_render():
    p = ann.Prompts()
    sys_p, user_p = p.render('caption_object', label='chair', size='medium', room_hint=', in the kitchen')
    assert 'chair' in user_p and 'in the kitchen' in user_p and '$' not in user_p
    for task in ('caption_object', 'name_region', 'disambiguate', 'find_rank', 'verify_candidate'):
        p.render(task)


def test_captions_are_applied_cached_and_logged(fake, tmp_path):
    m, _ = scenario_map()
    a = make_annotator(m, fake, log_path=str(tmp_path / 'vlm.jsonl'))
    a.calls = ann.CallLog(str(tmp_path / 'vlm.jsonl'))
    n = run_all(a)
    cap = [o for o in m.visible_objects() if o.caption]
    assert n > 5 and len(cap) > 5
    o = cap[0]
    assert o.caption.startswith('a ') and o.vlm_attrs['material'] == 'plastic'
    assert 'plastic' in m.attribute_list(o) and m.attribute_dict(o)['colour'] == 'red'
    calls_before = len(fake.calls)
    assert run_all(a) == 0 and len(fake.calls) == calls_before         # nothing new to caption
    # same view hash and label: a reset caption is answered from the cache, without a call
    key_obj = cap[0]
    key_obj.captions_done, key_obj.caption = 0, ''
    a.step()
    assert len(fake.calls) == calls_before and key_obj.caption
    lines = [json.loads(x) for x in (tmp_path / 'vlm.jsonl').read_text().splitlines()]
    real = [x for x in lines if not x['cached']]
    assert real and all({'prompt_hash', 'latency_ms', 'prompt_tokens', 'completion_tokens', 'task'} <= x.keys() for x in real)
    assert any(x['cached'] for x in lines)
    s = a.status()['log']
    assert s['calls'] == len(real) and s['latency_ms_mean'] > 0 and s['prompt_tokens'] > 0


def test_images_are_sent_downscaled(fake):
    m, _ = scenario_map()
    a = make_annotator(m, fake, max_image_px=48)
    a.step()
    c = fake.calls[0]
    assert c['n_images'] == 1
    assert 'response_format' in c['body']


def test_label_mismatch_relabels_or_demotes(fake):
    def responder(text, n, body):
        if 'crop of one object' in text:
            return json.dumps({'caption': 'a wooden table', 'label_matches': False, 'label': 'table', 'colour': 'brown'})
        return '{}'
    fake.responder = responder
    m = make_map(vlm__enabled=True)
    from helpers import red_crop
    rng = np.random.default_rng(0)
    feed(m, lambda i: [det('chair', (1, 1, 0.45), crop=red_crop(rng), box=(0, 0, 80, 80))], n=8)
    (o,) = m.visible_objects()
    a = make_annotator(m, fake, max_captions_per_object=3, recaption_quality_gain=0.0)
    for _ in range(3):
        a.step()
        o.captions_done = 0
        o.caption_quality = 0.0
    assert m.label(o) == 'table' and a.counters['relabelled'] == 1
    fake.responder = lambda t, n, b: json.dumps({'caption': 'a poster', 'label_matches': False, 'label': 'poster'})
    o.votes = {'chair': 5.0}
    o.label_bad_votes = o.label_ok_votes = 0
    for _ in range(3):
        o.captions_done = 0
        o.caption_quality = 0.0
        a.cache.clear()
        a.step()
    assert o.id not in m.objects and a.counters['demoted'] == 1


def test_invalid_answer_is_retried_once_then_applied(fake):
    fake.garbage_first = 1
    m, _ = scenario_map()
    a = make_annotator(m, fake, retries=1)
    assert a.step() == 1
    assert a.counters['invalid'] == 1 and a.counters['captions'] == 1
    assert len(fake.calls) == 2


def test_fenced_and_think_wrapped_json_is_understood(fake):
    fake.responder = lambda t, n, b: '<think>hmm</think>```json\n{"caption": "a chair", "label_matches": true,}\n```'
    m, _ = scenario_map()
    a = make_annotator(m, fake)
    a.step()
    assert a.counters['captions'] == 1


def test_endpoint_rejecting_response_format_is_handled(fake):
    fake.reject_response_format = True
    m, _ = scenario_map()
    a = make_annotator(m, fake, retries=1)
    assert a.step() == 1 and a.response_format == 'none' and a.counters['captions'] == 1


def test_circuit_breaker_opens_and_recovers_and_map_keeps_working(fake):
    clock = Clock()
    m, sc = scenario_map()
    a = make_annotator(m, fake, clock, fail_threshold=2, backoff_s=30.0)
    fake.fail_status = 503
    for _ in range(3):
        a.step()
    assert not a.available and a.consec_fail >= 2 and '503' in a.last_error
    n_calls = len(fake.calls)
    a.step()
    assert len(fake.calls) == n_calls, 'no calls while the circuit is open'
    for f in sc.frames[330:360]:                                      # mapping is unaffected
        m.process_frame(f)
    res = a.find('red chair', (1, 1, 0))
    assert res['method'] == 'heuristic' and res['matches']
    fake.fail_status = 0
    clock.t += 31.0
    assert a.available and a.step() == 1 and a.consec_fail == 0


def test_no_client_means_no_annotation_but_find_works():
    m, _ = scenario_map()
    a = ann.Annotator(m, m.cfg.vlm, None)
    assert not a.available and a.step() == 0
    out = a.find('the chair', (1, 1, 0))
    assert out['method'] == 'heuristic' and out['matches'] and out['vlm_calls'] == 0


def test_rate_and_budget_limits(fake):
    clock = Clock()
    m, _ = scenario_map()
    a = make_annotator(m, fake, clock, min_interval_s=5.0, max_calls_per_min=3, max_calls_total=4)
    assert a.step() == 1 and a.step() == 0                           # interval
    for _ in range(6):
        clock.t += 6.0
        a.step()
    assert len(fake.calls) == 3                                      # per-minute limit (3) reached
    clock.t += 61.0
    assert a.step() == 1 and len(fake.calls) == 4
    clock.t += 100.0
    assert a.step() == 0 and a.status()['budget_calls'] == 4         # total budget (4) exhausted
    m2, _ = scenario_map()
    b = make_annotator(m2, fake, clock, min_interval_s=0.0, max_calls_per_min=2, max_calls_total=100)
    n0 = len(fake.calls)
    for _ in range(5):
        b.step()
    assert len(fake.calls) - n0 == 2
    clock.t += 61.0
    assert b.step() == 1


def test_room_naming_cached_by_object_signature(fake):
    m, sc = scenario_map(frames=500)
    rooms.update_rooms(m, sc.grid, m.cfg.rooms)
    for r in m.regions.values():
        r.name, r.name_source, r.vlm_name_key = '', '', ''
    clock = Clock()
    a = make_annotator(m, fake, clock, caption_objects=False, disambiguate=False)
    run_all(a)
    named = {r.name: r.name_source for r in m.regions.values()}
    assert 'kitchen' in named and named['kitchen'] == 'vlm'
    assert a.counters['rooms'] >= 1
    room_calls = [c for c in fake.calls if 'An area of the map' in c['text']]
    assert all('objects' in c['text'] for c in room_calls) and room_calls[0]['n_images'] == 1   # montage of crops
    clock.t += 100.0
    before = len(fake.calls)
    run_all(a)
    assert len(fake.calls) == before                                  # same objects: no new call


def test_disambiguation_merges_or_separates(fake):
    from helpers import red_crop
    for same in (True, False):
        fake.same_object = same
        m = make_map(vlm__enabled=True)
        rng = np.random.default_rng(0)
        feed(m, lambda i: [det('chair', (1.0, 1.0, 0.45), crop=red_crop(rng), box=(0, 0, 80, 80)),
                           det('chair', (1.62, 1.0, 0.45), crop=red_crop(rng), box=(0, 0, 80, 80))], n=8)
        assert len(m.objects) == 2 and m.doubtful_pairs()
        a = make_annotator(m, fake, caption_objects=False, name_rooms=False)
        a.step()
        assert (len(m.objects) == 1) == same
        if not same:
            a.step()
            assert a.step() == 0 and not m.doubtful_pairs() or all(
                o.distinct_from for o in m.objects.values())


def test_find_uses_vlm_ranking_drops_invented_ids_and_checks_crops(fake):
    m, sc = scenario_map(frames=640)
    a = make_annotator(m, fake, find_recheck_top_k=2)
    run_all(a)                                                       # captions first
    fake.calls.clear()
    fake.hallucinate = True
    out = a.find('red chair', (1, 1, 0), limit=3)
    assert out['method'] == 'vlm' and out['vlm_calls'] >= 2 and out['matches']
    assert all(c['id'] in m.objects for c in out['matches'])
    assert out['matches'][0]['label'] in ('chair', 'armchair')
    assert any(c['text'].startswith('Description:') and c['n_images'] == 1 for c in fake.calls)
    fake.crop_match = False
    out2 = a.find('red chair', (1, 1, 0), limit=3)
    assert out2['matches'][0]['score'] < out['matches'][0]['score']  # crops that do not match lower the score
    assert 'crop_check' in out2['matches'][0] or len(out2['matches']) > 0


def test_find_without_word_match_lets_the_vlm_read_captions(fake):
    m, _ = scenario_map(frames=640)
    a = make_annotator(m, fake)
    run_all(a)
    assert finder_rank_empty(m, 'something to sit on')
    out = a.find('something plastic', (1, 1, 0))
    assert out['method'] == 'vlm' and out['matches'], 'the VLM saw all objects with attributes (plastic) from captions'


def finder_rank_empty(m, text):
    from x30_semantic_map import finder
    return finder.rank(m, text, (1, 1, 0))['matches'] == []


def test_find_times_out_into_the_heuristic_ranking(fake):
    fake.delay = 2.0
    m, _ = scenario_map(frames=640)
    a = make_annotator(m, fake)
    t0 = time.time()
    out = a.find('red chair', (1, 1, 0), timeout_s=0.6)
    assert time.time() - t0 < 2.5
    assert out['method'] == 'heuristic' and out['matches']


def test_mapping_is_not_blocked_by_a_slow_annotator(fake):
    fake.delay = 1.5
    m, sc = scenario_map(frames=330)
    a = make_annotator(m, fake)
    a.start(poll_s=0.05)
    try:
        time.sleep(0.4)                                               # a call is in flight now
        assert len(fake.calls) >= 1
        worst = 0.0
        for f in sc.frames[330:400]:
            t0 = time.time()
            m.process_frame(f)
            worst = max(worst, time.time() - t0)
        assert worst < 0.1, f'process_frame took {worst:.3f} s while the VLM call was running'
    finally:
        a.stop()
