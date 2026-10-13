import math

import numpy as np

from x30_follow import appearance
from x30_follow.lock import LockParams, TargetLock
from x30_follow.tracker import Det, PersonTracker


def person_img(torso, legs, h=220, w=70):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = (200, 200, 200)
    img[int(0.18 * h):int(0.52 * h)] = torso
    img[int(0.55 * h):int(0.92 * h)] = legs
    return img


NAVY, RED, GREEN, WHITE, BLACK, GREY = (90, 40, 20), (30, 30, 170), (40, 140, 40), (235, 235, 235), (25, 25, 25), (128, 128, 128)
SIG = {n: appearance.signature_from_bgr(person_img(t, l)) for n, (t, l) in
       {"navy": (NAVY, NAVY), "red": (RED, RED), "green": (GREEN, GREEN), "whiteblack": (WHITE, BLACK), "twin": (NAVY, NAVY)}.items()}


def det(x, y, t, who, pts=300):
    return Det(x, y, 0.9, pts, t, SIG[who])


def feed(tr, t, items):
    tr.update(t, [det(x, y, t, w) for x, y, w in items])


def test_signature_separates_outfits_and_matches_itself():
    d = appearance.distance
    assert d(SIG["navy"], SIG["twin"]) < 0.05
    for other in ("red", "green", "whiteblack"):
        assert d(SIG["navy"], SIG[other]) > 0.6
    assert appearance.signature(b"") is None and d(None, SIG["navy"]) is None


def test_lock_follows_the_target_through_a_crossing_with_a_different_outfit():
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0 + 0.5 * k * 0.25, 0.0, "navy")])
    lk.lock(tr.get(1), 2.0)
    for k in range(8, 40):                           # a red person walks the other way through the target (same place at k = 24)
        t = k * 0.25
        tx = 3.0 + 0.5 * t
        rx = tx + (24 - k) * 0.25 * 1.0
        feed(tr, t, [(tx, 0.0, "navy"), (rx, 0.3, "red")])
        dec = lk.select(tr, t)
        overlap = 21 <= k <= 29                      # both people within ~1 m: crops overlap, descriptors are unknown, the lock may hold briefly
        assert dec.state == "ok" or (overlap and dec.state == "ambiguous"), (k, dec)
        if dec.state == "ok":                        # never the wrong person
            assert abs(dec.track.x - tx) < 0.5 and abs(dec.track.y) < 0.2
    assert dec.state == "ok"


def test_a_twin_is_not_confused_with_the_target_while_tracking_is_continuous():
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0 + 0.5 * k * 0.25, 0.0, "navy")])
    lk.lock(tr.get(1), 2.0)
    for k in range(8, 40):                           # a same-outfit person walks 0.9 m beside the target: position continuity decides
        t = k * 0.25
        tx = 3.0 + 0.5 * t
        feed(tr, t, [(tx, 0.0, "navy"), (tx + 0.5, 0.9, "twin")])
        dec = lk.select(tr, t)
        assert dec.state == "ok" and abs(dec.track.y) < 0.4


def test_after_an_occlusion_a_twin_makes_it_ambiguous_until_it_walks_away():
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0 + 0.5 * k * 0.25, 0.0, "navy")])
    lk.lock(tr.get(1), 1.75)
    assert lk.select(tr, 1.75).state == "ok"
    t = 3.75                                         # nobody seen for 2 s (target behind an obstacle), then target and a twin appear
    px, py = lk.predicted(t)
    feed(tr, t, [(px, py - 0.4, "navy"), (px, py + 0.4, "twin")])
    feed(tr, t + 0.25, [(px + 0.25, py - 0.4, "navy"), (px + 0.25, py + 0.4, "twin")])
    dec = lk.select(tr, t + 0.25)
    assert dec.state == "ambiguous" and dec.candidates == 2
    states = []
    for k in range(1, 12):                           # the twin leaves sideways, the target keeps walking
        tt = t + 0.25 + k * 0.25
        feed(tr, tt, [(px + 0.25 + 0.5 * k * 0.25, py - 0.4, "navy"), (px + 0.25 + 0.5 * k * 0.25, py + 0.4 + 0.5 * k, "twin")])
        states.append(lk.select(tr, tt).state)
    assert states[-1] == "ok"


def test_lost_when_nobody_matches_and_the_unlocked_lock_returns_lost():
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0, 0.0, "navy")])
    assert lk.select(tr, 2.0).state == "lost"
    lk.lock(tr.get(1), 2.0)
    for k in range(8, 14):
        feed(tr, k * 0.25, [(3.0, 0.0, "red")])        # only a differently dressed person is left
    assert lk.select(tr, 3.5).state == "lost"


def test_tracker_does_not_merge_a_different_outfit_into_the_track():
    tr = PersonTracker()
    for k in range(6):
        feed(tr, k * 0.25, [(3.0, 0.0, "navy")])
    feed(tr, 1.5, [(3.1, 0.1, "red")])               # same place, other outfit: a new track, not the old one
    assert len(tr.tracks) == 2


def test_overlapping_detections_do_not_contaminate_the_signature():
    tr = PersonTracker()
    for k in range(6):
        feed(tr, k * 0.25, [(3.0, 0.0, "navy")])
    before = tr.get(1).sig.copy()
    for k in range(6, 14):                           # a red person right next to the target: both crops contain both people
        mixed = appearance.blend(SIG["navy"], SIG["red"], 0.5)
        tr.update(k * 0.25, [Det(3.0, 0.0, 0.9, 300, k * 0.25, mixed), Det(3.4, 0.2, 0.9, 300, k * 0.25, mixed)])
    assert np.allclose(tr.get(1).sig, before)
    assert appearance.distance(tr.get(1).sig, SIG["navy"]) < 0.05


def test_scales_to_a_crowd_and_keeps_the_target():
    """300 people: association and selection stay fast (KD-tree) and the lock keeps the right person."""
    import time
    rng = np.random.RandomState(1)
    base = rng.uniform(-40, 40, size=(300, 2))
    who = rng.choice(["red", "green", "whiteblack"], size=300)
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        t = k * 0.25
        tr.update(t, [det(0.0, 0.0, t, "navy")] + [det(float(b[0]), float(b[1]), t, str(w)) for b, w in zip(base, who)])
    target = min(tr.confirmed(), key=lambda x: math.hypot(x.x, x.y))
    lk.lock(target, 1.75)
    worst_update = worst_select = 0.0
    for k in range(8, 20):
        t = k * 0.25
        dets = [det(0.1 * (k - 8), 0.0, t, "navy")] + [det(float(b[0]), float(b[1]), t, str(w)) for b, w in zip(base, who)]
        t0 = time.perf_counter(); tr.update(t, dets); worst_update = max(worst_update, time.perf_counter() - t0)
        t0 = time.perf_counter(); dec = lk.select(tr, t); worst_select = max(worst_select, time.perf_counter() - t0)
        assert dec.state == "ok" and abs(dec.track.x - 0.1 * (k - 8)) < 0.3
    assert worst_update < 0.25 and worst_select < 0.05, (worst_update, worst_select)


def test_a_stopped_target_and_a_twin_walking_through_the_prediction_is_ambiguous_not_a_wrong_pick():
    """The target stops during a short occlusion; the extrapolated prediction runs on, a look-alike walks through it. Position alone would pick
    the look-alike: the descriptor cannot separate them, so the lock must hold (ambiguous) instead of following the wrong person."""
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0 + 0.5 * k * 0.25, 0.0, "navy")])
    lk.lock(tr.get(1), 1.75)                         # target walks +x at 0.5 m/s, last sure sighting at t = 1.75 near x = 3.9
    t = 3.5                                          # 1.75 s later the target has stopped at x = 3.95, the prediction is at about x = 4.8
    px, py = lk.predicted(t)
    feed(tr, t, [(3.95, 0.0, "navy"), (px, py + 0.1, "twin")])
    feed(tr, t + 0.25, [(3.95, 0.0, "navy"), (px + 0.35, py + 0.1, "twin")])
    dec = lk.select(tr, t + 0.25)
    assert dec.state == "ambiguous", dec


def test_stopped_target_is_chosen_after_the_twin_walks_on():
    """Target stops, a same-outfit person walks through where it would have been and on: while both are on the segment the lock holds,
    once the twin is clearly beyond it the lock picks the target (the one at the start of the segment), never the twin."""
    tr, lk = PersonTracker(), TargetLock()
    for k in range(8):
        feed(tr, k * 0.25, [(3.0 + 0.5 * k * 0.25, 0.0, "navy")])
    lk.lock(tr.get(1), 1.75)                         # target walks +x at 0.5 m/s, then stops at x = 3.95
    states, picks = [], []
    for k in range(8, 70):
        t = k * 0.25
        twin_x = 3.95 + 0.7 * (t - 3.0) - 1.0        # a look-alike walking +x at 0.7 m/s passing through the target at t = 4.4 s
        feed(tr, t, [(3.95, 0.0, "navy"), (twin_x, 0.2, "twin")])
        dec = lk.select(tr, t)
        states.append(dec.state)
        if dec.state == "ok":
            picks.append(dec.track.x)
    assert "ambiguous" in states
    assert states[-1] == "ok"
    assert all(abs(x - 3.95) < 0.3 for x in picks), picks         # never the walking twin
