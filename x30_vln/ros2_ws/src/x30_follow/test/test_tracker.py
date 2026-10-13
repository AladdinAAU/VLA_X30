import math

from x30_follow.tracker import Det, PersonTracker, TrackerParams


def det(x, y, t, score=0.9, pts=200):
    return Det(x, y, score, pts, t)


def test_track_follows_a_walking_person_and_keeps_its_id():
    tr = PersonTracker()
    ids = set()
    for k in range(40):                              # 0.8 m/s along +x, 4 detections per second
        t = k * 0.25
        out = tr.update(t, [det(3.0 + 0.8 * t, 0.0, t)])
        ids |= {o.id for o in out}
    assert ids == {1}
    trk = tr.get(1)
    assert abs(trk.vx - 0.8) < 0.15 and abs(trk.vy) < 0.1
    assert trk in tr.confirmed()


def test_weak_and_sparse_detections_are_ignored():
    tr = PersonTracker()
    assert tr.update(0.0, [det(2, 0, 0, score=0.3), det(3, 1, 0, pts=6)]) == []


def test_two_people_get_two_tracks_and_do_not_swap():
    tr = PersonTracker()
    for k in range(20):
        t = k * 0.25
        tr.update(t, [det(3.0 + 0.5 * t, 2.0, t), det(6.0, -2.0, t)])
    a, b = tr.get(1), tr.get(2)
    assert a is not None and b is not None
    assert a.y > 1.5 and b.y < -1.5


def test_duplicates_in_one_frame_are_merged():
    tr = PersonTracker()
    out = tr.update(0.0, [det(3.0, 0.0, 0.0, pts=300), det(3.2, 0.1, 0.0, pts=80)])
    assert len(out) == 1 and out[0].points == 300


def test_track_survives_a_short_gap_and_is_dropped_after_a_long_one():
    tr = PersonTracker(TrackerParams(drop_after_s=4.0))
    for k in range(8):
        tr.update(k * 0.25, [det(3.0 + 0.4 * k * 0.25, 0.0, k * 0.25)])
    tr.update(3.5, [])                               # nothing seen for 1.75 s: still alive and predicted ahead
    trk = tr.get(1)
    assert trk is not None
    px, _ = tr.predict(trk, 3.5)
    assert px > trk.x
    tr.update(9.0, [])
    assert tr.get(1) is None


def test_reacquire_inside_the_gate_keeps_the_id():
    tr = PersonTracker()
    for k in range(6):
        tr.update(k * 0.25, [det(3.0, 0.0, k * 0.25)])
    tr.update(3.0, [det(3.4, 0.2, 3.0)])
    assert set(tr.tracks) == {1}


def test_extrapolation_is_capped():
    tr = PersonTracker(TrackerParams(max_extrapolate_s=2.5))
    for k in range(8):
        tr.update(k * 0.25, [det(1.0 + 1.0 * k * 0.25, 0.0, k * 0.25)])
    trk = tr.get(1)
    x_far, _ = tr.predict(trk, trk.last_t + 100.0)
    assert x_far - trk.x <= trk.vx * 2.5 + 1e-6
    assert math.isfinite(x_far)
