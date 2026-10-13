import math

import numpy as np
import pytest

from x30_eval.core import metrics as M
from x30_eval.core.geometry import SE2, align_umeyama_2d, align_yaw, resample_path, wrap
from x30_eval.core.grid import grid_from_rects, shortest_path


def test_wrap_and_se2_inverse():
    assert wrap(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9) or wrap(3 * math.pi) == pytest.approx(math.pi, abs=1e-9)
    t = SE2(1.0, 2.0, 0.7)
    p = np.array([[0.3, -0.4], [2.0, 1.0]])
    assert np.allclose(t.inverse().apply(t.apply(p)), p)
    assert np.allclose(list(t.compose(t.inverse()).as_dict().values()), [0, 0, 0], atol=1e-9)


def test_spl_definition():
    assert M.spl(True, 10.0, 10.0) == 1.0
    assert M.spl(True, 10.0, 20.0) == pytest.approx(0.5)
    assert M.spl(True, 10.0, 8.0) == 1.0          # path shorter than optimum (noisy optimum) is clipped
    assert M.spl(False, 10.0, 10.0) == 0.0
    assert M.spl(True, None, 5.0) == 0.0


def test_ndtw_identical_and_shifted():
    ref = np.array([[0, 0], [5, 0], [5, 5]], float)
    assert M.ndtw(ref, ref) == pytest.approx(1.0)
    shifted = ref + [0.0, 1.0]
    v = M.ndtw(shifted, ref, d_th=3.0)
    assert 0.6 < v < 0.9
    far = ref + [0.0, 10.0]
    assert M.ndtw(far, ref) < 0.05
    assert M.sdtw(False, v) == 0.0 and M.sdtw(True, v) == v and M.sdtw(True, None) is None


def test_ndtw_independent_of_sampling_rate():
    ref = np.array([[0, 0], [10, 0]], float)
    dense = np.stack([np.linspace(0, 10, 500), np.full(500, 0.5)], axis=1)
    sparse = np.stack([np.linspace(0, 10, 11), np.full(11, 0.5)], axis=1)
    assert M.ndtw(dense, ref) == pytest.approx(M.ndtw(sparse, ref), abs=0.02)


def test_ate_recovers_frame_offset_and_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 200)
    gx, gy = 5 * np.cos(t / 4), 5 * np.sin(t / 4)
    gyaw = t / 4 + math.pi / 2
    off = SE2(3.0, -2.0, math.radians(25))                      # SLAM frame is rotated/translated vs world
    m = off.inverse().apply(np.stack([gx, gy], 1)) + rng.normal(0, 0.02, (200, 2))
    myaw = gyaw - off.yaw
    pairs = np.column_stack([t, m, myaw, gx, gy, gyaw])
    a = M.ate(pairs, "yaw")
    assert a["ate_rmse_m"] < 0.05
    assert a["frame_yaw_offset_deg"] == pytest.approx(25, abs=1.0)
    b = M.ate(pairs, "se2")
    assert b["ate_rmse_m"] < 0.05
    assert M.ate(pairs[:5]) is None
    pairs[:, 1] += np.linspace(0, 2.0, 200)                    # drift is NOT absorbed by the alignment
    assert M.ate(pairs, "yaw")["ate_rmse_m"] > 0.3


def test_alignments_agree():
    pts = np.array([[0, 0], [1, 0], [2, 1], [3, 3]], float)
    tf = SE2(1, 2, 0.5)
    q = tf.apply(pts)
    u = align_umeyama_2d(pts, q)
    assert np.allclose(u.apply(pts), q, atol=1e-9)
    y = align_yaw(pts, np.zeros(4), q, np.full(4, 0.5))
    assert np.allclose(y.apply(pts), q, atol=1e-9)


def test_goal_distance_object_surface_vs_centroid():
    objs = [{"label": "sofa", "x": 5.0, "y": 0.0, "sx": 2.0, "sy": 1.0}]
    goal = {"type": "object", "label": "couch"}
    syn = {"couch": ["sofa"]}
    xy = np.array([[2.0, 0.0]])
    assert M.distance_series(xy, goal, objs, syn, "centroid")[0] == pytest.approx(3.0)
    assert M.distance_series(xy, goal, objs, syn, "surface")[0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        M.distance_series(xy, {"type": "object", "label": "piano"}, objs, syn)


def test_any_instance_vs_named_instance():
    objs = [{"label": "chair", "name": "chair_1", "id": 1, "x": 0.0, "y": 0.0, "sx": 0.5, "sy": 0.5},
            {"label": "chair", "name": "chair_2", "id": 2, "x": 10.0, "y": 0.0, "sx": 0.5, "sy": 0.5}]
    goal = {"type": "object", "label": "chair", "instance": "chair_2"}
    xy = np.array([[0.5, 0.0]])
    assert M.distance_series(xy, goal, objs)[0] < 0.5                  # any instance
    assert M.instance_distance_series(xy, goal, objs)[0] > 9           # the named one is far


def test_semantic_map_metrics_matching_duplicates_and_alignment():
    gt = [{"label": "chair", "x": 1.0, "y": 1.0}, {"label": "table", "x": 4.0, "y": 1.0}, {"label": "plant", "x": 9.0, "y": 9.0}]
    align = SE2(10.0, 0.0, 0.0)
    mp = [{"label": "chair", "x": -9.0, "y": 1.1}, {"label": "chair", "x": -9.0, "y": 1.3},   # duplicate of the chair
          {"label": "table", "x": -6.0, "y": 1.0}, {"label": "lamp", "x": -5.0, "y": 5.0},   # lamp: spurious
          {"label": "bed", "x": 0, "y": 0, "bearing_only": True}]
    traj = np.array([[1.0, 1.0], [4.0, 1.0]])
    r = M.semantic_map_metrics(mp, gt, align, traj, match_radius=1.0, cover_radius=3.0)
    assert r["n_matched"] == 2 and r["n_bearing_only"] == 1
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["recall_covered"] == pytest.approx(1.0)                   # the plant was never near the path
    assert r["duplicate_rate"] == pytest.approx(1 / 4)
    assert r["precision"] == pytest.approx(2 / 4)
    assert r["pos_err_mean_m"] == pytest.approx(0.05, abs=0.01)


def test_collisions_from_grid_and_event_merging():
    g = grid_from_rects(10, 10, 0.1, [[5, 0, 5.2, 10]], border=False)
    poses = np.array([[t * 0.1, 1.0 + t * 0.1, 5.0, 0.0] for t in range(60)])      # drives into the wall at x=5
    ev = M.grid_collision_times(poses, g, 0.9, 0.45)
    assert len(ev) == 1
    assert M.grid_collision_times(poses[:10], g) == []
    assert M.merge_events([1.0, 1.4, 3.0, 3.2, 9.0], 1.0) == [1.0, 3.0, 9.0]


def test_shortest_path_goes_around_wall_and_respects_inflation():
    g = grid_from_rects(10, 10, 0.1, [[4, 0, 4.3, 7]], border=False)
    length, pts = shortest_path(g, (1, 1), (8, 1), inflate_m=0.3)
    assert 14.0 < length < 15.5                                         # straight line would be 7
    assert shortest_path(g, (1, 1), (8, 1), inflate_m=0.3)[0] >= shortest_path(g, (1, 1), (8, 1), inflate_m=0.0)[0]
    closed = grid_from_rects(10, 10, 0.1, [[4, 0, 4.3, 10]], border=False)
    assert shortest_path(closed, (1, 1), (8, 1)) is None
    region = shortest_path(g, (1, 1), lambda x, y: np.hypot(x - 8, y - 1) <= 1.5)
    assert region[0] < length


def test_resample_path_spacing():
    r = resample_path(np.array([[0, 0], [10, 0]]), 0.5)
    assert len(r) == 21 and r[-1][0] == 10
