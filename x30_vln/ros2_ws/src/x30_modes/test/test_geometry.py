import math

import numpy as np
import pytest

from x30_modes.geometry import (Footprint, GridMap, Pose2D, clearance_sweep, clearance_translation, max_safe_speed,
                                path_clearance, standoff_pose, stopping_distance, wrap_pi, yaw_from_quat, quat_from_yaw,
                                object_half_extent)

FP = Footprint(front=0.5, rear=0.5, half_width=0.3)


def pts(*xy):
    return np.array(xy, float).reshape(-1, 2)


def test_wrap_and_quat():
    assert wrap_pi(3 * math.pi) == pytest.approx(math.pi) or wrap_pi(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_pi(-0.5) == pytest.approx(-0.5)
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.0):
        assert yaw_from_quat(*quat_from_yaw(yaw)) == pytest.approx(yaw)


def test_forward_clearance_exact():
    # wall point 2.0 m ahead of base_link on the axis: footprint front is 0.5 -> 1.5 m free
    assert clearance_translation(pts((2.0, 0.0)), FP, (1, 0), 5.0) == pytest.approx(1.5)
    # same point seen while driving backwards: not in the way
    assert math.isinf(clearance_translation(pts((2.0, 0.0)), FP, (-1, 0), 5.0))
    # rear obstacle when reversing
    assert clearance_translation(pts((-1.5, 0.1)), FP, (-1, 0), 5.0) == pytest.approx(1.0)


def test_lateral_offset_misses_or_hits():
    assert math.isinf(clearance_translation(pts((2.0, 0.5)), FP, (1, 0), 5.0))      # 0.5 > half width 0.3
    assert clearance_translation(pts((2.0, 0.25)), FP, (1, 0), 5.0) == pytest.approx(1.5)


def test_strafe_and_diagonal():
    assert clearance_translation(pts((0.0, 1.0)), FP, (0, 1), 5.0) == pytest.approx(0.7)
    assert clearance_translation(pts((0.0, -1.0)), FP, (0, 1), 5.0) == math.inf
    # diagonal toward a point at 45 degrees: nearest approach distance along the direction
    d = clearance_translation(pts((1.5, 1.5)), FP, (1 / math.sqrt(2), 1 / math.sqrt(2)), 5.0)
    assert d == pytest.approx(1.2 / math.cos(math.pi / 4), abs=1e-6)     # y slab (half width 0.3) is entered last


def test_beyond_horizon():
    assert math.isinf(clearance_translation(pts((9.0, 0.0)), FP, (1, 0), 3.0))


def test_wall_in_margin_ring_blocks_ahead_but_does_not_freeze_escape():
    fp = FP.inflated(0.05)                       # half width 0.35
    outside = pts(*[(x, fp.half_width + 0.02) for x in np.linspace(-1, 2, 40)])
    assert math.isinf(clearance_translation(outside, fp, (1, 0), 3.0))          # parallel wall just outside the margin
    ring = pts(*[(x, fp.half_width - 0.02) for x in np.linspace(-1, 2, 40)])   # wall 2 cm inside the margin
    assert clearance_translation(ring, fp, (1, 0), 3.0) < 0.1                   # points ahead in the ring block
    assert clearance_translation(ring, fp, (0, 1), 3.0) == 0.0                  # moving into the wall is blocked
    assert math.isinf(clearance_translation(ring, fp, (0, -1), 3.0))            # moving away is free
    assert clearance_translation(ring, fp, (-1, 0), 3.0) < 0.1                  # reversing: points behind in the ring block


def test_contact_ahead_blocks_forward_but_allows_reverse():
    p = pts((0.45, 0.0))                      # point inside the footprint front region (collision)
    assert clearance_translation(p, FP, (1, 0), 3.0) == 0.0
    assert math.isinf(clearance_translation(p, FP, (-1, 0), 3.0))


def test_sweep_matches_translation():
    p = pts((2.0, 0.1), (3.0, -0.2))
    a = clearance_translation(p, FP, (1, 0), 4.0)
    b = clearance_sweep(p, FP, 0.4, 0.0, 0.0, 4.0, rot_radius=0.45, step=0.01)
    assert b == pytest.approx(a, abs=0.03)


def test_rotation_sweep_hits_side_obstacle():
    # pure rotation: a point just outside the nose swings into the body when turning left (corner sweeps)
    p = pts((0.0, 0.62))
    d = clearance_sweep(p, FP, 0.0, 0.0, 0.5, 3.0, rot_radius=0.45)
    assert math.isinf(d) or d > 0.0
    corner_hit = clearance_sweep(pts((0.4, 0.4)), FP, 0.0, 0.0, 0.5, 3.0, rot_radius=0.45)
    assert corner_hit < 3.0
    far = clearance_sweep(pts((2.0, 2.0)), FP, 0.0, 0.0, 0.5, 3.0, rot_radius=0.45)
    assert math.isinf(far)


def test_path_clearance_dispatch():
    assert path_clearance(pts((2.0, 0)), FP, 0.4, 0, 0, 3.0, 0.45) == pytest.approx(1.5)
    assert math.isinf(path_clearance(pts((2.0, 0)), FP, 0, 0, 0, 3.0, 0.45))


def test_stopping_distance_and_max_speed_inverse():
    for clear in (0.4, 0.8, 1.5, 3.0):
        v = max_safe_speed(clear, decel=1.0, latency=0.25, margin=0.2)
        if v > 0:
            assert stopping_distance(v, 1.0, 0.25) + 0.2 == pytest.approx(clear, abs=1e-6)
    assert max_safe_speed(0.15, 1.0, 0.25, 0.2) == 0.0
    assert math.isinf(max_safe_speed(math.inf, 1.0, 0.25, 0.2))


def test_object_half_extent():
    assert object_half_extent((0.6, 0.4), (1, 0)) == pytest.approx(0.3)
    assert object_half_extent((0.6, 0.4), (0, 1)) == pytest.approx(0.2)
    assert object_half_extent((0, 0), (1, 0)) == 0.0
    assert object_half_extent((10, 10), (1, 0), cap=1.2) == 1.2


def make_grid(w=100, h=100, res=0.1, origin=(-5.0, -5.0)):
    return GridMap(np.zeros((h, w), dtype=np.int8), res, origin[0], origin[1])


def test_grid_free_and_occupied():
    g = make_grid()
    assert g.disc_free(0, 0, 0.4)
    g.data[50, 55] = 100                       # cell at x=0.5, y=0.0
    assert not g.disc_free(0.45, 0.0, 0.3)
    assert g.disc_free(-1.0, 0.0, 0.3)
    assert not g.disc_free(10.0, 0.0, 0.3)     # outside = not free
    g.data[:, :] = -1
    assert not g.disc_free(0, 0, 0.3)
    assert g.disc_free(0, 0, 0.3, allow_unknown=True)


def test_standoff_pose_free_space():
    g = make_grid()
    pose, off = standoff_pose((0.0, 0.0), (0.4, 0.4), (3.0, 0.0), standoff=1.0, robot_front=0.5, grid=g, clearance=0.4)
    assert off == 0.0
    # centre distance = half extent 0.2 + standoff 1.0 + front 0.5
    assert math.hypot(pose.x, pose.y) == pytest.approx(1.7)
    assert pose.yaw == pytest.approx(math.pi)       # faces the object at the origin from +x


def test_standoff_pose_skips_blocked_side():
    g = make_grid()
    # block the straight-line candidate (object at origin, robot at +x, candidate at x=1.7)
    for yy in range(46, 55):
        for xx in range(60, 78):
            g.data[yy, xx] = 100
    pose, off = standoff_pose((0.0, 0.0), (0.4, 0.4), (3.0, 0.0), 1.0, 0.5, g, 0.3)
    assert off > 0.0
    assert g.disc_free(pose.x, pose.y, 0.3)


def test_standoff_pose_none_when_enclosed():
    g = make_grid()
    g.data[:, :] = 100
    g.data[48:53, 48:53] = 0
    assert standoff_pose((0.0, 0.0), (0.4, 0.4), (3.0, 0.0), 1.0, 0.5, g, 0.3) is None
