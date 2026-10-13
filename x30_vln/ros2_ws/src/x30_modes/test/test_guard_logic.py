import math

import numpy as np
import pytest

from x30_modes.guard_logic import (GuardConfig, SafetyGuardCore, gap_to_footprint, slew, REASON_ESTOP, REASON_NO_LIDAR,
                                   REASON_OBSTACLE, REASON_STALE, REASON_TILT)
from x30_modes.geometry import Footprint
from robot_model import SimRobot

DT = 0.05


def wall(x, half=1.5, n=300):
    return np.stack([np.full(n, x), np.linspace(-half, half, n)], axis=1)


def new_guard(**kw):
    return SafetyGuardCore(GuardConfig(**kw))


def run_open(g, cmd, secs, pts=None, t0=0.0, estop=False, tilt=None):
    out = None
    t = t0
    for _ in range(int(secs / DT)):
        out = g.step(t, DT, cmd, t, pts if pts is not None else np.zeros((0, 2)), t, estop, tilt)
        t += DT
    return out, t


def test_slew_limits():
    assert slew(0.0, 1.0, 0.1, accel=0.5, decel=1.0) == pytest.approx(0.05)
    assert slew(0.5, 0.0, 0.1, accel=0.5, decel=1.0) == pytest.approx(0.4)
    assert slew(0.3, -0.3, 0.1, 0.5, 1.0) == pytest.approx(0.2)      # reversing brakes first
    assert slew(0.0, 0.02, 0.1, 0.5, 1.0) == 0.02


def test_passes_command_with_limits_and_ramp():
    g = new_guard(vx_max=0.6)
    out, t = run_open(g, (1.0, 0.0, 0.0), 0.5)
    assert 0 < out.twist[0] < 0.6                                  # still accelerating (0.8 m/s2 -> 0.4 after 0.5 s)
    out, t = run_open(g, (1.0, 0.0, 0.0), 2.0, t0=t)
    assert out.twist[0] == pytest.approx(0.6)                      # clamped to the hard limit
    assert not out.limiting and not out.stopped


def test_accel_rate_limit_respected():
    g = new_guard()
    prev = 0.0
    t = 0.0
    for _ in range(40):
        o = g.step(t, DT, (0.5, 0.0, 0.0), t, np.zeros((0, 2)), t, False, None)
        assert o.twist[0] - prev <= g.cfg.accel_x * DT + 1e-9
        prev = o.twist[0]
        t += DT


def test_stale_command_zeroes():
    g = new_guard()
    out, t = run_open(g, (0.4, 0, 0), 2.0)
    assert out.twist[0] > 0.3
    o = g.step(t + 1.0, DT, (0.4, 0, 0), t, np.zeros((0, 2)), t + 1.0, False, None)   # command older than 0.5 s
    assert o.twist == (0.0, 0.0, 0.0) and o.reason == REASON_STALE


def test_estop_and_tilt_immediate_zero():
    g = new_guard()
    out, t = run_open(g, (0.4, 0, 0), 2.0)
    o = g.step(t, DT, (0.4, 0, 0), t, np.zeros((0, 2)), t, True, None)
    assert o.twist == (0.0, 0.0, 0.0) and o.reason == REASON_ESTOP and o.stopped
    o = g.step(t + DT, DT, (0.4, 0, 0), t + DT, np.zeros((0, 2)), t + DT, False, 50.0)
    assert o.reason == REASON_TILT and o.stopped and o.twist == (0.0, 0.0, 0.0)
    # recovers with a ramp, not a jump
    o = g.step(t + 2 * DT, DT, (0.4, 0, 0), t + 2 * DT, np.zeros((0, 2)), t + 2 * DT, False, 0.0)
    assert 0.0 < o.twist[0] <= 0.8 * DT + 1e-9


def test_no_lidar_stops_motion_but_not_idle():
    g = new_guard()
    o = g.step(0.0, DT, (0.4, 0, 0), 0.0, None, None, False, None)
    assert o.reason == REASON_NO_LIDAR and o.stopped and o.twist == (0.0, 0.0, 0.0)
    o = g.step(0.0, DT, (0.0, 0, 0), 0.0, None, None, False, None)
    assert o.reason == REASON_NO_LIDAR and not o.stopped
    g2 = new_guard(require_obstacles=False)
    o = g2.step(0.0, DT, (0.4, 0, 0), 0.0, None, None, False, None)
    assert o.twist[0] > 0


def test_stale_lidar_counts_as_missing():
    g = new_guard(obstacle_timeout_s=1.0)
    o = g.step(5.0, DT, (0.4, 0, 0), 5.0, np.zeros((0, 2)), 3.0, False, None)
    assert o.reason == REASON_NO_LIDAR


def test_obstacle_slows_then_stops_with_margin():
    """Closed loop on an ideal robot: output is the actual velocity; wall 3 m ahead of base_link."""
    cfg = GuardConfig(latency_s=0.25, decel=1.0, stop_margin=0.2)
    for speed in (0.2, 0.4, 0.6):
        g = SafetyGuardCore(cfg)
        x = 0.0
        t = 0.0
        w = 3.0
        for _ in range(int(20 / DT)):
            pts = wall(w - x)
            o = g.step(t, DT, (speed, 0, 0), t, pts, t, False, None)
            x += o.twist[0] * DT
            t += DT
        gap = (w - x) - cfg.front
        assert o.stopped or o.scale == 0.0
        assert gap >= cfg.stop_margin - 0.02, (speed, gap)
        assert gap < cfg.stop_margin + 0.15, (speed, gap)           # not stopping absurdly early
        assert o.reason == REASON_OBSTACLE


def test_reverse_uses_rear_clearance():
    g = new_guard()
    x, t = 0.0, 0.0
    for _ in range(int(20 / DT)):
        pts = wall(-2.0 - x)                       # wall behind, moves with robot motion (x decreasing)
        o = g.step(t, DT, (-0.3, 0, 0), t, pts, t, False, None)
        x += o.twist[0] * DT
        t += DT
    gap = (x - (-2.0)) - g.cfg.rear
    assert gap >= g.cfg.stop_margin - 0.02


def test_command_away_from_obstacle_is_not_limited():
    g = new_guard()
    pts = wall(0.7)                                # 0.2 m from the nose
    out, t = run_open(g, (-0.2, 0, 0), 1.0, pts)
    assert out.twist[0] < 0 and not out.limiting


def test_side_obstacle_does_not_block_forward():
    g = new_guard()
    pts = np.array([[0.2, 0.45]])                  # 0.13 m beside the footprint+margin
    out, t = run_open(g, (0.4, 0, 0), 2.0, pts)
    assert out.twist[0] == pytest.approx(0.4)


def test_self_points_filtered():
    g = new_guard()
    pts = np.array([[0.4, 0.0], [0.0, 0.2]])       # inside the raw footprint: the robot's own body
    out, t = run_open(g, (0.4, 0, 0), 2.0, pts)
    assert out.twist[0] == pytest.approx(0.4)


def test_hard_stop_zeroes_immediately():
    g = new_guard(hard_stop_margin=0.1)
    out, t = run_open(g, (0.4, 0, 0), 2.0)
    pts = wall(0.5 + 0.06 + 0.03)                  # 3 cm beyond the self-filter band, i.e. 9 cm from the nose, appears suddenly
    o = g.step(t, DT, (0.4, 0, 0), t, pts, t, False, None)
    assert o.twist == (0.0, 0.0, 0.0)


def test_gap_to_footprint():
    fp = Footprint(0.5, 0.5, 0.3)
    assert gap_to_footprint(np.array([[1.5, 0.0]]), fp) == pytest.approx(1.0)
    assert gap_to_footprint(np.array([[0.0, 0.0]]), fp) == 0.0
    assert math.isinf(gap_to_footprint(np.zeros((0, 2)), fp))


def test_status_reports_min_distance():
    g = new_guard()
    pts = wall(2.5)
    o, _ = run_open(g, (0.3, 0, 0), 0.2, pts)
    assert o.min_obstacle_distance == pytest.approx(2.0, abs=0.05)


def test_isolated_speckle_points_are_ignored_but_walls_are_not():
    g = new_guard()
    speckle = np.array([[0.9, 0.0], [1.5, 0.1]])           # two lone returns right in the path
    out, t = run_open(g, (0.4, 0, 0), 2.0, speckle)
    assert out.twist[0] == pytest.approx(0.4) and not out.limiting
    out, t = run_open(g, (0.4, 0, 0), 4.0, wall(0.8))  # dense wall still stops the robot
    assert out.limiting
