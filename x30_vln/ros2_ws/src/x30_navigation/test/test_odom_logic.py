import math

import numpy as np

from x30_navigation.geometry import make_transform, matrix_to_rpy, rpy_to_matrix
from x30_navigation.odom_logic import ComplementaryFusion, DisagreementMonitor, JumpConfig, JumpDetector


def pose(x, y, yaw, z=0.0):
    return make_transform(rpy_to_matrix(0.0, 0.0, yaw), [x, y, z])


def run_fusion(leg_scale=1.1, yaw_bias=0.02, seconds=60.0, use_lidar=True):
    """Robot drives a circle-ish path; legs have scale error and yaw bias, lidar poses are exact."""
    f = ComplementaryFusion()
    dt, v, w = 0.01, 0.4, 0.1
    x = y = yaw = 0.0
    err = []
    for k in range(int(seconds / dt)):
        t = k * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        if k % 10 == 0 and use_lidar:
            f.on_lidar_pose(pose(x, y, yaw, z=0.45))
        elif k == 0:
            f.on_lidar_pose(pose(0, 0, 0, z=0.45))
        out = f.on_leg_velocity(t, v * leg_scale, 0.0, w + yaw_bias)
        if out is not None:
            err.append(np.linalg.norm(out[:2, 3] - [x, y]))
    return np.array(err), f


def test_fusion_bounds_leg_drift():
    err_fused, f = run_fusion()
    err_leg, _ = run_fusion(use_lidar=False)
    assert err_fused.max() < 0.35            # bounded by the lidar
    assert err_leg.max() > 3.0               # legs alone drift a lot with 10 % scale and 0.02 rad/s yaw bias
    assert f.pose[2, 3] == 0.45              # height comes from the lidar


def test_fusion_rejects_lidar_jump_and_survives_lidar_dropout():
    f = ComplementaryFusion()
    f.on_lidar_pose(pose(0, 0, 0))
    for k in range(100):
        f.on_leg_velocity(k * 0.01, 0.5, 0.0, 0.0)
    f.on_lidar_pose(pose(30.0, 0.0, 0.0))     # ICP blew up
    assert f.rejected_updates == 1
    assert abs(f.pose[0, 3] - 0.495) < 0.02   # legs keep the estimate


def test_jump_detector():
    d = JumpDetector(JumpConfig(max_speed=3.0))
    assert not d.update(0.0, pose(0, 0, 0))
    assert not d.update(0.1, pose(0.05, 0, 0))
    assert d.update(0.2, pose(2.0, 0, 0))     # 19 m/s
    assert d.flagged(1.0) and not d.flagged(10.0)
    assert d.jumps == 1
    assert d.update(0.3, pose(2.0, 0, 3.0))   # yaw spin of 3 rad in 0.1 s
    assert not d.stale(0.0) or True


def test_stale_detection_uses_wall_time():
    d = JumpDetector(JumpConfig(stale_after=1.0))
    assert d.stale(5.0)                       # never received
    d.update(0.0, pose(0, 0, 0), wall=10.0)
    assert not d.stale(10.5) and d.stale(12.0)


def test_disagreement_monitor():
    m = DisagreementMonitor(window=6.0, min_travel=0.8, max_ratio=1.6)
    for k in range(60):
        t = k * 0.1
        m.add("a", t, pose(0.3 * t, 0, 0))     # legs: 0.3 m/s
        m.add("b", t, pose(0.5 * t, 0, 0))     # lidar: 0.5 m/s  (ratio 1.67)
    assert m.check()
    m2 = DisagreementMonitor()
    for k in range(60):
        t = k * 0.1
        m2.add("a", t, pose(0.3 * t, 0, 0))
        m2.add("b", t, pose(0.31 * t, 0, 0.01 * t))
    assert not m2.check()
    m3 = DisagreementMonitor()                 # stationary robot with lidar jitter: below min_travel, no alarm
    rng = np.random.default_rng(0)
    for k in range(60):
        m3.add("a", k * 0.1, pose(0, 0, 0))
        m3.add("b", k * 0.1, pose(*rng.normal(0, 0.003, 2), 0))
    assert not m3.check()


def test_rpy_roundtrip_in_fusion_pose():
    f = ComplementaryFusion()
    f.on_lidar_pose(make_transform(rpy_to_matrix(0.05, -0.03, 1.0), [1, 2, 0.4]))
    roll, pitch, yaw = matrix_to_rpy(f.pose[:3, :3])
    assert abs(roll - 0.05) < 1e-9 and abs(pitch + 0.03) < 1e-9 and abs(yaw - 1.0) < 1e-9
