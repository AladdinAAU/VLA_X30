import math

import numpy as np

from x30_navigation.cloud_ops import HeightBand, classify, obstacle_points, scan_from_points
from x30_navigation.geometry import matrix_to_quat, quat_to_matrix, rpy_to_matrix, matrix_to_rpy
from x30_navigation.ground import GroundEstimator, GroundParams, to_level_frame


def synthetic_room(pitch=0.0, roll=0.0, sensor_h=0.45, seed=0):
    """Floor + wall + a box + ceiling seen from a sensor at the origin, tilted by (roll, pitch)."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack((rng.uniform(-6, 6, 20000), rng.uniform(-6, 6, 20000), np.full(20000, -sensor_h)))
    wall = np.column_stack((np.full(4000, 5.0), rng.uniform(-5, 5, 4000), rng.uniform(-sensor_h, 2.0, 4000)))
    box = np.column_stack((rng.uniform(2.0, 2.5, 800), rng.uniform(-0.5, 0.0, 800), rng.uniform(-sensor_h, 0.2, 800)))
    ceiling = np.column_stack((rng.uniform(-6, 6, 8000), rng.uniform(-6, 6, 8000), np.full(8000, 2.5)))
    world = np.vstack((floor, wall, box, ceiling))
    r = rpy_to_matrix(roll, pitch, 0.0)
    return world @ r  # world points seen from a tilted body: p_body = R^T p_world


def test_quat_roundtrip():
    r = rpy_to_matrix(0.1, -0.2, 1.3)
    r2 = quat_to_matrix(*matrix_to_quat(r))
    assert np.allclose(r, r2, atol=1e-9)
    assert np.allclose(matrix_to_rpy(r), (0.1, -0.2, 1.3), atol=1e-9)


def test_ground_fit_level():
    pts = synthetic_room()
    est = GroundEstimator(GroundParams(smooth=1.0)).fit(pts)
    assert est.valid
    assert abs(est.height_above_ground - 0.45) < 0.02
    assert est.normal[2] > 0.999


def test_ground_fit_pitched_body_levels_the_floor():
    for pitch, roll in ((0.15, 0.0), (-0.12, 0.1), (0.0, 0.2)):
        pts = synthetic_room(pitch=pitch, roll=roll)
        est = GroundEstimator(GroundParams(smooth=1.0)).fit(pts)
        assert est.valid
        lev = to_level_frame(pts, est)
        band = HeightBand()
        ground, obs = classify(lev, band)
        # the floor must be removed (no floor points as obstacles) even though the body is pitched/rolled
        floor_like = lev[(np.abs(lev[:, 2]) < 0.05)]
        assert len(floor_like) > 15000
        assert not obs[np.abs(lev[:, 2]) < 0.05].any()


def test_ceiling_excluded_box_and_wall_kept():
    pts = synthetic_room()
    est = GroundEstimator(GroundParams(smooth=1.0)).fit(pts)
    lev = to_level_frame(pts, est)
    band = HeightBand(obstacle_max=0.9)
    _, obs = classify(lev, band)
    assert lev[obs][:, 2].max() <= 0.9 + 1e-6
    sel = obstacle_points(lev, band, voxel=0.05)
    assert (sel[:, 0] > 4.9).sum() > 100          # wall
    assert ((sel[:, 0] > 1.9) & (sel[:, 0] < 2.6)).sum() > 20  # box


def test_scan_picks_nearest_range():
    pts = synthetic_room()
    est = GroundEstimator(GroundParams(smooth=1.0)).fit(pts)
    lev = to_level_frame(pts, est)
    ranges = scan_from_points(lev, HeightBand(), num_beams=360)
    # beam straight ahead (azimuth 0) sees the box at x = 2.0 only for y in [-0.5, 0]; the wall elsewhere
    ahead = ranges[180 + 20]  # +20 deg: sees the wall at 5/cos(20)
    assert abs(ahead - 5.0 / math.cos(math.radians(20.5))) < 0.2
    assert np.isfinite(ranges[180 - 30:180 + 30]).all()


def test_fit_failure_keeps_last_estimate():
    est_fn = GroundEstimator(GroundParams(smooth=1.0))
    good = est_fn.fit(synthetic_room())
    bad = est_fn.fit(np.zeros((10, 3)))
    assert not bad.valid
    assert np.allclose(bad.normal, good.normal) and bad.offset == good.offset
