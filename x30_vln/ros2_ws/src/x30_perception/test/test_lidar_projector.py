"""Accuracy of the lidar-to-mask lifting on a synthetic room with known geometry."""

import numpy as np

from x30_perception import synthetic as syn
from x30_perception.camera_model import CameraModel
from x30_perception.geometry import apply, invert, make_transform
from x30_perception.lidar_projector import (
    ProjectorParams,
    erode_mask,
    lift_mask,
    lift_mask_from_depth,
    merge_scans,
    project_points,
    select_scans,
)

CAM = CameraModel.from_fov(640, 480, np.radians(90))
# optical frame of the camera expressed in the lidar frame: 15 cm ahead of and 20 cm below the lidar
T_LIDAR_OPT = syn.optical_pose((0.15, 0.0, -0.2))


def _scene():
    return syn.Scene(
        [
            syn.Billboard('box_a', (3.0, 0.0, 0.5), 0.6, 1.0),
            syn.Billboard('box_b', (5.0, 1.5, 0.75), 0.8, 1.5),
            syn.Billboard('box_c', (8.0, -1.5, 1.0), 1.0, 2.0),
            syn.Billboard('small', (2.5, -1.0, 0.8), 0.3, 0.3),
        ]
    )


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _static_lidar(t):
    return make_transform(np.eye(3), (0.0, 0.0, 1.0))


def _moving_lidar(t, v=0.6, wz=0.5):
    yaw = wz * t
    return make_transform(_yaw(yaw), (v * t * np.cos(yaw), v * t * np.sin(yaw), 1.0))


def _lift_all(scene, lidar_fn, t_img=0.15, scan_pts=80000, compensate=True, params=None):
    p = params or ProjectorParams()
    rng = np.random.default_rng(3)
    scans = [syn.simulate_scan(scene, lidar_fn, t0, scan_pts, rng=rng) for t0 in (0.0, 0.1, 0.2)]
    scans = select_scans(scans, t_img, 3)
    # "odom" == world; pose_fn is what TF would return: T_odom_lidar(t). Without compensation one pose is used for all points.
    pose_fn = lidar_fn if compensate else (lambda t: lidar_fn(t_img))
    pts_w, info = merge_scans(scans, pose_fn, time_bins=5, min_range_m=p.min_range_m)
    T_world_cam = lidar_fn(t_img) @ T_LIDAR_OPT
    proj = project_points(CAM, apply(invert(T_world_cam), pts_w), p)
    _, depth, labels = syn.render(scene, CAM, T_world_cam, with_texture=False)
    out = []
    for i, b in enumerate(scene.billboards):
        mask = labels == i
        if mask.sum() < 20:
            continue
        out.append((b, mask, lift_mask(proj, mask, CAM, p, T_out_cam=T_world_cam), depth))
    return out, proj, info, T_world_cam


def test_static_scene_position_error():
    results, _, info, _ = _lift_all(_scene(), _static_lidar)
    assert info['scans_used'] == 3
    assert len(results) == 4
    for b, mask, lift, depth in results:
        assert not lift.bearing_only, (b.label, lift.reason, lift.n_points)
        c = np.array(b.center)
        # the visible surface is the plane x = center.x: depth error must be at noise level, lateral error small
        assert abs(lift.centroid[0] - c[0]) < 0.08, b.label
        assert np.linalg.norm(lift.centroid[1:] - c[1:]) < 0.25 * max(b.width, b.height) + 0.05, b.label
        assert lift.extent[1] <= b.width * 1.1 + 0.1 and lift.extent[2] <= b.height * 1.1 + 0.1


def test_depth_image_path_agrees_with_lidar():
    results, _, _, T_cam = _lift_all(_scene(), _static_lidar)
    for b, mask, lift, depth in results:
        d = lift_mask_from_depth(depth, mask, CAM, ProjectorParams(), T_out_cam=T_cam)
        assert not d.bearing_only
        assert np.linalg.norm(d.centroid - lift.centroid) < 0.25


def test_tiny_or_unobserved_mask_is_bearing_only():
    _, proj, _, _ = _lift_all(_scene(), _static_lidar, scan_pts=300)
    mask = np.zeros((480, 640), bool)
    mask[200:206, 300:306] = True
    lift = lift_mask(proj, mask, CAM, ProjectorParams())
    assert lift.bearing_only and lift.centroid is None
    assert abs(np.linalg.norm(lift.bearing_cam) - 1) < 1e-9
    assert lift.bearing_cam[2] > 0.9


def test_background_bleed_is_rejected():
    """A thin post in front of a wall and a mask that is too wide (as a real segmenter produces): the wall points inside
    the mask must not pull the estimate back to the wall."""
    scene = syn.Scene([syn.Billboard('post', (3.0, 0.0, 0.8), 0.15, 1.6)])
    p = ProjectorParams()
    scans = [syn.simulate_scan(scene, _static_lidar, t0, 80000, rng=np.random.default_rng(5)) for t0 in (0.0, 0.1, 0.2)]
    pts_w, _ = merge_scans(scans, _static_lidar, 1, p.min_range_m)
    T_cam = _static_lidar(0) @ T_LIDAR_OPT
    proj = project_points(CAM, apply(invert(T_cam), pts_w), p)
    _, _, labels = syn.render(scene, CAM, T_cam, with_texture=False)
    ys, xs = np.nonzero(labels == 0)
    wide = np.zeros((480, 640), bool)
    wide[ys.min() : ys.max() + 1, xs.min() - 25 : xs.max() + 26] = True
    lift = lift_mask(proj, wide, CAM, p, T_out_cam=T_cam)
    assert not lift.bearing_only
    assert abs(lift.centroid[0] - 3.0) < 0.1


def test_motion_compensation_reduces_error():
    """Robot moving at 0.6 m/s while turning 0.5 rad/s: one pose per scan smears the object."""
    scene = _scene()
    err = {}
    for comp in (True, False):
        results, _, _, T_cam = _lift_all(scene, _moving_lidar, compensate=comp)
        e = [np.linalg.norm(lift.centroid[:2] - np.array(b.center[:2])) for b, _, lift, _ in results if not lift.bearing_only]
        err[comp] = (float(np.mean(e)), len(e))
    assert err[True][1] >= 3
    assert err[True][0] < 0.35
    assert err[True][0] < err[False][0]


def test_erode_keeps_image_border():
    m = np.zeros((20, 20), bool)
    m[:, :10] = True
    e = erode_mask(m, 2)
    assert e[:, :8].all() and not e[:, 8:].any()
