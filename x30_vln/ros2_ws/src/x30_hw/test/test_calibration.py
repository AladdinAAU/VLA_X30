"""Camera calibration on synthetic checkerboard views with known intrinsics; lidar projection and overlay round trip."""

import os

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

from x30_hw import config as cfgmod  # noqa: E402
from x30_hw.calib import calibrate_camera as cc  # noqa: E402
from x30_hw.calib import lidar_camera_tool as lct  # noqa: E402
from x30_hw.calib import projection as pj  # noqa: E402
from x30_hw.transforms import build_transforms  # noqa: E402

BOARD = (9, 6)
SQ = 0.04
K_TRUE = np.array([[820.0, 0, 645.0], [0, 815.0, 355.0], [0, 0, 1.0]])
SIZE = (1280, 720)


def render_board(rvec, tvec, k=K_TRUE, size=SIZE):
    s, m = 40, 1  # texture pixels per square, white margin in squares
    cols, rows = BOARD[0] + 1, BOARD[1] + 1
    tex = np.full(((rows + 2 * m) * s, (cols + 2 * m) * s), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[(r + m) * s:(r + m + 1) * s, (c + m) * s:(c + m + 1) * s] = 0
    th, tw = tex.shape
    # plane coordinates (metres) of the texture corners: inner corner (0,0) sits at texture grid (1+m, 1+m)
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    plane = np.array([[(x / s - 1 - m) * SQ, (y / s - 1 - m) * SQ, 0.0] for x, y in src])
    img_pts, _ = cv2.projectPoints(plane, rvec, tvec, k, None)
    h = cv2.getPerspectiveTransform(src, img_pts.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(tex, h, size, borderValue=200)


def make_views(n=14, seed=0):
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n):
        rvec = np.radians(rng.uniform(-30, 30, 3)) * np.array([1, 1, 0.5])
        tvec = np.array([rng.uniform(-0.2, 0.0), rng.uniform(-0.12, 0.05), rng.uniform(0.55, 0.9)])
        views.append(render_board(rvec, tvec))
    return views


def test_calibration_recovers_intrinsics(tmp_path):
    for i, v in enumerate(make_views()):
        cv2.imwrite(str(tmp_path / f"v{i:02d}.png"), v)
    out = tmp_path / "camera.yaml"
    rc = cc.main(["--images", str(tmp_path), "--board", "9x6", "--square", str(SQ), "--out", str(out)])
    assert rc == 0
    d = yaml.safe_load(out.read_text())
    k = np.array(d["camera_matrix"]["data"]).reshape(3, 3)
    assert d["rms_reprojection_error"] < 0.5 and d["calibrated"] and d["num_views"] >= 10
    assert k[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.03) and k[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.03)
    assert abs(k[0, 2] - K_TRUE[0, 2]) < 15 and abs(k[1, 2] - K_TRUE[1, 2]) < 15
    assert d["image_width"] == 1280 and d["distortion_model"] == "plumb_bob"


def test_too_few_views_refused(tmp_path):
    for i, v in enumerate(make_views(4)):
        cv2.imwrite(str(tmp_path / f"v{i}.png"), v)
    assert cc.main(["--images", str(tmp_path), "--square", str(SQ), "--out", str(tmp_path / "c.yaml")]) == 2
    assert not (tmp_path / "c.yaml").exists()


def test_projection_geometry_straight_ahead():
    m = pj.ExtrinsicModel([0.4, 0.0, 0.1], [0, 0, 0])
    uv, depth, valid = pj.project_points(np.array([[3.0, 0.0, 0.1], [3.0, 0.5, 0.1], [-1.0, 0, 0.1]]), m.cam_from_lidar(), K_TRUE)
    assert valid.tolist() == [True, True, False]
    assert uv[0] == pytest.approx([K_TRUE[0, 2], K_TRUE[1, 2]], abs=1e-6) and depth[0] == pytest.approx(2.6)
    assert uv[1][0] < K_TRUE[0, 2]  # +y (left) appears on the LEFT of the image


def test_adjustment_script_recovers_true_extrinsic():
    rng = np.random.default_rng(1)
    cloud = rng.uniform([1, -2, -0.3], [8, 2, 1.5], (500, 3))
    truth = pj.ExtrinsicModel([0.45, 0.02, 0.12], [0.0, 3.0, 2.0])
    uv_true, _, _ = pj.project_points(cloud, truth.cam_from_lidar(), K_TRUE)
    m = pj.ExtrinsicModel([0.40, 0.0, 0.10], [0.0, 0.0, 0.0])
    uv0, _, _ = pj.project_points(cloud, m.cam_from_lidar(), K_TRUE)
    for axis, delta in pj.parse_adjust_script("x+0.05,y+0.02,z+0.02,pitch+3,yaw+2"):
        m.adjust(axis, delta)
    uv1, _, _ = pj.project_points(cloud, m.cam_from_lidar(), K_TRUE)
    assert np.abs(uv1 - uv_true).max() < 1e-6
    assert np.abs(uv0 - uv_true).mean() > 20
    m.reset()
    assert m.xyz == [0.40, 0.0, 0.10]


def test_parse_script():
    assert pj.parse_adjust_script("x+0.05, yaw-2 ,pitch+1.5") == [("x", 0.05), ("yaw", -2.0), ("pitch", 1.5)]
    with pytest.raises(ValueError):
        pj.parse_adjust_script("x0.05")


def test_headless_tool_saves_overlay_that_static_tf_uses(tmp_path):
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "f.png"), img)
    np.save(tmp_path / "c.npy", np.random.default_rng(2).uniform([1, -2, -0.3], [8, 2, 1.5], (300, 3)))
    ov = tmp_path / "lidar_camera.yaml"
    rc = lct.main(["--image", str(tmp_path / "f.png"), "--cloud", str(tmp_path / "c.npy"), "--script", "x+0.05,yaw-2",
                   "--out-image", str(tmp_path / "o.png"), "--save", "--out", str(ov), "--intrinsics", str(tmp_path / "none.yaml")])
    assert rc == 0 and (tmp_path / "o.png").exists()
    data = yaml.safe_load(ov.read_text())
    cfg = cfgmod.load_robot_config()
    base = cfg["extrinsics"]["camera_link"]
    assert data["camera_link"]["xyz"][0] == pytest.approx(base["xyz"][0] + 0.05)
    assert data["camera_link"]["rpy_deg"][2] == pytest.approx(base["rpy_deg"][2] - 2.0)
    tfs = {c: (xyz, st) for p, c, xyz, q, st in build_transforms(cfg, data)}
    assert tfs["camera_link"][0][0] == pytest.approx(base["xyz"][0] + 0.05) and tfs["camera_link"][1] == "calibrated"


def test_robot_config_is_complete_and_consistent():
    cfg = cfgmod.load_robot_config()
    tfs = build_transforms(cfg)
    children = {c for _, c, *_ in tfs}
    assert {"imu_link", "lidar_link", "camera_link", "camera_optical_frame", "lidar_fl_link", "lidar_rr_link"} <= children
    assert "thermal_link" not in children  # no thermal camera on the stock robot
    b = cfg["stages"]["B"]
    assert (b["vx_forward"], b["vx_backward"], b["vy"], b["wz"]) == (0.2, 0.1, 0.1, 0.3)  # digest 5.1 bring-up row
    c = cfg["stages"]["C"]
    assert (c["vx_forward"], c["vx_backward"], c["vy"], c["wz"]) == (0.4, 0.15, 0.2, 0.5)  # INTERFACES section 2
    assert not cfg["stages"]["A"]["allow_motion"]
    assert cfg["nav_bridge"]["cmd_vel_timeout_ms"] <= 500 and not cfg["nav_bridge"]["startup_acquire_control"]
    assert cfg["extrinsics"]["imu_link"]["xyz"] == [0.13045, -0.001, 0.0149]
    topics = yaml.safe_load(open(os.path.join(cfgmod.configs_dir(), "ros1_bridge_topics.yaml")))["topics"]
    names = [t["topic"] for t in topics]
    assert "/cmd_vel" not in names and "/tf" not in names and "/lidar_points" in names and "/imu/data" in names
    unsafe = yaml.safe_load(open(os.path.join(cfgmod.configs_dir(), "ros1_bridge_topics_UNSAFE_with_cmd_vel.yaml")))["topics"]
    assert "/cmd_vel" in [t["topic"] for t in unsafe]
    no_imu = yaml.safe_load(open(os.path.join(cfgmod.configs_dir(), "ros1_bridge_topics_udp_imu.yaml")))["topics"]
    assert "/imu/data" not in [t["topic"] for t in no_imu]
