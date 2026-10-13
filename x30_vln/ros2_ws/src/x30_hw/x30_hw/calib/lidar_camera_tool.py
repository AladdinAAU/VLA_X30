"""Lidar-camera extrinsic fine-tuning: project the merged cloud onto a camera image, adjust 6 DoF, save.

Interactive (needs a display):
    calibrate_lidar_camera --image frame.png --cloud scan.npy --intrinsics configs/calibration/camera.yaml
    calibrate_lidar_camera --ros --accumulate 5                     # live /camera/color/image_raw + /lidar_points
Keys:  i/k x +/-   j/l y +/-   u/o z +/-   (camera_link axes: x forward, y left, z up; metres)
       w/s pitch +/-   a/d yaw +/-   q/e roll +/-   (degrees)      [ ] smaller/larger step    r reset   p save   ESC quit
Headless (also used by the tests):
    calibrate_lidar_camera --image f.png --cloud s.npy --script "x+0.05,yaw-2" --out-image check.png --save

Output: configs/calibration/lidar_camera.yaml  {camera_link: {xyz, rpy_deg}, meta: {...}}, consumed by
static_tf_publisher (parameter overlay_file / launch argument tf_overlay). Use scenes with strong depth edges (door frames,
corners, poles): a correct extrinsic makes the coloured points hug the object borders in the image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import yaml

from x30_hw import config as cfgmod
from x30_hw.calib import projection as pj

STEP_T = 0.01  # m
STEP_R = 0.5  # deg


def load_intrinsics(path: Optional[str], width: int, height: int, hfov_deg: float):
    if path and os.path.isfile(path):
        d = yaml.safe_load(open(path, encoding="utf-8"))
        k = np.array(d["camera_matrix"]["data"], float).reshape(3, 3)
        dist = np.array(d["distortion_coefficients"]["data"], float)
        sx, sy = width / d["image_width"], height / d["image_height"]
        k = k.copy()
        k[0] *= sx
        k[1] *= sy
        return k, dist
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    print(f"WARNING: no intrinsics file; using a PLACEHOLDER pinhole (hfov {hfov_deg} deg). Calibrate the camera first.")
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]]), np.zeros(5)


def model_from_config(cfg) -> pj.ExtrinsicModel:
    ex = cfg["extrinsics"]
    return pj.ExtrinsicModel(ex["camera_link"]["xyz"], ex["camera_link"]["rpy_deg"], ex["optical_rpy_deg"],
                             ex["lidar_link"]["xyz"], ex["lidar_link"]["rpy_deg"])


def render(model, image, cloud, k, dist, radius=2):
    uv, depth, valid = pj.project_points(cloud, model.cam_from_lidar(), k, dist, (image.shape[1], image.shape[0]))
    return pj.overlay_points(image, uv, depth, valid, radius), int(valid.sum())


def save_overlay(model, path, note="manual keyboard fine-tuning"):
    out = model.overlay()
    out["meta"] = {"method": note, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "initial_xyz": model.initial[0], "initial_rpy_deg": model.initial[1]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return out


def grab_ros(accumulate: int, timeout: float = 20.0):
    """Latest camera image and an accumulation of `accumulate` scans (lidar_link frame)."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2

    rclpy.init()
    node = Node("lidar_camera_tool")
    box = {"img": None, "scans": []}

    def on_img(m):
        box["img"] = m

    def on_pc(m):
        if len(box["scans"]) < accumulate:
            box["scans"].append(m)

    node.create_subscription(Image, "/camera/color/image_raw", on_img, qos_profile_sensor_data)
    node.create_subscription(PointCloud2, "/lidar_points", on_pc, qos_profile_sensor_data)
    t0 = time.time()
    while (box["img"] is None or len(box["scans"]) < accumulate) and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if box["img"] is None or not box["scans"]:
        raise SystemExit("no image or no cloud received")
    m = box["img"]
    img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)[..., ::-1].copy()  # rgb8 -> bgr
    pts = []
    for s in box["scans"]:
        dt = np.dtype({"names": [f.name for f in s.fields], "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"][:len(s.fields)],
                       "offsets": [f.offset for f in s.fields], "itemsize": s.point_step})
        a = np.frombuffer(bytes(s.data), dtype=dt)
        pts.append(np.stack([a["x"], a["y"], a["z"]], axis=1))
    cloud = np.concatenate(pts)
    return img, cloud[np.isfinite(cloud).all(axis=1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--cloud")
    ap.add_argument("--ros", action="store_true", help="grab image and cloud from the running ROS graph")
    ap.add_argument("--accumulate", type=int, default=5)
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hfov", type=float, default=100.0)
    ap.add_argument("--script", default="", help="headless adjustments, e.g. 'x+0.05,yaw-2'")
    ap.add_argument("--out-image", default=None)
    ap.add_argument("--save", action="store_true", help="write the overlay file")
    ap.add_argument("--out", default=None, help="overlay path (default configs/calibration/lidar_camera.yaml)")
    a = ap.parse_args(argv)

    import cv2

    cfg = cfgmod.load_robot_config(a.config)
    if a.ros:
        image, cloud = grab_ros(a.accumulate)
    else:
        if not (a.image and a.cloud):
            ap.error("--image and --cloud are required without --ros")
        image = cv2.imread(a.image)
        cloud = pj.load_cloud(a.cloud)
    h, w = image.shape[:2]
    intr_path = a.intrinsics or os.path.join(cfgmod.configs_dir(), cfg["camera"]["intrinsics_file"])
    k, dist = load_intrinsics(intr_path, w, h, a.hfov)
    model = model_from_config(cfg)
    out_path = a.out or os.path.join(cfgmod.configs_dir(), "calibration", "lidar_camera.yaml")

    for axis, delta in pj.parse_adjust_script(a.script):
        model.adjust(axis, delta)
    if a.script or not os.environ.get("DISPLAY"):
        vis, n = render(model, image, cloud, k, dist)
        print(f"{n} of {len(cloud)} points inside the image; extrinsic {model.overlay()}")
        if a.out_image:
            cv2.imwrite(a.out_image, vis)
        if a.save:
            save_overlay(model, out_path)
            print("saved", out_path)
        return 0

    step_t, step_r = STEP_T, STEP_R
    keys = {"i": ("x", +1), "k": ("x", -1), "j": ("y", +1), "l": ("y", -1), "u": ("z", +1), "o": ("z", -1),
            "w": ("pitch", +1), "s": ("pitch", -1), "a": ("yaw", +1), "d": ("yaw", -1), "q": ("roll", +1), "e": ("roll", -1)}
    while True:
        vis, n = render(model, image, cloud, k, dist)
        cv2.putText(vis, f"x{model.xyz[0]:.3f} y{model.xyz[1]:.3f} z{model.xyz[2]:.3f} r{model.rpy[0]:.1f} p{model.rpy[1]:.1f} "
                         f"y{model.rpy[2]:.1f} step {step_t * 1000:.0f}mm/{step_r:.1f}deg pts {n}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("lidar-camera overlay", vis)
        c = cv2.waitKey(50) & 0xFF
        if c == 27:
            break
        ch = chr(c) if 0 < c < 128 else ""
        if ch in keys:
            axis, sign = keys[ch]
            model.adjust(axis, sign * (step_t if axis in "xyz" else step_r))
        elif ch == "[":
            step_t, step_r = step_t / 2, step_r / 2
        elif ch == "]":
            step_t, step_r = step_t * 2, step_r * 2
        elif ch == "r":
            model.reset()
        elif ch == "p":
            save_overlay(model, out_path)
            print("saved", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
