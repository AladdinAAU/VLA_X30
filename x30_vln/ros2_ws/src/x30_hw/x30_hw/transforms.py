"""Static transforms from configs/robot_real.yaml (no ROS): list of (parent, child, xyz, quaternion xyzw, status)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

Transform = Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float, float], str]


def quat_from_rpy_deg(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    r, p, y = (math.radians(a) / 2.0 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def build_transforms(cfg: Dict[str, Any], overlay: Dict[str, Any] | None = None) -> List[Transform]:
    """base_link -> every sensor frame. `overlay` (calibration result) replaces xyz / rpy_deg per frame name."""
    f = cfg["frames"]
    ex = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg["extrinsics"].items()}
    for name, vals in (overlay or {}).items():
        if name in ex:
            ex[name].update({k: vals[k] for k in ("xyz", "rpy_deg") if k in vals})
            ex[name]["status"] = "calibrated"
    out: List[Transform] = []

    def add(parent: str, child: str, key: str) -> None:
        e = ex[key]
        out.append((parent, child, tuple(e["xyz"]), quat_from_rpy_deg(*e["rpy_deg"]), e.get("status", "")))

    base = f["base"]
    add(base, f["imu"], "imu_link")
    add(base, f["lidar"], "lidar_link")
    for k in ("lidar_fl_link", "lidar_fr_link", "lidar_rl_link", "lidar_rr_link"):
        add(base, k, k)
    add(base, f["camera"], "camera_link")
    out.append((f["camera"], f["camera_optical"], (0.0, 0.0, 0.0), quat_from_rpy_deg(*ex["optical_rpy_deg"]), "standard"))
    if ex["thermal_link"].get("enabled", False):
        add(base, f["thermal"], "thermal_link")
        out.append((f["thermal"], f["thermal_optical"], (0.0, 0.0, 0.0), quat_from_rpy_deg(*ex["optical_rpy_deg"]), "standard"))
    return out
