"""Loading of configs/robot_sim.yaml and derived geometry (no ROS imports, usable in unit tests and launch files)."""
from __future__ import annotations

import math
import os
from pathlib import Path

import yaml


def find_robot_config(path: str | os.PathLike | None = None) -> Path:
    """Resolve the robot config: explicit path, then $X30_CONFIGS/robot_sim.yaml, then a repository-relative guess."""
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get('X30_CONFIGS'):
        candidates.append(Path(os.environ['X30_CONFIGS']) / 'robot_sim.yaml')
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / 'configs' / 'robot_sim.yaml')
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError('robot_sim.yaml not found (set X30_CONFIGS or pass robot_config); tried: '
                            + ', '.join(str(c) for c in candidates[:4]))


def load_robot_config(path: str | os.PathLike | None = None) -> dict:
    with open(find_robot_config(path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def lidar_mount_pose(mount: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(xyz in metres, rpy in radians) of the sensor frame in the body frame.

    The sensor z axis is the boresight: (+-cos p, 0, sin p) with p the pitch above the horizon and the sign + for front
    mounts, - for rear mounts. Rotating about y by beta maps z to (sin beta, 0, cos beta), so beta = atan2(n_x, n_z).
    """
    pitch = math.radians(mount['boresight_pitch_deg'])
    sign = 1.0 if mount['side'] == 'front' else -1.0
    nx, nz = sign * math.cos(pitch), math.sin(pitch)
    beta = math.atan2(nx, nz)
    xyz = tuple(v / 1000.0 for v in mount['position_mm'])
    return xyz, (0.0, beta, 0.0)


def rot_y(beta: float):
    import numpy as np
    c, s = math.cos(beta), math.sin(beta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def lidar_mount_matrix(mount: dict):
    """4x4 transform sensor -> body (numpy)."""
    import numpy as np
    xyz, rpy = lidar_mount_pose(mount)
    T = np.eye(4)
    T[:3, :3] = rot_y(rpy[1])
    T[:3, 3] = xyz
    return T
