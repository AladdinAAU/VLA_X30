"""Generate the robot description (URDF text) for the two locomotion backends.

legged    : the xacro output unchanged (upstream gz_ros2_control hardware + the legged controller drive the joints).
kinematic : same URDF with the 12 leg joints frozen in the standing pose (fixed joints, the origin absorbs the joint angle),
            ros2_control and its Gazebo plugin removed, gravity off, a body-footprint bumper collision added and the Gazebo
            VelocityControl plugin added. The feet end up `spawn_clearance_m` above the ground when the model is spawned at the
            returned standing height, so no ground contact (and no friction) fights the commanded velocity.
Pure python apart from xacro; FK helpers are unit-tested without ROS.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .geometry import axis_angle_matrix, rpy_from_matrix, rpy_matrix  # noqa: F401

LEG_JOINT_SUFFIX = ('HipX', 'HipY', 'Knee')
LEG_PREFIXES = ('FR', 'FL', 'HR', 'HL')
FOOT_RADIUS_M = 0.03        # upstream x30_description foot collision sphere


def _floats(text: str | None, n: int) -> list[float]:
    return [float(v) for v in (text or '').split()] if text else [0.0] * n


def joint_origin(joint: ET.Element) -> np.ndarray:
    org = joint.find('origin')
    xyz = _floats(org.get('xyz') if org is not None else None, 3)
    rpy = _floats(org.get('rpy') if org is not None else None, 3)
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def set_joint_origin(joint: ET.Element, T: np.ndarray) -> None:
    org = joint.find('origin')
    if org is None:
        org = ET.SubElement(joint, 'origin')
    org.set('xyz', ' '.join(f'{v:.9g}' for v in T[:3, 3]))
    org.set('rpy', ' '.join(f'{v:.9g}' for v in rpy_from_matrix(T[:3, :3])))


def stand_angle(joint_name: str, stand_joints) -> float | None:
    for prefix in LEG_PREFIXES:
        for k, suffix in enumerate(LEG_JOINT_SUFFIX):
            if joint_name == f'{prefix}_{suffix}':
                return float(stand_joints[k])
    return None


def link_poses(root: ET.Element, base: str, angles: dict[str, float]) -> dict[str, np.ndarray]:
    """Forward kinematics of every link in the frame of `base` for the given joint angles (missing -> 0)."""
    children: dict[str, list[ET.Element]] = {}
    for j in root.findall('joint'):
        children.setdefault(j.find('parent').get('link'), []).append(j)
    poses = {base: np.eye(4)}
    stack = [base]
    while stack:
        link = stack.pop()
        for j in children.get(link, []):
            T = poses[link] @ joint_origin(j)
            q = angles.get(j.get('name'), 0.0)
            if j.get('type') in ('revolute', 'continuous') and q != 0.0:
                ax = j.find('axis')
                A = np.eye(4)
                A[:3, :3] = axis_angle_matrix(_floats(ax.get('xyz') if ax is not None else None, 3), q)
                T = T @ A
            child = j.find('child').get('link')
            poses[child] = T
            stack.append(child)
    return poses


def standing_height(root: ET.Element, stand_joints, base: str = 'TORSO') -> float:
    """Height of the TORSO origin above the ground when the feet (sphere radius FOOT_RADIUS_M) touch it."""
    angles = {f'{p}_{s}': stand_joints[k] for p in LEG_PREFIXES for k, s in enumerate(LEG_JOINT_SUFFIX)}
    poses = link_poses(root, base, angles)
    feet = [poses[f'{p}_FOOT'][2, 3] for p in LEG_PREFIXES if f'{p}_FOOT' in poses]
    if not feet:
        raise ValueError('no *_FOOT links in the URDF')
    return float(-min(feet) + FOOT_RADIUS_M)


def freeze_legs(root: ET.Element, stand_joints) -> int:
    n = 0
    for j in root.findall('joint'):
        q = stand_angle(j.get('name'), stand_joints)
        if q is None:
            continue
        ax = j.find('axis')
        T = joint_origin(j)
        A = np.eye(4)
        A[:3, :3] = axis_angle_matrix(_floats(ax.get('xyz') if ax is not None else None, 3), q)
        set_joint_origin(j, T @ A)
        j.set('type', 'fixed')
        for tag in ('axis', 'limit', 'dynamics', 'mimic', 'safety_controller'):
            for el in j.findall(tag):
                j.remove(el)
        n += 1
    return n


def add_kinematic_extras(root: ET.Element, bumper: dict | None) -> None:
    for el in root.findall('ros2_control'):
        root.remove(el)
    for g in root.findall('gazebo'):
        for p in g.findall('plugin'):
            if 'gz_ros2_control' in (p.get('filename', '') + p.get('name', '')):
                g.remove(p)
    for link in root.findall('link'):
        g = ET.SubElement(root, 'gazebo', reference=link.get('name'))
        ET.SubElement(g, 'gravity').text = 'false'
    if bumper:
        torso = next(link for link in root.findall('link') if link.get('name') == 'TORSO')
        col = ET.SubElement(torso, 'collision', name='x30_bumper')
        ET.SubElement(col, 'origin', xyz=' '.join(str(v) for v in bumper['xyz']), rpy='0 0 0')
        geo = ET.SubElement(col, 'geometry')
        ET.SubElement(geo, 'box', size=' '.join(str(v) for v in bumper['size']))
    g = ET.SubElement(root, 'gazebo')
    ET.SubElement(g, 'plugin', filename='gz-sim-velocity-control-system', name='gz::sim::systems::VelocityControl')


def process_xacro(xacro_file: str | Path, mappings: dict[str, str]) -> str:
    import xacro
    return xacro.process_file(str(xacro_file), mappings=mappings).toxml()


def simplify_visuals(root: ET.Element, cache_dir: Path, voxel: float) -> int:
    """Point every visual mesh at a decimated copy (see mesh_simplify); returns the number of meshes replaced."""
    from .mesh_simplify import simplified_copy
    n = 0
    for mesh in root.iter('mesh'):
        fn = mesh.get('filename', '')
        if not fn.startswith('file://') or fn.endswith('_mm.obj'):
            continue
        src = Path(fn[len('file://'):])
        if src.is_file():
            mesh.set('filename', 'file://' + str(simplified_copy(src, cache_dir, voxel)))
            n += 1
    return n


def build(xacro_file: str | Path, robot_config: str | Path, locomotion: str, thermal: bool, cfg: dict,
          mesh_cache: str | Path | None = None) -> tuple[str, float]:
    """Return (urdf_text, standing_height_m)."""
    text = process_xacro(xacro_file, {'GAZEBO': 'true', 'CLASSIC': 'false', 'robot_config': str(robot_config),
                                      'thermal': 'true' if thermal else 'false'})
    root = ET.fromstring(text)
    stand = cfg['robot']['stand_joints']
    height = standing_height(root, stand)
    voxel = cfg['robot'].get('visual_mesh_voxel_m', 0.0)
    if voxel and mesh_cache is not None:
        simplify_visuals(root, Path(mesh_cache), float(voxel))
    if locomotion == 'kinematic':
        freeze_legs(root, stand)
        add_kinematic_extras(root, {'xyz': (0.0, 0.0, 0.0), 'size': (0.98, 0.30, 0.20)})
    return ET.tostring(root, encoding='unicode'), height
