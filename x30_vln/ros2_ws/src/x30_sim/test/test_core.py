"""Unit tests that need no Gazebo and no running ROS graph (numpy / yaml / xml only)."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from x30_sim import bridge_config, config, geometry, lidar_model, odometry, robot_description, sdf_scene, worlds


@pytest.fixture(scope='module')
def cfg():
    return config.load_robot_config()


def test_box_chord_and_self_mask():
    origin = np.array([0.0, 0.0, 0.0])
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])
    chord = lidar_model.box_chord(origin, dirs, np.array([5.0, 5.0, 5.0]), (1.0, 2.0, -1, 1, -1, 1))
    assert chord == pytest.approx([1.0, 0.0, 0.0])
    # a hit inside a box counts even when the chord is short
    mask = lidar_model.self_mask(origin, dirs, np.array([1.05, 1.0, 1.0]), {'b': (1.0, 2.0, -1, 1, -1, 1)}, min_chord=0.5)
    assert mask.tolist() == [True, False, False]


def test_point_layout_is_contract():
    assert lidar_model.POINT_DTYPE.itemsize == 26
    assert [lidar_model.POINT_DTYPE.fields[n][1] for n in lidar_model.POINT_DTYPE.names] == [0, 4, 8, 12, 16, 17, 18]


def test_intensity_and_tags(cfg):
    rng = np.random.default_rng(0)
    inten = lidar_model.synth_intensity(rng, np.full(20000, 2.0), cfg['lidar']['intensity'])
    assert 1 <= np.median(inten) <= 8 and inten.max() <= 255
    valid = np.r_[np.ones(5000, bool), np.zeros(5000, bool)]
    tags = lidar_model.sample_tags(rng, valid, cfg['lidar']['tag_distribution'])
    assert (tags[:5000] == 0).mean() > 0.7 and (tags[5000:] == 192).mean() > 0.25


def test_lidar_mount_axes(cfg):
    """The sensor z axis must point along the documented boresight."""
    T = config.lidar_mount_matrix(cfg['lidar']['mounts']['fl'])          # front-up: 25 deg above the horizon, forwards
    z = T[:3, :3] @ np.array([0, 0, 1.0])
    assert z == pytest.approx([math.cos(math.radians(25)), 0, math.sin(math.radians(25))], abs=1e-9)
    T = config.lidar_mount_matrix(cfg['lidar']['mounts']['rr'])          # rear-down: backwards and 56 deg down
    z = T[:3, :3] @ np.array([0, 0, 1.0])
    assert z == pytest.approx([-math.cos(math.radians(56)), 0, -math.sin(math.radians(56))], abs=1e-9)


def test_ray_count_matches_bag(cfg):
    lc = cfg['lidar']
    assert lc['horizontal_samples'] * lc['vertical_samples'] * 4 == 79872


def test_odometry_exact_and_noisy(cfg):
    nominal = cfg['odometry']['nominal']
    exact = odometry.LegOdometry(0.0, nominal)
    path = [(1.0 + 0.1 * i, 2.0 + 0.05 * i, 0.3 + 0.01 * i) for i in range(100)]
    for p in path:
        out = exact.update(*p)
    assert out == pytest.approx(exact.exact_frame(*path[-1]), abs=1e-9)
    noisy = odometry.LegOdometry(1.0, nominal, seed=3)
    for p in path:
        n = noisy.update(*p)
    err = math.hypot(n[0] - out[0], n[1] - out[1])
    assert 0.0 < err < 0.2 * math.hypot(out[0], out[1])


def test_geometry_roundtrip():
    R = geometry.rpy_matrix(0.1, -0.4, 2.0)
    assert np.allclose(geometry.rpy_matrix(*geometry.rpy_from_matrix(R)), R)


URDF = """<robot name="t">
 <link name="TORSO"/><link name="FL_HIP"/><link name="FL_THIGH"/><link name="FL_FOOT"/>
 <joint name="FL_HipX" type="revolute"><origin xyz="0 0 0"/><parent link="TORSO"/><child link="FL_HIP"/><axis xyz="-1 0 0"/><limit lower="-1" upper="1"/></joint>
 <joint name="FL_HipY" type="revolute"><origin xyz="0 0 0"/><parent link="FL_HIP"/><child link="FL_THIGH"/><axis xyz="0 -1 0"/></joint>
 <joint name="FL_Knee" type="revolute"><origin xyz="0 0 -0.3"/><parent link="FL_THIGH"/><child link="FL_FOOT"/><axis xyz="0 -1 0"/></joint>
</robot>"""


def test_freeze_legs_keeps_foot_pose():
    root = ET.fromstring(URDF)
    stand = [0.0, -0.5, 1.0]
    before = robot_description.link_poses(root, 'TORSO', {'FL_HipX': 0.0, 'FL_HipY': -0.5, 'FL_Knee': 1.0})['FL_FOOT']
    assert robot_description.freeze_legs(root, stand) == 3
    assert all(j.get('type') == 'fixed' for j in root.findall('joint'))
    after = robot_description.link_poses(root, 'TORSO', {})['FL_FOOT']
    assert np.allclose(before, after, atol=1e-9)


def test_default_world_scene():
    spec = worlds.resolve_world('default_arena')
    objs, info = sdf_scene.parse_scene(spec, None)
    assert {'person', 'radiator', 'fire_extinguisher', 'box', 'barrel'} <= {o.label for o in objs}
    person = next(o for o in objs if o.label == 'person')
    assert person.position == pytest.approx((-1.5, -3.5, 0.9), abs=1e-6) and 'warm' in person.attributes
    assert 'hot' in next(o for o in objs if o.label == 'radiator').attributes
    assert info['bounds']['max'][0] > 5.9


def test_prepare_world(tmp_path):
    spec = worlds.resolve_world('lab_room')
    path, name = worlds.prepare_world(spec, tmp_path, 0.005, 1.0, 295.15)
    text = path.read_text()
    assert name == 'lab_room' and 'gz-sim-sensors-system' in text and '<max_step_size>0.005' in text and 'atmosphere' in text
    assert 'gz-sim-imu-system' not in text


def test_bridge_entries(cfg):
    names = {e['ros_topic_name'] for e in bridge_config.bridge_entries(cfg, 'kinematic', True)}
    assert {'/clock', '/imu/data', '/camera/color/image_raw', '/lidar/fl/points', '/thermal/raw', '/x30/cmd_vel_gz'} <= names
    assert '/x30/cmd_vel_gz' not in {e['ros_topic_name'] for e in bridge_config.bridge_entries(cfg, 'legged', False)}
