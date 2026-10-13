import math

import numpy as np

from x30_datasets.transforms import (SceneTransform, habitat_pose_to_world, quat_xyzw_to_matrix, rpy_to_matrix,
                                     matrix_to_rpy, wrap_angle, world_pose_to_habitat)


def test_rpy_roundtrip():
    for rpy in [(0.1, -0.2, 0.7), (0, 0, -3.0), (0.3, 0.4, 2.0)]:
        assert np.allclose(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy, atol=1e-9)


def test_habitat_position_axes():
    tf = SceneTransform.habitat(0.0)
    assert np.allclose(tf.point([1.0, 2.0, 3.0]), [1.0, -3.0, 2.0])  # y up -> z up, -z_h -> +y


def test_habitat_floor_shift():
    tf = SceneTransform.habitat(floor_y=0.5)
    assert abs(tf.point([0, 0.5, 0])[2]) < 1e-9
    assert abs(tf.point([0, 1.5, 0])[2] - 1.0) < 1e-9


def test_identity_rotation_faces_plus_y():
    # habitat identity heading is -z, which maps to +y in the world: yaw = +90 deg
    _, _, _, yaw = habitat_pose_to_world([0, 0, 0], [0, 0, 0, 1], SceneTransform.habitat())
    assert abs(yaw - math.pi / 2) < 1e-9


def test_heading_plus_x_is_yaw_zero():
    # rotate the habitat agent so that it looks along +x: -90 deg about y
    a = -math.pi / 2
    q = [0, math.sin(a / 2), 0, math.cos(a / 2)]
    assert np.allclose(quat_xyzw_to_matrix(q) @ [0, 0, -1], [1, 0, 0], atol=1e-9)
    _, _, _, yaw = habitat_pose_to_world([0, 0, 0], q, SceneTransform.habitat())
    assert abs(yaw) < 1e-9


def test_pose_roundtrip():
    tf = SceneTransform.habitat(0.2)
    for yaw in (-2.5, -0.3, 0.0, 1.2, 3.0):
        p, q = world_pose_to_habitat(1.5, -2.0, 0.0, yaw, tf)
        x, y, z, yaw2 = habitat_pose_to_world(p, q, tf)
        assert np.allclose([x, y, z], [1.5, -2.0, 0.0], atol=1e-9)
        assert abs(wrap_angle(yaw2 - yaw)) < 1e-9


def test_up_axis_transforms():
    assert np.allclose(SceneTransform.from_up_axis("y").point([1, 2, 3]), [1, -3, 2])
    assert np.allclose(SceneTransform.from_up_axis("z", scale=0.5).point([2, 2, 2]), [1, 1, 1])
    assert abs(SceneTransform.from_up_axis("y", floor=1.0).point([0, 1.0, 0])[2]) < 1e-9
