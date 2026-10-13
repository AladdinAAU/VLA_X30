import numpy as np
import pytest

from x30_perception.camera_model import CameraModel


def _grid():
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)], axis=1)


@pytest.mark.parametrize(
    'cam',
    [
        CameraModel.from_fov(640, 480, np.radians(90)),
        CameraModel(640, 480, 400, 400, 320, 240, (-0.25, 0.08, 0.001, -0.002, 0.0), 'plumb_bob'),
        CameraModel(640, 480, 300, 300, 320, 240, (0.05, -0.01, 0.0, 0.0), 'equidistant'),
    ],
)
def test_rays_project_roundtrip(cam):
    uv = _grid()
    rays = cam.rays(uv)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    uv2, valid = cam.project(rays * 3.7)
    assert valid.all()
    assert np.abs(uv2 - uv).max() < 1e-3


def test_pinhole_values_and_scaling():
    cam = CameraModel.from_fov(640, 480, np.radians(90))
    uv, valid = cam.project(np.array([[0, 0, 2.0], [2.0, 0, 2.0], [0, 0, -1.0]]))
    assert np.allclose(uv[0], [320, 240]) and np.allclose(uv[1], [640, 240])
    assert list(valid) == [True, True, False]
    half = cam.scaled(0.5)
    assert (half.width, half.height) == (320, 240)
    assert np.allclose(half.project(np.array([[2.0, 0, 2.0]]))[0], [320, 120])


def test_from_camera_info_ignores_zero_distortion():
    cam = CameraModel.from_camera_info(640, 480, [500, 0, 320, 0, 500, 240, 0, 0, 1], [0, 0, 0, 0, 0])
    assert cam.dist == ()
