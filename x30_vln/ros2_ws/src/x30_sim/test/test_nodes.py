"""Pure-function tests of the node modules (these import rclpy, so they run inside the sim container)."""
import numpy as np
import pytest

pytest.importorskip('rclpy')
from x30_sim import kinematic_driver as kd  # noqa: E402
from x30_sim import legged_bridge as lb  # noqa: E402
from x30_sim import thermal_processor as tp  # noqa: E402


def test_shape_axis_and_slew():
    assert kd.shape_axis(0.005, -0.6, 0.6, 0.012) == 0.0
    assert kd.shape_axis(2.0, -0.6, 0.6, 0.012) == 0.6
    assert kd.shape_axis(-0.3, -0.6, 0.6, 0.012) == -0.3
    assert kd.slew(0.0, 1.0, 0.1) == pytest.approx(0.1)
    assert kd.slew(0.95, 1.0, 0.1) == 1.0


def test_legged_normalise():
    assert lb.normalise(0.2, 0.4) == pytest.approx(0.5)
    assert lb.normalise(-5.0, 0.4) == -1.0


def test_thermal_conversion():
    raw = np.array([[29315, 30000]], dtype=np.uint16)      # 293.15 K and 300 K at 0.01 K per count
    c = tp.counts_to_celsius(raw, 0.01)
    assert c[0, 0] == pytest.approx(20.0, abs=0.01) and c[0, 1] == pytest.approx(26.85, abs=0.01)
    th = {'display_auto': True, 'display_min_span_c': 10.0, 'display_min_c': 15, 'display_max_c': 45}
    img = tp.to_display(np.full((4, 4), 22.0, np.float32), th)
    assert img.dtype == np.uint8 and 100 < img.mean() < 160          # a flat scene maps to mid grey
