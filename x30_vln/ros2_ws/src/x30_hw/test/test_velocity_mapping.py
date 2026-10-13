"""Speed mapping, signs, dead zone and the three-source gait table discrepancy (digest 3.4 and 3.6)."""

import pytest

from x30_hw import protocol as P

MOUNTAIN = P.DEFAULT_NAV_BRIDGE_TABLE[33]


def test_signs_plus_vy_left_and_plus_wz_ccw_need_negative_axes():
    f, lat, yaw = P.twist_to_axes(0.5, 0.4, 0.6, MOUNTAIN)
    assert f > 0  # forward is +
    assert lat < 0  # +vy (left) -> negative axis (robot: + = right)
    assert yaw < 0  # +wz (CCW) -> negative axis (robot: + = turn right)
    f, lat, yaw = P.twist_to_axes(-0.5, -0.4, -0.6, MOUNTAIN)
    assert f < 0 and lat > 0 and yaw > 0


def test_full_scale_values():
    assert P.twist_to_axes(1.0, 0, 0, MOUNTAIN)[0] == 32767
    assert P.twist_to_axes(-1.0, 0, 0, MOUNTAIN)[0] == -32767
    assert P.twist_to_axes(5.0, -5.0, 5.0, MOUNTAIN) == (32767, 32767, -32767)  # saturation, no overflow


def test_mountain_half_speed():
    f, lat, yaw = P.twist_to_axes(0.5, 0.4, 0.6, MOUNTAIN)
    assert f == int(0.5 * 32767)
    assert lat == -int(0.5 * 32767)  # 0.4 / 0.8
    assert yaw == -int(0.5 * 32767)  # 0.6 / 1.2


def test_dead_zone_is_two_percent():
    # mountain: 2 % of 1.0 m/s forward = 0.02 m/s -> 655 counts
    assert P.velocity_to_axis(0.019, 1.0) == 0
    assert P.velocity_to_axis(0.0205, 1.0) == int(0.0205 / 1.0 * 32767)
    assert abs(P.velocity_to_axis(0.0205, 1.0)) >= P.AXIS_DEAD_ZONE
    assert P.velocity_to_axis(-0.019, 1.0) == 0


def test_backward_uses_backward_limit():
    walk = P.API_APPENDIX_B[0]  # fwd 1.2, bwd 0.75
    assert P.twist_to_axes(0.375, 0, 0, walk)[0] == int(0.375 / 1.2 * 32767)
    assert P.twist_to_axes(-0.375, 0, 0, walk)[0] == -int(0.375 / 0.75 * 32767)


def test_axes_roundtrip_to_physical_velocity():
    limits = P.API_APPENDIX_B[33]
    for twist in [(0.4, 0.0, 0.0), (0.2, 0.15, 0.3), (-0.1, -0.2, -0.5), (0.0, 0.0, 0.0)]:
        axes = P.twist_to_axes(*twist, limits)
        back = P.axes_to_twist(*axes, limits)
        assert back == pytest.approx(twist, abs=2e-4)


def test_robot_side_sign_convention():
    """A positive lateral axis moves the robot RIGHT (negative body y); positive yaw axis turns right (CW)."""
    vx, vy, wz = P.axes_to_twist(0, 10000, 10000, P.API_APPENDIX_B[33])
    assert vy < 0 and wz < 0


def test_default_table_matches_api_appendix_b():
    for gait, lim in P.API_APPENDIX_B.items():
        assert P.DEFAULT_NAV_BRIDGE_TABLE[gait] == lim


def test_upstream_nav_bridge_table_discrepancy_is_documented():
    up, api = P.NAV_BRIDGE_UPSTREAM, P.API_APPENDIX_B
    # slope is over-driven 2.3x: asking 0.3 m/s commands full scale, the robot then does 0.7 m/s
    assert api[2][0] / up[2][0] == pytest.approx(2.333, abs=0.01)
    # walk lateral: upstream thinks 0.75, the robot's limit is 0.15 -> robot is 5x slower than requested
    assert up[0][2] / api[0][2] == pytest.approx(5.0)
    # only L-Walk and Mountain agree between the two
    agree = [g for g in api if up[g] == api[g]]
    assert set(agree) >= {32, 33} and 2 not in agree and 0 not in agree


def test_unlisted_gaits_use_the_largest_known_value():
    for gait in (34, 36, 3):
        known = [t[gait] for t in (P.NAV_BRIDGE_UPSTREAM, P.DOGX30) if gait in t]
        for axis in (0, 2, 3):
            assert P.DEFAULT_NAV_BRIDGE_TABLE[gait][axis] >= max(k[axis] for k in known) - 1e-9
