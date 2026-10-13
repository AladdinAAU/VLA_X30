"""Arming, interlocks, clamping, command timeout of the real_adapter gate."""

import math

import pytest

from x30_hw.safety_gate import GateLimits, Interlocks, RateMonitor, SafetyGate, clock_lag_level, rate_level

OK = Interlocks(estop=False, telemetry_ok=True, robot_ready=True, critical_error=False, battery_percent=80.0)


def gate(**kw):
    return SafetyGate(GateLimits(**kw))


def test_default_is_disarmed_and_forwards_nothing():
    g = gate()
    assert not g.armed
    assert g.evaluate((0.2, 0.0, 0.0), 0.0) == ((0.0, 0.0, 0.0), "disarmed")


def test_arm_requires_every_interlock():
    for field, bad in [("estop", True), ("telemetry_ok", False), ("robot_ready", False), ("critical_error", True)]:
        il = Interlocks(**{**OK.__dict__, field: bad})
        ok, msg = gate().arm(il)
        assert not ok and msg.startswith("refused"), field
    assert not gate().arm(Interlocks(**{**OK.__dict__, "battery_percent": None}))[0]
    assert not gate().arm(Interlocks(**{**OK.__dict__, "battery_percent": 20.0}))[0]
    assert gate().arm(OK) == (True, "armed")


def test_stage_a_never_arms():
    ok, msg = gate(allow_motion=False).arm(OK)
    assert not ok and "stage" in msg


def test_clamp_limits_and_lateral_rule():
    g = gate(vx_forward=0.6, vx_backward=0.2, vy=0.3, wz=0.8)
    g.arm(OK)
    (vx, vy, wz), st = g.evaluate((2.0, -2.0, 5.0), 0.0)
    assert (vx, vy, wz) == (0.6, 0.0, 0.8) and st == "clamped"  # vy removed because vx 0.6 > 0.5
    (vx, vy, wz), _ = g.evaluate((-2.0, 0.1, 0.0), 0.0)
    assert (vx, vy) == (-0.2, 0.1)
    (vx, vy, wz), st = g.evaluate((0.3, 0.1, 0.1), 0.0)
    assert (vx, vy, wz) == (0.3, 0.1, 0.1) and st == "ok"


def test_non_finite_becomes_zero():
    g = gate()
    g.arm(OK)
    out, _ = g.evaluate((math.nan, math.inf, -math.inf), 0.0)
    assert out == (0.0, 0.0, 0.0)
    (vx, vy, wz), flags = g.clamp(math.nan, 0.0, 0.0)
    assert vx == 0.0 and "non_finite" in flags


def test_stale_command_gives_zero():
    g = gate(cmd_timeout_s=0.4)
    g.arm(OK)
    assert g.evaluate((0.2, 0.0, 0.0), 0.39)[1] == "ok"
    assert g.evaluate((0.2, 0.0, 0.0), 0.41) == ((0.0, 0.0, 0.0), "stale")


@pytest.mark.parametrize("field,bad", [("estop", True), ("telemetry_ok", False), ("robot_ready", False),
                                        ("critical_error", True)])
def test_interlock_failure_disarms_and_latches(field, bad):
    g = gate()
    g.arm(OK)
    reason = g.check_interlocks(Interlocks(**{**OK.__dict__, field: bad}))
    assert reason and not g.armed
    assert g.evaluate((0.2, 0, 0), 0.0)[1] == "disarmed"
    # recovery of the interlock does not re-arm
    assert g.check_interlocks(OK) is None and not g.armed
    assert g.last_disarm_reason == reason


def test_battery_disarm_threshold():
    g = gate()
    g.arm(OK)
    assert g.check_interlocks(Interlocks(**{**OK.__dict__, "battery_percent": 15.0})) is None  # between 12 and 25: keep going
    assert g.check_interlocks(Interlocks(**{**OK.__dict__, "battery_percent": 10.0}))
    assert not g.armed


def test_rate_monitor_and_levels():
    m = RateMonitor(window_s=2.0)
    for i in range(20):
        m.tick(i * 0.1)
    assert m.rate(1.95) == pytest.approx(10.0, abs=1.0)
    assert m.rate(10.0) == 0.0
    assert rate_level(10, 10) == 0 and rate_level(5, 10) == 1 and rate_level(1, 10) == 2


def test_clock_lag_levels():
    assert clock_lag_level(0.12)[0] == 0
    assert clock_lag_level(-0.5)[0] == 2
    assert clock_lag_level(3.0)[0] == 1
