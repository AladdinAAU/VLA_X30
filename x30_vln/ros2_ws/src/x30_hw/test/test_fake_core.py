"""fake_x30 core behaviour in virtual time (state machine, axes, timeouts, e-stop, telemetry content)."""

import math

import pytest

from x30_hw import protocol as P
from x30_hw.tools.fake_core import FakeConfig, FakeRobot

ADDR = ("127.0.0.1", 50000)
S = P.BasicState


class Rig:
    """Drives a FakeRobot in virtual time with a client that can send heartbeats at 5 Hz."""

    def __init__(self, **cfg):
        cfg.setdefault("time_scale", 10.0)
        self.r = FakeRobot(FakeConfig(**cfg))
        self.t = 0.0
        self.out = []
        self.hb = False
        self._next_hb = 0.0

    def run(self, seconds, hb=None):
        hb = self.hb if hb is None else hb
        end = self.t + seconds
        while self.t < end - 1e-9:
            if hb and self.t >= self._next_hb:
                self.r.handle(P.simple_cmd(P.CMD_HEARTBEAT), ADDR, self.t)
                self._next_hb = self.t + 0.2
            self.out += self.r.advance(self.t)
            self.t += 0.005
        return self

    def send(self, code, value=0):
        self.r.handle(P.simple_cmd(code, value), ADDR, self.t)

    def tele(self, code):
        snap = P.TelemetrySnapshot()
        last = None
        for dest, data in self.out:
            pkt = P.parse_packet(data)
            if pkt and pkt.code == code:
                last = pkt
        assert last is not None, hex(code)
        P.decode_telemetry(last, snap)
        return snap

    def stand_to_rl(self, gait=P.CMD_GAIT_MOUNTAIN):
        self.hb = True
        self.run(0.5)
        self.send(P.CMD_QUERY)
        self.send(P.CMD_STAND_TOGGLE)
        self.run(1.0)
        assert self.r.state == S.INITIAL_STAND
        self.send(P.CMD_FORCE_STAND)
        self.run(0.5)
        assert self.r.state == S.FORCE_STAND
        self.send(P.CMD_MOTION_TOGGLE)
        self.run(0.5)
        assert self.r.state == S.STEPPING
        self.send(gait)
        self.run(0.6)


def test_full_stand_sequence_matches_digest():
    rig = Rig()
    rig.stand_to_rl()
    assert rig.r.state == S.RL_MODE and rig.r.gait == 33
    assert rig.r.nav_mode == 1  # firmware quirk: nav mode after stand until the manual command arrives
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    assert rig.r.nav_mode == 0
    m = rig.tele(P.RECV_MOTION).motion
    assert (m.basic_state, m.gait_state) == (16, 33)


def test_commands_without_heartbeat_are_dropped_and_flagged():
    rig = Rig()
    rig.run(0.2)
    rig.send(P.CMD_STAND_TOGGLE)
    rig.run(0.5)
    assert rig.r.state == S.SITTING
    assert any("without a live heartbeat" in v for v in rig.r.violations)


def test_toggle_during_transition_reverses():
    rig = Rig(time_scale=1.0)
    rig.hb = True
    rig.run(0.3)
    rig.send(P.CMD_STAND_TOGGLE)
    rig.run(0.5)
    assert rig.r.state == S.STANDING_UP
    rig.send(P.CMD_STAND_TOGGLE)  # duplicate send
    rig.run(0.3)
    assert rig.r.state == S.GOING_DOWN
    assert any("toggle semantics" in v for v in rig.r.violations)


def test_gait_and_motion_ignored_in_wrong_states():
    rig = Rig()
    rig.hb = True
    rig.run(0.3)
    rig.send(P.CMD_GAIT_MOUNTAIN)
    rig.send(P.CMD_FORCE_STAND)
    rig.send(P.CMD_MOTION_TOGGLE)
    rig.run(0.5)
    assert rig.r.state == S.SITTING and rig.r.stats["ignored_cmd"] == 3


def drive(rig, fwd=0, lat=0, yaw=0, seconds=2.0):
    end = rig.t + seconds
    while rig.t < end:
        for code, val in ((P.CMD_AXIS_FWD, fwd), (P.CMD_AXIS_LAT, lat), (P.CMD_AXIS_YAW, yaw)):
            rig.send(code, val)
        rig.run(0.02)


def test_axis_to_velocity_uses_api_table_and_signs():
    rig = Rig()
    rig.stand_to_rl()
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    axes = P.twist_to_axes(0.4, 0.2, 0.3, P.API_APPENDIX_B[33])
    drive(rig, *axes, seconds=3.0)
    vx, vy, wz = rig.r.v
    assert vx == pytest.approx(0.4, abs=0.01) and vy == pytest.approx(0.2, abs=0.01) and wz == pytest.approx(0.3, abs=0.01)
    assert vy > 0  # +vy is LEFT in ROS and in the robot's body frame


def test_truth_table_switch_exposes_nav_bridge_scaling_error():
    """If the robot really uses the dogx30 RL values (1.5 m/s) the same axes give 1.5x the speed."""
    rig = Rig(truth_table="dx")
    rig.stand_to_rl()
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    drive(rig, *P.twist_to_axes(0.4, 0, 0, P.API_APPENDIX_B[33]), seconds=3.0)
    assert rig.r.v[0] == pytest.approx(0.6, abs=0.02)


def test_dead_zone_and_axis_timeout():
    rig = Rig()
    rig.stand_to_rl()
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    drive(rig, fwd=600, seconds=2.0)  # below 655: no motion
    assert abs(rig.r.v[0]) < 1e-6
    drive(rig, fwd=16384, seconds=2.0)
    assert rig.r.v[0] > 0.4
    rig.run(0.5)  # axis silence < 1 s: still valid
    assert rig.r.v[0] > 0.3
    rig.run(2.0)  # > 1 s without axis packets: invalid -> decelerates to zero
    assert abs(rig.r.v[0]) < 0.01


def test_axes_ignored_in_nav_mode_and_in_standing_states():
    rig = Rig()
    rig.stand_to_rl()
    assert rig.r.nav_mode == 1
    drive(rig, fwd=20000, seconds=2.0)
    assert abs(rig.r.v[0]) < 1e-6


def test_axis_during_standing_up_is_flagged():
    rig = Rig(time_scale=1.0)
    rig.hb = True
    rig.run(0.3)
    rig.send(P.CMD_STAND_TOGGLE)
    rig.run(0.3)
    rig.send(P.CMD_AXIS_FWD, 0)
    assert any("axis packet" in v for v in rig.r.violations)


def test_soft_estop_and_recovery():
    rig = Rig()
    rig.stand_to_rl()
    rig.send(P.CMD_SOFT_ESTOP)
    rig.run(0.2)
    assert rig.r.state == S.SOFT_ESTOP and rig.r.emergency_source == 5
    assert rig.tele(P.RECV_RCS).rcs.emergency_source == 5
    rig.send(P.CMD_FORCE_STAND)
    rig.send(P.CMD_STAND_TOGGLE)  # recovery goes back to sitting
    rig.run(1.0)
    assert rig.r.state == S.SITTING and rig.r.emergency_source == 0


def test_estop_zero_report_quirk_keeps_lock():
    rig = Rig(estop_reports_zero_after_s=0.2)
    rig.stand_to_rl()
    rig.send(P.CMD_SOFT_ESTOP)
    rig.run(1.0)
    assert rig.r.state == S.SITTING and rig.r.estop_lock  # dx: basic_state 0 but joints locked
    rig.send(P.CMD_STAND_TOGGLE)
    rig.run(1.0)
    assert not rig.r.estop_lock


def test_heartbeat_loss_sets_wifi_error_and_eventually_sits_down():
    rig = Rig(time_scale=1.0)
    rig.hb = True
    rig.r.state = S.RL_MODE
    rig.r.gait = 33
    rig.run(1.0)
    assert not rig.r.error_state() & P.ERR_WIFI
    rig.run(1.5, hb=False)
    assert rig.r.error_state() & P.ERR_WIFI
    assert rig.tele(P.RECV_RCS).rcs.error_state & P.ERR_WIFI
    rig.run(12.0, hb=False)
    assert rig.r.state in (S.GOING_DOWN, S.SITTING)


def test_telemetry_registration_and_mute():
    rig = Rig(registered=[("127.0.0.1", 43897)])
    rig.run(1.2)
    assert all(d == ("127.0.0.1", 43897) for d, _ in rig.out)
    n = len(rig.out)
    rig.r.control({"cmd": "mute_telemetry", "seconds": 1.0})
    rig.run(0.9)
    assert len(rig.out) == n
    rig.run(0.5)
    assert len(rig.out) > n
    none = Rig(registered=[])
    none.run(1.0)
    assert none.out == []


def test_telemetry_rates_and_battery_error_bit():
    rig = Rig()
    rig.hb = True
    rig.run(4.0)
    codes = [P.parse_packet(d).code for _, d in rig.out]
    assert 790 <= codes.count(P.RECV_MOTION) <= 810  # 200 Hz
    assert 3 <= codes.count(P.RECV_SAFE) <= 5  # 1 Hz
    assert 1 <= codes.count(P.RECV_BATTERY) <= 3  # ~0.5 Hz
    assert codes.count(P.RECV_BODY_HEIGHT) == 1
    rig.r.control({"cmd": "battery", "percent": 10})
    rig.run(0.1)
    assert rig.tele(P.RECV_RCS).rcs.error_state & P.ERR_BATTERY_LOW


def test_step_response_has_latency_and_lag():
    rig = Rig(velocity_latency_s=0.1, lag_tau_s=0.25)
    rig.stand_to_rl()
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    t0 = rig.t
    axes = P.twist_to_axes(0.4, 0, 0, P.API_APPENDIX_B[33])
    samples = []
    while rig.t < t0 + 2.0:
        rig.send(P.CMD_AXIS_FWD, axes[0])
        rig.run(0.01)
        samples.append((rig.t - t0, rig.r.v[0]))
    first_move = next(t for t, v in samples if v > 0.01)
    assert 0.1 <= first_move <= 0.2
    t63 = next(t for t, v in samples if v >= 0.632 * 0.4)
    assert t63 - 0.1 == pytest.approx(0.25, abs=0.08)


def test_leg_odometry_scale_and_yaw():
    rig = Rig(odom_scale_error=0.9, yaw_drift_rad_s=0.01)
    rig.stand_to_rl()
    rig.send(P.CMD_MANUAL)
    rig.run(0.3)
    drive(rig, *P.twist_to_axes(0.4, 0, 0, P.API_APPENDIX_B[33]), seconds=5.0)
    true_d = math.hypot(*rig.r.true_pos[:2])
    rep_d = math.hypot(*rig.r.rep_pos[:2])
    assert rep_d / true_d == pytest.approx(0.9, abs=0.03)
    assert rig.r.rep_pos[2] > 0.04  # drifting yaw


def test_handheld_conflict_blocks_client():
    rig = Rig()
    rig.hb = True
    rig.run(0.3)
    rig.r.control({"cmd": "handheld", "seconds": 2.0})
    rig.send(P.CMD_STAND_TOGGLE)
    rig.run(0.5)
    assert rig.r.state == S.SITTING and rig.r.stats["ignored_handheld_active"] >= 1


def test_crawl_halves_walk_forward_limit():
    from x30_hw.tools.fake_core import truth_limits
    assert truth_limits("api", 0, True)[0] == pytest.approx(0.6)
    assert truth_limits("api", 33, True)[0] == 1.0
