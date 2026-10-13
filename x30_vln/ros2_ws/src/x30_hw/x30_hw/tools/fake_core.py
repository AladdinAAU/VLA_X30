"""Time-driven model of the X30 motion host (no sockets, no ROS, virtual time).

`FakeRobot.handle(datagram, addr, now)` consumes client commands, `FakeRobot.advance(now)` runs the state machine, the
velocity dynamics and returns the telemetry datagrams that are due. The UDP server wrapper is fake_x30.py.

What is modelled (all from docs/robot/x30-interfaces.md):
  * heartbeat session (>= 2 Hz), connection loss with a degrade chain (stop, keep still, sit down)       [3.2, 5]
  * basic state machine 0 -> 1 -> 2 -> 3 -> 4 -> 16, toggle semantics (a duplicate toggle reverses)    [3.3, 3.5]
  * axis packets: dead zone 655, 1 s validity, per-gait full-scale speed (api Appendix B)                [3.4, 3.6]
  * soft e-stop (state 6, emergency_source 5) and recovery by the stand toggle                           [3.2, 3.7]
  * manual / non-manual mode (axes are ignored in non-manual mode)                                       [3.2]
  * telemetry 0x1008 / 0x1009 / 0x100A at 200 Hz, 0x100B at 1 Hz, battery at 0.5 Hz, body height on change [3.7]
  * velocity dynamics: pure delay, first-order lag, acceleration limit, noise; leg odometry with scale/yaw drift
Everything that is not documented (durations, latency, degrade timings, dynamics) is a configuration value flagged as
"invented" below; the numbers are plausible placeholders, NOT measurements of the real robot.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from x30_hw import protocol as P

Addr = Tuple[str, int]


@dataclass
class FakeConfig:
    registered: List[Addr] = field(default_factory=lambda: [("127.0.0.1", P.TELEMETRY_PORT)])  # network.toml ips/ports
    register_sender: bool = False  # also register the sender of heartbeats (convenience for container tests)
    telemetry_port: int = P.TELEMETRY_PORT
    telemetry_hz: float = 200.0
    telemetry_requires_heartbeat: bool = False
    # link supervision (invented: real values live in overtime.toml on the motion host, not documented)
    heartbeat_timeout_s: float = 1.0
    degrade_stop_s: float = 1.0
    degrade_still_s: float = 3.0
    degrade_sitdown_s: float = 10.0
    axis_timeout_s: float = 1.0  # documented (api p.11-12)
    # state durations in seconds (invented; nav_bridge timeouts suggest ~10-16 s for the full stand-up)
    t_stand_up: float = 3.0
    t_force_stand: float = 0.5
    t_start_motion: float = 0.5
    t_rl_switch: float = 1.5
    t_sit_down: float = 3.0
    t_estop_recover: float = 0.5
    cmd_latency_s: float = 0.02
    time_scale: float = 1.0  # >1 makes state transitions faster (test speed-up), dynamics are unaffected
    # velocity dynamics (invented placeholders; the point of measure_step_response.py is to replace them)
    velocity_latency_s: float = 0.10
    lag_tau_s: float = 0.25
    max_accel: float = 1.0
    max_ang_accel: float = 2.0
    velocity_noise: float = 0.0
    odom_scale_error: float = 1.0  # reported distance = true distance * this
    yaw_drift_rad_s: float = 0.0
    truth_table: str = "api"  # which table the simulated robot really uses: api | dx | nb
    start_nav_mode_after_stand: bool = True  # firmware quirk assumed from nav_bridge ensureManualMode
    accept_run_gait: bool = False
    rl_to_mpc_via_force_stand: bool = False
    estop_reports_zero_after_s: Optional[float] = None  # dx field note: basic_state reverts to 0 while joints stay locked
    battery_level: int = 80
    battery_low_percent: int = 20
    robot_name: str = "FAKEX30"
    seed: int = 1


def truth_limits(table: str, gait: int, crawl: bool) -> Optional[P.Limits]:
    if table == "api":
        lim = P.API_APPENDIX_B.get(gait) or P.DOGX30.get(gait)
    elif table == "dx":
        lim = P.DOGX30.get(gait)
    elif table == "nb":
        lim = P.NAV_BRIDGE_UPSTREAM.get(gait)
    else:
        raise ValueError(f"unknown truth table {table}")
    if lim is None:
        return None
    if crawl and gait in (0, 2):
        lim = (lim[0] * 0.5, lim[1], lim[2], lim[3])
    return lim


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class FakeRobot:
    def __init__(self, cfg: Optional[FakeConfig] = None):
        self.cfg = cfg or FakeConfig()
        self.rng = random.Random(self.cfg.seed)
        self.registered: List[Addr] = list(self.cfg.registered)
        self.dt = 1.0 / self.cfg.telemetry_hz
        self.now = 0.0
        self._next_tick: Optional[float] = None
        self._events: List[Tuple[float, int, int, Callable[[], None]]] = []
        self._seq = 0
        self._trans_id = 0
        # protocol state
        self.state = P.BasicState.SITTING
        self.gait = 0
        self.crawl = False
        self.nav_mode = 0
        self.emergency_source = 0
        self.estop_lock = False  # joints locked (survives basic_state reverting to 0)
        self.extra_error_bits = 0
        self.battery_level = self.cfg.battery_level
        self.last_hb: Optional[float] = None
        self._hb_times: Deque[float] = deque()
        self._last_flood_report = -1e9
        self.client_addr: Optional[Addr] = None
        self.handheld_until = -1.0
        self.telemetry_mute_until = -1.0
        self.telemetry_muted_codes: set = set()
        self._first_rl_done = False
        # axes
        self.axis = {P.CMD_AXIS_FWD: 0, P.CMD_AXIS_LAT: 0, P.CMD_AXIS_YAW: 0}
        self.axis_time = {P.CMD_AXIS_FWD: -1e9, P.CMD_AXIS_LAT: -1e9, P.CMD_AXIS_YAW: -1e9}
        # dynamics
        self._target_hist: Deque[Tuple[float, Tuple[float, float, float]]] = deque()
        self.v = [0.0, 0.0, 0.0]  # true body velocity vx, vy, wz
        self.true_pos = [0.0, 0.0, 0.0]
        self.rep_pos = [0.0, 0.0, 0.0]
        self.distance_cm = 0.0
        self.t_start: Optional[float] = None
        # slow telemetry schedule
        self._next_safe = 0.0
        self._next_batt = 0.0
        self._body_height_dirty = True
        # accounting
        self.stats: Dict[str, int] = {}
        self.violations: List[str] = []
        self.log: List[Tuple[float, str]] = []

    # ------------------------------------------------------------------ helpers
    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def _violate(self, msg: str) -> None:
        self._count("violation")
        if len(self.violations) < 200:
            self.violations.append(f"{self.now:.3f}s {msg}")

    def _note(self, msg: str) -> None:
        if len(self.log) < 2000:
            self.log.append((self.now, msg))

    def _dur(self, seconds: float) -> float:
        return seconds / max(self.cfg.time_scale, 1e-6)

    def _schedule(self, delay: float, fn: Callable[[], None], cancellable: bool = True) -> None:
        self._seq += 1
        tid = self._trans_id if cancellable else -1
        heapq.heappush(self._events, (self.now + delay, self._seq, tid, fn))

    def _cancel_transitions(self) -> None:
        self._trans_id += 1

    def session_active(self, now: Optional[float] = None) -> bool:
        now = self.now if now is None else now
        return self.last_hb is not None and (now - self.last_hb) <= self.cfg.heartbeat_timeout_s

    def hb_age(self) -> float:
        return float("inf") if self.last_hb is None else self.now - self.last_hb

    # ------------------------------------------------------------------ state transitions
    def _enter(self, state: P.BasicState) -> None:
        if state != self.state:
            self._note(f"state {self.state.name} -> {state.name}")
        self.state = state
        if state == P.BasicState.RL_MODE and not self._first_rl_done:
            self._first_rl_done = True
            if self.cfg.start_nav_mode_after_stand:
                self.nav_mode = 1
        if state in (P.BasicState.SITTING, P.BasicState.GOING_DOWN, P.BasicState.STANDING_UP):
            self.gait = 0 if state == P.BasicState.SITTING else self.gait

    def _begin(self, via: P.BasicState, delay: float, then: P.BasicState, gait: Optional[int] = None) -> None:
        self._cancel_transitions()
        self._enter(via)

        def done() -> None:
            self._enter(then)
            if gait is not None:
                self.gait = gait

        self._schedule(self._dur(delay), done)

    def _apply_command(self, code: int, value: int) -> None:
        S = P.BasicState
        st = self.state
        if self.estop_lock and code not in (P.CMD_STAND_TOGGLE, P.CMD_SOFT_ESTOP):
            self._count("ignored_locked")
            return
        if code == P.CMD_SOFT_ESTOP:
            self._cancel_transitions()
            self.emergency_source = 5
            self.estop_lock = True
            self._enter(S.SOFT_ESTOP)
            if self.cfg.estop_reports_zero_after_s is not None:
                self._schedule(self.cfg.estop_reports_zero_after_s, lambda: self._enter(S.SITTING), cancellable=False)
            return
        if code == P.CMD_STAND_TOGGLE:
            if st == S.SITTING and not self.estop_lock:
                self._begin(S.STANDING_UP, self.cfg.t_stand_up, S.INITIAL_STAND)
            elif st == S.STANDING_UP:
                self._violate("stand toggle during Sit-to-Stand: reverses the action (toggle semantics)")
                self._begin(S.GOING_DOWN, self.cfg.t_sit_down / 2, S.SITTING)
            elif st == S.GOING_DOWN:
                self._violate("stand toggle during Stand-to-Sit: reverses the action (toggle semantics)")
                self._begin(S.STANDING_UP, self.cfg.t_stand_up / 2, S.INITIAL_STAND)
            elif st in (S.INITIAL_STAND, S.FORCE_STAND):
                self._begin(S.GOING_DOWN, self.cfg.t_sit_down, S.SITTING)
            elif st == S.RL_MODE:
                self._begin(S.RL_MODE, 0.0, S.INITIAL_STAND, gait=0)
            elif st == S.SOFT_ESTOP or self.estop_lock:
                self._cancel_transitions()
                self.emergency_source = 0
                self.estop_lock = False
                self._begin(S.SOFT_ESTOP if st == S.SOFT_ESTOP else st, self.cfg.t_estop_recover, S.SITTING)
            else:
                self._count("ignored_cmd")
            return
        if code == P.CMD_FORCE_STAND:
            if st == S.INITIAL_STAND:
                self._begin(S.INITIAL_STAND, self.cfg.t_force_stand, S.FORCE_STAND)
            else:
                self._count("ignored_cmd")
            return
        if code == P.CMD_MOTION_TOGGLE:
            if st == S.FORCE_STAND:
                self._begin(S.FORCE_STAND, self.cfg.t_start_motion, S.STEPPING, gait=0)
            elif st == S.STEPPING:
                self._begin(S.STEPPING, self.cfg.t_start_motion, S.FORCE_STAND)
            else:
                self._count("ignored_cmd")
            return
        if code in P.GAIT_CODE_TO_STATE:
            g = P.GAIT_CODE_TO_STATE[code]
            if code == P.CMD_GAIT_RUN and not self.cfg.accept_run_gait:
                self._count("ignored_cmd")
                return
            if st not in (S.STEPPING, S.RL_MODE):
                self._count("ignored_cmd")  # gait switching is only accepted while stepping / in RL mode
                return
            if g in P.RL_GAITS:
                self._cancel_transitions()
                self.gait = g
                self._schedule(self._dur(self.cfg.t_rl_switch), lambda: self._enter(S.RL_MODE))
            else:
                if st == S.RL_MODE:
                    if self.cfg.rl_to_mpc_via_force_stand:
                        self._begin(S.FORCE_STAND, self.cfg.t_rl_switch, S.FORCE_STAND, gait=0)
                        self.gait = g
                    else:
                        self._begin(S.RL_MODE, self.cfg.t_rl_switch, S.STEPPING, gait=g)
                else:
                    self.gait = g
            return
        if code == P.CMD_MANUAL:
            self._schedule(0.1, lambda: setattr(self, "nav_mode", 0), cancellable=False)
            return
        if code == P.CMD_NON_MANUAL:
            self._schedule(0.1, lambda: setattr(self, "nav_mode", 1), cancellable=False)
            return
        if code == P.CMD_BODY_HEIGHT:
            if value == 0 and self.gait in (0, 2) and st in (S.STEPPING, S.INITIAL_STAND, S.FORCE_STAND):
                self.crawl = True
                self._body_height_dirty = True
            elif value == 2 and st not in (S.RL_MODE,):
                self.crawl = False
                self._body_height_dirty = True
            else:
                self._count("ignored_cmd")
            return
        self._count("unknown_cmd")

    # ------------------------------------------------------------------ inbound
    def handle(self, data: bytes, addr: Addr, now: float) -> None:
        self.now = now
        pkt = P.parse_packet(data)
        if pkt is None or pkt.type != 0:
            self._count("malformed")
            return
        code = pkt.code
        if code != P.CMD_HEARTBEAT and code not in self.axis:
            self._note(f"rx 0x{code:08X} value={pkt.value} from {addr[0]}:{addr[1]}")
        if self.handheld_until > now:
            # dx/sys: the 2.4G handheld and a second heartbeat source cancel each other; the client is ignored
            self._count("ignored_handheld_active")
            if code == P.CMD_HEARTBEAT:
                self._violate("second 0x21 source: heartbeat while the handheld controller is active")
            return
        if code == P.CMD_HEARTBEAT:
            self._count("heartbeat")
            self._hb_times.append(now)
            while self._hb_times and now - self._hb_times[0] > 0.1:
                self._hb_times.popleft()
            if len(self._hb_times) > 20 and now - self._last_flood_report > 5.0:
                self._last_flood_report = now
                self._violate(f"heartbeat flood: {len(self._hb_times)} heartbeats within 100 ms (spec: >= 2 Hz)")
            self.last_hb = now
            self.client_addr = addr
            if self.cfg.register_sender and all(r[0] != addr[0] for r in self.registered):
                self.registered.append((addr[0], self.cfg.telemetry_port))
            return
        if not self.session_active(now):
            self._violate(f"command 0x{code:08X} without a live heartbeat (dropped)")
            return
        if code == P.CMD_QUERY:
            self._count("query")
            return
        if code in self.axis:
            self._count("axis_packets")
            if self.state in (P.BasicState.STANDING_UP, P.BasicState.GOING_DOWN, P.BasicState.SOFT_ESTOP):
                self._violate(f"axis packet 0x{code:08X}={pkt.value} during {self.state.name} "
                              "(hard motion risk, dx rule)")
            self.axis[code] = max(-P.AXIS_MAX, min(P.AXIS_MAX, pkt.value))
            self.axis_time[code] = now
            return
        if code == P.CMD_AXIS_PITCH:
            return
        self._count("cmd_" + hex(code))
        self._schedule(self.cfg.cmd_latency_s, lambda c=code, v=pkt.value: self._apply_command(c, v), cancellable=False)

    # ------------------------------------------------------------------ dynamics
    def _target_velocity(self, t: float) -> Tuple[float, float, float]:
        S = P.BasicState
        if self.state not in (S.STEPPING, S.RL_MODE) or self.nav_mode or not self.session_active(t):
            return (0.0, 0.0, 0.0)
        valid = [self.axis[c] if (t - self.axis_time[c]) <= self.cfg.axis_timeout_s else 0
                 for c in (P.CMD_AXIS_FWD, P.CMD_AXIS_LAT, P.CMD_AXIS_YAW)]
        lim = truth_limits(self.cfg.truth_table, self.gait, self.crawl)
        if lim is None:
            return (0.0, 0.0, 0.0)
        return P.axes_to_twist(valid[0], valid[1], valid[2], lim)

    def _physics(self, t: float) -> None:
        dt = self.dt
        tgt = self._target_velocity(t)
        self._target_hist.append((t, tgt))
        cut = t - self.cfg.velocity_latency_s
        delayed = (0.0, 0.0, 0.0)
        while len(self._target_hist) > 1 and self._target_hist[1][0] <= cut:
            self._target_hist.popleft()
        if self._target_hist and self._target_hist[0][0] <= cut:
            delayed = self._target_hist[0][1]
        tau = max(self.cfg.lag_tau_s, 1e-3)
        for i in range(3):
            dv = (delayed[i] - self.v[i]) * min(1.0, dt / tau)
            amax = (self.cfg.max_ang_accel if i == 2 else self.cfg.max_accel) * dt
            self.v[i] += max(-amax, min(amax, dv))
        vx, vy, wz = self.v
        yaw = self.true_pos[2]
        self.true_pos[0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        self.true_pos[1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        self.true_pos[2] = _wrap(yaw + wz * dt)
        s = self.cfg.odom_scale_error
        ryaw = self.rep_pos[2]
        self.rep_pos[0] += s * (vx * math.cos(ryaw) - vy * math.sin(ryaw)) * dt
        self.rep_pos[1] += s * (vx * math.sin(ryaw) + vy * math.cos(ryaw)) * dt
        self.rep_pos[2] = _wrap(ryaw + wz * dt + self.cfg.yaw_drift_rad_s * dt)
        self.distance_cm += s * math.hypot(vx, vy) * dt * 100.0

    def _supervise(self, t: float) -> None:
        if self.last_hb is None:
            return
        age = t - self.last_hb
        if age > self.cfg.heartbeat_timeout_s + self.cfg.degrade_sitdown_s and self.state in (
                P.BasicState.INITIAL_STAND, P.BasicState.FORCE_STAND, P.BasicState.STEPPING, P.BasicState.RL_MODE):
            self._note("heartbeat lost: degrade chain sits the robot down")
            self._count("degrade_sitdown")
            self._begin(P.BasicState.GOING_DOWN, self.cfg.t_sit_down, P.BasicState.SITTING)

    # ------------------------------------------------------------------ advance + telemetry
    def error_state(self) -> int:
        bits = self.extra_error_bits
        if self.last_hb is not None and self.hb_age() > self.cfg.heartbeat_timeout_s:
            bits |= P.ERR_WIFI
        if self.battery_level < self.cfg.battery_low_percent:
            bits |= P.ERR_BATTERY_LOW
        return bits

    def _telemetry(self, t: float) -> List[Tuple[Addr, bytes]]:
        if t < self.telemetry_mute_until:
            return []
        if self.cfg.telemetry_requires_heartbeat and not self.session_active(t):
            return []
        out: List[Tuple[Addr, bytes]] = []
        if self.t_start is None:
            self.t_start = t
        el = t - self.t_start
        noise = self.cfg.velocity_noise
        nv = [x + (self.rng.gauss(0.0, noise) if noise else 0.0) for x in self.v]
        reported_state = int(self.state)
        pkts: List[Tuple[int, bytes]] = []
        joy = (self.axis[P.CMD_AXIS_LAT] / P.AXIS_MAX, self.axis[P.CMD_AXIS_FWD] / P.AXIS_MAX,
               self.axis[P.CMD_AXIS_YAW] / P.AXIS_MAX, 0.0)
        pkts.append((P.RECV_RCS, P.RcsData(
            self.cfg.robot_name, int(self.distance_cm), int(self.distance_cm), int(el), int(el), int(el), int(el), joy,
            self.nav_mode, self.emergency_source, self.error_state()).pack()))
        pkts.append((P.RECV_MOTION, P.MotionStateData(
            reported_state, self.gait, 0.0, 0.0, tuple(self.rep_pos), tuple(nv), float(self.distance_cm),
            0, 0, (0, 0 if self.state != P.BasicState.STEPPING else 2) + (0,) * 8).pack()))
        g = 9.81
        pkts.append((P.RECV_SENSOR, P.SensorData(
            int(el * 1000) & 0x7FFFFFFF, (0.0, 0.0, math.degrees(self.rep_pos[2])), (0.0, 0.0, self.v[2]),
            (0.0, 0.0, g)).pack()))
        if t >= self._next_safe:
            self._next_safe = t + 1.0
            pkts.append((P.RECV_SAFE, P.SafeData().pack()))
        if t >= self._next_batt:
            self._next_batt = t + 2.0
            pkts.append((P.RECV_BATTERY, P.BatteryData(battery_level=self.battery_level).pack()))
        if self._body_height_dirty:
            self._body_height_dirty = False
            pkts.append((P.RECV_BODY_HEIGHT, P.body_height_msg(-1 if self.crawl else 0)))
        for code, data in pkts:
            if code in self.telemetry_muted_codes:
                continue
            for dest in self.registered:
                out.append((dest, data))
        return out

    def advance(self, now: float) -> List[Tuple[Addr, bytes]]:
        out: List[Tuple[Addr, bytes]] = []
        if self._next_tick is None:
            self._next_tick = now
        while self._next_tick <= now:
            t = self._next_tick
            self.now = t
            while self._events and self._events[0][0] <= t:
                _, _, tid, fn = heapq.heappop(self._events)
                if tid == -1 or tid == self._trans_id:
                    fn()
            self._supervise(t)
            self._physics(t)
            out.extend(self._telemetry(t))
            self._next_tick += self.dt
        self.now = now
        return out

    @property
    def next_tick(self) -> Optional[float]:
        return self._next_tick

    # ------------------------------------------------------------------ fault injection / introspection
    def control(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            return self.status()
        if cmd == "mute_telemetry":
            self.telemetry_mute_until = self.now + float(req.get("seconds", 1.0))
        elif cmd == "mute_code":
            self.telemetry_muted_codes.add(int(req["code"], 0) if isinstance(req["code"], str) else int(req["code"]))
        elif cmd == "unmute_codes":
            self.telemetry_muted_codes.clear()
        elif cmd == "hardware_estop":
            self._cancel_transitions()
            self.emergency_source = 1
            self.estop_lock = True
            self._enter(P.BasicState.SOFT_ESTOP)
        elif cmd == "battery":
            self.battery_level = int(req["percent"])
        elif cmd == "error_bits":
            self.extra_error_bits = int(req["value"])
        elif cmd == "handheld":
            self.handheld_until = self.now + float(req.get("seconds", 5.0))
        elif cmd == "set":
            key = req["key"]
            if not hasattr(self.cfg, key):
                return {"ok": False, "error": f"unknown config key {key}"}
            setattr(self.cfg, key, req["value"])
        elif cmd == "reset_stats":
            self.stats.clear()
            self.violations.clear()
        else:
            return {"ok": False, "error": f"unknown cmd {cmd}"}
        return {"ok": True}

    def status(self) -> dict:
        return {
            "ok": True, "now": self.now, "state": int(self.state), "state_name": self.state.name, "gait": self.gait,
            "nav_mode": self.nav_mode, "emergency_source": self.emergency_source, "crawl": self.crawl,
            "error_state": self.error_state(), "session_active": self.session_active(), "v": list(self.v),
            "true_pos": list(self.true_pos), "rep_pos": list(self.rep_pos), "axes": dict(
                (hex(k), v) for k, v in self.axis.items()),
            "stats": dict(self.stats), "violations": list(self.violations[-20:]),
            "n_violations": self.stats.get("violation", 0), "log": [f"{t:.2f} {m}" for t, m in self.log[-40:]], "registered": [list(r) for r in self.registered],
        }
