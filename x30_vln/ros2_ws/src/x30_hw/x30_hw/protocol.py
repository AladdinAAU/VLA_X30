"""X30 motion-host UDP protocol: framing, command codes, telemetry layouts, gait speed tables.

Pure Python (no ROS). Every number is taken from docs/robot/x30-interfaces.md (section numbers in comments),
which in turn cites the vendor API V1.0.5 (`api`), the nav_bridge sources (`nb`) and the dogx30 field notes (`dx`).
All multi-byte values are little-endian; telemetry structs use natural LP64 alignment (NOT packed), reproduced here
with explicit padding bytes in the struct format strings.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------------------------------
# Framing (digest 3.1)
# ---------------------------------------------------------------------------------------------------------------------
HEAD = struct.Struct("<III")  # code, paramters_size (value for type 0, payload bytes for type 1), type
HEAD_SIZE = HEAD.size  # 12

MOTION_HOST_PORT = 43893
TELEMETRY_PORT = 43897

# ---------------------------------------------------------------------------------------------------------------------
# Command codes (digest 3.2, prefix 0x21 = remote control terminal)
# ---------------------------------------------------------------------------------------------------------------------
CMD_HEARTBEAT = 0x21040001
CMD_QUERY = 0x21020001
CMD_STAND_TOGGLE = 0x21010202
CMD_FORCE_STAND = 0x2101020A
CMD_MOTION_TOGGLE = 0x21010201
CMD_AXIS_FWD = 0x21010130  # left stick Y: body x velocity
CMD_AXIS_LAT = 0x21010131  # left stick X: body y velocity, + = RIGHT
CMD_AXIS_YAW = 0x21010135  # right stick X: yaw rate, + = turn RIGHT
CMD_AXIS_PITCH = 0x21010102  # right stick Y: unused while stepping
AXIS_CODES = (CMD_AXIS_FWD, CMD_AXIS_LAT, CMD_AXIS_YAW, CMD_AXIS_PITCH)
CMD_NON_MANUAL = 0x21010C03
CMD_MANUAL = 0x21010C02
CMD_BODY_HEIGHT = 0x21010406  # 0 = crawl, 2 = normal
CMD_SOFT_ESTOP = 0x21010C0E

CMD_GAIT_WALK = 0x21010300
CMD_GAIT_SLOPE = 0x21010402
CMD_GAIT_OFFROAD = 0x21010401
CMD_GAIT_STAIR = 0x21010405
CMD_GAIT_STAIR_MULTI = 0x2101040A
CMD_GAIT_STAIR45_MULTI = 0x2101040B
CMD_GAIT_L_WALK = 0x21010420
CMD_GAIT_MOUNTAIN = 0x21010421
CMD_GAIT_SILENT = 0x21010422
CMD_GAIT_L_STAIR = 0x21010424  # not in api V1.0.5 (nb / dx only)
CMD_GAIT_RUN = 0x31010307  # not in api (nb only)

GAIT_CODE_TO_STATE: Dict[int, int] = {
    CMD_GAIT_WALK: 0,
    CMD_GAIT_OFFROAD: 1,
    CMD_GAIT_SLOPE: 2,
    CMD_GAIT_RUN: 3,
    CMD_GAIT_STAIR: 6,
    CMD_GAIT_STAIR_MULTI: 7,
    CMD_GAIT_STAIR45_MULTI: 8,
    CMD_GAIT_L_WALK: 32,
    CMD_GAIT_MOUNTAIN: 33,
    CMD_GAIT_SILENT: 34,
    CMD_GAIT_L_STAIR: 36,
}
RL_GAITS = (32, 33, 34, 36)

# Telemetry codes (digest 3.7)
RECV_RCS = 0x1008
RECV_MOTION = 0x1009
RECV_SENSOR = 0x100A
RECV_SAFE = 0x100B
RECV_BATTERY = 0x21050F0A
RECV_BODY_HEIGHT = 0x11050F08  # simple message: -1 crawl, 0 normal

AXIS_MAX = 32767
AXIS_DEAD_ZONE = 655


class BasicState(enum.IntEnum):
    SITTING = 0
    STANDING_UP = 1
    INITIAL_STAND = 2
    FORCE_STAND = 3
    STEPPING = 4
    GOING_DOWN = 5
    SOFT_ESTOP = 6
    RL_MODE = 16


class GaitState(enum.IntEnum):
    WALK = 0
    OFFROAD = 1
    SLOPE = 2
    RUN = 3
    STAIR = 6
    STAIR_MULTI = 7
    STAIR45_MULTI = 8
    L_WALK = 32
    MOUNTAIN = 33
    SILENT = 34
    L_STAIR = 36


ERROR_BITS = (
    "imu_error",  # 0 IMU timeout
    "wifi_error",  # 1 heartbeat timeout
    "driver_heat_warn",
    "driver_error",
    "motor_heat_warn",
    "battery_low_warn",
    "battery_heat_warn",
    "gpio_error",
    "cpu_heat_warn",
    "cpu_freq_warn",
)
ERR_IMU = 1 << 0
ERR_WIFI = 1 << 1
ERR_BATTERY_LOW = 1 << 5

EMERGENCY_SOURCES = {
    0: "none",
    1: "tail soft e-stop",
    2: "autonomous during motion",
    3: "autonomous during navigation",
    4: "autonomous during charging",
    5: "triggered by client",
    6: "triggered by RAS",
}
# digest 3.7 / dx: sources 2 and 3 appear during gait switches without a real lock; 1, 4, 5, 6 are real locks
EMERGENCY_LOCK_SOURCES = (1, 4, 5, 6)


# ---------------------------------------------------------------------------------------------------------------------
# Gait speed tables (digest 3.6). Tuple = (forward, backward, lateral, yaw) in m/s, m/s, m/s, rad/s, normal body height
# ---------------------------------------------------------------------------------------------------------------------
Limits = Tuple[float, float, float, float]

API_APPENDIX_B: Dict[int, Limits] = {
    0: (1.2, 0.75, 0.15, 0.45),
    2: (0.7, 0.75, 0.25, 0.5),
    1: (0.3, 0.4, 0.1, 0.5),
    6: (0.3, 0.3, 0.2, 0.8),
    7: (0.6, 0.3, 0.2, 0.8),
    8: (0.3, 0.2, 0.2, 0.8),
    32: (1.0, 1.0, 0.5, 1.2),
    33: (1.0, 1.0, 0.8, 1.2),
}
# upstream nav_bridge getGaitSpeedLimit (digest 3.6, column 'nav_bridge'): kept only for the discrepancy tests/docs
NAV_BRIDGE_UPSTREAM: Dict[int, Limits] = {
    0: (1.2, 0.7, 0.75, 0.5),
    1: (0.3, 0.3, 0.15, 0.45),
    2: (0.3, 0.3, 0.1, 0.5),
    6: (0.3, 0.3, 0.2, 0.8),
    7: (0.3, 0.3, 0.2, 0.8),
    8: (0.3, 0.3, 0.2, 0.8),
    32: (1.0, 1.0, 0.5, 1.2),
    33: (1.0, 1.0, 0.8, 1.2),
    34: (1.0, 1.0, 0.8, 1.2),
    36: (1.0, 1.0, 0.8, 1.2),
    3: (0.6, 0.25, 0.4, 0.5),
}
# dogx30 LimitsOf (fwd, lat, yaw); backward assumed = forward
DOGX30: Dict[int, Limits] = {
    0: (1.5, 1.5, 0.15, 0.45),
    2: (0.7, 0.7, 0.25, 0.5),
    1: (0.3, 0.3, 0.1, 0.5),
    6: (0.3, 0.3, 0.2, 0.8),
    7: (0.6, 0.6, 0.2, 0.8),
    8: (0.3, 0.3, 0.2, 0.8),
    32: (1.5, 1.5, 0.5, 1.2),
    33: (1.5, 1.5, 0.5, 1.2),
    34: (1.0, 1.0, 0.5, 1.0),
    36: (0.3, 0.3, 0.2, 0.8),
    3: (2.5, 2.5, 0.8, 1.2),
}
# Gaits that api App. B does not list: the table used by x30_nav_bridge takes the LARGEST known value per axis
# (a larger vmax makes the robot slower than requested, which is the safe side).
CONSERVATIVE_UNLISTED: Dict[int, Limits] = {
    34: (1.0, 1.0, 0.8, 1.2),
    36: (1.0, 1.0, 0.8, 1.2),
    3: (2.5, 0.25, 0.8, 1.2),
}
DEFAULT_NAV_BRIDGE_TABLE: Dict[int, Limits] = {**API_APPENDIX_B, **CONSERVATIVE_UNLISTED}


def velocity_to_axis(velocity: float, vmax: float) -> int:
    """Physical speed -> joystick axis count for one axis (mirror of nav_bridge velocityToAxisValue).

    The caller applies the ROS->robot sign flips (+vy left, +wz CCW need a negative axis)."""
    if vmax <= 0.0:
        return 0
    ratio = max(-1.0, min(1.0, velocity / vmax))
    raw = int(ratio * AXIS_MAX)  # truncation toward zero, like the C++ static_cast<int32_t>
    if abs(raw) < AXIS_DEAD_ZONE:
        return 0
    return raw


def twist_to_axes(vx: float, vy: float, wz: float, limits: Limits) -> Tuple[int, int, int]:
    """ROS twist (x fwd, y left, wz CCW) -> (axis_fwd, axis_lat, axis_yaw) per digest 3.4."""
    fwd, bwd, lat, yaw = limits
    return (
        velocity_to_axis(vx, fwd if vx >= 0.0 else bwd),
        velocity_to_axis(-vy, lat),
        velocity_to_axis(-wz, yaw),
    )


def axes_to_twist(axis_fwd: int, axis_lat: int, axis_yaw: int, limits: Limits) -> Tuple[float, float, float]:
    """Inverse used by the robot (and the fake): axis counts -> body velocity (x fwd, y left, wz CCW)."""
    fwd, bwd, lat, yaw = limits

    def dz(v: int) -> int:
        return 0 if abs(v) < AXIS_DEAD_ZONE else v

    f, la, ya = dz(axis_fwd), dz(axis_lat), dz(axis_yaw)
    vx = f / AXIS_MAX * (fwd if f >= 0 else bwd)
    vy = -la / AXIS_MAX * lat
    wz = -ya / AXIS_MAX * yaw
    return vx, vy, wz


# ---------------------------------------------------------------------------------------------------------------------
# Packing / unpacking
# ---------------------------------------------------------------------------------------------------------------------
def simple_cmd(code: int, value: int = 0) -> bytes:
    """type-0 command: the value rides in the paramters_size field as int32 two's complement."""
    return HEAD.pack(code & 0xFFFFFFFF, value & 0xFFFFFFFF, 0)


def complex_msg(code: int, payload: bytes) -> bytes:
    return HEAD.pack(code & 0xFFFFFFFF, len(payload), 1) + payload


@dataclass
class Packet:
    code: int
    type: int
    value: int  # int32 view of paramters_size (type 0)
    size: int  # paramters_size as unsigned (type 1)
    payload: bytes


def parse_packet(data: bytes) -> Optional[Packet]:
    """Parse one datagram. Returns None for short or inconsistent datagrams (type 1 must satisfy len == 12 + size)."""
    if len(data) < HEAD_SIZE:
        return None
    code, size, typ = HEAD.unpack_from(data, 0)
    if typ == 0:
        value = size - (1 << 32) if size & 0x80000000 else size
        return Packet(code, 0, value, size, data[HEAD_SIZE:])
    if typ == 1:
        if len(data) != HEAD_SIZE + size:
            return None
        return Packet(code, 1, 0, size, data[HEAD_SIZE:])
    return None


RCS = struct.Struct("<15sxiiqqqq4f10B2xI")
MOTION = struct.Struct("<BB2xff3f3ffII10B2x")
SENSOR = struct.Struct("<i9f12f12f12f")
SAFE = struct.Struct("<12f12B2f")
BATTERY = struct.Struct("<HhHHHHHHHBBBBBx4f")
assert (RCS.size, MOTION.size, SENSOR.size, SAFE.size, BATTERY.size) == (88, 60, 184, 68, 40)


@dataclass
class RcsData:
    robot_name: str = ""
    current_mileage: int = 0
    total_mileage: int = 0
    current_run_time: int = 0
    total_run_time: int = 0
    current_motion_time: int = 0
    total_motion_time: int = 0
    joystick: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    is_nav_mode: int = 0
    emergency_source: int = 0
    error_state: int = 0

    def pack(self) -> bytes:
        rcs_state = [self.is_nav_mode, self.emergency_source] + [0] * 8
        return complex_msg(
            RECV_RCS,
            RCS.pack(
                self.robot_name.encode()[:15], self.current_mileage, self.total_mileage, self.current_run_time,
                self.total_run_time, self.current_motion_time, self.total_motion_time, *self.joystick, *rcs_state,
                self.error_state,
            ),
        )

    @staticmethod
    def unpack(payload: bytes) -> "RcsData":
        f = RCS.unpack_from(payload)
        return RcsData(
            f[0].split(b"\0")[0].decode(errors="replace"), f[1], f[2], f[3], f[4], f[5], f[6], tuple(f[7:11]),
            f[11], f[12], f[21],
        )


@dataclass
class MotionStateData:
    basic_state: int = 0
    gait_state: int = 0
    max_forward_vel: float = 0.0
    max_backward_vel: float = 0.0
    leg_odom_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x m, y m, yaw rad (world, origin = start-up pose)
    leg_odom_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # body-frame vx, vy, wz
    robot_distance: float = 0.0  # cm
    touch_state: int = 0
    control_state: int = 0
    task_state: Tuple[int, ...] = (0,) * 10

    def pack(self) -> bytes:
        return complex_msg(
            RECV_MOTION,
            MOTION.pack(
                self.basic_state, self.gait_state, self.max_forward_vel, self.max_backward_vel, *self.leg_odom_pos,
                *self.leg_odom_vel, self.robot_distance, self.touch_state, self.control_state, *self.task_state,
            ),
        )

    @staticmethod
    def unpack(payload: bytes) -> "MotionStateData":
        f = MOTION.unpack_from(payload)
        return MotionStateData(
            f[0], f[1], f[2], f[3], tuple(f[4:7]), tuple(f[7:10]), f[10], f[11], f[12], tuple(f[13:23]),
        )


@dataclass
class SensorData:
    timestamp: int = 0
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    omega: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acc: Tuple[float, float, float] = (0.0, 0.0, 9.81)
    joint_pos: Tuple[float, ...] = (0.0,) * 12
    joint_vel: Tuple[float, ...] = (0.0,) * 12
    joint_tau: Tuple[float, ...] = (0.0,) * 12

    def pack(self) -> bytes:
        return complex_msg(
            RECV_SENSOR,
            SENSOR.pack(self.timestamp, *self.rpy_deg, *self.omega, *self.acc, *self.joint_pos, *self.joint_vel,
                        *self.joint_tau),
        )

    @staticmethod
    def unpack(payload: bytes) -> "SensorData":
        f = SENSOR.unpack_from(payload)
        return SensorData(f[0], tuple(f[1:4]), tuple(f[4:7]), tuple(f[7:10]), tuple(f[10:22]), tuple(f[22:34]),
                          tuple(f[34:46]))


@dataclass
class SafeData:
    motor_temperature: Tuple[float, ...] = (35.0,) * 12
    driver_temperature: Tuple[int, ...] = (30,) * 12
    cpu_temp: float = 45.0
    cpu_freq_mhz: float = 1800.0

    def pack(self) -> bytes:
        return complex_msg(RECV_SAFE, SAFE.pack(*self.motor_temperature, *self.driver_temperature, self.cpu_temp,
                                                self.cpu_freq_mhz))

    @staticmethod
    def unpack(payload: bytes) -> "SafeData":
        f = SAFE.unpack_from(payload)
        return SafeData(tuple(f[0:12]), tuple(f[12:24]), f[24], f[25])


@dataclass
class BatteryData:
    voltage: int = 72  # V
    current: int = -5  # A (before deeprcs 1.10.19) or 10 mA after; sign convention not documented
    remaining_capacity: int = 4000  # 10 mAh
    nominal_capacity: int = 5000
    cycles: int = 12
    production_date: int = 0
    balanced_low: int = 0
    balanced_high: int = 0
    protected_state: int = 0
    software_version: int = 1
    battery_level: int = 80
    mos_state: int = 0
    battery_quantity: int = 1
    battery_ntc: int = 4
    battery_temperature: Tuple[float, float, float, float] = (28.0, 28.5, 29.0, 29.5)

    def pack(self) -> bytes:
        return complex_msg(
            RECV_BATTERY,
            BATTERY.pack(
                self.voltage, self.current, self.remaining_capacity, self.nominal_capacity, self.cycles,
                self.production_date, self.balanced_low, self.balanced_high, self.protected_state,
                self.software_version, self.battery_level, self.mos_state, self.battery_quantity, self.battery_ntc,
                *self.battery_temperature,
            ),
        )

    @staticmethod
    def unpack(payload: bytes) -> "BatteryData":
        f = BATTERY.unpack_from(payload)
        return BatteryData(*f[:14], tuple(f[14:18]))


def body_height_msg(state: int) -> bytes:
    """0x11050F08 simple message: -1 crawl, 0 normal."""
    return simple_cmd(RECV_BODY_HEIGHT, state)


def decode_error_bits(error_state: int) -> List[str]:
    return [name for i, name in enumerate(ERROR_BITS) if error_state & (1 << i)]


def estop_condition(basic_state: int, emergency_source: int) -> bool:
    """Latched-safety rule used by nav_bridge: basic_state 6, or an emergency source that is a real joint lock."""
    return basic_state == BasicState.SOFT_ESTOP or emergency_source in EMERGENCY_LOCK_SOURCES


@dataclass
class TelemetrySnapshot:
    """Decoded latest values, filled by decode_telemetry (used by tools such as preflight and the recorder)."""

    rcs: Optional[RcsData] = None
    motion: Optional[MotionStateData] = None
    sensor: Optional[SensorData] = None
    safe: Optional[SafeData] = None
    battery: Optional[BatteryData] = None
    body_height: Optional[int] = None
    counts: Dict[int, int] = field(default_factory=dict)


def decode_telemetry(pkt: Packet, snap: TelemetrySnapshot) -> bool:
    """Decode one parsed telemetry packet into `snap`. Returns False for unknown codes or wrong payload sizes."""
    try:
        if pkt.code == RECV_RCS and pkt.type == 1 and pkt.size == RCS.size:
            snap.rcs = RcsData.unpack(pkt.payload)
        elif pkt.code == RECV_MOTION and pkt.type == 1 and pkt.size == MOTION.size:
            snap.motion = MotionStateData.unpack(pkt.payload)
        elif pkt.code == RECV_SENSOR and pkt.type == 1 and pkt.size == SENSOR.size:
            snap.sensor = SensorData.unpack(pkt.payload)
        elif pkt.code == RECV_SAFE and pkt.type == 1 and pkt.size == SAFE.size:
            snap.safe = SafeData.unpack(pkt.payload)
        elif pkt.code == RECV_BATTERY and pkt.type == 1 and pkt.size == BATTERY.size:
            snap.battery = BatteryData.unpack(pkt.payload)
        elif pkt.code == RECV_BODY_HEIGHT and pkt.type == 0:
            snap.body_height = pkt.value
        else:
            return False
    except struct.error:
        return False
    snap.counts[pkt.code] = snap.counts.get(pkt.code, 0) + 1
    return True
