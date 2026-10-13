"""Protocol tests against golden byte vectors.

The expected bytes are assembled field by field at the offsets stated in docs/robot/x30-interfaces.md section 3.7
(struct.pack_into on a zeroed buffer), independently of the struct format strings used by x30_hw.protocol.
"""

import struct

import pytest

from x30_hw import protocol as P


def test_simple_command_golden():
    # heartbeat 0x21040001, value 0, type 0, little endian
    assert P.simple_cmd(P.CMD_HEARTBEAT) == bytes.fromhex("01000421" "00000000" "00000000")
    # axis forward +16384 -> code 0x21010130
    assert P.simple_cmd(P.CMD_AXIS_FWD, 16384) == bytes.fromhex("30010121" "00400000" "00000000")
    # negative values are int32 two's complement in the paramters_size field
    assert P.simple_cmd(P.CMD_AXIS_LAT, -32767) == bytes.fromhex("31010121" "0180ffff" "00000000")
    assert P.simple_cmd(P.CMD_BODY_HEIGHT, 2) == bytes.fromhex("06040121" "02000000" "00000000")


def test_parse_simple_negative():
    pkt = P.parse_packet(P.simple_cmd(P.RECV_BODY_HEIGHT, -1))
    assert pkt.type == 0 and pkt.value == -1 and pkt.code == 0x11050F08


def test_parse_rejects_bad_sizes():
    assert P.parse_packet(b"\x00" * 11) is None
    bad = P.HEAD.pack(P.RECV_RCS, 88, 1) + b"\x00" * 87  # one byte short: recv_size != paramters_size + 12
    assert P.parse_packet(bad) is None
    assert P.parse_packet(P.HEAD.pack(1, 0, 7)) is None


def golden_rcs() -> bytes:
    b = bytearray(88)
    b[0:3] = b"X30"
    struct.pack_into("<i", b, 16, 1234)  # current_mileage
    struct.pack_into("<i", b, 20, 56789)  # total_mileage
    struct.pack_into("<q", b, 24, 100)
    struct.pack_into("<q", b, 32, 2000)
    struct.pack_into("<q", b, 40, 50)
    struct.pack_into("<q", b, 48, 900)
    struct.pack_into("<4f", b, 56, 0.5, -0.25, 0.0, 1.0)
    b[72] = 1  # is_nav_mode
    b[73] = 5  # emergency_source
    struct.pack_into("<I", b, 84, 0b100010)  # wifi_error + battery_low_warn
    return bytes(b)


def test_rcs_layout():
    assert P.RCS.size == 88
    payload = golden_rcs()
    msg = P.complex_msg(P.RECV_RCS, payload)
    assert msg[:12] == bytes.fromhex("08100000" "58000000" "01000000")
    d = P.RcsData.unpack(payload)
    assert (d.robot_name, d.current_mileage, d.total_mileage) == ("X30", 1234, 56789)
    assert (d.current_run_time, d.total_run_time, d.current_motion_time, d.total_motion_time) == (100, 2000, 50, 900)
    assert d.joystick == (0.5, -0.25, 0.0, 1.0)
    assert (d.is_nav_mode, d.emergency_source, d.error_state) == (1, 5, 0b100010)
    assert P.decode_error_bits(d.error_state) == ["wifi_error", "battery_low_warn"]
    assert P.RcsData(**{**d.__dict__}).pack() == msg


def golden_motion() -> bytes:
    b = bytearray(60)
    b[0] = 16
    b[1] = 33
    struct.pack_into("<2f", b, 4, 0.0, 0.0)
    struct.pack_into("<3f", b, 12, 1.5, -0.5, 0.25)
    struct.pack_into("<3f", b, 24, 0.3, -0.1, 0.2)
    struct.pack_into("<f", b, 36, 321.0)
    struct.pack_into("<I", b, 40, 0)
    struct.pack_into("<I", b, 44, 0b101)
    b[48] = 0
    b[49] = 2  # pos_ctrl_state
    return bytes(b)


def test_motion_layout():
    assert P.MOTION.size == 60
    payload = golden_motion()
    d = P.MotionStateData.unpack(payload)
    assert (d.basic_state, d.gait_state) == (16, 33)
    assert d.leg_odom_pos == (1.5, -0.5, 0.25)
    assert d.leg_odom_vel == pytest.approx((0.3, -0.1, 0.2))
    assert d.robot_distance == 321.0 and d.control_state == 5 and d.task_state[1] == 2
    assert d.pack() == P.complex_msg(P.RECV_MOTION, payload)


def golden_sensor() -> bytes:
    b = bytearray(184)
    struct.pack_into("<i", b, 0, 123456)
    struct.pack_into("<9f", b, 4, 1.0, -2.0, 90.0, 0.1, 0.2, 0.3, 0.0, 0.5, 9.8)
    for i in range(12):
        struct.pack_into("<f", b, 40 + 4 * i, 0.01 * i)  # joint_pos
        struct.pack_into("<f", b, 88 + 4 * i, 0.02 * i)  # joint_vel
        struct.pack_into("<f", b, 136 + 4 * i, 0.03 * i)  # joint_tau
    return bytes(b)


def test_sensor_layout():
    assert P.SENSOR.size == 184
    d = P.SensorData.unpack(golden_sensor())
    assert d.timestamp == 123456
    assert d.rpy_deg == (1.0, -2.0, 90.0)
    assert d.omega == pytest.approx((0.1, 0.2, 0.3))
    assert d.acc == pytest.approx((0.0, 0.5, 9.8))
    assert d.joint_pos[11] == pytest.approx(0.11) and d.joint_vel[11] == pytest.approx(0.22)
    assert d.joint_tau[11] == pytest.approx(0.33)
    assert d.pack()[12:] == golden_sensor()


def test_safe_layout():
    b = bytearray(68)
    for i in range(12):
        struct.pack_into("<f", b, 4 * i, 30.0 + i)
        b[48 + i] = 40 + i
    struct.pack_into("<f", b, 60, 55.5)
    struct.pack_into("<f", b, 64, 1992.0)
    d = P.SafeData.unpack(bytes(b))
    assert d.motor_temperature[11] == 41.0 and d.driver_temperature[11] == 51
    assert (d.cpu_temp, d.cpu_freq_mhz) == (55.5, 1992.0)
    assert P.SAFE.size == 68


def test_battery_layout_is_40_bytes_with_temperatures_at_24():
    """nav_bridge upstream declared this struct packed (39 bytes, temperatures at 23): that was the bug."""
    b = bytearray(40)
    struct.pack_into("<H", b, 0, 76)  # voltage
    struct.pack_into("<h", b, 2, -12)  # current
    struct.pack_into("<H", b, 4, 3900)
    struct.pack_into("<H", b, 6, 5000)
    struct.pack_into("<H", b, 8, 33)  # cycles
    b[18] = 7  # software_version
    b[19] = 81  # battery_level
    b[20] = 1
    b[21] = 2
    b[22] = 4
    struct.pack_into("<4f", b, 24, 25.0, 26.0, 27.0, 28.0)
    d = P.BatteryData.unpack(bytes(b))
    assert (d.voltage, d.current, d.remaining_capacity, d.nominal_capacity, d.cycles) == (76, -12, 3900, 5000, 33)
    assert (d.software_version, d.battery_level, d.mos_state, d.battery_quantity, d.battery_ntc) == (7, 81, 1, 2, 4)
    assert d.battery_temperature == (25.0, 26.0, 27.0, 28.0)
    assert P.BATTERY.size == 40
    assert d.pack()[12:] == bytes(b)


def test_decode_telemetry_dispatch():
    snap = P.TelemetrySnapshot()
    for msg in (P.RcsData(emergency_source=1).pack(), P.MotionStateData(basic_state=4).pack(),
                P.SensorData().pack(), P.SafeData().pack(), P.BatteryData().pack(), P.body_height_msg(-1)):
        assert P.decode_telemetry(P.parse_packet(msg), snap)
    assert snap.rcs.emergency_source == 1 and snap.motion.basic_state == 4 and snap.body_height == -1
    assert not P.decode_telemetry(P.parse_packet(P.complex_msg(P.RECV_RCS, b"\0" * 10)) or P.Packet(0, 1, 0, 0, b""), snap)


def test_estop_condition():
    assert P.estop_condition(6, 0)
    assert P.estop_condition(0, 1) and P.estop_condition(0, 5) and P.estop_condition(4, 6) and P.estop_condition(0, 4)
    assert not P.estop_condition(16, 2) and not P.estop_condition(16, 3)  # autonomous markers during gait switches
    assert not P.estop_condition(16, 0)
