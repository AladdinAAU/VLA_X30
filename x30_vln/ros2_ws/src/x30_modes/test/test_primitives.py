import math

import pytest

from robot_model import SimRobot
from x30_modes.primitives import (Primitive, TranslateController, TurnController, parse_bearing, parse_primitive_text,
                                  primitive_to_skill)


def test_parse_text():
    assert parse_primitive_text("move forward 75 cm") == Primitive("move_forward", 75)
    assert parse_primitive_text("Move forward 25.5 centimeters") == Primitive("move_forward", 25.5)
    assert parse_primitive_text("turn right 30 degrees") == Primitive("turn_right", 30)
    assert parse_primitive_text("turn_left(15)") == Primitive("turn_left", 15)
    assert parse_primitive_text("strafe left 20 cm") == Primitive("strafe_left", 20)
    assert parse_primitive_text("move backward 40") == Primitive("move_backward", 40)
    assert parse_primitive_text("stop") == Primitive("stop")
    assert parse_primitive_text("The next action is move forward 50 cm") == Primitive("move_forward", 50)
    with pytest.raises(ValueError):
        parse_primitive_text("dance")


def test_primitive_to_skill():
    assert primitive_to_skill(Primitive("move_forward", 75)) == ("move_relative", {"direction": "forward", "distance_cm": 75})
    assert primitive_to_skill(Primitive("strafe_right", 20)) == ("move_relative", {"direction": "right", "distance_cm": 20})
    assert primitive_to_skill(Primitive("turn_right", 30)) == ("turn_to", {"degrees": -30})
    assert primitive_to_skill(Primitive("turn_left", 30)) == ("turn_to", {"degrees": 30})
    assert primitive_to_skill(Primitive("stop")) == ("stop", {})


def test_parse_bearing():
    assert parse_bearing("left") == 90
    assert parse_bearing("Behind") == 180
    assert parse_bearing("front right") == -45
    assert parse_bearing(-30) == -30
    assert parse_bearing("45") == 45
    with pytest.raises(ValueError):
        parse_bearing("sideways")


def run_translate(direction, dist_cm, v_max=0.3, tau=0.15, delay=0.05, yaw0=0.3, coast=0.2):
    r = SimRobot(1.0, 2.0, yaw0, tau=tau, delay=delay)
    ctrl = TranslateController(r.pose, direction, dist_cm / 100.0, v_max=v_max, accel=0.5, decel=0.6, coast_s=coast)
    t = 0.0
    for _ in range(int(60 / r.dt)):
        cmd, done = ctrl.step(r.pose, t)
        if done:
            break
        r.step(cmd)
        t += r.dt
    for _ in range(int(2.0 / r.dt)):       # coast to rest with zero command
        r.step((0, 0, 0))
    return r, ctrl


@pytest.mark.parametrize("direction,dist,ux,uy", [("forward", 100, 1, 0), ("backward", 60, -1, 0), ("left", 50, 0, 1), ("right", 50, 0, -1)])
def test_translate_accuracy(direction, dist, ux, uy):
    r, ctrl = run_translate(direction, dist, yaw0=0.3)
    c, s = math.cos(0.3), math.sin(0.3)
    dx, dy = r.x - 1.0, r.y - 2.0
    along = dx * (c * ux - s * uy) + dy * (s * ux + c * uy)
    assert abs(along * 100 - dist) < 3.0, along
    assert abs(r.yaw - 0.3) < math.radians(3)


def test_translate_short_and_slow_robot():
    r, _ = run_translate("forward", 20, v_max=0.2, tau=0.4, delay=0.1, coast=0.5)
    assert abs((r.x - 1.0) * 100 - 20) < 4.0


def run_turn(deg, w_max=0.5, tau=0.15, delay=0.05):
    r = SimRobot(0, 0, 1.0, tau=tau, delay=delay)
    ctrl = TurnController(r.yaw, math.radians(deg), w_max, accel=1.0, decel=1.0)
    t = 0.0
    for _ in range(int(60 / r.dt)):
        cmd, done = ctrl.step(r.yaw, t)
        if done:
            break
        r.step(cmd)
        t += r.dt
    for _ in range(int(2.0 / r.dt)):
        r.step((0, 0, 0))
    return math.degrees(ctrl.tracker.update(r.yaw))


@pytest.mark.parametrize("deg", [90, -90, 30, -15, 180, -270, 360])
def test_turn_accuracy(deg):
    achieved = run_turn(deg)
    assert abs(achieved - deg) < 3.0, achieved


def test_turn_zero_is_done_immediately():
    ctrl = TurnController(0.0, 0.0, 0.5, 1.0, 1.0)
    cmd, done = ctrl.step(0.0, 0.0)
    assert done and cmd == (0.0, 0.0, 0.0)
