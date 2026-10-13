import math

from x30_follow.controller import FollowParams, command

P = FollowParams()


def run(px, py, pvx=0.0, pvy=0.0, prev=(0.0, 0.0, 0.0), dt=1.0):
    return command(px, py, pvx, pvy, P, prev, dt)


def test_far_person_ahead_drives_forward_and_stays_straight():
    vx, vy, wz = run(5.0, 0.0)
    assert 0.3 < vx <= P.vmax and vy == 0.0 and abs(wz) < 1e-9


def test_inside_deadband_no_motion():
    vx, _, wz = run(P.d_target + 0.05, 0.0)
    assert vx == 0.0 and wz == 0.0


def test_never_approaches_below_the_floor_and_backs_off():
    for d in (1.99, 1.8, 1.0, 0.5):
        vx, _, _ = run(d, 0.0)
        assert vx < 0.0 and vx >= -P.vback - 1e-9
    vx, _, _ = run(2.1, 0.0)
    assert vx <= 0.0                                 # just above the floor: still no forward command


def test_bearing_sign_and_turn_first():
    _, _, wz_left = run(4.0, 1.5)
    _, _, wz_right = run(4.0, -1.5)
    assert wz_left > 0 > wz_right
    vx_side, _, _ = run(0.1, 4.0)                    # person at 90 deg: turn, do not drive
    assert vx_side <= 0.05


def test_limits_hold():
    vx, _, wz = run(20.0, 20.0)
    assert vx <= P.vmax and abs(wz) <= P.wzmax


def test_feed_forward_helps_a_walking_person():
    base, _, _ = run(3.2, 0.0, 0.0, 0.0)
    ff, _, _ = run(3.2, 0.0, 0.8, 0.0)
    assert ff > base


def test_slew_limit():
    vx, _, wz = command(6.0, 3.0, 0, 0, P, (0.0, 0.0, 0.0), 0.05)
    assert vx <= P.ax * 0.05 + 1e-9 and abs(wz) <= P.aw * 0.05 + 1e-9
