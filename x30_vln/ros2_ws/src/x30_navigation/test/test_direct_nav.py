import numpy as np

from x30_navigation.direct_nav import TrackParams, plan_path, straight_line, track


def room_with_wall():
    g = np.zeros((100, 100), np.int8)
    g[:, 50] = 100
    g[40:60, 50] = 0            # a 1 m door at 0.05 m resolution
    return g


def test_plan_goes_through_the_door():
    p = plan_path(room_with_wall(), 0.05, np.zeros(2), (1.0, 1.0), (4.0, 1.0), radius=0.2)
    assert p is not None and np.linalg.norm(p[-1] - [4.0, 1.0]) < 1e-6
    crossing = p[np.argmin(np.abs(p[:, 0] - 2.5))]
    assert 2.0 <= crossing[1] <= 3.0


def test_no_path_through_closed_wall():
    g = room_with_wall()
    g[:, 50] = 100
    assert plan_path(g, 0.05, np.zeros(2), (1.0, 1.0), (4.0, 1.0), radius=0.2) is None


def test_track_turns_then_drives_and_slows():
    path = straight_line((0, 0), (3, 0))
    p = TrackParams()
    vx, _, wz, _ = track((0, 0, np.pi), path, p)
    assert vx == 0.0 and abs(wz) > 0.3                      # facing away: rotate in place
    vx, _, wz, _ = track((0, 0, 0), path, p)
    assert vx > 0.3 and abs(wz) < 1e-6
    vx_far, *_ = track((1.0, 0, 0), path, p)
    vx_near, *_ = track((2.9, 0, 0), path, p)
    assert vx_near < vx_far and vx_near >= p.v_min * 0.99
