import numpy as np
from scipy import ndimage
from x30_navigation.known_map import CtrlParams, KnownMap, control, footprint_contact, path_length


def make(b):
    return KnownMap(b, np.zeros(2), 0.1)


def room():
    b = np.zeros((100, 100), bool)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    b[:70, 50] = True          # wall with a 3 m gap at the top
    return b


def test_plan_goes_through_gap():
    km = make(room())
    p, r = km.plan([2.0, 2.0], [8.0, 2.0])
    assert r == 0.55 and p[:, 1].max() > 7.0 and min(km.clearance(q) for q in p) >= 0.5
    assert path_length(p) < 22


def test_narrow_tier_and_unreachable():
    b = room()
    b[:, 50] = True
    b[72:80, 50] = False       # 0.8 m door
    p, r = make(b).plan([2.0, 2.0], [8.0, 2.0])
    assert p is not None and r < 0.55
    b[:, 50] = True
    assert make(b).plan([2.0, 2.0], [8.0, 2.0]) is None


def test_control_no_rotation_in_tight_space():
    path = np.array([[0, 0], [0, 2.0]])
    vx, vy, wz, _ = control((0, 0, 0.0), path, 0.3, CtrlParams())
    assert vx >= 0 and vy > 0 and abs(wz) <= 0.25
    vx, vy, wz, _ = control((0, 0, 0.0), path, 2.0, CtrlParams())
    assert vx == 0 and wz > 0


def test_footprint_contact():
    b = np.zeros((50, 50), bool)
    b[:, 30] = True
    assert not footprint_contact(b, np.zeros(2), 0.1, (2.0, 2.0, 0.0))
    assert footprint_contact(b, np.zeros(2), 0.1, (2.7, 2.0, 0.0))


def test_nearest_reachable_moves_a_pocket_goal_to_the_reachable_side():
    """A goal inside an enclosed pocket (e.g. under a desk) is unreachable; the nearest reachable cell is on the open side."""
    blocked = np.zeros((60, 60), bool)
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
    blocked[20:40, 30] = blocked[20, 30:45] = blocked[39, 30:45] = blocked[20:40, 44] = True      # closed box: x 3.0-4.4, y 2.0-3.9
    km = KnownMap(blocked, np.array([0.0, 0.0]), 0.1)
    start, pocket = np.array([1.0, 3.0]), np.array([3.7, 3.0])
    assert km.plan(start, pocket) is None
    alt = km.nearest_reachable(start, pocket, 3.0)
    assert alt is not None
    xy, _ = alt
    assert km.plan(start, xy) is not None and np.hypot(*(xy - pocket)) < 3.0 and xy[0] < 3.0
    assert km.nearest_reachable(start, pocket, 0.2) is None
