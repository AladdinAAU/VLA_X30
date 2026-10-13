import math

import numpy as np

from x30_follow.trail import Breadcrumbs, GridMap


def walled_grid():
    """10 x 10 m, 0.1 m cells: a wall along x = 5 m from y = 0 to y = 6 m (a door gap above y = 6)."""
    d = np.zeros((100, 100), np.int8)
    d[0:60, 49:51] = 100
    return GridMap(d, 0.1, 0.0, 0.0)


def test_line_of_sight_is_blocked_by_the_wall_and_free_through_the_door():
    g = walled_grid()
    assert not g.line_free((2.0, 3.0), (8.0, 3.0))          # straight through the wall
    assert g.line_free((2.0, 8.0), (8.0, 8.0))              # above the wall
    assert g.line_free((2.0, 3.0), (4.0, 5.0))              # same side
    assert not g.free(5.0, 3.0) and g.free(2.0, 2.0) and not g.free(-1.0, 3.0)


def test_point_behind_follows_the_trail_around_a_corner():
    b = Breadcrumbs(step=0.3)
    for k in range(0, 21):                                  # walked east 6 m, then north 6 m (an L)
        b.add(k * 0.3, 0.0)
    for k in range(1, 21):
        b.add(6.0, k * 0.3)
    head = (6.0, 6.0)
    p = b.point_behind(head, 2.5)
    assert abs(p[0] - 6.0) < 1e-6 and abs(p[1] - 3.5) < 0.05             # on the leg she is walking on
    q = b.point_behind(head, 8.0)
    assert abs(q[1] - 0.0) < 0.05 and abs(q[0] - 4.0) < 0.35             # 6 m back down the second leg, then 2 m back along the first
    assert b.point_behind(head, 100.0) == b.pts[0]                       # trail shorter than asked: its oldest point
    assert Breadcrumbs().point_behind(head, 2.0) is None


def test_trail_is_spaced_and_trimmed():
    b = Breadcrumbs(step=0.5, max_len=5.0)
    for k in range(200):
        b.add(k * 0.1, 0.0)
    assert all(math.hypot(b.pts[i + 1][0] - b.pts[i][0], 0) >= 0.5 - 1e-9 for i in range(len(b.pts) - 1))
    length = sum(abs(b.pts[i + 1][0] - b.pts[i][0]) for i in range(len(b.pts) - 1))
    assert length <= 5.0 + 0.6 and b.last()[0] > 19.0
