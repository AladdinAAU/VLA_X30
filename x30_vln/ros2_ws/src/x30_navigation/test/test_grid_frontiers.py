import numpy as np

from x30_navigation.frontiers import FrontierParams, find_frontiers, select
from x30_navigation.grid import GridParams, LogOddsGrid


def ring_scan(origin, walls, n=720, rmax=10.0):
    """Hit points of a 2D lidar in a rectangular room [xmin,xmax,ymin,ymax] and ground points on the way."""
    xmin, xmax, ymin, ymax = walls
    az = np.linspace(-np.pi, np.pi, n, endpoint=False)
    d = np.column_stack((np.cos(az), np.sin(az)))
    t = np.full(n, np.inf)
    for axis, val in ((0, xmin), (0, xmax), (1, ymin), (1, ymax)):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (val - origin[axis]) / d[:, axis]
        t = np.where((tt > 0) & (tt < t), tt, t)
    hits = origin + d * t[:, None]
    ground = np.vstack([origin + d * (t[:, None] * s) for s in np.linspace(0.1, 0.95, 10)])
    return hits[t < rmax], ground


def test_room_is_mapped_free_inside_occupied_on_walls():
    g = LogOddsGrid(GridParams(resolution=0.05))
    origin = np.array([1.0, 1.0])
    hits, ground = ring_scan(origin, (-2, 4, -1, 3))
    for _ in range(3):
        g.integrate(origin, hits, ground)
    m = g.to_int8()
    c = g.world_to_cell(origin)
    assert m[c[1], c[0]] == 0
    wall_cell = g.world_to_cell(np.array([4.01, 1.0]))
    assert m[wall_cell[1], wall_cell[0]] == 100
    outside = g.world_to_cell(np.array([6.0, 1.0]))
    assert m[outside[1], outside[0]] == -1
    assert (m == 100).sum() > 100 and (m == 0).sum() > 5000


def test_grid_grows_when_robot_leaves():
    g = LogOddsGrid(GridParams(initial_size=10.0, grow_pad=5.0))
    g.integrate(np.zeros(2), *ring_scan(np.zeros(2), (-2, 2, -2, 2)))
    h0, w0 = g.shape
    far = np.array([30.0, 0.0])
    g.integrate(far, *ring_scan(far, (28, 32, -2, 2)))
    assert g.shape[1] > w0
    m = g.to_int8()
    c = g.world_to_cell(far)
    assert m[c[1], c[0]] == 0
    c0 = g.world_to_cell(np.zeros(2))
    assert m[c0[1], c0[0]] == 0  # old content preserved


def test_hit_priority_over_free_in_same_scan():
    g = LogOddsGrid(GridParams(resolution=0.05))
    origin = np.zeros(2)
    hits = np.array([[1.0, 0.0], [2.0, 0.0]])   # near hit and a farther hit on the same ray
    for _ in range(2):
        g.integrate(origin, hits, np.zeros((0, 2)))
    m = g.to_int8()
    c = g.world_to_cell(np.array([2.0, 0.0]))
    assert m[c[1], c[0]] == 100                  # farther hit is not erased by the ray to the nearer one


def _two_room_map():
    g = np.full((200, 200), -1, dtype=np.int8)
    g[50:150, 50:150] = 0
    g[50, 50:150] = g[149, 50:150] = g[50:150, 50] = g[50:150, 149] = 100
    g[51:149, 51:149] = 0
    g[95:105, 149] = 0         # door in the east wall, unknown behind it
    return g


def test_frontier_at_door_and_not_behind_wall():
    grid = _two_room_map()
    res = FrontierParams(downsample=1, robot_radius=0.1, min_frontier_length=0.2)
    out = find_frontiers(grid, 0.05, np.zeros(2), np.array([100 * 0.05, 100 * 0.05]), res)
    assert len(out.frontiers) >= 1
    fr = select(out.frontiers, "nearest")
    assert abs(fr.goal[1] - 100 * 0.05) < 0.6 and fr.goal[0] > 100 * 0.05
    assert 0.0 < out.explored_fraction < 1.0


def test_exclude_and_strategies():
    grid = _two_room_map()
    grid[95:105, 50] = 0       # a second door on the west
    p = FrontierParams(downsample=2, robot_radius=0.1, min_frontier_length=0.2)
    out = find_frontiers(grid, 0.05, np.zeros(2), np.array([5.0, 5.0]), p)
    assert len(out.frontiers) == 2
    first = select(out.frontiers, "nearest")
    second = select(out.frontiers, "nearest", exclude=[first.goal[:2]])
    assert second is not None and second.goal[:2] != first.goal[:2]
    assert select(out.frontiers, "nearest", exclude=[fr.goal[:2] for fr in out.frontiers]) is None
    west = select(out.frontiers, "biased", bias_xy=[0.0, 5.0])
    assert west.goal[0] < 5.0


def test_fully_explored_map_reports_one():
    grid = np.full((100, 100), -1, dtype=np.int8)
    grid[10:90, 10:90] = 0
    grid[10, 10:90] = grid[89, 10:90] = grid[10:90, 10] = grid[10:90, 89] = 100
    out = find_frontiers(grid, 0.05, np.zeros(2), np.array([2.5, 2.5]), FrontierParams(downsample=1, robot_radius=0.1))
    assert out.frontiers == [] and out.explored_fraction == 1.0


def test_body_hits_stop_carving_but_are_not_marked():
    """A wall closer than the body box: its points only cap the rays, so the free space does not run through the wall."""
    g = LogOddsGrid(GridParams(resolution=0.05, initial_size=20.0))
    ang = np.linspace(-0.5, 0.5, 60)
    wall = np.column_stack((np.full_like(ang, 0.6), 0.6 * np.tan(ang)))        # wall 0.6 m ahead, inside the body box
    far = np.column_stack((np.full(40, 3.0), np.linspace(-1.0, 1.0, 40)))     # a second wall behind it, seen over the top
    g.integrate(np.zeros(2), far, np.zeros((0, 2)), wall)
    cells = g.to_int8()
    ix = lambda x, y: tuple(g.world_to_cell(np.array([[x, y]]))[0])
    assert cells[ix(0.3, 0.0)[1], ix(0.3, 0.0)[0]] == 0               # in front of the wall: free
    assert cells[ix(1.5, 0.0)[1], ix(1.5, 0.0)[0]] == -1              # behind the wall: still unknown
    assert cells[ix(0.6, 0.0)[1], ix(0.6, 0.0)[0]] != 100             # not marked occupied (the legs would be)
