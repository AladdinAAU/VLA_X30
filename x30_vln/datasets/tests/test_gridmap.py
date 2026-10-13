import numpy as np

from x30_datasets import gridmap as gm


def _room():
    occ = np.zeros((20, 30), bool)
    occ[:, 15] = True
    occ[8:12, 15] = False  # door
    return occ


def test_astar_through_door():
    occ = _room()
    p = gm.astar(occ, (10, 2), (10, 27))
    assert p and p[0] == (10, 2) and p[-1] == (10, 27)
    assert all(not occ[c] for c in p)
    assert abs(gm.path_length(p, 1.0) - 25.0) < 1e-6


def test_astar_unreachable():
    occ = _room()
    occ[:, 15] = True
    assert gm.astar(occ, (10, 2), (10, 27)) is None


def test_distance_field_matches_astar():
    occ = _room()
    d = gm.distance_field(occ, [(10, 2)])
    p = gm.astar(occ, (10, 2), (3, 27))
    assert abs(d[3, 27] - gm.path_length(p, 1.0)) < 1e-6
    assert np.isinf(d[0, 15])


def test_shortcut_keeps_free_and_shortens():
    occ = _room()
    p = gm.astar(occ, (1, 1), (18, 28))
    s = gm.shortcut_path(occ, p)
    assert len(s) <= len(p) and s[0] == p[0] and s[-1] == p[-1]
    for a, b in zip(s, s[1:]):
        assert gm.line_of_sight(occ, a, b)


def test_inflation_and_clearance():
    occ = np.zeros((21, 21), bool)
    occ[10, 10] = True
    g = gm.GridMap(occ, 0.1, (0, 0))
    infl = g.inflated(0.35)
    assert infl[10, 10] and infl[10, 13] and not infl[10, 15]
    assert abs(g.clearance()[10, 14] - 0.4) < 1e-9


def test_map_save_load(tmp_path):
    occ = _room()
    g = gm.GridMap(occ, 0.05, (-1.0, 2.0))
    g.save(tmp_path / "m.yaml")
    h = gm.GridMap.load(tmp_path / "m.yaml")
    assert (h.occ == occ).all() and h.origin == (-1.0, 2.0) and h.resolution == 0.05
    assert g.world_to_cell(*g.cell_to_world(4, 7)) == (4, 7)
