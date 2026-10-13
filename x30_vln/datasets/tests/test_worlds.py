import json

from x30_datasets import gridmap as gm
from x30_datasets.gridmap import GridMap
from x30_datasets.procgen.builder import PRESETS, generate_world
from x30_datasets.sdfparse import parse_world, rasterize


def test_procgen_world_is_valid_and_deterministic(tmp_path):
    a = generate_world("apt_03", out_dir=tmp_path / "a")
    generate_world("apt_03", out_dir=tmp_path / "b")
    assert (tmp_path / "a" / "world.sdf").read_text() == (tmp_path / "b" / "world.sdf").read_text()
    assert a.entry["stats"]["min_door_width_m"] >= 1.0
    objs = json.loads((tmp_path / "a" / "objects.json").read_text())
    assert len({o["instance"] for o in objs}) == len(objs)
    assert {"sofa", "bed", "door", "window"} <= {o["label"] for o in objs}


def test_procgen_map_connected_and_doors_passable(tmp_path):
    res = generate_world("apt_01", out_dir=tmp_path)
    grid = GridMap.load(tmp_path / "map.yaml")
    infl = grid.inflated(0.35)
    sp = res.entry["spawn"]
    comp = gm.component_of(infl, grid.world_to_cell(sp["x"], sp["y"]))
    rooms = json.loads((tmp_path / "rooms.json").read_text())
    for d in rooms["doors"]:
        assert d["width"] >= 1.0
        cy, cx = grid.world_to_cell(d["x"], d["y"])
        assert comp[cy, cx], "door " + d["id"] + " not passable for the inflated robot"


def test_every_preset_generates(tmp_path):
    assert len(PRESETS) >= 8
    for wid in PRESETS:
        r = generate_world(wid, out_dir=tmp_path / wid)
        assert r.entry["stats"]["objects"] > 20


def test_sdf_roundtrip_rasterizes(tmp_path):
    generate_world("apt_03", out_dir=tmp_path)
    geo, missing = parse_world(tmp_path / "world.sdf")
    assert not missing
    g = rasterize(geo.geoms("collision"), (-2, -6, 12, 6), 0.1)
    assert 0.05 < g.occ.mean() < 0.5


def test_labels_have_ids():
    from x30_datasets import labels
    assert labels.label_id("chair") == 10 and labels.canonical("Couch") == "sofa" and labels.canonical("???") == "other"
