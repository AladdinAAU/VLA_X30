import numpy as np
import pytest

from x30_datasets.adapters import episodes_habitat as eh
from x30_datasets.adapters import habitat, synthetic
from x30_datasets.adapters.scene2sdf import greedy_boxes, scene2sdf
from x30_datasets.episodes import io as eio


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("syn")
    synthetic.write_hm3d(tmp / "hm3d_root")
    synthetic.write_mp3d(tmp / "mp3d_root")
    hm = habitat.import_scene(habitat.find_scenes(tmp / "hm3d_root")[0], "syn_hm3d",
                              out_dir=tmp / "worlds" / "syn_hm3d", models_dir=tmp / "models")
    mp = habitat.import_scene(habitat.find_scenes(tmp / "mp3d_root")[0], "syn_mp3d",
                              out_dir=tmp / "worlds" / "syn_mp3d", models_dir=tmp / "models")
    return tmp, hm, mp


def _registry(imported):
    _, hm, mp = imported
    return {"worlds": {"syn_hm3d": hm.entry, "syn_mp3d": mp.entry}, "episodes": {}}


def test_hm3d_layout_parsed(imported):
    _, hm, _ = imported
    assert habitat.label_summary(hm.objects) == {"chair": 2, "table": 1, "sofa": 1, "bed": 1}
    chair = [o for o in hm.objects if o["label"] == "chair"][0]
    assert np.allclose(chair["bbox"], [1.25, 0.95, 0.0, 1.75, 1.45, 0.9], atol=0.02)  # y-up -> z-up, floor at 0
    assert hm.stats["up_axis"] == "y" and abs(hm.stats["floor_z_source"]) < 1e-6


def test_mp3d_house_parsed(imported):
    tmp, _, mp = imported
    assert habitat.label_summary(mp.objects) == {"chair": 2, "table": 1, "sofa": 1, "bed": 1}
    sofa = [o for o in mp.objects if o["label"] == "sofa"][0]
    assert np.allclose(sofa["bbox"][:2], [6.5, 0.55], atol=0.02)
    assert sofa["room"].startswith("bedroom")  # x=7.5 lies in region 'b'
    assert (tmp / "worlds" / "syn_mp3d" / "rooms.json").exists()


def test_navigable_area_and_collision(imported):
    _, hm, _ = imported
    assert 20 < hm.stats["navigable_area_m2"] < 45  # 10 x 6 m house minus walls, furniture, inflation
    assert 5 < hm.stats["collision_boxes"] < 200
    assert 0 < hm.entry["spawn"]["x"] < 10 and 0 < hm.entry["spawn"]["y"] < 6


def test_greedy_boxes_cover_exactly():
    vox = np.random.default_rng(0).random((9, 8, 7)) > 0.6
    cover = np.zeros_like(vox)
    for i0, j0, k0, i1, j1, k1 in greedy_boxes(vox):
        assert not cover[i0:i1, j0:j1, k0:k1].any()
        cover[i0:i1, j0:j1, k0:k1] = True
    assert (cover == vox).all()


def test_scene_transform_in_entry_matches_habitat(imported):
    _, hm, _ = imported
    assert np.allclose(hm.entry["scene_transform"]["R"], [[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def test_objectnav_import(imported):
    tmp, _, _ = imported
    synthetic.write_objectnav(tmp / "on.json.gz", "hm3d/val/00800-TESTscene01/TESTscene01.basis.glb")
    out, stats = eh.import_objectnav(tmp / "on.json.gz", tmp, split_name="val", registry=_registry(imported))
    assert stats["imported"] == 3
    ep = out["syn_hm3d"][0]
    assert eio.schema_errors(ep) == []
    assert abs(ep["start"]["x"] - 1.0) < 1e-2 and abs(ep["start"]["y"] - 4.5) < 1e-2 and abs(ep["start"]["yaw"] - 0.3) < 1e-2
    assert ep["split"] == "heldout" and ep["goal"]["label"] == "chair"
    assert sorted(ep["goal"]["instances"]) == ["chair_01", "chair_02"]
    assert abs(ep["goal"]["x"] - 4.35) < 0.1 and abs(ep["goal"]["y"] - 2.5) < 0.1


def test_r2r_import(imported):
    tmp, _, _ = imported
    synthetic.write_r2r(tmp / "r2r.json.gz", "mp3d/TESTmp3dXYZ/TESTmp3dXYZ.glb")
    out, stats = eh.import_r2r(tmp / "r2r.json.gz", split_name="val_unseen", registry=_registry(imported))
    assert stats["imported"] == 2
    ep = out["syn_mp3d"][0]
    assert eio.schema_errors(ep) == []
    assert ep["mode"] == "velocity_follow" and "doorway" in ep["instruction"]
    assert abs(ep["reference_path"][1]["x"] - 5.0) < 1e-2 and abs(ep["reference_path"][1]["y"] - 3.0) < 1e-2
    assert abs(ep["goal"]["x"] - 8.0) < 1e-2 and ep["goal"]["radius"] == 3.0


def test_unknown_scene_skipped(imported):
    tmp, _, _ = imported
    synthetic.write_r2r(tmp / "r2r2.json.gz", "mp3d/OTHER/OTHER.glb")
    out, stats = eh.import_r2r(tmp / "r2r2.json.gz", registry=_registry(imported))
    assert out == {} and stats["skipped_scene"] == 2


def test_plain_mesh_z_up_obj(tmp_path):
    import trimesh
    room = trimesh.creation.box(extents=(6, 4, 0.1))
    room.apply_translation((3, 2, -0.05))
    wall = trimesh.creation.box(extents=(6, 0.1, 2.5))
    wall.apply_translation((3, 4.05, 1.25))
    trimesh.util.concatenate([room, wall]).export(str(tmp_path / "s.obj"))
    r = scene2sdf(tmp_path / "s.obj", "plain", up="z", out_dir=tmp_path / "w", models_dir=tmp_path / "m")
    assert r.stats["up_axis"] == "z" and r.stats["collision_boxes"] >= 1
