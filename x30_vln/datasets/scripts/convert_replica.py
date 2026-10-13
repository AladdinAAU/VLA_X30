"""Replica scene -> registered world.  python scripts/convert_replica.py apartment_0 [budget_tris]"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trimesh  # noqa: E402

from x30_datasets import registry  # noqa: E402
from x30_datasets.adapters import replica  # noqa: E402
from x30_datasets.adapters.scene2sdf import scene2sdf  # noqa: E402
from x30_datasets.paths import external_dir  # noqa: E402

LICENSE = "Replica Dataset Research Terms (Facebook Technologies): non-commercial research/education use only; see datasets/external/replica_src/LICENSE"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
    d = external_dir() / "replica" / name
    mesh, fobj = replica.read_mesh_semantic(d / "habitat" / "mesh_semantic.ply")
    inst = replica.instances(d, mesh, fobj)
    r = scene2sdf(Path("replica") / name, f"replica_{name}", up="z", scene=trimesh.Scene(mesh), instances=inst, budget_tris=budget,
                  voxel=0.1, license=LICENSE, source=f"facebookresearch/Replica-Dataset {name}", robot_radius=0.25, source_dataset="replica",
                  source_scene_id=name, object_models=True, ground_visible=False, ambient_light=0.7, sun_diffuse=0.5)
    registry.register_world(r.world_id, r.entry)
    print(json.dumps(r.stats))
