"""ReplicaCAD scene -> registered Gazebo world.  python scripts/convert_replicacad.py apt_0 [world_id]"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import registry  # noqa: E402
from x30_datasets.adapters import replicacad as rc  # noqa: E402
from x30_datasets.adapters.scene2sdf import scene2sdf  # noqa: E402
from x30_datasets.paths import external_dir  # noqa: E402

LICENSE = ("ReplicaCAD: LICENSE.txt in the dataset says CC BY-NC 4.0 (the Hugging Face card says CC BY 4.0); "
           "treated as non-commercial, attribution required")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1]
    wid = sys.argv[2] if len(sys.argv) > 2 else f"replicacad_{name}"
    root = external_dir() / "replicacad"
    c = rc.compose(root, name)
    r = scene2sdf(Path("replicacad") / name, wid, up="y", floor=0.0, scene=c.scene, instances=c.instances, voxel=0.08,
                  license=LICENSE, source=f"ai-habitat/ReplicaCAD_dataset {name}", robot_radius=0.25, source_dataset="replicacad",
                  source_scene_id=name, tex_max=int(__import__("os").environ.get("X30_TEX_MAX", 2048)), object_models=True, ground_visible=False, ambient_light=0.7, sun_diffuse=0.5)
    registry.register_world(wid, r.entry)
    print(json.dumps(r.stats))
