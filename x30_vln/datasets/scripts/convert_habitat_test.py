"""Habitat test scenes (glb stored Z up, no semantics) -> registered worlds.  python scripts/convert_habitat_test.py apartment_1 [budget]"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_datasets import registry  # noqa: E402
from x30_datasets.adapters.scene2sdf import scene2sdf  # noqa: E402
from x30_datasets.paths import external_dir  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
    src = next((external_dir() / "habitat_test").rglob(f"{name}.glb"))
    r = scene2sdf(src, f"habtest_{name.replace('-', '_')}", up="z", budget_tris=budget, voxel=0.1,
                  license="Habitat test scenes: Hugging Face card cc-by-nc-4.0 (no licence file in the archive); apartment_1 derives from Replica (research terms)",
                  source="habitat-test-scenes.zip (dl.fbaipublicfiles.com/habitat)", robot_radius=0.25, source_dataset="habitat-test-scenes",
                  source_scene_id=src.name, object_models=True, ground_visible=False, ambient_light=0.7, sun_diffuse=0.5)
    registry.register_world(r.world_id, r.entry)
    print(json.dumps(r.stats))
