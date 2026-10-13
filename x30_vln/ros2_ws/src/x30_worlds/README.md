# x30_worlds

Data-only ament package: Gazebo Harmonic worlds, their models and ground truth for the X30 VLN trials.

```
worlds/<id>/world.sdf      the world (labelled GT object models: <x30:label>/<x30:instance>, convention in docs/requests/sim-to-worlds.md)
worlds/<id>/objects.json   ground-truth objects (instance, class, pose, 3D box, room)
worlds/<id>/map.yaml+png   2D occupancy map of the obstacle layer (robot height band)
worlds/<id>/meta.json      the registry entry of the world (same content as datasets/registry.yaml)
models/<id>_scene/meshes   textured visual meshes of imported scans (scene.obj = body, ceiling.obj = ceiling, shared tex_*.jpg)
config/labels.yaml         class table (id, kind, synonyms, aliases for external datasets)
hooks/                     adds share/x30_worlds/models to GZ_SIM_RESOURCE_PATH
```

Install: `colcon build --packages-select x30_worlds` copies everything above (the real-scan models are about 30-60 MB each).
For development the simulator container can symlink `share/x30_worlds/{worlds,models}` to the source tree instead (done in the `x30-sim`
container for the checks in `docs/reports/worlds.md`).

Worlds are registered in `datasets/registry.yaml`; the simulator loads them by id (`world:=replica_apartment_2`).
Content, sources, licences and how to add datasets: `docs/WORLDS.md`. The generators live in `datasets/` (`x30_datasets`, `scripts/convert_*.py`).

Licences: generated procedural worlds are CC0; imported scans keep the licence of their dataset (Replica research terms,
ReplicaCAD CC BY-NC 4.0 per its LICENSE.txt, Habitat test scenes cc-by-nc-4.0, AWS RoboMaker MIT-0, Fuel models CC-BY-4.0). The real-scan
worlds are non-commercial: do not redistribute them.
