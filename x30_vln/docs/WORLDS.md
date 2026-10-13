# Worlds and datasets

Single reference for the simulation worlds (`ros2_ws/src/x30_worlds`), the dataset code (`datasets/x30_datasets`) and the episodes.
Verification numbers: `docs/reports/worlds.md`. Priority 1 are the **real-scan worlds** (textured scans with ground truth); the procedural
flat-shaded worlds stay for fast tests.

## Worlds

| family | ids | source | licence |
|---|---|---|---|
| real scan, Replica | `replica_apartment_0/1/2`, `replica_frl_apartment_0`, `replica_hotel_0`, `replica_office_0`, `replica_room_0` | facebookresearch/Replica-Dataset v1.0 (all 18 scenes downloaded, 7 converted) | Replica Dataset Research Terms: non-commercial research / education only; terms text in `datasets/external/replica_src/LICENSE`. The terms say that accessing the data means agreeing to them. |
| real-scan-like, ReplicaCAD | `replicacad_apt_0/1/2/3` (FRL apartment with 4 object layouts, doors opened) | Hugging Face `ai-habitat/ReplicaCAD_dataset` (153 MB, "interactive" variant, textures without baked lighting) | `LICENSE.txt` in the dataset: **CC BY-NC 4.0**; the Hugging Face card says CC BY 4.0. Treated as non-commercial, attribution required. |
| scan, no semantics | `habtest_apartment_1`, `habtest_van_gogh_room` | Habitat test scenes zip, `dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip` (94.6 MB; `skokloster-castle` is in the archive, not converted) | no licence file in the archive; Hugging Face card `ai-habitat/habitat_test_scenes`: cc-by-nc-4.0. `habtest_apartment_1` is the Replica apartment_1 scan (same extent) |
| AWS RoboMaker | `aws_bookstore`, `aws_hospital`, `aws_small_house`, `aws_small_warehouse` | cloned repos, converted (`datasets/x30_datasets/adapters/aws.py`) | MIT-0 (Fuel models CC-BY-4.0) |
| procedural | `apt_01..04`, `office_01/02`, `lab_01/02`, `warehouse_01/02` | `x30_datasets.procgen` | CC0 |
| mesh test | `sponza_test` | Khronos Sponza | Crytek licence |

Real-scan worlds in numbers (from `registry.yaml`, `python datasets/scripts/world_table.py`; reachable = largest connected free area of the
robot-centre set, planning radius in brackets):

| world | source scene | extent m | visual tris | ceiling tris | collision boxes | reachable m2 (r) | GT objects | spawn |
|---|---|---|---|---|---|---|---|---|
| habtest_apartment_1 | apartment_1.glb | 7.9x10.74x2.94 | 249911 | 36196 | 553 | 23.5 (0.35) | 0 | 0.92, -4.21 |
| habtest_van_gogh_room | van-gogh-room.glb | 6.77x3.54x3.86 | 11272 | 98 | 164 | 5.7 (0.25) | 0 | 3.29, -0.5 |
| replica_apartment_0 | apartment_0 | 9.37x14.77x5.35 | 248269 | 46508 | 1503 | 30.1 (0.25) | 185 | 0.58, -5.54 |
| replica_apartment_1 | apartment_1 | 10.74x7.9x2.94 | 249279 | 54739 | 525 | 34.6 (0.25) | 75 | 4.08, 0.92 |
| replica_apartment_2 | apartment_2 | 9.44x10.2x2.81 | 248928 | 50038 | 791 | 27.0 (0.25) | 103 | 0.96, 3.42 |
| replica_frl_apartment_0 | frl_apartment_0 | 7.25x12.65x3.08 | 248639 | 5531 | 994 | 23.5 (0.25) | 176 | 3.99, -1.51 |
| replica_hotel_0 | hotel_0 | 8.59x4.18x2.75 | 249475 | 38247 | 455 | 5.3 (0.25) | 83 | 4.07, 1.17 |
| replica_office_0 | office_0 | 4.4x5.01x2.99 | 249670 | 46484 | 340 | 3.4 (0.25) | 28 | 0.22, -1.83 |
| replica_room_0 | room_0 | 7.76x4.7x2.81 | 249490 | 53906 | 293 | 8.3 (0.25) | 64 | 0.65, 2.29 |
| replicacad_apt_0 | apt_0 | 7.26x13.0x3.13 | 147002 | 2026 | 1381 | 24.9 (0.25) | 118 | 2.44, -0.23 |
| replicacad_apt_1 | apt_1 | 7.26x13.0x3.13 | 149018 | 2076 | 1252 | 30.4 (0.25) | 125 | 2.19, -0.13 |
| replicacad_apt_2 | apt_2 | 7.26x13.0x3.13 | 151206 | 2077 | 1284 | 21.6 (0.25) | 124 | -0.71, -2.83 |
| replicacad_apt_3 | apt_3 | 7.26x13.0x3.13 | 151070 | 2067 | 1228 | 27.3 (0.25) | 123 | 2.0, -3.18 |

All of them are registered with `spawn`, `bounds`, `objects_file`, `license`, `source_dataset`, `source_scene_id`, `robot_radius_m`.

### What a converted real-scan world contains
* `models/<id>_scene/meshes/scene.obj` (visual body), `ceiling.obj` (downward faces above 2 m, a separate model so a world can drop it; the lidar needs it),
  `tex_*.jpg` shared. Scans are decimated to about 250 k triangles and **re-baked into one 2048 px atlas** (Replica meshes are 9 M triangles with vertex
  colours; quadric decimation + xatlas unwrap + nearest-colour bake, `x30_datasets/budget.py`). ReplicaCAD is used as is (about 150 k triangles, textures capped at 2048 px).
* `scene_collision`: invisible box collision of everything between 0.12 and 2 m (voxels 0.08-0.1 m merged into 300-1500 boxes); the floor is the ground plane
  (not drawn, the scan floor is). Meshes are shifted so the floor is z = 0 and rotated Z-up (Replica and Habitat test glb are stored Z-up; ReplicaCAD Y-up).
* One labelled model per ground-truth object (`<x30:label>`, `<x30:instance>`, box collision with `collide_bitmask` 0 so it never blocks; the simulator reads
  class and box from it, `/gt/objects` shows the same count as `objects.json`). Replica: boxes from the faces carrying each instance id; class from
  `info_semantic.json`; structure classes (wall, floor, ceiling, plugs, switches, vents, ...) are not objects. ReplicaCAD: boxes of the placed assets, class from the
  object config's semantic id (name overrides for known wrong ids). Classes are mapped onto `config/labels.yaml` (27 classes added, aliases for the external names).
* Lighting: Replica textures are baked; the world uses ambient 0.7 and a 0.5 sun without shadows.

## Commands
```
cd datasets            # PYTHONPATH=. for python -m
python scripts/convert_replica.py apartment_2 [triangle_budget]     # about 5-20 min per scene (xatlas), 8 GB RAM
python scripts/convert_replicacad.py apt_1                          # seconds; env X30_TEX_MAX caps textures
python scripts/convert_habitat_test.py apartment_1 [budget]
python -m x30_datasets gen-episodes <world> --n-object 30 --n-pose 20 --n-follow 12
python -m x30_datasets validate
python scripts/door_check.py <world>          # doors with reachable floor next to them
python scripts/make_view_poses.py <world> 20  # + scripts/sim_capture_world.sh <world>: frames from the simulator  + scripts/eval_detector.py <world>
```
Run a world: `docker exec x30-sim bash /x30/datasets/scripts/sim_start_world.sh replica_apartment_2` (domain 70, own Gazebo partition) or the usual
`ros2 launch x30_sim sim.launch.py world:=<id> locomotion:=kinematic`. The simulator resolves `worlds/` and `models/` in the installed share; either rebuild
`x30_worlds` or symlink them to the source tree.

## Adding a dataset
* Licensed data (HM3D, MP3D, ScanNet, ARKitScenes, Gibson) is **not** downloaded by anyone but the user, after accepting its licence. Put it under
  `datasets/external/<name>/` and import with `python -m x30_datasets import-scene <path>` (adapters `habitat.py`: HM3D `*.basis.glb` + `.semantic.txt/glb`, MP3D `.house`,
  ScanNet; they were verified only on synthetic mini-datasets). HM3D / MP3D ObjectNav and R2R episode files: `python -m x30_datasets import-episodes <file.json.gz>`.
* Another textured scan: write a small adapter returning a `trimesh.Scene` plus instances `{id, label, bbox_min, bbox_max}` (see `adapters/replica.py`,
  about 60 lines) and call `scene2sdf(..., scene=, instances=, budget_tris=, object_models=True)`; register with `registry.register_world`.
* Episodes: `gen-episodes`; split dev/heldout is by room; `validate` checks the schema.

## Episodes
`datasets/episodes/<world>_{object_nav,goal_pose,velocity_follow}.yaml`, schema `datasets/x30_datasets/schemas/episode.schema.json`, dev/heldout split
(fraction 0.6 dev). ObjectNav uses "any instance of the class within 1.5 m" scoring over the real instances of the scan. Worlds with a small reachable area give few
or no episodes (`replica_office_0`: none; `replica_room_0`: no object_nav). object_nav counts are limited by the classes that occur in the reachable part.

## Known limits
* No ground truth for the `habtest_*` worlds. Real-scan floors must be flat for the kinematic backend; stairs of the scans (Replica apartment_0 has some) are solid steps for collision.
* Scan noise: floating specks become small obstacles; the speckle in the baked texture is visible.
* ReplicaCAD worlds use 7 GB (up to 14 GB while loading) of Gazebo memory in software rendering, Replica worlds 1.2-1.9 GB (see report).
* A rough world of the real lab from the accumulated bag cloud was **not** made.
