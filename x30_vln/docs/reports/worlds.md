# Worlds report (refocus on real-scan worlds)

Status 2026-10-10. Everything below was run unless marked "not done". Table of the worlds, sources, licences, commands: `docs/WORLDS.md`. Figures: `docs/figures/worlds/`.
Raw data: `results/worlds_check/<world>/` (frames, overlays, `detector_eval.json`, `poses.json`).

## 1. Downloads (user-approved, official sources)
| dataset | size | source | licence as found |
|---|---|---|---|
| ReplicaCAD (interactive) | 153 MB, 602 files | Hugging Face `ai-habitat/ReplicaCAD_dataset` | LICENSE.txt: CC BY-NC 4.0; HF card: CC BY 4.0 (discrepancy; treated as non-commercial) |
| Habitat test scenes | 94.6 MB zip (3 scenes) | `dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip` | no licence file; HF card cc-by-nc-4.0 |
| Replica v1.0 | 33.9 GB (17 release parts, all 18 scenes); extracted without the Ptex textures to 3.9 GB, parts deleted afterwards | GitHub release of facebookresearch/Replica-Dataset (the URLs of its `download.sh`, sizes checked with the releases API) | Replica Dataset Research Terms: non-commercial research/education; "by accessing ... Researcher agrees". No interactive acceptance was asked and none was given by me; the user is bound by the terms |

Not downloaded: HM3D, MP3D, ScanNet, ARKitScenes, Gibson. The baked-lighting ReplicaCAD variant (525 MB) was not needed: the interactive one does not look flat (figures).
Replica's colour is stored as Ptex (several GB per scene, no reader here); the vertex colours of `habitat/mesh_semantic.ply` (about 4.5 M vertices) are used instead.

## 2. Conversion (numbers from `registry.yaml`)
13 real-scan worlds registered (7 Replica, 4 ReplicaCAD, 2 Habitat test). Visual triangles: Replica 9.1 M -> about 249 k (apartment_0 baked in 202 s), ReplicaCAD 147-151 k without decimation,
atlas 2048 px; 290-1500 collision boxes; 28-185 ground-truth objects per scan (table: `python datasets/scripts/world_table.py`).
Downloaded but not converted: Replica `office_1..4`, `room_1/2`, `frl_apartment_1..5`, Habitat `skokloster-castle`. (`office_3` exceeded 100 min in xatlas and was stopped.)
Reachable area (largest free component of the robot-centre set, radius 0.25 m = body width 0.3 m + margin): 21.6-30.4 m2 in ReplicaCAD, 23.4-27 m2 in the big Replica apartments, 3.4-8.3 m2 in the small rooms.
At radius 0.35 the numbers are lower (replica_apartment_0 13.0, replicacad_apt_2 11.8 m2): clutter and 0.7-0.8 m passages. Door check (`scripts/door_check.py`, only a proxy: reachable floor within 0.7 m of the door box centre):
all doors of ReplicaCAD apt_0/1, Replica apartment_2 (4/4), frl_apartment_0 (5/5), room_0 pass at both radii. A 1 m long robot has to be aligned to pass a 0.8 m door; the planner models a circle.
The voxel collision rounds gaps by up to one voxel (0.08-0.1 m).

## 3. Simulator (x30-sim:latest, domain 70, kinematic, software rendering, all sensors: lidar 4 x 19968 rays, RGB 640x480 + depth, thermal 384x288)
| world | RTF | Gazebo RSS | notes |
|---|---|---|---|
| replica_apartment_0 | 1.00 (0.999-1.001, 40 s) | 1.9 GB, +1 MB/s over 40 s | 248 k triangles |
| replica_frl_apartment_0 | 1.00 (20 s) | 1.4 GB constant | |
| replica_apartment_2 | 1.00 (20 s) | 1.2 GB constant | |
| replicacad_apt_0 | 1.00 | 7 GB steady, 11-14 GB during the first minutes | with textures capped at 256 px it was also 7 GB, so it is not texture memory; cause not found |
| replicacad_apt_1 | 1.00 | 7.3 GB | |
| apt_01 (procedural, comparison) | 0.81 | 3.6 GB | run disturbed by a leftover process (topic rates 17-35 Hz); the sim report gives 0.82 |

`/gt/objects` count equals `objects.json` (replicacad_apt_0: 118 objects, 122 models). Looked at: camera frames (textured furniture, floors, walls: `*_camera.jpg`) and a lidar top view
(`replica_apartment_0_lidar_bev.png`: walls and furniture edges at plausible places). Visual artefacts: dark speckles from the baked scan noise, no shadows.
Not tested: legged mode on these worlds, long soaks, dropping the ceiling model, the GUI, a robot driving through an object carrier box (`collide_bitmask 0`).

## 4. Detector check
YOLOE-11l-seg via `POST /segment` (perception container started for the test and stopped afterwards), conf 0.25, vocabulary of 20 phrases: chair, sofa, table, bed, tv, lamp, plant, cabinet, refrigerator, sink, toilet, door, window, picture, cushion, desk, shelf, stool, bench, bicycle.
20 frames per world from poses that look at a ground-truth object 1.5-4.5 m away with line of sight (`make_view_poses.py`), robot camera. A ground-truth object counts as visible if its class is in the vocabulary, its projected
box centre is in the image, depth 0.6-6 m, at least 24 px and 35 % of the box in the image, and it is not blocked in the 2D map. It is detected if a same-class detection overlaps it (IoU >= 0.15 or centre inside).
A detection is a false positive if no same-class ground-truth object of any size or occlusion overlaps it.

| world | type | visible GT | detected | recall | detections | false positives | FP rate |
|---|---|---|---|---|---|---|---|
| apt_01 | procedural flat-shaded | 12 | 0 | 0.00 | 14 | 10 | 0.71 |
| replicacad_apt_0 | ReplicaCAD | 51 | 12 | 0.24 | 111 | 66 | 0.59 |
| replicacad_apt_1 | ReplicaCAD | 56 | 19 | 0.34 | 70 | 43 | 0.61 |
| replica_apartment_0 | Replica scan | 79 | 22 | 0.28 | 74 | 28 | 0.38 |
| replica_apartment_2 | Replica scan | 88 | 9 | 0.10 | 67 | 37 | 0.55 |
| replica_frl_apartment_0 | Replica scan | 55 | 14 | 0.25 | 46 | 22 | 0.48 |
| real-scan total (5 worlds) | | 329 | 76 | 0.23 | 368 | 196 | 0.53 |

Per class, five real-scan worlds together (visible GT detected): chair 35/68, door 3/44, table 2/44, window 0/30, cabinet 3/26, plant 3/22, picture 7/19, lamp 3/17, cushion 0/16, sofa 8/13, bicycle 9/10, refrigerator 2/7, shelf 1/6, sink 0/5, bed 0/2 (per world in `detector_eval.json`).
Frequent false positives in the scans: `desk` (19 in apartment_2, 12 in apartment_0), `picture`, `bed`, `shelf` and `cabinet` in ReplicaCAD (mostly counters and wall cabinets).
Overlays: `docs/figures/worlds/detector_overlays*.jpg`.

**Answer: partly.** On the flat-shaded procedural world the detector finds 0 of 12 visible objects (no occlusion filter: 2 of 151) and its 14 detections are almost all wrong. In the textured worlds it finds 23 % of the visible objects and the
false-positive rate is 0.53 against 0.71. Distinct-looking classes (bicycle, sofa, chair) are found; doors, windows, tables, cushions are essentially not. The `elevator` / `mirror` flood of `semantic_map.md` does not appear, but this vocabulary has no elevator, so that comparison is not like for like.
The remaining false positives are confusions between similar furniture classes, not hallucinated rooms. The real-scan worlds remove the "garbage labels on flat shading" problem; they do not make open-vocabulary recall high.
Limits: 20 frames per world, one detector setting (no confidence sweep), one procedural world with few visible objects, a camera at 0.55 m that sees mostly legs and low items, occlusion judged on the 2D map only.

## 5. Episodes
Generated for every real-scan world (`gen-episodes --n-object 30 --n-pose 20 --n-follow 12`), object_nav / goal_pose / velocity_follow: replica_apartment_0 28/20/12, apartment_1 12/20/12, apartment_2 13/20/12, frl_apartment_0 8/20/12,
hotel_0 6/20/12, office_0 none (reachable 3.4 m2), room_0 0/17/4, replicacad_apt_0 8/20/12, apt_1 22/20/12, apt_2 22/20/12, apt_3 12/20/12, habtest_apartment_1 0/20/2, van_gogh_room 0/4/1.
`validate`: 28 worlds, 81 episode sets, 1687 episodes, 0 problems; pytest 46 passed (unchanged suite; new adapters have no unit tests).
Generator fixes: the tier search no longer stalls on a distance tier that does not exist in a small world; `robot_radius_m` per world in the registry is honoured. The Habitat ObjectNav / R2R import path is unchanged.

## 6. Not done / open
* Optional rough world of the real lab from the bag cloud: not started.
* Remaining Replica scenes and skokloster not converted (see section 2).
* ReplicaCAD memory (7 GB steady, up to 14 GB) is unexplained; legged mode and long runs not tested on any new world.
* Habitat test scenes carry no ground truth; `habtest_apartment_1` duplicates Replica apartment_1.
* Object boxes are axis-aligned; `blocks_robot` is a heuristic; there is no `rooms.json` for Replica (no region labels in the files used).
* The Replica terms and the ReplicaCAD licence discrepancy matter before any redistribution (non-commercial only).
* `x30-sim` container: `share/x30_worlds/{worlds,models}` of its installed workspace were replaced by symlinks to the source tree (originals kept as `worlds_orig`, `models_orig`).
