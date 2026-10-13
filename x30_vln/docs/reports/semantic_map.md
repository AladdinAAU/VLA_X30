# Semantic map report (x30_semantic_map)

Status 2026-10-09. Package `ros2_ws/src/x30_semantic_map` (README there), config `configs/semantic_map.yaml`, figures `docs/figures/semantic_map/`,
raw numbers `docs/reports/semantic_map_{integration,live_vlm,live_vlm_v1_prompt,sim_chain}.json`. Everything below was run unless marked "not tested".

## What exists
| Part | State |
|---|---|
| Core (pure python, no ROS): association, merging, forgetting, dynamic objects, ray triangulation, persistence, rooms, relations, finder, summary, annotated map, annotator | done, 58 unit tests pass on the host and in the x30-framework image (Python 3.10, ROS python env) |
| ROS node `semantic_mapper`, launch, package files | done; topics and the 7 documented services plus `render`, `feedback`, `status` |
| Annotator against a fake OpenAI server | done (17 tests: validation, retries, circuit breaker, budget, caching, `find` timeout fallback) |
| Annotator against the real Qwen3-VL 8B server | measured, section 2 |
| Chain sim -> perception client -> mapper | comes up and runs; quantitative recall is not meaningful, section 3 |

Interfaces follow `INTERFACES.md` section 4 and the requests `perception-to-semantic_map.md` / `modes-to-semantic_map.md`:
the mapper consumes only `/perception/detections`; `/semantic_map/query` returns `{"count", "objects":[{id,label,confidence,position:[x,y,z],size:[..],bearing_only,caption,attributes,room,last_seen,...}]}`
and `/semantic_map/summary` returns `{"text", ...}`; bearing-only objects are flagged and never confirmed without triangulation.

## 1. ROS integration test (x30-framework, ROS_DOMAIN_ID 77, `scripts/run_ros_integration.sh`)
A scripted publisher plays the synthetic apartment as `/perception/detections` at 5 Hz (noisy duplicates, a walking person, bearing-only rays,
false positives, a chair that is moved, NaN thermal values) together with TF, `/map` and `camera_info`; the node runs with the VLM off.
31 checks, all pass (`RESULT PASS`, run twice; second run after the prompt change below):
* map: 18 objects, recall 0.90 (18/20 ground-truth objects), mean position error 0.045 m (median 0.039), 0 duplicate instances, the moved chair is gone from its old place,
  at most one person instance, no thermal attributes without thermal data, 'refrigerator' gets room 'kitchen', 3 regions named (corridor, kitchen, living room);
* 8 of 146 bearing-only rays were triangulated into objects;
* services return the documented JSON: `query` (label, attribute, near/radius), `find` (heuristic), `relations`, `summary` (390 tokens, with rooms), `render` (image 700x500 and text),
  `feedback`, `status`, `save` (24 objects), `reset`, `load` (24 restored, queries work afterwards); error requests (`find` without text, unknown id, missing file) return `ok=false` with `{"error": ...}` and the node stays alive;
* mapper counters: 639 frames, 1130 detections, 64 objects created, 15 merges, 20 expired unconfirmed, 3 removed as dynamic.

Caveat: this uses the repo's own synthetic generator, so recall/error describe the mapper's logic under modelled noise, not the detector.

## 2. Live annotator against Qwen3-VL-8B (llama.cpp, Q4_K_M, port 8000; `test/live_vlm_check.py`)
Sample: 15 hand-cropped objects from two rendered indoor scenes (AWS RoboMaker bookstore and hospital world screenshots; one image each), 6 `find` requests, 6 room-naming cases.
These are small, so treat percentages as indicative (one case = 6.7 points). The GPU was otherwise idle (perception server running, no requests).
* Latency per call (30 caption calls, 336 px crop, JSON output; over all 64 calls about 235 prompt and 42 completion tokens on average): mean 3.0 s, median 3.0 s, max 3.7 s. `find` (rank + crop re-check, 2-3 calls): 4.8-7.7 s.
  Captioning all 15 objects of the map sequentially took 39 s.
* Captions: 15/15 answered, 0 failed calls of 64; keyword plausibility 15/15 ("Tall silver refrigerator ...", "Person wearing blue outfit ...");
  VLM colour plausible in 9/11 cases where a plausible set was defined.
* Label check (is the detector label right?): the true label was confirmed 15/15. A deliberately wrong label was rejected only 8/15 (53 %) with the final prompt (6/15 = 40 % with the first prompt,
  `semantic_map_live_vlm_v1_prompt.json`). The model's own `label` is mostly right (stool, bookshelf, fridge, bed); misses are of two kinds: "unclear" on small crops (5 cases, treated as match by design)
  and echoing the supplied wrong label (tv, bed). The change that gained 2 cases: the prompt asks for the model's own name first and only then for `label_matches`. Consequence: the VLM label vote is a weak
  veto, the mapper only demotes after `rejected_label_votes` bad votes, so it will rarely remove a wrong detection by itself.
* Dominant colour from pixels agrees with the VLM colour in only 2/15 crops (backgrounds and shading dominate small crops); VLM colour is the better attribute, pixel colour is a fallback only.
* Room naming from object lists: 6/6 accepted (names from object lists such as fridge + sink + microwave -> kitchen; accepted sets are in `ROOM_CASES` of the script).
* `find` on the captioned map: top-1 correct 6/6 with the VLM ranking versus 3/6 with the heuristic only (the heuristic cannot answer "something to sit on" or "where can I get a cold drink").
* Fallback with the endpoint down (nothing listening on the port): the circuit breaker opens after the first connection error ("mapper VLM unavailable ... pausing annotation for 15 s"),
  the map keeps all 15 objects, `find` answers with the heuristic ranking immediately (0.0 s, 2 matches). Also covered by `test_circuit_breaker_opens_and_recovers_and_map_keeps_working` and `test_find_times_out_into_the_heuristic_ranking`.
* Not tested: the ROS node with `vlm_enabled:=true` against the live server (the node uses the same `Annotator` class, so only the wiring is unverified); the optional small `vlm2` model; crops from real X30 camera images.

## 3. Simulator chain (x30-sim `lab_01` kinematic -> perception server -> `perception_client` -> `semantic_mapper`, ROS_DOMAIN_ID 40; `test/sim_chain_check.py`)
The chain comes up with no code changes: sim (Jazzy) and framework (Humble) containers discover each other over CycloneDDS on `x30_net`; the client reached the perception server (about 2.3 detection messages per second; the sim report gives a real-time factor of 0.74 for `lab_01` kinematic), a static TF `map->odom` (identity, the sim publishes `odom->base_link`) was enough, and the mapper built objects and answered `query` and `summary`.
What does not work yet is a quantitative comparison with `/gt/objects`:
* **The detector does not see the simulated lab as a lab.** The world is built from flat-shaded primitives (`docs/figures/semantic_map/sim_lab01_camera.jpg`). In the last run 699 detections in 247 s were 331 `elevator`,
  85 `window`, 81 `mirror`, 43 `table`, 42 `chair`, 32 `bed`; the lab has no elevator, mirror or bed, and the real classes (cabinet, server rack, laptop, fire extinguisher, whiteboard) were almost absent.
  The mapper merged the 331 elevator sightings into one instance, so no duplicates, but it keeps that wrong object. Filed in `docs/requests/semantic_map-to-perception.md`.
* **My test driver could not cover the lab.** With only a lidar clearance check (no foot-level sensing, no navigation stack) the robot wedged against obstacles within 1-2 minutes in most of five attempts
  (it covered 13-87 m of path); one 150 s stretch before leaving the building produced 413 detections. Navigation of the sim is the job of the navigation/modes areas; I did not debug it further.
* Numbers of the last run (spawn moved to the densest corner, 247 s, 13 m of path): ground-truth objects within 4 m of the path 26, of which 6 had a label that the map produced at all; recall 2/26 = 0.08 (lower bound, no occlusion test),
  matched position error 0.36 m (n = 2; an earlier 400 s run: 3 matches, mean 0.17 m), precision 0.4 (2 of 5 map objects have a same-label ground-truth object within 1.5 m). These numbers measure the detector on simulated frames
  and the driver, not the mapper, and should not be quoted as mapper accuracy. Use section 1 for the mapper.
* Not tested: the chain with `/map` (no SLAM running, so no rooms), `depth_source: sim_depth`, the legged backend, `apt_01`, a real RGB bag (the recorded bag has only lidar).
  Needed to make this test meaningful: an "oracle detections" publisher derived from `/gt/objects` and the camera pose (sim or eval area), or a world with textured meshes.

## Known limits and design caveats
* Position semantics: perception centroids are on the visible surface (docs/reports/perception.md); the mapper stores them as received. A bias of about half the object depth towards the robot is expected for large objects. There is no compensation.
* An unconfirmed object needs `confirm.min_obs: 4` sightings inside `tentative_ttl_s: 20`. At the measured 2.3 Hz a fast pass leaves true objects below it. In one sim run with 490 detections 111 objects were created and 88 expired unconfirmed;
  I could not separate hallucinations (intended to expire) from true objects lost to the rule in that data. If the real perception rate is below 2 Hz or the robot passes quickly, lower `min_obs` for large static categories.
* VLM label check is a weak veto (above). Captions of crops below about 40 px are "unclear".
* Thermal attributes are off by default (stock X30 has no thermal camera) and were only tested with synthetic statistics, not with the real `thermal.attributes_from_stats` rules.
* Rooms come from free-space geometry of `/map` and object rules; the segmentation was tested on the synthetic apartment only.
* Memory/CPU of the node on long runs and with hundreds of objects were not measured; the persistence format has a version field but no migration code.
* Windows hosts: write Python-edited shell scripts with LF line endings (a CRLF `container_check.sh` fails with `set: pipefail: invalid option name` inside the container).

## How to reproduce
```
scripts/run_ros_integration.sh                       # section 1 (about 4 min incl. pytest), container x30-semmap-test, volume x30-semmap-ws
powershell scripts\x30.ps1 up models                 # then: cd ros2_ws/src/x30_semantic_map && python test/live_vlm_check.py --out live.json   (section 2); powershell scripts\x30.ps1 down models afterwards
# section 3: x30-sim container with world:=lab_01 locomotion:=kinematic (ROS_DOMAIN_ID=40) + a framework container on x30_net with
#   ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id odom ; ros2 launch x30_perception perception.launch.py use_sim_time:=true ;
#   ros2 run x30_semantic_map semantic_mapper --ros-args -p use_sim_time:=true ; python3 test/sim_chain_check.py --duration 300
```

## Next steps
1. Oracle detections from `/gt/objects` (project through the camera model, add the measured detector noise) to score the mapper in simulation without the detector.
2. Run the node with `vlm_enabled:=true` in the sim chain and measure the effect on `find` (live check only used the library).
3. Tune `confirm.min_obs` and `ttl` on real camera frames once available; calibrate the camera-lidar extrinsics first (perception report).
4. Add a surface-to-centre offset option for large objects if navigation targets turn out to stop short.
