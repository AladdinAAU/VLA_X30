# x30_semantic_map

Persistent 3D semantic object map for the X30 VLN framework, a model-based annotator (mapper VLM) and a MapNav-style
annotated top-down map. ROS 2 Humble, Python 3.10. The core is pure python and does not import ROS.

## What it does
- Consumes only `/perception/detections` (`x30_msgs/Detection3DArray`, `map` frame) plus `/map`, TF and optional
  `/camera/color/camera_info`; builds object instances with label votes, a robust position/size estimate, sighting
  counts, best views (3 distinct), colour name, size class, optional thermal attributes (off by default: the stock X30
  has no thermal camera, NaN values are ignored), moving/stationary state for people.
- Association: label compatibility (synonym groups and categories from `configs/semantic_map.yaml`), 3D distance gated by
  object size, box IoU, recency; one-to-one per frame; same-frame double labels (bottle + cup) become label votes.
- Forgetting: unconfirmed objects expire; confirmed static objects are removed when the camera looked at their place
  (field of view from `camera_info`, line of sight on `/map`) and saw nothing for `forget.miss_limit` counted frames;
  people expire after a ttl; rejected/expired places are remembered and need extra observations to come back.
- Bearing-only detections are kept as rays and triangulated (weighted least squares, uncertainty and chi-square checks,
  line of sight) when rays from different poses agree; a later 3D detection upgrades or absorbs the triangulated object.
- Rooms: free space is eroded by the door clearance, cores are regions; names from object rules, shape (corridor) or the VLM.
- Spatial relations: near / next to, on / under (vertical gap + footprint overlap), left / right / in front / behind
  (egocentric or as seen from a viewer pose), between.
- `summary`: nearest-first text with distance and bearing relative to the robot, rooms, captions, unexplored directions
  (frontiers of `/map`), trimmed to a token budget.
- Annotated map: raster (free / occupied / unknown, tinted named rooms, trajectory, robot arrow with heading, object
  footprints with `id label`, optional captions, scale bar) north-up or robot-centred, and a character-grid text variant.
- Annotator (`annotator.py`, asynchronous): captions + attributes + label check per object crop, room naming from objects
  plus a crop montage, doubtful-pair disambiguation (merge or keep apart), `find` re-ranking with crop re-checks. Rate and
  budget limited, cached by (task, object, view hash), retries + one repair retry, validated JSON, circuit breaker, JSONL
  log of every call (prompt hash, latency, tokens). Without a VLM everything else works and `find` uses `finder.py`.
  Prompts are files in `x30_semantic_map/prompts/`.

## Run
```
ros2 launch x30_semantic_map semantic_mapper.launch.py use_sim_time:=false vlm_enabled:=auto   # auto = on when configured
ros2 run x30_semantic_map semantic_mapper --ros-args -p config_file:=/x30/configs/semantic_map.yaml
```
Config: `configs/semantic_map.yaml` (found through `config_file`, `$X30_SEMANTIC_MAP_CONFIG`, `$X30_CONFIG_DIR`,
`$X30_CONFIGS`, or a `configs/` folder above the source). The VLM endpoint is `mapper_vlm` in `configs/models.yaml`
(`X30_MAPPER_URL`, `X30_MAPPER_MODEL`, ... override), used through `x30_common.VLMClient`. Mapper log of every VLM call:
set `vlm.log_path`.

## Interface
Topics: `/semantic_map/objects` (`SemanticObjectArray`, 1 Hz, transient local; confirmed and triangulated objects),
`/semantic_map/markers`, `/semantic_map/annotated_map` (`Image` rgb8, only rendered when subscribed).
Services (`x30_msgs/srv/Json`, request and response are JSON strings; `ok=false` carries `{"error": ...}`):

| Service | Request | Response |
|---|---|---|
| `query` | `{label, near:[x,y], radius, attributes:[..], room, limit, include_tentative, min_confidence}` | `{count, objects:[...], robot_pose}` |
| `find` | `{text, limit, use_vlm, timeout_s}` | `{method: vlm\|heuristic, matches:[{id,label,score,reasons,position,...}], parsed, vlm_calls, latency_ms}` |
| `relations` | `{id, pose:[x,y,yaw], radius, limit}` | `{relations:[{id,label,dist_m,relations,seen_from_robot}], between, from_robot, text}` |
| `summary` | `{pose, max_objects, token_budget, include_tentative}` | `{text, tokens_est, objects_shown, objects_total, truncated, objects}` |
| `reset`, `save`, `load` | `{}`, `{path}`, `{path}` | `save` writes JSON plus `<stem>_crops/` |
| `render` (extra) | `{mode: image\|text, egocentric, show_captions, max_size_px, include_tentative}` | image: `{png_base64, width, height}`; text: `{text}` |
| `feedback` (extra) | `{id, action: reject\|confirm\|relabel\|merge, label, into, reason}` | `{applied}` |
| `status` (extra) | `{}` | counters, regions, annotator state and VLM call statistics |

`SemanticObject.confidence` = mean detector score x (1 - exp(-observations / 3)) x label vote purity (x 0.6 for
triangulated objects). `attributes` are plain words (`red`, `large`, `plastic`, `closed`, `warm`, `person-like`, `moving`).

## Tests
```
cd ros2_ws/src/x30_semantic_map && python -m pytest test -q          # 58 tests, no ROS, no GPU (fake OpenAI server inside)
scripts/run_ros_integration.sh                                       # builds + tests + node + ROS integration in x30-framework (31 checks)
python test/live_vlm_check.py --url http://localhost:8000/v1         # live annotator measurement against the running VLM (never starts one)
python3 test/sim_chain_check.py --duration 300                       # inside a framework container with sim + perception + mapper running
```
`run_ros_integration.sh` starts one throw-away container `x30-semmap-test` (removed on exit) and keeps its build in the
named volume `x30-semmap-ws` (`docker volume rm x30-semmap-ws` to delete). It writes `docs/reports/semantic_map_integration.json`
and `docs/figures/semantic_map/asm_ros_integration.png`. `SEMMAP_RATE` (default 5 Hz) sets the playback
rate of the synthetic detections; replaying faster than real time shortens the time between observations and changes the
counted-observation logic, so keep it at 5 for numbers that mean something.
`sim_chain_check.py` needs the simulator (`x30-sim`, world `lab_01`), the perception server, `perception_client` and `semantic_mapper`
with `use_sim_time` and a static TF `map->odom`; see `docs/reports/semantic_map.md` section 3 for what it can and cannot tell you.
Shell scripts must keep LF line endings.

## Known limits
See `docs/reports/semantic_map.md`.
