# x30_perception

Open-vocabulary instance segmentation for the X30 VLN framework, with 3D positions obtained by projecting the lidar into the
camera image (the robot has no depth camera) and optional thermal statistics. Contract: `docs/INTERFACES.md` sections 1, 2, 4.
Measured results and limits: `docs/reports/perception.md`.

```
 /camera/color/image_raw ─┐                       ┌── HTTP POST /segment (JPEG + vocabulary) ──▶ perception_server (GPU, YOLOE-seg)
 /camera/color/camera_info ┤  perception_client ───┤◀── instances: label, score, box, COCO-RLE mask, timing
 /lidar_points ───────────┤  (rate-limited,       └── per mask: lidar_projector (project, occlusion test, depth-cluster,
 /thermal/* (optional) ───┤   async worker)           3D clustering, median) + thermal stats ──▶ /perception/detections
 TF (odom, map, sensors) ─┘                                                                    /perception/detections_image
                                                                                               /perception/diagnostics
```

## Contents
| Path | What |
|---|---|
| `perception_server/` (top-level package, installed next to x30_perception) | FastAPI model server (self-contained, no ROS): `app.py`, `backends.py` (yoloe, yolo_world, fake, SAM refiner), `pipeline.py`, `rle.py`, `server.py` (uvicorn entry point for the infra image) |
| `x30_perception/perception_client_node.py` | ROS node `perception_client` |
| `x30_perception/lidar_projector.py` | pure numpy: scan selection, motion-compensated merge, projection, occlusion test, per-mask robust 3D estimate |
| `x30_perception/camera_model.py`, `geometry.py`, `pointcloud.py` | pinhole + plumb_bob/equidistant model, 4x4 helpers, PointCloud2 decoding |
| `x30_perception/thermal.py` | mono8 -> deg C, mask -> thermal pixels, mean/max, attribute rules |
| `x30_perception/lifting.py`, `config.py`, `overlay.py`, `ros_image.py`, `server_client.py` | filtering/aliases/thresholds, vocabulary + calibration loading, debug overlay, image decoding, HTTP client |
| `x30_perception/synthetic.py`, `synthetic_stream.py` | synthetic world (photo billboards, virtual lidar) and a ROS node that publishes it |
| `config/perception_client.yaml` | all node parameters with defaults and comments |
| `../../../configs/perception.yaml` | vocabulary, aliases, per-class thresholds, thermal rules, server defaults |
| `launch/perception.launch.py` | `ros2 launch x30_perception perception.launch.py [use_sim_time:=true] [server_url:=..] [depth_source:=lidar]` |
| `scripts/` | `smoke_test.py`, `bench_server.py`, `lidar_coverage.py`, `synthetic_accuracy.py` |
| `test/` | pytest (`test/assets` holds two photos from the ultralytics package, used as textures) |

## Run
Server (GPU; in the compose stack it is the `perception` service, `docker compose ... --profile models up -d`):
```
# from ros2_ws/src/x30_perception, inside an image with torch + ultralytics + fastapi:
YOLOE_WEIGHTS=yoloe-11l-seg.pt PERCEPTION_CACHE=/cache PERCEPTION_CONFIG=/x30/configs/perception.yaml \
  uvicorn perception_server.app:app --host 0.0.0.0 --port 8100
```
Environment: `PERCEPTION_BACKEND` (yoloe | yolo_world | fake), `PERCEPTION_WEIGHTS`/`YOLOE_WEIGHTS`, `PERCEPTION_CACHE` (weights +
`mobileclip_blt.ts`), `PERCEPTION_DEVICE`, `PERCEPTION_HALF`, `PERCEPTION_REFINER=sam` (box-prompted mask refinement, request opts in
with `refine=true`), `PERCEPTION_CONFIG` (default vocabulary), `PERCEPTION_WARMUP`. On the Windows host use `127.0.0.1`, not
`localhost` (IPv6 attempt first: +2 s per request).

HTTP API (port 8100): `POST /segment {image: base64 jpeg, vocab, conf, imgsz, iou, max_det, regions, region_imgsz, masks, refine}` ->
`{instances: [{label, score, box, mask_rle: {size, counts}, mask_source}], ms, image, backend, timing: {decode, wait, vocab, infer, refine, rle}}`;
`GET /health`; `POST /vocab {vocab}` (pre-embed); `POST /embed {texts}` (text embeddings; image embeddings answer 501, the YOLOE bundle has
only the text tower). `regions` re-runs fractions of the image at `region_imgsz` and merges (small far objects).

Client (framework container): `ros2 launch x30_perception perception.launch.py use_sim_time:=true`. Needs TF
`map -> odom -> base_link -> {camera_optical_frame, lidar_link}`; falls back to `odom` (frame_id says which) without `map`, and to one
uncompensated scan without `odom -> base_link`. Without any lidar or extrinsics every detection is `bearing_only`.

Task vocabulary at run time (`x30_msgs/srv/Json`, `/perception/vocabulary`): `{"op": "add", "labels": ["red fire hydrant"], "ttl_s": 600}`,
`remove`, `set`, `reset`, `get`, `stats`. A new phrase costs one slower frame (about 0.4 s server side), later frames are normal.

## Published
- `/perception/detections` (`x30_msgs/Detection3DArray`): header stamp = image stamp, frame `map` (or `odom`), `camera_position`, per detection
  `label` (alias-resolved), `score`, `box_xyxy` (pixels of the camera image), `mask_rle` (JSON, **size = the resolution that was sent to the
  server**, equal to the image size unless `send_max_side` shrinks it), `point_count`, `centroid`/`extent` (map-frame axis-aligned, from the
  visible surface), `bearing` (map frame), `bearing_only`, `depth_source` (`lidar`|`sim_depth`|`none`), `thermal_mean_c`/`thermal_max_c` (NaN
  without thermal), `crop` (JPEG).
- `/perception/detections_image` (overlay, only produced when someone subscribes), `/perception/diagnostics` (`DiagnosticArray`: server rtt/inference,
  lift/thermal/process ms, image-to-publish latency, scans used, compensated, drops, TF fallbacks).

## Real robot notes
- The camera is a wide-angle USB camera with unknown intrinsics: use `camera_info` (equidistant or plumb_bob) or `calibration_file`
  (ROS camera_calibration YAML, optionally `lidar_to_camera_xyzrpy`). Extrinsics normally come from TF; `extrinsics_override` is the last resort.
- `/thermal/*` is optional hardware: the node never waits for it. `thermal.norm_range_c` in `configs/perception.yaml` enables the
  approximate mono8 -> deg C mapping; attribute rules are `x30_perception.thermal.attributes_from_stats` (used by the semantic mapper).
- The merged lidar cloud contains the robot's own body (62 % of points within 0.6 m in the recorded bag): `min_range_m: 0.6` removes it.
  The recorded lidar has no returns below about -10.5 deg elevation, so image regions that low are never lifted (see report).

## Tests
```
cd ros2_ws/src/x30_perception && python -m pytest test -q          # host/any image: pure parts (server tests need fastapi + httpx)
python scripts/synthetic_accuracy.py                                # accuracy sweeps, no GPU
python scripts/lidar_coverage.py ../../../datasets/real/lab_2026-10-06 --yaw 0   # needs `pip install rosbags`
python scripts/bench_server.py --url http://127.0.0.1:8100          # latency of a running server
# whole chain (server running; framework container; synthetic world):
ros2 run x30_perception synthetic_stream --ros-args -p assets_dir:=<repo>/ros2_ws/src/x30_perception/test/assets &
ros2 launch x30_perception perception.launch.py & python3 scripts/smoke_test.py --duration 30   # exit 0 = PASS
```
