# Perception report (x30_perception)

Status 2026-10-09. Everything below was run unless marked "not tested". Raw numbers: `results/perception/*.json`, overlay `results/perception/overlay_synthetic.png`.
Package: `ros2_ws/src/x30_perception` (README there), config `configs/perception.yaml`, requests in `docs/requests/perception-to-{infra,semantic_map,hw}.md`.

## 1. What exists

| Part | State |
|---|---|
| `perception_server` (FastAPI, port 8100): `POST /segment`, `GET /health`, `POST /vocab`, `POST /embed` (text only), backends `yoloe` (default), `yolo_world`, `fake`, SAM refiner hook, per-phrase text-embedding cache, enlarged-region passes, RLE masks, timing breakdown | works with YOLOE-11l-seg on the RTX 3090 Ti |
| `perception_client` ROS node: async, rate-limited, drops when busy; lidar lifting; thermal statistics; `/perception/detections`, `/detections_image`, `/diagnostics`; service `/perception/vocabulary`; launch + parameter YAML | works end to end against the synthetic stream with the real server |
| `lidar_projector` (pure numpy): N-scan accumulation, per-time-slice motion compensation through TF, pinhole/plumb_bob/equidistant, occlusion test, depth-gap split, 3D connectivity filter, median centroid, extent, `bearing_only` | unit-tested + swept (section 4) |
| Thermal helpers: mono8 -> deg C, mask -> thermal pixels, mean/max, rule-based attributes from `configs/perception.yaml` | tested on synthetic thermal images only |
| `depth_source: sim_depth` ablation, `none` | sim_depth tested on the synthetic stream |
| Tests | host: 17 passed, 6 skipped (no fastapi); framework image: 16 passed, 7 skipped; perception image: 23 passed (incl. RLE cross-check against pycocotools). `ruff` (E,F,W, 130 cols) clean |

Real-robot facts honoured (digest section 6): one wide-angle camera of unknown intrinsics (read from `camera_info` or `calibration_file`, equidistant model supported), no stock thermal camera (the node never waits for `/thermal/*`, values are NaN without it), no depth camera (`depth_source: lidar` is the default), lidar `lidar_link` cloud with per-point `timestamp` in milliseconds and header stamp = scan start (verified on the bag).

## 2. Server latency and memory (RTX 3090 Ti, YOLOE-11l-seg fp16, 74 phrases, 640x480 input, HTTP from the host, `scripts/bench_server.py`, n=20)

The GPU was shared with other agents' jobs during the runs (other processes held about 9-11 GB), so treat as indicative.

| imgsz | image | detections (conf 0.1) | RTT median / p95 (ms) | server total (ms) | of which inference (ms) |
|---|---|---|---|---|---|
| 640 | bus.jpg | 9 | 27.5 / 80.9 | 22.9 | 19.6 |
| 640 | zidane.jpg | 2 | 23.1 / 33.8 | 19.6 | 18.0 |
| 960 | bus / zidane | 8 / 2 | 31.2 / 28.8 | 24.9 / 23.6 | 21.8 / 21.8 |
| 1280 | bus / zidane | 7 / 3 | 30.1 / 34.0 | 25.4 / 26.0 | 22.3 / 23.9 |

- 1280 px costs about +4 ms over 640 px on this GPU; response about 3-4 kB (RLE masks are small). The detection count changes with imgsz (7-9 on the bus photo), so imgsz is a recall setting, not just latency.
- An extra enlarged-region pass (`regions`, 1280 px crop) adds about 25 ms (RTT 50 ms vs 28 ms).
- Changing the vocabulary (new task phrase, or going back) costs one slower frame: about 365-430 ms RTT, text embedding itself 2.5 ms for cached phrases; the whole 74-phrase set is embedded once at start-up. Ultralytics' own `get_text_pe` rebuilds the text encoder on every call (2 s); the server calls the internal one with `cache_clip_model=True`.
- GPU memory: about 1.2 GB (nvidia-smi delta after load) to 1.6 GB (after the 640/960/1280 sweep); torch allocated 435 MB, peak 980 MB. Start-up with warm-up about 40 s.
- Measured on the host the request through `localhost` took 2 s longer than through `127.0.0.1` (IPv6 attempt first): use `127.0.0.1` from the Windows host; containers use the service name.

## 3. Lidar density for a virtual head camera (real bag `lab_2026-10-06`, 178 scans, `scripts/lidar_coverage.py`)

Virtual 640x480 pinhole camera, 90 deg HFOV (fx = 320 px), placed at the `lidar_link` origin looking along +x (the real camera pose is unknown), own body removed with `min_range_m` 0.6. Scans merged without motion compensation (0.3 s, robot nearly static in the bag).

Facts about the cloud: 80 k points/scan, of which 62 % lie within 0.6 m of the sensor (the robot's own body seen by the four lidars), leaving about 30 k useful points. Elevation distribution: 1st percentile -10.5 deg, median +16.7 deg, 99th percentile +77 deg: **there are no returns below about -10.5 deg**, so a horizontal camera gets no lidar points in image rows below about row 300 of 480 (the lower 37 % of the image). Floor-standing objects are only liftable where they extend above that line, e.g. beyond about 2.8 m for a sensor 0.5 m above the floor (sensor height not documented).

Points per image, front view (yaw 0): 7 474 (one scan), 22 425 (three scans); rear view (yaw 180): 6 544 / 19 630. Pixels with at least one point: 2.3 % (1 scan), 6.5 % (3 scans). Range of those points in front (3 scans): 0.6-1 m 6 205, 1-2 m 5 240, 2-4 m 9 813, 4-7 m 900, 7-12 m 190, >12 m 73 (the lab is small; far density is limited by the room, not by the scan pattern).

Angular density, points per pixel, 3 scans, image split in 4x4 cells (rows top to bottom):

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| row 1 | 0.072 | 0.056 | 0.045 | 0.110 |
| row 2 | 0.074 | 0.094 | 0.084 | 0.109 |
| row 3 | 0.071 | 0.185 | 0.187 | 0.081 |
| row 4 | 0.000 | 0.000 | 0.000 | 0.000 |

A lidar samples directions, so points per pixel does not depend on distance; the density on a surface is `points_per_pixel * (f / d)^2`. With the central density 0.137 px^-1 (rows 2-3, cols 2-3) and 3 scans:

| distance | pixel footprint | points per m^2 | side of a square patch with 5 points | with 20 points |
|---|---|---|---|---|
| 1 m | 3.1 mm | 14 079 | 1.9 cm | 3.8 cm |
| 3 m | 9.4 mm | 1 564 | 5.7 cm | 11.3 cm |
| 5 m | 15.6 mm | 563 | 9.4 cm | 18.8 cm |
| 10 m | 31.3 mm | 141 | 18.8 cm | 37.7 cm |

In image pixels: a mask of about 6x6 px (central) to 10x10 px (sparsest cell, 0.045 px^-1) holds 5 points; 12x12 to 21x21 px holds 20. **Practical rule from this and from the synthetic sweep:** with three scans a mask of at least about 20 px side (about 400 px^2) gives a usable estimate; below about 10 px side (or with one scan, three times more) expect `bearing_only`. A 0.4 m object is therefore liftable out to roughly 8-10 m; a 0.2 m object to roughly 5 m. A single scan has one third of these densities. Caveat: the density comes from one 17 s recording in one room; it is a lower-quality estimate for far objects and it ignores the motion-induced smear that real use has.

## 4. 3D position accuracy

### 4.1 Synthetic world, ground-truth masks (`scripts/synthetic_accuracy.py`, no segmenter, flat targets, perfect calibration)
Virtual lidar 80 k points/scan, uniform over -7..52 deg elevation (Livox-like FOV, not its scan pattern), 2 cm range noise, 3 scans, camera 640x480/87 deg.

| object | distance | mask px | points used | position error (m) |
|---|---|---|---|---|
| 0.2 m square | 1.5 / 3 / 5 / 8 / 12 m | 4556 / 702 / 210 / 72 / 36 | 574 / 106 / 51 / 26 / 14 | 0.008 / 0.008 / 0.017 / 0.030 / 0.032 |
| 0.4 m square | 1.5 / 3 / 5 / 8 / 12 m | 18090 / 2862 / 870 / 306 / 144 | 2348 / 542 / 154 / 82 / 36 | 0.005 / 0.014 / 0.011 / 0.023 / 0.035 |
| 0.8 m square | 3 / 5 / 8 / 12 m | 11556 / 3540 / 1260 / 576 | 2327 / 812 / 270 / 114 | 0.009 / 0.023 / 0.019 / 0.030 |

No `bearing_only` cases in this sweep (the smallest, 36 px at 12 m, still had 14 points). Per-frame numpy cost on one host core: projection of 240 k merged points 50 ms, 3.7 ms per object.

Sensitivity (4 flat objects at 3-8 m): horizontal pointing error of the camera-lidar extrinsic 0 / 0.5 / 1 / 2 deg -> median position error 0.015 / 0.032 / 0.048 / 0.083 m, points kept 778 / 767 / 716 / 622. Robot moving at 0.6 m/s and turning 0.5 rad/s: motion-compensated median error 0.014 m (max 0.30 m, an object partly hidden by another one); uncompensated, one pose per scan: 0.023 m median but **max 5.1 m** (points of a different surface get into the mask).
**Failure mode found:** a 0.3 m wide post at 3 m in front of a wall at 14 m lifts correctly up to 4 deg of pointing error (0.15 m error at 4 deg), but at 8 deg the mask collects only wall points and the object is placed 11 m away. The depth-cluster rule prefers the nearest cluster with at least 40 % of the biggest cluster's points; a thin object whose points all miss the mask has no nearer cluster to prefer. Real extrinsics must be calibrated to better than about 2 deg.

### 4.2 Full chain with the real YOLOE server (synthetic stream, `scripts/smoke_test.py`, 30 s, PASS)
Scene: photo billboards (two people, a bus photo with passengers) at 3.0, 4.6 and 6.0 m; robot moving 0.4 m / 0.15 rad; thermal camera simulated. YOLOE detects the people (0.9) and, after adding the phrase `bus` through `/perception/vocabulary`, the bus (0.75); it also detects the passengers inside the bus photo as people. Position error is measured against the billboard plane (the surface the lidar sees):

| object | detections | median / max distance from plane | points (median) | thermal GT / measured |
|---|---|---|---|---|
| person at 3.0 m | 78 | 1 mm / 2 mm | 983 | 34 / 34.0 C |
| bus at 6.0 m | 38 | 1 mm / 2 mm | 627 | 21 / 21.1 C |
| person at 4.6 m | 78 | 1 mm / 4 mm | 286 | 34 / 34.0 C |

The plane error is a lower bound: flat targets, exact calibration, TF without latency problems. The lateral distance from the billboard centre (0.37-0.41 m median) just reflects that YOLOE masks cover part of the photo. Same run with `depth_source: sim_depth`: 3-5 cm median plane error (up to 0.22 m max): this path depends on the camera pose at the image stamp, so it exposes the (unisolated) timing/TF accuracy of the synthetic publisher, while the lidar path is insensitive to pose error along a plane. Both runs: 55-61 frames in 30 s on a loaded machine (about 2 Hz at `rate_hz: 2.5`), `frame_id: map`.

Client timings from the diagnostics topic (shared 24-core machine, other agents running): server RTT 29-48 ms, lifting 56-93 ms (3 scans x 60 k points, one core), total process 96-147 ms, image-stamp to publish 450-780 ms. The latter is dominated by the synthetic renderer (0.2-0.5 s per image before it is published) and by the rate limiter, not by the client: with a real camera expect camera latency + up to 1/rate_hz + about 0.15 s.

### 4.3 Not measured
- **Gazebo / `/gt/objects`: not tested** (the simulator image with camera, depth and lidar was not available to me). Run: `ros2 launch x30_perception perception.launch.py use_sim_time:=true`, compare `/perception/detections` with `/gt/objects` after yaw alignment, once with `depth_source: lidar` and once with `sim_depth`.
- Real camera images: none exist (the bag has only `/lidar_points`), so detection quality on the X30 camera, indoor class thresholds, and real extrinsics are untested.
- The scan pattern of the real lidar was not simulated (uniform random directions instead), and the real 4-lidar merge may have per-lidar timing differences.

## 5. What works for which classes (honest)
- Evidence is thin: two photographs (bus.jpg, zidane.jpg) and synthetic scenes. YOLOE-11l with text prompts finds people (score 0.8-0.93), the bus (0.75 with the phrase added) and its passengers (0.7-0.8) reliably, even when the photo is pasted as a flat billboard.
- The indoor vocabulary and thresholds in `configs/perception.yaml` are a first guess and **untuned**. From the drone project (outdoor, same model family): large structures (buildings 0.05-0.3) scored far below small objects (vehicles 0.5-0.75), so doors, windows, stairs, elevators, pillars and carpets have lower thresholds (0.12-0.15) in the config; far small objects need 1280 px or the `regions` pass (both cheap here). These expectations are carried over, not measured indoors.
- Geometry limits regardless of class: objects must be at least about 20 px across (3 scans) to be lifted, low floor-level objects within about 3 m fall below the lidar's lowest beam (-10.5 deg), glass and mirrors follow lidar behaviour (not tested), moving objects (people) smear over 0.3 s of accumulated scans, and the centroid is on the visible surface (for a 0.5 m deep object about 0.25 m closer than its centre).

## 6. Optional backends
- `yolo_world` (yolov8s-worldv2) and the SAM refiner (`mobile_sam.pt`) install and run through the same interface: in a throwaway container YOLO-World returned 5 boxes on the bus photo in 1.1 s (first call, includes CLIP download of 338 MB) and the refiner produced 5 masks. Neither is accuracy-compared with YOLOE; the weights are not in `/cache` (first use downloads them into the working directory). `/embed` answers 501 for images (the YOLOE bundle has only the text tower); text embeddings work.

## 7. Known issues and next steps
1. Run the chain in Gazebo against `/gt/objects` and on a real bag with camera frames once recorded; tune thresholds per class; decide imgsz 640 vs 960 on real frames.
2. Calibrate camera intrinsics and camera-lidar extrinsics of the X30; until then bearing/point errors are unknown (section 4.1 gives the sensitivity). A wrong extrinsic can silently place thin objects on the wall behind them: consider a consistency check against lidar-only clustering.
3. The visible-surface bias and flat `extent` could be improved with a box fit; people smear while walking (no tracking/dynamic filtering yet).
4. Attributes from thermal data need an ambient estimate in the message (request to semantic_map) or a field in `Detection3D`.
5. The Docker image `x30-perception:latest` was not built by me; the server was tested in `drone-perception:latest` with the exact entrypoint command (`uvicorn server:app --app-dir /app/perception_server`).
6. The client decodes lazily and runs on one worker thread; at 80 k points x 3 scans it needs about 60-90 ms per frame on a loaded core; if more is needed, cull to the camera FOV before the TF transform or lower `num_scans`.
7. Test hygiene: the framework image's pytest 6.2.5 needs `-p no:anyio` (reported to infra); tests of the web part run in the perception image only.
