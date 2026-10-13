# perception -> infra

## 1. Server mount path: nothing to change
The server is the top-level directory `ros2_ws/src/x30_perception/perception_server/` (a Python package, also installed by colcon next to
`x30_perception`, which shares its RLE code with the ROS client). `server.py` in it puts `/app` on `sys.path` and imports
`perception_server.app`, so the existing entrypoint `uvicorn server:app --app-dir /app/perception_server` and the mount in
`compose.models.yml` work unchanged. Tested with the old `drone-perception:latest` image and exactly this command; I have not built
`x30-perception:latest`.

## 2. Optional: default vocabulary and weights variable
- Mount `../configs:/x30/configs:ro` into `perception` and set `PERCEPTION_CONFIG=/x30/configs/perception.yaml` so a request without a
  vocabulary falls back to the project vocabulary (the ROS client always sends its own, so this only matters for manual curl tests).
- `YOLOE_WEIGHTS` from `.env` is honoured (`PERCEPTION_WEIGHTS` takes precedence).
- Docker Desktop on Windows: clients on the host must use `http://127.0.0.1:8100`; `localhost` costs +2 s per request (IPv6 first). Worth a line in docs/INFRA.md.
- Optional backends (`PERCEPTION_BACKEND=yolo_world`, `PERCEPTION_REFINER=sam`) download weights on first use (`yolov8s-worldv2.pt` 25 MB, CLIP ViT-B/32 338 MB, `mobile_sam.pt`)
  into the working directory = `/cache` when run via the entrypoint. Not needed for the default YOLOE setup.

## 3. pytest in the framework image
`pytest 6.2.5` in `x30-framework` conflicts with the `anyio` pytest plugin (installed with httpx): `pytest` fails at start-up with
`No module named '_pytest.scope'` unless run as `pytest -p no:anyio`. `x30-test` may want that flag.
