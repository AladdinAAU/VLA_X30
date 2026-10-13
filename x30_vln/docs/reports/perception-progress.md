# Perception progress notes

- M1 (done): pure modules (camera model, lidar projector, thermal, RLE, config), synthetic world, 16 host tests pass.
- M2 (done): real bag coverage (results/perception/coverage_*.json); server (YOLOE) running in container x30-perception-test; bench in results/perception/bench_server.json (use 127.0.0.1, not localhost: 2 s IPv6 delay on this host).
- M3 (in progress): client node + launch + synthetic stream built in /tmp/ws of container x30-perception-ros (ROS_DOMAIN_ID 77); next: unit tests in container, smoke test, accuracy numbers, README, docs/reports/perception.md, requests file for INFRA (mount path).
- M3 (done): client + launch + synthetic stream run in container x30-perception-ros with the real YOLOE server; scripts/smoke_test.py PASS (plane error ~1 mm on planar billboards, thermal OK, vocabulary service OK), results/perception/smoke.json. Unit tests: 17 pass on host, 16 pass + 7 skipped (no fastapi) in framework image, 23 pass in perception image.
- Remaining: accuracy-vs-distance script, optional backends (yolo_world/sam) quick attempt, README, docs/reports/perception.md, requests/perception-to-infra.md, ruff cleanup.
- DONE: all deliverables written; see docs/reports/perception.md. Containers x30-perception-test/-ros removed.
