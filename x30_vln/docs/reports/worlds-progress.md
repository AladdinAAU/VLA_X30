# WORLDS progress notes
- M1: core pkg (transforms, sdfgen, sdfparse, gridmap), procgen (10 worlds generated + registered), episode generators + schema: done, run.
- M2: scene2sdf adapter run on Khronos Sponza (Y-up gltf -> Z-up, 667 collision boxes, 239 m2 navigable). Next: habitat adapters, importers, CLI, tests, AWS, Gazebo validation.
- M3: habitat/mp3d adapters + importers + synthetic data + CLI + 46 pytest tests passing. Next: AWS conversion, procgen episodes for all worlds, gazebo validation, docs.
- M4: AWS worlds converted (4), Fuel models fetched (43, CC-BY-4.0), episodes generated for 15 worlds, validate OK. Next: ROS pkg files, gazebo validation (x30-sim image), docs, report.
- M5 (refocus to real-scan worlds): downloaded ReplicaCAD (153 MB, LICENSE.txt says CC BY-NC 4.0, HF card says CC BY 4.0), Habitat test scenes (94.6 MB zip, HF card cc-by-nc-4.0), Replica release parts aa..aq (33.9 GB, all present, NOT yet extracted; terms say access = agreement, research/non-commercial) under datasets/external/{replicacad,habitat_test,replica_src}. Do not re-download.
  Code: x30_datasets/budget.py (decimate + atlas re-bake, ceiling split), adapters/replicacad.py, scene2sdf extended (scene=, budget_tris, tex_max, object_models, spawn), sdfgen object_model_xml + xmlns:x30, labels.yaml +27 classes. 46 pytest pass.
  Converted: replicacad_apt_0 (147k tris, 1381 collision boxes, 118 GT objects, reachable 18.3 m2 at r=0.35 of 47 m2 free). Not yet registered/simulated.
  Next: extract Replica (exclude textures/ ptex), replica adapter (vertex colours -> bake), register worlds, simulator check, detector check, episodes, docs.
- M6: converted replicacad_apt_0/apt_2, habtest_apartment_1, habtest_van_gogh_room, replica_apartment_0 (scripts/convert_*.py, registered). Replica extraction of remaining scenes in progress (datasets/external/replica_src/extract.log). Next: sim check.
- M7: sim check + detector check done for replica_apartment_0, replicacad_apt_0, apt_01 (see results/worlds_check/*/detector_eval.json). Next: convert more Replica scenes, episodes, docs.
- M8: 13 real-scan worlds, episodes, sim + detector checks, docs/WORLDS.md, docs/reports/worlds.md written. Remaining: gallery.
