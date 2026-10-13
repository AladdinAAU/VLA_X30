sim_lab01_map_vs_gt_exploring.png: /map (red occupied, blue free) over the lab_01 ground truth (dark) after 190 s of frontier exploration; precision 0.94, recall 0.79 at 0.15 m.
sim_lab01_map_vs_gt_end.png: same comparison at the end of the 20-goal benchmark run s5 (robot at the east end of the corridor).
stub_rooms_map_vs_gt.png: /map over the stub world 'rooms' after a scripted drive without Nav2 (precision 0.69, recall 0.77, walls registered within one cell).
waypoints_replica_apartment_1.png: final ground-truth error of each waypoint in the Gazebo world replica_apartment_1 (kinematic locomotion, `/nav/follow_waypoints`), direct backend (1 of 12 reached) and Nav2 backend (2 of 10), dashed line = 0.4 m tolerance, green = reached. Source: results/waypoints/bench_*.json.
known_map_lab_01.png: known-map navigation (A* on the ground-truth grid, no SLAM / Nav2) in lab_01, kinematic sim: 12 of 12 waypoints reached, 55.8 m driven (A* 62.1 m), 0 collisions, RTF 1.0. Planned paths blue, /gt/pose trajectory orange, yellow = reached. Animation: known_map_lab_01.gif (docs/NAVIGATION_KNOWN_MAP.md).
known_map_replica_frl_apartment_0.png: same run in the Replica scan replica_frl_apartment_0: 12 of 12 reached, 31.3 m (A* 36.7 m), 0 collisions, RTF 0.34. The waypoints lie in the part of the flat with 0.55 m clearance (limits in docs/NAVIGATION_KNOWN_MAP.md).
known_map_lab_01_cam1.jpg: simulator RGB camera during the lab_01 known-map run (about 60 s in).
known_map_lab_01_cam3.jpg: simulator RGB camera during the lab_01 known-map run (about 180 s in).
known_map_replica_frl_apartment_0_cam1.jpg: simulator RGB camera during the replica_frl_apartment_0 known-map run (about 120 s wall).
known_map_replica_frl_apartment_0_cam3.jpg: simulator RGB camera during the replica_frl_apartment_0 known-map run (about 360 s wall).
