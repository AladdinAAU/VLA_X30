# Figures: worlds

- `replica_apartment_0_camera.jpg`: four frames from the simulated robot camera (640x480, 100 deg, 0.55 m above the floor) in `replica_apartment_0` (Replica scan, vertex colours re-baked to one 2048 px atlas, 248k triangles). Visible: dark speckle from the baked scan noise.
- `replica_apartment_0_lidar_bev.png`: one merged `/lidar_points` scan of the same world, body frame, points between z = -0.3 and 1.5 m, top view; walls and furniture edges are visible, sparse because the four Livox patterns sample 20 k rays each.
- `replicacad_apt_0_camera.jpg`: four camera frames in `replicacad_apt_0` (ReplicaCAD stage + 118 object instances, 147 k triangles, doors opened).
- `replica_frl_apartment_0_map.png`, `replicacad_apt_1_map.png`: occupancy map of the obstacle layer (black = blocked), ground-truth object boxes (blue), robot-centre reachable set for a 0.25 m planning radius (green).
- `detector_overlays.jpg` (3 x 2 grid, left to right, top to bottom): procedural `apt_01` (two frames), `replicacad_apt_0` (two), `replica_apartment_0` (two). Green box = detection matching a same-class ground-truth object, red = detection without one, cyan = visible ground-truth object boxes (projected, unoccluded in the 2D map). The `apt_01` frames face flat walls and corners; the detector finds almost nothing.
- `detector_overlays_2.jpg`: the same layout for `replica_frl_apartment_0` (two), `replica_apartment_2` (two), `replicacad_apt_1` (two).
