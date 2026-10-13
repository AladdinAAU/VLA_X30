# Captions: first end-to-end episodes (see docs/END_TO_END.md)

* `architecture.png`: containers, nodes and the topics/services/actions actually used in the run.
* `ep1_sofa_topdown.gif`: episode 1 (sofa) top-down: occupancy map, robot path (green), robot heading (orange), GT sofa bbox (red dashed), semantic-map objects (blue) with the current agent turn and skill in the title.
* `ep1_sofa_topdown.png`, `ep2_chair_topdown.png`, `ep3_tv_topdown.png`: final path, stop pose (star), GT target bbox, semantic-map objects.
* `ep*_semantic_map.png`: the annotated semantic map image rendered by `/semantic_map/render` at the end of each episode.
* `ep*_cam_start.jpg`, `ep*_cam_turnNN.jpg`, `ep*_cam_end.jpg`: camera frames at the start, at decision points (agent turn NN) and at the end. Ep1 end: the robot faces the sofa it verified (`ep1_sofa_cam_end.jpg`).
