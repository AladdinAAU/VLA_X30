# Captions: hospital rooms and the Room C doorway (see docs/INSTRUCTION_TO_GOAL.md)

* `aws_hospital_regions.png`: Automatic room segmentation of the hospital 2D map (36 regions; the corridor, region 2, swallows most of the building).
* `aws_hospital_rooms.png`: The 7 operator-defined rooms A-G of the central column (configs/rooms/aws_hospital.yaml) over the automatic regions.
* `roomC_door_zoom.png`: Room C doorway (about 0.9 m wide, x 2.3-3.2 at y 0): planned path (green, offline, same planner) and the first run's actual trajectory (blue); red = robot footprint at the final pose, rotated about 45 deg across the door.
* `roomC_plan_vs_actual.png`: Overview of the planned path vs the first run's trajectory for Room C.
* `roomC_replay_before_fix.gif`: Room C, first run (multi-turn policy, strict guard): the robot stops at the door mouth, guard holds, run fails.
* `roomC_replay.gif`: Room C, locate_nav policy + relaxed guard: reaches the room in 60 s sim time.
* `roomC_demo_from_bag.mp4`: Room C demo built from a rosbag of the episode (camera at the sim rate, pose, planner path, guard state, cmd_vel; sim-time paced). Recorded with scripts/e2e/record_episode.sh, exported with bag_export.py, composed with bag_demo.py.
