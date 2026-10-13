# Captions: follow_person (see docs/FOLLOW_PERSON.md)

* `follow_person_demo.mp4`: follow_person in aws_hospital recorded as a rosbag: live camera, map with the robot, the ground-truth person (magenta) and the tracker estimate (cyan x), 2 m floor and 2.5 m target circles, state/distance/bearing/cmd_vel/guard readout (sim-time paced, 80 s).
* `follow_person_frame.png`: one frame of the demo (walk t = 42 s, distance 2.96 m).
* `follow_distance_plot.png`: ground-truth robot-person distance over the walk (person 0.5 m/s, 4 s pause): never below the 2.0 m floor.
* `follow_person_crowd_demo.mp4`: follow_person with the lock-on service in a crowd (aws_hospital): navy-scrubs target and four other people in four other outfits (red scrubs, green scrubs, dark top, maroon dress) crossing, passing head-on in the corridor lane and being overtaken; the map shows everybody (coloured by outfit), the red 2 m circle and the green 2.5 m target circle; the camera is live; the guard readout shows the robot holding while the red nurse passes (95 s sim).
* `follow_person_crowd_frame.png`: one frame of the crowd demo at walk t = 23 s (red nurse passing at the robot's side).
* `follow_person_twin_demo.mp4`: look-alike test (twin_rec4): a second nurse in the SAME navy scrubs overtakes the robot and passes the paused target; the lock holds twice (state ambiguous, red text) and never follows the twin (magenta = true target, blue-dot = twin).
