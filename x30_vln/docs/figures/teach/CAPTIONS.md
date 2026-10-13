# Captions: learner phase / guided tour (see docs/TEACH_PLACES.md)

* `guided_tour_demo.mp4`: the robot follows a nurse through aws_hospital; she says "This is Marcus." in Room C, "This is Alexander." in Room E and "Here we deliver stuff." in the corridor; the language model turns each sentence into a saved place (orange star + name on the map appears when it is saved, the robot's reply is shown). Recorded as a rosbag, sim-time paced (110 s).
* `guided_tour_frame_marcus.png`: the frame right after the first sentence: 'Marcus' saved as a room (7 m2), owner Marcus. The robot is still 7.6 m behind the nurse in the hall: the place is anchored at the person's position, not the robot's.
