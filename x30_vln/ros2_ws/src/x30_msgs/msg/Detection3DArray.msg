std_msgs/Header header                # frame_id = frame of centroid/bearing (normally "map")
builtin_interfaces/Time image_stamp   # stamp of the RGB frame the detections come from
geometry_msgs/Point camera_position   # camera origin in header.frame_id at image_stamp
x30_msgs/Detection3D[] detections
