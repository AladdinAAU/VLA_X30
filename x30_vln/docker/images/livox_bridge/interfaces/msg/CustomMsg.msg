# Wire format from Livox-SDK/livox_ros_driver2, matching the vendored ROB7 definitions.
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
CustomPoint[] points
