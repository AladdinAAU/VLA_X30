std_msgs/Header header
bool limiting                         # guard is reducing the command
bool stopped                          # guard output is zero because of an obstacle / watchdog / estop
float32 min_obstacle_distance         # m, along the commanded direction (inf if none)
float32 scale                         # 0..1 applied to the command
string reason                         # "", "obstacle", "estop", "stale_cmd", "no_lidar", "tilt"
