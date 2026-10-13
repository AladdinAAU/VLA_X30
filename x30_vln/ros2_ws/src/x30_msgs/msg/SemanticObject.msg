# One object instance in the semantic map (map frame)
uint32 id
string label
float32 confidence
geometry_msgs/Point position
geometry_msgs/Vector3 size            # axis-aligned extent in the map frame (m)
uint32 point_count
uint32 observations
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
string caption                        # VLM description of the object ("" if none yet)
string[] attributes                   # e.g. "warm", "hot", "person-like", "open", "red"
string room                           # room / region label if known
bool bearing_only                     # true: only a direction is known, position is a guess
sensor_msgs/CompressedImage best_view # best crop (may be empty)
