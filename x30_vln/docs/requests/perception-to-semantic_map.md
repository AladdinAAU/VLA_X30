# perception -> semantic_map (and integrator)

1. **Interfaces added by perception (please add to INTERFACES.md section 4):**
   - `/perception/diagnostics` (`diagnostic_msgs/DiagnosticArray`, one status `x30_perception/perception_client`, key/values with latencies and counters).
   - Service `/perception/vocabulary` (`x30_msgs/srv/Json`): `{"op": "add|remove|set|reset|get|stats", "labels": [...], "ttl_s": 600}` -> `{"phrases": [...], "server_conf": 0.12}`.
     The mapper / agent can use it to add task phrases ("red fire hydrant") before searching for them.
   - `Detection3D.mask_rle` is a JSON string whose `size` is the resolution sent to the server (equal to the image size at the default `send_max_side: 960` for 640x480);
     `box_xyxy` is in camera-image pixels. Use `size` when decoding.
2. **Attributes from thermal data.** `Detection3D` has no attribute field. Use `x30_perception.thermal.attributes_from_stats(mean_c, max_c, ambient_c, label, rules)`
   with `rules` = `configs/perception.yaml: thermal.rules` (ambient = median of the thermal image; the client does not publish it, so use
   rules with `min_mean_c`/`min_max_c` only, or ask for a field). Without `/thermal/*` the values are NaN and no attribute should be set.
3. **Position semantics.** `centroid` is the median of lidar points on the visible surface of the object in the map frame, i.e. on the side facing the robot.
   For objects with depth the geometric centre is about half the object depth further away. `extent` is the visible-surface extent (2-98 percentile), not a full 3D box.
   `bearing_only` detections have no centroid (zeros): place them by triangulating bearings from different `camera_position`s.
4. Labels are alias-resolved (`configs/perception.yaml: aliases`), e.g. office chair/armchair/stool -> `chair`; the raw prompt is not published.
   Several overlapping detections of the same label (duplicate prompts) are already merged (IoU 0.6).
