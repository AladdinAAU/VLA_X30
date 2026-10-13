# One segmented object observation from a single camera frame, lifted to 3D by lidar projection (or sim depth)
string label
float32 score
int32[4] box_xyxy                     # pixels in the RGB image
string mask_rle                       # JSON {"size":[h,w],"counts":"..."} (COCO RLE), "" if no mask
uint32 point_count                    # lidar (or depth) points inside the mask used for the 3D estimate
geometry_msgs/Point centroid          # in header.frame_id of the array (normally "map"); valid if !bearing_only
geometry_msgs/Vector3 extent          # axis-aligned size (m) from the lifted points
geometry_msgs/Vector3 bearing         # unit vector from the camera towards the object, in header.frame_id
bool bearing_only                     # true: too few 3D points, only the bearing is meaningful
string depth_source                   # "lidar" | "sim_depth" | "none"
float32 thermal_mean_c                # mean temperature of the mask region in the thermal image (NaN if unavailable)
float32 thermal_max_c                 # max temperature of the region (NaN if unavailable)
sensor_msgs/CompressedImage crop      # jpeg crop of the object (may be empty)
