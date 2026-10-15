# X30 Livox to FAST-LIO

This folder adds **one** FAST-LIO container to the existing, working Livox Compose stack. It keeps the Noetic/Foxy custom-message bridge, Humble interface package, host networking, DDS settings, and robot hostname mapping in `../docker/compose.livox.yml`. Nothing here drives the robot.

## Files

`Dockerfile` pins the ROS 2 FAST-LIO fork; `compose.yml` adds the mapper; `fast_lio.yaml` sets its inputs; `launch.py` starts the mapper and `select_lidar.py`; `run.sh` gives short commands; `check_bag.py` checks playback inputs.

## Inputs and frames

The robot's ROS 1 `/livox/lidar` is `livox_ros_driver2/CustomMsg`; the existing bridge forwards it unchanged to ROS 2. FAST-LIO needs that native point timing plus a time-aligned `sensor_msgs/Imu` stream. This configuration uses `/imu/data` and one Livox `lidar_id` from the possibly multiplexed native topic. Set `LIVOX_LIDAR_ID` after inspecting messages; `auto` chooses the first observed ID. The merged `/lidar_points` (`PointCloud2`, frame `lidar_link`) is useful for display but is **not** this mapper's input. The existing Livox bridge also forwards four `/livox/imu_192_168_2_202` through `_205` streams; edit `common.imu_topic` to one of those only after confirming that its timestamps and physical mounting match the selected LiDAR.

FAST-LIO's origin is its first IMU pose. Its odometry/map frames are local FAST-LIO frames; do not assume they equal the robot's `map`, `odom`, `base_link`, or `lidar_link`. The identity extrinsic in `fast_lio.yaml` is a starting value with online estimation enabled, **not** a measured mounting transform. Before using its pose for robot localization or navigation, measure the LiDAR-to-IMU transform, verify axes and time sync, and establish the transform to the robot body. Keep this mapping domain separate from any robot TF broadcaster until the frame relationship is confirmed.

## Build

Use Ubuntu/WSL with Docker Engine or Docker Desktop Linux containers, internet for the first FAST-LIO source fetch, and the existing `x30-ros1-bridge:latest` and `x30-framework:latest` base images. Run from the repository root:

```bash
cd x30_vln
bash fast_lio_integration/run.sh build
```

The build uses the same Livox bridge source preparation as the working setup. The FAST-LIO fork is pinned to commit `2fffc570a25d0df172720bac034fbdb6a13d2162` on its ROS 2 branch, with its `ikd-Tree` submodule.

## Live robot

Use a Linux host network that can reach the ROS 1 publisher and accept its callback. Set the workstation address on the robot network, not the robot's address:

```bash
cd x30_vln
export ROS_MASTER_URI=http://192.168.1.106:11311
export ROS_IP=192.168.1.133       # replace using: ip route get 192.168.1.106
export LIVOX_LIDAR_ID=0            # replace with the ID observed on /livox/lidar
bash fast_lio_integration/run.sh live
docker logs -f x30-fast-lio
```

Inspect the actual streams before relying on the map:

```bash
docker exec x30-livox-framework bash -lc 'ros2 topic list -t --no-daemon'
docker exec x30-livox-framework bash -lc 'timeout 10 ros2 topic echo /livox/lidar livox_ros_driver2/msg/CustomMsg --no-daemon --field lidar_id --once'
docker exec x30-livox-framework bash -lc 'timeout 10 ros2 topic echo /imu/data sensor_msgs/msg/Imu --no-daemon --field header --once'
docker exec x30-livox-framework bash -lc 'ros2 topic hz /livox/lidar'
```

Check FAST-LIO output with `docker exec x30-livox-framework bash -lc 'ros2 topic list -t --no-daemon'`. Its upstream output topics include `/Odometry` and `/cloud_registered`. Output publication alone does not validate alignment or accuracy.

## Bag playback

Record a **complete** ROS 2 bag while live, including `/livox/lidar`, `/imu/data`, `/tf`, and `/tf_static`. Confirm nonzero message counts with `ros2 bag info`. Store the bag directory under `x30_vln/datasets/real/`. Playback stops the live bridge and starts the mapper with simulated time:

```bash
cd x30_vln
bash fast_lio_integration/run.sh bag datasets/real/YOUR_COMPLETE_BAG
```

The script rejects a bag with no SQLite data or without both native points and IMU. It plays with `--clock` and a three-second startup delay. Run `bash fast_lio_integration/run.sh stop` afterward. For another mapping run, use a fresh ROS domain/container restart so old mapper state is not reused.

The existing `livox_20261014_112245` metadata lists 34 `/lidar_points` messages, 2,893 `/tf`, and 3 `/tf_static`, but **zero** native `/livox/lidar` and IMU. Its `.db3` is also absent in this checkout. The older `lab_2026-10-06` bag has only `/lidar_points`. Neither can run inertial FAST-LIO. Replaying those clouds successfully as ROS topics would not be a successful FAST-LIO mapping test.

On 2026-10-15, the bridge, Humble interface, and FAST-LIO images built in Ubuntu WSL with Docker Desktop. The mapper and selector started and advertised the expected topics. The older bag played through Humble's existing metadata compatibility helper, and a subscriber received a `/lidar_points` header with frame `lidar_link`. Both bags failed the FAST-LIO input check for the reasons above. No robot connection, FAST-LIO pose, or map was verified in that test.

## Troubleshooting

- No ROS 1 points: check `ROS_IP`, `ROS_MASTER_URI`, robot network route, and `docker exec x30-livox-bridge getent hosts ysc-perception`; use `rostopic echo -n 1 /livox/lidar/header` inside the bridge.
- ROS 1 points but no ROS 2 points: check bridge logs and use `ros2 topic ... --no-daemon`; the existing Livox bridge supplies the custom type on both sides.
- FAST-LIO has no pose: verify native point and IMU rates, matching wall-clock stamps, one selected `lidar_id`, IMU axes, and measured extrinsics. Check `docker logs x30-fast-lio`.
- Bag rejected: its `.db3` and both required topics must be present. Record a new complete bag; merged clouds cannot reconstruct Livox per-point timing.
- Docker daemon unavailable: start Docker Desktop or a Linux Docker Engine before building or running. Host-network robot callback behavior must be tested on the actual workstation.
