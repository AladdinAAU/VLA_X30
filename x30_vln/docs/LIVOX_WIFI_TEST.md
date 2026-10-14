# Manual Livox test on robot Wi-Fi

Run these steps locally after connecting to `X30021148_5G`. Leave the robot Wi-Fi connected until the test is complete.
Chat may be unavailable on this network; these commands run on the workstation without a chat connection.
The ROS 1 master is `192.168.1.106:11311`. The first live test reached it and listed its topics, but captured zero messages:
the bridge could not resolve publisher hostname `ysc-perception` and was shut down by a duplicate `/ros_bridge` registration.
The Livox overlay now adds the ROB7 hostname mapping and gives the bridge a distinct ROS 1 node name.

Prepare the bridge source and build native Livox CustomMsg support first. Preparation can reuse the cached bridge image;
only a first download without that image needs normal Wi-Fi. The Docker build itself uses local source:

```bash
cd /home/a/xxx/VLA_X30/x30_vln
bash docker/images/livox_bridge/prepare-source.sh
docker compose -f docker/compose.yml -f docker/compose.real.yml -f docker/compose.livox.yml \
    --profile real build ros1_bridge framework
```

Then follow the steps below on robot Wi-Fi. This uses the same topic bridge with additional native Livox and IMU topics.

## 1. Confirm The Workstation Address

```bash
cd /home/a/xxx/VLA_X30/x30_vln
ip -brief address show wlp2s0
ip route get 192.168.1.106
```

Read the address after `src` in the route output. The first test assigned `192.168.1.133`; use the actual current address.
The route should go through robot Wi-Fi, not the campus Wi-Fi gateway.

## 2. Start The Bridge

Set the master and this workstation's address, replacing the example `ROS_IP` if needed:

```bash
export ROS_MASTER_URI=http://192.168.1.106:11311
export ROS_IP=192.168.1.133
livox_compose() {
    docker compose -f docker/compose.yml -f docker/compose.real.yml -f docker/compose.livox.yml \
        --profile real "$@"
}
livox_compose up -d ros1_bridge framework tools
docker exec x30-livox-bridge getent hosts ysc-perception
docker logs --tail 40 x30-livox-bridge
```

`ROS_MASTER_URI` identifies the directory of ROS 1 nodes. `ROS_IP` tells the robot where to call back to this workstation.
Getting a topic list proves access to the directory, but actual data comes from the publisher's advertised hostname and port.
`getent` should return `192.168.1.106` for `ysc-perception`. Bridge logs should show the selected topics being created,
without hostname-resolution errors or a duplicate-node shutdown.

## 3. Prove Data Arrives In ROS 1

```bash
docker exec x30-livox-bridge x30-bridge-entrypoint ros1 rostopic info /lidar_points
docker exec x30-livox-bridge timeout 10 x30-bridge-entrypoint ros1 rostopic echo -n 1 /lidar_points/header
```

The first command identifies the registered publishers. The second subscribes to an actual message and prints only its header,
so the terminal is not flooded with point data. A timestamp and `frame_id` confirm delivery; a timeout is still a data-path failure.
If this step fails, inspect the bridge log and publisher host before moving to ROS 2.

## 4. Prove The Bridge Delivers ROS 2 Data

```bash
docker exec x30-livox-framework bash -lc 'ros2 topic echo /lidar_points sensor_msgs/msg/PointCloud2 --no-daemon --field header --once'
docker exec x30-livox-framework bash -lc 'ros2 topic hz /lidar_points'
```

The first command proves a converted message reached Humble. The second measures the live rate; stop it with Ctrl+C after
about ten seconds. Around 10 Hz is expected from the earlier recorded cloud, but Wi-Fi performance must be measured.
Listing the ROS 2 topic alone does not prove that any cloud was received.

Check the native FAST-LIO candidate stream:

```bash
docker exec x30-livox-bridge x30-bridge-entrypoint ros1 rostopic type /livox/lidar
docker exec x30-livox-framework bash -lc 'ros2 interface show livox_ros_driver2/msg/CustomMsg'
docker exec x30-livox-framework bash -lc 'timeout 10 ros2 topic echo /livox/lidar livox_ros_driver2/msg/CustomMsg --no-daemon --field point_num --once'
docker exec x30-livox-framework bash -lc 'timeout 10 ros2 topic echo /imu/data sensor_msgs/msg/Imu --no-daemon --field header --once'
```

The ROS 1 type must be `livox_ros_driver2/CustomMsg`. A nonzero `point_num` proves native points reached ROS 2.
Inspect the header and `lidar_id` similarly to identify the device; compare its timing with the selected IMU.
The bridge also carries the four `/livox/imu_192_168_2_202` through `_205` topics so a device-specific IMU can be selected.
Use the merged `/lidar_points` cloud for the existing 3D visualization.

## 5. View Cloud And TF

Connect Foxglove to `ws://localhost:8766`, add a 3D panel, and select `/lidar_points`. Use the actual cloud `frame_id`
as the display frame first (`lidar_link` in the earlier recording). This shows the live scan without requiring external TF.

```bash
docker exec x30-livox-framework bash -lc 'ros2 topic echo /tf --no-daemon --once'
docker exec x30-livox-framework bash -lc 'ros2 topic echo /tf_static --no-daemon --qos-durability transient_local --once'
docker exec x30-livox-framework bash -lc 'ros2 run tf2_ros tf2_echo odom lidar_link'
```

The last command assumes the robot uses `odom` and `lidar_link`; substitute the observed frame names.
If it resolves, use `odom` as Foxglove's display frame. TF may be absent even when the cloud arrives: those are separate checks.
Do not add guessed transforms or launch our own TF publishers into this domain.

## 6. Record A Short Export

```bash
docker exec -it x30-livox-framework bash -lc \
    'mkdir -p /x30/datasets/real; ros2 bag record -o /x30/datasets/real/livox_$(date +%Y%m%d_%H%M%S) /livox/lidar /lidar_points /imu/data /livox/imu_192_168_2_202 /livox/imu_192_168_2_203 /livox/imu_192_168_2_204 /livox/imu_192_168_2_205 /tf /tf_static'
```

After about 20 seconds, press Ctrl+C to close the bag. It is stored under this repository's `x30_vln/datasets/real/`.
Check it with `ros2 bag info` inside the framework container; the cloud topic must have a nonzero message count.
Keep the terminal outputs from steps 2-5 to diagnose any failed stage after returning to campus Wi-Fi.

To stop this setup, use `livox_compose down` in the shell where you defined the helper.

## ROS 2 CLI Daemon Failure

An `xmlrpc.client.Fault` mentioning `!rclpy.ok()` during `get_publishers_info_by_topic` comes from the local
ROS 2 CLI daemon's failed context. Its XML-RPC server can report "running" even though its ROS node cannot service requests.
Stop that daemon and use direct discovery:

```bash
docker exec x30-livox-framework bash -lc 'ros2 daemon stop'
docker exec x30-livox-framework bash -lc 'timeout 15 ros2 topic echo /livox/lidar livox_ros_driver2/msg/CustomMsg --no-daemon --field point_num --once'
```

`--no-daemon` gives the CLI its own ROS 2 node for discovery and subscription. Keep robot Wi-Fi connected for the test;
a timeout after switching back to campus Wi-Fi is a separate lack of live data. `xhost` controls X11 GUI access and has no effect on this CLI error.
