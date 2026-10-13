# DEEP Robotics Jueying X30 / X30 Pro - Interface Digest

Purpose: exact interface facts for the VLN framework (dev PC ROS2 Humble in Docker, ROS1 Noetic bridge container,
robot motion host by UDP). Engineering digest only, no implementation code.

**Source keys** (page numbers are the printed page numbers of the PDFs)

| Key | File |
| --- | --- |
| `api` | `reference/text/api.txt` = Jueying X30 API Documentation (beta) V1.0.5-0, 2025-08-21, 68 p. Where the text extraction garbled a table (App. B, gait table p.42, formulas p.13-14) the PDF page was rendered and read visually. |
| `man` | `reference/text/manual.txt` = X30 Pro Application Manual V2.0.4-0, 2025-11-29, 69 p. Figures on p.6, p.67, p.69 read visually. |
| `nb` | `reference/repos/nav_bridge` (`nb/README`, `nb/CHARGE`, `nb/yaml`, `nb/hpp` = x30_protocol.hpp, `nb/cpp` = x30_nav_bridge.cpp, `nb/ae` = action_executor.cpp) |
| `py` | `reference/repos/X30/README.md` (tkinter UDP controller, actually Python source) |
| `dx` | `reference/repos/dogx30` (`dx/proto` = docs/x30-protocol.md, `dx/hw` = hardware-integration.md, `dx/sys` = system-architecture.md, `dx/media` = media-architecture.md, `dx/pc` = pointcloud-architecture.md, `dx/src` = rk3588/ headers+src, `dx/deploy` = deploy/) |
| `rob7` | `reference/repos/Rob7_Gr760_P7` (our prework: `Docker/Main/docker-compose.yaml`, `Docker/Main/bridge/topics.yaml`) |
| `bag` | rosbag analysis supplied in the task brief (single topic `/lidar_points`) |
| `legacy` | `Robot Documentation/Legacy file.docx` (short note from an earlier student group) |

Marker **(inferred)** = not stated in a source; derived by me. **not documented** = absent from every supplied file.
`dx` is a third-party project (RK3588 payload gateway for another X30 unit). Its facts are "field notes", not vendor
documentation, and are labelled as such.

---

## 0. Ten facts that matter most

1. The docs name **.105 = perception host** and **.106 = navigation host (intelligent controller)** [api p.38, p.57; man p.6].
   Our prework talks to ROS master `http://192.168.1.106:11311` [rob7 compose], i.e. the navigation host. The brief calls
   .106 the "perception host"; that is a naming mismatch, not a different machine.
2. Motion is commanded **only** by UDP to `192.168.1.103:43893`, little-endian, 12-byte header, heartbeat >= 2 Hz [api p.6-9].
3. Telemetry comes back only to IP:port pairs **registered in `network.toml` on the motion host**
   (default `ips=[.105,.106]`, `ports=[43897,43897]`). Our dev PC (192.168.1.131 in rob7) must be added or nav_bridge receives nothing [man p.56-57; dx/hw; dx/proto].
4. Velocity input = joystick axes `[-32767,32767]`, scaled by the **max speed of the current gait** (App. B), dead zone 655,
   50 Hz, invalid after 1 s [api p.11-14]. The ROS twist is therefore mapped *per gait*.
5. **nav_bridge's per-gait speed table is wrong for most gaits** versus API App. B (see 3.6). Only the default gait
   MOUNTAIN (33) and L_WALK (32) match. Slope is 2.3x over-driven.
6. nav_bridge works in **manual mode** (simulated 0x21 joystick) so the robot's elevation-map collision avoidance is
   **not** in the loop (inferred from api p.40-41, man p.45). Treat the robot as having no obstacle protection.
7. nav_bridge timestamps IMU/odom with the **dev-PC clock at receive time**, publishes **zero covariances**, 2-D leg
   odometry only, and holds the last `/cmd_vel` for **5 s** by default.
8. 4 Livox (Mid-360 class) lidars are at 192.168.2.202-205 on a separate switch (192.168.2.0/24); the robot publishes a merged
   `/lidar_points` (10 Hz) plus per-lidar IMU topics `/livox/imu_192_168_2_20x` [bag; rob7 topics.yaml; dx/sys].
9. The X30 has **one wide-angle USB2.0 camera** on the perception host and **no thermal camera** in any vendor document.
   No intrinsics, FOV, resolution, or ROS image topic is documented. A field-found RTSP URL exists (see 6).
10. Body mass, dimensions, battery capacity, speed/stair/slope specs, camera and thermal specs, controller latency and
    odometry drift are **not in the supplied files** (see 8).

---

## 1. Hardware inventory

### 1.1 Compute hosts

| Host | IP(s) | Role | OS / software | Notes / source |
| --- | --- | --- | --- | --- |
| Motion host | 192.168.1.103 (eth port 1) | gait control, 12 joints, receives IMU; UDP 43893 | not documented; `~/jy_exe` (motion program `jy_exe`), conf `motion.toml`, `network.toml`, `overtime.toml`; scripts `restart.sh`, `stop.sh`, `show_log.sh`; logs kept 7 days | `ssh ysc@192.168.1.103`; vendor default password is the single character `'` (backtick/single quote as printed) [man p.7, p.10] |
| Perception host | 192.168.1.105 (port 1), 192.168.2.105 (port 0 -> lidar switch) | sensor drivers, elevation-map (vmap) collision avoidance + footstep planning, LIO for multi-frame stairs, charging manager; USB2.0 wide-angle camera | ROS 1 (Noetic stated only in `legacy`); `~/jy_cog/transfer/setup.bash` provides custom msgs | [api p.29-37, p.38-56; man p.6] |
| Navigation host = "intelligent controller" | 192.168.1.106 (port 1), 192.168.2.106 (port 2 -> lidar switch) | mapping, localization, navigation (`jy_cog`: slam, nav, localization, player, broker/robot_server, app) | Ubuntu desktop with VNC; ROS 1 master (rob7: `:11311`) | VNC `192.168.1.106`, password `123456`; user `ysc`; `/home/ysc/jy_cog` [man p.11-13]. Bolts to the back in 3 positions A/B/C; own IMU; counts as 3.0 kg load at (0.05, 0, 0.10) m [man p.58-60, p.68] |
| Lidar switch | 192.168.2.x | 4x Livox on 192.168.2.0/24 | n/a | lidar IPs 192.168.2.202-205 [bag brief; rob7 topics.yaml] |
| Router / debug port | 192.168.1.x | tail "Debug Ethernet Port" via a router; Wi-Fi SSID `X30*****` (serial no.), password `12345678` | n/a | [man p.6-7]. dx notes the router may be NAT, not bridge: verify DHCP gives 192.168.1.x [dx/sys] |
| 2.4G receiver | on 192.168.1.x switch | original handheld controller link | n/a | [man p.6] |
| Dev PC (ours) | 192.168.1.131 | ROS2 Humble + ros1_bridge (foxy image) | Docker host network | `ROS_IP=192.168.1.131`, `ROS_DOMAIN_ID=0` [rob7 compose] |

Network topology (man p.6 figure): `Router - Switch(192.168.1.x) - {Motion .103, Perception .105, Intelligent controller .106, 2.4G receiver}`;
`Perception host port0 .2.105` and `Controller port2 .2.106` both attach to the **Laser-radar switch** with the 4 lidars.
IMU feeds both motion host and perception host. Wide-angle camera -> perception host over **USB 2.0**.

Payload interfaces (body): power 72 V / 600 W max; Ethernet port 1 = 192.168.1.*, Ethernet port 2 = 192.168.2.*; mount to both rails,
payload height <= 500 mm, payload 0-20 kg (`motion.toml load`) [man p.53-56, p.8]. Intelligent-controller ports: LAN1/3/4 12 V
(36 W), LAN5/6 24 V (72 W), 24 V DC 480 W, 2x USB2.0; total payload power <= 600 W [man p.61-65].

### 1.2 Sensors

| Sensor | Count | Spec (documented) | Mounting / extrinsics | Source |
| --- | --- | --- | --- | --- |
| Lidar, Livox (model by inference Mid-360) | 4: front-up, front-down, rear-up, rear-down | FOV figure: **360 deg horizontal, +52 deg / -7 deg vertical**; manual text says only "LiDAR"; model name appears only in dx ("Livox Mid-360 x4") and in the topic name `..._mid360/obstacle_pointcloud` | position in body frame (mm): front-up (443.67, 0, 54.15); front-down (432.97, 0, -55.19); rear-up (-465.76, 0, 34.84); rear-down (-410.74, 0, -56.01). Angles in figure: front-up **25 deg**, front-down **46 deg**, rear-up **5 deg**, rear-down **56 deg**. Axis definition is not stated; arrows suggest the sensor axis is pitched up-forward 25, down-forward 46, backward ~5 down, down-backward 56 (inferred). Table valid for body version >= 401 [man p.2, p.67; dx/hw] | [man p.66-69] |
| Lidar IPs | | 192.168.2.202, .203, .204, .205 | which IP is which mount is not documented | [bag brief; rob7 topics.yaml] |
| Merged cloud | 1 | `/lidar_points`, PointCloud2, frame `lidar_link`, ~80 k pts/scan @10 Hz, fields x,y,z,intensity f32, tag u8, line u8, timestamp f64, point_step 26 (= 16+1+1+8, offsets 0,4,8,12,16,17,18 (inferred)). 4 x 200 kpts/s x 0.1 s = 80 k, consistent with 4 Mid-360 | published by robot [bag; api p.38] |
| Raw lidar | 1 | `/livox/lidar` `livox_ros_driver2/CustomMsg` 10 Hz (header, timebase u64, point_num, lidar_id, rsvd[3], points{offset_time u32,x,y,z,reflectivity u8,tag u8,line u8}) | | [api p.38; rob7 ros1_ws msg] |
| Built-in IMU (body) | 1 | `/imu/data` sensor_msgs/Imu 200 Hz; vendor node name `/yesense_imu_node` suggests a Yesense IMU (inferred) | origin in body frame (130.45, -1, 14.9) mm; axes aligned with body frame (x fwd, z up) [man p.67, p.32, p.49] | |
| Livox internal IMUs | 4 | `/livox/imu_192_168_2_202..205` sensor_msgs/Imu; rate not documented (Mid-360 internal IMU is nominally 200 Hz, vendor datasheet, unverified) | not documented | [rob7 topics.yaml] |
| Controller IMU | 1 | present in intelligent controller; aligned with body axes | origin A (87.15, -1, 117) / B (47.15, -1, 117) / C (7.15, -1, 117) mm depending on mount slot | [man p.68] |
| Joint encoders / torque | 12 joints | pos rad, vel rad/s, torque N*m at 200 Hz (UDP 0x100A); ROS `/joint_states` | order FL,FR,HL,HR x (HipX, HipY, Knee) | [api p.23-25, p.43] |
| RGB camera | 1 wide-angle, USB 2.0 -> perception host | model, resolution, FOV, intrinsics, mounting pose **not documented**. Controller app shows a video stream ("save camera view" at waypoints) | not documented | [man p.6, p.40; api: zero camera mentions] |
| Thermal / IR camera | **none on the X30** | not in any vendor file. dx uses a third-party dual-light PTZ ball (see 6.3) | n/a | |
| Depth camera, GPS | none documented | | | |
| Foot contact | `touch_state` field | reserved ("Reserve") | | [api p.20-21] |
| Temperatures | 12 motor, 12 driver, CPU, 4 battery NTC | | | [api p.25-26] |

### 1.3 Body, performance, power

| Item | Value | Source |
| --- | --- | --- |
| Body mass, dimensions, standing height | **not documented** (body-frame origin = geometric centre; body is ~0.9 m long judging from lidar x = +/-0.45 m (inferred)) | [man p.66] |
| Payload | 0-20 kg; payload CoM `load_x` +/-0.3 m, `load_y` +/-0.1 m, `load_z` 0-0.3 m; never enable `enable_twist` with a load | [man p.8-9] |
| Speed limits per gait | see table 3.6 (api App. B; **conflicting** values in nb, dx) | [api p.65] |
| Body height | 2 gears: Crawl (cmd 0), Normal (cmd 2); crawl only for Walk/Slope; max forward of Walk/Slope halves in crawl | [api p.14-15, p.65] |
| Stairs / slope capability | no angle/step height given. Obstacle height thresholds: 8 cm (walk/slope), 28 cm (off-road/stair) ; recommended /step_z_max 0.08 walk, 0.12 slope, 0.25 stair (m). `legacy`: "difficulties walking up stairs even in stair mode". Waypoint before upstairs >= 0.8 m from stair edge | [api p.36-37, p.53-54; man p.40; legacy] |
| Battery | nominal 72 V, charger 84 V (dx field note); fields: voltage V, current (A, or 10 mA from deeprcs >= 1.10.19), remaining/nominal capacity (10 mAh), cycles, 4 temps, % level; 0.5 Hz. Capacity value, runtime **not documented** | [api p.25-26, p.39-40; dx/hw] |
| Mapping limits | fast/indoor: 50x50 m (multi-floor 10 floors), large scene 500x500 m (slam >= 2.1.0); older slam: 200x100 m; no narrow stairs, open outdoor, smooth corridors | [man p.15, p.22-24] |
| Software versions referenced | jy_exe >= 2.2.45 (new save-data code), >= 1.10.19 (current in 10 mA), >= 1.10.28 (multi-frame), launcher 0.0.5, fast-lio 1.0.3, jycog-system 0.5.5; nav: localization 2.0.5, nav 2.1.9, player 1.0.6, broker 1.1.6, app 2.3.2; app >= 2.3.1 for vel-source persistence | [api p.16, p.26, p.30, p.34; man p.33] |

---

## 2. Network map

### 2.1 Endpoints

| Address | Port / proto | Service | Direction | Source |
| --- | --- | --- | --- | --- |
| 192.168.1.103 | **43893 / UDP** | motion commands (heartbeat, axes, gait...) | client -> robot | [api p.8; man p.57] |
| client IP | **43897 / UDP** (default `target_port`) | telemetry 0x1008/1009/100A/100B/21050F0A/11050F08 | robot -> registered IPs | `network.toml`: `ip='192.168.1.105'`, `target_port=43897`, `local_port=43893`, `ips=[".105",".106"]`, `ports=[43897,43897]`; arrays are index-paired; restart motion program via `~/jy_exe/scripts/restart.sh` [man p.56-57] |
| 192.168.1.105 | **43899 / UDP** | elevation map: vel source 0x3101EE03, map mode 0x3101EE01, avoidance mode 0x3101EE02, height threshold, query 0x3101EEFF; after first packet robot streams state at 10 Hz | request / event stream | [api p.29-37, p.67-68] |
| 192.168.1.105 | **60000 / UDP** | LIO on/off 0x0BAA0001 (value 1/0), reply 0 ok, -1 failed | request-response | [api p.34-35, p.66] |
| 192.168.1.105 | **3333 / UDP** | auto-charge 0x91910250 (1 start,0 stop,2 clear), 0x91910253 query; reply echoes code with state; nb local port 49004 | request-response | [api p.27-29; nb/CHARGE] |
| 192.168.1.105 | 8554 / RTSP, path `/test` | body camera, **not vendor-documented**, field-confirmed in dx | pull | [dx/proto; dx/media] |
| 192.168.1.106 | 11311 / TCP (XML-RPC) | ROS 1 master used by our bridge | | [rob7 compose]. API does not state any ROS_MASTER_URI; dx guessed `.105:11311` and never verified. Possibly two masters, one per host (inferred; docs list sensor topics for both hosts) |
| 192.168.1.106 | 30000 / TCP | "body monitor" (official 2026 doc per dx): 16-byte LE header (magic `EB 91 EB 90`, u16 XML length, u16 request id, 8 reserved) + UTF-8 XML; poll `Type=1002, Command=1`; MotionState 0 sit,1 standing-up,2 stand,3 torque,4 walk,5 sitting-down,6 e-stop,7 fall,16 RL; all-zero items = query failed | request-response | [dx/proto] (not in api/man) |
| 192.168.1.106 | VNC (port not documented) | desktop for mapping / RViz | | [man p.11-13] |
| 192.168.2.202-205 | Livox SDK2 UDP | lidars. Mid-360 default ports (lidar side) 56100 cmd, 56200 push, 56300 points, 56400 IMU (Livox SDK2 convention, **not in supplied docs except 56300/56400 in dx**; unverified) | | [dx/hw] |
| 192.168.1.131 | local 43897/UDP, 49004/UDP | nav_bridge telemetry bind, charge reply bind | | [nb/yaml] |

Note: 43897 is both the robot's default telemetry port and the port nav_bridge binds with `SO_REUSEADDR` [nb/udp_transport.cpp]. On a host that also runs another UDP
listener on 43897, unicast datagrams reach only one of the sockets (which one is undefined).

### 2.2 ROS1 topics published by the robot (vendor list)

Types and rates exactly as in the docs; "-" = rate not given. Hosts: sensor and state topics exist on perception host and, per api p.57, "the same" on the navigation host.

| Topic | Type | Rate | Source |
| --- | --- | --- | --- |
| `/imu/data` | sensor_msgs/Imu | 200 Hz | api p.38; man p.49 |
| `/lidar_points` | sensor_msgs/PointCloud2 | 10 Hz | api p.38; bag |
| `/livox/lidar` | livox_ros_driver2/CustomMsg | 10 Hz | api p.38 |
| `/livox/imu_192_168_2_202`...`205` | sensor_msgs/Imu | - | rob7 |
| `/leg_odom` | nav_msgs/Odometry (world frame pose, 2-D) | ~200 Hz | api p.42; man p.49 |
| `/robot_basic_state`, `/robot_gait_state` | std_msgs/Int32 | - | api p.41-42 |
| `/robot_velocity` | geometry_msgs/Twist | - | api p.42 |
| `/control_mode` | std_msgs/UInt8 (0 manual, 1 non-manual) | - | api p.40 |
| `/battery/level` UInt8, `/battery/current` Float32, `/battery/voltage` Float32, `/battery_state` message_transformer_cpp/BatteryData | | - | api p.39 |
| `/joint_states` sensor_msgs/JointState; `/motor_temperature` Float32MultiArray; `/driver_temperature` UInt8MultiArray | | - | api p.43 |
| `/mileage/current_mileage`, `/mileage/total_mileage` Int32; `/run_time/{current,total}_{run,motion}_time` Int64; `/motion_cpu/{frequency,temperature}` Float32 | | - | api p.42-44 |
| `/handle_state` Twist (joystick), `/cmd_vel` Twist (navigation), `/cmd_vel_corrected` Twist (after elevation-map correction; executed by motion program in non-manual mode) | | - | api p.44-45 |
| `/deeprobotics_local_height_map_mid360/obstacle_pointcloud` PointCloud2 (obstacles in CoM frame; name reassembled from split text) | | - | api p.45-46 |
| `/brake_mode`(+`_state`), `/vel_source`(+`_state`), `/height_map_mode`(+`_state`) Int32; `/step_z_max`, `/slow_t`, `/stop_t` (+`_state`) Float64; `/charge_manager_state` Int32 | | - | api p.46-56 |
| Navigation host: `/initialpose` PoseWithCovarianceStamped, `/map_request/pcd` String, `/odom` Odometry, `/globalmap` PointCloud2, `/location_status` Int8 (0 ok, 1 unavailable; order inferred from garbled table), `/map` OccupancyGrid, `/move_base/goal` MoveBaseActionGoal, `/move_base_direct_line/goal`, `/move_base/GlobalPlanner/plan`, `/move_base/LocalPlanner/local_plan`, `/move_base/LocalPlanner/global_plan` Path, `/move_base/obs_state` Int32, `/cmd_vel` Twist | | - | api p.57-63 |

Custom msg definitions require `source ~/jy_cog/transfer/setup.bash` (CustomMsg, BatteryData) or `~/jy_cog/broker|nav/setup.bash` (MoveBase*) [api p.39-40, p.60-61].

### 2.3 Our prework bridge (rob7)

`ros:foxy-ros1-bridge` container, `ROS_MASTER_URI=http://192.168.1.106:11311`, `ROS_IP=192.168.1.131`, `parameter_bridge` list
(`bridge/topics.yaml`): `/cmd_vel` (Twist), `/livox/imu_192_168_2_202..205` (Imu), `/lidar_points` (PointCloud2), `/tf` (TFMessage).
Caveats (inferred): `parameter_bridge` normally creates both directions per listed topic, so **`/cmd_vel` published in ROS2 also reaches the robot's ROS1 `/cmd_vel`**
(navigation-host topic that, in non-manual navigation mode, feeds the elevation-map module [api p.62-63]) while nav_bridge simultaneously sends the same velocity over UDP.
`/livox/lidar` CustomMsg and `/imu/data` are **not** bridged; nav_bridge supplies `/imu/data` from UDP instead. The main dev image is `ros:humble-ros-base-jammy`; the bridge image is foxy.

---

## 3. UDP protocol (motion host)

### 3.1 Framing [api p.6-8]

All little-endian. Packet = 12-byte head, optional payload.

```
struct CommandHead { uint32 code; uint32 paramters_size; uint32 type; }   // 12 B
type 0  simple : paramters_size field carries the command VALUE (int32 two's complement, e.g. -1 = 0xFFFFFFFF); no payload
type 1  complex: paramters_size = payload length in bytes; send length = 12 + paramters_size; receive check recv_size == paramters_size + 12
```
Prefix of command codes: `0x21` = remote-control terminal, `0x31` = autonomous module (e.g. navigation) [api p.9]. Every table below uses 0x21 unless noted.
Sending a command never overrides robot logic (e.g. "start motion" while lying is ignored) [api p.8].
The py script and nb pack simple commands with `struct.pack("<III", code, value, 0)` [py; nb/hpp].

### 3.2 Command codes to 192.168.1.103:43893

| Name | Code | Value | Payload meaning / notes | Source |
| --- | --- | --- | --- | --- |
| Heartbeat | `0x21040001` | - | send continuously, **>= 2 Hz**; timeout => motion program treats link as down. nb 5 Hz (200 ms), dx 10 Hz, py exactly 2 Hz (no margin) | api p.9; nb/yaml; dx/src; py |
| Connect confirm (query) | `0x21020001` | - | send **once** after heartbeat starts | api p.9 |
| Stand/Sit toggle | `0x21010202` | - | toggles Sitting<->Standing; also the documented way out of e-stop/fall (back to Sitting) [state diagram]; dx: "unload" after soft e-stop | api p.9-10; dx/proto |
| Torque-controlled stand | `0x2101020A` | - | Initial Standing -> Torque-controlled Standing | api p.10 |
| Start/Stop stepping | `0x21010201` | - | toggle Torque-controlled Standing <-> Stepping (sending twice = stop) | api p.10; dx/proto |
| Axis: left stick Y | `0x21010130` | int32 [-32767,32767] | Stepping: body-X velocity (+ fwd). Torque stand: body height | api p.11-12 |
| Axis: left stick X | `0x21010131` | same | Stepping: body-Y velocity, **+ = right** (sign opposite to ROS +y left). Torque stand: roll | api p.11, p.13 |
| Axis: right stick X | `0x21010135` | same | Stepping: yaw rate, **+ = turn right** (ROS +wz is CCW/left). Torque stand: yaw | api p.11, p.13-14 |
| Axis: right stick Y | `0x21010102` | same | Torque stand: pitch; no function stepping | api p.11 |
| Non-manual mode | `0x21010C03` | - | robot follows velocity from perception host (elevation-map-corrected) | api p.14, p.40 |
| Manual mode | `0x21010C02` | - | robot executes raw joystick velocity | api p.14 |
| Body height | `0x21010406` | 0 = Crawl, 2 = Normal | no crawl in off-road/stair/run; crawl only with Walk or Slope | api p.14-15 |
| Gait Walk | `0x21010300` | - | | api p.15 |
| Gait Slope | `0x21010402` | - | | api p.15 |
| Gait Off-Road | `0x21010401` | - | | api p.15 |
| Gait Stair (solid/grid/no-riser) | `0x21010405` | - | needs elevation map mode 3/4/5 | api p.15 |
| Gait Stair multi-frame | `0x2101040A` | - | needs map mode 20 | api p.15 |
| Gait 45 deg stair multi-frame | `0x2101040B` | - | needs map mode 20 | api p.15 |
| Gait L-Walk | `0x21010420` | - | RL gait; no height/terrain switching | api p.15; man p.44 |
| Gait Mountain | `0x21010421` | - | RL gait | api p.15 |
| Gait Silent | `0x21010422` | - | RL gait | api p.15 |
| Gait L-Stair | `0x21010424` | - | **not in api V1.0.5**; nb and dx only (dx: "V1.0.6, solid stairs only") | nb/hpp; dx/proto |
| Gait Run | `0x31010307` | - | **not in api**; nb only. dx: no separate Run code exists (D key on RC) | nb/hpp; dx/proto |
| RL sit / RL stand | `0x21010222` / `0x21010223` | - | **undocumented legacy firmware**; dx later found only 0x21010202 executes on current firmware | dx/proto; dx/src |
| **Soft e-stop** | `0x21010C0E` | - | robot lies down immediately, joint protection | api p.16 |
| Save data + exit | `0x010C01` (jy_exe >= 2.2.45) / `18` (older) | - | saves last 5 min, exits motion program; robot should be lying | api p.16; man p.10 |

Sent to the **perception host** (.105):

| Name | Code | Port | Value | Source |
| --- | --- | --- | --- | --- |
| Velocity input source | `0x3101EE03` | 43899 | 1 joystick (assist mode), 2 navigation | api p.29-30 |
| Elevation map mode | `0x3101EE01` | 43899 | 3 solid, 4 grid, 5 stair w/o risers, 20 multi-frame (18 = preparation, reported only) | api p.31-35 |
| Avoidance mode | `0x3101EE02` | 43899 | 1 decelerate, 2 detour | api p.36 |
| Obstacle height threshold | `0x3100EE04` (as printed; likely typo for `0x3101EE04`, nb uses the latter; inferred) | 43899 | 1 = 8 cm, 2 = 28 cm | api p.36; nb/hpp |
| Query stream start | `0x3101EEFF` | 43899 | 0 | api p.67 |
| LIO on/off | `0x0BAA0001` | 60000 | 1 / 0 (robot must stand still) | api p.34, p.66 |
| Charge start/stop/clear | `0x91910250` | 3333 | 1 / 0 / 2 | api p.27-28 |
| Charge query | `0x91910253` | 3333 | 0 | api p.28 |

Feedback from perception host: `0x3100EE03` (vel source), `0x3100EE01` (map mode), `0x3100EE02` (avoid mode), 10 Hz, after any packet received [api p.30-36].
Charge states: 0x0000 idle, 0x0001 startup, 0x0002 charging, 0x0003 exit, 0x0004 power failure, 0x0005 charger connected ("be charging with the charger"), 0x1002-0x100A location/dock failures [api p.28-29; nb uses different names].

### 3.3 Basic state machine [api p.10, p.21]

`basic_state`: 0 Sitting, 1 Sit-to-Stand, 2 Initial Standing, 3 Torque-controlled Standing, 4 Stepping, 5 Stand-to-Sit, 6 Emergency Stop/Fall.
**16 = RL state** is in the state diagram (entered from Stepping by an L-Walk/Mountain/Silent gait command; left by any other gait command) and in nb/py/dx, but absent from the printed state table [api p.10, p.21; nb/hpp].
Any state can fall to 6. Path: `0 -(0x21010202)-> 1 -> 2 -(0x2101020A)-> 3 -(0x21010201)-> 4 -(RL gait cmd)-> 16`; `6 -(0x21010202)-> 0`.
`gait_state`: 0 Walk, 1 Off-Road, 2 Slope, 3 Run, 6 Stair, 7 Stair multi-frame, 8 45deg Stair multi-frame, 32 L-Walk, 33 Mountain, 34 Silent, (36 L-Stair: nb/dx only) [api p.22, p.42].
Field note (dx): after an RL stand, telemetry can keep reporting `basic_state=0`; and after e-stop `basic_state` often reverts to 0 while joints stay locked, so check `emergency_source`. nb does not handle either.

### 3.4 Axis scaling [api p.12-14 (formulas read from rendered pages)]

```
vx   =  (y / 32767) * v_max_fwd_or_bwd(gait)      if 655 <= |y| <= 32767 else 0     (y = cmd 0x21010130)
vy   = -(x / 32767) * v_max_lat(gait)             (x = cmd 0x21010131; body +y = left)
wz   = -(x / 32767) * w_max(gait)                 (x = cmd 0x21010135; +wz = CCW)
```
So for a ROS Twist (vx, vy, wz): `axis_x = vx/vmax*32767`, `axis_y = -vy/vmax_lat*32767`, `axis_w = -wz/wmax*32767`, clamp to +/-32767, zero if |axis| < 655 [nb/hpp].
Resolution: 1 axis count = vmax/32767; dead zone = 2.0 % of vmax (e.g. Mountain: 0.020 m/s fwd, 0.016 m/s lat, 0.024 rad/s yaw; Walk lateral 0.003 m/s).
50 Hz required; no axis packet for >1 s => command invalid (stepping: decelerate to zero; torque stand: pose returns to neutral) [api p.11-12].
Each axis is its own UDP packet, so one velocity tick = 3 packets. The py script scales with 30000 instead of 32767 [py].

### 3.5 Takeover / start-up sequence

Documented (vendor): register client in `network.toml`; start heartbeat >= 2 Hz; send `0x21020001` once; then commands [api p.9; man p.56-57].
Practical sequence implemented by nav_bridge `~/stand` (empirical, "real-robot behaviour takes precedence over the manual" [nb/README 12]):

| Step | Action | Wait / timing [nb/ae, nb/cpp] |
| --- | --- | --- |
| 0 | node start: latch control => heartbeat `0x21040001` every 200 ms + one `0x21020001` | `startup_acquire_control=true` |
| 1 | before each key command: "warm-up" = heartbeat pulses every 100 ms for 400 ms | kControlWarmupMs 400 / pulse 100 |
| 2 | if SOFT_ESTOP(6): `0x21010202`, wait LYING_DOWN 5 s, up to 3 attempts | |
| 3 | LYING_DOWN: `0x21010202`, resend every 4 s until 1/2/3/4 within 16 s; then wait 2/3/4 within 6.5 s | |
| 4 | INITIAL_STAND: `0x2101020A`, wait FORCE_STAND 5 s | |
| 5 | FORCE_STAND: `0x21010201` (resend 3 s), wait STEPPING within 8 s | |
| 6 | send gait code (default `0x21010421` Mountain), wait `basic_state==16 && gait==33` within 12 s | |
| 7 | after stand: repeat `0x21010C02` (manual mode) every 200 ms up to 20 s until `is_nav_mode==0` | |
| 8 | axes at 50 Hz only while (STEPPING with any supported gait) or (RL_MODE with L_WALK/MOUNTAIN/SILENT/L_STAIR) and no service running | `isCmdVelCompatibleState` |
| lie | RL: `0x21010202`; STEPPING: `0x21010201`; then wait stable stand 3 s, `0x21010202`, wait LYING_DOWN | up to 12 s |

Toggle commands (0x21010202, 0x21010201) are non-idempotent; nb retries them only after observing no state change. dx rule: do not send axes during Sit-to-Stand/Stand-to-Sit/e-stop (stand-up is hardened into a hard motion otherwise) and pause axes ~300 ms after a stand/sit toggle [dx/proto].
Only one 0x21 source at a time: dx measured that the original 2.4G handheld and a second heartbeat+zero-axes source cancel each other (both lose control) [dx/sys].

### 3.6 Per-gait speed limits (normal body height) - three sources disagree

api App. B (table read from the rendered page [api p.65]) in m/s, rad/s: forward / backward / lateral / yaw

| Gait | State | **api** | nav_bridge `getGaitSpeedLimit` | dogx30 `LimitsOf` (fwd/lat/yaw) |
| --- | --- | --- | --- | --- |
| Walk | 0 | 1.2 / 0.75 / 0.15 / 0.45 | 1.2 / 0.7 / **0.75** / 0.5 | 1.5 / 0.15 / 0.45 |
| Slope | 2 | 0.7 / 0.75 / 0.25 / 0.5 | **0.3 / 0.3 / 0.1** / 0.5 | 0.7 / 0.25 / 0.5 |
| Off-Road | 1 | 0.3 / 0.4 / 0.1 / 0.5 | 0.3 / 0.3 / 0.15 / 0.45 | 0.3 / 0.1 / 0.5 |
| Stair (solid/grid/no-riser) | 6 | 0.3 / 0.3 / 0.2 / 0.8 | same | 0.3 / 0.2 / 0.8 |
| Stair multi-frame | 7 | 0.6 / 0.3 / 0.2 / 0.8 | 0.3 / 0.3 / 0.2 / 0.8 | 0.6 / 0.2 / 0.8 |
| 45deg Stair multi-frame | 8 | 0.3 / 0.2 / 0.2 / 0.8 | 0.3 / 0.3 / 0.2 / 0.8 | 0.3 / 0.2 / 0.8 |
| L-Walk | 32 | 1.0 / 1.0 / 0.5 / 1.2 | same | 1.5 / 0.5 / 1.2 |
| **Mountain** | 33 | 1.0 / 1.0 / 0.8 / 1.2 | same | 1.5 / 0.5 / 1.2 |
| Silent | 34 | not listed | 1.0 / 1.0 / 0.8 / 1.2 | 1.0 / 0.5 / 1.0 |
| L-Stair | 36 | not listed | 1.0 / 1.0 / 0.8 / 1.2 | 0.3 / 0.2 / 0.8 |
| Run | 3 | not listed | 0.6 / 0.25 / 0.4 / 0.5 | 2.5 / 0.8 / 1.2 |

Crawl height: Walk and Slope forward max = 50 % [api p.65]. The nb rows for Walk/Slope/Off-Road look shifted by one table column relative to the PDF (inferred: transcribed from garbled text).
Consequence: the robot always maps axis to the *true* vmax. If nb's vmax < true vmax, a requested speed up to nb's vmax already commands full axis
(Slope: ask 0.3 m/s, robot does up to 0.7). If nb's vmax > true, the robot is slower than requested (Walk lateral: ask 0.15, robot does 0.03). Which of api vs dx is correct for RL gaits on the current firmware is unverified (gap).

### 3.7 Telemetry (robot -> registered IP:port; all type 1 complex, natural LP64 alignment, **not packed**)

| Code | Struct | Size | Rate | Contents [api p.17-26] |
| --- | --- | --- | --- | --- |
| `0x1008` | RcsData | 88 B | 200 Hz | `char robot_name[15]`@0; `int32 current_mileage`@16 (cm); `total_mileage`@20; `int64 current_run_time`@24, `total_run_time`@32, `current_motion_time`@40, `total_motion_time`@48 (s); `float joystick[4]`@56 (LX,LY,RX,RY, [-1,1]); `rcs_state[10]`@72: [0]=`is_nav_mode` (0 manual,1 non-manual), [1]=`emergency_source`; `uint32 error_state`@84 |
| `0x1009` | MotionStateData | 60 B | 200 Hz | `u8 basic_state`@0; `u8 gait_state`@1; pad2; `max_fwd/bwd_vel`@4,@8 (deprecated); `float leg_odom_pos[3]`@12 = {x m, y m, yaw rad}, world frame, origin = robot position at start-up, axes = body axes at start-up; `float leg_odom_vel[3]`@24 = body-frame {vx, vy, wz}; `robot_distance`@36 (cm); `touch_state`@40 (reserved); `control_state`@44 (bits narrow_walk, pose_safe_flag, joint_limit_flag; reserved); `task_state[10]`@48: [0] deprecated, [1] `pos_ctrl_state` 0 exit,1 init,2 executing,3 done,4 aborted |
| `0x100A` | ControllerSensorData | 184 B | 200 Hz | IMU(40 B): `int32 timestamp`(unit not given), `roll,pitch,yaw` **degrees**, `omega xyz` rad/s, `acc xyz` m/s2; then `joint_pos`@40, `joint_vel`@88, `joint_tau`@136 each float[12] (FL,FR,HL,HR x HipX,HipY,Knee) |
| `0x100B` | ControllerSafeData | 68 B | 1 Hz | `float motor_temperature[12]`@0; `u8 driver_temperature[12]`@48; `float cpu_temp`@60; `float cpu_freq_MHz`@64 |
| `0x21050F0A` | BatterySensorData | 40 B | 0.5 Hz | u16 voltage@0 (V), i16 current@2 (A before deeprcs 1.10.19, 10 mA after), u16 remaining@4, nominal@6 (10 mAh), cycles@8, production_date@10, balanced_low@12, balanced_high@14, protected_state@16; u8 software_version@18, **battery_level@19 (%)**, mos_state@20, battery_quantity@21, battery_ntc@22; pad; `float battery_temperature[4]`@24 |
| `0x11050F08` | simple (type 0) | 12 B | on change | body-height state in `paramters_size`: **-1 Crawl, 0 Normal** (note: command uses 0/2) |

`error_state` bits (LSB first): 0 imu_error (IMU timeout), 1 wifi_error (**heartbeat timeout**), 2 driver_heat_warn, 3 driver_error, 4 motor_heat_warn, 5 battery_low_warn,
6 battery_heat_warn, 7 gpio_error, 8 cpu_heat_warn, 9 cpu_freq_warn [api p.18-19].
`emergency_source`: 1 tail soft e-stop, 2 autonomous during motion, 3 autonomous during navigation, 4 autonomous during charging, 5 triggered by client, 6 triggered by RAS [api p.19].
dx field note: sources 2/3 appear during gait switches without a real e-stop; real lock = `basic_state 6` or source 1, 4-6 [dx/proto].
Sizes match dx `static_assert`s and my offset calculation; the original packed layout in the py script differs (see 3.8).

### 3.8 Inconsistencies between docs, nav_bridge, py script

| # | Item | api | nb | py / dx | Impact |
| --- | --- | --- | --- | --- | --- |
| 1 | Per-gait vmax | App. B | wrong rows (3.6) | dx higher RL values | wrong velocity scaling |
| 2 | RcsData parse | natural alignment, 88 B | natural, `int64` times, ok | **py**: `<15siiiiii4f10BI` = 69 B, no padding, 32-bit times => mileage/error offsets wrong | do not copy py parser |
| 3 | BatterySensorData | 40 B natural | declared `packed` => 39 B, `battery_temperature` at offset 23 instead of 24 (unused field, so level/cycles still correct) | dx 40 B | temps wrong if used |
| 4 | Axis full-scale | 32767 | 32767 | py 30000 | 8 % lower speed in py |
| 5 | Body-height values | cmd 0/2, state -1/0 | consistent with api | py same | easy to confuse |
| 6 | basic_state 16 / gait 36 / Run & L-Stair codes | absent from V1.0.5 text (RL in diagram only) | present | dx cites V1.0.6 | firmware-version dependent |
| 7 | Heartbeat rate | >= 2 Hz | 5 Hz | py 2 Hz, dx 10 Hz | py has zero margin |
| 8 | `/cmd_vel` stale handling | n/a | holds last value 5 s | dx watchdog 300 ms | see 5 |
| 9 | Perception-host IP | .105 (api) | .105 charging | brief/rob7: ROS master .106 | naming only |
| 10 | Obstacle-height code | `0x3100EE04` | `0x3101EE04` | dx copies api | verify on robot |
| 11 | nb `publish_tf` | | code default `true`, yaml `false` | | yaml wins when launched with params |
| 12 | Stand code | `0x21010202` | `0x21010202` | dx early used `0x2101022x` | fine on current firmware |
| 13 | Charge-state naming | api 0x0004 "power failure", 0x0005 "charger plugged" | names differ (`pile_error`, `safety_warning`) | | cosmetic |

---

## 4. How nav_bridge works

### 4.1 Interfaces (ROS2 node `nav_bridge_node`, package `nav_bridge`, C++, ament_cmake) [nb/README; nb/cpp]

| Kind | Name | Type | Notes |
| --- | --- | --- | --- |
| Sub | `/cmd_vel` | geometry_msgs/Twist | queue 10; cached atomically; **any message latches control ownership** (starts heartbeat) |
| Pub | `/imu/data` | sensor_msgs/Imu | frame `imu_link`; stamp = `this->now()` at UDP receive; orientation = RPY(deg->rad) quaternion; angular vel, accel copied; **no covariance set (all 0)**; ~200 Hz |
| Pub | `/leg_odom` | nav_msgs/Odometry | `odom` -> `base_link`; stamp = receive time; x,y,yaw only (z=0, roll=pitch=0); twist = body-frame vx,vy,wz; **pose and twist covariance all zero**; 200 Hz |
| Pub | `/robot_basic_state`, `/robot_gait_state`, `/robot_body_height_state`, `/charge_manager_state` | std_msgs/Int32 | raw codes (16/36 included); charge state polled every 1 s |
| Pub | `/robot_sum_odom` (Float32, km = total_mileage/100000), `/robot_speed` (Float32, vx m/s) | | |
| Pub | `/battery/level` (UInt8), `/battery/cycles` (Int32), `/battery_text` (rviz_2d_overlay_msgs/OverlayText) | | needs `rviz_2d_overlay_msgs` (not in our Main Dockerfile) |
| Pub | `/inspection_task_hub/heartbeat/nav_bridge` | diagnostic_msgs/DiagnosticArray | 1 Hz liveness, status always OK |
| TF | `odom -> base_link` | | only if `publish_tf` true (yaml: false) |
| Srv | `~/stand`, `~/lie`, `~/soft_estop`, `~/release_control` | std_srvs/Trigger | `stand` blocks for tens of seconds (worst case ~45 s from the timeouts in 3.5); target = RL_MODE + `stand_target_gait`; `stand` also forces manual mode afterwards |
| Srv | `~/set_gait`, `~/set_body_height`, `~/charge_command` | rcl_interfaces/SetParameters | param names `gait` (int or name), `body_height` (0/2 or CRAWL/NORMAL), `charge_command` (0-3 or start/stop/reset/query). Gait switch waits up to 8 s (RL) / 12 s (non-RL); crawl only from WALK/SLOPE |
| Params | `motion_host_ip` 192.168.1.103, `motion_host_port` 43893, `local_recv_port` 43897, `heartbeat_interval_ms` 200, `cmd_vel_rate_hz` 50, `cmd_vel_timeout_ms` **5000**, `startup_acquire_control` **true**, `stand_target_gait` 33, frames `imu_link`/`odom`/`base_link`, `publish_tf`, charge: host 192.168.1.105, port 3333, cfg port 43899, local 49004, `enable_charge_state_query` true | | [nb/yaml] |

Threads: separate std::thread heartbeat loop (survives blocking services), receive loop (50 ms poll), ROS wall timer at 50 Hz for axes, charge receive/query threads.
Connection: marked lost after 2 s with no UDP packets (log only; heartbeat continues) [nb/cpp].
Velocity path: `/cmd_vel` -> per-gait clamp to +/-vmax (silent saturation) -> axis ints (+vy, +wz sign flip) -> 3 simple UDP packets/20 ms; stale input => zero axes (not release).
Backward vs forward uses `max_backward` when vx < 0. State gating: no axes unless the robot is in a compatible state or a service is running.
Does **not** do: non-manual mode, velocity-source switching (except before charging), elevation-map settings, stairs terrain-mode coordination, publishing joint states, motor/driver temperatures, error bits (logs only), battery voltage/current/temps, `emergency_source`, e-stop state topic.
Charging: START sends `0x3101EE03=2` to .105:43899 then `0x91910250=1` to .105:3333; if in RL mode first exits RL via `forceStand()`; STOP then returns to manual mode [nb/CHARGE].

### 4.2 Known limitations / TODO-like statements in the repo [nb/README 13]
- only motion host bridged; `percept_udp_` used only for charging; takeover is an engineering heuristic, no explicit ack; stand/lie paths tuned to current firmware and "may need revisiting if firmware changes"; robot name usually empty; unhandled codes may appear. Behaviour in conflict with manuals: "trust real state feedback".

### 4.3 Changes needed for our framework (recommendations)

1. **Register the dev PC** in `network.toml` on .103 (`ips`/`ports` pairs) and restart the motion program; otherwise zero telemetry (nb does not mention this).
2. **Fix the speed table** to api App. B, and decide per-gait limits by measurement; add a hard config clamp (`max_vx/vy/wz`) independent of gait; make saturation visible (warn).
3. **Staleness**: set `cmd_vel_timeout_ms` to 300-500 ms (default 5000 ms keeps driving on stale commands); add accel/jerk limiting, and zero axes if `/cmd_vel` has not arrived for N ms.
4. **Control ownership**: set `startup_acquire_control=false`; add an explicit `acquire`/`enable` service. Today any `/cmd_vel` or startup starts heartbeats, which fights the original 2.4G handheld (single 0x21 source rule, 3.5).
5. **Timestamps/clocks**: stamp IMU/odom from a synchronised clock (chrony/PTP between PC, .103, .106) or add the robot's time; current receive-time stamps vs robot-stamped `/lidar_points` (ROS1 host clock) will not align for SLAM/deskew.
6. **Covariances and frames**: fill pose/twist/IMU covariances (see 7); `/leg_odom` has no z, roll, pitch; give it a distinct name (e.g. `/x30/leg_odom`), keep `publish_tf=false` when SLAM owns `odom->base_link`; add static TFs `base_link->{imu_link, lidar_link,...}` from man p.67 (needs calibrated angles).
7. **Namespace collisions**: nb publishes `/imu/data`, `/leg_odom`, `/cmd_vel`, `/battery/level`, `/charge_manager_state`, same names as the robot's ROS1 topics; remap to `/x30/...` to avoid accidental bridging loops. Do not bridge `/cmd_vel` to the ROS1 master unless intentional (dual-path risk, 2.3).
8. **Expose missing state**: publish `error_state` bits, `emergency_source`, `is_nav_mode`, battery voltage/current/temps, joint states, motor/driver temps, and an explicit "link-alive" diagnostic; use `emergency_source` and `basic_state==6` as latched safety input to the planner.
9. **Obstacle protection decision**: nb stays in manual mode (no elevation-map avoidance, inferred). To use the robot's collision layer: `0x21010C03` + `0x3101EE03=2` (to .105:43899) + publish velocity as the robot's navigation `/cmd_vel` (or `/cmd_vel_corrected`) on the ROS1 side [api p.44-45, p.62-63]; would need new code and validation; keep UDP path as default.
10. **Dependencies**: add `ros-humble-rviz-2d-overlay-msgs` or drop `battery_text`; use sensor-data QoS for 200 Hz topics; guard long-blocking services with a multi-threaded executor.
11. **Gait policy**: default MOUNTAIN is an RL gait: no body-height/terrain changes; stairs need other gait + terrain-mode coordination (not in nb). Decide whether VLN stays on RL gait only.
12. **Sim adapter**: provide a Gazebo node exposing the same topics/services (cmd_vel in; leg_odom, imu, states out) so the framework is unchanged.
13. Review `tf` race: nb applies `this->now()`; under `use_sim_time` mixing real nb with sim is not valid.

---

## 5. Safety-relevant facts

| Topic | Fact | Source |
| --- | --- | --- |
| Heartbeat timeout | motion program "assumes the connection is disconnected" if heartbeat (>= 2 Hz) stops; sets `wifi_error` bit; degrade chain from `overtime.toml [Wifi]`: `StopMotion` (s, >0) < `KeepStill` < `Sitdown` (each from disconnection). **Default values not documented** (read from .103 `~/jy_exe/conf/overtime.toml`) | api p.9, p.19; man p.9 |
| Axis timeout | > 1 s without axis command => invalid; stepping decelerates to zero | api p.11-12 |
| UDP stops entirely | robot degrades per overtime.toml; it does not hold last velocity beyond 1 s of silence | (inferred from the two rows above) |
| Telemetry stops | no effect on robot; client must detect (nb: 2 s) | nb/cpp |
| Soft e-stop | `0x21010C0E`: lies down at once, joint protection; recover with `0x21010202`; sources 1-6 reported; tail e-stop, handheld stop button, or SW1+SW4 up | api p.16, p.19; man p.36 |
| Hardware | handheld stop button / "SW1 and SW4 up" for runaway; tail soft e-stop (source 1); whether the handheld e-stop works while another source sends heartbeats: **untested** | man p.36; dx/sys |
| Fall | `basic_state 6` ("e-stop/fall"); TCP 30000 additionally reports 7 = fall | api p.21; dx/proto |
| Low battery / thermal | `battery_low_warn`, `battery_heat_warn`, `motor_heat_warn`, `driver_heat_warn`, `cpu_*` bits only; robot behaviour on low battery (auto-sit, speed derate) **not documented** | api p.19 |
| Obstacle avoidance | elevation-map collision layer "always enabled" for navigation modes, **except composite-ground sections**; modes decelerate/detour; thresholds 8/28 cm; correction/braking times `/slow_t`, `/stop_t` (defaults not documented) | man p.45; api p.29-37, p.53-55 |
| Navigation rule | all people >= 2 m from robot during autonomy; avoid sharp terrain; blind-area obstacles | man p.36 |
| Payload | `enable_twist` must stay off with payload (fall risk) | man p.9 |
| Stairs | needs correct elevation-map mode first, otherwise gait command silently ignored; multi-frame mode only when standing still, 10 s timeout | api p.15, p.33-35 |
| Crawl | only Walk/Slope; never switch to crawl in off-road/stair/run | api p.15 |
| Gait lockouts | RL gaits (L-Walk/Mountain/Silent): no body-height or terrain switching | api p.15; man p.44 |
| Toggle commands | 0x21010202, 0x21010201 are toggles; duplicate sends reverse the action | api p.10; dx |
| Mapping | do not agitate/block lidar; stand still 5-10 s before mapping | man p.15, p.26 |

### 5.1 Recommended velocity limits for indoor autonomy trials (engineering recommendation, not vendor values; (inferred))

Assume gait MOUNTAIN (RL), nb with the fixed table. Start in a cleared area >= 3 m radius, handheld in reach, people >= 2 m.

| Stage | vx fwd | vx back | vy | wz | Notes |
| --- | --- | --- | --- | --- | --- |
| Bring-up / teleop through `/cmd_vel` | 0.2 | 0.1 | 0.1 | 0.3 | verify signs, latency, stop distance |
| Indoor autonomy (initial) | **0.4** | **0.15** | **0.2** | **0.5** | <= 40 % of vendor max (1.0/1.0/0.8/1.2) |
| Indoor autonomy (after validation, open rooms) | 0.6 | 0.2 | 0.3 | 0.8 | only after measured stopping distance and clutter test |
| Constraints | accel <= 0.5 m/s2, `/cmd_vel` timeout <= 0.5 s, no vy while vx > 0.5, no commands during stand/lie/gait services, no stairs/ramps | | | | |

Obstacle clearance: robot has no protection on the UDP path; the VLN stack must stop within sensing range using at least the measured braking distance (gap 8).

---

## 6. Camera and thermal camera access

### 6.1 What the vendor documents
Nothing. `api.txt` has **0** occurrences of camera/image/video/rtsp/thermal/infrared; `manual.txt` only draws "Wide-angle Camera" (USB 2.0 into the perception host) on its architecture figure
and mentions "save camera view" in the Controller app [api; man p.6, p.40]. `legacy`: "specific ROS packages for lidar and IMUs, but not for the visual camera".
No ROS image topic is listed in api p.38-63. No intrinsics, distortion, FOV, resolution, mounting pose, exposure/white balance controls, time sync.

### 6.2 Field-found body-camera stream (third-party, unverified on our unit)
`rtsp://192.168.1.105:8554/test` (perception host eth0; single stream, no sub-stream; path "looks like a factory default"; may change with firmware; probably the USB camera transcoded in software on the perception host (inferred); codec/resolution **not measured**, dx media.json guess: H.264 720p25 ~2 Mbps) [dx/proto; dx/media; dx/deploy/media.json]. dx tool `tools/rtsp_probe.py <url>` does RTSP DESCRIBE to read the SDP without ffmpeg.
A long-running pull competes for CPU with the elevation map on the perception host (stairs/avoidance safety) [dx/media].

Getting frames into ROS2 (no code, plan): (a) on the dev PC run a GStreamer/ffmpeg RTSP reader (e.g. `rtspsrc location=... protocols=tcp latency=0 ! decodebin ! videoconvert` into an `appsink`, or `ffmpeg -rtsp_transport tcp -i URL`)
inside the ROS2 container and publish `sensor_msgs/Image` (bgr8/rgb8) + `CameraInfo` from our own calibration; (b) alternatively `gscam2`/`image_transport` style node. Use TCP interleaved (more robust over Wi-Fi), expect extra latency from the robot-side transcode and jitter buffer; stamp frames at arrival and measure offset (gap).

### 6.3 Thermal camera
**Not part of the X30 in any supplied document.** The only thermal reference is in `dx`: a third-party **dual-light PTZ ball ("布控球")** mounted as payload on a separate payload network 192.168.10.0/24 (not the robot's networks):

| Item | Value | Status |
| --- | --- | --- |
| Visible stream | `rtsp://192.168.10.168:554/11` (dx default; also used for `ptz_vis_sub`) | configured default, not vendor spec |
| Thermal stream (template) | `rtsp://admin:PASSWORD@192.168.10.12:554/Streaming/Channels/201` main, `/202` sub (Hikvision-style path; IP and path are placeholders in `mediamtx.yml`, `ptz_ir_rtsp` empty by default) | unverified |
| Target encoder settings | main 1080p25 H.265 3 Mbps CBR (visible), thermal main "720p25" H.265 2 Mbps, sub 640x480 (480p10) H.264 400 kbps | desired settings, not measured sensor resolution |
| PTZ control | HTTP port 80, EVERET/"anv" CGI `/cgi-bin/anv/ptz_cgi`, factory user `admin` (password convention noted in dx/app-protocol.md); GB28181 for PTZ/intercom; SIP not implemented | [dx/media; dx/app-protocol.md] |
| Camera model, thermal sensor resolution (e.g. 256x192 vs 384x288), FOV, NETD, radiometry | **not documented** | |

If our robot has no such payload, there is no thermal source; if it does, obtain the model from the payload nameplate before simulating it.

---

## 7. Simulation fidelity checklist (Gazebo Harmonic)

Frames: body (base) frame origin = geometric centre of robot, x forward, y left, z up, yaw CCW positive [man p.66; api p.22-23]. Heights above ground are **not documented** (gap).

| Real X30 feature | Value from docs | How to reproduce in Gazebo Harmonic |
| --- | --- | --- |
| 4x Livox (Mid-360 class) mounts | positions in 1.2; angles 25/46/5/56 deg (axis definition unclear) | 4 `gpu_lidar` (gz-sensors, `gz-sim-sensors-system`, render engine ogre2) links on `base_link` at the listed xyz; pitch from figure (try both interpretations, validate against ground-plane/body-shadow in the real bag); `update_rate` 10; topic remapped then merged |
| Lidar scan pattern | Livox non-repetitive rosette; Mid-360: 360 deg x (-7 .. +52) deg, 200 k pts/s, 10 Hz => 20 k pts/lidar/frame, ~80 k merged [man p.69; bag]; range spec ~40 m @10 % reflectivity (up to ~70 m @80 %), min 0.1 m, range noise ~2 cm class (Livox datasheet values from memory, **verify**) | Harmonic has no non-repetitive scan; approximate with `gpu_lidar` horizontal 360 deg x ~720 samples (0.5 deg) and vertical 59 deg x ~28-32 beams (~20-23 k rays), `min 0.1`, `max 40`(or 70), gaussian noise stddev ~0.02 m; rotate sample phase each frame via a small custom system/plugin or accept regular pattern; reproduce point drop (tag) optionally |
| Merged cloud format | `/lidar_points` PointCloud2, frame `lidar_link`, fields x,y,z,intensity(f32), tag(u8), line(u8), timestamp(f64), point_step 26, 10 Hz [bag] | merger node: transform 4 clouds to `lidar_link` (= body frame, inferred, verify with bag), add `tag=0`, `line`=beam index, `timestamp`=per-point time offset, `intensity`=0-255 reflectivity; self-body occlusion from the visual/collision mesh |
| Per-lidar IMU | `/livox/imu_192_168_2_20x` | optional: 4 `imu` sensors at lidar origins, 200 Hz |
| Body IMU | 200 Hz, origin (0.13045, -0.001, 0.0149) m, axes = body, roll/pitch/yaw in deg + gyro rad/s + accel m/s2 [man p.67; api p.24] | `imu` sensor (gz-sim-imu-system), `update_rate` 200, frame `imu_link`; noise: **not documented**; placeholder from a static bag recording (Allan variance) before tuning |
| RGB camera | one wide-angle USB2.0, specs **not documented**; stream is likely 720p-class H.264 (inferred) | `camera` sensor, provisional 1280x720 @ 15-25 Hz, horizontal FOV ~100-120 deg (placeholder, **must be calibrated**), mount pose placeholder at head/front, `lens` distortion from calibration; JPEG-like noise optional |
| Thermal camera | **none on robot** (6.3) | only if payload exists: `thermal_camera` sensor (Harmonic supports it) with resolution/FOV of the actual module; else omit |
| Depth / GPS | none | omit |
| Leg odometry | 200 Hz, x,y,yaw in world frame from start pose, body-frame velocities; planar; drift **not documented**; origin at power-up [api p.20-22] | custom odom publisher: integrate ground-truth body twist, add per-step multiplicative scale error and yaw random-walk (values to be measured), z = 0, roll = pitch = 0; publish `/x30/leg_odom` at 200 Hz with realistic non-zero covariances; stamp with sim time |
| Velocity interface | axis -> gait vmax mapping, dead zone 2 %, 50 Hz, 1 s timeout [api p.11-14] | `gz-sim-velocity-control-system` or a custom holonomic base driver on the floating base; clamp to the active gait table (Mountain 1.0 fwd/1.0 back/0.8 lat/1.2 yaw); apply dead zone 0.02*vmax; zero after command timeout |
| Latency / dynamics | not documented; nb adds <= 20 ms tick + DDS; robot internal 200 Hz loop | add a first-order lag + 40-100 ms delay on commanded velocity as placeholder; replace with measured step response (gap) |
| Acceleration limits | not documented | apply 0.5-1.0 m/s2 / 1-2 rad/s2 placeholder; measure |
| Robot self-model | mass, size, COM, leg kinematics not in docs | URDF/SDF from measurements or vendor model; base can be a rigid body with 12 fixed joints if gait physics are not needed |
| Services/state | same ROS2 topics/services as nb (3-4) | sim adapter node; basic_state/gait_state transitions with realistic delays (stand ~ 10-16 s per nb timeouts) |
| Time | robot clocks unsynced with PC | run sim with `/clock`, keep one time base |

---

## 8. Gaps - measure on the real robot

1. Body mass, size, standing height (base to ground), leg geometry, CoM; battery capacity/runtime at 0.4 m/s; charge behaviour.
2. Lidar extrinsics: true rotation of each of the 4 lidars (and which IP -> which mount: 202-205); whether `lidar_link` == `base_link`; time sync across lidars; per-point `timestamp` semantics; intensity vs reflectivity; actual Livox model/firmware; whether /lidar_points already includes factory extrinsics.
3. Body camera: model, native resolution/FOV/frame rate, mounting pose and height, intrinsics/distortion, RTSP path/codec/bitrate/latency on our firmware, CPU impact on the perception host, USB vs other, any ROS image topic.
4. Thermal camera: existence on our unit; if present model/resolution/FOV/protocol/URL.
5. ROS: exact `ROS_MASTER_URI`s (one or two masters: .105 and/or .106), ROS distro/Ubuntu on each host, message md5 for custom types, SSH access/credentials for .105/.106, time sync (chrony) and clock offsets between PC and hosts, rates of all topics (esp. `/livox/imu_*`, `/cmd_vel_corrected`, `/robot_*`).
6. Motion host: firmware (`jy_exe`, `deeprcs`) versions; defaults of `overtime.toml` (StopMotion/KeepStill/Sitdown); `network.toml` after registering 192.168.1.131; whether `0x21010222/3`, `0x21010424`, Run code and the 0x3101EE04 vs 0x3100EE04 code work on this firmware; whether 16/36 states exist.
7. True per-gait speed limits and the effect of crawl; accel/decel limits; command-to-motion latency; step response; stopping distance at 0.2/0.4/0.6 m/s; max safe lateral/yaw rates; minimum non-zero velocity actually executed (dead zone behaviour in RL gaits).
8. Leg-odometry: scale error, yaw drift per 10 m / per rotation, behaviour on slip and in RL gait, wrap-around of yaw, reset at motion-program restart.
9. IMU: units/axes confirmation of roll/pitch/yaw (degrees, frame), noise/bias, gravity sign, `timestamp` unit, latency vs lidar.
10. Safety behaviour: what happens at low battery, over-temperature, heartbeat loss (observe overtime chain), 1 s axis silence in RL gait, e-stop recovery sequence; whether the handheld/tail e-stop works while nav_bridge heartbeats; behaviour with two simultaneous 0x21 sources; whether elevation-map avoidance is active under UDP manual mode (expect not).
11. Stairs/slope/step-height limits for RL vs MPC gaits; behaviour of Mountain gait on thresholds, door sills, carpets; narrow passage width (`narrow_walk` flag meaning).
12. Wi-Fi/Ethernet link: topology (router NAT vs bridge), bandwidth for 80 k pts x 10 Hz (point_step 26 => ~2.1 MB/frame, ~21 MB/s, ~166 Mbit/s raw) over Wi-Fi; recommended wired connection for the dev PC.
13. nav_bridge: validate fixed speed table; verify odometry frame/sign conventions (+y left, +yaw CCW) by test motions; verify `stand`/`lie` reliability, robot name, 5 s stale-command behaviour; battery temperature offset.
14. Navigation-host capabilities: whether `jy_cog` localization/occupancy map can be used as ground truth for VLN evaluation; map/pose topics in the 'map' frame and their TF tree (not documented).
