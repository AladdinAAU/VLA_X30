# X30 simulation and navigation lab

This repository has two ROS stacks. The current Docker based stack is in `x30_vln/`. The older, local Gazebo/ROS stack remains under `src/`. The commands below describe the Docker stack that was launched in this session.

## What is running

The simulation uses a simulated, articulated DEEP Robotics X30 model and its simulated walking controller. It is not connected to a physical X30. The simulator runs ROS 2 Jazzy with Gazebo; navigation and the mode manager run in ROS 2 Humble in a second container. They communicate across Docker networking using ROS domain 10.

Data/control path:

```text
joystick GUI --> /cmd_vel_teleop --+
                                   +--> mode mux --> safety guard --> /cmd_vel --> simulated X30 controller
Nav2 ---------> /cmd_vel_nav ------+
Gazebo sensors and odometry -------------------------------------------> Nav2 and RViz
```

The mode manager selects `object_nav`; the mux arbitrates teleoperation and navigation. The safety guard can stop all motion (for example, when it detects excessive tilt).

## Docker services

Run these commands from the repository root in PowerShell. Docker Desktop must be running. If Docker reports a named pipe or access denied error, Docker Desktop is unavailable to the current Windows session; start/sign in to Docker Desktop and retry.

Compose definitions are in `x30_vln/docker/`:

- `compose.yml`: shared services and networks.
- `compose.models.yml`: model assets and volumes.
- `compose.sim.yml`: simulator service/profile.
- `compose.sim.gpu.yml`: GPU rendering and WSLg display mounts.
- `compose.real.yml`: hardware-oriented configuration; it was not used for this simulation.

Start/recreate the simulator with its GUI and legged controller:

```powershell
Set-Location .\x30_vln
docker compose -f docker/compose.yml -f docker/compose.models.yml -f docker/compose.sim.yml -f docker/compose.sim.gpu.yml --profile sim up -d --force-recreate sim
```

Compose creates the container and configures WSLg/GPU display access. The ROS launch is a separate step (the compose file's default launch is kinematic/headless). Run this after compose has started `x30-sim`:

```powershell
docker exec -d -e ROS_DOMAIN_ID=10 x30-sim bash -lc 'source /opt/ros/jazzy/setup.bash; ros2 launch x30_sim sim.launch.py world:=lab_01 locomotion:=legged gui:=true render:=gpu > /tmp/x30-sim.log 2>&1'
```

The `legged` setting starts articulated walking; `kinematic` is a simpler body-motion mode and does not provide the same legged behavior. To inspect service defaults and launch arguments, read `x30_vln/docker/compose.sim.yml` and `x30_vln/ros2_ws/src/x30_sim/launch/sim.launch.py`. After launch, `docker exec x30-sim bash -lc 'tail -n 100 /tmp/x30-sim.log'` shows startup logs. The Gazebo client can also be restarted inside the running container with `docker exec -d x30-sim gz sim -g` if only the GUI window closed.

Useful container checks:

```powershell
docker ps -a
docker logs --tail 100 x30-sim
docker logs --tail 100 x30-nav-fw
docker exec x30-sim bash -lc 'ros2 topic echo --once /x30/ready'
docker exec x30-sim bash -lc 'ros2 topic echo --once /x30/safety'
docker exec x30-sim bash -lc 'ros2 topic echo --once /x30/health'
```

## Navigation and modes

The simulator is on ROS domain 10. The nav container normally has its own domain setting, so the launch explicitly sets domain 10 and pins CycloneDDS to the nav container's IP on `x30_net`. In this session that address was `172.19.0.3`; Docker may assign a different address after recreating containers. Inspect it before reusing the temporary DDS XML:

```powershell
docker inspect x30-nav-fw
docker exec x30-nav-fw ip -brief addr
```

In the session, `/tmp/x30-nav-dds.xml` inside `x30-nav-fw` selected that container's `x30_net` interface. It was then launched with:

```bash
docker exec -d -e ROS_DOMAIN_ID=10 -e CYCLONEDDS_URI=file:///tmp/x30-nav-dds.xml x30-nav-fw bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; ros2 launch x30_navigation navigation.launch.py profile:=sim use_sim_time:=true odom_tf:=false limits:=legged > /tmp/x30-nav2.log 2>&1'
```

The modes were built in the nav container (needed only if its workspace does not already have the install artifacts) and started as follows:

```bash
docker exec x30-nav-fw bash -lc 'source /opt/ros/humble/setup.bash; cd /ws; colcon build --symlink-install --packages-up-to x30_modes'
docker exec -d -e ROS_DOMAIN_ID=10 -e CYCLONEDDS_URI=file:///tmp/x30-nav-dds.xml x30-nav-fw bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; ros2 launch x30_modes modes.launch.py profile:=sim use_sim_time:=true nodes:=mux,guard,manager > /tmp/x30-modes.log 2>&1'
```

Select navigation mode (or use `teleop` when available):

```bash
docker exec -e ROS_DOMAIN_ID=10 -e CYCLONEDDS_URI=file:///tmp/x30-nav-dds.xml x30-nav-fw bash -lc 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; ros2 service call /x30/set_mode x30_msgs/srv/SetMode "{mode: object_nav}"'
```

Check the navigation launch log with `docker exec x30-nav-fw tail -n 100 /tmp/x30-nav2.log`; the mode log is `/tmp/x30-modes.log`. ROS CLI commands should be run inside one of the ROS containers with the matching domain and DDS environment. The Humble and Jazzy containers can emit ROS serialization and TF timing warnings when communicating across distributions.

## Joystick

The on-screen joystick GUI source is the older package at `src/src/x30_teleop_bridge/x30_teleop_bridge/joystick_gui.py`. During the session it was copied into the simulator container as `/tmp/x30_joystick_gui.py` and launched publishing to the mux's teleoperation input:

```bash
docker cp src/src/x30_teleop_bridge/x30_teleop_bridge/joystick_gui.py x30-sim:/tmp/x30_joystick_gui.py
docker exec -d -e ROS_DOMAIN_ID=10 x30-sim bash -lc 'source /opt/ros/jazzy/setup.bash; python3 /tmp/x30_joystick_gui.py --ros-args -p topic:=/cmd_vel_teleop > /tmp/x30-joystick.log 2>&1'
```

This is a desktop GUI with on-screen controls and keyboard input, not a physical gamepad driver. It uses the same WSLg display forwarding configured for the simulator. The older package documents nominal limits of 0.4 m/s forward, 0.3 m/s sideways, and 0.5 rad/s turning. Safety guard limits still apply. If modes/mux are not running, the joystick's direct simulation input is `/cmd_vel`; with the mux running, use `/cmd_vel_teleop`.

## RViz

The navigation RViz config is Humble's default Nav2 config at `/opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz`. It was launched in a separate Humble RViz container because the Jazzy RViz installation did not have the Humble Nav2 display plugins used by this config. The map did appear, but that `x30-rviz` process exited with code 139 after TF queue/extrapolation warnings. Therefore RViz should currently be treated as unstable in this mixed-distro setup; the simulator's Gazebo GUI is separate and is not RViz.

## Health and recovery

Check these before trying to drive:

- `/x30/ready`: simulator/controller readiness.
- `/x30/safety`: whether the safety guard is stopping or scaling velocity, and its reason.
- `/x30/health`: sensor rates and real-time factor.
- `/cmd_vel_teleop`, `/cmd_vel_nav`, `/cmd_vel`: incoming teleop, Nav2, and final controller command streams.
- `/map`, `/scan`, `/odom`, `/tf`: navigation inputs and transforms.

The last observed state in this session had Nav2 lifecycle nodes active and a 600 by 600 `/map`, but the articulated robot had tipped. `/x30/safety` reported `stopped: true`, `scale: 0.0`, `reason: tilt`; the documented `/sim/set_robot_pose` reset attempt failed because the legged controller did not leave trotting. The simulator real-time factor was about 0.41, below real time. The safety guard was therefore blocking movement, and the system was not in a drive-ready state at that point. Check the current safety and health topics because container state can change between sessions.

## Stopping

Stop individual GUI windows/processes with Ctrl+C when they are attached to a terminal. Detached ROS processes can be stopped by stopping their container:

```powershell
docker stop x30-sim
docker stop x30-nav-fw
docker stop x30-rviz
```

To stop only simulator ROS processes from inside the simulator, the stack also contains `x30_vln/scripts/sim/kill_sim.sh`. Avoid running it until inspecting the script and confirming that stopping all simulator ROS nodes is intended. `docker compose ... down` tears down compose-managed services and is broader than closing a GUI.

## Repository map

| Path | Purpose |
|---|---|
| `x30_vln/docker/` | Docker Compose definitions for shared, simulation, GPU, model, and real-robot profiles |
| `x30_vln/ros2_ws/src/x30_sim/` | Gazebo world, robot model/controller, sensors, simulator launch |
| `x30_vln/ros2_ws/src/x30_navigation/` | Nav2 launch and navigation configuration |
| `x30_vln/ros2_ws/src/x30_modes/` | Command mux, safety guard, and mode manager |
| `x30_vln/ros2_ws/src/x30_msgs/` | Custom X30 ROS message/service definitions |
| `x30_vln/ros2_ws/src/x30_common/` | Shared configuration and utilities |
| `x30_vln/configs/navigation/` | Navigation profiles and parameters |
| `x30_vln/configs/modes/` | Mode and command-selection configuration |
| `x30_vln/datasets/registry.yaml` | Dataset/world registry, including `lab_01` |
| `x30_vln/scripts/` | Simulation and development helper scripts |
| `src/src/x30_teleop_bridge/` | Older joystick GUI source used for this session |
| `src/src/x30_sim_bringup/` | Older local ROS simulation launch stack; separate from the current Docker launch |

For more details, read the package guides, especially `x30_vln/ros2_ws/src/x30_sim/README.md`, `x30_vln/ros2_ws/src/x30_navigation/README.md`, and `x30_vln/ros2_ws/src/x30_modes/README.md`.
