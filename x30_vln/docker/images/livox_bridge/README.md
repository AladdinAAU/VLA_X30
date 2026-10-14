# Native Livox interfaces for the existing bridge

The robot already publishes ROS 1 `livox_ros_driver2/CustomMsg` on `/livox/lidar`.
The message definitions here match the vendored ROB7 packages and the
[Livox driver2 definitions](https://github.com/Livox-SDK/livox_ros_driver2/tree/master/msg).
They are intentionally interface-only packages with the same package name, so their ROS wire types match the robot.

`bridge.Dockerfile` builds the messages for Noetic and Foxy, then builds upstream `ros1_bridge` 0.9.7 with both packages sourced.
The ROS 2 package exports an explicit package mapping rule, required because `livox_ros_driver2` does not use the bridge's `_msgs` naming convention.
`humble.Dockerfile` builds the same ROS 2 interfaces for Humble and extends the framework's usual environment initialization.
That makes the native messages available to topic inspection, bag recording, Foxglove, and later FAST-LIO consumers.
The bridge entrypoint loads the appropriate overlays; its `ros1` helper loads only Noetic and the ROS 1 interfaces.

No sensor driver runs here and no point timing is reconstructed. The bridge translates the existing native messages.
The `compose.livox.yml` overlay selects these images; the base real-robot bridge image remains available separately.

See `docs/LIVOX_LIVE.md` for builds and the loopback regression proof. Build context is `docker/images/`.
Run `bash docker/images/livox_bridge/prepare-source.sh` from `x30_vln/` before building.
It prepares ignored local source under `upstream/`, preferring a cached image. Docker builds require no GitHub access.
`build-bridge.sh` refreshes the mapping factories and preserves timestamps of unchanged files for incremental C++ builds.
An optional `BASE_IMAGE` build argument can reuse a separately tagged prior Livox image's compiled workspace.
Message-only packages with this name should not be combined with a full driver package of the same name in one workspace.
