# sim -> worlds: label, temperature and registry conventions

The simulator (`x30_sim`) reads world SDF files; it never edits them. It copies the SDF to `/tmp/x30_sim`, makes relative URIs absolute and adds any
missing Gazebo systems (physics, user commands, scene broadcaster, sensors/ogre2, contact), sets `max_step_size` (0.001 legged, 0.005 kinematic) and
`real_time_factor`, and adds an `<atmosphere>` (ambient for the thermal camera) when the world has none. Do not add the `gz-sim-imu-system` to the world
(the robot model carries it; two instances duplicate every IMU message).

## Ground-truth objects (`/gt/objects`, `/gt/scene_info`)
A top-level `<model>` (or `<include>`) is a ground-truth object when its class is found, first match wins:
1. a child element `<x30:label>chair</x30:label>` (optional `<x30:instance>chair_3</x30:instance>`; the instance name defaults to the model name).
   Declare `xmlns:x30="http://x30.local/sdf"` on `<sdf>` (the simulator repairs a missing declaration when it parses, but Gazebo needs it).
2. a Gazebo Label plugin `<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>10</label></plugin>`; the integer is mapped through
   `x30_worlds/config/labels.yaml` (id -> class).
3. the model name (or the include URI) minus a trailing instance suffix (`chair_3` -> `chair`) when it is a class or synonym of `labels.yaml`.
Classes `wall`, `floor`, `ceiling` (and models named `ground*`) are structure: not objects, only used for `bounds` in `/gt/scene_info`.
Object pose = centre of the model's collision (else visual) bounding box in the world frame; size = its extent. Meshes without a primitive collision get
size 0.5 m (the simulator does not load meshes); give such models a box collision or an `<x30:size>`... (not implemented yet; ask if needed).
`SemanticObject.caption` carries the instance name, `attributes` carries `warm` (>= 305 K) / `hot` (>= 330 K) from the thermal temperature.

## Thermal camera
Set a temperature (Kelvin) with `<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal"><temperature>310</temperature></plugin>`
**inside the `<visual>`** element (not the model; verified with Harmonic 8.15). Untagged geometry is at the ambient `thermal.ambient_c` (22 C, uniform: the
simulator writes `<atmosphere type="adiabatic">` with gradient 0). Typical values: person 309-310 K, radiator 340-345 K, cold drink/extinguisher 290-295 K.

## Registry
`datasets/registry.yaml`: `worlds: {<id>: {sdf: <path relative to the x30_worlds package>, spawn: {x, y, z, yaw}}}`. `spawn.z` is added to the standing height.
Model lookup uses `<x30_worlds>/models`, `datasets/external/*/models` and `GZ_SIM_RESOURCE_PATH`. Fuel (https) URIs are not fetched (no network in the sim).
Worlds must be flat for the `kinematic` backend (the robot keeps a constant height); stairs/ramps need `legged`.
