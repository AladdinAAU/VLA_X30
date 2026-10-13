# VLN / ObjectNav papers digest for the DEEP Robotics X30 project

Scope: engineering digest of 6 papers (7 text files) for a modular VLN framework on a DEEP Robotics X30 quadruped (ROS2 + ROS1 bridge, Docker, Gazebo first).
Source: plain-text PDF extractions in `reference/text/`. Conventions: statements are from the papers unless tagged `(inferred)` (my deduction) or `(from my knowledge)` (outside the papers; verify before relying on it). Tables in the extractions are often scrambled; where I could not attribute a number safely I say so.

File identification (checked from first page and full read):
- `CasualNav.txt` = **CausalNav** (arXiv 2601.01872, IEEE RA-L 2026, Tsinghua SIGS). Filename is misspelled.
- `qween.txt` = **the same paper as `Qwen-RobotNav_...txt`** (arXiv 2606.18112v3, Qwen Team, June 2026). It is NOT the Qwen3-VL technical report. `qween.txt` is the longer extraction (has conclusion, Fig. 16-18 captions, reference list); the other one stops after section 5.6.
- Consequence: the inputs contain **no Qwen3-VL-specific material on prompting, tool-calling, grounding output format or video input**, beyond what Qwen-RobotNav says about the Qwen3-VL architecture and its own prompt/tag conventions. Anything on those topics below is tagged `(from my knowledge)`.
- Also present in the folder but out of scope (X30 docs): `api.txt` (Jueying X30 API v1.0.5-0), `manual.txt` (X30 Pro Application Manual v2.0.4-0). I only used them for 2 interface facts, tagged `(api.txt)`.

---------------------------------------------------------------------------------------------------

## 1. Per-paper digest

### 1.1 CausalNav - scene-graph RAG navigation, dynamic outdoor (Tsinghua SIGS / Peng Cheng Lab)

**Problem / task type.** Language-directed *abstract goal* navigation in large dynamic outdoor campus/urban scenes ("I'm hungry, take me to find something to eat"; "take me to the Information Building"). Open-vocabulary, long range (100 m to >500 m). Not step-by-step VLN; closer to semantic goal retrieval + point-to-point navigation.

**Architecture (3 stages, all modules named).**
- Perception: YOLO-World (open-vocab 2D boxes + masks) on RGB -> ByteTrack for temporal association. 3D object position by projecting LiDAR points into the mask (K*H*P, calibrated LiDAR-camera extrinsics, ref [26]) and taking the centroid of a min-volume 3D box; authors state depth cameras perform poorly outdoors.
- Ego-motion: FAST-LIO2 (LiDAR-inertial odometry); RTK GNSS/INS used on the real robot for map-frame alignment and 5 cm accuracy.
- Dynamic filtering: CenterPoint (TensorRT) BEV detection + LIO-SEGMOT (asynchronous multi-object tracking fused with LIO) classify objects dynamic / static / quasi-static. A "spatial-temporal corridor" (stack of historical 3D boxes) is dropped from the graph when displacement exceeds a threshold over k steps.
- Map/memory: the **Embodied Graph** (see below). Local mapping with dynamic removal: RH-Map (3D region-wise hash map) gives the feasible region F.
- Reasoning: LLM does (a) hierarchical clustering summaries (cluster node text) and (b) hierarchical retrieval scoring. Local open models evaluated: phi4-14B, Qwen3-14B, Gemma3-12B, DeepSeek-R1-Distill-14B, vs GPT-4o.
- Global planner: if target not reachable via past trajectory -> coarse route from offline road map or Google Maps/Amap API; if connected via trajectory -> Dijkstra on ego-node graph. Output: waypoint list W.
- Local planner/controller: informed-RRT* in F, B-spline smoothing, orientation per point, then NMPC with discrete Control Barrier Function constraints (dynamic obstacles = predicted pedestrians/vehicles, safety radius d_safe). Control input u = [v, omega] (wheeled Hunter SE).

**Sensors.** Sim: RealSense D435i + 3D LiDAR on a Gazebo ground robot. Real: Agilex Hunter SE, RealSense D435i, RS-Helios LiDAR, RTK GNSS/INS, IMU, i9-13900H + RTX 4070.

**Action interface.** Continuous [v, omega] from NMPC-CBF; the LLM never emits motion, only selects graph nodes (target location).

**Map/memory structure.** Multi-level graph G: object nodes {class text c_i, 3D bbox, world position}; ego-vehicle nodes {position, velocity} appended each time movement exceeds d (edges = history trajectory, used for global path); building nodes (from offline map data, GNSS->local frame transform); cluster nodes {LLM-summarised text, centroid}. Clustering is bottom-up, recursive, using sim = (1-lambda)*exp(-haversine/sigma) + lambda*cosine(embeddings) (embeddings make "trash can" vs "garbage bin" merge). Updated online (Alg. 1) at 1 Hz; dynamic nodes removed; object nodes inserted/updated by tracking. Resolution: no raster map; metric positions + haversine distance.

**How the LLM consumes it.** Top-down retrieval: at each level the LLM scores each candidate node's text description against the query, softmax with temperature -> p(node|q); path score = product of p along parent->child chain (valid links only); if robot location known, hybrid re-rank = alpha*spatial_proximity + (1-alpha)*semantic path score. Ablation: accuracy/recall peak near 0.5 for the mixing weight and 1.5 for the sharpness (symbols lost in extraction). Prompt text is NOT shown in the paper.

**Datasets / sim / metrics.** No standard benchmark (authors: none suitable and open). Custom **Gazebo urban environment** with moving pedestrians and vehicles (scene not named/released as far as the text says). 25 random tasks x 10 trials, short/medium/large range. Metrics SR, SPL, Collision Count (CC), Trajectory Length (TL); success = within 10 m of target. Baselines: ViNT, NoMaD, GNM (given image topological maps) and CityWalker (given CausalNav's waypoints).

**Results.** Sim (small/medium/large): CausalNav SR 100/92/80, SPL 88.9/82.2/66.0, CC 0.2/0.6/1.2; CityWalker SR 100/85/80, SPL 82.4/73.6/68.3, CC 1.2/3.4/4.5 (table is scrambled; attribution to rows inferred from ordering). LLM table: SR 82-85% for the four open 12-14B models vs GPT-4o 88% (SPL 67.6-72.1 vs 75.3); best open = DeepSeek-R1-Distill-14B. Online graph updating: SR 78 -> 90, SPL 54.7 -> 80.1, CC 1.8 -> 1.1. Real: 130 m object-level task - only ViNT and CausalNav succeed; 512 m building-level task - only CausalNav succeeds (others collide).

**Compute / latency.** Sim PC: i9-14900K + one RTX 3090. Rates: detection/tracking/ego-motion 30 Hz, corridor filter 20 Hz, local mapping+planning 10 Hz, graph clustering/update 1 Hz. Per-cycle latency 105 ms (NoMaD 95, ViNT 150, GNM 110, CityWalker 180). Re-planning "within about 100 ms". LLM latency per query not reported.

**Limitations (stated).** Scalability of the graph, robustness to extreme lighting/weather, long-horizon consistency; future: graph compression, memory recall, multimodal fusion.

**Engineering take.** Best reference for: LiDAR-camera fusion for object localisation, dynamic-object filtering, and showing that 12-14B local LLMs suffice when retrieval is hierarchical. Weak for indoor/legged.

---

### 1.2 MapNav - Annotated Semantic Map as VLM memory (HKUST-GZ / BAAI)

**Problem / task type.** Instruction-following VLN in continuous environments (VLN-CE): R2R-CE and RxR-CE style multi-step route instructions; also "semantic instruction following" in real rooms (e.g. "move forward to the refrigerator and then turn right").

**Architecture (end-to-end VLM + a map builder).**
- Segmentation: Mask2Former on RGB (ablation: YOLOv8 SR 31.5 / SPL 29.6; MobileSAM 34.8 / 31.6; Mask2Former 36.5 / 34.3 on R2R val-unseen - segmentation quality matters).
- Mapping: RGB-D -> point cloud -> project to 2D top-down using pose; semantic mask aligned with point cloud writes into per-class channels.
- Memory: Annotated Semantic Map (ASM), below. Replaces historical RGB frames.
- VLM: LLaVA-OneVision style: SigLIP-so400m-patch14-384 vision encoder (hidden 1152) shared for both inputs, spatial-unpad patch-merge, **two separate 2-layer GELU MLP projectors** (observation vs map), Qwen2-7B-Instruct LLM (28 layers, ctx 32,768). Training: vision encoder frozen, projector + LLM tuned, 1 epoch, lr 1e-6, bf16, 8x A100 ~30 h (240 GPU-h, 500k step samples).
- Low-level: none in-model; VLM text -> regex -> discrete action.

**Sensors.** RGB-D + pose per step. Real robot: Unitree **Go2 Edu** with Intel RealSense D435i mounted 40 cm above the robot; RGB-D streamed to an A100 server; actions returned to the Go2. Source of the pose in the real system is **not stated** (paper only mentions "pose estimation errors").

**Action space.** Discrete: move forward / turn left / turn right / stop (Habitat low-level repertoire). Output is free text; case-insensitive regex with synonym sets (P_FORWARD = {"move forward","proceed",...}, P_TURN-LEFT = {"turn left","rotate left",...}, P_STOP = {"stop","halt","wait",...}). Step length / turn angle not given in the text.

**Map structure.** Tensor M of size C x W x H, C = C_n + 4 (C_n object classes). Channels 1-4: physical obstacles, explored area, agent current position, historical positions (trajectory). Agent starts at (W/2, H/2) each episode. ASM generation: connected-component analysis per class channel; regions above a minimum-area threshold get a centroid; the class name is drawn as **text at the centroid** on the rendered map image (plus drawn trajectory and agent marker). Map resolution / grid cell size / image size are not stated.

**How the VLM consumes it.** Input sequence V_t = [<TASK> instruction </TASK>; <OBS> E_t </OBS>; <MAP> E^M_t </MAP>] (special tokens), i.e. the **map is just a second image**. Fig. 4: GPT-4o and MapNav describe an ASM correctly (objects, own position, nearest object) but not the raw top-down or unlabeled semantic map; attention maps peak >0.8 on text labels for ASM vs <0.3-0.4 for the others (Vicuna-13B, VLM-Visualizer).

**Datasets / sim / metrics.** Habitat, **R2R-CE val-unseen (1,839 episodes)** and **RxR-CE val-unseen (1,517)** over Matterport3D. Metrics NE, OS, SR, SPL, nDTW. Training data ~1M step-wise samples: 300k expert (GT trajectories, R2R+RxR), 200k DAgger, 25k collision-recovery, + ~600k general VL co-training (LLaVA-OneVision style). Real: 50 instructions in 5 rooms (meeting room, office, lecture hall, tea room, living room), simple vs semantic.

**Results.** R2R-CE val-unseen: w/o ASM + current RGB: SR 27.1 / SPL 23.5; with ASM + current RGB: SR 36.5 / SPL 34.3; ASM + current + 2 history frames: SR 39.7 / SPL 37.2 (beats NaVid all-frames 37.4/35.9). Ablations (R2R val-unseen): RGB+Depth input is *worse* than RGB-only (VLM not pretrained on depth); original top-down map < unannotated semantic map < ASM. Real (SR, simple / semantic instructions, per room): MapNav 70-90% simple, 60-70% semantic; beats NaVid and WS-MGMap in all rooms.

**Compute / memory.** Constant 0.17 MB map memory vs NaVid frame memory growing to 276 MB at 300 steps; 0.25 s/step vs 1.22 s/step (-79.5%); A100 server for inference.

**Limitations (stated).** Segmentation errors under occlusion/lighting; dynamic scenes and partial observability; future: interactive navigation / manipulation. Not stated but visible: depends on accurate pose/depth; discrete 4-action space only; indoor only.

**Engineering take.** The cleanest recipe for turning a metric semantic grid into something a stock VLM can read (labelled top-down image). Fine-tuning was needed for action output; zero-shot ASM understanding was shown only qualitatively with GPT-4o.

---

### 1.3 NaVILA - VLA with language mid-level actions + visual locomotion RL (UCSD / USC / NVIDIA)

**Problem / task type.** Instruction-following VLN (VLN-CE) for legged robots (Unitree Go2, Unitree H1 in sim, Booster T1 humanoid real); long-horizon route instructions, stairs, slopes, outdoors. Optional voice front-end.

**Architecture (2-level, two timescales).**
- High level: **VILA-family VLM (8B)**: vision encoder + downsample/MLP projector + LLM; starts from VILA stage-2 checkpoint; all three parts unfrozen in SFT (1 epoch, lr 1e-4 cosine, warm-up 0.03). Image-based (no video encoder), 196 tokens per frame; tested 8-64 history frames; **real world uses 8 frames** for latency.
- Output: language mid-level action, e.g. "move forward 75 cm" / "turn right 30 degrees" / "stop"; parsed by a regular-expression extractor (authors: all outputs matched in all experiments).
- Conversion to velocity: actions map to fixed command velocities (forward 0.5 m/s, turn +/- pi/6 rad/s; extraction garbled, read as such) applied for a duration matching the requested distance/angle.
- Low level: **single-stage PPO locomotion policy** (Isaac Lab, ~60k FPS on RTX 4090): input = velocity command + proprioception history + 2.5D height map from LiDAR; output = 12 leg joint position targets (stiffness/damping -> torque). Height map: Unitree L1 LiDAR (360x90 FOV, 15 Hz) -> voxel grid, lowest point per voxel, max filter over last 5 scans (sim params: 32 channels, voxel 0.06 m, window X[-0.8,0.2] Y[-0.8,0.8] Z[0.05,0.5] m). LiDAR chosen because it sees glass and works in sunlight.
- No explicit map, no SLAM, no planner; history frames are the only memory.

**Navigation prompt (quote of structure).** "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations: <frames> and current observation: <frame>. Your assigned task is: <instruction>. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed." Answer template: "The next action is" + {move forward d cm | turn left x degrees | turn right x degrees | stop}. Latest frame is the "current observation"; the earlier frames are uniformly sampled with the first frame always kept. No new special tokens. Auxiliary prompt for trajectory summarisation: "Assume you are a robot designed for navigation. You are provided with captured image sequences: <frame3><frame6>... describe the navigation trajectory of the robot."

**Training data.** R2R-CE + RxR-CE in Habitat via shortest-path follower; consecutive actions merged (max 3, e.g. 25 cm + 25 cm -> 50 cm) and "stop" rebalanced; EnvDrop augmented instructions; trajectory-summarisation task; ScanQA (multi-view images); general VQA (LLaVA-1.5 mix, ShareGPT4V, Video-ChatGPT); **2K YouTube human touring videos -> 20K trajectories** (entropy-based sampling, MASt3R metric pose -> step actions, VLM captioning + LLM rewrite) - first demonstration that human video helps continuous VLN.

**Sim / benchmarks / metrics.** VLN-CE R2R/RxR val-unseen (Habitat); ScanQA validation; new **VLN-CE-Isaac** (Isaac Sim, R2R scenes, 1,077 traversable trajectories out of 1,839 val-unseen, Go2 and H1 with joint-level physics). Metrics NE, OS, SR, SPL, nDTW.

**Results.** R2R-CE val-unseen: SR 54.0 / SPL 49.0 (49.7 / 45.5 without YouTube data); RxR-CE 49.3 SR / 44.0 SPL with single RGB; cross-dataset (R2R-trained only) RxR SR 34.3 vs NaVid 23.8. VLN-CE-Isaac Go2: vision policy SR 50.2 vs blind 36.2 vs oracle controller 51.3. Low-level policy vs ROA: linear-vel error 0.066 vs 0.161, collision 0.81 vs 3.09. Real world: 25 instructions x 3 repeats, workspace/home/outdoor, **88% SR overall, 75% on complex instructions** (abstract); beats GPT-4o on SR/NE (per-environment table scrambled); same VLA without retraining on Booster T1; speech control ("Hey Robot", speech recognition per ref [70], Whisper (inferred)).

**Compute / latency.** VLA served on a single RTX 4090 at ~1 FPS (images transmitted to server); FP16: 594.6 ms, 18.5 GB at 1,737 context tokens; AWQ W4A16: 367.8 ms, 8.6 GB, SR 49.7 -> 48.2 (R2R-CE). Authors list on-robot deployment as future work. Training: 4 A100 nodes x 18 h for final stage; stages 1-2 inherited from VILA.

**Limitations (stated).** No error recovery after deviation (Fig. 13); spatial understanding/generalisation limited; image-VLM is compute-heavy; suggests larger realistic-sim training and explicit reasoning data. Code and data "to be released upon publication" - release status unverified.

**Engineering take.** The reference design for mode (a): strict, parseable language actions, async slow VLM + fast locomotion policy, history = a handful of frames. X30 would swap the locomotion policy for its own gait controller fed with Twist (see 4).

---

### 1.4 OrionNav - open-architecture LLM + scene-graph ObjectNav on a Go2 (NYU)

**Problem / task type.** Zero-shot real-world *object/room navigation* from natural language ("find a potted plant", "find a break room", "find an office") in unknown office buildings; replans as the scene changes.

**Architecture (all ROS2).**
- Localization/mapping: SLAM Toolbox (2D LiDAR SLAM, pose-graph optimisation, loop closure) using Hesai LiDAR + Go2 legged-inertial odometry -> 2D occupancy grid + pose.
- Segmentation: **FC-CLIP** (frozen ConvNeXt-L CLIP backbone, in-vocab + out-of-vocab classifiers, per-class text embeddings from templates and synonyms, synonyms mapped to canonical label). Edge-optimised: FP16, custom CUDA kernels for multi-scale deformable attention, softmax approximation, vectorised top-k. 560 object classes / 1,306 text categories, extendable at runtime.
- Object map: mask centers back-projected with depth + intrinsics -> world frame via SLAM pose; DBSCAN data association merges to instances; positions re-anchored after pose-graph updates.
- Hierarchical scene graph: DBSCAN on object coordinates -> room clusters; **LLaMA 3 (4-bit, on-device)** labels each cluster from its object-label list + candidate room set + guidelines. Tree: floor -> room -> object, node properties = level, label, instance id, location, parent/child.
- Planner: **GPT-4-Turbo (cloud API)** with action primitives `goto`, `search_room`, `explore_globally`.
- Navigation: ROS2 Nav2 stack - cost-aware A* global planner on inflated occupancy grid, **MPPI** local controller, **CBF-QP safety filter** on raw LiDAR points between MPPI and the velocity tracker; exploration = m-explore-ros2 frontiers weighted by distance Fd and size Fn (formula garbled in extraction; a weight of 50 is set to favour closer frontiers).
- Locomotion: Go2's built-in controller via velocity commands (inferred from "low-level velocity tracker").

**Sensors / compute.** Go2, top-mounted Hesai LiDAR, RealSense D455 (640x480 @ 30 Hz). Two Jetsons: AGX Orin (semantics) and Orin Nano (SLAM, nav, control). Segmentation ~10 Hz, object map 2 Hz, scene graph <= 0.5 Hz. Only the planner LLM is cloud.

**Prompt / tool schema (structure only).** System prompt states role, the three primitives, reasoning guidelines and output format; user part = task string + scene graph **as JSON** (rooms, objects) + feedback + action history. Output: chosen primitive + arguments (room/object ids from the graph) + a feedback explanation. Loop (Alg. 1): get latest graph -> LLM -> RobotApi.send(action) -> if `goto` succeeded ask the agent "Task done?" (verification) else feedback "Task has not been accomplished." / recent action logs. Empty graph -> first do a 360-degree scan.

**Experiments.** Four office environments; no standard dataset. Scenario A short-range object retrieval 32/36, B long-range 41/45, C room navigation 12/15; **total 85/96 = 88.5%**. Failure breakdown: 3 localisation (2 bad SLAM maps, 1 odometry), 3 exploration, 4 segmentation, 1 CBF; **zero planner/scene-graph failures**. Baselines: VLFM could not complete real tasks (needs accurate odometry, fixed map size; larger maps cut mapping rate 0.9 -> 0.5 -> 0.17 Hz; in HM3D sim with injected pose noise VLFM SR 52.5 -> 42.3); Clio could not run in real time on Orin and fragmented rooms; ablations on a changing scene (9 trials): Frontier-Search 1/9, Object-Map-Search (no room labels) ~56%, OrionNav 100%, with task times 2-5 min.

**Limitations.** No section; evident: relies on cloud GPT-4-Turbo, 2D SLAM only, office-style rooms, perception misses cause failures, exploration quality.

**Engineering take.** Closest to mode (b): perception -> map -> room graph -> LLM primitive -> Nav2. Shows the LLM reasoning layer was not the failure source.

---

### 1.5 SysNav - three-level cross-embodiment ObjectNav (CMU)

**Problem / task type.** Real-world building-scale ObjectNav, including attribute and spatial constraints ("find the white chair in the bedroom", "blue trash can in the classroom", "microwave near the refrigerator", "person sleeping on the yellow sofa"). Goal G = (category, constraints). Success when the robot reaches a pose where an object satisfying the constraints is observable within a distance.

**Architecture (high / mid / low).**
- Sensors: **360-degree panoramic RGB (Ricoh Theta Z1)** + **Livox Mid-360 LiDAR**; same on a Mecanum wheeled base, Unitree Go2, Unitree G1 humanoid (Go2/G1 built-in LiDAR also for their locomotion policy).
- Perception: YOLOv8x-WorldV2 (detection) + SAM 2.1 hiera-b+ (segmentation); 3D object point cloud reconstructed from mask + LiDAR + robot pose; instances merged.
- Scene representation R (3 layers + 5 edge types):
  - Room nodes: walls found by planar fitting + z-axis point distribution, dilate walls to split free space into disconnected components = rooms; attributes {2D top-down room mask, category (kitchen...), best-view RGB}. Room-room edges via doorways.
  - Viewpoint nodes: added when new coverage (voxels within d_cover) exceeds threshold; attributes {position, coverage region, panorama}. RGB stored only at viewpoints (compact).
  - Object nodes: {class, confidence, 3D point cloud, 3D bbox, best-view RGB, self-attributes}. **Attributes (color, state) and object-object spatial relations are inferred on demand by the VLM from stored images**, only when the task asks.
  - Edges: room-viewpoint, room-object (containment), viewpoint-object (visibility), object-object (only when required by the task).
- High-level reasoning: **Gemini-2.5-flash (cloud)**. Two query modes (structure below).
- Mid-level: rooms are the minimum decision unit for the VLM; inside a room *classical* exploration (TARE-style two-level planner, ref [35]): sample pose candidates in the local horizon scored by uncovered surface points, stochastic selection, TSP over candidates, repeated K times, rolling window local->global, TSP global path. Cross-room: early-stop and room-query (below).
- Low-level: CMU "base autonomy" stack (ref [36]; waypoint following, collision avoidance, terrain traversability) -> embodiment-specific motion control. Interface = waypoints; robot action a = (v, omega) continuous.

**Prompt structure (paraphrase of Fig. 2).** *Room-query:* "Task: find the white chair in the bedroom. Context: History: I have explored bedroom1, living1...; Candidates: Laundry1 with washing machine; Bedroom2 with chair, bed;... Question: which room should be explored next?" -> Answer: room id + Reason. *Early-stop:* "Task... Current state: exploring Living Room1 with sofa, pillow, table, chair...; New observation: discovered Study Room1 with white chair, table...; Question: should the robot stop exploring Living Room1 and switch to Study Room2?" -> Answer: Early Stop + reason. Contexts are built from room attributes + object lists + goal; trajectory included for room-query.

**Datasets / sim / metrics.** Habitat Challenge 2023 setup (640x480 RGB-D; 0.25 m forward step, 30-degree turn; depth -> point cloud with given pose): HM3D-v1 (2,000 episodes, 20 scenes, 6 categories), HM3D-v2 (1,000 eps, 36 scenes, 6 cats), MP3D (2,195 eps, 11 scenes, 21 cats), HM3D-OVON val-unseen (3,000 eps, 36 scenes, 49 cats): 8,195 sim episodes. Metrics SR, SPL; real: SPT = SR*(1 - t/T_timeout) and average time AT.

**Results.** Sim SR/SPL: HM3D-v1 63.7/30.5, HM3D-v2 80.8/37.2, MP3D 50.7/18.1, HM3D-OVON 54.9/26.1 (best on all; OVON +14.1 SR, +6.5 SPL over best baseline; SPL gain small due to deliberate over-coverage). Real: 190 episodes (78 qualitative, 112 quantitative on the wheeled robot, 10 categories, 3 buildings; timeouts 3 min easy, 4 min medium/hard). SR/SPT/AT: SysNav easy 100/83.8/29.1 s, medium 97.5/67.9/72.8 s, hard (3 rooms) 98.3/71.8/67.6 s; VLFM 58.3/47.5/25.6 SR; InstructNav 45.8/55.0/37.2 SR. Claimed 4-5x efficiency gain, first reliable building-scale ObjectNav.

**Compute.** Laptop i9-14900HX + RTX 4090 for everything except the cloud VLM. VLM latency not reported.

**Limitations.** No dedicated section. Authors' own caveats: VLMs have weak 3D spatial reasoning (hence VLM limited to room-level choices); over-coverage costs SPL; VLM is cloud.

**Engineering take.** Strongest evidence that "VLM chooses the room, classical planner explores inside the room" is robust; also the cleanest source for mode (b) logic and for LiDAR-centric (Livox) perception.

---

### 1.6 Qwen-RobotNav - parameterised navigation foundation model on Qwen3-VL (Qwen Team, `qween.txt` = same paper)

**Problem / task type.** One model for VLN (instruction following), PointNav, ObjNav (incl. open-vocabulary), target Tracking, and driving; plus an *agentic* layer (EQA, long-horizon search) where a stronger LLM planner calls the model as a tool. Sizes 2B/4B/8B (4B and 8B benchmarked).

**Architecture.**
- Backbone: Qwen3-VL (SigLIP-2 ViT with native dynamic resolution + 2D-RoPE, 2-layer MLP patch merger, DeepStack injection of multi-layer ViT tokens into early LLM layers). Fine-tuned end-to-end from the pretrained Qwen3-VL checkpoint.
- Action head: 4-layer MLP (hidden 512, GELU) on the LLM's final hidden state -> **K = 8 waypoints x (x, y, theta) = 24 numbers** in the robot frame; normalised to [-1,1] by per-dataset 99th-percentile scale factors; MSE loss. Loss L = L_traj + lambda*L_VL (lambda = 1) - joint with next-token loss on nav-reasoning VQA.
- Observation: multi-camera, multi-timestep RGB only (no depth, no LiDAR, no odometry in the model). **Task-adaptive observation encoding** with 4 inference-time knobs: token budget B in [2048, 4096], temporal decay gamma in [1,3] (gamma = 2 -> newest frame gets ~7.4x the oldest), per-camera weights w_c (default front/right/back/left = 2.0/1.0/0.5/1.0), frame sampling mode random vs latest. Constrained allocation with per-image floor b_min in [1,8] and ceiling b_max in [128,256] tokens; each image is rescaled to its token budget. Everything randomised in training so any setting works at inference.
- Identity via plain text: each step is "Time step t" followed by "Front View <image>", "Front Right View <image>", ... (descriptive camera names beat numeric azimuths). Embodiment via system preamble: "Imagine you are a robot programmed for navigation tasks" vs "...a car programmed for autonomous driving"; a new robot = new prompt template, no new parameters.
- Task modes: VLN, PointNav (coordinates or motion primitives such as "Move forward 2.0 meters", "Turn left 90 degrees"; 62K samples), ObjNav, Tracking.

**Agentic layer.** Upper planner (Qwen3.6-Plus, cloud) calls `nav_qwennav(L_i, task_mode_i, config_i)` returning waypoint trajectory W_i; other tools: object detection, scene understanding, semantic grounding (auxiliary, no waypoints). Planner heuristics: broad history (larger B, smaller gamma, random sampling) for search/revisit; small B, large gamma, latest frames for approach/tracking. A harness turns each rollout into a **trajectory-evidence record** `{subgoal, task_mode, config(B,gamma,m), progress, salient[], outcome, key_frames[#18,#31]}` and keeps an **evidence notebook** ("[step 47] Kitchen entered and searched; countertop and dining table checked. No mug observed. Corridor shelf remains a possible candidate region from key frame #12."). Two-level memory: single-episode evidence + persistent cross-episode memory; key frames can be re-fetched ("visual recall"). Tiers talk only in natural language.

**Training data (15.6M).** 85% trajectory / 15% VL. Instruction following 5.63M (VLN-CE R2R 1,491K + RxR 4,140K, MP3D scenes; Qwen-Image-Edit style-transfer of renders to photorealism; 3 LLM paraphrases per instruction); point-goal 984K (HM3D+MP3D, curriculum, 922K coordinate + 62K command primitives); object-goal 2.0M (skeleton-of-navmesh exploration trajectories with backtracking, VLM-annotated open-vocabulary goals, HM3D-OVON annotations); tracking 1,486K (EVT-Bench); driving ~3.2M (nuScenes 78K + OpenScene 3,138K); 40K text-to-video-generated samples (VLM filter, monocular pose, kinematic filter); VL: 1.0M general (VQAv2, RefCOCO, Objects365...) + 873K nav reasoning (NavGPT-2 style, 12 action classes, history/scene/progress/action chains) + 362K discrete multi-turn (CVDN, SOON, REVERIE, SRDF). Camera height randomised 0.5-1.5 m, HFoV 90-120 degrees, step speeds 0.05-0.25 m. 8B trained in 2,816 H100-GPU-h, peak lr 2e-5 (backbone) / 1e-4 (head).

**Results (selected).** VLN-CE val-unseen: panoramic 8B R2R SR 72.1 / SPL 66.6, RxR SR 76.5 / SPL 65.7; **monocular (single front camera)** 4B R2R SR 66.9 / SPL 60.5, 8B RxR SR 73.4 / SPL 63.5 (previous best monocular DualVLN: R2R SR 64.3, RxR SR 61.4; NaVILA: R2R SR 54.0, RxR SR 49.3). VLN-PE (flash controller): 8B SR 65.5. HM3D-v2 ObjectNav (RGB only) 4B SR 75.6 / SPL 30.6; MP3D 4B SR 52.2; HM3D-OVON 4B SR 57.7 / 60.1 / 53.1 (seen/synonyms/unseen, single front camera); EVT-Bench tracking rate 90.0; NAVSIM PDMS 91.4; EQA: HM-EQA 76.7 acc, 77% fewer steps (abstract). Real-world: only qualitative (Go2 in an exhibition hall, 21.78 m route; reverse-walk command; apartment; an agentic "green umbrella at Cotti Coffee" episode) - **no real-world success-rate table**.

**Deployment / latency.** 4B on Unitree Go2 built-in camera: edge on Jetson Thor (FP8 + TensorRT) 204 ms (4.9 Hz) vs remote server 196 ms (5.1 Hz) end-to-end incl. network; edge is steadier. No RTX 3090 numbers. How waypoints are converted to leg commands is **not described** in the text.

**Limitations.** No limitations section. Visible: tracking SR trails specialists (78.6 vs 86-87); lower SPL on OVON from "reach-first" exploration; driving closed-loop (AlpaSim) weak; no depth/LiDAR safety layer; cloud planner for agentic mode; weight/licence status not in text (project links: github.com/QwenLM/Qwen-RobotNav).

**Engineering take.** Candidate drop-in "mode (a)" executor and the template for a mode switch (task mode + context config); also a prompt/tag convention reusable with plain Qwen3-VL.

---

## 2. Cross-paper comparison (modules x papers)

| Module | CausalNav | MapNav | NaVILA | OrionNav | SysNav | Qwen-RobotNav |
|---|---|---|---|---|---|---|
| Task | Abstract-goal nav, outdoor | VLN-CE instruction following | VLN-CE instruction following (legged) | ObjectNav + room nav (real) | ObjectNav w/ attribute+spatial constraints | VLN / PointNav / ObjNav / Tracking / driving / EQA |
| Robot (real) | Agilex Hunter SE (wheeled) | Unitree Go2 Edu | Go2 (+ Booster T1; H1 sim) | Go2 | Mecanum wheeled, Go2, G1 | Go2 (qualitative); mobile robot (agentic demo) |
| Sensors | RGB (D435i) + LiDAR (RS-Helios) + IMU + RTK GNSS | RGB-D (D435i) + pose | Monocular RGB (+ Unitree L1 LiDAR for locomotion) | RGB-D (D455) + Hesai LiDAR + leg odometry | Panoramic RGB (Theta Z1) + Livox Mid-360 | Multi-view or single-front RGB only |
| Segmentation / detection | YOLO-World + ByteTrack; CenterPoint (LiDAR) | Mask2Former | none (VLM sees pixels) | FC-CLIP (FP16 optimised) | YOLOv8x-WorldV2 + SAM 2.1 | none (VLM) ; optional detector/grounding tools via planner |
| Localisation / SLAM | FAST-LIO2 (+RTK) | pose given (source unstated) | none | SLAM Toolbox 2D + leg odom | LiDAR-based (stack in ref [36]; SLAM algorithm not named in text) | none in model |
| Map / memory | Hierarchical graph: ego-trajectory, building, object, LLM clusters; RH-Map local voxel | Top-down multi-channel semantic grid -> labelled image (ASM), constant size | 8 history frames (uniform + first) | Object map (DBSCAN) + room tree (JSON) | Room / viewpoint / object graph with images | Frame history under token budget; evidence notebook (agentic) |
| Reasoning model | Local LLM 12-14B / GPT-4o (retrieval + summarising) | Qwen2-7B VLM (fine-tuned) | VILA-8B VLA (fine-tuned) | GPT-4-Turbo (plan) + LLaMA3 4-bit (room labels) | Gemini-2.5-flash (room-level) | Qwen3-VL 4B/8B (+ Qwen3.6-Plus planner) |
| Planner | Global: map APIs/Dijkstra; local informed-RRT* + B-spline | none (policy) | none (policy) | Nav2: A* + MPPI; frontier exploration | TARE-style TSP coverage in-room; VLM picks room | none (policy) ; planner LLM decomposes goals |
| Low-level | NMPC-CBF -> [v, omega] | discrete actions -> robot SDK (unspecified) | RL joint-position policy w/ LiDAR height map | MPPI + CBF-QP -> Go2 velocity | Base autonomy -> embodiment-specific control | unspecified; model emits 8 waypoints |
| Action interface | Continuous v, omega | 4 discrete actions as text | Text "move forward 75 cm" -> velocity | goto / search_room / explore_globally -> Nav2 | Waypoints -> (v, omega) | 8 x (x, y, theta) waypoints |
| Sim / benchmark | Custom Gazebo urban (not benchmark) | Habitat R2R-CE, RxR-CE | Habitat R2R/RxR-CE, ScanQA, VLN-CE-Isaac | HM3D noise test (VLFM); real offices | Habitat HM3D v1/v2, MP3D, HM3D-OVON | VLN-CE, VLNVerse, VLN-PE, HM3D/MP3D ObjNav, OVON, EVT-Bench, NAVSIM, EQA sets |
| Headline result | 512 m real long-range success | R2R SR 36.5 (current RGB + ASM) | R2R SR 54.0; real 88% / 75% complex | 85/96 real | 98-100% real SR; OVON SR 54.9 | R2R SR 72.1 (pano); RxR 76.5; mono R2R 66.9 |
| Compute | i9 + RTX 3090 (sim), 4070 laptop-class (real); 105 ms/cycle | 8xA100 train; A100 server | RTX 4090, ~1 FPS, 594 ms (FP16) / 368 ms (AWQ) | 2 Jetsons + cloud LLM | i9-14900HX + RTX 4090 + cloud VLM | Jetson Thor FP8 ~204 ms; 8B train 2,816 H100-h |
| LLM local or cloud | Local (shown viable) | Local server | Local server | Cloud planner, local room-labeller | Cloud | Model local; planner cloud |
| Stated limits | scale, weather, long-horizon | seg errors, dynamic scenes | no error recovery, compute | not formal (SLAM, perception) | VLM weak in 3D (design) | no real success rates; tracking SR |

---

## 3. Datasets and simulation worlds

### 3.1 Datasets / benchmarks mentioned in the papers

Licence/access items marked `(from my knowledge)` are not in the papers - verify the current terms.

| Name | What it is | Used by | Access / licence | Format | Loadable into Gazebo Harmonic? |
|---|---|---|---|---|---|
| **Matterport3D (MP3D)** | Real-building RGB-D scans (reconstructed textured meshes), scene base for R2R/RxR and MP3D ObjectNav (21 categories) | MapNav, NaVILA, SysNav (11 val scenes), Qwen-RobotNav | Terms-of-use agreement with Matterport, academic/non-commercial (from my knowledge) | Raw: textured .obj/.ply + jpg; Habitat: .glb + .navmesh + semantic .ply | Feasible but laborious: mesh -> SDF `<mesh>` for visual, trimesh/decimated collision, static model; huge textures; cannot redistribute |
| **HM3D** (Habitat-Matterport 3D; v1 / v2 = semantics v0.1 / v0.2) | Large indoor scan set; HM3D ObjectNav episodes (6 categories) | SysNav (v1: 2,000 eps/20 scenes; v2: 1,000 eps/36 scenes), Qwen-RobotNav (v2), HM3D-OVON | Matterport/Habitat access request, research use (from my knowledge) | .glb (+ .basis.glb), navmesh, semantic .glb, `scene_dataset_config.json` | Same as MP3D; scenes are noisy single meshes (holes, no explicit collision geometry (from my knowledge)), so lidar/contact behaviour must be validated |
| **R2R / R2R-CE** | Room-to-Room instructions; CE = continuous-environment re-episode set; val-unseen 1,839 episodes | MapNav, NaVILA, Qwen-RobotNav | Instruction data public; useless without MP3D scenes (from my knowledge) | gzipped JSON: scene_id, start position/rotation, goal, reference path, instruction text | Parseable for start/goal/instruction; only meaningful if the same MP3D scene is imported |
| **RxR / RxR-CE** | Longer multilingual dense-grounded instructions; val-unseen 1,517 (per MapNav) | MapNav, NaVILA, Qwen-RobotNav | As above | JSON | As above |
| **HM3D-OVON** | Open-vocabulary ObjectNav episodes on HM3D (val-unseen: 3,000 eps, 36 scenes, 49 categories per SysNav) | SysNav, Qwen-RobotNav | Needs HM3D | Habitat episode JSON | Goal strings reusable for "find the X" tests; geometry requires HM3D |
| **VLN-CE-Isaac** (NaVILA) | R2R val-unseen scenes in Isaac Sim, 1,077 traversable trajectories, Go2/H1 | NaVILA | Release status unclear (paper promises code/data) | Isaac Sim USD (from my knowledge) | No (Isaac); but the trajectory/instruction list could be re-used if MP3D access exists |
| ScanQA / ScanNet | 3D QA on scans (spatial understanding eval) | NaVILA | ScanNet needs terms-of-use (from my knowledge) | images + QA JSON | Not needed for Gazebo |
| EnvDrop, CVDN, SOON, REVERIE, SRDF | Augmented / auxiliary VLN data | NaVILA, Qwen-RobotNav | public (from my knowledge) | JSON | Training only |
| YouTube touring videos (2K -> 20K trajectories) | Human egocentric video for navigation pre-training | NaVILA | Not released as far as the text says | video | Training only |
| VLNVerse, VLN-PE, EVT-Bench, NAVSIM, nuScenes, OpenScene, AlpaSim/NuRec, HM-EQA, MT-EQA, EXPRESS-Bench | Benchmarks/training sets for physics VLN, tracking, driving, EQA | Qwen-RobotNav | Varies | Varies | Out of scope for Gazebo |
| Custom Gazebo urban world | CausalNav's sim (pedestrians + vehicles) | CausalNav | Not stated as released | Gazebo world | Native Gazebo (classic), name/availability unknown |
| Real-office / lab scenes | OrionNav, SysNav, MapNav, NaVILA real tests | all | Private | - | - |

Practical reading for this project (inferred): Habitat episode files are plain JSON and can be parsed for *instruction text + start pose + goal*, but the scene geometry must come from Matterport (access agreement). With no Matterport access, keep the *instruction style* (R2R/RxR phrasing, "walk past X, turn left, stop at Y") and author new episodes on Gazebo worlds.

Mesh-import notes `(from my knowledge; verify on the Harmonic install)`: gz-sim loads Collada (.dae), .obj, .stl and glTF/.glb meshes via gz-common; very large textured scans need decimation and separate low-poly collision meshes; for lidar simulation ray-casts hit collision/visual geometry, so check both; Livox non-repetitive scan patterns are not built into Gazebo Harmonic's `gpu_lidar` (community plugins exist for Classic) - plan to emulate with rotating multi-sensor configs.

### 3.2 Free Gazebo-compatible fallbacks (all `(from my knowledge)`; verify licence and Harmonic compatibility)

Indoor:
- AWS RoboMaker worlds (MIT-0, ROS packages with SDF models): small house (`aws-robomaker-small-house-world`, best for "find the chair/sofa/fridge" tests), hospital, bookstore, small warehouse. Written for Gazebo Classic: world files and some plugins need porting to `gz-sim` (SDF mostly reusable; replace classic plugins, check material scripts).
- TurtleBot3 simulation worlds (Apache-2.0): turtlebot3_world, turtlebot3_house (small, simple, good for smoke tests).
- Clearpath `clearpath_gz` worlds (office, warehouse, construction, pipeline; native for newer Gazebo).
- Gazebo Fuel (app.gazebosim.org): OpenRobotics model collections (furniture, "Cafe", "Depot", office/house sets, hundreds of chairs, sofas, tables, plants usable for object-goal tests) - check each item's licence (mostly CC-BY / CC-BY-SA / Apache).
- Free scan/synthetic scene sets with permissive terms, if a Habitat-style benchmark is wanted: ReplicaCAD (CC-BY), HSSD (non-commercial) and ProcTHOR/AI2-THOR houses (Apache-2.0) - these can be exported as glb/obj and wrapped into SDF.
Outdoor:
- Fuel worlds/models: Sonoma Raceway, Baylands (PX4 SITL) style terrains, DARPA SubT tunnel/urban/cave circuit worlds (SubT assets on Fuel), OSRF `citysim` (Classic; roads, pedestrians, vehicles - closest to CausalNav's setup).
- Heightmap + building models from Fuel for a campus-like scene (authoring effort).
Approach (inferred): start with small house + hospital (rooms/doors for room-graph logic), warehouse (large aisles), and 1 outdoor world; place `Fuel` object models of known categories and build a ground-truth object list for automatic ObjectNav success checks.

---

## 4. What to reuse (prioritised, 15 items)

Context: SLAM + segmentation + semantic map from 4 Livox + RGB + thermal; a local Qwen3-VL-class VLM on one RTX 3090 Ti (24 GB); planner; two switchable modes: (a) instruction following with mid-level velocity/language actions, (b) object navigation via semantic map + Nav2.

1. **Strict language-action grammar + regex parser (NaVILA).** Why: removes any action decoder, makes VLM outputs auditable, and is cheap to swap models behind. Hint: define actions `move forward <d> cm`, `turn left|right <x> degrees`, `stop` (optionally `move backward`), parse with case-insensitive patterns incl. synonyms (MapNav's synonym sets), reject/re-ask on parse failure, clamp d and x. Convert to a timed velocity profile on `/cmd_vel` (X30 `/cmd_vel` Twist and `/cmd_vel_corrected` elevation-map-corrected variant exist per `api.txt` 2.3; the motion host also needs periodic commands, axis command rate 50 Hz per `api.txt` 1.2.3 - verify which path the ROS bridge uses). Prefer closing the loop on SLAM/odom displacement instead of pure open-loop timing (inferred).
2. **Two-timescale architecture: slow VLM, fast controller (NaVILA, CausalNav).** Why: VLA ~1 FPS vs controller real-time; CausalNav runs 30/20/10/1 Hz tiers. Hint: VLM node publishes an action request asynchronously; a motion executor node owns the robot, enforces velocity/accel limits, and stops on timeout, collision-risk or new VLM veto; never put the VLM in the safety loop.
3. **Qwen-RobotNav as primary mode-(a) candidate and as the interface template (Qwen-RobotNav).** Why: Qwen3-VL based, monocular numbers still strong (R2R SR 66.9, RxR 73.4), 4B fits 24 GB with room for perception (inferred: ~8 GB bf16), outputs 8 waypoints that map to a local tracker. Hint: benchmark 4B and 8B on the 3090 Ti for latency (Thor reached ~204 ms); mount camera + prompt template for the X30 ("Front View" tag, "Imagine you are a robot programmed for navigation tasks"); feed waypoints to a waypoint follower (Nav2 follow-path or the base-autonomy style tracker) rather than raw velocity. Check weights/licence in the repo first (not in the text). Keep NaVILA-style language actions as the fallback when the weights are unusable (plain Qwen3-VL zero-shot is unproven: see pitfalls).
4. **Parameterised mode switch: `navigate(mode, instruction, config)` (Qwen-RobotNav).** Why: one executor, different context strategies; matches the two-mode requirement. Hint: define a tool schema with `mode in {instruction_following, object_goal}` (later pointnav/tracking), `config = {token_budget, recency_decay, frame_sampling: random|latest}`; defaults: instruction following -> larger B, gamma ~2-3 (their ablation: SR peaks near gamma 3.0, B 4608), approach/verify -> small B, latest frames. Mode (b) can use the same tool for the *approach* phase and Nav2 for the global leg.
5. **Hierarchical semantic structure: rooms -> viewpoints/objects, VLM picks the room (SysNav) with DBSCAN+LLM room labelling as the cheap variant (OrionNav).** Why: SysNav's real SR 98-100% vs 25-58% for VLFM; OrionNav 100% vs 56% without room labels; the VLM is used where it is good (semantics) not where it is weak (3D geometry). Hint: 4 Livox give a dense 3D cloud -> use SysNav's wall-plane + z-distribution room segmentation, fall back to DBSCAN on object positions + LLM label; keep per-room object lists and a best-view crop per object/room.
6. **Scene graph serialised as compact JSON + a small primitive set + verification loop (OrionNav).** Why: zero LLM/graph failures in 96 real trials. Hint: primitives `goto(object_or_room_id)`, `search_room(room_id)`, `explore(frontier|global)`, `done(object_id)`; each call returns structured feedback (success, new objects seen, recent action log); ask the VLM "task done?" with the candidate's best-view crop before terminating; start with a 360-degree spin when the graph is empty (use yaw rotation or the 4 Livox + camera sweep). Prefer constrained/JSON-schema decoding with the local VLM `(from my knowledge: vLLM guided decoding / tool-call parsers)`.
7. **Event-triggered VLM queries: early-stop and room-query templates (SysNav).** Why: keeps VLM load low on one GPU and avoids backtracking. Hint: implement exactly two query types - (1) "new room discovered while exploring room A: switch?" (2) "room finished without target: which unexplored room next?" with candidate rooms + object lists + history (trajectory) + goal; return `{choice, reason}`; cache per room-state hash to avoid repeated calls.
8. **Instance-level semantic object map with LiDAR-camera lifting and pose-graph re-anchoring (CausalNav + OrionNav).** Why: depth cameras are unreliable outdoors (CausalNav); OrionNav re-anchors objects after loop closure. Hint: for each mask, collect LiDAR points projected into it (needs LiDAR-camera extrinsics; do per-lidar extrinsics for 4 Livox), take robust 3D centroid + bbox, associate by DBSCAN/nearest-neighbour + class, store with the keyframe id so a SLAM map update moves them (inferred); publish both a 3D marker set and a 2D semantic occupancy layer for Nav2.
9. **Open-vocabulary perception stack sized for a shared 24 GB GPU (SysNav, CausalNav, OrionNav).** Why: YOLO-World + SAM 2 (SysNav) or YOLO-World + ByteTrack (CausalNav) are light; FC-CLIP needed FP16 and custom kernels to run on Orin. Hint: YOLO-World (open vocab, runtime-editable class list) for detection, SAM2 hiera-b+ only on selected boxes, ByteTrack for IDs; budget VRAM (VLM + detector + SAM) and run perception at 5-10 Hz; thermal channel is unprecedented here (see pitfalls) - use it as an additional detector for people/heat sources (inferred).
10. **Annotated Semantic Map rendering as a VLM-readable memory image (MapNav).** Why: 0.17 MB constant memory, +9 SR points over no map, VLM attention snaps to text labels; GPT-4o read ASM zero-shot. Hint: render the semantic grid from the map node as a top-down image with agent arrow, trajectory, explored/obstacle shading, and class names drawn at region centroids via connected components with min-area threshold; supply as 2nd image in mode (a) and as context for the planner in mode (b); do not feed depth images to the VLM (MapNav ablation: RGB+Depth was worse). Zero-shot usefulness with Qwen3-VL is unverified - A/B test.
11. **Hierarchical LLM retrieval with hybrid spatial score for "find X" queries (CausalNav).** Why: made 12-14B local models almost as good as GPT-4o (SR 82-85 vs 88). Hint: query graph level by level: floor/area -> room -> object; score = alpha*proximity + (1-alpha)*semantic path score; embed labels to merge synonyms ("trash can"/"garbage bin"); returns ranked candidate instances to Nav2 `NavigateToPose`; if none, trigger exploration.
12. **Dynamic-object filtering before map insertion (CausalNav).** Why: moving people/vehicles pollute the graph and costmaps, graph update gave SR +12. Hint: ByteTrack IDs + displacement threshold over k steps (corridor filter); keep dynamic objects only in the local costmap layer; use thermal to flag people (inferred); apply to Livox cloud with a dynamic-removal local map (RH-Map idea).
13. **Safety stack under all VLM output: MPPI + CBF filter (OrionNav) / NMPC-CBF (CausalNav) + X30's own correction.** Why: neither Habitat-style nor VLM outputs guarantee collision freedom; NaVILA's blind vs vision policy gap was 14 SR points. Hint: Nav2 MPPI controller for mode (b); a CBF-QP velocity filter on the fused Livox cloud between any planner/VLM command and `/cmd_vel`; also keep the X30 elevation-map correction path in the loop `(api.txt)`; use LiDAR (not RGB depth) for glass/sunlight cases (NaVILA).
14. **Memory budgeting: frame subsampling and evidence records (NaVILA, Qwen-RobotNav).** Why: local Qwen3-VL context is limited. Hint: NaVILA recipe = latest frame + uniformly sampled history (first frame always), 8 frames, textual markers "historical observations:" / "current observation:"; for long runs, convert each finished segment into a small JSON evidence record `{subgoal, mode, progress, salient[], outcome, key_frames[]}` plus a notebook (searched rooms, rejected hypotheses) so the prompt stays short; store key frames on disk for later re-inspection.
15. **Quantised serving + latency targets (NaVILA, Qwen-RobotNav, CausalNav).** Why: AWQ W4A16 halves memory (18.5 -> 8.6 GB) and cuts latency 40% with ~1.5 SR point loss; edge FP8 reached ~5 Hz. Hint: serve the VLM through an OpenAI-compatible local server (vLLM or similar `(from my knowledge)`), 4B bf16 or 8B 4-bit, one process, request queue with priorities (safety verification > room query > instruction step); log per-call latency; target <=1-2 Hz for mode (a) and event-driven for mode (b). Also decouple "simulated locomotion" from navigation for Gazebo-first tests (VLN-PE flash-controller idea: a kinematic velocity-driven base, inferred) before integrating real gait.

---

## 5. Pitfalls and open questions

1. **No paper validates an off-the-shelf Qwen3-VL as a language-action navigator.** NaVILA, MapNav and Qwen-RobotNav all fine-tune (15.6M samples for Qwen-RobotNav; NaVILA's 20K YouTube trajectories added ~4-5 SR points; GPT-4o without tuning lost to NaVILA in real tests). Expect zero-shot Qwen3-VL to be much weaker; plan on Qwen-RobotNav weights or a small fine-tune/DAgger loop, and decide early whether weights/licences allow it (not stated in the text).
2. **GPU budget on one RTX 3090 Ti (24 GB).** NaVILA-8B FP16 is 18.5 GB for its context; adding detector, SAM2 and Qwen-style visual tokens will not fit unless quantised or the 4B model is used; the papers' strong systems run on RTX 4090 (24 GB, ~30-60% faster than a 3090 Ti (from my knowledge)) or Jetson Thor. Real latencies on this hardware are unknown.
3. **Cloud LLM/VLM dependence in 3 of 6 systems** (OrionNav GPT-4-Turbo, SysNav Gemini-2.5-flash, Qwen-RobotNav's agentic Qwen3.6-Plus). Only CausalNav shows local 12-14B models viable (and only for graph retrieval). A local 4-8B VLM for room-level reasoning is plausible but unvalidated; keep prompts short and structured.
4. **Action-space and step-size mismatch.** Habitat-era benchmarks use 0.25 m / 15-30 degree teleport-like steps and sometimes pass through gaps a quadruped cannot; X30 gait limits, turn-in-place behaviour and command timeout rules must define the Gazebo action model. NaVILA's quoted speeds (0.5 m/s, +/- pi/6 rad/s) are for Go2 - X30 limits are in `api.txt` Appendix B (not digested here).
5. **Pose/odometry dependence and drift.** MapNav's ASM needs pose each step (source unspecified); VLFM degraded with pose noise (OrionNav); OrionNav's failures included bad SLAM maps. With 4 Livox you must solve multi-lidar extrinsics, time sync and SLAM choice (none of the papers uses multi-LiDAR; SysNav uses one Mid-360).
6. **Thermal camera has no precedent in these papers.** There is no recipe for thermal segmentation/fusion or for feeding thermal to a VLM; treat it as a separate detector track (people/hot objects/low light) and an open research item.
7. **Semantic-map staleness and false positives.** Dynamic objects (CausalNav), perception misses (OrionNav 4/96 failures), and wrong-instance merges. Need object lifetime/decay, re-observation checks and VLM verification on a best-view crop before declaring success.
8. **Open-loop language actions without error recovery.** NaVILA's own failure case; Qwen-RobotNav and MapNav also lack explicit recovery. Add stuck/loop detection, backtrack-to-viewpoint (SysNav viewpoint nodes make this natural, inferred) and a human override.
9. **Dataset licences and sim fidelity.** MP3D/HM3D require signed agreements and cannot be redistributed; Habitat meshes are noisy and heavy for Gazebo; Livox patterns and legged dynamics are not reproduced by stock Gazebo sensors; synthetic-to-real appearance gap (Qwen team used style transfer to photorealistic images, MapNav relied on real tests). Budget time for world authoring.
10. **Evaluation protocol gaps.** Papers use SR/SPL/NE on fixed benchmarks, 10 m (CausalNav) or sensor-visible distance success (SysNav); real tests are small (25-190 episodes) and Qwen-RobotNav gives no real success rates. Define our own metrics early: SR, SPL-like path ratio, SPT/time, collisions/interventions, VLM latency, parse-failure rate, per mode.
