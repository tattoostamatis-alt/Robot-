# home_robot

A home tidying/assistant robot: a Roomba 692 base with a 360° LiDAR, a depth
camera, an IMU, a robot arm and a mic array, running a full ROS 2 Jazzy
autonomy + voice + LLM stack on an AMD Ryzen AI mini-PC.

> Package metadata: `home_robot` (ROS 2 Jazzy, `ament_cmake` + Python nodes).
> Live workspace is `~/robot_ws`; the stale `~/ros2_robot_ws` only provides
> vendored drivers (sllidar, create_bringup).

## Hardware

| Part | Used for |
|---|---|
| iRobot Roomba 692 | drive base (`roomba_driver.py`, raw-termios serial, keep-alive) |
| SLAMTEC RPLiDAR C1 | 360° scan → `/scan` (arm box-filtered out of the plane) |
| Intel RealSense D435 | depth for localization + RGB-D for object detection |
| BNO085 IMU | orientation + yaw-rate for the EKF (mounted upside-down, gyro watchdog) |
| Waveshare RoArm-M3 | 5-DoF arm for pick-and-place (MoveIt) |
| reSpeaker XVF3800 | 6-ch mic array (ch4 = AEC'd ASR beam) for the voice stack |
| AMD Ryzen AI (NPU + Radeon 860M iGPU) | Qwen3-VL/embeddings on NPU, YOLO on iGPU |

## Software stack

- **Localization** — `robot_localization` EKF (IMU + wheel odom) + AMCL on a
  saved map, with AprilTag one-shot relocalization and a depth+LiDAR global
  localizer for a random start pose.
- **Navigation** — Nav2 with the **MPPI** controller, direction-aware
  `collision_monitor` (`cmd_vel` → `cmd_vel_safe`), keepout zones.
- **Mapping** — `slam_toolbox` (2D) and RTAB-Map (3D).
- **Perception** — YOLO11n object detector on the iGPU (~63 fps), object
  tracker, face/pose detectors.
- **Voice / LLM** — openWakeWord wake word ("Ρομπότ Μαξ"), faster-whisper STT
  (Greek), edge-tts TTS, an LLM bridge to a local Qwen3-VL (Lemonade/NPU) with
  tool-calling, and a ChromaDB RAG long-term memory.
- **Arm** — MoveIt 3D control + a detection-driven pick-and-place bridge.
- **Teleop** — PS5 DualSense joystick, keyboard teleop.
- **Orchestration** — task planner → mission executor → recovery manager.

## Launch modes

```bash
cd ~/robot_ws && colcon build --packages-select home_robot --symlink-install && source install/setup.bash
```

| Launch | What it does |
|---|---|
| `bringup.launch.py` | the everything-stack; each subsystem gated by a `use_*` flag |
| `localize.launch.py map:=kela3 use_obstacle_safety:=true` | AMCL + Nav2 driving on a saved map (add `use_perception:=true`, below) |
| `slam_only.launch.py` | build a new map with slam_toolbox |
| `view_map.launch.py` | inspect a saved map in RViz |
| `sim.launch.py` | headless Gazebo Nav2 sim (no hardware) |
| `arm_moveit.launch.py` | MoveIt 3D arm control |

`use_obstacle_safety:=true` is required for the robot to move in localize mode
(nothing relays `cmd_vel → cmd_vel_safe` otherwise).

## Recent features (2026-07-06)

Four features added together — all **code-complete and unit-tested, not yet
hardware-verified**. HW-test + tuning checklist in
[`scripts/RUNBOOK_spin_retest.md`](scripts/RUNBOOK_spin_retest.md) (PART 3) and
[`TODO.md`](TODO.md).

- **Semantic object memory** (`object_memory_node.py`, `object_memory.py`) —
  remembers *where objects are* in the map frame. Each `detected_objects`
  sighting is transformed to `map` and merged into the nearest known instance
  of the same label; the result persists to disk and is exposed as
  `/object_memory` + RViz markers, an `object_memory/query`→`/answer` service,
  and concise facts pushed to the RAG store so voice recall ("πού είναι το X;")
  works with no new LLM tool. Flag: `use_object_memory`.
- **Dynamic obstacle costmap layers** — `obstacle_prediction_node` and
  `semantic_costmap_node` publish `/predicted_obstacles` + `/semantic_obstacles`
  (people/pets get extra personal-space inflation, moving objects get their
  path projected ahead) into Nav2's local costmap. Now reachable during
  navigation via `localize.launch.py use_perception:=true`.
- **Pick-place visual servoing** (`pick_place_node.py`) — closed-loop XY
  refinement of the grasp: while hovering, the arm re-locks the same object
  from live detections and nudges until the estimate settles, instead of
  committing to one noisy snapshot. Eye-to-hand (body-mounted D435). *Calibrate
  `tf_base_arm` first* — servoing can't fix a calibration bias.
- **Voice barge-in** — say the wake word while the robot is talking and it stops
  (`tts/stop` → `sd.stop()`) and listens, instead of muting the mic. Relies on
  the XVF3800's hardware AEC; flag `allow_barge_in` (default on).

Enable the first three together:

```bash
ros2 launch home_robot localize.launch.py map:=kela3 \
  use_obstacle_safety:=true use_perception:=true use_arm:=true   # use_arm only for pick
```

## Label-free object clustering (`perception_clusters`)

Geometric object detection adapted from the **Interbotix LoCoBot** stack — the
label-free complement to `pick_place_node.py` (which needs a YOLO label). A PCL
pipeline on the D435 cloud (CropBox → Voxel → RANSAC plane removal → Euclidean
clustering) reports the centroid of *any* object sitting on a support surface.

The heavy lifting is in two **vendored** packages (BSD-3-Clause, Trossen) that
live in `src/` *outside* this repo — like `roarm_*` and `m-explore-ros2`. On a
fresh machine, fetch them first, then build:

```bash
cd ~/robot_ws
./src/home_robot/scripts/fetch_vendored.sh   # sparse-checks-out the 2 pkgs into src/
colcon build --packages-select interbotix_perception_msgs interbotix_perception_pipelines home_robot
source install/setup.bash
```

`vendor.repos` pins the exact upstream commit (see the script/file headers).

Run it (needs the D435 pointcloud up — `bringup` with `pointcloud.enable:=true`
— and the `base_link → arm_base` TF):

```bash
ros2 launch home_robot perception_clusters.launch.py                    # on-demand: PCL runs per service call
ros2 launch home_robot perception_clusters.launch.py enable_pipeline:=true  # continuous, for RViz tuning
```

- `~/clusters` (JSON) and `~/cluster_markers` (RViz MarkerArray) report the
  objects in `arm_base`; the launch **picks/moves nothing**, it's verification.
- Tune the crop box in [`config/pcl_filter_params.yaml`](config/pcl_filter_params.yaml)
  to the RoArm-M3 reach (box is in the camera depth-optical frame: x=right,
  y=down, z=forward). `enable_pipeline:=true` + RViz makes tuning live.
- **Status: HW-untested** on real D435 data — smoke-tested wiring only.

## Tests

Robot-free unit + wiring tests (no ROS graph needed):

```bash
cd ~/robot_ws/src/home_robot && python3 -m pytest tests/ -q
```

## Repo layout

- `home_robot/nodes/` — the ROS 2 nodes (one file per node).
- `home_robot/*.py` — dependency-free cores (`object_memory.py`,
  `voice_gate.py`) that back a node but stay unit-testable.
- `launch/`, `config/`, `maps/`, `worlds/`, `description/` — launch files,
  params/RViz/nav2 config, saved maps, sim worlds, robot model.
- `firmware/` — ESP32 IMU firmware (BNO085).
- `scripts/` — preflight, nav-test recording, runbooks.
- `deploy/` — system files outside the workspace (systemd, udev, `~/bin`); see
  [`deploy/README.md`](deploy/README.md).
- `TODO.md` — open work, grouped by whether the robot must be powered on.
