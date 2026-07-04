# deploy/ — system files outside the ROS workspace

Everything the robot needs that does **not** live in the colcon workspace:
helper scripts in `~/bin`, systemd units, udev rules, DDS config. These are
the masters — when you change the live copy on the machine, copy it back
here and commit. `./install.sh` pushes them onto a fresh machine.

| file | installs to | why |
|---|---|---|
| `bin/start_sllidar_c1` | `~/bin` | lidar + arm box-filter (`/scan_raw` → `/scan`), run by `ros-sllidar-c1.service` |
| `bin/start_roomba_692` | `~/bin` | create_bringup driver with `config/roomba_692.yaml` |
| `bin/start_realsense_d435` | `~/bin` | D435 with aligned depth + pointcloud |
| `bin/start_foxglove_bridge` | `~/bin` | Foxglove ws on :8765, run by `ros-foxglove-bridge.service` |
| `systemd/ros-sllidar-c1.service` | `/etc/systemd/system` | lidar at boot |
| `systemd/ros-foxglove-bridge.service` | `/etc/systemd/system` | foxglove at boot |
| `config/cyclonedds.xml` | `~/.config` | MaxAutoParticipantIndex=99 — default 9 is too low for the ~20-node stack |
| `config/99-ros-robot-serial.rules` | `/etc/udev/rules.d` | `/dev/sllidar`, `/dev/roomba` symlinks |
| `../config/99-robot-devices.rules` | `/etc/udev/rules.d` | `/dev/lidar`, `/dev/imu`, `/dev/arm` etc. — lidar & arm are both CP2102N, matched by serial |
| `config/roomba_692.yaml` | `~/ros2_robot_ws/config` | create_driver params (`/dev/roomba`, base_footprint, publish_tf) |

## Gotchas

- The start scripts source **both** workspaces: `~/robot_ws` (home_robot) and
  `~/ros2_robot_ws`. The old workspace is stale for home_robot code but still
  provides the vendored drivers: `sllidar_ros2`, `create_bringup`
  (+ `roomba_692.yaml`). Don't delete it without moving those.
- `lemond.service` (Lemonade/NPU LLM for the voice stack) is packaged by
  Lemonade itself under `/usr/lib/systemd/system` — not vendored here.
- udev matches lidar/arm by **serial number** (both are 10c4:ea60); a
  replacement unit means updating the serial in `99-robot-devices.rules`.
