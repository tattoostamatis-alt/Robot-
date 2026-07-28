#!/usr/bin/env bash
# Install the out-of-repo pieces of the robot bringup onto a fresh machine
# (or restore them after a disk failure). Idempotent — safe to re-run.
#
#   cd ~/robot_ws/src/home_robot/deploy && ./install.sh
#
# Not covered here (see README.md): the ROS 2 workspaces themselves
# (~/robot_ws and ~/ros2_robot_ws) and apt/pip dependencies.
set -eo pipefail
cd "$(dirname "$0")"

echo "── helper scripts → ~/bin"
mkdir -p ~/bin
cp -v bin/* ~/bin/
chmod +x ~/bin/start_*

echo "── CycloneDDS config → ~/.config"
mkdir -p ~/.config
cp -v config/cyclonedds.xml ~/.config/

echo "── udev rules → /etc/udev/rules.d (sudo)"
sudo cp -v ../config/99-robot-devices.rules /etc/udev/rules.d/
sudo cp -v config/99-ros-robot-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger

echo "── roomba config → ~/ros2_robot_ws/config"
mkdir -p ~/ros2_robot_ws/config
cp -v config/roomba_692.yaml ~/ros2_robot_ws/config/

echo "── systemd units → /etc/systemd/system (sudo)"
sudo cp -v systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ros-sllidar-c1.service ros-foxglove-bridge.service
# The keep-alive is the one robot service that must be on at boot: it wakes the
# robot with the PC and holds it awake. The -resume companion re-runs its wake
# burst after suspend.
sudo systemctl enable roomba-keepalive.service roomba-keepalive-resume.service

echo
echo "Done. Start now with:"
echo "  sudo systemctl start ros-sllidar-c1 ros-foxglove-bridge"
