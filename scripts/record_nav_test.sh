#!/usr/bin/env bash
# Record everything needed to diagnose a nav test offline (e.g. the RViz-goal
# spin/oscillation bug): TF, scan, odom/IMU/AMCL poses, the whole cmd_vel
# chain, plan and costmap. Start it BEFORE sending the goal, Ctrl-C after.
#
#   ~/robot_ws/src/home_robot/scripts/record_nav_test.sh [label]
#
# Bags land in ~/nav_test_bags/<date>_<label>. Size is modest (the big
# camera/image topics are deliberately NOT recorded).
#
# Analyze later, e.g.:
#   ros2 bag info ~/nav_test_bags/<name>
#   ros2 bag play <name> --topics /tf /scan   # replay into RViz
set -eo pipefail

LABEL=${1:-test}
OUT=~/nav_test_bags/$(date +%Y%m%d_%H%M%S)_${LABEL}
mkdir -p ~/nav_test_bags

echo "Καταγράφω στο $OUT — Ctrl-C για τέλος."
exec ros2 bag record -o "$OUT" \
    /tf /tf_static \
    /scan \
    /odom \
    /imu/data \
    /amcl_pose \
    /particle_cloud \
    /goal_pose \
    /plan \
    /cmd_vel /cmd_vel_nav /cmd_vel_smoothed /cmd_vel_safe \
    /local_costmap/costmap \
    /collision_monitor_state \
    /navigate_to_pose/_action/status
