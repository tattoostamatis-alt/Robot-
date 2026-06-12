#!/usr/bin/env bash
# Save the current slam_toolbox pose-graph for "lifelong mapping" — lets
# the next bringup resume localizing in and extending this map instead of
# starting from empty.
#
# Usage: ./save_map.sh [name]   (default name: home)
#
# Run this while slam_toolbox is up (e.g. during/after a mapping session
# with bringup.launch.py use_slam:=true). Afterwards, set in
# config/nav2_params.yaml under slam_toolbox:
#   map_file_name: <repo>/maps/<name>
# (uncomment map_start_pose too if not resuming at the dock) to resume
# from it on the next bringup.

set -euo pipefail
NAME="${1:-home}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/maps"
mkdir -p "$DIR"

ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "{filename: '$DIR/$NAME'}"

echo
echo "Saved pose-graph to $DIR/$NAME.posegraph + $DIR/$NAME.data"
echo "To resume next session, set in config/nav2_params.yaml (slam_toolbox):"
echo "  map_file_name: $DIR/$NAME"
