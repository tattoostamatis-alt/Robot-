#!/usr/bin/env bash
# Fetch the two vendored perception packages that home_robot's
# perception_clusters.launch.py depends on, straight into robot_ws/src/.
#
# vcstool can't grab subdirectories, so this sparse-checks-out only the two
# packages we need from interbotix_ros_toolboxes (BSD-3-Clause) -- exactly
# reproducing the in-tree copies. Idempotent: skips packages already present.
# See vendor.repos for the pinned commit and project_robot_pcl_clusters memory.
set -euo pipefail

REPO="https://github.com/Interbotix/interbotix_ros_toolboxes.git"
COMMIT="f52234371b9ec1cb1f5ee8241da32e6b8476fa5c"  # humble @ 2026-01-14, keep in sync with vendor.repos
# packages we want: <sparse path in upstream> -> <dest dir name under src/>
declare -A PKGS=(
  ["interbotix_perception_toolbox/interbotix_perception_msgs"]="interbotix_perception_msgs"
  ["interbotix_perception_toolbox/interbotix_perception_pipelines"]="interbotix_perception_pipelines"
)

# src/ is two levels up from this script (src/home_robot/scripts/ -> src/)
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Target workspace src: ${SRC_DIR}"

need_fetch=false
for dest in "${PKGS[@]}"; do
  if [[ -d "${SRC_DIR}/${dest}" ]]; then
    echo "  [skip] ${dest} already present"
  else
    need_fetch=true
  fi
done
if ! $need_fetch; then
  echo "All vendored packages already present. Nothing to do."
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
echo "Sparse-cloning ${REPO} @ ${COMMIT:0:7} ..."
git clone --filter=blob:none --sparse "${REPO}" "${TMP}/repo"
git -C "${TMP}/repo" sparse-checkout set "${!PKGS[@]}"
git -C "${TMP}/repo" checkout "${COMMIT}"

for path in "${!PKGS[@]}"; do
  dest="${PKGS[$path]}"
  if [[ -d "${SRC_DIR}/${dest}" ]]; then continue; fi
  echo "  [copy] ${dest}"
  cp -r "${TMP}/repo/${path}" "${SRC_DIR}/${dest}"
  cp "${TMP}/repo/LICENSE" "${SRC_DIR}/${dest}/LICENSE"   # carry BSD attribution
done

echo "Done. Now build:"
echo "  cd ${SRC_DIR}/.. && colcon build --packages-select interbotix_perception_msgs interbotix_perception_pipelines"
