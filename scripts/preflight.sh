#!/usr/bin/env bash
# Pre-flight check before a robot test session: devices, topics, TF chain,
# nav lifecycle, duplicate nodes/publishers, system health. Read-only — it
# changes nothing. Run after bringup/localize are up:
#
#   ~/robot_ws/src/home_robot/scripts/preflight.sh          # full check
#   ~/robot_ws/src/home_robot/scripts/preflight.sh --quick  # skip topic-rate checks
#
# Exit code: 0 all good, 1 at least one FAIL (warnings don't fail).
set -uo pipefail

QUICK=false
[ "${1:-}" = "--quick" ] && QUICK=true

PASS=0; FAIL=0; WARN=0
ok()   { echo "  ✔ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✖ $1"; FAIL=$((FAIL+1)); }
meh()  { echo "  ⚠ $1"; WARN=$((WARN+1)); }

# The ros2 CLI daemon caches discovery info and keeps reporting nodes that
# died (ghost "duplicates", missing publishers). Restart it so every check
# below sees the live graph. CLI-side only — touches no robot node.
echo "── Φρεσκάρισμα ROS discovery cache"
ros2 daemon stop  > /dev/null 2>&1
ros2 daemon start > /dev/null 2>&1
sleep 4

echo "── Συσκευές (udev symlinks)"
for d in roomba lidar imu; do
    [ -e /dev/$d ] && ok "/dev/$d" || bad "/dev/$d λείπει (καλώδιο/udev;)"
done
[ -e /dev/arm ] && ok "/dev/arm" || meh "/dev/arm λείπει (οκ αν ο βραχίονας είναι εκτός)"

echo "── Σύστημα"
read -r load _ < /proc/loadavg
cores=$(nproc)
awk -v l="$load" -v c="$cores" 'BEGIN{exit !(l < c)}' \
    && ok "load $load (πυρήνες: $cores)" || meh "υψηλό load $load — δες top πριν κατηγορήσεις το nav"
mem_avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
awk -v m="$mem_avail" 'BEGIN{exit !(m > 1.5)}' \
    && ok "διαθέσιμη RAM ${mem_avail}G" || bad "μόνο ${mem_avail}G RAM διαθέσιμη"
disk_avail=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$disk_avail" -gt 5 ] && ok "δίσκος ${disk_avail}G ελεύθερα" || meh "μόνο ${disk_avail}G δίσκος"

echo "── ROS graph"
NODES=$(timeout 10 ros2 node list 2>/dev/null)
[ -n "$NODES" ] && ok "ros2 node list απαντά ($(echo "$NODES" | wc -l) nodes)" \
                || bad "κανένα ROS node — τρέχει το bringup;"
# laser_filters' scan_to_scan_filter_chain registers TWO DDS nodes named
# /lidar_arm_filter from its single process — a known package quirk, not an
# orphan. Filter it out and verify it by process count instead.
DUPES=$(echo "$NODES" | sort | uniq -d | grep -v "^/lidar_arm_filter$")
[ -z "$DUPES" ] && ok "κανένα διπλό node" \
                || bad "ΔΙΠΛΑ nodes (ορφανά; → RViz flicker/TF jumps): $(echo $DUPES | tr '\n' ' ')"
# Known orphan offenders: duplicate processes → duplicate TF publishers
# "=" catches remapped node names too, e.g. ekf_node's argv only contains
# "ekf_filter_node" as "...--ros-args -r __node:=ekf_filter_node..." — the
# binary itself is named ekf_node, so "[/ ]" alone never matched it.
for pat in imu_node ekf_filter_node scan_to_scan_filter_chain; do
    n=$(pgrep -cf "[/ =]$pat" || true)
    [ "${n:-0}" -le 1 ] && ok "$pat: $n process" || bad "$pat: $n processes — σκότωσε το ορφανό"
done

echo "── Topics"
for t in /scan /odom /map; do
    # retry — right after the daemon restart above, DDS discovery can take
    # longer than any single timeout to repopulate a topic's publisher list.
    # Capture into a variable first: piping straight into `grep -q` lets grep
    # exit (and close the pipe) the instant it sees a match, SIGPIPE-ing
    # `timeout`/ros2 mid-write and corrupting the pipeline's exit status.
    found=false
    for attempt in 1 2 3; do
        out=$(timeout 6 ros2 topic info $t 2>/dev/null)
        echo "$out" | grep -q "Publisher count: [1-9]" && { found=true; break; }
        sleep 2
    done
    $found && ok "$t έχει publisher" || bad "$t χωρίς publisher"
done
if ! $QUICK; then
    hz=$(timeout 8 ros2 topic hz /scan --window 10 2>/dev/null | grep -oE 'average rate: [0-9.]+' | head -1 | grep -oE '[0-9.]+')
    if [ -n "${hz:-}" ]; then
        awk -v h="$hz" 'BEGIN{exit !(h > 5)}' && ok "/scan @ ${hz} Hz" || bad "/scan μόνο ${hz} Hz"
    else
        bad "/scan δεν στέλνει δεδομένα"
    fi
fi

echo "── TF αλυσίδα"
for pair in "map odom" "odom base_link" "base_link laser"; do
    set -- $pair
    # a fresh tf2_echo process needs a few seconds to build its buffer from
    # /tf_static + /tf before the first lookup succeeds — retry, don't 1-shot.
    # Captured into a variable for the same reason as the topic-info check
    # above (direct pipe into `grep -q` races with the writer's SIGPIPE).
    found=false
    for attempt in 1 2 3; do
        out=$(timeout 6 ros2 run tf2_ros tf2_echo "$1" "$2" 2>/dev/null)
        echo "$out" | grep -q "Translation" && { found=true; break; }
        sleep 2
    done
    if $found; then
        ok "TF $1 → $2"
    else
        bad "TF $1 → $2 λείπει $([ "$1" = map ] && echo '(τρέχει localize/AMCL;)')"
    fi
done

# ‼️ A transform with TWO publishers is worse than one that is missing: the
# lookup keeps succeeding, and just returns whichever writer was last. Found
# 2026-08-02 — an orphaned odom_tf_broadcaster from a Gazebo session was
# fighting ekf_filter_node, so map→base_link swung 0.49 m and 14.6° with the
# robot standing perfectly still. Every symptom pointed at the lidar ("the red
# lines point somewhere else"), while /scan, /odom, the IMU and the laser TF
# were all provably fine. Nothing else in this script would have caught it: the
# node names differ, so the duplicate-node check above stays quiet.
# Counting message rate does NOT work here (a single writer at 16 Hz looks like
# two at 8), so name the culprits instead: odom->base_link has exactly one legal
# owner, and which one depends on the mode.
#   ekf_filter_node      — the real robot (bringup)
#   odom_tf_broadcaster  — the Gazebo sim (sim.launch.py), which has no EKF
# Both alive at once means a sim leftover survived, and the two disagree.
ekf_n=$(pgrep -cf "[/ =]ekf_filter_node" || true)
otb_n=$(pgrep -cf "[/ =]odom_tf_broadcaster" || true)
if [ "${ekf_n:-0}" -ge 1 ] && [ "${otb_n:-0}" -ge 1 ]; then
    bad "odom→base_link έχει ΔΥΟ publishers (ekf_filter_node + odom_tf_broadcaster)"
    bad "  ↳ ορφανό από συνεδρία Gazebo· η πόζα θα χοροπηδά και το scan θα «δείχνει αλλού»"
elif [ "${ekf_n:-0}" -ge 1 ] || [ "${otb_n:-0}" -ge 1 ]; then
    ok "odom→base_link: ένας publisher"
else
    bad "odom→base_link: κανένας publisher (ούτε EKF ούτε odom_tf_broadcaster)"
fi

echo "── Nav2 lifecycle"
NAV_OK=true
for n in controller_server planner_server behavior_server velocity_smoother collision_monitor bt_navigator; do
    state=$(timeout 5 ros2 lifecycle get /$n 2>/dev/null | cut -d' ' -f1)
    if [ "$state" = "active" ]; then
        ok "/$n active"
    else
        bad "/$n: ${state:-δεν αποκρίνεται} → τρέξε scripts/ensure_nav_active.sh"
        NAV_OK=false
    fi
done

echo
echo "Σύνοψη: $PASS ✔  $WARN ⚠  $FAIL ✖"
if [ $FAIL -eq 0 ]; then
    echo "Όλα έτοιμα για test. 🚀"
    exit 0
else
    $NAV_OK || echo "Tip: bash ~/robot_ws/src/home_robot/scripts/ensure_nav_active.sh"
    exit 1
fi
