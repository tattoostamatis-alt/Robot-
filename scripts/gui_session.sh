#!/usr/bin/env bash
# gui_session.sh — run the real Qt GUIs (RViz, MoveIt, Gazebo) on headless VNC
# displays so the web dashboard can stream them into the browser.
#
# The dashboard does NOT shell out to websockify: it bridges the RFB socket to a
# browser WebSocket itself (see web_dashboard_node.py), so everything stays on
# port 8080 behind the one dashboard token. This script's only job is owning the
# X displays and the app processes.
#
#   gui_session.sh start  rviz|moveit|gazebo
#   gui_session.sh stop   rviz|moveit|gazebo
#   gui_session.sh status [app]        # prints "running <vnc_port>" or "stopped"
#
# Display map — RViz keeps :2 on purpose. `robot max` already starts :2 for the
# phone (RealVNC → :5902), and running a SECOND RViz against the same graph has
# a documented failure mode: two RViz instances made AMCL deactivate and every
# Nav2 goal fail with planner error 208. So the dashboard attaches to the same
# session rather than spawning a rival one.
#
#   rviz   :2  → 5902   (shared with the phone's RealVNC session)
#   gazebo :3  → 5903
#   moveit :4  → 5904

set -u

WS=/home/dimi/robot_ws
VNCDIR=/home/dimi/.vnc

case "${2:-}" in
  rviz)   DISP=2; GEOM=1600x900 ;;
  gazebo) DISP=3; GEOM=1600x900 ;;
  moveit) DISP=4; GEOM=1600x900 ;;
  '')     [ "${1:-}" = status ] || { echo "usage: $0 {start|stop|status} {rviz|gazebo|moveit}" >&2; exit 2; } ;;
  *)      echo "unknown app '${2}' (rviz|gazebo|moveit)" >&2; exit 2 ;;
esac
APP="${2:-}"
PORT=$((5900 + ${DISP:-0}))

# Per-app X session script. Each one sources ROS, starts a bare window manager
# (openbox — without one, Qt dialogs come up undecorated and unmovable) and
# execs the app as the session leader, so killing the VNC display kills the app.
#
# LIBGL_ALWAYS_SOFTWARE: Xvnc has no GPU. Gazebo's Ogre2 renderer additionally
# needs its own software backend selected, which is why it sets more than RViz.
write_xstartup() {
  local f="$VNCDIR/xstartup-$1"
  case "$1" in
    rviz)
      cat > "$f" <<'EOF'
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
exec rviz2 -d /home/dimi/robot_ws/install/home_robot/share/home_robot/config/robot.rviz
EOF
      ;;
    moveit)
      # arm_moveit.launch.py brings up move_group + its own RViz with the MoveIt
      # motion-planning panel (drag the gripper, plan, execute). It must run
      # ALONGSIDE bringup's arm_driver, never with a second one — the bridge in
      # that launch file forwards MoveIt's trajectories to the existing driver.
      cat > "$f" <<'EOF'
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
exec ros2 launch home_robot arm_moveit.launch.py
EOF
      ;;
    gazebo)
      # The headless-sim launch file plus the Gazebo GUI client. Note this is a
      # SIMULATION: it publishes its own /clock, /scan and /odom. Do not run it
      # while driving the real robot — see the orphan-/clock note in the docs.
      cat > "$f" <<'EOF'
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
exec ros2 launch home_robot sim.launch.py headless:=false use_rviz:=false
EOF
      ;;
  esac
  chmod +x "$f"
}

is_up() {   # $1 = display number
  ss -ltn 2>/dev/null | grep -q ":$((5900 + $1))\b"
}

case "${1:-}" in

  start)
    if is_up "$DISP"; then
      echo "running $PORT"
      exit 0
    fi
    write_xstartup "$APP"
    # -localhost no so RealVNC on the phone can still reach these directly;
    # the dashboard's own bridge connects over 127.0.0.1 either way.
    tigervncserver ":$DISP" -localhost no -geometry "$GEOM" \
      -xstartup "$VNCDIR/xstartup-$APP" >/dev/null 2>&1
    for _ in $(seq 1 20); do
      is_up "$DISP" && { echo "running $PORT"; exit 0; }
      sleep 1
    done
    echo "failed (see $VNCDIR/*:$DISP.log)" >&2
    exit 1
    ;;

  stop)
    # RViz shares :2 with `robot max`'s phone session. Tearing that down from a
    # browser tab would silently kill the phone's RViz too, so refuse.
    if [ "$APP" = rviz ]; then
      echo "refusing: :2 is the shared session started by 'robot max'" >&2
      exit 1
    fi
    tigervncserver -kill ":$DISP" >/dev/null 2>&1
    # tigervncserver -kill only signals Xvnc; ros2 launch children survive it,
    # and an orphaned gz server holds /clock so the next sim start dies silently.
    pkill -9 -f "DISPLAY=:$DISP" 2>/dev/null
    [ "$APP" = gazebo ] && pkill -9 -f 'gz sim' 2>/dev/null
    echo stopped
    ;;

  status)
    if [ -n "$APP" ]; then
      is_up "$DISP" && echo "running $PORT" || echo stopped
    else
      for a in rviz:2 gazebo:3 moveit:4; do
        n=${a%%:*}; d=${a##*:}
        is_up "$d" && echo "$n running $((5900 + d))" || echo "$n stopped"
      done
    fi
    ;;

  *)
    echo "usage: $0 {start|stop|status} {rviz|gazebo|moveit}" >&2
    exit 2
    ;;
esac
