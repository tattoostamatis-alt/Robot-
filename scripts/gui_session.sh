#!/usr/bin/env bash
# gui_session.sh — run the real Qt GUIs (RViz, MoveIt, Gazebo) on headless VNC
# displays so the web dashboard can stream them into the browser.
#
# The dashboard does NOT shell out to websockify: it bridges the RFB socket to a
# browser WebSocket itself (see web_dashboard_node.py), so everything stays on
# port 8080 behind the one dashboard token. This script's only job is owning the
# X displays and the app processes.
#
#   gui_session.sh start  rviz|moveit|gazebo|rtabmap|rtabview
#   gui_session.sh stop   rviz|moveit|gazebo|rtabmap|rtabview
#   gui_session.sh status [app]        # prints "running <vnc_port>" or "stopped"
#
# Display map — RViz keeps :2 on purpose. `robot max` already starts :2 for the
# phone (RealVNC → :5902), and running a SECOND RViz against the same graph has
# a documented failure mode: two RViz instances made AMCL deactivate and every
# Nav2 goal fail with planner error 208. So the dashboard attaches to the same
# session rather than spawning a rival one.
#
#   rviz     :2  → 5902   (shared with the phone's RealVNC session)
#   gazebo   :3  → 5903
#   moveit   :4  → 5904
#   rtabmap  :5  → 5905
#   rtabview :7  → 5907   (read-only viewer for a SAVED snapshot, arg 3 = filename)

set -u

WS=/home/dimi/robot_ws
VNCDIR=/home/dimi/.vnc

# ‼️ Gazebo runs on its OWN ROS domain. The robot lives on domain 0 (see
# deploy/systemd/*.service); the simulation is a second, complete robot —
# sim.launch.py brings up its own AMCL, map_server, planner, controller,
# collision_monitor, the lot — and on a shared domain those collide by name
# with the live ones.
#
# Measured 2026-08-02 by starting this tab against a running `robot max`:
# 18 duplicated nodes, /scan and /odom with two publishers each, and — worst —
# the sim's startup /initialpose landed in the REAL AMCL and teleported the
# robot's pose from (-7.01, 1.17) to (-6.09, 0.49). The UI note "don't open it
# while driving the real robot" understated it twice over: the damage does not
# need the robot to be driving, and no note can prevent a click.
#
# A separate domain makes the two graphs invisible to each other, so the tab is
# safe to open at any time. Nothing inside the sim notices — it is self-
# contained, and the user watches it over VNC either way.
SIM_DOMAIN_ID=77

case "${2:-}" in
  rviz)     DISP=2; GEOM=1600x900 ;;
  gazebo)   DISP=3; GEOM=1600x900 ;;
  moveit)   DISP=4; GEOM=1600x900 ;;
  rtabmap)  DISP=5; GEOM=1600x900 ;;
  rtabview) DISP=7; GEOM=1600x900 ;;
  '')       [ "${1:-}" = status ] || { echo "usage: $0 {start|stop|status} {rviz|gazebo|moveit|rtabmap|rtabview}" >&2; exit 2; } ;;
  *)        echo "unknown app '${2}' (rviz|gazebo|moveit|rtabmap|rtabview)" >&2; exit 2 ;;
esac
APP="${2:-}"
PORT=$((5900 + ${DISP:-0}))

# rtabview needs a THIRD arg: which saved snapshot (see save_snapshot in
# web_dashboard_node.py) to open. Read-only, its own process, its own display —
# never touches the live `rtabmap` app's database or domain.
RTAB_SAVED_DIR="/home/dimi/.home_robot/rtabmap/saved"
if [ "$APP" = rtabview ] && [ "${1:-}" = start ]; then
  SNAP_FILE="${3:-}"
  case "$SNAP_FILE" in
    '' | *[!A-Za-z0-9_.-]* )
      echo "rtabview needs a valid snapshot filename as arg 3" >&2; exit 2 ;;
  esac
  SNAP_PATH="$RTAB_SAVED_DIR/$SNAP_FILE"
  [ -f "$SNAP_PATH" ] || { echo "no such snapshot: $SNAP_FILE" >&2; exit 2; }
fi

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
# RViz restores the window size stored in robot.rviz, which is smaller than the
# Xvnc geometry — the dashboard tab then shows it floating on black.
( for _ in $(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "RViz" && {
      wmctrl -r "RViz" -b add,maximized_vert,maximized_horz; break; }
    sleep 2
  done ) &
exec rviz2 -d /home/dimi/robot_ws/install/home_robot/share/home_robot/config/robot.rviz
EOF
      ;;
    moveit)
      # arm_moveit.launch.py brings up move_group + its own RViz with the MoveIt
      # motion-planning panel (drag the gripper, plan, execute). It must run
      # ALONGSIDE bringup's arm_driver, never with a second one — the bridge in
      # that launch file forwards MoveIt's trajectories to the existing driver.
      #
      # Stays on the REAL domain, unlike gazebo: the whole point is to drive
      # the arm that is actually attached. `set -m` + the pgid file is only so
      # `stop` can take move_group and its RViz down together.
      cat > "$f" <<EOF
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
# Same as gazebo: RViz opens at its .rviz-saved size, leaving black around it
# in the tab. Match the display instead.
( for _ in \$(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "RViz" && {
      wmctrl -r "RViz" -b add,maximized_vert,maximized_horz; break; }
    sleep 2
  done ) &
set -m
ros2 launch home_robot arm_moveit.launch.py &
echo \$! > "$VNCDIR/app-moveit.pgid"
wait \$!
EOF
      ;;
    rtabmap)
      # RTAB-Map building the 3D map of the house from the D435, plus its own
      # Qt GUI (loop closures, the pose graph, the accumulated cloud).
      #
      # Stays on the REAL domain — unlike gazebo, the entire point is the house
      # the robot is actually standing in. What makes that safe is in
      # rtabmap.launch.py: publish_tf is false, so AMCL keeps sole ownership of
      # map->odom, and the node runs in the `rtabmap` namespace, so its own
      # /map, /global_path and friends cannot collide with map_server's and
      # Nav2's. Both were verified live before this tab existed; the namespace
      # one had already started publishing to the real /map when it was caught.
      #
      # ‼️ align_depth: the lean camera from localize.launch.py starts with
      # align_depth.enable false, and RTAB-Map needs depth registered to the
      # colour image or every point lands in the wrong place. It is settable at
      # runtime (23 Hz aligned depth appears within ~5 s, measured), so switch
      # it on here and back off in `stop` rather than making every session pay
      # for it. The dashboard does the same dance for the 3D tab's pointcloud.
      cat > "$f" <<EOF
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
ros2 param set /camera/camera align_depth.enable true >/dev/null 2>&1
# rtabmap_viz opens at its .ini-saved size, so the tab would show it floating
# on black exactly like RViz and Gazebo did.
# ‼️ The title is "RTAB-Map* [ROS]" — WITH the hyphen. Grepping for "rtabmap"
# never matches it, so the window stayed at its saved size inside a 1600x900
# display and the tab showed it floating on black.
( for _ in \$(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "RTAB-Map" && {
      wmctrl -r "RTAB-Map" -b add,maximized_vert,maximized_horz; break; }
    sleep 2
  done ) &
set -m
ros2 launch home_robot rtabmap.launch.py &
echo \$! > "$VNCDIR/app-rtabmap.pgid"
wait \$!
EOF
      ;;
    rtabview)
      # Read-only viewer for one SAVED snapshot (SNAP_PATH, validated above) —
      # rtabmap-databaseViewer opens a .db file directly with no ROS launch at
      # all, so this can never touch the live `rtabmap` app's database (which
      # runs --delete_db_on_start) or interrupt an in-progress mapping session.
      cat > "$f" <<EOF
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export LIBGL_ALWAYS_SOFTWARE=1
source /opt/ros/jazzy/setup.bash
openbox &
( for _ in \$(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "RTAB-Map Database Viewer" && {
      wmctrl -r "RTAB-Map Database Viewer" -b add,maximized_vert,maximized_horz; break; }
    sleep 2
  done
  # Auto-drive to the assembled 3D point cloud. The tool's default view is
  # per-frame "Constraints" (two camera images side by side) — useful for
  # debugging matches, useless for "what does the room look like". Sequence
  # verified by hand 2026-08-12 against this exact 1600x900 maximized layout:
  #   View menu > 3D view (opens a docked 3D viewport, Cloud checkbox OFF by
  #   default) > tick Cloud > Edit menu > View 3D map... (opens an "Export 3D
  #   Clouds" settings dialog) > OK (assembles every frame into one cloud,
  #   10-15s for a couple rooms).
  #
  # Tried also ticking Meshing here for a solid surface instead of a sparse
  # cloud (genuinely clearer when it lands) — reverted 2026-08-12. Even with
  # every click re-activating the window first, it silently no-op'd the
  # Meshing checkbox on about half of repeated identical runs (point-cloud-
  # only fallback, ~15s instead of ~130s, with no way to tell from outside
  # the app that it happened) — no cheap way to verify a Qt widget's checked
  # state from this script to retry on. Flaky automation of a "clearer" view
  # is worse than reliable automation of an OK one; if this is revisited,
  # the two extra clicks are in the git history of this file at this commit.
  #
  # ‼️ EVERY click below needs the window re-activated immediately first —
  # measured 2026-08-12, a plain mousemove+click right after any menu/dialog
  # opens silently no-ops about half the time otherwise (focus hasn't
  # actually landed on the new widget yet even though it is on screen). This
  # part reproduced reliably (2/2 clean runs) and is kept.
  # Best-effort only: coordinates drift if this app ever changes its default
  # layout, but nothing here is destructive — worst case the user lands on
  # the plain per-frame view and can click through it by hand.
  activate() { xdotool search --name "\$1" windowactivate 2>/dev/null; }
  sleep 2
  # A saved snapshot's embedded Core parameters can differ from this app's
  # defaults (they will, for anything saved off the live robot's settings),
  # which pops a blocking "Update parameters..." dialog BEFORE any menu is
  # usable — every click below silently lands on/behind it otherwise. Only
  # shows up sometimes (seen both ways across otherwise-identical runs, cause
  # unclear), so poll a few times rather than assume either way.
  for _ in \$(seq 1 5); do
    wmctrl -l 2>/dev/null | grep -q "Update parameters" && {
      activate "Update parameters"; sleep 1
      xdotool mousemove 876 464 click 1; sleep 1; break; }
    sleep 1
  done
  activate "RTAB-Map Database Viewer"; sleep 1
  xdotool mousemove 93 31 click 1; sleep 1            # View menu
  xdotool mousemove 127 205 click 1; sleep 2          # → 3D view
  activate "RTAB-Map Database Viewer"; sleep 1
  xdotool mousemove 69 272 click 1; sleep 1           # tick Cloud
  activate "RTAB-Map Database Viewer"; sleep 1
  xdotool mousemove 55 31 click 1; sleep 1            # Edit menu
  xdotool mousemove 105 340 click 1; sleep 2          # → View 3D map...
  activate "Export 3D Clouds"; sleep 1
  xdotool mousemove 879 867 click 1                   # OK on the export dialog
  # The assemble progress dialog is titled just "rtabmap-databaseViewer" (the
  # bare binary name, not the main window's "RTAB-Map Database Viewer") and
  # self-closes ("Close when done" is checked by default) — wait it out so a
  # screenshot/VNC view taken right after this script returns shows the
  # finished cloud. The initial sleep gives it time to actually appear first
  # — otherwise the very next poll can run before the dialog exists, find no
  # match, and the loop "finishes" immediately while assembly has barely
  # started.
  sleep 3
  for _ in \$(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "rtabmap-databaseViewer" || break
    sleep 2
  done
) &
exec rtabmap-databaseViewer "$SNAP_PATH"
EOF
      ;;
    gazebo)
      # The headless-sim launch file plus the Gazebo GUI client. This is a
      # SIMULATION — a whole second robot, with its own /clock, /scan, /odom
      # and its own Nav2 — so it is confined to SIM_DOMAIN_ID and cannot see
      # or be seen by the real one. See the note at the top of this file.
      #
      # `set -m` puts the launch in its own process group so `stop` can kill
      # the whole tree by PGID; without it the ROS nodes outlived every attempt
      # to shut the session down.
      cat > "$f" <<EOF
#!/bin/bash
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
# LIBGL_ALWAYS_SOFTWARE alone is not enough here — it steers GLX, and Mesa says
# so out loud on the EGL path ("Not allowed to force software rendering when API
# explicitly selects a hardware device"). GALLIUM_DRIVER + the loader override
# pick llvmpipe for both. The engine itself is chosen in sim.launch.py.
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export OGRE_RTT_MODE=Copy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=$SIM_DOMAIN_ID
source /opt/ros/jazzy/setup.bash
source /home/dimi/robot_ws/install/setup.bash
openbox &
# Gazebo opens at its saved ~/.gz size, not the display's, so the tab showed a
# window floating on black. openbox does not maximise anything by itself.
( for _ in \$(seq 1 60); do
    wmctrl -l 2>/dev/null | grep -q "Gazebo Sim" && {
      wmctrl -r "Gazebo Sim" -b add,maximized_vert,maximized_horz; break; }
    sleep 2
  done ) &
set -m
ros2 launch home_robot sim.launch.py headless:=false use_rviz:=false &
echo \$! > "$VNCDIR/app-gazebo.pgid"
wait \$!
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
    # Kill the app's process group FIRST, while we still know its PGID — once
    # Xvnc goes the pid file is all we have.
    #
    # ‼️ This used to be `pkill -9 -f "DISPLAY=:$DISP"`, which never matched a
    # single process: DISPLAY is an environment variable, not part of any
    # command line, and pkill -f only reads /proc/PID/cmdline. So `stop` printed
    # "stopped", tigervncserver dropped the X display, and every ROS node the
    # launch had started went on running reparented to init — 18 of them, still
    # publishing. Confirmed by /proc/<pid>/cmdline on 2026-08-02.
    #
    # The xstartup scripts put the launch in its own process group (set -m) and
    # write its PGID here, so a negative kill reaches the launch, its ROS nodes
    # and the gz server in one signal.
    pgidf="$VNCDIR/app-$APP.pgid"
    if [ -r "$pgidf" ]; then
      pgid=$(cat "$pgidf")
      if [ -n "$pgid" ]; then
        kill -TERM "-$pgid" 2>/dev/null      # let ROS shut down cleanly
        for _ in $(seq 1 10); do
          kill -0 "-$pgid" 2>/dev/null || break
          sleep 1
        done
        kill -9 "-$pgid" 2>/dev/null         # whatever ignored SIGTERM
      fi
      rm -f "$pgidf"
    fi
    tigervncserver -kill ":$DISP" >/dev/null 2>&1
    # Belt and braces for the sim: an orphaned gz server holds /clock, and the
    # next start then fails silently by falling back to a namespaced one.
    [ "$APP" = gazebo ] && pkill -9 -f 'gz sim' 2>/dev/null
    # Hand the camera back the way we found it. Aligned depth is a second 23 Hz
    # image stream on a machine that already runs at load ~6; leaving it on
    # after the mapping session would be a slow leak nobody would think to look
    # for. Best-effort: if the camera is not up there is nothing to restore.
    #
    # ‼️ set +u FIRST. This file runs under `set -u`, and ROS's own setup.bash
    # reads unbound variables (AMENT_TRACE_SETUP_FILES on line 8) — so sourcing
    # it aborts the subshell before ros2 is ever reached. With stderr sent to
    # /dev/null that failed in complete silence: `stop` printed "stopped" and
    # aligned depth stayed on. Measured after the first stop, which is the only
    # reason it was noticed at all.
    if [ "$APP" = rtabmap ]; then
      ( set +u
        source /opt/ros/jazzy/setup.bash
        timeout 40 ros2 param set /camera/camera align_depth.enable false ) >/dev/null 2>&1
    fi
    echo stopped
    ;;

  status)
    if [ -n "$APP" ]; then
      is_up "$DISP" && echo "running $PORT" || echo stopped
    else
      for a in rviz:2 gazebo:3 moveit:4 rtabmap:5; do
        n=${a%%:*}; d=${a##*:}
        is_up "$d" && echo "$n running $((5900 + d))" || echo "$n stopped"
      done
    fi
    ;;

  *)
    echo "usage: $0 {start|stop|status} {rviz|gazebo|moveit|rtabmap|rtabview}" >&2
    exit 2
    ;;
esac
