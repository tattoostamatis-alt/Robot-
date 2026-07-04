#!/usr/bin/env bash
# Self-heal the Nav2 lifecycle chain: activate any managed node that is not
# `active`. Normally navigation_launch autostarts everything and this is a
# no-op, but on 2026-07-03 part of the chain (behavior_server,
# velocity_smoother, collision_monitor) came up inactive under full-stack
# boot load and goals silently went nowhere. Run by a TimerAction in
# bringup.launch.py; safe to run by hand anytime.
set -uo pipefail

NODES="controller_server smoother_server planner_server route_server
       behavior_server velocity_smoother collision_monitor bt_navigator
       waypoint_follower docking_server"

for attempt in 1 2 3; do
  all_active=true
  for n in $NODES; do
    state=$(timeout 5 ros2 lifecycle get /$n 2>/dev/null | cut -d' ' -f1)
    case "$state" in
      active) ;;
      unconfigured)
        all_active=false
        echo "[ensure_nav_active] /$n unconfigured -> configure+activate"
        timeout 20 ros2 lifecycle set /$n configure >/dev/null 2>&1
        timeout 20 ros2 lifecycle set /$n activate  >/dev/null 2>&1
        ;;
      inactive)
        all_active=false
        echo "[ensure_nav_active] /$n inactive -> activate"
        timeout 20 ros2 lifecycle set /$n activate >/dev/null 2>&1
        ;;
      *)
        # not discovered (yet) or mid-transition — recheck next attempt
        all_active=false
        echo "[ensure_nav_active] /$n state '$state' — will recheck"
        ;;
    esac
  done
  $all_active && { echo "[ensure_nav_active] nav chain fully active"; exit 0; }
  sleep 10
done

echo "[ensure_nav_active] WARNING: nav chain still not fully active after 3 attempts"
exit 1
