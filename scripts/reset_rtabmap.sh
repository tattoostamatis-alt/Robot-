#!/bin/bash
# Reset RTAB-Map's on-disk database after a crash or corrupted session.
# Backs up the existing .db file with a timestamp before removing it.
# Usage:  ./reset_rtabmap.sh [db_path]
# Default db_path: ~/robot_ws/maps/rtabmap.db

DB="${1:-$HOME/robot_ws/maps/rtabmap.db}"

if [ ! -f "$DB" ]; then
    echo "No database found at $DB — nothing to reset."
    exit 0
fi

BAK="${DB%.db}_$(date +%Y%m%d_%H%M%S).db"
mv "$DB" "$BAK"
echo "Backed up to: $BAK"
echo "Database reset. Next bringup will start fresh."
