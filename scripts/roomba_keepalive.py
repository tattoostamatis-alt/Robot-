#!/usr/bin/env python3
"""Keep the Roomba from ever falling asleep while the mini PC is powered.

Why this exists: the Open Interface puts the robot to sleep after 5 minutes of
inactivity in Passive mode, and once asleep it ignores serial entirely -- the
only reliable way back is a human pressing CLEAN (confirmed 2026-07-21 on the
new 879 chassis: every query returned 0 bytes until the button was pressed).
roomba_driver.py already pulses BRC every 60s, but it only runs on demand, so
between sessions the robot is unattended and drifts off to sleep.

What it does: pulses the BRC line (RTS/DTR, same sequence as pycreate2.wake())
once a minute, which resets the 5-minute sleep timer -- see OI spec pg 7.

Two deliberate design choices:

1. It does NOT put the robot into Safe/Full mode. Passive is the only OI mode
   in which the Roomba keeps charging on its dock, and this daemon runs 24/7 --
   forcing Full here would quietly flatten the battery while it sits docked.
   Pulsing BRC is enough to prevent sleep on its own.

2. It stands down whenever roomba_driver.py is running. Only one process can
   own /dev/roomba, and the driver does its own keep-alive, so grabbing the
   port here would break live driving.
"""

import subprocess
import sys
import time

import serial

PORT = '/dev/roomba'
BAUD = 115200
INTERVAL_S = 60.0
DRIVER_PATTERN = 'roomba_driver.py'


def driver_running() -> bool:
    """True when the ROS driver owns the port and is handling keep-alive."""
    return subprocess.run(
        ['pgrep', '-f', DRIVER_PATTERN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def pulse_brc() -> None:
    """Same RTS/DTR toggle sequence as pycreate2.wake()."""
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        ser.rts = True
        ser.dtr = True
        time.sleep(1)
        ser.rts = False
        ser.dtr = False
        time.sleep(1)
        ser.rts = True
        ser.dtr = True
        time.sleep(1)


def log(msg: str) -> None:
    print(msg, flush=True)          # journald picks this up


def main() -> int:
    log(f'roomba keep-alive started (port={PORT}, every {INTERVAL_S:.0f}s)')
    while True:
        if driver_running():
            log('roomba_driver.py is running — standing down this cycle')
        else:
            try:
                pulse_brc()
            except serial.SerialException as e:
                # Port missing (robot unplugged) or briefly grabbed by someone
                # else: not fatal, just try again next cycle.
                log(f'BRC pulse skipped: {e}')
            except Exception as e:                       # noqa: BLE001
                log(f'BRC pulse failed unexpectedly: {e!r}')
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
