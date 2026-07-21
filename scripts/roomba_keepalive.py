#!/usr/bin/env python3
"""Keep the Roomba awake forever by holding it in OI Full mode.

Background: the Open Interface sleeps after 5 minutes of inactivity in Passive
mode, and once asleep this robot/cable ignores the BRC (RTS/DTR) wake pulse
entirely -- only a human pressing CLEAN brings it back (confirmed 2026-07-21 on
the 879). A BRC-pulse keep-alive is therefore unreliable here.

Full mode does NOT have the 5-minute sleep timer, so the fix is to hold the
robot in Full mode: every cycle we (re)assert start -> full. Once it is in Full
mode it never sleeps, so CLEAN is only ever needed ONCE after a power loss.

TRADE-OFF the user explicitly accepted (2026-07-21, "as xalaei mpataria"):
Full/Safe modes do NOT charge on the dock, so this slowly drains the battery
while the robot sits idle. That is the price of never having to press CLEAN.

It still stands down whenever roomba_driver.py is running -- only one process
can own /dev/roomba, and the driver manages OI mode itself while driving.
"""
import subprocess
import sys
import time

import serial

PORT = '/dev/roomba'
BAUD = 115200
INTERVAL_S = 60.0
DRIVER_PATTERN = 'roomba_driver.py'

# OI opcodes (iRobot Open Interface spec).
OI_START = 128   # enter OI, Passive mode
OI_FULL = 132    # Full mode: full control, and NO 5-minute sleep timer


def driver_running() -> bool:
    """True when the ROS driver owns the port and is managing OI itself."""
    return subprocess.run(
        ['pgrep', '-f', DRIVER_PATTERN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def assert_full_mode() -> None:
    """Best-effort BRC wake, then start -> Full. Idempotent: re-sending while
    already in Full mode is harmless and re-asserts it if anything reset it."""
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        # BRC wake (works on some cables, no-op on others -- harmless either way).
        ser.rts = True
        ser.dtr = True
        time.sleep(0.1)
        ser.rts = False
        ser.dtr = False
        time.sleep(0.1)
        # Enter OI, then Full mode (no auto-sleep).
        ser.write(bytes([OI_START]))
        time.sleep(0.2)
        ser.write(bytes([OI_FULL]))
        time.sleep(0.05)


def log(msg: str) -> None:
    print(msg, flush=True)          # journald picks this up


def main() -> int:
    log(f'roomba keep-alive started (hold Full mode, port={PORT}, '
        f'every {INTERVAL_S:.0f}s)')
    while True:
        if driver_running():
            log('roomba_driver.py is running — standing down this cycle')
        else:
            try:
                assert_full_mode()
            except serial.SerialException as e:
                # Port missing (robot unplugged) or briefly grabbed by someone
                # else: not fatal, retry next cycle.
                log(f'keep-alive skipped: {e}')
            except Exception as e:                       # noqa: BLE001
                log(f'keep-alive failed unexpectedly: {e!r}')
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
