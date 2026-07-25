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

‼️ VERIFY, DON'T ASSUME (2026-07-25). This used to write start->full and call it
a success without ever checking whether anything answered, and it logged nothing
on the happy path -- so "working perfectly" and "shouting at a robot that has
been asleep since yesterday" produced identical (empty) journals. That is
exactly what happened: the robot slept overnight while the PC was off, the
service came up 7s after boot, poked a sleeping robot once a minute for 13
minutes, and reported nothing wrong. Every cycle now ends with a sensor query
and logs awake/asleep transitions, because a keep-alive that cannot tell you it
has failed is worse than none -- it hides the failure.

Note this daemon can only *prevent* sleep, never cure it: with BRC unwired,
nothing over serial wakes a sleeping robot. It also cannot cover the hours when
the PC is off, which is the window the robot actually dozes off in.
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
OI_START = 128     # enter OI, Passive mode
OI_FULL = 132      # Full mode: full control, and NO 5-minute sleep timer
OI_SENSORS = 142   # request one sensor packet
PKT_CHARGE = 25    # battery charge (2 bytes) -- used purely as a liveness probe


def driver_running() -> bool:
    """True when the ROS driver owns the port and is managing OI itself."""
    return subprocess.run(
        ['pgrep', '-f', DRIVER_PATTERN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def assert_full_mode() -> bool:
    """Best-effort BRC wake, then start -> Full. Idempotent: re-sending while
    already in Full mode is harmless and re-asserts it if anything reset it.

    Returns True only if the robot actually answered a sensor query afterwards.
    A sleeping Roomba silently swallows every byte, so writing the opcodes
    proves nothing on its own -- the reply is the only evidence the OI is live.
    """
    with serial.Serial(PORT, BAUD, timeout=0.4) as ser:
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
        # Liveness probe. reset_input_buffer() first so a stale byte from an
        # earlier cycle cannot be mistaken for this query's reply.
        ser.reset_input_buffer()
        ser.write(bytes([OI_SENSORS, PKT_CHARGE]))
        return len(ser.read(2)) == 2


def log(msg: str) -> None:
    print(msg, flush=True)          # journald picks this up


def main() -> int:
    log(f'roomba keep-alive started (hold Full mode, port={PORT}, '
        f'every {INTERVAL_S:.0f}s)')
    # Track three distinct states, not a bool: 'driver' (stood down), 'awake'
    # (we are holding it) and 'asleep'. Folding the first two into True hides
    # the handover -- when the ROS driver stops, the very line you want is
    # "keep-alive has taken over and the robot still answers". None = unknown,
    # so the first cycle after a restart always reports what it found. Only
    # transitions are logged: at once a minute, logging every cycle would bury
    # the one line that matters under a day of "still fine".
    state = None
    while True:
        if driver_running():
            if state != 'driver':
                log('roomba_driver.py is running — standing down (it keeps the '
                    'robot awake itself)')
                state = 'driver'
        else:
            try:
                awake = assert_full_mode()
                new_state = 'awake' if awake else 'asleep'
                if new_state != state:
                    if awake:
                        log('robot is AWAKE and held in Full mode by keep-alive')
                    else:
                        log('robot is ASLEEP — it ignores serial entirely and '
                            'BRC is not wired, so only a CLEAN press wakes it. '
                            'Keep-alive cannot recover this on its own.')
                    state = new_state
            except serial.SerialException as e:
                # Port missing (robot unplugged) or briefly grabbed by someone
                # else: not fatal, retry next cycle.
                log(f'keep-alive skipped: {e}')
                state = None
            except Exception as e:                       # noqa: BLE001
                log(f'keep-alive failed unexpectedly: {e!r}')
                state = None
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
