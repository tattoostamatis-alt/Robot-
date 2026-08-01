#!/usr/bin/env python3
"""Is the Roomba awake? Ask the Open Interface and say so, in one command.

Why this exists: a sleeping Roomba is SILENT, not noisy. It swallows every byte
without complaint, so "the robot does not move", "nav keeps timing out", "the
pose jumps" and "the base is asleep" all look identical from ROS. Several
debugging sessions have been spent on phantom navigation bugs that were only a
dozing base -- see the project notes on dead-base artifacts. Run this first.

Read-only as far as the robot is concerned: it sends 142 (query one sensor
packet) and, only if that gets no answer, 128 (Start -> Passive). Neither
touches the motors. Packet 35 is the OI mode itself, which is both the liveness
probe and the useful answer; battery packets are no use on this robot because
the pack is gone (powerbank), so charge readings are garbage.

‼️ Stands down if the port is busy. The roomba_driver owns /dev/roomba while it
runs and two processes cannot share the line -- opening it underneath a running
driver has broken a live session before. If the driver is up, ask ROS instead:

    ros2 topic hz /odom

‼️ Cannot wake the robot. BRC (mini-DIN pin 5) is physically dead on this unit
(verified 2026-07-28 at the electrical level: the DTR line really moves, the
robot answers on neither 115200 nor 19200). Only a CLEAN press wakes it.

Raw termios, never pyserial: the stock read path returns short/partial OI
replies on this adapter, which would misframe the answer and read as "asleep".

    ~/robot_ws/src/home_robot/scripts/roomba_awake.py

Exit status: 0 awake, 1 asleep/off, 2 could not check (port busy or missing).
"""
import fcntl
import os
import select
import struct
import subprocess
import sys
import termios
import time

PORT = '/dev/roomba'

OI_START = 128      # enter OI, Passive mode
OI_SENSORS = 142    # request one sensor packet
PKT_MODE = 35       # OI mode, 1 byte

MODE_NAMES = {0: 'OFF', 1: 'PASSIVE', 2: 'SAFE', 3: 'FULL'}

# ‼️ The 879 sleeps after ~90 SECONDS in Passive, not the 5 minutes the OI spec
# suggests (measured 2026-07-27). Only Passive runs that timer -- Safe and Full
# stay awake indefinitely, which is why the driver parks the robot in Full.
MODE_NOTES = {
    0: 'σβηστή — θέλει CLEAN',
    1: 'Passive: θα κοιμηθεί σε ~90s αν δεν την κρατήσει κάποιος ξύπνια',
    2: 'Safe: δεν κοιμάται, αλλά κάθε abort την υποβιβάζει σε Passive',
    3: 'Full: δεν κοιμάται ποτέ (εδώ την βάζει ο driver)',
}


def port_is_owned() -> bool:
    """True when another process holds the serial port open.

    Asks the kernel who has the device open rather than matching process names:
    `pgrep -f roomba_driver.py` matches the whole command line of every process,
    so a grep or an editor looks like a running driver (that false positive is
    documented in roomba_keepalive.py, which uses this same check).
    """
    try:
        return subprocess.run(['fuser', '-s', PORT],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False        # no fuser: proceed rather than refuse to run


def configure(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0                      # raw in/out/local
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # 8N1, no modem ctrl
    attrs[4] = attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0      # select() below gates the timing, not VTIME
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    # os.open() asserts DTR on Linux, which pulls BRC low the moment the port
    # opens. Release it so the line idles high, exactly as the driver does.
    fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack('I', termios.TIOCM_DTR))


def read_exact(fd: int, n: int, timeout: float = 0.5) -> bytes:
    """Accumulate until n bytes arrive or the timeout elapses.

    A single read() can return short while the reply is still in flight (FTDI
    latency timer), and a short read here would be indistinguishable from a
    robot that never answered.
    """
    buf = b''
    deadline = time.monotonic() + timeout
    while len(buf) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def query_mode(fd: int):
    """One OI-mode query. Returns the mode int, or None if nothing answered."""
    # Flush first, so a stale byte cannot be mistaken for this query's reply.
    termios.tcflush(fd, termios.TCIFLUSH)
    os.write(fd, bytes((OI_SENSORS, PKT_MODE)))
    data = read_exact(fd, 1)
    return data[0] if data else None


def main() -> int:
    if not os.path.exists(PORT):
        print(f'✘ {PORT} δεν υπάρχει — έλεγξε το USB και το udev rule.')
        return 2

    if port_is_owned():
        print(f'⏸  Το {PORT} το κρατάει άλλη διεργασία (μάλλον ο roomba_driver).')
        print('   Δεν το ανοίγω — δύο διεργασίες δεν μοιράζονται τη σειριακή.')
        print('   Ρώτα το ROS: ros2 topic hz /odom')
        return 2

    print(f'{PORT} -> {os.path.realpath(PORT)}')
    fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure(fd)
        mode = query_mode(fd)
        if mode is None:
            # No reply may just mean the OI was never opened this power cycle,
            # so try the handshake once before calling it asleep.
            print('   καμία απάντηση· στέλνω 128 (Start), δεν κινεί τροχούς...')
            os.write(fd, bytes((OI_START,)))
            time.sleep(0.3)
            mode = query_mode(fd)
    finally:
        os.close(fd)

    if mode is None:
        print()
        print('✘ ΚΟΙΜΑΤΑΙ (ή είναι σβηστή) — 0 bytes στο OI.')
        print('  Ξύπνα την: ΠΑΤΑ CLEAN πάνω στη βάση.')
        print('  (Το BRC pin 5 είναι νεκρό σε αυτό το ρομπότ — δεν γίνεται από εδώ.)')
        return 1

    name = MODE_NAMES.get(mode, f'άγνωστο ({mode})')
    print()
    print(f'✔ ΞΥΠΝΙΑ — απαντάει στο OI, mode {mode} ({name}).')
    note = MODE_NOTES.get(mode)
    if note:
        print(f'  {note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
