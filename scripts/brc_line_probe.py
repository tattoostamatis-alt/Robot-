#!/usr/bin/env python3
"""Does the USB adapter physically move DTR/RTS at all? Toggle slowly and probe.

Why this exists: BRC (mini-DIN pin 5) wakes a sleeping Roomba, and on this robot
it has never worked. Software has been ruled out -- a USB-serial adapter gives
the host exactly two controllable lines, DTR and RTS, and 2026-08-01 swept both,
in both polarities, at two pulse widths, at both bauds, against a confirmed
sleeping robot: ten attempts, zero replies. There is no third line left to try.

‼️ What TIOCMGET does NOT prove. Reading the line back confirms the FTDI's
register changed, i.e. the host and the chip agree. It says nothing about a
physical pin: plenty of cheap USB-TTL modules never break DTR/RTS out to a pad
at all, and the register still reads back perfectly. Earlier notes in this repo
claimed the readback proved "the line really moves" -- it does not, and that
claim sent the diagnosis toward a cut wire when the adapter itself is at least
as likely.

So: this drives one line at a time, slowly enough to follow with a multimeter,
and announces every transition. Probe pad-to-GND on the adapter.

  moves (≈0 V <-> 3.3/5 V) -> the adapter is fine; the break is between that pad
                              and pin 5. Wire it.
  never moves              -> the adapter does not expose the line. No amount of
                              code fixes that; use an adapter that breaks it out.

FTDI drives modem-control lines ACTIVE LOW: asserted reads ~0 V, released reads
high. Either way what matters here is only whether the voltage CHANGES.

Roomba 7-pin mini-DIN: 1,2 = Vpwr | 3 = RXD | 4 = TXD | 5 = BRC | 6,7 = GND.
Data works on this robot, so 3/4/6/7 are known good -- only pin 5 is in question.

    ~/robot_ws/src/home_robot/scripts/brc_line_probe.py          # both lines
    ~/robot_ws/src/home_robot/scripts/brc_line_probe.py DTR      # just one

‼️ Stop the keep-alive first, or it will grab the port mid-probe:
    sudo systemctl stop roomba-keepalive.service
    ... probe ...
    sudo systemctl start roomba-keepalive.service
"""
import fcntl
import os
import struct
import subprocess
import sys
import termios
import time

PORT = '/dev/roomba'
HOLD_S = 3.0        # long enough to read a multimeter without rushing
CYCLES = 8          # per line

LINES = {'DTR': termios.TIOCM_DTR, 'RTS': termios.TIOCM_RTS}


KEEPALIVE = 'roomba-keepalive.service'


def port_is_owned() -> bool:
    """Same fuser check the keep-alive uses -- two owners on one serial line is
    exactly the corruption everything here tries to avoid."""
    try:
        return subprocess.run(['fuser', '-s', PORT],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False


def keepalive_running() -> bool:
    """‼️ fuser alone is not enough HERE, unlike everywhere else in this repo.

    The keep-alive holds the port for a fraction of each 30 s cycle, so a
    point-in-time fuser check almost always finds it free -- and then, halfway
    through the probe, it opens the port and os.open() ASSERTS DTR. That injects
    transitions this script did not command, on the very line being measured,
    while someone is watching a multimeter. So ask systemd whether the service
    is up at all, not whether it happens to hold the port this instant.
    """
    try:
        return subprocess.run(['systemctl', 'is-active', '--quiet', KEEPALIVE]
                              ).returncode == 0
    except FileNotFoundError:
        return False


def main() -> int:
    wanted = [a.upper() for a in sys.argv[1:]] or list(LINES)
    unknown = [w for w in wanted if w not in LINES]
    if unknown:
        print(f'Άγνωστη γραμμή: {unknown}. Διάλεξε από {list(LINES)}.')
        return 2

    if not os.path.exists(PORT):
        print(f'✘ {PORT} δεν υπάρχει.')
        return 2
    if keepalive_running():
        print(f'✘ Το {KEEPALIVE} τρέχει — κάθε κύκλος του τραβάει το DTR και θα')
        print('  μολύνει τη μέτρηση στη μέση. Σταμάτησέ το πρώτα:')
        print(f'    sudo systemctl stop {KEEPALIVE}')
        return 2
    if port_is_owned():
        print(f'✘ Το {PORT} το κρατάει άλλη διεργασία (ο ROS driver;).')
        return 2

    fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        a = termios.tcgetattr(fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = termios.B115200
        a[6][termios.VMIN] = a[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, a)

        def drive(bit, on):
            fcntl.ioctl(fd, termios.TIOCMBIS if on else termios.TIOCMBIC,
                        struct.pack('I', bit))

        for bit in LINES.values():      # park everything, incl. the edge os.open made
            drive(bit, False)

        print('Βάλε το ΜΑΥΡΟ probe σε GND του αντάπτορα και το ΚΟΚΚΙΝΟ στο pad')
        print('της γραμμής που δοκιμάζουμε. Ψάχνουμε ΑΛΛΑΓΗ τάσης, όχι τιμή.\n')

        for name in wanted:
            bit = LINES[name]
            print(f'── {name} — {CYCLES} κύκλοι × {HOLD_S:.0f}s ──')
            for i in range(1, CYCLES + 1):
                for on in (True, False):
                    drive(bit, on)
                    level = 'ASSERTED (~0 V)' if on else 'RELEASED (high)'
                    print(f'  {i}/{CYCLES}  {name} = {level}', flush=True)
                    time.sleep(HOLD_S)
            drive(bit, False)
            print()
    except KeyboardInterrupt:
        print('\n(διακόπηκε)')
    finally:
        for bit in LINES.values():
            try:
                fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack('I', bit))
            except OSError:
                pass
        os.close(fd)

    print('Άλλαξε τάση κάποιο pad;')
    print('  ΝΑΙ -> ο αντάπτορας δίνει τη γραμμή· λείπει η καλωδίωση ως το pin 5.')
    print('  ΟΧΙ -> ο αντάπτορας δεν τη βγάζει καθόλου· θέλει άλλον αντάπτορα.')
    print('\nΜην ξεχάσεις: sudo systemctl start roomba-keepalive.service')
    return 0


if __name__ == '__main__':
    sys.exit(main())
