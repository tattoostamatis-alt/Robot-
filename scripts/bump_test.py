#!/usr/bin/env python3
"""Is the physical bumper wired up? Read OI packet 7 for 25 s and report.

Packet 7 (Bumps and Wheel Drops), bit 0 = bump right, bit 1 = bump left,
bits 2-3 = wheel drops. Also reads packet 45 (Light Bumper) — the non-contact
IR bumper on 600/800 series, bits 0-5 left..right.

Raw termios (never pyserial on this CH340). Driver must be stopped.
"""
import os, time, termios

PORT = '/dev/roomba'
DUR = 25.0

fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
try:
    a = termios.tcgetattr(fd)
    a[0] = a[1] = a[3] = 0
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[4] = a[5] = termios.B115200
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 2
    termios.tcsetattr(fd, termios.TCSANOW, a)
    time.sleep(0.2)
    os.write(fd, bytes([128])); time.sleep(0.3)
    os.write(fd, bytes([131])); time.sleep(0.3)

    print('ΠΑΤΑ ΤΟΝ ΠΡΟΦΥΛΑΚΤΗΡΑ ΜΕ ΤΟ ΧΕΡΙ (25s)...', flush=True)
    seen_bump, seen_light, n = set(), set(), 0
    t0 = time.time()
    while time.time() - t0 < DUR:
        os.write(fd, bytes([149, 2, 7, 45]))   # packet 7 + packet 45
        time.sleep(0.1)
        try:
            d = os.read(fd, 2)
        except BlockingIOError:
            d = b''
        if len(d) == 2:
            n += 1
            b, lb = d[0], d[1]
            if b: seen_bump.add(b)
            if lb: seen_light.add(lb)
            if b or lb:
                names = []
                if b & 1: names.append('BUMP-ΔΕΞΙΑ')
                if b & 2: names.append('BUMP-ΑΡΙΣΤΕΡΑ')
                if b & 4: names.append('ρόδα-αριστ')
                if b & 8: names.append('ρόδα-δεξ')
                if lb:    names.append(f'light=0b{lb:06b}')
                print('  ' + ', '.join(names), flush=True)

    print(f'\nδείγματα: {n}')
    print(f'  μηχανικό bump (p7):  {"ΝΑΙ " + str(sorted(seen_bump)) if seen_bump else "ΠΟΤΕ — καμία ένδειξη"}')
    print(f'  light bumper (p45):  {"ΝΑΙ " + str(sorted(seen_light)) if seen_light else "ΠΟΤΕ — καμία ένδειξη"}')
finally:
    os.close(fd)
