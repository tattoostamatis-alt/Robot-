#!/usr/bin/env python3
"""Read the three IR receivers while the robot stands still.

Stationary robot = the geometry is frozen, so any flicker left is the
receiver or the base, not the approach angle.

OI packets: 17 = omni, 52 = left, similar 53 = right.
Bits (on top of the 0xA0 Home Base base value):
  1 = Force Field, 4 = Green Buoy, 8 = Red Buoy
"""
import os, time, termios, collections

PORT = '/dev/roomba'
DUR = 20.0


def decode(v):
    if v == 0:
        return '—'
    parts = []
    if v & 8:
        parts.append('RED')
    if v & 4:
        parts.append('GREEN')
    if v & 1:
        parts.append('force')
    return '+'.join(parts) if parts else f'?{v}'


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
    os.write(fd, bytes([131])); time.sleep(0.3)   # Safe: no motion anyway

    counts = {'omni': collections.Counter(), 'left': collections.Counter(),
              'right': collections.Counter()}
    n = 0
    t0 = time.time()
    while time.time() - t0 < DUR:
        # query packets 17, 52, 53 in one shot: opcode 149, count 3
        os.write(fd, bytes([149, 3, 17, 52, 53]))
        time.sleep(0.12)
        try:
            d = os.read(fd, 3)
        except BlockingIOError:
            d = b''
        if len(d) == 3:
            n += 1
            counts['omni'][d[0]] += 1
            counts['left'][d[1]] += 1
            counts['right'][d[2]] += 1

    print(f'δείγματα: {n} σε {DUR:.0f}s (ρομπότ ΑΚΙΝΗΤΟ)\n')
    for name in ('omni', 'left', 'right'):
        c = counts[name]
        tot = sum(c.values()) or 1
        zero = c.get(0, 0)
        print(f'{name.upper():6}  σήμα {100*(tot-zero)/tot:5.1f}% του χρόνου')
        for v, k in c.most_common(4):
            print(f'         {v:4} {decode(v):14} {100*k/tot:5.1f}%')
        print()
finally:
    os.close(fd)
