#!/usr/bin/env python3
"""Wake the Roomba when the PC boots, then keep it awake for as long as the PC
is on.

Two jobs, in order:

1. WAKE. On start (boot, resume from suspend, or a service restart) pulse the
   BRC line and try to talk OI. The robot has usually dozed off while the PC was
   off, and that is exactly the window nothing else covers.
2. HOLD. Once it answers, re-assert Full mode every cycle. Full mode has no
   inactivity sleep timer, so the robot never dozes off again while we run.

TRADE-OFF the user explicitly accepted (2026-07-21, "as xalaei mpataria"):
Full/Safe modes do NOT charge on the dock, so this slowly drains the battery
while the robot sits idle. That is the price of never having to press CLEAN.

It stands down whenever roomba_driver.py is running -- only one process can own
/dev/roomba, and the driver manages OI mode itself while driving.

‼️ TIMING. In Passive the 879 sleeps somewhere between 3 and 5 minutes -- awake
at 180 s, gone by 300 s (re-measured 2026-08-01). Essentially the OI spec's 5
minutes. The cycle below must stay well under that; at the old 60s it could poke
a robot that had already dozed off in the gap.

  The header used to claim ~90 SECONDS, "measured 2026-07-27". That number does
  not reproduce, and the likely reason is a trap worth knowing about: os.open()
  asserts DTR on Linux, i.e. it pulls BRC LOW the instant the port opens (see
  roomba_driver.py, which releases it explicitly for this reason). A liveness
  check that REOPENS the port therefore delivers its own BRC falling edge, so a
  robot that dozed off and was woken by that edge reads as "still awake" -- and
  the moment it finally reads asleep looks like the sleep threshold. Measure
  this by opening the port ONCE and keeping it open, never by reopening.

  Nothing here changed as a result: 30 s is comfortably inside either figure.
  The old number was simply wrong about the hardware, presented as measured.

‼️ VERIFY, DON'T ASSUME (2026-07-25). This used to write start->full and call it
a success without ever checking whether anything answered, and it logged nothing
on the happy path -- so "working perfectly" and "shouting at a robot that has
been asleep since yesterday" produced identical (empty) journals. Every cycle
now ends with a sensor query and logs awake/asleep transitions, because a
keep-alive that cannot tell you it has failed is worse than none.

‼️ BRC IS PHYSICALLY DEAD (mini-DIN pin 5). First found 2026-07-28; RE-TESTED
2026-08-01 with the reopen artefact above ruled out, same result. That run is the
one to reproduce, because it is the only design that can tell the difference:

    port opened ONCE and held; 128 -> Passive confirmed; awake at 180 s; asleep
    (0 bytes) at 300 s with no falling edge delivered in between; then a single
    pulse, DTR read back LOW mid-pulse via TIOCMGET; 0 bytes at 115200 AND at
    19200 (baud switched on the same fd, still no reopen).

So the wake path here is correct and proven live at the electrical level -- our
side of the wire really moves -- and the signal still does not reach the robot.
Only a CLEAN press wakes it; this daemon says so instead of failing silently.
Suspects, all physical: a cut pin-5 wire, the wrong pin in the mini-DIN, a bad
contact, or the power bank cutting output. When any is fixed, wake starts working
here with no code change.

‼️ One pulse per attempt, always. Three inside 5 s put the robot on 19200 baud
(OI spec), and a robot silently sitting on the wrong baud looks exactly like a
dead one -- which is why the test above checks 19200 before concluding anything.

✔ A POWER CYCLE WAKES IT (found 2026-08-01). Unplugging the robot from the power
bank for ~10 s and plugging it back in brings the OI straight back, no CLEAN.
Attribution is solid rather than assumed: this daemon's own handshake -- which
sends 128 -- had been answered with silence three times at 13:28-13:29, so the
robot was genuinely asleep and not merely sitting in OFF (a powered robot in OFF
ignores a bare query but DOES answer 128, which is the trap to avoid here); the
power cycle was the only thing that happened before it answered 128 at 13:35.

  Not automatable as wired today: the robot hangs off a POWER BANK, which the PC
  cannot switch. It becomes automatable the moment there is a PC-controlled
  switch in that DC line (a USB relay is the cheap option) -- at which point this
  daemon can power-cycle instead of asking, and the CLEAN press disappears for
  good. Deliberately not written until that switch exists: untestable code for
  absent hardware is how the BRC path came to look finished for months.
"""
import fcntl
import os
import struct
import subprocess
import sys
import termios
import time

PORT = '/dev/roomba'

# ── Cycle timing ──────────────────────────────────────────────────────────────
# 30s, comfortably inside the 3-5 min Passive-mode sleep window (see header).
INTERVAL_S = 30.0

# ── OI opcodes (iRobot Open Interface spec) ──────────────────────────────────
OI_START = 128     # enter OI, Passive mode
OI_FULL = 132      # Full mode: full control, and NO inactivity sleep timer
OI_SENSORS = 142   # request one sensor packet
PKT_CHARGE = 25    # battery charge (2 bytes) -- used purely as a liveness probe

# ── BRC pulse shape ──────────────────────────────────────────────────────────
BRC_IDLE_HIGH_S = 0.3    # let the line settle HIGH so the falling edge is real
BRC_PULSE_S = 0.6        # OI wants the line held low >500ms to register
BRC_SETTLE_S = 0.5       # the OI does not listen the instant BRC goes back high
# ‼️ SAFETY, not tuning: per the OI spec, three BRC pulses within 5 seconds
# switch the robot to 19200 baud. One pulse per cycle stays far clear of that,
# and the boot-time retries below are spaced deliberately wide for the same
# reason -- a robot silently sitting on the wrong baud rate looks exactly like a
# dead one.
BOOT_WAKE_ATTEMPTS = 3
BOOT_WAKE_GAP_S = 12.0


def port_is_owned() -> bool:
    """True when another process holds the serial port open.

    Asks the kernel who has the device open rather than matching process names.
    ``pgrep -f roomba_driver.py`` (what this used to do) matches the *whole
    command line* of every process, so anything that merely mentions the string
    -- a grep, an editor, a shell one-liner during debugging -- looked like a
    running driver and made this daemon stand down while nothing was actually
    keeping the robot awake. Observed 2026-07-25, where one of our own diagnostic
    commands produced exactly that false positive.

    Owning the port is also the question that actually matters: the reason to
    stand down is that two processes cannot share the serial line, not that some
    file is named roomba_driver.py.
    """
    return subprocess.run(
        ['fuser', '-s', PORT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _set_brc_low(fd: int, low: bool) -> None:
    """Drive the BRC line. On a USB-TTL adapter, asserting DTR pulls the pin to
    0V, which is the active (wake) level BRC wants."""
    fcntl.ioctl(fd,
                termios.TIOCMBIS if low else termios.TIOCMBIC,
                struct.pack('I', termios.TIOCM_DTR))


def _brc_pulse(fd: int) -> None:
    """One clean falling edge on BRC.

    ‼️ The release-first step is the whole point. Opening a tty asserts DTR --
    the BRC line is already LOW before we touch it -- so the old sequence
    (``ser.dtr = True`` then False) drove low from low: no falling edge, ever.
    The robot saw BRC held down forever and the only real edge it ever got was
    the port being opened. Same bug the ROS driver had; fixed there 2026-07-28.
    """
    _set_brc_low(fd, False)
    time.sleep(BRC_IDLE_HIGH_S)
    _set_brc_low(fd, True)
    time.sleep(BRC_PULSE_S)
    _set_brc_low(fd, False)
    time.sleep(BRC_SETTLE_S)


def _configure(fd: int, baud: int = termios.B115200) -> None:
    """8N1, no flow control, 0.5s read timeout.

    Raw termios rather than pyserial: pyserial asserts DTR on open with no way
    to stop it, which is precisely the edge-killing behaviour above, and the
    driver already learned that its ioctls hang on the CH340 after a hub reset.
    """
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0          # iflag, oflag, lflag: raw
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[4] = attrs[5] = baud                  # ispeed, ospeed
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 5                 # 0.5s
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def _handshake(fd: int) -> bool:
    """start -> Full, then prove the robot is really listening.

    A sleeping Roomba silently swallows every byte, so writing the opcodes
    proves nothing on its own -- the reply is the only evidence the OI is live.
    """
    os.write(fd, bytes([OI_START]))
    time.sleep(0.3)
    os.write(fd, bytes([OI_FULL]))
    time.sleep(0.2)
    # Flush first so a stale byte from an earlier cycle cannot be mistaken for
    # this query's reply.
    termios.tcflush(fd, termios.TCIFLUSH)
    os.write(fd, bytes([OI_SENSORS, PKT_CHARGE]))
    time.sleep(0.4)
    try:
        return len(os.read(fd, 2)) == 2
    except BlockingIOError:
        return False


def assert_full_mode(pulse: bool) -> bool:
    """BRC wake (optional), then start -> Full. Idempotent: re-sending while
    already in Full mode is harmless and re-asserts it if anything reset it.

    ``pulse`` is skipped once the robot is known awake -- it is already in Full
    mode, so there is nothing to wake, and not pulsing keeps us further away
    from the 3-pulses/5s baud-switch rule.

    Returns True only if the robot actually answered a sensor query.
    """
    fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        _configure(fd)
        if pulse:
            _brc_pulse(fd)
        if _handshake(fd):
            return True
        if not pulse:
            return False
        # Nothing at 115200. A stray triple-pulse in the past can have left the
        # robot on 19200, where it is alive but mute to us -- indistinguishable
        # from asleep unless we actually look.
        _configure(fd, termios.B19200)
        if _handshake(fd):
            log('robot answered at 19200 baud — it had been switched by BRC '
                'triple-pulsing. Reset it with a CLEAN press (or 10s power off) '
                'to get back to 115200.')
            return True
        return False
    finally:
        os.close(fd)


def log(msg: str) -> None:
    print(msg, flush=True)          # journald picks this up


def notify(title: str, body: str) -> None:
    """Best-effort desktop popup, so a sleeping robot is visible without
    reading the journal -- the whole failure mode here is being silent."""
    env = dict(os.environ,
               DISPLAY=os.environ.get('DISPLAY', ':0'),
               DBUS_SESSION_BUS_ADDRESS=os.environ.get(
                   'DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{os.getuid()}/bus'))
    try:
        subprocess.run(['notify-send', '-u', 'critical', title, body],
                       env=env, timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:                                    # noqa: BLE001
        pass                                             # never fatal


def boot_wake() -> bool:
    """Aggressive wake burst at startup.

    This runs when the PC has just booted or resumed, i.e. after the robot has
    been unattended for hours -- the one moment it is almost certainly asleep
    and worth several attempts rather than one. Attempts are spaced
    BOOT_WAKE_GAP_S apart to stay clear of the baud-switch rule.
    """
    for attempt in range(1, BOOT_WAKE_ATTEMPTS + 1):
        if port_is_owned():
            log('startup wake skipped: the ROS driver already owns the port')
            return True
        try:
            if assert_full_mode(pulse=True):
                log(f'startup wake: robot is AWAKE (attempt {attempt})')
                return True
            log(f'startup wake: no answer (attempt {attempt}/{BOOT_WAKE_ATTEMPTS})')
        except OSError as e:
            log(f'startup wake: port unavailable ({e})')
        if attempt < BOOT_WAKE_ATTEMPTS:
            time.sleep(BOOT_WAKE_GAP_S)
    log('startup wake FAILED — CLEAN press, or unplug the robot from the power '
        'bank for ~10s and plug it back in (verified 2026-08-01: a power cycle '
        'brings the OI back with no CLEAN at all). BRC (mini-DIN pin 5) does not '
        'reach the robot, so serial cannot do either for us.')
    notify('Roomba is asleep',
           'The PC is up but the robot did not answer. Press CLEAN, or power-cycle '
           'it at the power bank for ~10s — either brings it back.')
    return False


def main() -> int:
    log(f'roomba keep-alive started (wake, then hold Full mode; port={PORT}, '
        f'every {INTERVAL_S:.0f}s)')
    awake = boot_wake()
    # Track three distinct states, not a bool: 'driver' (stood down), 'awake'
    # (we are holding it) and 'asleep'. Folding the first two into True hides
    # the handover -- when the ROS driver stops, the very line you want is
    # "keep-alive has taken over and the robot still answers". Seeded from the
    # boot burst so it does not re-log what boot_wake just said. Only
    # transitions are logged: at twice a minute, logging every cycle would bury
    # the one line that matters under a day of "still fine".
    state = 'awake' if awake else 'asleep'
    while True:
        time.sleep(INTERVAL_S)
        if port_is_owned():
            if state != 'driver':
                log(f'{PORT} is owned by another process (the ROS driver) — '
                    'standing down; it keeps the robot awake itself')
                state = 'driver'
            continue
        try:
            # Only pulse BRC when we think it is asleep: an awake robot is
            # already in Full mode and needs no edge.
            awake = assert_full_mode(pulse=(state != 'awake'))
            new_state = 'awake' if awake else 'asleep'
            if new_state != state:
                if awake:
                    log('robot is AWAKE and held in Full mode by keep-alive')
                else:
                    log('robot went ASLEEP (or lost power) — press CLEAN, or '
                        'power-cycle it at the power bank for ~10s. BRC wake over '
                        'serial does not reach it, so we cannot do either.')
                    notify('Roomba fell asleep',
                           'Press CLEAN, or unplug it from the power bank for ~10s.')
                state = new_state
        except OSError as e:
            # Port missing (robot unplugged) or briefly grabbed by someone
            # else: not fatal, retry next cycle.
            log(f'keep-alive skipped: {e}')
            state = None
        except Exception as e:                           # noqa: BLE001
            log(f'keep-alive failed unexpectedly: {e!r}')
            state = None


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
