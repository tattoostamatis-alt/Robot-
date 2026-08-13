#!/usr/bin/env python3
"""Direction of Arrival + LED control — ReSpeaker XVF3800.

Reads hardware DoA from the XVF3800 DSP and controls the onboard LED ring
to give visual feedback about the robot's listening state.

Topics published:
  /doa/angle       (Float32) — angle [0-359°] when speech detected
  /doa/wake        (Float32) — angle sampled at wake_word moment
  /voice_activity  (Bool)    — hardware VAD, on the DSP's own speech flag
  /cmd_vel_safe    (Twist)   — turn toward the speaker (‼️ not /cmd_vel)
  /doa/rotate_state (Bool)   — latched; is turning toward the speaker armed?

Topics subscribed:
  /wake_word    (String)  — triggers LISTENING LED state (+ turn, only if armed)
  /speech_text  (String)  — transitions LED back to IDLE
  /doa/rotate_enable (Bool) — the dashboard's switch; arms/disarms the turn
  /emergency_stop    (Bool) — latched; never turn while it is set

‼️ Turning the base toward whoever spoke is OFF unless armed, from the web
dashboard (Ρυθμίσεις → «Στρίψε προς τη φωνή») or `rotate_on_wake:=true`. It is
the one thing here that moves the robot with no command given, and a wake word
is not a command.

LED states:
  IDLE       — DoA mode (mode=4): 1 LED points toward detected speaker
  LISTENING  — breath blue: robot is recording the command
  PROCESSING — breath orange: Whisper is transcribing
  IDLE again — DoA mode resumes after speech_text or timeout

DoA convention (XVF3800): 0° = front of device, increases clockwise.
ROS angular.z positive = counter-clockwise — rotation is negated accordingly.
"""

import json
import math
import os
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
import usb.core
import usb.util


# Where the web toggle is remembered, so switching it on does not have to be
# redone after every `robot max`. Same directory as the gesture bindings.
_STATE_PATH = os.path.expanduser('~/.ros/home_robot_doa_rotate.json')


VID = 0x2886
PID = 0x001A

# LED effect modes
LED_OFF         = 0
LED_BREATH      = 1
LED_RAINBOW     = 2
LED_SINGLE      = 3
LED_DOA         = 4
LED_RING        = 5

# Colors (0xRRGGBB as uint32)
COLOR_BLUE      = 0x0000FF
COLOR_ORANGE    = 0xFF6600
COLOR_GREEN     = 0x00FF00
COLOR_WHITE     = 0xFFFFFF
COLOR_OFF       = 0x000000

# _PARAMS: (resid, cmdid, data_count, access, data_type)
_PARAMS = {
    'DOA_VALUE':    (20, 18,  2, 'ro', 'uint16'),
    'LED_EFFECT':   (20, 12,  1, 'rw', 'uint8'),
    'LED_BRIGHTNESS':(20, 13, 1, 'rw', 'uint8'),
    'LED_SPEED':    (20, 15,  1, 'rw', 'uint8'),
    'LED_COLOR':    (20, 16,  1, 'rw', 'uint32'),
    'LED_DOA_COLOR':(20, 17,  2, 'rw', 'uint32'),
    'LED_GAMMIFY':  (20, 14,  1, 'rw', 'uint8'),
}

TIMEOUT = 100_000


class XVF3800:
    def __init__(self, dev):
        self._dev = dev
        self._lock = threading.Lock()

    def _ctrl_in(self, cmdid, resid, length):
        return self._dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | cmdid, resid, length, TIMEOUT)

    def _ctrl_out(self, cmdid, resid, payload):
        self._dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, payload, TIMEOUT)

    def read_doa(self):
        """Returns (speech_detected: bool, angle_deg: int 0-359)."""
        resid, cmdid, cnt, _, dtype = _PARAMS['DOA_VALUE']
        # cnt=2 uint16s → 4 bytes + 1 status byte
        with self._lock:
            resp = self._ctrl_in(cmdid, resid, cnt * 2 + 1)
        angle, speech = struct.unpack_from('<HH', resp.tobytes(), 1)
        return bool(speech), int(angle)

    def set_led_effect(self, mode: int):
        resid, cmdid, _, _, _ = _PARAMS['LED_EFFECT']
        with self._lock:
            self._ctrl_out(cmdid, resid, bytes([mode]))

    def set_led_brightness(self, brightness: int):
        resid, cmdid, _, _, _ = _PARAMS['LED_BRIGHTNESS']
        with self._lock:
            self._ctrl_out(cmdid, resid, bytes([brightness]))

    def set_led_speed(self, speed: int):
        resid, cmdid, _, _, _ = _PARAMS['LED_SPEED']
        with self._lock:
            self._ctrl_out(cmdid, resid, bytes([speed]))

    def set_led_color(self, color: int):
        resid, cmdid, _, _, _ = _PARAMS['LED_COLOR']
        with self._lock:
            self._ctrl_out(cmdid, resid, struct.pack('<I', color))

    def set_led_doa_colors(self, base_color: int, doa_color: int):
        resid, cmdid, _, _, _ = _PARAMS['LED_DOA_COLOR']
        with self._lock:
            self._ctrl_out(cmdid, resid, struct.pack('<II', base_color, doa_color))

    def close(self):
        usb.util.dispose_resources(self._dev)


def _find_device():
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    return XVF3800(dev) if dev else None


class DoaNode(Node):

    # LED state machine values
    _STATE_IDLE       = 'idle'
    _STATE_LISTENING  = 'listening'
    _STATE_PROCESSING = 'processing'

    def __init__(self):
        super().__init__('doa_node')

        self.declare_parameter('poll_hz',           10.0)
        # ‼️ OFF by default, and it stays off until the web dashboard turns it
        # on (Ρυθμίσεις → «Στρίψε προς τη φωνή»). This used to default to True,
        # which meant the base turned itself the instant the wake word fired —
        # before any command was spoken, and with no yes/no confirmation, the
        # guard every other motion path got after the "robot executes overheard
        # conversation" bug. Measured 2026-08-04 on a plain `robot max` with
        # nobody talking to the robot: 6 unrequested turns in 100 s, several of
        # them 155°, off wake words the user never addressed to it. The LED ring
        # already shows where the speaker is; turning the wheels is the nicety,
        # and a nicety does not get to move a robot on its own.
        self.declare_parameter('rotate_on_wake',    False)
        self.declare_parameter('rotate_speed',      0.6)
        self.declare_parameter('min_angle_deg',     20.0)
        self.declare_parameter('led_enabled',       True)
        self.declare_parameter('led_brightness',    150)
        # Per-state colours, 0xRRGGBB — pulled out of _apply_led_state's body
        # (they used to be hardcoded there) so the dashboard's mic-settings
        # card can retune them live, same mechanism as led_brightness below.
        self.declare_parameter('led_color_idle_base',    0x111111)
        self.declare_parameter('led_color_idle_pointer', 0x0000FF)
        self.declare_parameter('led_color_listening',    COLOR_BLUE)
        self.declare_parameter('led_color_processing',   COLOR_ORANGE)
        # Privacy mute (wake_word_node's 'muted' param, relayed over
        # mic/muted): takes over the ring from whatever state the voice
        # pipeline is in, so the ring never shows "idle/listening" while the
        # mic is actually off.
        self.declare_parameter('led_color_muted',        0xFF0000)
        self.declare_parameter('listen_timeout',    12.0)  # max STT window seconds
        # ‼️ Do not turn the base toward a voice while something else is driving
        # it. The only guard used to be "not already rotating", so saying the
        # wake word mid-goto put this node and Nav2's controller on cmd_vel at
        # the same time — six independent publishers share that topic and the
        # last write of each 20 Hz driver tick wins. Turning toward the speaker
        # is a nicety; not fighting the navigator is not.
        self.declare_parameter('rotate_block_s',    1.5)   # s since the last drive command
        # One wake word, one turn. The XVF3800 re-fires the wake word every
        # ~1.5 s while somebody keeps talking, and each one used to start its
        # own rotation: the log for a single conversation shows 205° → turn
        # -155°, then 205° → turn -155° again 7 s later, because a turn that is
        # still in flight has not changed the measured DoA yet. Anything inside
        # this window belongs to the utterance we are already handling.
        self.declare_parameter('rotate_cooldown_s', 8.0)

        poll_hz               = self.get_parameter('poll_hz').value
        self._rotate_on_wake  = self.get_parameter('rotate_on_wake').value
        # Guarded: this is a divisor in _rotation_duration, and 0 would take
        # the node down with a ZeroDivisionError on the first turn.
        self._rotate_speed    = max(0.01, float(self.get_parameter('rotate_speed').value))
        self._min_angle_deg   = self.get_parameter('min_angle_deg').value
        self._led_enabled     = self.get_parameter('led_enabled').value
        self._led_brightness  = self.get_parameter('led_brightness').value
        self._led_color_idle_base    = self.get_parameter('led_color_idle_base').value
        self._led_color_idle_pointer = self.get_parameter('led_color_idle_pointer').value
        self._led_color_listening    = self.get_parameter('led_color_listening').value
        self._led_color_processing   = self.get_parameter('led_color_processing').value
        self._led_color_muted        = self.get_parameter('led_color_muted').value
        self._listen_timeout  = self.get_parameter('listen_timeout').value
        self._rotate_block_s  = self.get_parameter('rotate_block_s').value
        self._rotate_cooldown = self.get_parameter('rotate_cooldown_s').value

        # A launch parameter of True is an explicit "start with it on"; it must
        # not be silently downgraded by a stale file. Otherwise the remembered
        # web toggle wins, so the setting survives a restart.
        if not self._rotate_on_wake:
            self._rotate_on_wake = self._load_enabled()

        self._angle_pub   = self.create_publisher(Float32, 'doa/angle', 10)
        self._wake_pub    = self.create_publisher(Float32, 'doa/wake',  10)
        # ‼️ RELATIVE 'cmd_vel'; bringup remaps it to cmd_vel_safe when
        # obstacle_safety_node is not up to relay it. Left on the absolute
        # /cmd_vel — which has ZERO subscribers, measured live 2026-08-03 —
        # every "turn toward the speaker" went nowhere and the feature just
        # looked broken. Check with `ros2 topic info -v /cmd_vel_safe` before
        # changing this.
        self._cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Hardware voice-activity flag, straight off the XVF3800 DSP. It comes
        # back with every DoA read at no extra cost, and nothing was publishing
        # it: STT gates on an energy threshold instead, which is why a fan spike
        # can open a recording. Published so anything that wants a real VAD has
        # one to subscribe to.
        #
        # Latched, and only on transitions. This is STATE ("someone is
        # speaking"), not an event stream: publishing all 10 polls a second
        # would bury a bag in duplicates, but plain volatile QoS on top of that
        # means a node that starts later hears nothing until the next time
        # somebody happens to speak. transient_local gives it the last value on
        # connect. ‼️ From the CLI that means
        # `ros2 topic echo /voice_activity --qos-durability transient_local`;
        # a plain echo shows nothing and looks dead (same trap as
        # /emergency_stop).
        self._vad_pub = self.create_publisher(
            Bool, 'voice_activity',
            QoSProfile(depth=1,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE))

        # The web dashboard's switch, and the state it paints itself from.
        # Latched both ways: the dashboard is usually started (or reloaded)
        # long after this node, and a volatile publisher would leave its
        # checkbox guessing.
        _latched = QoSProfile(depth=1,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=QoSReliabilityPolicy.RELIABLE)
        self._rotate_state_pub = self.create_publisher(
            Bool, 'doa/rotate_state', _latched)
        self.create_subscription(
            Bool, 'doa/rotate_enable', self._on_rotate_enable, _latched)

        self.create_subscription(String, 'wake_word',   self._on_wake_word,   10)
        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)
        # ‼️ Nothing else in this node consulted the e-stop. The driver drops
        # the wheel writes, so the robot did stay still — but this node went on
        # publishing a turn for the whole sweep, and the moment the latch was
        # released the leftover twist was still in flight. An e-stopped robot
        # must have nobody asking it to move.
        self.create_subscription(
            Bool, '/emergency_stop', self._on_estop, _latched)
        # Commanded intent, upstream of the collision monitor — the same signal
        # recovery_manager watches. Non-zero here means a navigator (or teleop
        # via the smoother) owns the base right now.
        self.create_subscription(Twist, 'cmd_vel_smoothed', self._on_drive_cmd, 10)
        # wake_word_node's privacy mute, relayed here so the ring can show it —
        # same QoS it publishes with (see that node's muted_pub comment).
        self.create_subscription(
            Bool, 'mic/muted', self._on_mic_muted, _latched)

        self._dev        = None
        self._last_angle = 0.0
        self._last_speech = None      # so the first reading always publishes
        self._rotating   = False
        self._last_drive = 0.0        # monotonic time of the last non-zero drive cmd
        self._last_rotate = 0.0       # monotonic time of the last turn we started
        self._estop      = False
        self._muted       = False
        self._led_state  = self._STATE_IDLE
        self._listen_timer = None
        self._lock       = threading.Lock()

        self._rotate_state_pub.publish(Bool(data=bool(self._rotate_on_wake)))
        self.add_on_set_parameters_callback(self._on_param_change)

        threading.Thread(target=self._poll_loop, args=(poll_hz,), daemon=True).start()
        self.get_logger().info(
            'DoA node started — turn-toward-speaker is '
            + ('ON' if self._rotate_on_wake else 'OFF (enable it in the web dashboard)'))

    # ── Web toggle ─────────────────────────────────────────────────
    def _load_enabled(self) -> bool:
        try:
            with open(_STATE_PATH, encoding='utf-8') as fh:
                return bool(json.load(fh).get('rotate_on_wake', False))
        except (OSError, ValueError):
            return False

    def _save_enabled(self, enabled: bool):
        try:
            tmp = _STATE_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({'rotate_on_wake': bool(enabled)}, fh)
            os.replace(tmp, _STATE_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save DoA rotate state: {exc}')

    def _on_rotate_enable(self, msg: Bool):
        enabled = bool(msg.data)
        if enabled == self._rotate_on_wake:
            return
        self._rotate_on_wake = enabled
        self._save_enabled(enabled)
        self._rotate_state_pub.publish(Bool(data=enabled))
        self.get_logger().info(
            f'Turn toward speaker {"ENABLED" if enabled else "DISABLED"} from the dashboard')

    def _on_estop(self, msg: Bool):
        self._estop = bool(msg.data)

    def _on_mic_muted(self, msg: Bool):
        self._muted = bool(msg.data)
        self._apply_led_state()

    def _on_param_change(self, params):
        """Live retuning from the dashboard's mic-settings card — brightness
        and the per-state colours used to be read once in __init__ and never
        again. Re-applies immediately rather than waiting for the next state
        transition, so a colour picker feels connected to the hardware."""
        for p in params:
            if p.name == 'led_enabled':
                self._led_enabled = p.value
            elif p.name == 'led_brightness':
                self._led_brightness = max(0, min(255, int(p.value)))
            elif p.name == 'led_color_idle_base':
                self._led_color_idle_base = p.value
            elif p.name == 'led_color_idle_pointer':
                self._led_color_idle_pointer = p.value
            elif p.name == 'led_color_listening':
                self._led_color_listening = p.value
            elif p.name == 'led_color_processing':
                self._led_color_processing = p.value
            elif p.name == 'led_color_muted':
                self._led_color_muted = p.value
            else:
                continue
            self._apply_led_state()
        return SetParametersResult(successful=True)

    # ── USB polling ────────────────────────────────────────────────
    def _poll_loop(self, hz):
        interval = 1.0 / hz
        while rclpy.ok():
            if self._dev is None:
                self._dev = _find_device()
                if self._dev is None:
                    self.get_logger().warn('ReSpeaker not found, retrying...', throttle_duration_sec=10)
                    time.sleep(2)
                    continue
                self._apply_led_state()

            try:
                speech, angle = self._dev.read_doa()
                with self._lock:
                    self._last_angle = float(angle)
                if speech:
                    self._angle_pub.publish(Float32(data=float(angle)))
                # Edge-triggered: the flag is polled at 10 Hz but only the
                # transitions are interesting, and publishing 10 identical
                # messages a second would bury anything else in a bag.
                if speech != self._last_speech:
                    self._last_speech = speech
                    self._vad_pub.publish(Bool(data=speech))
            except Exception as e:
                self.get_logger().warn(f'DoA read error: {e}', throttle_duration_sec=5)
                self._dev = None

            time.sleep(interval)

    # ── LED control ────────────────────────────────────────────────
    def _apply_led_state(self):
        if not self._led_enabled or self._dev is None:
            return
        try:
            self._dev.set_led_brightness(self._led_brightness)
            # Mute overrides whichever of idle/listening/processing the voice
            # pipeline thinks it's in — the ring must never look like it's
            # still listening while the mic is actually off.
            if self._muted:
                self._dev.set_led_color(self._led_color_muted)
                self._dev.set_led_effect(LED_SINGLE)
            elif self._led_state == self._STATE_IDLE:
                self._dev.set_led_doa_colors(self._led_color_idle_base,
                                             self._led_color_idle_pointer)
                self._dev.set_led_effect(LED_DOA)
            elif self._led_state == self._STATE_LISTENING:
                self._dev.set_led_color(self._led_color_listening)
                self._dev.set_led_speed(10)
                self._dev.set_led_effect(LED_BREATH)
            elif self._led_state == self._STATE_PROCESSING:
                self._dev.set_led_color(self._led_color_processing)
                self._dev.set_led_speed(8)
                self._dev.set_led_effect(LED_BREATH)
        except Exception as e:
            self.get_logger().warn(f'LED error: {e}', throttle_duration_sec=5)

    def _set_led_state(self, state: str):
        with self._lock:
            self._led_state = state
        self._apply_led_state()

    # ── Wake word ──────────────────────────────────────────────────
    def _on_wake_word(self, _msg: String):
        with self._lock:
            angle = self._last_angle
        self._wake_pub.publish(Float32(data=angle))
        self.get_logger().info(f'Wake word DoA: {angle:.0f}°')

        self._set_led_state(self._STATE_LISTENING)

        # Cancel previous timeout if any
        if self._listen_timer is not None:
            self._listen_timer.cancel()
        self._listen_timer = threading.Timer(
            self._listen_timeout, self._on_listen_timeout)
        self._listen_timer.daemon = True
        self._listen_timer.start()

        if not self._rotate_on_wake:
            return
        if self._rotating or self._base_is_busy():
            return
        if self._estop:
            self.get_logger().info('E-stop latched — not turning toward the speaker')
            return
        if (time.monotonic() - self._last_rotate) < self._rotate_cooldown:
            self.get_logger().info(
                'Already turned for this speaker — ignoring the repeat wake word')
            return
        self._last_rotate = time.monotonic()
        threading.Thread(target=self._rotate_toward, args=(angle,), daemon=True).start()

    def _on_drive_cmd(self, msg: Twist):
        if (abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)) > 0.01:
            self._last_drive = time.monotonic()

    def _base_is_busy(self) -> bool:
        """True while someone else is driving, so we keep off cmd_vel."""
        busy = (time.monotonic() - self._last_drive) < self._rotate_block_s
        if busy:
            self.get_logger().info(
                'Wake word while navigating — not turning toward the speaker')
        return busy

    def _on_listen_timeout(self):
        self.get_logger().info('Listen timeout — returning to idle')
        self._set_led_state(self._STATE_IDLE)

    # ── Speech text received (STT done) ───────────────────────────
    def _on_speech_text(self, _msg: String):
        if self._listen_timer is not None:
            self._listen_timer.cancel()
            self._listen_timer = None
        self._set_led_state(self._STATE_PROCESSING)
        # Brief processing flash, then back to idle
        threading.Timer(3.0, lambda: self._set_led_state(self._STATE_IDLE)).start()

    # ── Rotation ───────────────────────────────────────────────────
    def _rotate_toward(self, angle_deg: float):
        if angle_deg > 180:
            angle_deg -= 360
        angle_rad = math.radians(angle_deg)

        if abs(angle_rad) < math.radians(self._min_angle_deg):
            return

        duration  = abs(angle_rad) / self._rotate_speed
        direction = -1.0 if angle_rad > 0 else 1.0

        self.get_logger().info(
            f'Rotating {math.degrees(angle_rad):.0f}° toward speaker '
            f'({duration:.1f}s @ {self._rotate_speed} rad/s)')

        self._rotating = True
        twist = Twist()
        twist.angular.z = direction * self._rotate_speed
        try:
            t_end = time.monotonic() + duration
            while time.monotonic() < t_end and rclpy.ok():
                # A goal can start after we did; yield the base the moment a
                # navigator asks for it rather than fighting it to the end of
                # the sweep. (Our own turn goes out on cmd_vel, not
                # cmd_vel_smoothed, so this never self-triggers.)
                if (time.monotonic() - self._last_drive) < self._rotate_block_s:
                    self.get_logger().info(
                        'Navigation took the base — abandoning the turn')
                    break
                if self._estop or not self._rotate_on_wake:
                    self.get_logger().info('Turn cancelled mid-sweep')
                    break
                self._cmd_vel_pub.publish(twist)
                time.sleep(0.05)
        finally:
            self._cmd_vel_pub.publish(Twist())
            self._rotating = False


def main():
    rclpy.init()
    node = DoaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._dev:
            node._dev.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
