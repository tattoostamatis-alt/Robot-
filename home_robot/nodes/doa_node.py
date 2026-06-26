#!/usr/bin/env python3
"""Direction of Arrival + LED control — ReSpeaker XVF3800.

Reads hardware DoA from the XVF3800 DSP and controls the onboard LED ring
to give visual feedback about the robot's listening state.

Topics published:
  /doa/angle  (Float32) — angle [0-359°] when speech detected
  /doa/wake   (Float32) — angle sampled at wake_word moment

Topics subscribed:
  /wake_word   (String)  — triggers LISTENING LED state + rotate toward speaker
  /speech_text (String)  — transitions LED back to IDLE

LED states:
  IDLE       — DoA mode (mode=4): 1 LED points toward detected speaker
  LISTENING  — breath blue: robot is recording the command
  PROCESSING — breath orange: Whisper is transcribing
  IDLE again — DoA mode resumes after speech_text or timeout

DoA convention (XVF3800): 0° = front of device, increases clockwise.
ROS angular.z positive = counter-clockwise — rotation is negated accordingly.
"""

import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
import usb.core
import usb.util


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
        self.declare_parameter('rotate_on_wake',    True)
        self.declare_parameter('rotate_speed',      0.6)
        self.declare_parameter('min_angle_deg',     20.0)
        self.declare_parameter('led_enabled',       True)
        self.declare_parameter('led_brightness',    150)
        self.declare_parameter('listen_timeout',    12.0)  # max STT window seconds

        poll_hz               = self.get_parameter('poll_hz').value
        self._rotate_on_wake  = self.get_parameter('rotate_on_wake').value
        self._rotate_speed    = self.get_parameter('rotate_speed').value
        self._min_angle_deg   = self.get_parameter('min_angle_deg').value
        self._led_enabled     = self.get_parameter('led_enabled').value
        self._led_brightness  = self.get_parameter('led_brightness').value
        self._listen_timeout  = self.get_parameter('listen_timeout').value

        self._angle_pub   = self.create_publisher(Float32, 'doa/angle', 10)
        self._wake_pub    = self.create_publisher(Float32, 'doa/wake',  10)
        self._cmd_vel_pub = self.create_publisher(Twist,  'cmd_vel',   10)

        self.create_subscription(String, 'wake_word',   self._on_wake_word,   10)
        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)

        self._dev        = None
        self._last_angle = 0.0
        self._rotating   = False
        self._led_state  = self._STATE_IDLE
        self._listen_timer = None
        self._lock       = threading.Lock()

        threading.Thread(target=self._poll_loop, args=(poll_hz,), daemon=True).start()
        self.get_logger().info('DoA node started')

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
            if self._led_state == self._STATE_IDLE:
                self._dev.set_led_doa_colors(0x111111, 0x0000FF)
                self._dev.set_led_effect(LED_DOA)
            elif self._led_state == self._STATE_LISTENING:
                self._dev.set_led_color(COLOR_BLUE)
                self._dev.set_led_speed(10)
                self._dev.set_led_effect(LED_BREATH)
            elif self._led_state == self._STATE_PROCESSING:
                self._dev.set_led_color(COLOR_ORANGE)
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

        if self._rotate_on_wake and not self._rotating:
            threading.Thread(target=self._rotate_toward, args=(angle,), daemon=True).start()

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
