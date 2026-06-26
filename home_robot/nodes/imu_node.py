#!/usr/bin/env python3
"""ESP32 + BNO085 IMU -> sensor_msgs/Imu.

Reads the "IMU,qw,qi,qj,qk,gx,gy,gz,ax,ay,az" lines streamed by the
bno085_imu.ino firmware and republishes as sensor_msgs/Imu on imu/data.

Uses raw termios instead of pyserial — the CH340 on this ESP32 hangs on
TIOCMBIS/TIOCMBIC ioctls after a USB hub reset, causing [Errno 110].
Raw os.open + termios bypasses those ioctls entirely.
"""

import fcntl
import io
import os
import termios
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

_BAUD_MAP = {
    9600:   termios.B9600,
    19200:  termios.B19200,
    38400:  termios.B38400,
    57600:  termios.B57600,
    115200: termios.B115200,
}


def _open_raw_serial(port: str, baud: int) -> io.FileIO:
    """Open a serial port without any modem-status ioctls."""
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    speed = _BAUD_MAP[baud]
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0                                        # iflag: no input processing
    attrs[1] = 0                                        # oflag: no output processing
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag: 8N1, no modem ctrl
    attrs[3] = 0                                        # lflag: raw
    attrs[4] = speed                                    # ispeed
    attrs[5] = speed                                    # ospeed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # Switch to blocking I/O
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    return io.FileIO(fd, mode='rb', closefd=True)


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.declare_parameter('port', '/dev/imu')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('frame_id', 'imu_link')

        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value
        self.frame_id = self.get_parameter('frame_id').value

        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)

        self._fio: io.FileIO | None = None
        self._open_serial()

        self._stop = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_serial(self):
        if self._fio is not None:
            try:
                self._fio.close()
            except Exception:
                pass
        raw = _open_raw_serial(self.port, self.baud)
        self._fio = raw
        self._reader = io.BufferedReader(raw, buffer_size=4096)
        self.get_logger().info(f'Opened IMU serial on {self.port}')

    def _readline(self) -> bytes:
        """Read one \n-terminated line with a 1-second timeout via select."""
        import select
        buf = b''
        deadline = time.monotonic() + 1.0
        fd = self._fio.fileno()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return buf  # timeout — return whatever we have
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return buf
            chunk = os.read(fd, 256)
            if not chunk:
                raise OSError('EOF on serial port')
            buf += chunk
            if b'\n' in buf:
                line, _ = buf.split(b'\n', 1)
                return line + b'\n'

    def _read_loop(self):
        while not self._stop:
            try:
                line = self._readline()
            except OSError as e:
                self.get_logger().warn(f'Serial error: {e!r}, reopening in 1s')
                self._reopen_after_delay()
                continue

            if not line.startswith(b'IMU,'):
                continue

            fields = line.decode(errors='replace').strip().split(',')
            if len(fields) != 11:
                continue

            try:
                qw, qi, qj, qk, gx, gy, gz, ax, ay, az = (float(f) for f in fields[1:])
            except ValueError:
                continue

            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.orientation.w = qw
            msg.orientation.x = qi
            msg.orientation.y = qj
            msg.orientation.z = qk
            msg.orientation_covariance = [
                0.01, 0.0,  0.0,
                0.0,  0.01, 0.0,
                0.0,  0.0,  0.02,
            ]

            msg.angular_velocity.x = gx
            msg.angular_velocity.y = gy
            msg.angular_velocity.z = gz
            msg.angular_velocity_covariance = [
                0.001, 0.0,   0.0,
                0.0,   0.001, 0.0,
                0.0,   0.0,   0.001,
            ]

            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az
            msg.linear_acceleration_covariance = [
                0.04, 0.0,  0.0,
                0.0,  0.04, 0.0,
                0.0,  0.0,  0.04,
            ]

            self.imu_pub.publish(msg)

    def _reopen_after_delay(self):
        time.sleep(1.0)
        try:
            self._open_serial()
        except Exception as e:
            self.get_logger().error(f'Reopen failed: {e!r}')

    def destroy_node(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        if self._fio is not None:
            try:
                self._fio.close()
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
