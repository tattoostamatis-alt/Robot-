#!/usr/bin/env python3
"""ESP32 + BNO085 IMU -> sensor_msgs/Imu.

Reads the "IMU,qw,qi,qj,qk,gx,gy,gz,ax,ay,az" lines streamed by the
bno085_imu.ino firmware (absolute orientation quaternion from the
BNO085's onboard sensor fusion + calibrated gyro + linear acceleration,
~100Hz) and republishes them as sensor_msgs/Imu on imu/data, frame_id
imu_link, for robot_localization's ekf_node to fuse with wheel odometry.
"""

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial


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

        self.ser = None
        self._open_serial()

        self._stop = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_serial(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        # Asserting DTR while opening triggers the CH340's auto-reset
        # (DTR wired to the ESP32 EN pin) — and resetting the ESP32 here
        # re-runs its IMU init, which can hang and need a physical USB
        # power-cycle to recover. Setting dtr=False before open() avoids
        # the reset pulse entirely, so this just attaches to the
        # already-running/streaming firmware instead.
        self.ser = serial.Serial()
        self.ser.port = self.port
        self.ser.baudrate = self.baud
        self.ser.timeout = 1.0
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.get_logger().info(f'Opened IMU serial on {self.port}')

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline()
            except (serial.SerialException, OSError) as e:
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
            # Full 3-axis absolute orientation from the BNO085's onboard
            # sensor fusion (accel+gyro+magnetometer), not just yaw. Fixed
            # diagonal covariance is good enough for EKF fusion here.
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
        import time
        time.sleep(1.0)
        try:
            self._open_serial()
        except Exception as e:
            self.get_logger().error(f'Reopen failed: {e!r}')

    def destroy_node(self):
        self._stop = True
        # Join before closing the port — the read thread's blocking
        # readline() (timeout=1.0) needs to return on its own first, or
        # closing self.ser out from under it raises a TypeError inside
        # pyserial's os.read() (fd becomes None mid-call).
        self._thread.join(timeout=2.0)
        if self.ser is not None:
            try:
                self.ser.close()
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
