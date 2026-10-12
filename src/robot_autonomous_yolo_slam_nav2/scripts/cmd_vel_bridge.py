#!/usr/bin/env python3
# ============================================================
# BRIDGE: /cmd_vel (Twist) -> /motor/target_rpm (Float32MultiArray [L, R])
# ============================================================
# Nav2 / tay cam phat lenh tren /cmd_vel kieu Twist (m/s, rad/s).
# Firmware ESP32 nghe /motor/target_rpm kieu Float32MultiArray [left, right]
# (RPM). Node nay doi chieu + gioi han toc do + ramp mem + timeout ve 0.
#
# Cong thuc:
#   vL = v - w * WHEEL_BASE / 2
#   vR = v + w * WHEEL_BASE / 2
#   rpm = v_banh / (PI * DUONG_KINH_BANH) * 60
#
# MAX_RPM = 90: cho robot chay ~0,31 m/s.
# tuong vao tuong — giu o 60 cho an toan, doi muc nay o day khi can.
# ============================================================

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

WHEEL_BASE = 0.2385          # m, do bang thuoc
WHEEL_DIAMETER = 0.065       # m
CIRC = math.pi * WHEEL_DIAMETER

MAX_RPM = 126.0
ACCEL_LIM = 60.0
KICK_START_RPM = 25.0        # vuot ma sat tinh khi banh dang dung yen
KEEP_ROLLING_RPM = 25.0
CMD_TIMEOUT = 0.5            # kieu twist_mux timeout: qua han khong co
                             # /cmd_vel thi ve RPM 0, khong giu ga cu


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/motor/target_rpm')
        self.cmd = self.create_publisher(Float32MultiArray,
                                           str(self.get_parameter('output_topic').value), 10)
        self.create_subscription(
            Twist,
            str(self.get_parameter('input_topic').value),
            self.cb,
            10,
        )
        self.rpm_l = 0.0
        self.rpm_r = 0.0
        self.target_l = 0.0
        self.target_r = 0.0
        self.last_cmd = None
        self.last = None
        self.create_timer(0.05, self.ramp)   # 20 Hz, khop PID 10 Hz cua ESP32

    def cb(self, msg):
        v = msg.linear.x
        w = msg.angular.z
        self.last_cmd = self.get_clock().now()

        half = WHEEL_BASE / 2.0
        v_l = v - w * half
        v_r = v + w * half

        self.target_l = max(-MAX_RPM, min(MAX_RPM, v_l / CIRC * 60.0))
        self.target_r = max(-MAX_RPM, min(MAX_RPM, v_r / CIRC * 60.0))

    @staticmethod
    def _rolling_floor(current_rpm, target_rpm):
        if target_rpm == 0.0:
            return 0.0
        floor = KICK_START_RPM if current_rpm == 0.0 else KEEP_ROLLING_RPM
        if 0.0 < target_rpm < floor:
            return floor
        if -floor < target_rpm < 0.0:
            return -floor
        return target_rpm

    def ramp(self):
        now = self.get_clock().now()
        dt = 0.05 if self.last is None else (now - self.last).nanoseconds * 1e-9
        self.last = now
        step = ACCEL_LIM * max(0.0, min(dt, 0.2))

        # Het han lenh -> ve 0, khong giu ga cu dam tuong
        if self.last_cmd is None or \
                (now - self.last_cmd).nanoseconds * 1e-9 > CMD_TIMEOUT:
            self.target_l = 0.0
            self.target_r = 0.0

        self.rpm_l = self._approach(
            self.rpm_l, self._rolling_floor(self.rpm_l, self.target_l), step)
        self.rpm_r = self._approach(
            self.rpm_r, self._rolling_floor(self.rpm_r, self.target_r), step)

        m = Float32MultiArray()
        m.data = [self.rpm_l, self.rpm_r]
        self.cmd.publish(m)

    @staticmethod
    def _approach(cur, tgt, step):
        if cur < tgt:
            return min(tgt, cur + step)
        if cur > tgt:
            return max(tgt, cur - step)
        return cur


def main():
    rclpy.init()
    node = CmdVelBridge()
    node.target_l = 0.0
    node.target_r = 0.0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
