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
# MAX_RPM = 126: cho robot chay ~0,43 m/s (126/60 * PI * 0.065).
# Doi muc nay o day khi can giam toc do toi da.
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
# He so hieu chuan toc do firmware (1.0 = tin firmware 1:1).
OUTPUT_SCALE = 1.0
KICK_START_RPM = 25.0        # vuot ma sat tinh khi banh dang dung yen
KEEP_ROLLING_RPM = 25.0
CMD_TIMEOUT = 0.5            # kieu twist_mux timeout: qua han khong co
                             # /cmd_vel thi ve RPM 0, khong giu ga cu


def resolve_bridge_timeout(
    target_l: float,
    target_r: float,
    rpm_l: float,
    rpm_r: float,
    expired: bool,
) -> tuple[float, float, float, float]:
    """Fail-safe resolver: expired commands hard-zero, live ones pass through.

    On timeout every stale queued/ramped value is discarded so motion cannot
    revive from ramp memory; only a fresh cb() may command motion again.
    """
    if expired:
        return (0.0, 0.0, 0.0, 0.0)
    return (target_l, target_r, rpm_l, rpm_r)


def twist_to_targets(v: float, w: float) -> tuple[float, float]:
    """Convert Twist (v, w) to clamped wheel RPM targets.

    Non-finite inputs hard-zero: min/max clamping lets NaN through to full
    RPM, so NaN/Inf must never reach the motor command path.
    """
    if not math.isfinite(v) or not math.isfinite(w):
        return (0.0, 0.0)
    half = WHEEL_BASE / 2.0
    v_l = v - w * half
    v_r = v + w * half
    target_l = max(-MAX_RPM, min(MAX_RPM, v_l / CIRC * 60.0)) * OUTPUT_SCALE
    target_r = max(-MAX_RPM, min(MAX_RPM, v_r / CIRC * 60.0)) * OUTPUT_SCALE
    return (target_l, target_r)


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
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        self.last_cmd = self.get_clock().now()

        # Non-finite Twist hard-zeroes targets; link stays alive but commands
        # nothing until the next finite message.
        self.target_l, self.target_r = twist_to_targets(v, w)

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

        # Het han lenh -> hard-zero 0, khong giu ga cu dam tuong.
        # Hard-zero (khong ramp giam dan): bridge dung gui RPM != 0 nen
        # firmware timeout duoc kich hoat thay vi bi reset lien tuc, va lenh
        # cu/queued khong the lam song lai chuyen dong.
        expired = self.last_cmd is None or \
            (now - self.last_cmd).nanoseconds * 1e-9 > CMD_TIMEOUT
        if expired:
            self.target_l, self.target_r, self.rpm_l, self.rpm_r = \
                resolve_bridge_timeout(
                    self.target_l, self.target_r, self.rpm_l, self.rpm_r, True
                )
            m = Float32MultiArray()
            m.data = [0.0, 0.0]
            self.cmd.publish(m)
            return

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
