#!/usr/bin/env python3
"""Encoder odometry that: /motor/state -> /wheel/odom.

ESP32 (firmware lds02rr_wifi + motor) chi pub /motor/state 10Hz:
  [targetL, targetR, rpmL, rpmR, encL, encR, pwmL, pwmR]
Khong co odom. Node nay tich phan xung encoder thanh pose diff-drive
cho odom_tf_bridge + slam_toolbox dung (thay fake odom = 0).

Thong so khop firmware + URDF: TPR 227, banh D=65mm, co so 0.2385m.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class MotorOdom(Node):
    def __init__(self):
        super().__init__('motor_odom')
        self.declare_parameter('wheel_diameter', 0.065)
        self.declare_parameter('ticks_per_rev', 227.0)
        self.declare_parameter('wheel_base', 0.2385)
        self.declare_parameter('publish_rate', 20.0)

        self.diameter = float(self.get_parameter('wheel_diameter').value or 0.065)
        self.tpr = float(self.get_parameter('ticks_per_rev').value or 227.0)
        self.base = float(self.get_parameter('wheel_base').value or 0.2385)
        rate = float(self.get_parameter('publish_rate').value or 20.0)

        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.prev_enc = None
        self.rpm = (0.0, 0.0)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Float32MultiArray, '/motor/state',
                                 self.state_cb, qos)
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odom', 10)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info('motor_odom ready (real encoder odom)')

    def state_cb(self, msg):
        if len(msg.data) < 6:
            return
        self.rpm = (msg.data[2], msg.data[3])
        enc = (msg.data[4], msg.data[5])
        if self.prev_enc is None:
            self.prev_enc = enc
            return
        d_ticks = (enc[0] - self.prev_enc[0], enc[1] - self.prev_enc[1])
        self.prev_enc = enc
        # ESP32 reboot -> xung ve 0 (delta am lon); WiFi rot goi cung gay
        # nhay dot ngot. Qua 2000 xung/mau (>2.5s chay max) thi resync,
        # bo mau nay thay vi cong don rac vao pose.
        if abs(d_ticks[0]) > 2000 or abs(d_ticks[1]) > 2000:
            return
        m_per_tick = math.pi * self.diameter / self.tpr
        d_l = d_ticks[0] * m_per_tick
        d_r = d_ticks[1] * m_per_tick
        d_c = (d_l + d_r) * 0.5
        d_th = (d_r - d_l) / self.base
        th_mid = self.th + d_th * 0.5
        self.x += d_c * math.cos(th_mid)
        self.y += d_c * math.sin(th_mid)
        self.th += d_th

    def _publish(self):
        if self.prev_enc is None:
            return
        rps_l = self.rpm[0] / 60.0
        rps_r = self.rpm[1] / 60.0
        v = (rps_l + rps_r) * 0.5 * math.pi * self.diameter
        w = (rps_r - rps_l) * math.pi * self.diameter / self.base
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id = 'odom'
        o.child_frame_id = 'base_footprint'
        o.pose.pose.position.x = self.x
        o.pose.pose.position.y = self.y
        o.pose.pose.orientation.z = math.sin(self.th * 0.5)
        o.pose.pose.orientation.w = math.cos(self.th * 0.5)
        o.pose.covariance[0] = 0.05
        o.pose.covariance[7] = 0.05
        o.pose.covariance[35] = 0.1
        o.twist.twist.linear.x = v
        o.twist.twist.angular.z = w
        o.twist.covariance[0] = 0.1
        o.twist.covariance[35] = 0.2
        self.odom_pub.publish(o)


def main():
    rclpy.init()
    node = MotorOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
