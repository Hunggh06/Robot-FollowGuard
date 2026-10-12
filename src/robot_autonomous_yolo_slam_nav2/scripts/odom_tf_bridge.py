#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBridge(Node):
    # Loc tia gan: giu vat cach lidar >= MIN_RANGE, bo vong nhieu sat cam.
    MIN_RANGE = 0.25
    # Scan mat ~1 vong quet + WiFi moi toi host: noi dung cu hon stamp
    # neu dong dau "bay gio" thi TF tra sai thoi diem, cang chay cang
    # nhoe. Tru hao latency uoc luong.
    SCAN_LATENCY = 0.15
    def __init__(self):
        super().__init__('odom_tf_bridge')
        self.declare_parameter('odom_topic', '/wheel/odom')
        odom_topic = str(self.get_parameter('odom_topic').value
                         or '/wheel/odom')
        self.tf = TransformBroadcaster(self)
        self.last_pose = None
        self.pose = [0.0, 0.0]
        self.vel = [0.0, 0.0]
        self.quat = Quaternion()
        self.quat.w = 1.0
        self.declare_parameter('relay_scan', False)
        self.relay_scan = bool(self.get_parameter('relay_scan').value)
        self.scan_pub = None
        if self.relay_scan:
            self.scan_pub = self.create_publisher(LaserScan, '/scan_host', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        if self.relay_scan:
            self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_timer(0.02, self._tf_fresh_timer)

    def _tf_fresh_timer(self):
        if self.last_pose is None:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_odom(stamp)
        self._publish_tf(stamp, self.pose[0], self.pose[1], 0.0, self.quat)

    def _publish_tf(self, stamp, x, y, z, q):
        t = TransformStamped()
        t.header.frame_id = 'odom'
        t.header.stamp = stamp
        # Ground projection (khong doi thanh base_link: banh se chui duoi luoi RViz).
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation = q
        self.tf.sendTransform(t)

    def _publish_odom(self, stamp):
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = 'odom'
        o.child_frame_id = 'base_footprint'
        o.pose.pose.position.x = self.pose[0]
        o.pose.pose.position.y = self.pose[1]
        o.pose.pose.orientation = self.quat
        o.twist.twist.linear.x = self.vel[0]
        o.twist.twist.angular.z = self.vel[1]
        self.odom_pub.publish(o)

    def odom_cb(self, msg):
        # Dung gia tri vi tri cua ESP32. Tich phan van toc o host se lech dan
        # (odo ESP32 sai ~16% scale) va khi khoi dong lai thi "resync" gay
        # nhay vi tri vai met giua 2 frame -> robot nhung trong RViz.
        # Giu nguyen odo ESP32 chi doi timestamp sang host clock.
        p = msg.pose.pose
        self.pose[0] = p.position.x
        self.pose[1] = p.position.y
        t = msg.twist.twist
        self.vel[0] = t.linear.x
        self.vel[1] = t.angular.z
        self.quat = p.orientation
        self.last_pose = p

    @staticmethod
    def _yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * q.z * q.z)

    def scan_cb(self, msg):
        if self.scan_pub is None:
            return
        latency_ns = int(self.SCAN_LATENCY * 1e9)
        msg.header.stamp = (self.get_clock().now() -
                            Duration(nanoseconds=latency_ns)).to_msg()
        n = len(msg.ranges)
        if n > 1:
            msg.angle_increment = (msg.angle_max - msg.angle_min) / (n - 1)
        for i, r in enumerate(msg.ranges):
            if r < self.MIN_RANGE:
                msg.ranges[i] = math.inf
        self.scan_pub.publish(msg)


def main():
    rclpy.init()
    node = OdomTfBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
