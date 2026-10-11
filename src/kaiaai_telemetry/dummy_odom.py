#!/usr/bin/env python3
# ============================================================
# ODOM GIA — dong vai static transform nhung co timestamp that
# ============================================================
#
# VAN DE:
#   static_transform_publisher phat odom -> base_footprint mot lan, co
#   timestamp 0. Robot dung yen -> khong co transform moi nao. /scan den
#   voi gio hien tai, nhung transform cache cua slam_toolbox chi co du
#   lieu tu luc khoi dong, nen no bao:
#
#     "the timestamp on the message is earlier than all the data in the
#      transform cache"
#
#   va loai bo TAT CA message -> /map khong bao gi.
#
# CACH SUA:
#   Phat odom -> base_footprint lien tuc voi timestamp = hien tai, gia tri
#   danh tinh (robot chua co odom chinh xac). Nhieu transform dung thoi
#   gian -> cache luon co du lieu moi, moi scan deu khop.
#
#   Day la GIAI PHAP TAM THOI cho khi robot dung lai. Khi TPR + WHEEL_BASE
#   chuan xong, thay node nay bang /odom that tu node kaiaai_telemetry.
#
# Dung:
#   ros2 run kaiaai_telemetry dummy_odom
# ============================================================

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class DummyOdom(Node):
    def __init__(self):
        super().__init__("dummy_odom")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.02, self.publish)

    def publish(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = DummyOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
