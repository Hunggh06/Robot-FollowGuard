#!/usr/bin/env python3
"""Lay yaw-rate tu IMU dien thoai: Gx -> /imu/yaw (frame base_link).

Dien thoai lap landscape, camera nhin truoc: truc Gx trung truc yaw
robot (test xoay trai 90 do: Gx duong). EKF chi can van toc yaw nen
node nay trich Gx sang angular_velocity.z, bo orientation va gia toc
tuyen tinh (magnet trong nha nhieu, khong tin duoc). Day la phep
trich yaw-only cho gia lap co dinh, khong phai bien doi day du he truc.
"""
import math
from typing import Final

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


FALLBACK_YAW_VARIANCE: Final = 0.05
UNAVAILABLE_AXIS_VARIANCE: Final = 1e6


def remap_imu(message: Imu, bias_x: float) -> Imu:
    """Extract phone X gyro as base_link yaw with explicit partial covariance."""
    source_x_variance = message.angular_velocity_covariance[0]
    yaw_variance = (
        source_x_variance
        if math.isfinite(source_x_variance) and source_x_variance > 0.0
        else FALLBACK_YAW_VARIANCE
    )

    result = Imu()
    result.header.stamp = message.header.stamp
    result.header.frame_id = 'base_link'
    result.angular_velocity.z = message.angular_velocity.x - bias_x
    result.angular_velocity_covariance[0] = (
        -1.0
        if source_x_variance == -1.0
        else UNAVAILABLE_AXIS_VARIANCE
    )
    result.angular_velocity_covariance[4] = UNAVAILABLE_AXIS_VARIANCE
    result.angular_velocity_covariance[8] = yaw_variance
    result.orientation_covariance[0] = -1.0
    result.linear_acceleration_covariance[0] = -1.0
    return result


class ImuRemap(Node):
    def __init__(self):
        super().__init__('imu_remap')
        self.declare_parameter('gyro_bias_x', 0.0)
        self.bias = float(self.get_parameter('gyro_bias_x').value or 0.0)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Imu, '/phone/imu', self.cb, qos)
        self.pub = self.create_publisher(Imu, '/imu/yaw', 10)

    def cb(self, msg):
        self.pub.publish(remap_imu(msg, self.bias))


def main():
    rclpy.init()
    node = ImuRemap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
