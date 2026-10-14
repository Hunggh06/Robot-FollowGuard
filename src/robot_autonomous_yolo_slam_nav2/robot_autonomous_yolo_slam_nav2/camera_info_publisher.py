"""Publish measured CameraInfo values alongside phone camera images."""

from __future__ import annotations

import sys
from typing import Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .camera_calibration import CalibrationError
from .camera_info_calibration import CameraCalibration, load_camera_calibration


class CameraInfoPublisherNode(Node):
    """Republish validated calibration with each image timestamp."""

    def __init__(self) -> None:
        super().__init__("phone_camera_info")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("image_topic", "/phone/camera/back/image")
        self.declare_parameter("camera_info_topic", "/phone/camera/back/camera_info")
        self.declare_parameter("frame_id", "phone_camera_back")
        self._calibration: CameraCalibration | None = None
        self._camera_info_publisher = None
        self._image_subscription = None
        self._frame_id = str(self.get_parameter("frame_id").value)
        calibration_file = str(self.get_parameter("calibration_file").value)
        if not calibration_file:
            self.get_logger().error("calibration_file is empty; CameraInfo publisher is disabled")
            return
        try:
            self._calibration = load_camera_calibration(calibration_file)
        except CalibrationError as error:
            self.get_logger().error(f"CameraInfo publisher disabled: {error}")
            return
        info_topic = str(self.get_parameter("camera_info_topic").value)
        image_topic = str(self.get_parameter("image_topic").value)
        self._camera_info_publisher = self.create_publisher(
            CameraInfo,
            info_topic,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._size_warned = False

    def _image_callback(self, image: Image) -> None:
        calibration = self._calibration
        publisher = self._camera_info_publisher
        if calibration is None or publisher is None:
            return
        if (image.width > 0 and image.height > 0
                and image.width * 3 == image.height * 4):
            width = int(image.width)
            height = int(image.height)
            scale_x = width / calibration.image_width
            scale_y = height / calibration.image_height
            k = list(calibration.camera_matrix)
            p = list(calibration.projection_matrix)
            k[0] *= scale_x
            k[2] *= scale_x
            k[4] *= scale_y
            k[5] *= scale_y
            p[0] *= scale_x
            p[2] *= scale_x
            p[5] *= scale_y
            p[6] *= scale_y
        else:
            if not getattr(self, "_size_warned", False):
                try:
                    self.get_logger().warning(
                        f"image {image.width}x{image.height} != "
                        f"calibration {calibration.image_width}x{calibration.image_height}; "
                        "CameraInfo van publish voi stamp copy, can calibrate lai"
                    )
                except Exception:
                    pass
                self._size_warned = True
            width = calibration.image_width
            height = calibration.image_height
            k = list(calibration.camera_matrix)
            p = list(calibration.projection_matrix)
        try:
            frame_id = getattr(image.header, "frame_id", "")
        except Exception:
            frame_id = ""
        if frame_id and frame_id != self._frame_id:
            try:
                self.get_logger().warning(
                    f"image frame '{frame_id}' != '{self._frame_id}'",
                    throttle_duration_sec=5.0,
                )
            except Exception:
                pass
        message = CameraInfo()
        # Phone clock is not synced with host clock (measured ~1.3 s offset
        # over WiFi); stamp receipt time so downstream sync compares one clock.
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.width = width
        message.height = height
        message.distortion_model = calibration.distortion_model
        message.d = list(calibration.distortion_coefficients)
        message.k = k
        message.r = list(calibration.rectification_matrix)
        message.p = p
        message.binning_x = calibration.binning_x
        message.binning_y = calibration.binning_y
        message.roi.x_offset = int(calibration.roi.get("x_offset", 0))
        message.roi.y_offset = int(calibration.roi.get("y_offset", 0))
        message.roi.height = int(calibration.roi.get("height", 0))
        message.roi.width = int(calibration.roi.get("width", 0))
        message.roi.do_rectify = bool(calibration.roi.get("do_rectify", False))
        publisher.publish(message)


def main(args: Sequence[str] | None = None) -> int:
    rclpy.init(args=list(args) if args is not None else None)
    node = CameraInfoPublisherNode()
    try:
        if node._image_subscription is None:
            return 1
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
