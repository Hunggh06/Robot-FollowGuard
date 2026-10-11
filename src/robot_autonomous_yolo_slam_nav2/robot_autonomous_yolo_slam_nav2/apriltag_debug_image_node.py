"""Publish an annotated camera image for AprilTag inspection in rqt."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point


class AprilTagDebugImageNode(Node):
    """Detect tag36h11 markers and publish an annotated camera image."""

    def __init__(self) -> None:
        super().__init__("apriltag_debug_image")
        self.declare_parameter("image_topic", "/phone/camera/back/image")
        self.declare_parameter("output_topic", "/apriltag_debug/annotated")
        self.declare_parameter("publish_annotated_image", True)
        self.declare_parameter("detection_topic", "/apriltag/detections")
        self.declare_parameter("family", "36h11")
        self.declare_parameter("target_id", 0)

        image_topic = str(self.get_parameter("image_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        detection_topic = str(self.get_parameter("detection_topic").value)
        self._family = str(self.get_parameter("family").value)
        self._target_id = int(str(self.get_parameter("target_id").value))
        self._bridge = CvBridge()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(
            dictionary,
            cv2.aruco.DetectorParameters(),
        )
        self._publisher = (
            self.create_publisher(Image, output_topic, 5)
            if self.get_parameter("publish_annotated_image").value else None
        )
        self._detection_publisher = self.create_publisher(
            AprilTagDetectionArray,
            detection_topic,
            5,
        )
        self._subscription = self.create_subscription(
            Image,
            image_topic,
            self._callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Detecting tag{self._family} ID {self._target_id}; "
            f"detections={detection_topic}; "
            f"annotated={output_topic if self._publisher is not None else 'disabled'}"
        )

    def _callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"camera conversion failed: {error}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        annotated = frame.copy() if self._publisher is not None else None
        target_seen = False
        tag_count = 0
        detection_array = AprilTagDetectionArray()
        detection_array.header = message.header
        detections: list[AprilTagDetection] = []
        if ids is not None:
            for marker_corners, raw_id in zip(corners, ids.reshape(-1).tolist()):
                points = np.asarray(marker_corners, dtype=np.int32).reshape(4, 2)
                tag_id = int(raw_id)
                is_target = tag_id == self._target_id
                target_seen = target_seen or is_target
                tag_count += 1
                color = (0, 255, 0) if is_target else (255, 0, 0)
                if annotated is not None:
                    cv2.polylines(annotated, [points], True, color, 2)
                center = tuple(np.mean(points, axis=0).astype(int))
                detection = AprilTagDetection()
                detection.family = f"tag{self._family}"
                detection.id = tag_id
                detection.hamming = 0
                detection.goodness = 1.0
                detection.decision_margin = 1.0
                detection.centre = Point(x=float(center[0]), y=float(center[1]))
                detection.corners = [
                    Point(x=float(point[0]), y=float(point[1])) for point in points
                ]
                detections.append(detection)
                if annotated is not None:
                    cv2.putText(
                        annotated,
                        f"tag {tag_id}",
                        (int(center[0]) - 30, int(center[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

        detection_array.detections = detections
        self._detection_publisher.publish(detection_array)
        if annotated is None or self._publisher is None:
            return
        status = "FOUND" if target_seen else "NOT FOUND"
        cv2.putText(
            annotated,
            f"tag{self._family} ID {self._target_id}: {status} | tags={tag_count}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        try:
            output = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"annotated image conversion failed: {error}")
            return
        output.header = message.header
        self._publisher.publish(output)


def main(args: list[str] | None = None) -> None:
    """Run the rqt AprilTag debug image publisher."""
    rclpy.init(args=args)
    node = AprilTagDebugImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
