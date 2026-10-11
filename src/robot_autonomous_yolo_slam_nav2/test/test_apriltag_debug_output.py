import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge

from robot_autonomous_yolo_slam_nav2.apriltag_debug_image_node import AprilTagDebugImageNode


def test_disabling_tag_only_image_keeps_detection_publisher(monkeypatch) -> None:
    rclpy.init(args=["--ros-args", "-p", "publish_annotated_image:=false"])
    node = AprilTagDebugImageNode()
    try:
        assert node._publisher is None
        assert node._detection_publisher is not None
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image[80:160, 120:200] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, 0, 80), cv2.COLOR_GRAY2BGR,
        )
        message = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
        detections = []
        monkeypatch.setattr(node._detection_publisher, "publish", detections.append)

        def fail_if_drawing(*args, **kwargs):
            raise AssertionError("tag-only drawing should be disabled")

        monkeypatch.setattr(cv2, "putText", fail_if_drawing)
        monkeypatch.setattr(cv2, "polylines", fail_if_drawing)
        node._callback(message)

        assert len(detections) == 1
        assert [tag.id for tag in detections[0].detections] == [0]
    finally:
        node.destroy_node()
        rclpy.shutdown()
