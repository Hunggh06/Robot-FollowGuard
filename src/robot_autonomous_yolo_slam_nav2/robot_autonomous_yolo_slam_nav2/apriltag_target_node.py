"""ROS adapter from apriltag_ros detections to the local target contract."""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Integral
from typing import Any

from rclpy.node import Node
from std_srvs.srv import Trigger

from .apriltag_target_state import AprilTagTargetSelector, TagDetection


def _stamp_ns(header: Any) -> int:
    sec = header.stamp.sec
    nanosec = header.stamp.nanosec
    if (
        isinstance(sec, bool)
        or isinstance(nanosec, bool)
        or not isinstance(sec, Integral)
        or not isinstance(nanosec, Integral)
    ):
        raise ValueError("invalid ROS timestamp")
    sec_value = int(sec)
    nanosec_value = int(nanosec)
    if sec_value < 0 or not 0 <= nanosec_value < 1_000_000_000:
        raise ValueError("invalid ROS timestamp")
    return sec_value * 1_000_000_000 + nanosec_value


def _corners(message: Any) -> tuple[tuple[float, float], ...] | None:
    try:
        values = tuple(getattr(message, "corners", ()))
        points = tuple((float(point.x), float(point.y)) for point in values)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(points) != 4:
        return None
    return points if all(math.isfinite(value) for point in points for value in point) else None


def _tag_id(message: Any) -> int | None:
    values = getattr(message, "id", ())
    if isinstance(values, (str, bytes)):
        return None
    try:
        sequence = tuple(values) if isinstance(values, Iterable) else (values,)
        if len(sequence) != 1:
            return None
        value = sequence[0]
        if isinstance(value, bool) or not isinstance(value, Integral):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def detection_message_to_observation(
    message: Any,
    family: str,
    stamp_ns: int,
) -> TagDetection | None:
    tag_id = _tag_id(message)
    corners = _corners(message)
    if tag_id is None or corners is None:
        return None
    return TagDetection(
        family=family,
        tag_id=tag_id,
        corners=(corners[0], corners[1], corners[2], corners[3]),
        stamp_ns=stamp_ns,
    )


def detection_frame_matches(header: Any, expected_frame: str) -> bool:
    frame_id = getattr(header, "frame_id", None)
    return isinstance(frame_id, str) and frame_id == expected_frame


class AprilTagTargetNode(Node):
    """Publish one configured AprilTag as the existing target message."""

    def __init__(self) -> None:
        super().__init__("apriltag_target_node")
        self.declare_parameter("image_topic", "/phone/camera/back/image")
        self.declare_parameter("camera_info_topic", "/phone/camera/back/camera_info")
        self.declare_parameter("camera_frame", "phone_camera_back")
        self.declare_parameter("detections_topic", "/apriltag/detections")
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("target_tag_id", 0)
        self.declare_parameter("tag_size_m", 0.075)
        self.declare_parameter("stable_frames", 3)
        self.declare_parameter("lost_frames", 3)
        self.declare_parameter("observations_topic", "~/observations")
        self.declare_parameter("target_topic", "~/target")
        self._image_topic = str(self.get_parameter("image_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._tag_size_m = float(str(self.get_parameter("tag_size_m").value))
        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._family = str(self.get_parameter("tag_family").value)
        self._tag_id = int(str(self.get_parameter("target_tag_id").value))
        self._selector = AprilTagTargetSelector(
            family=self._family,
            tag_id=self._tag_id,
            stable_frames=int(str(self.get_parameter("stable_frames").value)),
            lost_frames=int(str(self.get_parameter("lost_frames").value)),
        )
        from robot_autonomous_yolo_slam_nav2.msg import PersonObservation, PersonTarget

        self._target_type = PersonTarget
        self._observation_type = PersonObservation
        self._target_pub = self.create_publisher(
            self._target_type,
            str(self.get_parameter("target_topic").value),
            5,
        )
        self._observation_pub = self.create_publisher(
            self._observation_type,
            str(self.get_parameter("observations_topic").value),
            5,
        )
        self.create_service(Trigger, "/person_follow/reset_target", self._reset)
        try:
            import importlib

            AprilTagDetectionArray = getattr(
                importlib.import_module("apriltag_msgs.msg"),
                "AprilTagDetectionArray",
            )
        except ImportError as error:
            raise RuntimeError(
                "apriltag_msgs is required; install apriltag_ros before starting "
                "apriltag_target_node"
            ) from error
        detection_topic = str(self.get_parameter("detections_topic").value)
        self.create_subscription(AprilTagDetectionArray, detection_topic, self._callback, 5)
        self.get_logger().info(
            f"Subscribed to {detection_topic} for {self._family}:{self._tag_id}; "
            f"detector image={self._image_topic}, camera_info={self._camera_info_topic}, "
            f"tag_size_m={self._tag_size_m}"
        )

    def _reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._selector.reset()
        response.success = True
        response.message = "AprilTag target reset; searching for configured tag"
        return response

    def _callback(self, message: Any) -> None:
        header = getattr(message, "header", None)
        if header is None:
            self.get_logger().warning("ignoring AprilTag message without a header")
            return
        now_ns = self.get_clock().now().nanoseconds
        try:
            stamp_ns = _stamp_ns(header)
            raw_detections = tuple(message.detections)
        except (AttributeError, TypeError, ValueError):
            stamp_ns = 0
            raw_detections = ()
        if not detection_frame_matches(header, self._camera_frame):
            raw_detections = ()
        detections = tuple(
            parsed
            for parsed in (
                detection_message_to_observation(
                    item,
                    self._family,
                    stamp_ns,
                )
                for item in raw_detections
            )
            if parsed is not None
        )
        selection = self._selector.update(detections, now_ns)
        target = self._target_message(header, selection)
        self._target_pub.publish(target)
        self._observation_pub.publish(self._observation_message(header, selection))

    def _target_message(self, header: Any, selection: Any) -> Any:
        message = self._target_type()
        message.header = header
        message.track_id = self._tag_id
        message.state = selection.state.value
        message.valid = selection.detection is not None
        message.confidence = 1.0 if message.valid else 0.0
        if selection.detection is not None:
            message.bbox_x, message.bbox_y, message.bbox_width, message.bbox_height = (
                selection.detection.bbox
            )
        return message

    def _observation_message(self, header: Any, selection: Any) -> Any:
        message = self._observation_type()
        message.header = header
        message.track_id = self._tag_id
        message.state = selection.state.value
        message.valid = selection.detection is not None
        message.confidence = 1.0 if message.valid else 0.0
        if selection.detection is not None:
            message.bbox_x, message.bbox_y, message.bbox_width, message.bbox_height = (
                selection.detection.bbox
            )
        return message


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = AprilTagTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
