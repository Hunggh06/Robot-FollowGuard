from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from rclpy.node import Node
from std_msgs.msg import Bool, Header

from robot_autonomous_yolo_slam_nav2.msg import HandRaise

from .hand_raise_logic import HandRaiseConfig, Keypoint, PoseDetection, target_leg_raised
from .person_tracking import bbox_iou
from .target_state import TargetSelectionResult

if TYPE_CHECKING:
    from ultralytics import YOLO


class HandRaisePerception:
    def __init__(self, node: Node) -> None:
        self._node = node
        node.declare_parameter("hand_raise_topic", "/person_follow/hand_raise")
        node.declare_parameter("confirm_request_topic", "/person_follow/confirm_requested")
        node.declare_parameter("hand_raise_model", "/home/hungdo/ros2_ws/yolov8n-pose.pt")
        node.declare_parameter("hand_raise_imgsz", 416)
        node.declare_parameter("hand_raise_hz", 5.0)
        node.declare_parameter("hand_raise_keypoint_confidence", 0.5)
        self._model: YOLO | None = None
        self._load_failed = False
        self._requested = False
        self._request_ns = 0
        self._last_inference_ns = 0
        self._last_source_ns = 0
        self._last_poses: tuple[PoseDetection, ...] = ()
        self._last_raised = False
        self._config = HandRaiseConfig(
            keypoint_confidence=float(str(node.get_parameter("hand_raise_keypoint_confidence").value)))
        self._publisher = node.create_publisher(
            HandRaise, str(node.get_parameter("hand_raise_topic").value), 5)
        self._subscription = node.create_subscription(
            Bool, str(node.get_parameter("confirm_request_topic").value), self._request_callback, 1)

    def _request_callback(self, message: Bool) -> None:
        if not message.data:
            self._last_source_ns = 0
        self._requested = bool(message.data)
        self._request_ns = self._node.get_clock().now().nanoseconds

    def publish(
        self, frame: NDArray[np.uint8], selection: TargetSelectionResult, header: Header,
        *, source_stamp_ns: int,
    ) -> tuple[tuple[PoseDetection, ...], bool]:
        now_ns = self._node.get_clock().now().nanoseconds
        if not self._requested or not 0 <= now_ns - self._request_ns <= 500_000_000:
            self._last_poses, self._last_raised = (), False
            return (), False
        target = selection.target_observation
        message = HandRaise(header=header, track_id=0, raised=False)
        if source_stamp_ns <= self._last_source_ns:
            self._publisher.publish(message)
            return self._last_poses, self._last_raised
        self._last_source_ns = source_stamp_ns
        if target is None:
            self._last_poses, self._last_raised = (), False
        else:
            message.track_id = int(target.track_id)
            ambiguous = any(
                bbox_iou(target.bbox, other.bbox) >= 0.3
                for other in selection.non_target_observations
            )
            if ambiguous:
                self._last_poses, self._last_raised = (), False
            else:
                hz = float(str(self._node.get_parameter("hand_raise_hz").value))
                if hz <= 0.0 or now_ns - self._last_inference_ns < 1_000_000_000 / hz:
                    return self._last_poses, self._last_raised
                self._last_inference_ns = now_ns
                poses = self._detect(frame)
                message.raised = target_leg_raised(poses, target.bbox, self._config)
                self._last_poses = poses
                self._last_raised = bool(message.raised)
                best_iou = max(
                    (bbox_iou(pose.bbox, target.bbox) for pose in poses), default=0.0)
                self._node.get_logger().info(
                    f"hand_raise track={message.track_id} poses={len(poses)} "
                    f"best_iou={best_iou:.2f} raised={message.raised}",
                    throttle_duration_sec=1.0,
                )
        self._publisher.publish(message)
        return self._last_poses, self._last_raised

    def _detect(self, frame: NDArray[np.uint8]) -> tuple[PoseDetection, ...]:
        if self._load_failed:
            return ()
        try:
            model = self._model
            if model is None:
                from ultralytics import YOLO
                model_path = Path(str(self._node.get_parameter("hand_raise_model").value))
                if not model_path.is_file():
                    self._load_failed = True
                    self._node.get_logger().error(f"Hand-raise model missing: {model_path}")
                    return ()
                model = YOLO(str(model_path), task="pose")
                self._model = model
            results = model.predict(
                frame, imgsz=int(str(self._node.get_parameter("hand_raise_imgsz").value)),
                conf=0.5, device="cpu", verbose=False,
            )
            poses: list[PoseDetection] = []
            for result in results:
                if result.boxes is None or result.keypoints is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                points = result.keypoints.data.cpu().numpy()
                for box, joints in zip(boxes, points):
                    if joints.shape != (17, 3):
                        continue
                    poses.append(PoseDetection(
                        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        tuple(Keypoint(float(p[0]), float(p[1]), float(p[2])) for p in joints),
                    ))
            return tuple(poses)
        except (OSError, RuntimeError, ValueError) as error:
            self._node.get_logger().error(
                f"Hand-raise inference unavailable: {error}", throttle_duration_sec=10.0)
            return ()
