"""ROS and model runtime boundary for camera person tracking."""

from __future__ import annotations

import importlib

import numpy as np
from boxmot import ReIDModel
from boxmot.trackers.bbox.botsort import BotSort
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from ultralytics import YOLO

from .reid_target_helpers import (
    TrackerRows,
    selection_to_ros_messages,
    tracker_rows_to_observations,
)
from .target_state import TargetSelector

_generated = importlib.import_module("robot_autonomous_yolo_slam_nav2.msg")
_PersonObservation = getattr(_generated, "PersonObservation")
_PersonTarget = getattr(_generated, "PersonTarget")


class ReidTargetNode(Node):
    """ROS adapter running YOLO, BoT-SORT, OSNet, and target selection."""

    def __init__(self) -> None:
        super().__init__("reid_target_node")
        self.declare_parameter("image_topic", "/phone/camera/back/image")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("imgsz", 416)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("reid_weights", "osnet_x0_25_msmt17.pt")
        self.declare_parameter("stable_frames", 3)
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("lost_frames", 3)
        self.declare_parameter("log_every", 30)
        self.declare_parameter("observations_topic", "~/observations")
        self.declare_parameter("target_topic", "~/target")
        topic = str(self.get_parameter("image_topic").value)
        model_name = str(self.get_parameter("model").value)
        imgsz = int(str(self.get_parameter("imgsz").value))
        confidence = float(str(self.get_parameter("conf").value))
        device = str(self.get_parameter("device").value)
        reid_weights = str(self.get_parameter("reid_weights").value)
        self._log_every = int(str(self.get_parameter("log_every").value))
        self._model = YOLO(model_name)
        reid_model = ReIDModel(reid_weights, device=device, half=False)
        self._tracker = BotSort(reid_model=reid_model, with_reid=True, frame_rate=20)
        warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self._model.predict(
                warmup_frame,
                imgsz=imgsz,
                conf=confidence,
                device=device,
                verbose=False,
            )
            self._tracker.update(np.zeros((0, 6), dtype=np.float32), warmup_frame)
        except (RuntimeError, TypeError, ValueError) as error:
            self.get_logger().warning(f"model warmup failed: {error}")
        self._selector = TargetSelector(
            stable_frames=int(str(self.get_parameter("stable_frames").value)),
            min_confidence=float(str(self.get_parameter("min_confidence").value)),
            lost_frames=int(str(self.get_parameter("lost_frames").value)),
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._bridge = CvBridge()
        self._observations_pub = self.create_publisher(
            _PersonObservation,
            str(self.get_parameter("observations_topic").value),
            5,
        )
        self._target_pub = self.create_publisher(
            _PersonTarget,
            str(self.get_parameter("target_topic").value),
            5,
        )
        self.create_subscription(Image, topic, self._image_callback, qos)
        self.create_service(Trigger, "/person_follow/reset_target", self._reset_target)
        self._frame_count = 0
        self._imgsz = imgsz
        self._confidence = confidence
        self._device = device
        self.get_logger().info(f"Subscribed to {topic}")

    def _reset_target(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._selector.reset()
        response.success = True
        response.message = "target reset; searching for a stable person"
        return response

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"image conversion failed: {error}")
            return
        try:
            predictions = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._confidence,
                classes=[0],
                device=self._device,
                verbose=False,
            )
            detections = np.zeros((0, 6), dtype=np.float32)
            for prediction in predictions:
                boxes = getattr(prediction, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    break
                xyxy = np.asarray(getattr(boxes, "xyxy"), dtype=np.float32).reshape(-1, 4)
                conf = np.asarray(getattr(boxes, "conf"), dtype=np.float32).reshape(-1, 1)
                classes = np.asarray(getattr(boxes, "cls"), dtype=np.float32).reshape(-1, 1)
                detections = np.hstack((xyxy, conf, classes))
                break
            tracks = self._tracker.update(detections, frame)
            track_array = np.asarray(tracks, dtype=np.float32)
            if track_array.size == 0:
                rows: TrackerRows = ()
            elif track_array.ndim == 1 and track_array.size % 8 == 0:
                rows = tuple(
                    tuple(float(value) for value in row)
                    for row in track_array.reshape(-1, 8).tolist()
                )
            elif track_array.ndim == 2:
                rows = tuple(
                    tuple(float(value) for value in row)
                    for row in track_array.tolist()
                )
            else:
                rows = ()
        except (RuntimeError, TypeError, ValueError) as error:
            self.get_logger().error(f"YOLO or tracker processing failed: {error}")
            return
        result = self._selector.update(tracker_rows_to_observations(rows))
        observations, target = selection_to_ros_messages(result, message.header)
        for observation in observations:
            self._observations_pub.publish(observation)
        self._target_pub.publish(target)
        self._frame_count += 1
        if self._frame_count % self._log_every == 0:
            self.get_logger().info(
                f"processed {self._frame_count} frames in state {result.state.value}",
            )
