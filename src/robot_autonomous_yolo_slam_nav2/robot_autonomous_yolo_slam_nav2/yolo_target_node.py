"""YOLO person detection + light IoU tracker + AprilTag ID confirm.

- YOLO detect nguoi (yolov8n, imgsz 416), Tracker nhe duy tri ID.
- AprilTag 36h11 ID 0 la tin hieu nhan dang target duy nhat,
  khong bam tag truc tiep.
- Robot bam NGUOI; tag xac nhan / tai xac nhan target, chi quet trong bbox nguoi.
- Tag mat tam thoi -> van FOLLOWING neu tracker giu ID; mat qua lost_timeout_s
  thi clear va can tag moi de lock lai. Khong co tag topic, khong full-frame fallback.
"""

from __future__ import annotations

import math

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from .target_state import TrackId, TrackObservation
from .person_annotation import PersonAnnotationPublisher, TagOverlay
from .person_tag_detector import PersonTagDetector, tagged_tracks
from .person_tracking import PersonTracker
from .tag_person_identity import TagPersonIdentity


def presence_observation_messages(observations, header, state, obs_type):
    """Build one presence message per tracked person for search inspection.

    Messages carry valid=False so metric identity selection ignores them;
    the follow node counts fresh stamps as person presence during a lost
    spin search. The existing target/empty message keeps its authoritative
    meaning and must be published after these.
    """
    messages = []
    for observation in observations:
        x1, y1, x2, y2 = observation.bbox
        message = obs_type()
        message.header = header
        message.track_id = int(observation.track_id)
        message.confidence = float(observation.confidence)
        message.bbox_x = float(x1)
        message.bbox_y = float(y1)
        message.bbox_width = float(x2 - x1)
        message.bbox_height = float(y2 - y1)
        message.state = state
        message.valid = False
        message.footpoint_valid = False
        message.footpoint_x = 0.0
        message.footpoint_y = 0.0
        message.footpoint_confidence = 0.0
        messages.append(message)
    return tuple(messages)


class YoloTargetNode(Node):
    """YOLO + tracker nhe, AprilTag xac nhan target, publish PersonTarget."""

    def __init__(self) -> None:
        super().__init__("yolo_target_node")
        self.declare_parameter("image_topic", "/phone/camera/back/image")
        self.declare_parameter("camera_frame", "phone_camera_back")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("imgsz", 416)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("target_tag_id", 0)
        self.declare_parameter("tag_roi_upscale", 4.0)
        self.declare_parameter("lost_timeout_s", 2.0)
        self.declare_parameter("ambiguity_iou", 0.3)
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("lost_frames", 10)
        self.declare_parameter("iou_match_thresh", 0.3)
        self.declare_parameter("log_every", 30)
        self.declare_parameter("observations_topic", "~/observations")
        self.declare_parameter("target_topic", "~/target")

        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._family = str(self.get_parameter("tag_family").value)
        self._tag_id = int(str(self.get_parameter("target_tag_id").value))
        self._iou_thresh = float(str(self.get_parameter("iou_match_thresh").value))
        self._log_every = int(str(self.get_parameter("log_every").value))
        self._imgsz = int(str(self.get_parameter("imgsz").value))
        self._confidence = float(str(self.get_parameter("conf").value))
        self._device = str(self.get_parameter("device").value)
        model_name = str(self.get_parameter("model").value)

        from robot_autonomous_yolo_slam_nav2.msg import PersonObservation, PersonTarget

        self._obs_type = PersonObservation
        self._target_type = PersonTarget
        self._target_pub = self.create_publisher(
            self._target_type, str(self.get_parameter("target_topic").value), 5
        )
        self._obs_pub = self.create_publisher(
            self._obs_type, str(self.get_parameter("observations_topic").value), 5
        )
        self.create_service(Trigger, "/person_follow/reset_target", self._reset)

        # Lazy import ultralytics de test khong can torch.
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_name)
            warmup = np.zeros((240, 320, 3), dtype=np.uint8)
            try:
                self._model.predict(
                    warmup, imgsz=self._imgsz, conf=self._confidence,
                    device=self._device, verbose=False,
                )
            except Exception as error:  # noqa: BLE001
                self.get_logger().warning(f"YOLO warmup failed: {error}")
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(f"YOLO load failed ({model_name}): {error}")
            raise

        self._selector = TagPersonIdentity(
            lost_frames=int(str(self.get_parameter("lost_frames").value)),
            lost_timeout_s=float(str(self.get_parameter("lost_timeout_s").value)),
            ambiguity_iou=float(str(self.get_parameter("ambiguity_iou").value)),
        )
        self._tracker = PersonTracker(self._iou_thresh)
        self._tag_detector = PersonTagDetector(
            upscale=float(str(self.get_parameter("tag_roi_upscale").value)),
        )
        self._min_confidence = float(str(self.get_parameter("min_confidence").value))
        self._bridge = CvBridge()
        self._annotation = PersonAnnotationPublisher(self)
        self._frame_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        image_topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(Image, image_topic, self._image_callback, qos)
        self.get_logger().info(
            f"YOLO {model_name} imgsz={self._imgsz} on {image_topic}, "
            f"one-shot ROI tag confirm {self._family}:{self._tag_id}"
        )

    def _reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._selector.reset()
        self._tracker.reset()
        resp.success = True
        resp.message = "YOLO target reset; searching for stable person"
        return resp

    def _image_callback(self, msg: Image) -> None:
        received_stamp = self.get_clock().now().to_msg()
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"image conversion failed: {error}")
            return
        try:
            preds = self._model.predict(
                frame, imgsz=self._imgsz, conf=self._confidence,
                classes=[0], device=self._device, verbose=False,
            )
            boxes: list[tuple[float, float, float, float, float]] = []
            for pred in preds:
                b = getattr(pred, "boxes", None)
                if b is None or len(b) == 0:
                    break
                xyxy = np.asarray(b.xyxy, dtype=np.float32).reshape(-1, 4)
                conf = np.asarray(b.conf, dtype=np.float32).reshape(-1)
                for (x1, y1, x2, y2), c in zip(xyxy.tolist(), conf.tolist()):
                    if math.isfinite(x1 + y1 + x2 + y2 + c):
                        boxes.append((float(x1), float(y1), float(x2), float(y2), float(c)))
                break
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(f"YOLO predict failed: {error}")
            return

        tracked = self._tracker.update(boxes)
        observations: list[TrackObservation] = []
        image_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        for (tid, (x1, y1, x2, y2), conf) in tracked:
            if conf < self._min_confidence:
                continue
            observations.append(
                TrackObservation(
                    track_id=TrackId(int(tid)),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        tags: tuple[TagOverlay, ...] = ()
        needs_tag = self._selector.needs_confirmation(observations, image_stamp_ns)
        if needs_tag:
            tags = self._tag_detector.detect(frame, observations)
        confirmed_ids = tagged_tracks(observations, tags, self._tag_id)
        result = self._selector.update(observations, confirmed_ids, image_stamp_ns)
        header = Header()
        header.stamp = received_stamp
        header.frame_id = self._camera_frame
        target_msg = self._target_type()
        target_msg.header = header
        obs_msg = self._obs_type()
        obs_msg.header = header
        if result.target_observation is not None:
            x1, y1, x2, y2 = result.target_observation.bbox
            target_msg.track_id = int(result.target_observation.track_id)
            target_msg.confidence = float(result.target_observation.confidence)
            target_msg.bbox_x, target_msg.bbox_y = float(x1), float(y1)
            target_msg.bbox_width, target_msg.bbox_height = float(x2 - x1), float(y2 - y1)
            target_msg.state = result.state.value
            target_msg.valid = True
            target_msg.identity_confirmed = True
            target_msg.footpoint_valid = False
            target_msg.footpoint_x = 0.0
            target_msg.footpoint_y = 0.0
            target_msg.footpoint_confidence = 0.0
            obs_msg.track_id = target_msg.track_id
            obs_msg.confidence = target_msg.confidence
            obs_msg.bbox_x, obs_msg.bbox_y = target_msg.bbox_x, target_msg.bbox_y
            obs_msg.bbox_width, obs_msg.bbox_height = target_msg.bbox_width, target_msg.bbox_height
            obs_msg.state = result.state.value
            obs_msg.valid = True
            obs_msg.footpoint_valid = False
            obs_msg.footpoint_x = 0.0
            obs_msg.footpoint_y = 0.0
            obs_msg.footpoint_confidence = 0.0
        else:
            target_msg.track_id = 0
            target_msg.confidence = 0.0
            target_msg.state = result.state.value
            target_msg.valid = False
            obs_msg.track_id = 0
            obs_msg.confidence = 0.0
            obs_msg.state = result.state.value
            obs_msg.valid = False
        for presence in presence_observation_messages(
            observations, header, result.state.value, self._obs_type
        ):
            self._obs_pub.publish(presence)
        self._target_pub.publish(target_msg)
        self._obs_pub.publish(obs_msg)
        self._annotation.publish(frame, tracked, result, msg.header, tags)
        self._frame_count += 1
        if self._frame_count % self._log_every == 0:
            self.get_logger().info(
                f"yolo frames={self._frame_count} state={result.state.value} "
                f"tracks={len(tracked)} tags={len(tags)} tag_search={needs_tag}"
            )


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = YoloTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
