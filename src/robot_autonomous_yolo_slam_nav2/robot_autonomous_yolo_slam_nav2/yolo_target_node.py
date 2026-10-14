"""YOLO person detection + light IoU tracker + AprilTag ID confirm.

Spec (docs/superpowers/specs/2026-10-11-apriltag-reacquisition-design.md):
- YOLO detect nguoi (yolov8n, imgsz 416), Tracker nhe duy tri ID.
- AprilTag 36h11 ID 0 la tin hieu nhan dang target, khong bam tag truc tiep.
- Robot bam NGUOI; tag xac nhan / tai xac nhan target, chi quet trong bbox nguoi.
- Tag mat tam thoi -> van FOLLOWING neu tracker giu ID; mat qua lost_timeout_s
  thi clear va can tag moi de lock lai. Khong co tag topic, khong full-frame fallback.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from .target_state import TrackId, TrackObservation
from .color_band_person_identity import ColorBandPersonIdentity
from .person_annotation import PersonAnnotationPublisher, TagOverlay, BandOverlay
from .person_color_bands_detector import (
    ColorBandsConfig,
    PersonColorBandsDetector,
    banded_tracks,
)
from .person_tag_detector import PersonTagDetector, tagged_tracks
from .person_tracking import PersonTracker
from .reid_branch import ReidBranch, ReidBranchConfig
from .tag_person_identity import TagPersonIdentity


_ROLLBACK_MODES: Final = ("color_bands", "apriltag")


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
        self.declare_parameter("identity_mode", "apriltag")
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("target_tag_id", 0)
        self.declare_parameter("tag_roi_upscale", 4.0)
        self.declare_parameter("band_leg_roi_fraction", 0.5)
        self.declare_parameter("band_confirm_frames", 3)
        self.declare_parameter("band_min_area_px", 100.0)
        self.declare_parameter("band_max_color_ratio", 0.4)
        self.declare_parameter("band_min_sep_ratio", 0.25)
        self.declare_parameter("band_max_dx_ratio", 0.3)
        self.declare_parameter("band_red_hue_low_a", 0)
        self.declare_parameter("band_red_hue_high_a", 10)
        self.declare_parameter("band_red_hue_low_b", 170)
        self.declare_parameter("band_red_hue_high_b", 179)
        self.declare_parameter("band_red_sat_min", 70)
        self.declare_parameter("band_red_val_min", 50)
        self.declare_parameter("band_blue_hue_low", 90)
        self.declare_parameter("band_blue_hue_high", 130)
        self.declare_parameter("band_blue_sat_min", 70)
        self.declare_parameter("band_blue_val_min", 50)
        reid_defaults = ReidBranchConfig()
        self.declare_parameter("reid_checkpoint", reid_defaults.checkpoint_path)
        self.declare_parameter("reid_expected_sha256", reid_defaults.expected_sha256)
        self.declare_parameter("reid_match_threshold", reid_defaults.match_threshold)
        self.declare_parameter("reid_match_margin", reid_defaults.match_margin)
        self.declare_parameter("reid_match_hz", 5.0)
        self.declare_parameter("reid_min_crop_height_px", reid_defaults.min_crop_height_px)
        self.declare_parameter("reid_enroll_samples", reid_defaults.enroll_samples)
        self.declare_parameter("reid_enroll_timeout_s", 3.0)
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
        self._band_selector = ColorBandPersonIdentity(
            lost_frames=int(str(self.get_parameter("lost_frames").value)),
            lost_timeout_s=float(str(self.get_parameter("lost_timeout_s").value)),
            ambiguity_iou=float(str(self.get_parameter("ambiguity_iou").value)),
            confirm_frames=int(str(self.get_parameter("band_confirm_frames").value)),
        )
        self._bands_detector = PersonColorBandsDetector(self._bands_config_from_params())
        self._min_confidence = float(str(self.get_parameter("min_confidence").value))
        self._reid: ReidBranch | None = None
        if self._resolve_identity_mode() == "reid":
            self._reid = ReidBranch(self._reid_config_from_params())
        self.create_service(Trigger, "/person_follow/enroll_target", self._enroll)
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
            f"one-shot ROI tag confirm {self._family}:{self._tag_id}, "
            f"identity_mode={self._resolve_identity_mode()}"
        )

    def _bands_config_from_params(self) -> ColorBandsConfig:
        """Build the band detector config from declared yaml-tunable params."""
        return ColorBandsConfig(
            leg_roi_fraction=float(str(self.get_parameter("band_leg_roi_fraction").value)),
            min_area_px=float(str(self.get_parameter("band_min_area_px").value)),
            max_color_ratio=float(str(self.get_parameter("band_max_color_ratio").value)),
            min_sep_ratio=float(str(self.get_parameter("band_min_sep_ratio").value)),
            max_dx_ratio=float(str(self.get_parameter("band_max_dx_ratio").value)),
            red_hue_low_a=int(str(self.get_parameter("band_red_hue_low_a").value)),
            red_hue_high_a=int(str(self.get_parameter("band_red_hue_high_a").value)),
            red_hue_low_b=int(str(self.get_parameter("band_red_hue_low_b").value)),
            red_hue_high_b=int(str(self.get_parameter("band_red_hue_high_b").value)),
            red_sat_min=int(str(self.get_parameter("band_red_sat_min").value)),
            red_val_min=int(str(self.get_parameter("band_red_val_min").value)),
            blue_hue_low=int(str(self.get_parameter("band_blue_hue_low").value)),
            blue_hue_high=int(str(self.get_parameter("band_blue_hue_high").value)),
            blue_sat_min=int(str(self.get_parameter("band_blue_sat_min").value)),
            blue_val_min=int(str(self.get_parameter("band_blue_val_min").value)),
        )

    def _resolve_identity_mode(self) -> str:
        """Return the selected identity mode, defaulting invalid values to apriltag."""
        mode = str(self.get_parameter("identity_mode").value).strip().lower()
        if mode == "reid":
            return mode
        if mode in _ROLLBACK_MODES:
            return mode
        self.get_logger().warning(
            f"unknown identity_mode={mode!r}; falling back to apriltag"
        )
        return "apriltag"

    def _reid_config_from_params(self) -> ReidBranchConfig:
        """Build the ReID branch config from declared yaml-tunable params."""
        match_hz = float(str(self.get_parameter("reid_match_hz").value))
        return ReidBranchConfig(
            lost_frames=int(str(self.get_parameter("lost_frames").value)),
            lost_timeout_s=float(str(self.get_parameter("lost_timeout_s").value)),
            ambiguity_iou=float(str(self.get_parameter("ambiguity_iou").value)),
            enroll_samples=int(str(self.get_parameter("reid_enroll_samples").value)),
            enroll_timeout_ns=int(float(str(self.get_parameter("reid_enroll_timeout_s").value)) * 1_000_000_000),
            min_crop_height_px=int(str(self.get_parameter("reid_min_crop_height_px").value)),
            match_threshold=float(str(self.get_parameter("reid_match_threshold").value)),
            match_margin=float(str(self.get_parameter("reid_match_margin").value)),
            match_interval_ns=int(1_000_000_000 / match_hz) if match_hz > 0.0 else 200_000_000,
            checkpoint_path=str(self.get_parameter("reid_checkpoint").value),
            expected_sha256=str(self.get_parameter("reid_expected_sha256").value),
        )

    def _enroll(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if self._resolve_identity_mode() != "reid" or self._reid is None:
            resp.success = False
            resp.message = "enroll_target needs identity_mode=reid"
            return resp
        ok, message = self._reid.start_enrollment(self.get_clock().now().nanoseconds)
        resp.success = ok
        resp.message = message
        return resp

    def _reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._selector.reset()
        self._band_selector.reset()
        if self._reid is not None:
            self._reid.reset()
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

        mode = self._resolve_identity_mode()
        tags: tuple[TagOverlay, ...] = ()
        bands: tuple[BandOverlay, ...] = ()
        reid_state: str | None = None
        reid_score = 0.0
        needs_tag = False
        needs_bands = False
        if mode == "color_bands":
            needs_bands = self._band_selector.needs_confirmation(observations, image_stamp_ns)
            band_overlays = (
                self._bands_detector.detect(frame, observations) if needs_bands else ()
            )
            banded_ids = banded_tracks(observations, band_overlays)
            result = self._band_selector.update(observations, banded_ids, image_stamp_ns)
            bands = tuple(
                BandOverlay(
                    track_id=overlay.track_id,
                    red_center=overlay.red_center,
                    blue_center=overlay.blue_center,
                )
                for overlay in band_overlays
            )
        elif mode == "reid":
            if self._reid is None:
                self._reid = ReidBranch(self._reid_config_from_params())
            outcome = self._reid.update(frame, observations, image_stamp_ns)
            result = outcome.result
            reid_state = outcome.status
            reid_score = outcome.active_score
        else:
            needs_tag = self._selector.needs_confirmation(observations, image_stamp_ns)
            tags: tuple[TagOverlay, ...] = (
                self._tag_detector.detect(frame, observations) if needs_tag else ()
            )
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
        self._annotation.publish(frame, tracked, result, msg.header, tags, bands, reid_state)
        self._frame_count += 1
        if self._frame_count % self._log_every == 0:
            self.get_logger().info(
                f"yolo frames={self._frame_count} state={result.state.value} "
                f"tracks={len(tracked)} tags={len(tags)} tag_search={needs_tag} "
                f"mode={mode} bands={len(bands)} bands_search={needs_bands} "
                f"reid={reid_state} reid_score={reid_score:.2f}"
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
