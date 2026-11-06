from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from cv_bridge import CvBridge
from numpy.typing import NDArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .hand_raise_logic import PoseDetection
from .person_tracking import bbox_iou
from .target_state import TargetSelectionResult


TrackedPerson = tuple[int, tuple[float, float, float, float], float]

_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


@dataclass(frozen=True, slots=True)
class TagOverlay:
    family: str
    tag_id: int
    corners: tuple[tuple[float, float], ...]


def _draw_pose(
    frame: NDArray[np.uint8], pose: PoseDetection, color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(round(k.x)), int(round(k.y)))
        if k.visible(0.3) and 0 <= k.x < width and 0 <= k.y < height else None
        for k in pose.keypoints
    ]
    for first, second in _SKELETON:
        if points[first] is not None and points[second] is not None:
            cv2.line(frame, points[first], points[second], color, 2)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, color, -1)


def annotate_people(
    frame: NDArray[np.uint8],
    tracks: list[TrackedPerson],
    selection: TargetSelectionResult,
    tags: tuple[TagOverlay, ...] = (),
    poses: tuple[PoseDetection, ...] = (),
    raised: bool = False,
) -> NDArray[np.uint8]:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    target_id = None if selection.target_observation is None else selection.target_observation.track_id
    for track_id, (x1, y1, x2, y2), confidence in tracks:
        left, top = max(0, min(width - 1, int(x1))), max(0, min(height - 1, int(y1)))
        right, bottom = max(0, min(width - 1, int(x2))), max(0, min(height - 1, int(y2)))
        selected = track_id == target_id
        color = (0, 255, 0) if selected else (255, 180, 0)
        label = f"{'TARGET' if selected else 'person'} ID {track_id} {confidence:.2f}"
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.putText(annotated, label, (left, max(44, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for tag in tags:
        points = np.asarray(tag.corners, dtype=np.int32).reshape(4, 2)
        cv2.polylines(annotated, [points], True, (0, 255, 255), 2)
        anchor = (max(0, int(points[0, 0])), max(44, int(points[0, 1]) - 5))
        cv2.putText(annotated, f"{tag.family} ID {tag.tag_id}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    target_box = None if selection.target_observation is None else selection.target_observation.bbox
    best_index, best_iou = -1, 0.0
    if target_box is not None:
        for index, pose in enumerate(poses):
            overlap = bbox_iou(pose.bbox, target_box)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
    for index, pose in enumerate(poses):
        if index == best_index and best_iou >= 0.3:
            _draw_pose(annotated, pose, (0, 255, 0) if raised else (0, 0, 255))
        else:
            _draw_pose(annotated, pose, (160, 160, 160))
    cv2.rectangle(annotated, (0, 0), (width - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        annotated, f"state={selection.state.value} target={target_id if target_id is not None else 'none'}",
        (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )
    tag_state = "CONFIRM" if (tags and target_id is not None) else ("SKIP" if target_id is not None else "SEARCH")
    pose_state = "RAISED" if raised else ("down" if poses else "off")
    status_line = f"people={len(tracks)} tags={len(tags)} tag={tag_state} pose={pose_state}"
    cv2.putText(annotated, status_line, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return annotated


class PersonAnnotationPublisher:
    def __init__(self, node: Node) -> None:
        node.declare_parameter("annotated_topic", "/person_follow/annotated")
        self._bridge = CvBridge()
        self._publisher = node.create_publisher(Image, str(node.get_parameter("annotated_topic").value), 5)

    def publish(
        self,
        frame: NDArray[np.uint8],
        tracks: list[TrackedPerson],
        selection: TargetSelectionResult,
        header: Header,
        tags: tuple[TagOverlay, ...] | None = None,
        poses: tuple[PoseDetection, ...] = (),
        raised: bool = False,
    ) -> None:
        if tags is None:
            tags = ()
        output = self._bridge.cv2_to_imgmsg(
            annotate_people(frame, tracks, selection, tags, poses, raised), encoding="bgr8")
        output.header = header
        self._publisher.publish(output)
