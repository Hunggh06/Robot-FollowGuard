from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from cv_bridge import CvBridge
from numpy.typing import NDArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .target_state import TargetSelectionResult, TrackId


TrackedPerson = tuple[int, tuple[float, float, float, float], float]


@dataclass(frozen=True, slots=True)
class TagOverlay:
    family: str
    tag_id: int
    corners: tuple[tuple[float, float], ...]


def annotate_people(
    frame: NDArray[np.uint8],
    tracks: list[TrackedPerson],
    selection: TargetSelectionResult,
    tags: tuple[TagOverlay, ...] = (),
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
    cv2.rectangle(annotated, (0, 0), (width - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        annotated, f"state={selection.state.value} target={target_id if target_id is not None else 'none'}",
        (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )
    tag_state = "CONFIRM" if tags else ("SKIP" if target_id is not None else "SEARCH")
    status_line = f"people={len(tracks)} tags={len(tags)} tag={tag_state}"
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
    ) -> None:
        if tags is None:
            tags = ()
        output = self._bridge.cv2_to_imgmsg(annotate_people(frame, tracks, selection, tags), encoding="bgr8")
        output.header = header
        self._publisher.publish(output)
