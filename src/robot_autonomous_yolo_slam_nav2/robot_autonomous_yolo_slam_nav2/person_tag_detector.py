from __future__ import annotations

from collections.abc import Sequence
import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .person_annotation import TagOverlay
from .target_state import TrackId, TrackObservation


def resized_roi(gray_roi, scale: float):
    return cv2.resize(gray_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


class PersonTagDetector:
    """Search bounded person ROIs and return tag corners in source-image pixels."""

    def __init__(self, upscale: float = 4.0) -> None:
        self._upscale = upscale
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = 0.01
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, frame: NDArray[np.uint8], people: Sequence[TrackObservation]) -> tuple[TagOverlay, ...]:
        height, width = frame.shape[:2]
        tags: list[TagOverlay] = []
        for person in people:
            x1, y1, x2, y2 = person.bbox
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
            if right - left < 8 or bottom - top < 8:
                continue
            roi = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            scale = max(1.0, min(self._upscale, 640.0 / max(roi.shape)))
            self._accumulate(resized_roi(roi, scale), left, top, scale, scale, tags)
        if not tags:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            full_scale = max(1.0, min(2.0, 640.0 / max(gray.shape)))
            self._accumulate(resized_roi(gray, full_scale), 0, 0, full_scale, full_scale, tags)
        return tuple(tags)

    def _accumulate(self, resized, left: int, top: int,
                    scale_x: float, scale_y: float,
                    tags: list[TagOverlay]) -> None:
        corners, ids, _ = self._detector.detectMarkers(resized)
        if ids is None:
            return
        for raw_corners, tag_id in zip(corners, ids.reshape(-1).tolist()):
            points = tuple((float(x) / scale_x + left, float(y) / scale_y + top)
                           for x, y in raw_corners.reshape(4, 2))
            center = (sum(p[0] for p in points) / 4, sum(p[1] for p in points) / 4)
            duplicate = any(
                tag.tag_id == tag_id and math.hypot(
                    center[0] - sum(p[0] for p in tag.corners) / 4,
                    center[1] - sum(p[1] for p in tag.corners) / 4,
                ) < 3.0 for tag in tags
            )
            if not duplicate:
                tags.append(TagOverlay("tag36h11", int(tag_id), points))


def tagged_tracks(
    people: Sequence[TrackObservation], tags: Sequence[TagOverlay], target_tag_id: int,
) -> tuple[TrackId, ...]:
    accepted = tuple(tag for tag in tags if tag.family == "tag36h11" and tag.tag_id == target_tag_id)
    if len(accepted) != 1:
        return ()
    center_x = sum(p[0] for p in accepted[0].corners) / 4
    center_y = sum(p[1] for p in accepted[0].corners) / 4
    matches = tuple(p.track_id for p in people
                    if p.bbox[0] <= center_x <= p.bbox[2] and p.bbox[1] <= center_y <= p.bbox[3])
    return matches if len(matches) == 1 else ()
