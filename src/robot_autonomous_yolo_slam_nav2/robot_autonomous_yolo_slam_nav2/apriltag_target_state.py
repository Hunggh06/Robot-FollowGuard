"""Fail-closed target state for one configured AprilTag identity."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence, TypeAlias


Point2D: TypeAlias = tuple[float, float]
BoundingBox: TypeAlias = tuple[float, float, float, float]
TAG_FAMILY = "tag36h11"


class TagTargetState(Enum):
    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TagDetection:
    family: str
    tag_id: int
    corners: tuple[Point2D, Point2D, Point2D, Point2D]
    stamp_ns: int

    @property
    def bbox(self) -> BoundingBox:
        xs = tuple(point[0] for point in self.corners)
        ys = tuple(point[1] for point in self.corners)
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        return left, top, right - left, bottom - top

    @property
    def area(self) -> float:
        _, _, width, height = self.bbox
        return width * height


@dataclass(frozen=True, slots=True)
class TagSelection:
    state: TagTargetState
    detection: TagDetection | None

    @property
    def forward_motion_allowed(self) -> bool:
        return self.state is TagTargetState.TRACKING and self.detection is not None


class AprilTagTargetSelector:
    """Lock only one family/ID and never switch to another detection."""

    def __init__(
        self,
        family: str = TAG_FAMILY,
        tag_id: int = 0,
        stable_frames: int = 3,
        lost_frames: int = 3,
        max_age_ns: int = 500_000_000,
        min_area_px: float = 25.0,
    ) -> None:
        self._family = family
        self._tag_id = tag_id
        self._stable_frames = stable_frames
        self._lost_frames = lost_frames
        self._max_age_ns = max_age_ns
        self._min_area_px = min_area_px
        self._state = TagTargetState.SEARCHING
        self._stable_streak = 0
        self._lost_streak = 0

    @property
    def state(self) -> TagTargetState:
        return self._state

    def update(
        self,
        detections: Sequence[TagDetection],
        now_ns: int,
    ) -> TagSelection:
        if self._state is TagTargetState.STOPPED:
            return TagSelection(self._state, None)
        valid = tuple(self._valid_detection(item, now_ns) for item in detections)
        candidates = tuple(item for item in valid if item is not None)
        if len(candidates) != 1:
            return self._missing()
        detection = candidates[0]
        if self._state is TagTargetState.SEARCHING:
            self._stable_streak += 1
            if self._stable_streak < self._stable_frames:
                return TagSelection(self._state, None)
            self._state = TagTargetState.TRACKING
            self._lost_streak = 0
            return TagSelection(self._state, detection)
        self._lost_streak = 0
        return TagSelection(self._state, detection)

    def reset(self) -> None:
        self._state = TagTargetState.SEARCHING
        self._stable_streak = 0
        self._lost_streak = 0

    def stop(self) -> None:
        self._state = TagTargetState.STOPPED

    def _valid_detection(self, detection: TagDetection, now_ns: int) -> TagDetection | None:
        if detection.family != self._family or detection.tag_id != self._tag_id:
            return None
        if (
            detection.stamp_ns <= 0
            or detection.stamp_ns > now_ns
            or now_ns - detection.stamp_ns > self._max_age_ns
        ):
            return None
        if not all(
            math.isfinite(value)
            for point in detection.corners
            for value in point
        ):
            return None
        if detection.area < self._min_area_px:
            return None
        return detection

    def _missing(self) -> TagSelection:
        self._stable_streak = 0
        if self._state is TagTargetState.SEARCHING:
            return TagSelection(self._state, None)
        self._lost_streak += 1
        if self._lost_streak >= self._lost_frames:
            self._state = TagTargetState.LOST
        return TagSelection(self._state, None)
