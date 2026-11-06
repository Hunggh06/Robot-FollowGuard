from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .person_tracking import bbox_iou
from .target_state import BoundingBox, TrackId


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    confidence: float

    def visible(self, minimum: float) -> bool:
        # A confident joint can legitimately sit on the top/left image edge
        # (x or y == 0), so confidence alone distinguishes it from the
        # (0, 0, 0) padding Ultralytics emits for undetected joints.
        return (
            math.isfinite(self.x + self.y + self.confidence)
            and minimum <= self.confidence <= 1.0
        )


@dataclass(frozen=True, slots=True)
class PoseDetection:
    bbox: BoundingBox
    keypoints: tuple[Keypoint, ...]


@dataclass(frozen=True, slots=True)
class HandRaiseConfig:
    keypoint_confidence: float = 0.5
    match_iou: float = 0.5
    leg_lift_ratio: float = 0.15
    ankle_above_knee_ratio: float = 0.10


_LEGS = ((11, 13, 15), (12, 14, 16))


def _distance(first: Keypoint, second: Keypoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def target_leg_raised(
    poses: Sequence[PoseDetection], target: BoundingBox,
    config: HandRaiseConfig = HandRaiseConfig(),
) -> bool:
    """Accept one COCO pose whose ankle has clearly left the ground.

    With both ankles visible the lower one is treated as the planted foot, so
    the other ankle must rise by a fraction of that leg's length. When only one
    ankle is visible, the foot must instead be lifted above its own knee. Either
    way the raised leg needs a confident hip, knee, and ankle.
    """
    matches = [pose for pose in poses if bbox_iou(pose.bbox, target) >= config.match_iou]
    if len(matches) != 1:
        return False
    points = matches[0].keypoints
    if len(points) != 17:
        return False
    minimum = config.keypoint_confidence
    complete = [leg for leg in _LEGS if all(points[i].visible(minimum) for i in leg)]
    if not complete:
        return False
    ankles = [index for index in (15, 16) if points[index].visible(minimum)]
    if len(ankles) == 2:
        standing = max(ankles, key=lambda index: points[index].y)
        lifted = min(ankles, key=lambda index: points[index].y)
        lifted_leg = next((leg for leg in _LEGS if leg[2] == lifted), None)
        if lifted_leg is None or lifted_leg not in complete:
            return False
        standing_leg = next((leg for leg in _LEGS if leg[2] == standing), None)
        if standing_leg is not None and standing_leg in complete:
            scale = _distance(points[standing_leg[0]], points[standing_leg[2]])
        else:
            scale = _distance(points[lifted_leg[0]], points[lifted_leg[2]])
        if scale < 5.0:
            return False
        lift = points[standing].y - points[lifted].y
        return lift >= config.leg_lift_ratio * scale
    ankle = ankles[0]
    leg = next(leg for leg in _LEGS if leg[2] == ankle)
    if leg not in complete:
        return False
    hip, knee, _ = leg
    knee_to_hip = _distance(points[hip], points[knee])
    if knee_to_hip < 5.0:
        return False
    return points[knee].y - points[ankle].y >= config.ankle_above_knee_ratio * knee_to_hip


@dataclass(frozen=True, slots=True)
class GestureSample:
    track_id: TrackId
    stamp_ns: int
    raised: bool


class HandRaiseHold:
    """Mutable accumulator of distinct, fresh observations for a single identity."""

    def __init__(self, hold_s: float = 1.0, max_gap_s: float = 0.5) -> None:
        self._hold_ns = int(hold_s * 1_000_000_000)
        self._gap_ns = int(max_gap_s * 1_000_000_000)
        self.reset()

    def reset(self) -> None:
        self._track_id: TrackId | None = None
        self._since_ns = 0
        self._last_ns = 0

    def update(self, sample: GestureSample, now_ns: int) -> None:
        if (
            not sample.raised or sample.track_id <= 0 or sample.stamp_ns <= 0
            or not 0 <= now_ns - sample.stamp_ns <= self._gap_ns
        ):
            self.reset()
            return
        if sample.track_id == self._track_id and sample.stamp_ns == self._last_ns:
            return
        if (
            sample.track_id != self._track_id
            or not 0 < sample.stamp_ns - self._last_ns <= self._gap_ns
        ):
            self._since_ns = sample.stamp_ns
        self._track_id = sample.track_id
        self._last_ns = sample.stamp_ns

    def ready(self, now_ns: int, track_id: TrackId) -> bool:
        return (
            self._track_id == track_id
            and 0 <= now_ns - self._last_ns <= self._gap_ns
            and self._last_ns - self._since_ns >= self._hold_ns
        )
