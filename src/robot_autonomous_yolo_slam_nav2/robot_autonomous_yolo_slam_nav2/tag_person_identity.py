from __future__ import annotations

from collections.abc import Sequence

from .target_state import TargetSelectionResult, TargetState, TrackId, TrackObservation


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


class TagPersonIdentity:
    """Retain one tag-confirmed identity until tracking evidence becomes ambiguous."""

    def __init__(self, lost_frames: int = 10, lost_timeout_s: float = 2.0, ambiguity_iou: float = 0.3) -> None:
        self._lost_frames = lost_frames
        self._lost_timeout_ns = int(lost_timeout_s * 1_000_000_000)
        self._ambiguity_iou = ambiguity_iou
        self.reset()

    @property
    def state(self) -> TargetState:
        return self._state

    def reset(self) -> None:
        self._state = TargetState.SEARCHING
        self._target_id: TrackId | None = None
        self._last_seen_ns = 0
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._missing_frames = 0
        self._awaiting_confirmation = False

    def needs_confirmation(self, observations: Sequence[TrackObservation], stamp_ns: int) -> bool:
        if self._target_id is None or self._awaiting_confirmation:
            return True
        target = next((p for p in observations if p.track_id == self._target_id), None)
        if target is None or stamp_ns - self._last_seen_ns >= self._lost_timeout_ns:
            return True
        if self._last_bbox is not None and bbox_iou(target.bbox, self._last_bbox) < 0.1:
            return True
        return any(
            p.track_id != target.track_id and bbox_iou(target.bbox, p.bbox) >= self._ambiguity_iou
            for p in observations
        )

    def update(
        self, observations: Sequence[TrackObservation], tagged_ids: Sequence[TrackId], stamp_ns: int,
    ) -> TargetSelectionResult:
        needs_tag = self.needs_confirmation(observations, stamp_ns)
        target = next((p for p in observations if p.track_id == self._target_id), None)
        if needs_tag:
            candidates = tuple(p for p in observations if p.track_id in tagged_ids)
            if len(candidates) == 1:
                target = candidates[0]
                self._target_id = target.track_id
                self._awaiting_confirmation = False
            else:
                if self._target_id is not None:
                    self._missing_frames += 1
                    long_gap = stamp_ns - self._last_seen_ns >= self._lost_timeout_ns
                    if long_gap and target is None:
                        self._target_id = None
                        self._last_bbox = None
                        self._missing_frames = 0
                        self._awaiting_confirmation = True
                        self._state = TargetState.LOST
                        return self._result(observations, None)
                    self._awaiting_confirmation = (
                        self._awaiting_confirmation or target is not None
                        or self._missing_frames >= self._lost_frames or long_gap
                    )
                    self._state = TargetState.LOST if self._awaiting_confirmation else TargetState.TRACKING
                return self._result(observations, None)
        if target is not None:
            self._state = TargetState.TRACKING
            self._missing_frames = 0
            self._last_seen_ns = stamp_ns
            self._last_bbox = target.bbox
        return self._result(observations, target)

    def _result(self, observations: Sequence[TrackObservation], target: TrackObservation | None) -> TargetSelectionResult:
        return TargetSelectionResult(
            self._state, target,
            tuple(p for p in observations if target is None or p.track_id != target.track_id),
            target is not None and self._state is TargetState.TRACKING,
        )
