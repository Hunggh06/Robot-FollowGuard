"""Lock the first stable person track without automatic identity switching."""

from dataclasses import dataclass
from enum import Enum
from typing import NewType, Sequence, TypeAlias

from typing_extensions import assert_never

TrackId = NewType("TrackId", int)
BoundingBox: TypeAlias = tuple[float, float, float, float]


class TargetState(Enum):
    """States exposed by the target identity state machine."""

    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One tracker observation, including optional appearance data."""

    track_id: TrackId
    confidence: float
    bbox: BoundingBox
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class TargetSelectionResult:
    """The locked target, non-target observations, and motion safety state."""

    state: TargetState
    target_observation: TrackObservation | None
    non_target_observations: tuple[TrackObservation, ...]
    forward_motion_allowed: bool


class TargetSelector:
    """Mutable first identity lock layer; metric target association is a later adapter."""

    __slots__ = (
        "_candidate_streak",
        "_candidate_track_id",
        "_lost_streak",
        "_lost_frames",
        "_min_confidence",
        "_stable_frames",
        "_state",
        "_target_track_id",
    )

    def __init__(
        self,
        stable_frames: int = 3,
        min_confidence: float = 0.5,
        lost_frames: int = 3,
    ) -> None:
        self._stable_frames = stable_frames
        self._min_confidence = min_confidence
        self._lost_frames = lost_frames
        self._state = TargetState.SEARCHING
        self._candidate_track_id: TrackId | None = None
        self._candidate_streak = 0
        self._target_track_id: TrackId | None = None
        self._lost_streak = 0

    @property
    def state(self) -> TargetState:
        """Return the current state without changing the selector."""
        return self._state

    def update(
        self,
        observations: Sequence[TrackObservation],
    ) -> TargetSelectionResult:
        """Consume one frame of tracker observations."""
        match self._state:
            case TargetState.SEARCHING:
                return self._update_searching(observations)
            case TargetState.TRACKING:
                return self._update_tracking(observations)
            case TargetState.LOST:
                return self._result(observations, None)
            case TargetState.STOPPED:
                return self._result(observations, None)
            case unreachable:
                assert_never(unreachable)

    def reset(self) -> None:
        """Explicitly clear the lock and return to searching."""
        self._state = TargetState.SEARCHING
        self._candidate_track_id = None
        self._candidate_streak = 0
        self._target_track_id = None
        self._lost_streak = 0

    def stop(self) -> None:
        """Stop motion and keep the selector stopped until an explicit reset."""
        self._state = TargetState.STOPPED

    def _update_searching(
        self,
        observations: Sequence[TrackObservation],
    ) -> TargetSelectionResult:
        eligible = tuple(
            observation
            for observation in observations
            if observation.confidence >= self._min_confidence
        )
        if not eligible:
            self._candidate_track_id = None
            self._candidate_streak = 0
            return self._result(observations, None)

        candidate_observation = next(
            (
                observation
                for observation in eligible
                if observation.track_id == self._candidate_track_id
            ),
            None,
        )
        if candidate_observation is not None:
            self._candidate_streak += 1
        else:
            candidate_observation = eligible[0]
            self._candidate_track_id = candidate_observation.track_id
            self._candidate_streak = 1

        if self._candidate_streak < self._stable_frames:
            return self._result(observations, None)

        self._target_track_id = candidate_observation.track_id
        self._state = TargetState.TRACKING
        self._lost_streak = 0
        return self._result(observations, candidate_observation)

    def _update_tracking(
        self,
        observations: Sequence[TrackObservation],
    ) -> TargetSelectionResult:
        target_track_id = self._target_track_id
        assert target_track_id is not None
        target_observation = next(
            (
                observation
                for observation in observations
                if observation.track_id == target_track_id
            ),
            None,
        )
        if target_observation is None:
            self._lost_streak += 1
            if self._lost_streak >= self._lost_frames:
                self._state = TargetState.LOST
            return self._result(observations, None)

        self._lost_streak = 0
        return self._result(observations, target_observation)

    def _result(
        self,
        observations: Sequence[TrackObservation],
        target_observation: TrackObservation | None,
    ) -> TargetSelectionResult:
        target_track_id = (
            target_observation.track_id if target_observation is not None else None
        )
        non_target_observations = tuple(
            observation
            for observation in observations
            if observation.track_id != target_track_id
        )
        return TargetSelectionResult(
            state=self._state,
            target_observation=target_observation,
            non_target_observations=non_target_observations,
            forward_motion_allowed=(
                self._state is TargetState.TRACKING
                and target_observation is not None
            ),
        )
