"""Shared person-track state types."""

from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeAlias

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