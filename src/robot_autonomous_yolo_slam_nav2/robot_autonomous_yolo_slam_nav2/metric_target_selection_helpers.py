"""Target selection and obstacle projection helpers for metric association."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias, TypeVar

from .metric_association import MetricPersonObstacle, Point3D
from .reid_target_helpers import PersonObservationFields, PersonTargetFields
from .target_state import TargetSelectionResult, TargetState, TrackId, TrackObservation


HeaderT = TypeVar("HeaderT")
Point3DList: TypeAlias = tuple[Point3D, ...]


def derive_target_selection(
    target: PersonTargetFields[HeaderT],
    observations: Sequence[PersonObservationFields[HeaderT]],
) -> TargetSelectionResult | None:
    """Build a locked selection from current messages without identity switching."""
    if target.track_id < 0 or target.state != TargetState.TRACKING.value:
        return None
    if not _valid_detection(
        target.track_id,
        target.confidence,
        target.bbox_x,
        target.bbox_y,
        target.bbox_width,
        target.bbox_height,
    ):
        return None

    target_observation = TrackObservation(
        track_id=TrackId(target.track_id),
        confidence=target.confidence,
        bbox=_bbox_from_fields(target),
    )
    non_target_observations = tuple(
        TrackObservation(
            track_id=TrackId(observation.track_id),
            confidence=observation.confidence,
            bbox=_bbox_from_fields(observation),
        )
        for observation in observations
        if observation.valid
        and observation.track_id != target.track_id
        and _valid_detection(
            observation.track_id,
            observation.confidence,
            observation.bbox_x,
            observation.bbox_y,
            observation.bbox_width,
            observation.bbox_height,
        )
    )
    return TargetSelectionResult(
        state=TargetState.TRACKING,
        target_observation=target_observation,
        non_target_observations=non_target_observations,
        forward_motion_allowed=True,
    )


def metric_obstacles_to_xyz(
    obstacles: Sequence[MetricPersonObstacle],
) -> Point3DList:
    """Return finite metric non-target points in PointCloud2 tuple form."""
    return tuple(
        obstacle.point
        for obstacle in obstacles
        if obstacle.valid
        and obstacle.source_stamp_ns > 0
        and all(math.isfinite(value) for value in obstacle.point)
    )


def _bbox_from_fields(
    fields: PersonTargetFields[HeaderT] | PersonObservationFields[HeaderT],
) -> tuple[float, float, float, float]:
    """Convert position-size message fields into an association bbox."""
    return (
        fields.bbox_x,
        fields.bbox_y,
        fields.bbox_x + fields.bbox_width,
        fields.bbox_y + fields.bbox_height,
    )


def _valid_detection(
    track_id: int,
    confidence: float,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
) -> bool:
    """Check finite positive detection fields at the message boundary."""
    return (
        track_id >= 0
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
        and all(
            math.isfinite(value)
            for value in (bbox_x, bbox_y, bbox_width, bbox_height)
        )
        and bbox_width > 0.0
        and bbox_height > 0.0
    )
