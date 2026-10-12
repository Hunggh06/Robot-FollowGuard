"""Pure tracker-row and target-message projections for metric association."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar

from .target_state import TargetSelectionResult, TrackId, TrackObservation

TrackerRow: TypeAlias = Sequence[float]
TrackerRows: TypeAlias = Sequence[TrackerRow]
Point3D: TypeAlias = tuple[float, float, float]
HeaderT = TypeVar("HeaderT")


class _PointMessage(Protocol):
    x: float
    y: float
    z: float


class _ObservationMessage(Protocol[HeaderT]):
    header: HeaderT
    track_id: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    state: str
    valid: bool


class _TargetMessage(Protocol[HeaderT]):
    header: HeaderT
    track_id: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    state: str
    valid: bool
    point_base_link: _PointMessage
    range_m: float
    bearing_rad: float


@dataclass(frozen=True, slots=True)
class PersonObservationFields(Generic[HeaderT]):
    """Fields for one raw tracked person."""

    header: HeaderT
    track_id: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    state: str
    valid: bool


@dataclass(frozen=True, slots=True)
class PersonTargetFields(Generic[HeaderT]):
    """Fields at the camera-to-metric target association boundary."""

    header: HeaderT
    track_id: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    state: str
    valid: bool
    point_base_link: Point3D
    range_m: float
    bearing_rad: float


def tracker_rows_to_observations(rows: TrackerRows) -> tuple[TrackObservation, ...]:
    """Convert BoT-SORT rows into finite person observations."""
    observations: list[TrackObservation] = []
    for row in rows:
        try:
            if len(row) < 7:
                continue
            values = tuple(float(row[index]) for index in range(7))
        except (IndexError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        x1, y1, x2, y2, track_id, confidence, class_id = values
        if class_id != 0.0 or not track_id.is_integer():
            continue
        if x2 < x1 or y2 < y1 or not 0.0 <= confidence <= 1.0:
            continue
        observations.append(
            TrackObservation(
                track_id=TrackId(int(track_id)),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            ),
        )
    return tuple(observations)


def _observation_fields(
    observation: TrackObservation,
    header: HeaderT,
    state: str,
) -> PersonObservationFields[HeaderT]:
    x1, y1, x2, y2 = observation.bbox
    return PersonObservationFields(
        header=header,
        track_id=observation.track_id,
        confidence=observation.confidence,
        bbox_x=x1,
        bbox_y=y1,
        bbox_width=x2 - x1,
        bbox_height=y2 - y1,
        state=state,
        valid=True,
    )


def selection_to_message_fields(
    result: TargetSelectionResult,
    header: HeaderT,
) -> tuple[tuple[PersonObservationFields[HeaderT], ...], PersonTargetFields[HeaderT]]:
    """Project selection output into raw and metric-boundary fields."""
    state = result.state.value
    people = tuple(
        _observation_fields(observation, header, state)
        for observation in (*result.non_target_observations, result.target_observation)
        if observation is not None
    )
    target = result.target_observation
    if target is None:
        target_id = 0
        confidence = 0.0
        bbox_x = 0.0
        bbox_y = 0.0
        bbox_width = 0.0
        bbox_height = 0.0
    else:
        x1, y1, x2, y2 = target.bbox
        target_id = target.track_id
        confidence = target.confidence
        bbox_x = x1
        bbox_y = y1
        bbox_width = x2 - x1
        bbox_height = y2 - y1
    return people, PersonTargetFields(
        header=header,
        track_id=target_id,
        confidence=confidence,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        state=state,
        valid=False,
        point_base_link=(0.0, 0.0, 0.0),
        range_m=0.0,
        bearing_rad=0.0,
    )


def selection_to_ros_messages(
    result: TargetSelectionResult,
    header: HeaderT,
) -> tuple[tuple[_ObservationMessage[HeaderT], ...], _TargetMessage[HeaderT]]:
    """Create generated ROS messages at the runtime boundary."""
    generated = importlib.import_module("robot_autonomous_yolo_slam_nav2.msg")
    people_fields, target_fields = selection_to_message_fields(result, header)
    people: list[_ObservationMessage[HeaderT]] = []
    for fields in people_fields:
        message: _ObservationMessage[HeaderT] = getattr(generated, "PersonObservation")()
        message.header = fields.header
        message.track_id = fields.track_id
        message.confidence = fields.confidence
        message.bbox_x = fields.bbox_x
        message.bbox_y = fields.bbox_y
        message.bbox_width = fields.bbox_width
        message.bbox_height = fields.bbox_height
        message.state = fields.state
        message.valid = fields.valid
        people.append(message)
    target: _TargetMessage[HeaderT] = getattr(generated, "PersonTarget")()
    target.header = target_fields.header
    target.track_id = target_fields.track_id
    target.confidence = target_fields.confidence
    target.bbox_x = target_fields.bbox_x
    target.bbox_y = target_fields.bbox_y
    target.bbox_width = target_fields.bbox_width
    target.bbox_height = target_fields.bbox_height
    target.state = target_fields.state
    target.valid = target_fields.valid
    target.point_base_link.x = target_fields.point_base_link[0]
    target.point_base_link.y = target_fields.point_base_link[1]
    target.point_base_link.z = target_fields.point_base_link[2]
    target.range_m = target_fields.range_m
    target.bearing_rad = target_fields.bearing_rad
    return tuple(people), target
