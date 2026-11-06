"""Pure target-message field projections shared by metric association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

Point3D: TypeAlias = tuple[float, float, float]
HeaderT = TypeVar("HeaderT")


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