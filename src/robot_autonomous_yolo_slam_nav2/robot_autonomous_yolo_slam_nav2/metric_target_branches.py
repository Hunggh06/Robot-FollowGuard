"""Pure LiDAR position decisions for the metric target runtime.

No ROS I/O here: the runtime builds the typed inputs (selection and LiDAR
points) and publishes the returned decision. People obstacles come only from
projected non-target observations, never from raw scan points: the target's
own legs and walls must not appear in the person cloud and freeze the
follower. There is no camera range estimation: only LiDAR geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metric_association import (
    BaseToCameraTransform,
    CameraCalibration,
    ImageSize,
    LidarPoint,
    MetricPersonObservation,
    Point3D,
    associate_metric_observations,
)
from .metric_target_scan_helpers import is_fresh, timestamps_within_skew
from .metric_target_selection_helpers import metric_obstacles_to_xyz
from .target_state import TargetSelectionResult


@dataclass(frozen=True, slots=True)
class LidarInputs:
    """Converted scan points plus the calibration used to project them."""

    points: tuple[LidarPoint, ...]
    calibration: CameraCalibration
    camera_transform: BaseToCameraTransform
    image_size: ImageSize
    stamp_ns: int
    position_confidence: float


@dataclass(frozen=True, slots=True)
class BranchDecision:
    """One branch outcome: source, geometry, and person-only obstacles."""

    source: str
    observation: MetricPersonObservation | None
    confidence: float
    provenance: str
    obstacles: tuple[Point3D, ...]


def usable_scan_stamp(
    scan_stamp_ns: int,
    target_stamp_ns: int,
    now_ns: int,
    max_scan_age_s: float,
    max_scan_skew_s: float,
) -> int | None:
    """Return the scan stamp when fresh and skew-bounded, else None."""
    if not is_fresh(scan_stamp_ns, now_ns, max_scan_age_s):
        return None
    if not timestamps_within_skew(target_stamp_ns, scan_stamp_ns, max_scan_skew_s):
        return None
    return scan_stamp_ns


def lidar_target_and_obstacles(
    selection: TargetSelectionResult,
    lidar: LidarInputs,
) -> tuple[MetricPersonObservation | None, tuple[Point3D, ...]]:
    """Associate box-projected points; obstacles are non-target people only."""
    result = associate_metric_observations(
        selection,
        lidar.points,
        lidar.calibration,
        lidar.camera_transform,
        lidar.image_size,
        lidar.stamp_ns,
    )
    obstacles = metric_obstacles_to_xyz(result.obstacles)
    return result.target_observation, obstacles


def decide_branch(
    selection: TargetSelectionResult,
    lidar: LidarInputs | None,
) -> BranchDecision:
    """LiDAR geometry when the target box holds fresh scan points.

    Obstacles always come from projected non-target observations, never
    from raw scan points. Anything else reports no geometry.
    """
    obstacles: tuple[Point3D, ...] = ()
    observation: MetricPersonObservation | None = None
    if lidar is not None:
        observation, obstacles = lidar_target_and_obstacles(selection, lidar)
    if observation is not None and lidar is not None:
        return BranchDecision(
            "lidar", observation, lidar.position_confidence,
            "lidar_bbox", obstacles,
        )
    return BranchDecision("none", None, 0.0, "no_geometry", obstacles)
