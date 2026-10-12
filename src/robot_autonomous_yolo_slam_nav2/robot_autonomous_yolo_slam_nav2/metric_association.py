"""Associate camera person boxes with LiDAR points in ``base_link``."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .target_state import BoundingBox, TargetSelectionResult, TrackId, TrackObservation

Point3D: TypeAlias = tuple[float, float, float]
RotationMatrix: TypeAlias = tuple[Point3D, Point3D, Point3D]
Pixel: TypeAlias = tuple[float, float]
ImageSize: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class BaseToCameraTransform:
    """Rigid transform with ``camera = rotation * base_link + translation``."""

    rotation: RotationMatrix
    translation: Point3D

    def to_camera(self, point_base_link: Point3D) -> Point3D:
        """Transform one base-link point into the camera frame."""
        x, y, z = point_base_link
        row_x, row_y, row_z = self.rotation
        tx, ty, tz = self.translation
        return (
            row_x[0] * x + row_x[1] * y + row_x[2] * z + tx,
            row_y[0] * x + row_y[1] * y + row_y[2] * z + ty,
            row_z[0] * x + row_z[1] * y + row_z[2] * z + tz,
        )


@dataclass(frozen=True, slots=True)
class LidarPoint:
    """One LiDAR point expressed in the ``base_link`` frame."""

    point_base_link: Point3D

    @property
    def point(self) -> Point3D:
        """Return the point using the short name used by metric consumers."""
        return self.point_base_link


@dataclass(frozen=True, slots=True)
class MetricPersonObservation:
    """A valid metric point for one tracked person."""

    track_id: TrackId
    point_base_link: Point3D
    source_stamp_ns: int

    @property
    def point(self) -> Point3D:
        return self.point_base_link

    @property
    def range_m(self) -> float:
        x, y, _ = self.point_base_link
        return math.hypot(x, y)

    @property
    def bearing_rad(self) -> float:
        x, y, _ = self.point_base_link
        return math.atan2(y, x)

    @property
    def valid(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class MetricPersonObstacle:
    """A valid metric point for a non-target person."""

    track_id: TrackId
    point_base_link: Point3D
    source_stamp_ns: int

    @property
    def point(self) -> Point3D:
        return self.point_base_link

    @property
    def range_m(self) -> float:
        x, y, _ = self.point_base_link
        return math.hypot(x, y)

    @property
    def bearing_rad(self) -> float:
        x, y, _ = self.point_base_link
        return math.atan2(y, x)

    @property
    def valid(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class MetricAssociationResult:
    """Target and non-target metric points from one timestamped scan."""

    target_observation: MetricPersonObservation | None
    non_target_obstacles: tuple[MetricPersonObstacle, ...]
    source_stamp_ns: int

    @property
    def target(self) -> MetricPersonObservation | None:
        """Return the target observation using the short consumer name."""
        return self.target_observation

    @property
    def obstacles(self) -> tuple[MetricPersonObstacle, ...]:
        """Return non-target obstacles using the short consumer name."""
        return self.non_target_obstacles


def project_point(
    point_base_link: Point3D,
    calibration: CameraCalibration,
    base_to_camera: BaseToCameraTransform,
    image_size: ImageSize,
) -> Pixel | None:
    """Project a base-link point into a valid half-open image pixel."""
    width, height = image_size
    if width <= 0 or height <= 0 or calibration.fx <= 0 or calibration.fy <= 0:
        return None
    if not all(math.isfinite(value) for value in point_base_link):
        return None
    camera_x, camera_y, camera_z = base_to_camera.to_camera(point_base_link)
    if camera_z <= 0 or not all(
        math.isfinite(value) for value in (camera_x, camera_y, camera_z)
    ):
        return None
    pixel = (
        calibration.fx * camera_x / camera_z + calibration.cx,
        calibration.fy * camera_y / camera_z + calibration.cy,
    )
    if not all(math.isfinite(value) for value in pixel):
        return None
    pixel_x, pixel_y = pixel
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        return None
    return pixel


def _point_in_bbox(pixel: Pixel, bbox: BoundingBox) -> bool:
    pixel_x, pixel_y = pixel
    left, top, right, bottom = bbox
    return left <= pixel_x < right and top <= pixel_y < bottom


def _nearest_point(
    observation: TrackObservation,
    lidar_points: Sequence[LidarPoint],
    calibration: CameraCalibration,
    base_to_camera: BaseToCameraTransform,
    image_size: ImageSize,
    exclude_bboxes: tuple[BoundingBox, ...] = (),
    expected_width_m: float = 0.45,
    expected_near_factor: float = 0.6,
    expected_far_factor: float = 2.5,
) -> Point3D | None:
    left, _top, right, _bottom = observation.bbox
    width_px = right - left
    depth_lo, depth_hi = 0.0, math.inf
    if (
        math.isfinite(width_px)
        and width_px > 0.0
        and math.isfinite(calibration.fx)
        and calibration.fx > 0.0
        and math.isfinite(expected_width_m)
        and expected_width_m > 0.0
    ):
        expected = calibration.fx * expected_width_m / width_px
        if math.isfinite(expected) and expected > 0.0:
            depth_lo, depth_hi = (
                expected * expected_near_factor,
                expected * expected_far_factor,
            )
    candidates = []
    for index, lidar_point in enumerate(lidar_points):
        pixel = project_point(
            lidar_point.point_base_link, calibration, base_to_camera, image_size
        )
        if pixel is None or not _point_in_bbox(pixel, observation.bbox):
            continue
        if any(_point_in_bbox(pixel, other) for other in exclude_bboxes):
            continue
        x, y, _ = lidar_point.point_base_link
        distance = math.hypot(x, y)
        if not depth_lo <= distance <= depth_hi:
            continue
        candidates.append((x * x + y * y, index, lidar_point.point_base_link))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def associate_metric_observations(
    selection: TargetSelectionResult,
    lidar_points: Sequence[LidarPoint],
    calibration: CameraCalibration,
    base_to_camera: BaseToCameraTransform,
    image_size: ImageSize,
    source_stamp_ns: int,
) -> MetricAssociationResult:
    """Return valid target and non-target metric points for one scan."""
    if source_stamp_ns <= 0:
        return MetricAssociationResult(None, (), source_stamp_ns)

    target_observation = None
    if selection.target_observation is not None:
        others = tuple(
            observation.bbox for observation in selection.non_target_observations
        )
        point = _nearest_point(
            selection.target_observation, lidar_points, calibration, base_to_camera, image_size,
            exclude_bboxes=others,
        )
        if point is not None:
            target_observation = MetricPersonObservation(
                selection.target_observation.track_id, point, source_stamp_ns
            )
    obstacles = tuple(
        MetricPersonObstacle(observation.track_id, point, source_stamp_ns)
        for observation in selection.non_target_observations
        for point in (
            _nearest_point(
                observation, lidar_points, calibration, base_to_camera, image_size,
                exclude_bboxes=tuple(
                    other.bbox
                    for other in (selection.target_observation,)
                    if other is not None
                ) + tuple(
                    another.bbox
                    for another in selection.non_target_observations
                    if another.track_id != observation.track_id
                ),
            ),
        )
        if point is not None
    )
    return MetricAssociationResult(target_observation, obstacles, source_stamp_ns)


CameraIntrinsics = CameraCalibration
RigidTransform = BaseToCameraTransform
MetricObservation = MetricPersonObservation
AssociationResult = MetricAssociationResult
associate_lidar_points = associate_metric_observations
