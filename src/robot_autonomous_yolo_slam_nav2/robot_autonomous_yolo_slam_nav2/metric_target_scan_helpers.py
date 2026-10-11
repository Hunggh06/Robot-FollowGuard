"""Scan, calibration, and transform helpers for metric association."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .metric_association import (
    BaseToCameraTransform,
    BoundingBox,
    CameraCalibration,
    ImageSize,
    LidarPoint,
    Point3D,
    RotationMatrix,
)


@dataclass(frozen=True, slots=True)
class LaserScanFields:
    """The LaserScan values required for polar conversion."""

    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LaserToBaseTransform:
    """Rigid transform mapping laser-frame points into base_link."""

    rotation: RotationMatrix
    translation: Point3D

    def to_base(self, point_laser: Point3D) -> Point3D:
        """Transform one point from the laser frame into base_link."""
        x, y, z = point_laser
        row_x, row_y, row_z = self.rotation
        tx, ty, tz = self.translation
        return (
            row_x[0] * x + row_x[1] * y + row_x[2] * z + tx,
            row_y[0] * x + row_y[1] * y + row_y[2] * z + ty,
            row_z[0] * x + row_z[1] * y + row_z[2] * z + tz,
        )


@dataclass(frozen=True, slots=True)
class CameraInfoFields:
    """The CameraInfo values required to parse pinhole calibration."""

    width: int
    height: int
    k: tuple[float, ...]
    d: tuple[float, ...] = ()
    distortion_model: str = ""


@dataclass(frozen=True, slots=True)
class ParsedCameraInfo:
    """Validated camera calibration and image dimensions."""

    calibration: CameraCalibration
    image_size: ImageSize


def timestamps_within_skew(
    first_stamp_ns: int,
    second_stamp_ns: int,
    max_skew_s: float,
) -> bool:
    return (
        first_stamp_ns > 0
        and second_stamp_ns > 0
        and math.isfinite(max_skew_s)
        and max_skew_s >= 0.0
        and abs(first_stamp_ns - second_stamp_ns) <= max_skew_s * 1_000_000_000
    )


def laser_scan_to_lidar_points(
    scan: LaserScanFields,
    laser_to_base: LaserToBaseTransform,
) -> tuple[LidarPoint, ...]:
    """Convert finite, in-range polar samples into base-link LiDAR points."""
    scan_values = (
        scan.angle_min,
        scan.angle_increment,
        scan.range_min,
        scan.range_max,
    )
    if (
        not all(math.isfinite(value) for value in scan_values)
        or scan.range_min < 0.0
        or scan.range_max < scan.range_min
    ):
        return ()

    points: list[LidarPoint] = []
    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        if (
            not math.isfinite(distance)
            or not math.isfinite(angle)
            or distance < scan.range_min
            or distance > scan.range_max
        ):
            continue
        point_base = laser_to_base.to_base(
            (distance * math.cos(angle), distance * math.sin(angle), 0.0),
        )
        if all(math.isfinite(value) for value in point_base):
            points.append(LidarPoint(point_base))
    return tuple(points)


def parse_camera_info(
    info: CameraInfoFields,
    fallback_image_size: ImageSize | None = None,
) -> ParsedCameraInfo | None:
    """Parse CameraInfo K values without inventing missing intrinsics."""
    if len(info.k) < 9:
        return None
    width, height = info.width, info.height
    if width <= 0 or height <= 0:
        if fallback_image_size is None:
            return None
        width, height = fallback_image_size
    fx, cx = info.k[0], info.k[2]
    fy, cy = info.k[4], info.k[5]
    if (
        width <= 0
        or height <= 0
        or not all(math.isfinite(value) for value in (fx, fy, cx, cy))
        or fx <= 0.0
        or fy <= 0.0
    ):
        return None
    return ParsedCameraInfo(CameraCalibration(fx, fy, cx, cy), (width, height))


def is_fresh(stamp_ns: int, now_ns: int, max_data_age_s: float) -> bool:
    """Return whether a timestamp is present, not future-dated, and recent."""
    if not math.isfinite(max_data_age_s) or max_data_age_s < 0.0:
        return False
    if stamp_ns <= 0 or now_ns < stamp_ns:
        return False
    return now_ns - stamp_ns <= int(max_data_age_s * 1_000_000_000.0)


def quaternion_to_rotation(
    x: float,
    y: float,
    z: float,
    w: float,
) -> RotationMatrix | None:
    """Normalize a quaternion and return its rotation matrix when valid."""
    values = (x, y, z, w)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    x_n, y_n, z_n, w_n = (value / norm for value in values)
    return (
        (
            1.0 - 2.0 * (y_n * y_n + z_n * z_n),
            2.0 * (x_n * y_n - z_n * w_n),
            2.0 * (x_n * z_n + y_n * w_n),
        ),
        (
            2.0 * (x_n * y_n + z_n * w_n),
            1.0 - 2.0 * (x_n * x_n + z_n * z_n),
            2.0 * (y_n * z_n - x_n * w_n),
        ),
        (
            2.0 * (x_n * z_n - y_n * w_n),
            2.0 * (y_n * z_n + x_n * w_n),
            1.0 - 2.0 * (x_n * x_n + y_n * y_n),
        ),
    )


def bearing_window_point(
    bbox: BoundingBox,
    calibration: CameraCalibration,
    base_to_camera: BaseToCameraTransform,
    image_size: ImageSize,
    scan: LaserScanFields,
    laser_to_base: LaserToBaseTransform,
    half_window_rad: float,
) -> Point3D | None:
    """Return the nearest scan point near the bbox-center bearing.

    Fallback when no projected point lands inside the bbox (fragile
    TF/calibration/timing): unproject the bbox center to a base-frame
    ray, then take the nearest valid scan return within ``half_window_rad``
    of that bearing. Returns None when nothing usable is found.
    """
    left, top, right, bottom = bbox
    width, height = image_size
    if not all(
        math.isfinite(value)
        for value in (left, top, right, bottom, float(width), float(height))
    ):
        return None
    if (
        right <= left
        or bottom <= top
        or width <= 0
        or height <= 0
        or calibration.fx <= 0.0
        or calibration.fy <= 0.0
        or not math.isfinite(half_window_rad)
        or half_window_rad <= 0.0
    ):
        return None
    ray_cam = (
        ((left + right) / 2.0 - calibration.cx) / calibration.fx,
        ((top + bottom) / 2.0 - calibration.cy) / calibration.fy,
        1.0,
    )
    rows = base_to_camera.rotation
    ray_base = (
        rows[0][0] * ray_cam[0] + rows[1][0] * ray_cam[1] + rows[2][0] * ray_cam[2],
        rows[0][1] * ray_cam[0] + rows[1][1] * ray_cam[1] + rows[2][1] * ray_cam[2],
    )
    if not all(math.isfinite(value) for value in ray_base):
        return None
    if ray_base[0] == 0.0 and ray_base[1] == 0.0:
        return None
    bearing = math.atan2(ray_base[1], ray_base[0])
    increment = scan.angle_increment
    count = len(scan.ranges)
    if (
        count == 0
        or not math.isfinite(scan.angle_min)
        or not math.isfinite(increment)
        or increment <= 0.0
        or not math.isfinite(scan.range_min)
        or not math.isfinite(scan.range_max)
        or scan.range_max < scan.range_min
    ):
        return None
    center = int(round((bearing - scan.angle_min) / increment))
    span = max(1, int(math.ceil(half_window_rad / increment)))
    best: tuple[float, int] | None = None
    for index in range(max(0, center - span), min(count, center + span + 1)):
        distance = scan.ranges[index]
        if (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
            and (best is None or distance < best[0])
        ):
            best = (distance, index)
    if best is None:
        return None
    distance, index = best
    angle = scan.angle_min + index * increment
    point = laser_to_base.to_base(
        (distance * math.cos(angle), distance * math.sin(angle), 0.0),
    )
    if not all(math.isfinite(value) for value in point):
        return None
    return point
