import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.metric_association import (  # noqa: E402
    BaseToCameraTransform,
    CameraCalibration,
)
from robot_autonomous_yolo_slam_nav2.metric_target_scan_helpers import (  # noqa: E402
    LaserScanFields,
    LaserToBaseTransform,
    bearing_window_point,
)

CALIBRATION = CameraCalibration(fx=136.92, fy=136.78, cx=159.02, cy=118.77)
OPTICAL = ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
CAMERA = BaseToCameraTransform(OPTICAL, (-0.055, 0.063, 0.177))
IDENTITY_LASER = LaserToBaseTransform(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    (0.0, 0.0, 0.0),
)


def _scan(ranges, angle_min=-math.pi):
    return LaserScanFields(
        angle_min=angle_min,
        angle_increment=math.pi / 180.0,
        range_min=0.15,
        range_max=6.0,
        ranges=tuple(ranges),
    )


def test_returns_nearest_return_inside_bearing_window() -> None:
    ranges = [float("inf")] * 360
    ranges[176] = 2.5
    ranges[183] = 1.2
    scan = _scan(ranges, angle_min=-math.pi)
    bbox = (0.0, 0.0, 320.0, 240.0)
    point = bearing_window_point(
        bbox, CALIBRATION, CAMERA, (320, 240), scan, IDENTITY_LASER, 0.35,
    )
    assert point is not None
    assert math.hypot(point[0], point[1]) == 1.2 or abs(
        math.hypot(point[0], point[1]) - 1.2
    ) < 1e-6


def test_returns_none_when_window_empty() -> None:
    scan = _scan([float("inf")] * 360)
    assert (
        bearing_window_point(
            (0.0, 0.0, 320.0, 240.0),
            CALIBRATION,
            CAMERA,
            (320, 240),
            scan,
            IDENTITY_LASER,
            0.35,
        )
        is None
    )


def test_returns_none_for_degenerate_inputs() -> None:
    scan = _scan([1.0] * 360)
    assert (
        bearing_window_point(
            (10.0, 10.0, 5.0, 5.0),
            CALIBRATION,
            CAMERA,
            (320, 240),
            scan,
            IDENTITY_LASER,
            0.35,
        )
        is None
    )
    assert (
        bearing_window_point(
            (0.0, 0.0, 320.0, 240.0),
            CALIBRATION,
            CAMERA,
            (320, 240),
            scan,
            IDENTITY_LASER,
            0.0,
        )
        is None
    )
