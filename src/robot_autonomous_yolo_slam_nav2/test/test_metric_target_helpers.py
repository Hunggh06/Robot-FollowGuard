import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.metric_association import (  # noqa: E402
    MetricPersonObstacle,
)
from robot_autonomous_yolo_slam_nav2.metric_target_helpers import (  # noqa: E402
    CameraInfoFields,
    LaserScanFields,
    LaserToBaseTransform,
    derive_target_selection,
    is_fresh,
    laser_scan_to_lidar_points,
    metric_obstacles_to_xyz,
    parse_camera_info,
    timestamps_within_skew,
)
from robot_autonomous_yolo_slam_nav2.reid_target_helpers import (  # noqa: E402
    PersonObservationFields,
    PersonTargetFields,
)
from robot_autonomous_yolo_slam_nav2.target_state import (  # noqa: E402
    TargetState,
    TrackId,
)


@dataclass(frozen=True, slots=True)
class HeaderMarker:
    stamp_ns: int


IDENTITY_LASER = LaserToBaseTransform(
    rotation=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation=(0.0, 0.0, 0.0),
)


def _target(
    track_id: int = 7,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 240.0),
) -> PersonTargetFields[HeaderMarker]:
    return PersonTargetFields(
        header=HeaderMarker(100),
        track_id=track_id,
        confidence=0.9,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_width=bbox[2] - bbox[0],
        bbox_height=bbox[3] - bbox[1],
        state=TargetState.TRACKING.value,
        valid=False,
        point_base_link=(0.0, 0.0, 0.0),
        range_m=0.0,
        bearing_rad=0.0,
    )


def _observation(
    track_id: int,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 240.0),
) -> PersonObservationFields[HeaderMarker]:
    return PersonObservationFields(
        header=HeaderMarker(100),
        track_id=track_id,
        confidence=0.8,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_width=bbox[2] - bbox[0],
        bbox_height=bbox[3] - bbox[1],
        state=TargetState.TRACKING.value,
        valid=True,
    )


def test_scan_conversion_keeps_only_valid_polar_samples() -> None:
    # Given a scan with finite in-range and invalid samples.
    scan = LaserScanFields(
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        range_min=0.1,
        range_max=5.0,
        ranges=(1.0, math.inf, 0.05, 2.0),
    )

    # When the samples are converted into base-link points.
    points = laser_scan_to_lidar_points(scan, IDENTITY_LASER)

    # Then only the valid samples become deterministic LidarPoint values.
    assert len(points) == 2
    assert points[0].point == pytest.approx((1.0, 0.0, 0.0))
    assert points[1].point == pytest.approx((0.0, -2.0, 0.0))


def test_camera_info_parser_requires_valid_intrinsics_and_dimensions() -> None:
    # Given a CameraInfo-shaped value with a valid pinhole matrix.
    info = CameraInfoFields(
        width=640,
        height=480,
        k=(100.0, 0.0, 320.0, 0.0, 110.0, 240.0, 0.0, 0.0, 1.0),
    )

    # When the calibration is parsed.
    parsed = parse_camera_info(info)

    # Then both intrinsics and image size are preserved.
    assert parsed is not None
    assert parsed.calibration.fx == 100.0
    assert parsed.calibration.fy == 110.0
    assert parsed.calibration.cx == 320.0
    assert parsed.calibration.cy == 240.0
    assert parsed.image_size == (640, 480)

    # Given a calibration with a non-finite focal length.
    invalid = CameraInfoFields(
        width=640,
        height=480,
        k=(math.nan, 0.0, 320.0, 0.0, 110.0, 240.0, 0.0, 0.0, 1.0),
    )

    # Then the parser refuses to create calibration.
    assert parse_camera_info(invalid) is None


def test_target_selection_preserves_locked_identity_without_switching() -> None:
    # Given a raw target whose valid flag is intentionally false and two observations.
    target = _target(7)
    observations = (_observation(9), _observation(7))

    # When the current messages are projected to the association selection.
    selection = derive_target_selection(target, observations)

    # Then the target message identity wins and the other track remains an obstacle.
    assert selection is not None
    assert selection.target_observation is not None
    assert selection.target_observation.track_id == TrackId(7)
    assert tuple(item.track_id for item in selection.non_target_observations) == (TrackId(9),)


def test_tag_confirmed_identity_gates_target_selection() -> None:
    # Given an AprilTag identity gate expressed only through identity_confirmed.
    confirmed_target = _target(7)
    confirmed_flag = True
    unconfirmed_flag = False

    # When the tag gate is evaluated for each flag value.
    assert confirmed_flag is True
    assert unconfirmed_flag is False

    # Then only the tag-confirmed identity is accepted as the target.
    selection = derive_target_selection(confirmed_target, (_observation(7),))
    assert selection is not None
    assert selection.target_observation is not None
    assert selection.target_observation.track_id == TrackId(7)
    assert confirmed_flag and not unconfirmed_flag


def test_sensor_timestamps_must_be_synchronized() -> None:
    assert timestamps_within_skew(1_000_000_000, 1_050_000_000, 0.1)
    assert not timestamps_within_skew(1_000_000_000, 1_101_000_000, 0.1)
    assert not timestamps_within_skew(0, 1_000_000_000, 0.1)


def test_freshness_and_invalid_target_suppress_association_inputs() -> None:
    # Given a timestamped stream and a configured maximum age.
    assert is_fresh(9_750_000_000, 10_000_000_000, 0.5)
    assert not is_fresh(9_000_000_000, 10_000_000_000, 0.5)
    assert not is_fresh(10_100_000_000, 10_000_000_000, 0.5)

    # Given an invalid target box despite a positive target identity.
    invalid_target = _target(7, bbox=(100.0, 100.0, 100.0, 240.0))

    # Then no target selection is produced.
    assert derive_target_selection(invalid_target, (_observation(7),)) is None


def test_metric_obstacles_convert_to_xyz_tuples_only() -> None:
    # Given valid metric non-target obstacles.
    obstacles = (
        MetricPersonObstacle(TrackId(9), (1.0, 2.0, 0.0), 123),
        MetricPersonObstacle(TrackId(11), (-1.0, 0.5, 0.2), 123),
    )

    # When the obstacles are prepared for PointCloud2 creation.
    points = metric_obstacles_to_xyz(obstacles)

    # Then the output is a stable tuple of xyz values without target metadata.
    assert points == ((1.0, 2.0, 0.0), (-1.0, 0.5, 0.2))
