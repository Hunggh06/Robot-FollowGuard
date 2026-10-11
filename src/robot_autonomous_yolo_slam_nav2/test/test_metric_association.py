import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.metric_association import (  # noqa: E402
    BaseToCameraTransform,
    CameraCalibration,
    LidarPoint,
    MetricAssociationResult,
    MetricPersonObstacle,
    MetricPersonObservation,
    associate_metric_observations,
)
from robot_autonomous_yolo_slam_nav2.target_state import (  # noqa: E402
    BoundingBox,
    TargetSelectionResult,
    TargetState,
    TrackId,
    TrackObservation,
)


CALIBRATION = CameraCalibration(fx=100.0, fy=100.0, cx=320.0, cy=240.0)
IDENTITY_TRANSFORM = BaseToCameraTransform(
    rotation=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation=(0.0, 0.0, 0.0),
)
IMAGE_SIZE = (640, 480)


def _observation(track_id: int, bbox: BoundingBox) -> TrackObservation:
    return TrackObservation(track_id=TrackId(track_id), confidence=0.9, bbox=bbox)


def _selection(
    target: TrackObservation | None,
    non_targets: tuple[TrackObservation, ...] = (),
) -> TargetSelectionResult:
    return TargetSelectionResult(
        state=TargetState.TRACKING if target is not None else TargetState.SEARCHING,
        target_observation=target,
        non_target_observations=non_targets,
        forward_motion_allowed=target is not None,
    )


def _point(x: float, y: float, z: float) -> LidarPoint:
    return LidarPoint(point_base_link=(x, y, z))


def test_association_selects_nearest_target_point_deterministically() -> None:
    # Given a locked target whose bbox contains two valid projected LiDAR points.
    target = _observation(7, (300.0, 220.0, 340.0, 260.0))
    points = (_point(0.5, -0.8, 4.0), _point(1.0, -0.6, 8.0))

    # When the target is associated with the scan.
    result = associate_metric_observations(
        _selection(target), points, CALIBRATION, IDENTITY_TRANSFORM, IMAGE_SIZE, 42
    )

    # Then the nearest point is returned with the target identity.
    assert result.target_observation == MetricPersonObservation(
        track_id=TrackId(7), point_base_link=(0.5, -0.8, 4.0), source_stamp_ns=42
    )
    assert result.non_target_obstacles == ()


def test_association_separates_target_and_non_target_people() -> None:
    # Given distinct target and non-target bboxes.
    target = _observation(7, (300.0, 220.0, 340.0, 260.0))
    non_target = _observation(9, (350.0, 220.0, 390.0, 260.0))
    points = (_point(0.5, -0.8, 4.0), _point(1.0, 0.0, 2.0))

    # When both observations are associated with the same scan.
    result = associate_metric_observations(
        _selection(target, (non_target,)),
        points,
        CALIBRATION,
        IDENTITY_TRANSFORM,
        IMAGE_SIZE,
        100,
    )

    # Then only the locked track is the target and the other is an obstacle.
    assert result.target_observation is not None
    assert result.target_observation.track_id == TrackId(7)
    assert result.target_observation.point_base_link == (0.5, -0.8, 4.0)
    assert result.non_target_obstacles == (
        MetricPersonObstacle(
            track_id=TrackId(9), point_base_link=(1.0, 0.0, 2.0), source_stamp_ns=100
        ),
    )


@pytest.mark.parametrize(
    "point",
    (_point(0.0, 0.0, 0.0), _point(0.0, 0.0, -1.0), _point(4.0, 0.0, 1.0)),
)
def test_association_rejects_invalid_depth_and_out_of_image_points(
    point: LidarPoint,
) -> None:
    # Given a target bbox and a point that cannot produce a valid image pixel.
    target = _observation(7, (300.0, 220.0, 340.0, 260.0))

    # When the invalid point is associated.
    result = associate_metric_observations(
        _selection(target), (point,), CALIBRATION, IDENTITY_TRANSFORM, IMAGE_SIZE, 42
    )

    # Then no target metric point is produced.
    assert result.target_observation is None


def test_association_omits_bbox_without_a_lidar_association() -> None:
    # Given a target bbox that does not contain the scan point's projection.
    target = _observation(7, (300.0, 220.0, 340.0, 260.0))

    # When the scan point is associated.
    result = associate_metric_observations(
        _selection(target), (_point(2.0, 0.0, 1.0),), CALIBRATION, IDENTITY_TRANSFORM, IMAGE_SIZE, 42
    )

    # Then the unassociated target is invalid and omitted.
    assert result.target_observation is None
    assert result.non_target_obstacles == ()


def test_association_preserves_source_stamp_on_result_and_points() -> None:
    # Given target and non-target observations with valid projected points.
    target = _observation(7, (300.0, 220.0, 340.0, 260.0))
    non_target = _observation(9, (350.0, 220.0, 390.0, 260.0))

    # When the scan carries a source stamp.
    result = associate_metric_observations(
        _selection(target, (non_target,)),
        (_point(0.5, -0.8, 4.0), _point(1.0, 0.0, 2.0)),
        CALIBRATION,
        IDENTITY_TRANSFORM,
        IMAGE_SIZE,
        987654321,
    )

    # Then the adapter preserves it for downstream freshness checks.
    assert isinstance(result, MetricAssociationResult)
    assert result.source_stamp_ns == 987654321
    assert result.target_observation is not None
    assert result.target_observation.source_stamp_ns == 987654321
    assert result.non_target_obstacles[0].source_stamp_ns == 987654321


def test_near_point_claimed_by_other_person_is_not_given_to_target() -> None:
    # Given a far target whose bbox also contains a near stranger's returns.
    target = _observation(7, (400.0, 200.0, 440.0, 280.0))
    stranger = _observation(9, (390.0, 220.0, 410.0, 270.0))
    near = _point(0.85, -0.08, 1.0)
    near_own = _point(1.5, 0.2, 2.0)
    far = _point(1.0, -0.1, 1.0)

    # When associated together.
    result = associate_metric_observations(
        _selection(target, (stranger,)),
        (near, near_own, far),
        CALIBRATION,
        IDENTITY_TRANSFORM,
        IMAGE_SIZE,
        42,
    )

    # Then the target keeps its own farther point, stranger keeps the near one.
    assert result.target_observation is not None
    assert result.target_observation.point_base_link == (1.0, -0.1, 1.0)
    assert result.non_target_obstacles == (
        MetricPersonObstacle(
            track_id=TrackId(9), point_base_link=(1.5, 0.2, 2.0), source_stamp_ns=42
        ),
    )


def test_returns_far_from_bbox_implied_depth_are_rejected() -> None:
    # Given a small box implying a ~2.25 m person and a near return inside it.
    target = _observation(7, (300.0, 200.0, 320.0, 260.0))
    points = (_point(0.3, -0.6, 3.0),)

    # When the only in-box return is much nearer than the implied depth.
    result = associate_metric_observations(
        _selection(target), points, CALIBRATION, IDENTITY_TRANSFORM, IMAGE_SIZE, 42
    )

    # Then no metric point is produced instead of attaching clutter.
    assert result.target_observation is None
