import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.metric_association import (  # noqa: E402
    BaseToCameraTransform,
    CameraCalibration,
    LidarPoint,
)
from robot_autonomous_yolo_slam_nav2.metric_target_branches import (  # noqa: E402
    LidarInputs,
    decide_branch,
    usable_scan_stamp,
)
from robot_autonomous_yolo_slam_nav2.target_state import (  # noqa: E402
    TargetSelectionResult,
    TargetState,
    TrackId,
    TrackObservation,
)

OPTICAL = ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
CAMERA = BaseToCameraTransform(OPTICAL, (0.063, 0.177, 0.055))
CALIBRATION = CameraCalibration(fx=136.92, fy=136.78, cx=159.02, cy=118.77)


def _selection() -> TargetSelectionResult:
    return TargetSelectionResult(
        state=TargetState.TRACKING,
        target_observation=TrackObservation(
            TrackId(5), 0.9, (150.0, 110.0, 190.0, 150.0)),
        non_target_observations=(
            TrackObservation(TrackId(7), 0.8, (100.0, 110.0, 130.0, 140.0)),
        ),
        forward_motion_allowed=True,
    )


def _lidar() -> LidarInputs:
    return LidarInputs(
        points=(
            LidarPoint((2.0, 0.0, 0.0)),
            LidarPoint((2.5, 0.8, 0.0)),
            LidarPoint((-3.0, 0.0, 0.0)),
        ),
        calibration=CALIBRATION,
        camera_transform=CAMERA,
        image_size=(320, 240),
        stamp_ns=1_000_000_000,
        position_confidence=0.95,
    )


def test_lidar_reports_target_and_excludes_target_and_walls() -> None:
    decision = decide_branch(_selection(), _lidar())
    assert decision.source == "lidar"
    assert decision.observation is not None
    assert decision.observation.track_id == 5
    assert decision.confidence == 0.95
    assert decision.provenance == "lidar_bbox"
    assert decision.obstacles == ((2.5, 0.8, 0.0),)


def test_empty_target_box_reports_none_but_keeps_obstacles() -> None:
    empty_box = LidarInputs(
        points=_lidar().points,
        calibration=CALIBRATION,
        camera_transform=CAMERA,
        image_size=(320, 240),
        stamp_ns=1_000_000_000,
        position_confidence=0.95,
    )
    selection = TargetSelectionResult(
        state=TargetState.TRACKING,
        target_observation=TrackObservation(
            TrackId(5), 0.9, (0.0, 0.0, 10.0, 10.0)),
        non_target_observations=(
            TrackObservation(TrackId(7), 0.8, (100.0, 110.0, 130.0, 140.0)),
        ),
        forward_motion_allowed=True,
    )
    decision = decide_branch(selection, empty_box)
    assert decision.source == "none"
    assert decision.observation is None
    assert decision.obstacles == ((2.5, 0.8, 0.0),)


def test_no_geometry_reports_none_without_obstacle_leak() -> None:
    decision = decide_branch(_selection(), None)
    assert decision.source == "none"
    assert decision.observation is None
    assert decision.confidence == 0.0
    assert decision.obstacles == ()


def test_usable_scan_stamp_gates_age_and_skew() -> None:
    assert usable_scan_stamp(1_000_000_000, 1_000_000_000, 2_000_000_000, 1.5, 1.0)
    assert usable_scan_stamp(1_000_000_000, 1_000_000_000, 9_000_000_000, 1.5, 1.0) is None
    assert usable_scan_stamp(1_000_000_000, 5_000_000_000, 5_100_000_000, 1.5, 1.0) is None
