import math
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
)
from robot_autonomous_yolo_slam_nav2.metric_target_scan_helpers import (  # noqa: E402
    LaserScanFields,
    LaserToBaseTransform,
)
from robot_autonomous_yolo_slam_nav2.target_kalman_filter import (  # noqa: E402
    CameraFusionInputs,
    KalmanFusionConfig,
    TargetKalmanFilter,
    estimate_camera_measurement,
    extract_lidar_measurement,
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
IMAGE_SIZE = (320, 240)
IDENTITY_LASER = LaserToBaseTransform(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    (0.0, 0.0, 0.0),
)


def _selection(
    track_id: int = 5,
    bbox: tuple[float, float, float, float] = (145.0, 40.0, 175.0, 160.0),
) -> TargetSelectionResult:
    return TargetSelectionResult(
        state=TargetState.TRACKING,
        target_observation=TrackObservation(TrackId(track_id), 0.9, bbox),
        non_target_observations=(
            TrackObservation(TrackId(7), 0.8, (90.0, 100.0, 130.0, 150.0)),
        ),
        forward_motion_allowed=True,
    )


def _camera_inputs(
    track_id: int = 5,
    bbox: tuple[float, float, float, float] = (145.0, 40.0, 175.0, 160.0),
    stamp_ns: int = 1_000_000_000,
    foot_valid: bool = True,
) -> CameraFusionInputs:
    return CameraFusionInputs(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        calibration=CALIBRATION,
        camera_transform=CAMERA,
        image_size=IMAGE_SIZE,
        stamp_ns=stamp_ns,
        footpoint_valid=foot_valid,
        footpoint_x=0.5 * (bbox[0] + bbox[2]),
        footpoint_y=bbox[3],
        footpoint_confidence=0.85 if foot_valid else 0.0,
    )


def _lidar_inputs(
    points: tuple[LidarPoint, ...] = (
        LidarPoint((2.0, 0.0, 0.0)),
        LidarPoint((2.5, 0.8, 0.0)),
    ),
    stamp_ns: int = 1_000_000_000,
    scan: LaserScanFields | None = None,
) -> LidarInputs:
    return LidarInputs(
        points=points,
        calibration=CALIBRATION,
        camera_transform=CAMERA,
        image_size=IMAGE_SIZE,
        stamp_ns=stamp_ns,
        position_confidence=0.95,
        scan=scan,
        laser_transform=IDENTITY_LASER if scan is not None else None,
    )


def test_camera_measurement_has_tight_bearing_and_valid_depth() -> None:
    cam = _camera_inputs(bbox=(145.0, 40.0, 175.0, 160.0))
    meas = estimate_camera_measurement(cam)
    assert meas is not None
    assert 1.2 <= meas.range_m <= 2.6
    assert abs(meas.bearing_rad) < 0.15
    # Forward-looking ray: x variance (depth) is much larger than y variance (lateral)
    assert meas.covariance_2x2[0][0] > meas.covariance_2x2[1][1] * 2.0


def test_kalman_fuses_lidar_and_camera_and_bridges_scan_gaps() -> None:
    kf = TargetKalmanFilter()
    sel = _selection(bbox=(145.0, 40.0, 175.0, 160.0))
    lidar = _lidar_inputs(stamp_ns=1_000_000_000)
    cam1 = _camera_inputs(stamp_ns=1_000_000_000)

    d1 = decide_branch(sel, lidar, kalman_filter=kf, camera=cam1)
    assert d1.source == "fused"
    assert d1.observation is not None
    assert abs(d1.observation.range_m - 2.0) < 0.15
    assert d1.obstacles == ((2.5, 0.8, 0.0),)

    # Next camera frame arrives 100ms later while LiDAR scan is temporarily absent;
    # Kalman Filter coasts LiDAR depth while updating lateral bearing from camera.
    cam2 = _camera_inputs(
        bbox=(135.0, 40.0, 165.0, 160.0),
        stamp_ns=1_100_000_000,
    )
    d2 = decide_branch(sel, None, kalman_filter=kf, camera=cam2)
    assert d2.source == "fused"
    assert d2.observation is not None
    assert "kalman_coast_lidar" in d2.provenance
    assert abs(d2.observation.range_m - 2.0) < 0.25
    # Moving bbox left in image shifts bearing toward +y in base_link
    assert d2.observation.point[1] > d1.observation.point[1]


def test_bearing_window_fallback_recovers_lidar_when_bbox_vertically_misses_scan() -> None:
    # Bbox is high in the image so 2D z=0 LiDAR ray does not project inside [top, bottom],
    # but bearing_window_point finds the 2.0m leg return along the camera azimuth.
    high_bbox = (145.0, 10.0, 175.0, 90.0)
    sel = _selection(bbox=high_bbox)
    scan = LaserScanFields(
        angle_min=-0.2,
        angle_increment=0.1,
        range_min=0.1,
        range_max=8.0,
        ranges=(2.0, 2.0, 2.0, 2.0, 2.0),
    )
    lidar = _lidar_inputs(
        points=(LidarPoint((2.5, 0.8, 0.0)),),
        stamp_ns=1_000_000_000,
        scan=scan,
    )
    meas, obstacles = extract_lidar_measurement(sel, lidar)
    assert meas is not None
    assert meas.provenance == "lidar_bearing_window"
    assert abs(meas.range_m - 2.0) < 0.15
    assert obstacles == ((2.5, 0.8, 0.0),)


def test_kalman_resets_when_track_id_changes() -> None:
    kf = TargetKalmanFilter()
    sel5 = _selection(track_id=5)
    d5 = decide_branch(
        sel5,
        _lidar_inputs(stamp_ns=1_000_000_000),
        kalman_filter=kf,
        camera=_camera_inputs(track_id=5, stamp_ns=1_000_000_000),
    )
    assert d5.observation is not None
    assert kf.track_id == 5

    sel8 = _selection(track_id=8)
    d8 = decide_branch(
        sel8,
        None,
        kalman_filter=kf,
        camera=_camera_inputs(track_id=8, stamp_ns=1_100_000_000),
    )
    assert kf.track_id == 8
    assert d8.source == "camera"
    assert not kf.has_lidar_lock


def test_kalman_rejects_background_wall_jump_with_mahalanobis_gate() -> None:
    kf = TargetKalmanFilter(KalmanFusionConfig())
    sel = _selection()
    for step in range(4):
        ts = 1_000_000_000 + step * 100_000_000
        decide_branch(
            sel,
            _lidar_inputs(
                points=(LidarPoint((1.8, 0.0, 0.0)),),
                stamp_ns=ts,
            ),
            kalman_filter=kf,
            camera=_camera_inputs(stamp_ns=ts),
        )
    assert kf.range_m is not None
    assert abs(kf.range_m - 1.8) < 0.1

    # Single-frame LiDAR outlier jumping to a 4.8m background wall
    ts_jump = 1_400_000_000
    d_jump = decide_branch(
        sel,
        _lidar_inputs(
            points=(LidarPoint((4.8, 0.0, 0.0)),),
            stamp_ns=ts_jump,
        ),
        kalman_filter=kf,
        camera=_camera_inputs(stamp_ns=ts_jump),
    )
    assert d_jump.observation is not None
    # Gated innovation keeps filtered range close to the person rather than jumping to 4.8m
    assert d_jump.observation.range_m < 2.6
