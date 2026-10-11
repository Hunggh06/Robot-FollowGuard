import math
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.reid_target_node import (  # noqa: E402
    selection_to_message_fields,
    tracker_rows_to_observations,
)
from robot_autonomous_yolo_slam_nav2.target_state import (  # noqa: E402
    BoundingBox,
    TargetSelectionResult,
    TargetSelector,
    TargetState,
    TrackId,
    TrackObservation,
)


@dataclass(frozen=True, slots=True)
class HeaderMarker:
    stamp_ns: int


def _observation(track_id: int, bbox: BoundingBox = (1.0, 2.0, 11.0, 22.0)) -> TrackObservation:
    return TrackObservation(
        track_id=TrackId(track_id),
        confidence=0.9,
        bbox=bbox,
    )


def test_tracker_rows_keep_person_rows_and_reject_other_classes() -> None:
    rows = (
        (1.0, 2.0, 11.0, 22.0, 7.0, 0.9, 0.0, 0.0),
        (3.0, 4.0, 13.0, 24.0, 9.0, 0.99, 1.0, 0.0),
        (5.0, 6.0, 15.0, 26.0, 11.0, 0.8, 0.0, 0.0),
    )

    observations = tracker_rows_to_observations(rows)

    assert observations == (
        _observation(7),
        TrackObservation(TrackId(11), 0.8, (5.0, 6.0, 15.0, 26.0)),
    )


def test_tracker_rows_reject_malformed_and_non_finite_rows() -> None:
    rows = (
        (1.0, 2.0, 3.0),
        (1.0, math.nan, 3.0, 4.0, 7.0, 0.9, 0.0),
        (1.0, 2.0, 3.0, 4.0, 7.0, math.inf, 0.0),
        (4.0, 2.0, 3.0, 4.0, 7.0, 0.9, 0.0),
        (1.0, 2.0, 3.0, 4.0, 7.5, 0.9, 0.0),
    )

    assert tracker_rows_to_observations(rows) == ()


def test_message_fields_preserve_header_and_mark_raw_people() -> None:
    target = _observation(7)
    non_target = _observation(9, (20.0, 30.0, 40.0, 50.0))
    result = TargetSelectionResult(
        state=TargetState.TRACKING,
        target_observation=target,
        non_target_observations=(non_target,),
        forward_motion_allowed=True,
    )

    people, target_message = selection_to_message_fields(result, HeaderMarker(42))

    assert tuple(message.track_id for message in people) == (9, 7)
    assert all(message.valid for message in people)
    assert people[0].header == HeaderMarker(42)
    assert people[0].bbox_width == 20.0
    assert people[0].bbox_height == 20.0
    assert target_message.state == "tracking"
    assert target_message.valid is False
    assert target_message.point_base_link == (0.0, 0.0, 0.0)
    assert target_message.range_m == 0.0
    assert target_message.bearing_rad == 0.0


def test_message_fields_keep_target_invalid_before_metric_association() -> None:
    result = TargetSelectionResult(
        state=TargetState.SEARCHING,
        target_observation=None,
        non_target_observations=(_observation(9),),
        forward_motion_allowed=False,
    )

    people, target_message = selection_to_message_fields(result, HeaderMarker(99))

    assert tuple(message.track_id for message in people) == (9,)
    assert target_message.track_id == 0
    assert target_message.state == "searching"
    assert target_message.valid is False
    assert target_message.point_base_link == (0.0, 0.0, 0.0)


def test_reset_is_the_only_recovery_from_stopped() -> None:
    selector = TargetSelector(stable_frames=1, min_confidence=0.5, lost_frames=1)
    selector.stop()

    stopped = selector.update((_observation(7),))
    assert stopped.state is TargetState.STOPPED
    assert stopped.target_observation is None
    assert stopped.forward_motion_allowed is False

    selector.reset()
    tracking = selector.update((_observation(11),))
    assert tracking.state is TargetState.TRACKING
    assert tracking.target_observation == _observation(11)


def test_ros_executable_wrapper_imports_package_entrypoint() -> None:
    # Given the installed ROS executable is launched as a standalone script.
    wrapper = Path(__file__).parents[1] / "scripts" / "reid_target_node.py"

    # When the wrapper is loaded without invoking its main function.
    namespace = runpy.run_path(str(wrapper), run_name="reid_target_node_wrapper")

    # Then it exposes the package entrypoint without relative-import failure.
    assert callable(namespace["main"])
