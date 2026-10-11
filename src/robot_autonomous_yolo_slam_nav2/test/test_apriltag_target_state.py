import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.apriltag_target_state import (  # noqa: E402
    AprilTagTargetSelector,
    TagDetection,
    TagTargetState,
)


def _detection(tag_id: int = 0, family: str = "tag36h11") -> TagDetection:
    return TagDetection(
        family=family,
        tag_id=tag_id,
        corners=((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        stamp_ns=100,
    )


def test_accepts_only_tag36h11_id_zero_after_stable_frames() -> None:
    selector = AprilTagTargetSelector(stable_frames=2, lost_frames=2)

    assert selector.update((_detection(),), now_ns=100).state is TagTargetState.SEARCHING
    result = selector.update((_detection(),), now_ns=100)

    assert result.state is TagTargetState.TRACKING
    assert result.detection == _detection()


def test_rejects_wrong_id_and_family_without_candidate() -> None:
    selector = AprilTagTargetSelector(stable_frames=1, lost_frames=2)

    wrong_id = selector.update((_detection(tag_id=1),), now_ns=100)
    wrong_family = selector.update((_detection(family="tag25h9"),), now_ns=100)

    assert wrong_id.state is TagTargetState.SEARCHING
    assert wrong_id.detection is None
    assert wrong_family.state is TagTargetState.SEARCHING
    assert wrong_family.detection is None


def test_duplicate_target_detections_are_rejected() -> None:
    selector = AprilTagTargetSelector(stable_frames=1, lost_frames=2)

    result = selector.update((_detection(), _detection()), now_ns=100)

    assert result.state is TagTargetState.SEARCHING
    assert result.detection is None


def test_target_loss_blocks_motion_after_timeout() -> None:
    selector = AprilTagTargetSelector(stable_frames=1, lost_frames=2)
    selector.update((_detection(),), now_ns=100)

    first_missing = selector.update((), now_ns=100)
    lost = selector.update((), now_ns=100)

    assert first_missing.state is TagTargetState.TRACKING
    assert first_missing.detection is None
    assert lost.state is TagTargetState.LOST
    assert lost.detection is None


def test_future_dated_detection_is_rejected() -> None:
    selector = AprilTagTargetSelector(stable_frames=1, lost_frames=2)

    result = selector.update((_detection(),), now_ns=99)

    assert result.state is TagTargetState.SEARCHING
    assert result.detection is None


def test_non_finite_corners_are_rejected() -> None:
    selector = AprilTagTargetSelector(stable_frames=1, lost_frames=2)
    malformed = TagDetection(
        family="tag36h11",
        tag_id=0,
        corners=((math.nan, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        stamp_ns=100,
    )

    result = selector.update((malformed,), now_ns=100)

    assert result.state is TagTargetState.SEARCHING
    assert result.detection is None
