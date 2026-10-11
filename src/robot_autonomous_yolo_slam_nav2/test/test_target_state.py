import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.target_state import (  # noqa: E402
    BoundingBox,
    TargetSelector,
    TargetState,
    TrackId,
    TrackObservation,
)


def _observation(track_id: int, confidence: float = 0.9) -> TrackObservation:
    return TrackObservation(
        track_id=TrackId(track_id),
        confidence=confidence,
        bbox=BoundingBox((0.0, 0.0, 10.0, 10.0)),
    )


def test_enrolls_after_consecutive_stable_frames() -> None:
    # Given a selector that requires two stable frames.
    selector = TargetSelector(stable_frames=2, min_confidence=0.8, lost_frames=2)

    # When the same eligible track is observed once.
    first_result = selector.update((_observation(7),))

    # Then enrollment has not happened yet and forward motion is blocked.
    assert first_result.state is TargetState.SEARCHING
    assert first_result.target_observation is None
    assert first_result.forward_motion_allowed is False

    # When the same eligible track is observed for the second consecutive frame.
    second_result = selector.update((_observation(7),))

    # Then the track is enrolled and forward motion is allowed.
    assert second_result.state is TargetState.TRACKING
    assert second_result.target_observation == _observation(7)
    assert second_result.forward_motion_allowed is True


def test_locked_target_remains_target_when_another_track_appears() -> None:
    # Given a selector that has already enrolled track 7.
    selector = TargetSelector(stable_frames=1, min_confidence=0.8, lost_frames=2)
    selector.update((_observation(7),))
    other_person = _observation(9, confidence=0.99)
    target_person = _observation(7, confidence=0.85)

    # When both the locked target and a higher-confidence person are observed.
    result = selector.update((other_person, target_person))

    # Then the original track remains the target and the other is non-target.
    assert result.state is TargetState.TRACKING
    assert result.target_observation == target_person
    assert result.non_target_observations == (other_person,)
    assert result.forward_motion_allowed is True


def test_target_loss_reaches_lost_and_blocks_forward_motion() -> None:
    # Given a selector tracking track 7 with a two-frame loss threshold.
    selector = TargetSelector(stable_frames=1, min_confidence=0.8, lost_frames=2)
    selector.update((_observation(7),))

    # When two consecutive frames contain no target track.
    first_missing = selector.update(())
    lost_result = selector.update(())

    # Then the selector enters LOST and keeps forward motion disabled.
    assert first_missing.state is TargetState.TRACKING
    assert first_missing.target_observation is None
    assert first_missing.forward_motion_allowed is False
    assert lost_result.state is TargetState.LOST
    assert lost_result.target_observation is None
    assert lost_result.forward_motion_allowed is False


def test_explicit_reset_returns_to_searching_and_allows_new_enrollment() -> None:
    # Given a selector that has stopped after losing its locked target.
    selector = TargetSelector(stable_frames=1, min_confidence=0.8, lost_frames=1)
    selector.update((_observation(7),))
    selector.update(())

    # When the caller explicitly resets the selector.
    selector.reset()

    # Then it searches again and can enroll a new track.
    assert selector.state is TargetState.SEARCHING
    result = selector.update((_observation(11),))
    assert result.state is TargetState.TRACKING
    assert result.target_observation == _observation(11)


def test_low_confidence_and_empty_frames_have_no_candidate() -> None:
    # Given a selector whose confidence threshold rejects the only track.
    selector = TargetSelector(stable_frames=2, min_confidence=0.8, lost_frames=2)
    low_confidence = _observation(7, confidence=0.79)

    # When a low-confidence observation is followed by an empty frame.
    low_result = selector.update((low_confidence,))
    empty_result = selector.update(())

    # Then both frames remain SEARCHING with motion disabled.
    assert low_result.state is TargetState.SEARCHING
    assert low_result.target_observation is None
    assert low_result.non_target_observations == (low_confidence,)
    assert low_result.forward_motion_allowed is False
    assert empty_result.state is TargetState.SEARCHING
    assert empty_result.target_observation is None
    assert empty_result.non_target_observations == ()
    assert empty_result.forward_motion_allowed is False


def test_stop_blocks_motion_until_explicit_reset() -> None:
    # Given a selector tracking a target.
    selector = TargetSelector(stable_frames=1, min_confidence=0.8, lost_frames=2)
    target = _observation(7)
    selector.update((target,))

    # When the caller stops the selector and another frame arrives.
    selector.stop()
    result = selector.update((target,))

    # Then STOPPED remains safe and does not expose a drive target.
    assert result.state is TargetState.STOPPED
    assert result.target_observation is None
    assert result.non_target_observations == (target,)
    assert result.forward_motion_allowed is False
