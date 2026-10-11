import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.follow_goal_helpers import (  # noqa: E402
    CancelGrace,
    FollowConfig,
    FollowDecisionStatus,
    ObstaclePoint,
    TargetGeometry,
    decide_follow_goal,
    grace_advance,
    update_close_latch,
)


def _config() -> FollowConfig:
    return FollowConfig(
        follow_distance_m=1.0,
        max_target_age_s=0.5,
        corridor_half_width_m=0.4,
        goal_update_distance_m=0.2,
        base_frame="base_link",
    )


def _target(
    *,
    state: str = "tracking",
    valid: bool = True,
    point: tuple[float, float, float] = (3.0, 0.0, 0.0),
    range_m: float = 3.0,
    bearing_rad: float = 0.0,
    age_s: float = 0.1,
) -> TargetGeometry:
    return TargetGeometry(
        state=state,
        valid=valid,
        point_base_link=point,
        range_m=range_m,
        bearing_rad=bearing_rad,
        age_s=age_s,
    )


def test_goal_is_behind_target_at_follow_distance() -> None:
    # Given a fresh tracking target three metres ahead.
    target = _target()

    # When a follow decision is computed with a one metre stand-off.
    decision = decide_follow_goal(target, (), _config(), None)

    # Then the goal is two metres ahead on the target bearing.
    assert decision.status is FollowDecisionStatus.GOAL
    assert decision.goal_base_link == pytest.approx((2.0, 0.0))
    assert decision.goal_yaw_rad == pytest.approx(0.0)


def test_inside_follow_distance_holds_without_goal() -> None:
    # Given a fresh target already inside the configured follow distance.
    target = _target(point=(0.7, 0.0, 0.0), range_m=0.7)

    # When a follow decision is computed.
    decision = decide_follow_goal(target, (), _config(), None)

    # Then no forward goal is created and the runtime is asked to hold.
    assert decision.status is FollowDecisionStatus.HOLD
    assert decision.goal_base_link is None
    assert decision.cancel_active_goal


@pytest.mark.parametrize(
    ("state", "valid", "age_s"),
    (("lost", True, 0.1), ("tracking", False, 0.1), ("tracking", True, 0.6)),
)
def test_invalid_or_stale_target_cancels(
    state: str,
    valid: bool,
    age_s: float,
) -> None:
    # Given a target that violates exactly one validity requirement.
    target = _target(state=state, valid=valid, age_s=age_s)

    # When a follow decision is computed.
    decision = decide_follow_goal(target, (), _config(), None)

    # Then the active Nav2 goal is cancelled and no replacement is allowed.
    assert decision.status is FollowDecisionStatus.CANCEL
    assert decision.goal_base_link is None
    assert decision.cancel_active_goal


def test_non_target_in_corridor_holds_without_forward_goal() -> None:
    # Given a target ahead and a non-target person before it in the corridor.
    target = _target()
    obstacles = (ObstaclePoint(point_base_link=(1.0, 0.2, 0.0)),)

    # When a follow decision is computed.
    decision = decide_follow_goal(target, obstacles, _config(), None)

    # Then the corridor block prevents a forward goal.
    assert decision.status is FollowDecisionStatus.HOLD
    assert decision.goal_base_link is None
    assert decision.cancel_active_goal


def test_clear_path_creates_goal() -> None:
    # Given a target and an obstacle outside the configured lateral corridor.
    target = _target()
    obstacles = (ObstaclePoint(point_base_link=(1.0, 0.5, 0.0)),)

    # When a follow decision is computed.
    decision = decide_follow_goal(target, obstacles, _config(), None)

    # Then the clear path still produces the behind-target goal.
    assert decision.status is FollowDecisionStatus.GOAL
    assert decision.goal_base_link == pytest.approx((2.0, 0.0))


def test_small_base_frame_goal_churn_is_suppressed() -> None:
    # Given an active goal and a new candidate less than the update threshold away.
    target = _target(point=(3.1, 0.0, 0.0), range_m=3.1)
    previous_goal = (2.0, 0.0)

    # When a follow decision is computed.
    decision = decide_follow_goal(target, (), _config(), previous_goal)

    # Then the existing goal is left active instead of being replaced.
    assert decision.status is FollowDecisionStatus.HOLD
    assert decision.goal_base_link is None
    assert not decision.cancel_active_goal


def test_nonfinite_metric_data_cancels() -> None:
    # Given a target with a non-finite metric point.
    target = _target(point=(math.nan, 0.0, 0.0))

    # When a follow decision is computed.
    decision = decide_follow_goal(target, (), _config(), None)

    # Then the invalid metric data cannot produce a goal.
    assert decision.status is FollowDecisionStatus.CANCEL


def test_future_target_cancels() -> None:
    decision = decide_follow_goal(_target(age_s=-0.01), (), _config(), None)

    assert decision.status is FollowDecisionStatus.CANCEL


def test_inconsistent_target_geometry_cancels() -> None:
    target = _target(point=(4.0, 0.0, 0.0), range_m=3.0)

    decision = decide_follow_goal(target, (), _config(), None)

    assert decision.status is FollowDecisionStatus.CANCEL


def test_wrong_target_frame_cancels() -> None:
    target = TargetGeometry(
        state="tracking",
        valid=True,
        point_base_link=(3.0, 0.0, 0.0),
        range_m=3.0,
        bearing_rad=0.0,
        age_s=0.1,
        frame_id="camera_link",
    )

    decision = decide_follow_goal(target, (), _config(), None)

    assert decision.status is FollowDecisionStatus.CANCEL


def test_close_range_cancels_once_then_holds_quietly_until_resume() -> None:
    # Given default hysteresis (stop at 1.0 m, resume at 1.2 m).
    config = _config()

    # When the person steps inside the stand-off.
    latched, action = update_close_latch(False, 0.9, config)

    # Then the approach stops immediately.
    assert (latched, action) == (True, "cancel_now")

    # When the person hovers at the boundary.
    latched, action = update_close_latch(latched, 1.05, config)

    # Then no further cancel storm is emitted.
    assert (latched, action) == (True, "hold")

    # When the person steps back out past resume distance.
    latched, action = update_close_latch(latched, 1.25, config)

    # Then following resumes.
    assert (latched, action) == (False, "follow")


def test_lost_target_returns_to_recovery_instead_of_holding_close() -> None:
    # Given a latched close hold.
    config = _config()

    # When the target disappears entirely.
    latched, action = update_close_latch(True, None, config)

    # Then the latch releases so recovery/search can take over.
    assert (latched, action) == (False, "follow")


def test_single_frame_blip_does_not_fire_cancel() -> None:
    # Given no prior cancel pressure.
    grace = CancelGrace()

    # When fewer than five consecutive bad cycles occur.
    for _ in range(4):
        grace, fire = grace_advance(grace, True, 5)
        assert fire is False

    # Then the fifth consecutive bad cycle fires once.
    grace, fire = grace_advance(grace, True, 5)
    assert fire is True

    # When a healthy cycle intervenes.
    grace, fire = grace_advance(grace, False, 5)

    # Then the counter resets instead of firing late.
    assert (grace.consecutive, fire) == (0, False)
