"""Pure geometry and safety decisions for a moving-target Nav2 follower."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from typing_extensions import assert_never

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


class FollowDecisionStatus(Enum):
    """Action requested from the Nav2 runtime."""

    GOAL = "goal"
    HOLD = "hold"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class TargetGeometry:
    """Metric target data after the ROS message boundary."""

    state: str
    valid: bool
    point_base_link: Point3D
    range_m: float
    bearing_rad: float
    age_s: float
    frame_id: str = "base_link"


@dataclass(frozen=True, slots=True)
class ObstaclePoint:
    """One non-target person point expressed in base_link."""

    point_base_link: Point3D


@dataclass(frozen=True, slots=True)
class FollowConfig:
    """Safety and movement thresholds for the follower."""

    follow_distance_m: float = 1.0
    follow_resume_distance_m: float = 1.2
    max_target_age_s: float = 0.5
    corridor_half_width_m: float = 0.22
    goal_update_distance_m: float = 0.1
    base_frame: str = "base_link"


@dataclass(frozen=True, slots=True)
class FollowDecision:
    """One immutable decision for the Nav2 action-client runtime."""

    status: FollowDecisionStatus
    goal_base_link: Point2D | None
    goal_yaw_rad: float | None
    cancel_active_goal: bool
    reason: str


def _finite_point(point: Point3D) -> bool:
    return all(math.isfinite(value) for value in point)


def _valid_target(target: TargetGeometry, config: FollowConfig) -> bool:
    expected_range = math.hypot(target.point_base_link[0], target.point_base_link[1])
    expected_bearing = math.atan2(target.point_base_link[1], target.point_base_link[0])
    bearing_error = math.atan2(
        math.sin(target.bearing_rad - expected_bearing),
        math.cos(target.bearing_rad - expected_bearing),
    )
    point_is_nonzero = sum(value * value for value in target.point_base_link) > 0.0
    return (
        target.state == "tracking"
        and target.valid
        and target.frame_id == config.base_frame
        and math.isfinite(target.range_m)
        and target.range_m > 0.0
        and _finite_point(target.point_base_link)
        and point_is_nonzero
        and math.isfinite(target.bearing_rad)
        and math.isfinite(target.age_s)
        and target.age_s >= 0.0
        and target.age_s <= config.max_target_age_s
        and math.isclose(target.range_m, expected_range, rel_tol=1e-6, abs_tol=1e-6)
        and abs(bearing_error) <= 1e-6
    )


def _blocked_by_obstacle(
    target: TargetGeometry,
    obstacles: tuple[ObstaclePoint, ...],
    config: FollowConfig,
) -> bool:
    bearing_cos = math.cos(target.bearing_rad)
    bearing_sin = math.sin(target.bearing_rad)
    for obstacle in obstacles:
        if not _finite_point(obstacle.point_base_link):
            continue
        x, y, _ = obstacle.point_base_link
        along = x * bearing_cos + y * bearing_sin
        lateral = abs(-x * bearing_sin + y * bearing_cos)
        if (
            0.0 <= along < target.range_m
            and lateral <= config.corridor_half_width_m
        ):
            return True
    return False


def _hold(reason: str, cancel_active_goal: bool) -> FollowDecision:
    return FollowDecision(
        status=FollowDecisionStatus.HOLD,
        goal_base_link=None,
        goal_yaw_rad=None,
        cancel_active_goal=cancel_active_goal,
        reason=reason,
    )


def _cancel(reason: str) -> FollowDecision:
    return FollowDecision(
        status=FollowDecisionStatus.CANCEL,
        goal_base_link=None,
        goal_yaw_rad=None,
        cancel_active_goal=True,
        reason=reason,
    )


def decide_follow_goal(
    target: TargetGeometry,
    obstacles: tuple[ObstaclePoint, ...],
    config: FollowConfig,
    previous_goal_base_link: Point2D | None,
) -> FollowDecision:
    """Choose a safe base-link goal, hold, or cancellation action."""
    if not _valid_target(target, config):
        return _cancel("target_invalid_or_stale")
    if _blocked_by_obstacle(target, obstacles, config):
        return _hold("target_corridor_blocked", True)

    goal_distance = target.range_m - config.follow_distance_m
    if goal_distance <= 0.0:
        return _hold("target_inside_follow_distance", True)
    goal = (
        goal_distance * math.cos(target.bearing_rad),
        goal_distance * math.sin(target.bearing_rad),
    )
    if previous_goal_base_link is not None:
        delta = math.hypot(
            goal[0] - previous_goal_base_link[0],
            goal[1] - previous_goal_base_link[1],
        )
        if delta < config.goal_update_distance_m:
            return _hold("goal_update_below_threshold", False)
    return FollowDecision(
        status=FollowDecisionStatus.GOAL,
        goal_base_link=goal,
        goal_yaw_rad=target.bearing_rad,
        cancel_active_goal=False,
        reason="clear_path",
    )


def decision_requires_cancel(decision: FollowDecision) -> bool:
    """Return whether a runtime must cancel its active Nav2 goal."""
    match decision.status:
        case FollowDecisionStatus.GOAL:
            return decision.cancel_active_goal
        case FollowDecisionStatus.HOLD:
            return decision.cancel_active_goal
        case FollowDecisionStatus.CANCEL:
            return True
        case unreachable:
            assert_never(unreachable)


CloseAction = Literal["cancel_now", "hold", "follow"]


def update_close_latch(
    latched: bool,
    fresh_range_m: float | None,
    config: FollowConfig,
) -> tuple[bool, CloseAction]:
    """Hysteresis around the follow stand-off so boundary noise cannot churn goals.

    Returns (latched, action) with action one of "cancel_now", "hold", "follow".
    "cancel_now" stops an approach immediately; "hold" keeps position quietly.
    """
    if fresh_range_m is not None and fresh_range_m <= config.follow_distance_m:
        return True, "cancel_now"
    if latched:
        if fresh_range_m is None:
            return False, "follow"
        if fresh_range_m >= config.follow_resume_distance_m:
            return False, "follow"
        return True, "hold"
    return False, "follow"


@dataclass(frozen=True, slots=True)
class CancelGrace:
    """Consecutive cancel-worthy cycles; prevents one-frame blips cancelling goals."""

    consecutive: int = 0


def grace_advance(grace: CancelGrace, needed: bool, limit: int) -> tuple[CancelGrace, bool]:
    """Count consecutive cancel requests; fire only after `limit` cycles."""
    if not needed:
        return CancelGrace(0), False
    advanced = CancelGrace(grace.consecutive + 1)
    return advanced, advanced.consecutive >= limit


def estimate_person_speed(
    previous: tuple[tuple[float, float], float] | None,
    current: tuple[tuple[float, float], float],
) -> float | None:
    """Estimate map-frame speed between two stamped positions.

    Returns None when the pair is unusable (missing, non-finite, or
    outside the 0.2-3.0 s window); returns 0.0 for a static person.
    """
    if previous is None:
        return None
    (x0, y0), t0 = previous
    (x1, y1), t1 = current
    if not all(math.isfinite(value) for value in (x0, y0, t0, x1, y1, t1)):
        return None
    dt_s = t1 - t0
    if dt_s < 0.2 or dt_s > 3.0:
        return None
    return math.hypot(x1 - x0, y1 - y0) / dt_s


def search_spin_command(
    bearing_rad: float | None,
    speed_m_s: float | None,
    range_m: float,
    default_rate_rad_s: float,
    min_rate_rad_s: float,
    max_rate_rad_s: float,
    deadband_rad: float,
) -> float | None:
    """Yaw rate that keeps facing a lost person, or None when aimless.

    Positive bearing (target left) spins left. The rate matches the
    person's last measured speed (omega = v / r) so the robot turns
    with them instead of a fixed blind spin.
    """
    if bearing_rad is None or not math.isfinite(bearing_rad):
        return None
    if abs(bearing_rad) <= deadband_rad:
        return 0.0
    if (
        speed_m_s is None
        or not math.isfinite(speed_m_s)
        or speed_m_s <= 0.0
        or not math.isfinite(range_m)
        or range_m <= 0.0
    ):
        rate = default_rate_rad_s
    else:
        rate = speed_m_s / max(range_m, 0.5)
    rate = min(max(rate, min_rate_rad_s), max_rate_rad_s)
    return rate if bearing_rad > 0.0 else -rate
