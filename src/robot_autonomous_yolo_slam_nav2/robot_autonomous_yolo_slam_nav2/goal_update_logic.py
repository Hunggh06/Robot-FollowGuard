"""Map-frame goal identity and replacement policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from builtin_interfaces.msg import Time


GoalMode = Literal["follow", "recovery"]


@dataclass(frozen=True, slots=True)
class GoalSnapshot:
    """Map-frame goal identity used to suppress only true duplicates."""

    position: tuple[float, float]
    yaw_rad: float
    mode: GoalMode


@dataclass(frozen=True, slots=True)
class GoalUpdateThresholds:
    """Map-pose thresholds for replacing an active goal."""

    position_m: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class GoalRequest:
    """One Nav2 request plus the mode that owns it."""

    goal_base_link: tuple[float, float]
    yaw_rad: float
    stamp: Time
    mode: GoalMode
    token: str | None = None


def goal_requires_update(
    previous: GoalSnapshot | None,
    current: GoalSnapshot,
    thresholds: GoalUpdateThresholds,
) -> bool:
    """Return whether map pose or goal mode changed enough to resend."""
    if previous is None or previous.mode != current.mode:
        return True
    position_delta = math.hypot(
        current.position[0] - previous.position[0],
        current.position[1] - previous.position[1],
    )
    yaw_delta = abs(
        math.atan2(
            math.sin(current.yaw_rad - previous.yaw_rad),
            math.cos(current.yaw_rad - previous.yaw_rad),
        )
    )
    return position_delta >= thresholds.position_m or yaw_delta >= thresholds.yaw_rad
