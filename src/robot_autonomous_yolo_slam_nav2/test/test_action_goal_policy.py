import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.goal_update_logic import (  # noqa: E402
    GoalSnapshot,
    GoalUpdateThresholds,
    goal_requires_update,
)


def test_look_yaw_change_requires_update_without_xy_motion() -> None:
    previous = GoalSnapshot((1.0, 0.0), 0.0, "recovery")
    current = GoalSnapshot((1.0, 0.0), math.pi / 2.0, "recovery")

    assert goal_requires_update(previous, current, GoalUpdateThresholds(0.2, 0.1))


def test_reacquisition_mode_change_requires_update_without_pose_motion() -> None:
    previous = GoalSnapshot((1.0, 0.0), 0.0, "recovery")
    current = GoalSnapshot((1.0, 0.0), 0.0, "follow")

    assert goal_requires_update(previous, current, GoalUpdateThresholds(0.2, 0.1))


def test_unchanged_map_goal_is_suppressed() -> None:
    goal = GoalSnapshot((1.0, 0.0), 0.0, "follow")

    assert not goal_requires_update(goal, goal, GoalUpdateThresholds(0.2, 0.1))
