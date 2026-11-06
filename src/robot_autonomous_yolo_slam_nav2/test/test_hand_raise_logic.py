"""Behavioral checks for identity-bound foot lifting, without ROS or a model."""

import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


def _logic():
    return importlib.import_module("robot_autonomous_yolo_slam_nav2.hand_raise_logic")


def _pose(leg: str = "left"):
    logic = _logic()
    points = [logic.Keypoint(50.0, 40.0, 0.9) for _ in range(17)]
    points[11] = logic.Keypoint(40.0, 100.0, 0.9)
    points[12] = logic.Keypoint(60.0, 100.0, 0.9)
    points[13] = logic.Keypoint(40.0, 140.0, 0.9)
    points[14] = logic.Keypoint(60.0, 140.0, 0.9)
    points[15] = logic.Keypoint(40.0, 180.0, 0.9)
    points[16] = logic.Keypoint(60.0, 180.0, 0.9)
    if leg == "left":
        points[15] = logic.Keypoint(40.0, 150.0, 0.9)
    if leg == "right":
        points[16] = logic.Keypoint(60.0, 150.0, 0.9)
    return logic.PoseDetection((0.0, 0.0, 100.0, 200.0), tuple(points))


@pytest.mark.parametrize("leg", ["left", "right"])
def test_either_foot_lifted_is_accepted(leg: str) -> None:
    pose = _pose(leg)
    assert _logic().target_leg_raised((pose,), pose.bbox) is True


def test_both_feet_on_ground_are_rejected() -> None:
    pose = _pose("down")
    assert _logic().target_leg_raised((pose,), pose.bbox) is False


def test_small_foot_lift_is_rejected() -> None:
    from dataclasses import replace
    pose = _pose("down")
    points = list(pose.keypoints)
    points[15] = replace(points[15], y=170.0)
    assert _logic().target_leg_raised(
        (replace(pose, keypoints=tuple(points)),), pose.bbox) is False


@pytest.mark.parametrize("index", [11, 13, 15])
def test_uncertain_leg_joint_is_rejected(index: int) -> None:
    from dataclasses import replace
    pose = _pose("left")
    points = list(pose.keypoints)
    points[index] = replace(points[index], confidence=0.1)
    uncertain = replace(pose, keypoints=tuple(points))
    assert _logic().target_leg_raised((uncertain,), pose.bbox) is False


def test_confident_joint_on_image_border_is_visible() -> None:
    logic = _logic()
    assert logic.Keypoint(0.0, 0.0, 0.9).visible(0.5) is True
    assert logic.Keypoint(0.0, 0.0, 0.0).visible(0.5) is False
    assert logic.Keypoint(0.0, 0.0, float("nan")).visible(0.5) is False


def test_one_visible_leg_accepts_ankle_above_knee() -> None:
    logic = _logic()
    points = [logic.Keypoint(50.0, 40.0, 0.9) for _ in range(17)]
    points[11] = logic.Keypoint(40.0, 100.0, 0.9)
    points[13] = logic.Keypoint(40.0, 140.0, 0.9)
    points[15] = logic.Keypoint(40.0, 120.0, 0.9)
    points[12] = logic.Keypoint(60.0, 100.0, 0.0)
    points[14] = logic.Keypoint(60.0, 140.0, 0.0)
    points[16] = logic.Keypoint(60.0, 180.0, 0.0)
    pose = logic.PoseDetection((0.0, 0.0, 100.0, 200.0), tuple(points))
    assert logic.target_leg_raised((pose,), pose.bbox) is True


def test_one_visible_leg_ankle_below_knee_is_rejected() -> None:
    logic = _logic()
    points = [logic.Keypoint(50.0, 40.0, 0.9) for _ in range(17)]
    points[11] = logic.Keypoint(40.0, 100.0, 0.9)
    points[13] = logic.Keypoint(40.0, 140.0, 0.9)
    points[15] = logic.Keypoint(40.0, 160.0, 0.9)
    points[12] = logic.Keypoint(60.0, 100.0, 0.0)
    points[14] = logic.Keypoint(60.0, 140.0, 0.0)
    points[16] = logic.Keypoint(60.0, 180.0, 0.0)
    pose = logic.PoseDetection((0.0, 0.0, 100.0, 200.0), tuple(points))
    assert logic.target_leg_raised((pose,), pose.bbox) is False


def test_another_persons_lifted_foot_is_not_used() -> None:
    from dataclasses import replace
    target = _pose("down")
    other = replace(_pose("left"), bbox=(200.0, 0.0, 300.0, 200.0))
    assert _logic().target_leg_raised((other, target), target.bbox) is False


def test_ambiguous_pose_match_is_rejected() -> None:
    pose = _pose("left")
    assert _logic().target_leg_raised((pose, pose), pose.bbox) is False


def test_hold_requires_full_second_of_distinct_samples() -> None:
    logic = _logic()
    hold = logic.HandRaiseHold()
    for stamp in (1_000_000_000, 1_250_000_000, 1_500_000_000, 1_750_000_000):
        hold.update(logic.GestureSample(7, stamp, True), stamp)
    assert hold.ready(1_750_000_000, 7) is False
    hold.update(logic.GestureSample(7, 2_000_000_000, True), 2_000_000_000)
    assert hold.ready(2_000_000_000, 7) is True


@pytest.mark.parametrize("interruption", ["lowered", "gap", "identity", "future", "stale", "backward"])
def test_interruption_restarts_the_hold(interruption: str) -> None:
    logic = _logic()
    hold = logic.HandRaiseHold()
    for stamp in (1_000_000_000, 1_250_000_000, 1_500_000_000):
        hold.update(logic.GestureSample(7, stamp, True), stamp)
    samples = {
        "lowered": (7, 1_750_000_000, False, 1_750_000_000),
        "gap": (7, 2_250_000_000, True, 2_250_000_000),
        "identity": (8, 1_750_000_000, True, 1_750_000_000),
        "future": (7, 3_000_000_000, True, 1_750_000_000),
        "stale": (7, 1_600_000_000, True, 2_250_000_000),
        "backward": (7, 1_200_000_000, True, 1_750_000_000),
    }
    tid, stamp, raised, now = samples[interruption]
    hold.update(logic.GestureSample(tid, stamp, raised), now)
    assert hold.ready(now, tid) is False


def test_duplicate_samples_cannot_refresh_or_complete_hold() -> None:
    logic = _logic()
    hold = logic.HandRaiseHold()
    for now in (1_000_000_000, 1_250_000_000, 1_500_000_000, 2_000_000_000):
        hold.update(logic.GestureSample(7, 1_000_000_000, True), now)
    assert hold.ready(2_000_000_000, 7) is False


def test_ready_hold_expires_and_cannot_transfer_identity() -> None:
    logic = _logic()
    hold = logic.HandRaiseHold()
    for stamp in (1_000_000_000, 1_500_000_000, 2_000_000_000):
        hold.update(logic.GestureSample(7, stamp, True), stamp)
    assert hold.ready(2_000_000_000, 8) is False
    assert hold.ready(2_600_000_000, 7) is False
