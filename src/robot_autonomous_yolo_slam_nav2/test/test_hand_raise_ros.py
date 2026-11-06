import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.clock import ROSClock
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Bool, Header

from robot_autonomous_yolo_slam_nav2.msg import HandRaise, PersonTarget

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture
def node():
    rclpy.init(args=[])
    instance = Node("hand_raise_test")
    instance.set_parameters([Parameter("use_sim_time", value=True)])
    _set_time(instance, 10_000_000_000)
    yield instance
    instance.destroy_node()
    rclpy.shutdown()


def _target(node: Node, track_id: int = 7) -> PersonTarget:
    message = PersonTarget()
    message.header.stamp = node.get_clock().now().to_msg()
    message.track_id = track_id
    message.valid = True
    message.identity_confirmed = True
    message.state = "tracking"
    return message


def _set_time(node: Node, stamp_ns: int) -> None:
    clock = node.get_clock()
    assert isinstance(clock, ROSClock)
    clock.set_ros_time_override(Time(nanoseconds=stamp_ns))


def _confirmation(node: Node):
    module = importlib.import_module("robot_autonomous_yolo_slam_nav2.zone_hand_raise")
    return module.ZoneHandRaise(node)


def _feed(confirmation, target: PersonTarget, duration_ns: int = 1_000_000_000):
    now = confirmation._node.get_clock().now().nanoseconds
    for offset in range(0, duration_ns + 1, 250_000_000):
        message = HandRaise(track_id=target.track_id, raised=True)
        stamp = now + offset
        _set_time(confirmation._node, stamp)
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        confirmation.receive(message)


def test_correct_person_hold_is_consumed_once_when_waiting(node: Node) -> None:
    confirmation = _confirmation(node)
    target = _target(node)
    now = node.get_clock().now().nanoseconds
    confirmation.set_waiting(True, now - 2_000_000_000)
    confirmation.update_target(target)
    _feed(confirmation, target)
    assert confirmation.consume(node.get_clock().now().nanoseconds) is True
    assert confirmation.consume(node.get_clock().now().nanoseconds) is False


@pytest.mark.parametrize("condition", ["inactive", "wrong_id", "lost", "unconfirmed", "old_window", "stale"])
def test_unqualified_evidence_cannot_confirm(node: Node, condition: str) -> None:
    confirmation = _confirmation(node)
    target = _target(node)
    now = node.get_clock().now().nanoseconds
    confirmation.set_waiting(condition != "inactive", now - 2_000_000_000)
    confirmation.update_target(target)
    if condition == "wrong_id":
        target.track_id = 8
    if condition == "lost":
        target.valid = False
        confirmation.update_target(target)
    if condition == "unconfirmed":
        target.identity_confirmed = False
        confirmation.update_target(target)
    if condition == "old_window":
        confirmation.set_waiting(False, now)
        confirmation.set_waiting(True, now + 500_000_000)
    if condition == "stale":
        target.header.stamp.sec -= 2
        confirmation.update_target(target)
    _feed(confirmation, target)
    assert confirmation.consume(node.get_clock().now().nanoseconds) is False


def test_target_change_clears_completed_hold(node: Node) -> None:
    confirmation = _confirmation(node)
    target = _target(node)
    now = node.get_clock().now().nanoseconds
    confirmation.set_waiting(True, now - 2_000_000_000)
    confirmation.update_target(target)
    _feed(confirmation, target)
    confirmation.update_target(_target(node, 8))
    assert confirmation.consume(node.get_clock().now().nanoseconds) is False


def test_pose_inference_is_gated_by_live_request_and_target(node: Node, monkeypatch) -> None:
    from robot_autonomous_yolo_slam_nav2.target_state import (
        TargetSelectionResult, TargetState, TrackId, TrackObservation,
    )
    module = importlib.import_module("robot_autonomous_yolo_slam_nav2.hand_raise_perception")
    perception = module.HandRaisePerception(node)
    calls: list[int] = []
    monkeypatch.setattr(perception, "_detect", lambda frame: calls.append(1) or ())
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    target = TrackObservation(TrackId(7), 0.9, (0.0, 0.0, 100.0, 200.0))
    result = TargetSelectionResult(TargetState.TRACKING, target, (), True)
    header = Header(stamp=node.get_clock().now().to_msg())
    perception.publish(frame, result, header, source_stamp_ns=1_000_000_000)
    assert calls == []
    perception._request_callback(Bool(data=True))
    perception.publish(frame, result, header, source_stamp_ns=1_000_000_000)
    assert calls == [1]
    perception._request_callback(Bool(data=False))
    perception.publish(frame, result, header, source_stamp_ns=1_000_000_000)
    assert calls == [1]


def test_zone_guard_no_longer_subscribes_to_speech(node: Node) -> None:
    from robot_autonomous_yolo_slam_nav2.zone_guard_runtime import ZoneGuardRuntime
    zone = ZoneGuardRuntime()
    try:
        assert "/phone/speech_text" not in {sub.topic_name for sub in zone.subscriptions}
    finally:
        zone.destroy_node()


def test_overlapping_other_person_skips_pose_inference(node: Node, monkeypatch) -> None:
    from robot_autonomous_yolo_slam_nav2.target_state import (
        TargetSelectionResult, TargetState, TrackId, TrackObservation,
    )
    module = importlib.import_module("robot_autonomous_yolo_slam_nav2.hand_raise_perception")
    perception = module.HandRaisePerception(node)
    calls: list[int] = []
    monkeypatch.setattr(perception, "_detect", lambda frame: calls.append(1) or ())
    target = TrackObservation(TrackId(7), 0.9, (0.0, 0.0, 100.0, 200.0))
    other = TrackObservation(TrackId(8), 0.9, (10.0, 0.0, 110.0, 200.0))
    result = TargetSelectionResult(TargetState.TRACKING, target, (other,), True)
    perception._request_callback(Bool(data=True))
    perception.publish(np.zeros((200, 120, 3), dtype=np.uint8), result,
                       Header(stamp=node.get_clock().now().to_msg()), source_stamp_ns=1_000_000_000)
    assert calls == []


def test_repeated_camera_stamp_cannot_refresh_pose_evidence(node: Node, monkeypatch) -> None:
    from robot_autonomous_yolo_slam_nav2.target_state import (
        TargetSelectionResult, TargetState, TrackId, TrackObservation,
    )
    module = importlib.import_module("robot_autonomous_yolo_slam_nav2.hand_raise_perception")
    perception = module.HandRaisePerception(node)
    calls: list[int] = []
    monkeypatch.setattr(perception, "_detect", lambda frame: calls.append(1) or ())
    target = TrackObservation(TrackId(7), 0.9, (0.0, 0.0, 100.0, 200.0))
    result = TargetSelectionResult(TargetState.TRACKING, target, (), True)
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    perception._request_callback(Bool(data=True))
    perception.publish(frame, result, Header(stamp=node.get_clock().now().to_msg()),
                       source_stamp_ns=1_000_000_000)
    _set_time(node, 10_250_000_000)
    perception.publish(frame, result, Header(stamp=node.get_clock().now().to_msg()),
                       source_stamp_ns=1_000_000_000)
    assert calls == [1]
