from __future__ import annotations

from rclpy.node import Node
from std_msgs.msg import Bool

from robot_autonomous_yolo_slam_nav2.msg import HandRaise, PersonTarget

from .hand_raise_logic import GestureSample, HandRaiseHold
from .target_state import TrackId


class ZoneHandRaise:
    def __init__(self, node: Node, max_target_age_s: float = 1.0) -> None:
        self._node = node
        node.declare_parameter("hand_raise_topic", "/person_follow/hand_raise")
        node.declare_parameter("confirm_request_topic", "/person_follow/confirm_requested")
        node.declare_parameter("hand_raise_hold_s", 1.0)
        node.declare_parameter("hand_raise_max_gap_s", 0.5)
        self._hold = HandRaiseHold(
            float(str(node.get_parameter("hand_raise_hold_s").value)),
            float(str(node.get_parameter("hand_raise_max_gap_s").value)),
        )
        self._max_target_age_ns = int(max_target_age_s * 1_000_000_000)
        self._waiting = False
        self._since_ns = 0
        self._track_id = TrackId(0)
        self._target_stamp_ns = 0
        self._target_eligible = False
        self._request_pub = node.create_publisher(
            Bool, str(node.get_parameter("confirm_request_topic").value), 1)
        self._subscription = node.create_subscription(
            HandRaise, str(node.get_parameter("hand_raise_topic").value), self.receive, 5)

    def set_waiting(self, waiting: bool, now_ns: int) -> None:
        if waiting != self._waiting:
            self._hold.reset()
            self._since_ns = now_ns
        self._waiting = waiting
        self._request_pub.publish(Bool(data=waiting))

    def update_target(self, target: PersonTarget) -> None:
        stamp_ns = (
            int(target.header.stamp.sec) * 1_000_000_000 + int(target.header.stamp.nanosec))
        held_continuity = (
            self._target_eligible
            and int(target.track_id) == int(self._track_id)
            and target.position_source == "held_invalid"
            and target.identity_confirmed
            and target.state == "tracking"
        )
        eligible = (
            (target.valid and target.identity_confirmed and target.state == "tracking")
            or held_continuity
        )
        if not eligible or int(target.track_id) != self._track_id:
            self._hold.reset()
        self._track_id = TrackId(int(target.track_id))
        if target.valid or not held_continuity:
            self._target_stamp_ns = stamp_ns
        self._target_eligible = bool(eligible)

    def _target_fresh(self, now_ns: int) -> bool:
        return (
            self._target_eligible and self._track_id > 0
            and 0 <= now_ns - self._target_stamp_ns <= self._max_target_age_ns
        )

    def receive(self, message: HandRaise) -> None:
        now_ns = self._node.get_clock().now().nanoseconds
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        if (
            not self._waiting or not self._target_fresh(now_ns)
            or int(message.track_id) != self._track_id or stamp_ns < self._since_ns
        ):
            self._hold.reset()
            return
        self._hold.update(
            GestureSample(TrackId(int(message.track_id)), stamp_ns, bool(message.raised)), now_ns)

    def consume(self, now_ns: int) -> bool:
        accepted = (
            self._waiting and self._target_fresh(now_ns)
            and self._hold.ready(now_ns, self._track_id)
        )
        if accepted:
            self.set_waiting(False, now_ns)
        return accepted
