import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _make_node():
    import rclpy
    from robot_autonomous_yolo_slam_nav2.zone_guard_runtime import (
        ZoneGuardRuntime,
    )

    rclpy.init(args=[])
    return ZoneGuardRuntime()


def test_clicks_latch_square_polygon() -> None:
    import rclpy
    from geometry_msgs.msg import PointStamped

    node = _make_node()
    try:
        assert node._polygon is None
        for x, y in ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)):
            message = PointStamped()
            message.header.frame_id = "map"
            message.point.x = x
            message.point.y = y
            node._clicked_callback(message)
        assert node._polygon == ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        assert node._phase == "ARMED"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_extra_clicks_ignored_after_latch() -> None:
    import rclpy
    from geometry_msgs.msg import PointStamped

    node = _make_node()
    try:
        for x, y in ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (9.0, 9.0)):
            message = PointStamped()
            message.header.frame_id = "map"
            message.point.x = x
            message.point.y = y
            node._clicked_callback(message)
        assert node._polygon is not None
        assert len(node._polygon) == 4
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_confirm_uses_gesture_subscription_instead_of_voice() -> None:
    import rclpy

    node = _make_node()
    try:
        topics = {subscription.topic_name for subscription in node.subscriptions}
        assert "/person_follow/hand_raise" in topics
        assert "/phone/speech_text" not in topics
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_say_queues_until_speaker_discovered() -> None:
    import rclpy

    node = _make_node()
    try:
        node._say("Bạn đã đi vào vùng cấm, quay đầu ngay lập tức")
        assert node._pending_say is not None
        node._tick()
        assert node._pending_say is not None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_guide_goal_defaults_to_polygon_centroid() -> None:
    import rclpy
    from geometry_msgs.msg import PointStamped

    node = _make_node()
    try:
        assert node._guide_goal_point() is None
        for x, y in ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)):
            message = PointStamped()
            message.header.frame_id = "map"
            message.point.x = x
            message.point.y = y
            node._clicked_callback(message)
        assert node._guide_goal_point() == (2.0, 1.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_click_progress_visible_before_latch() -> None:
    import rclpy
    from geometry_msgs.msg import PointStamped

    node = _make_node()
    try:
        for x, y in ((0.0, 0.0), (4.0, 0.0)):
            message = PointStamped()
            message.header.frame_id = "map"
            message.point.x = x
            message.point.y = y
            node._clicked_callback(message)
        assert node._polygon is None
        assert node._clicks == [(0.0, 0.0), (4.0, 0.0)]
        node._publish_clicks()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_verify_exposes_no_direct_twist_publisher() -> None:
    import rclpy

    node = _make_node()
    try:
        assert not hasattr(node, "_spin_pub")
        publishers = list(node._publishers) if hasattr(node, "_publishers") else []
        topics = [getattr(pub, "topic_name", "") for pub in publishers]
        assert "/cmd_vel" not in topics
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_verify_source_has_no_twist_or_cmd_vel() -> None:
    source = Path(__file__).parents[1] / "robot_autonomous_yolo_slam_nav2" / "zone_guard_runtime.py"
    text = source.read_text()
    assert "_spin_pub" not in text
    assert "Twist" not in text
    assert '"/cmd_vel"' not in text


def test_verify_spin_routes_bounded_nav2_yaw_goal() -> None:
    import math
    import rclpy
    from robot_autonomous_yolo_slam_nav2 import zone_guard_runtime as zg

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "ARRIVED_VERIFY"
        node._spin_accum_rad = 0.0
        node._last_spin_ns = 0
        node._scan_points = ((1.0, 0.0),)
        node._scan_stamp_ns = node.get_clock().now().nanoseconds
        node._last_seen_bearing_rad = 1.0
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        sent = []

        class _Future:
            def add_done_callback(self, callback) -> None:
                return None

        class _Client:
            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                return _Future()

        node._action_client = _Client()  # type: ignore[assignment]
        real_pose = zg.robot_map_pose  # type: ignore[attr-defined]
        zg.robot_map_pose = lambda *args, **kwargs: (1.0, 2.0, 0.0)  # type: ignore[attr-defined]
        try:
            node._verify_spin(node.get_clock().now().nanoseconds, False, None)
        finally:
            zg.robot_map_pose = real_pose  # type: ignore[attr-defined]
        assert len(sent) == 1
        pose = sent[0].pose
        assert pose.pose.position.x == 1.0
        assert pose.pose.position.y == 2.0
        yaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
        assert 0.0 < abs(yaw) <= math.pi
        assert node._spin_accum_rad > 0.0
        assert node._spin_accum_rad <= 2.0 * 2.0 * math.pi
        assert node._phase == "ARRIVED_VERIFY"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_verify_spin_holds_when_nav_unavailable() -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2 import zone_guard_runtime as zg

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "ARRIVED_VERIFY"
        node._spin_accum_rad = 0.0
        node._scan_points = ((1.0, 0.0),)
        node._scan_stamp_ns = node.get_clock().now().nanoseconds
        node._last_seen_bearing_rad = 1.0
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        sent = []

        class _Future:
            def add_done_callback(self, callback) -> None:
                return None

        class _Client:
            def server_is_ready(self) -> bool:
                return False

            def send_goal_async(self, goal):
                sent.append(goal)
                return _Future()

        node._action_client = _Client()  # type: ignore[assignment]
        real_pose = zg.robot_map_pose  # type: ignore[attr-defined]
        zg.robot_map_pose = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        try:
            node._verify_spin(node.get_clock().now().nanoseconds, False, None)
        finally:
            zg.robot_map_pose = real_pose  # type: ignore[attr-defined]
        assert sent == []
        assert node._spin_accum_rad == 0.0
        assert node._phase == "ARRIVED_VERIFY"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_verify_cap_enters_hold_without_twist() -> None:
    import math
    import rclpy

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "ARRIVED_VERIFY"
        max_rev = float(str(node.get_parameter("search_max_spin_revolutions").value))
        node._spin_accum_rad = max_rev * 2.0 * math.pi
        node._scan_points = ((1.0, 0.0),)
        node._scan_stamp_ns = node.get_clock().now().nanoseconds
        node._last_seen_bearing_rad = 1.0
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        node._verify_spin(node.get_clock().now().nanoseconds, False, None)
        assert node._phase == "VERIFY_HOLD"
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_gesture_is_ignored_before_confirmation_window() -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.msg import HandRaise, PersonTarget

    node = _make_node()
    try:
        now_ns = node.get_clock().now().nanoseconds
        target = PersonTarget(track_id=7, valid=True, identity_confirmed=True, state="tracking")
        target.header.stamp = node.get_clock().now().to_msg()
        node._target_callback(target)
        message = HandRaise(header=target.header, track_id=7, raised=True)
        node._hand_confirm.receive(message)
        assert node._hand_confirm._hold._last_ns == 0
        node._hand_confirm.set_waiting(True, now_ns - 1)
        node._hand_confirm.receive(message)
        assert node._hand_confirm._hold._last_ns >= now_ns
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_speaker_prompt_does_not_block_visual_confirmation() -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.msg import HandRaise, PersonTarget

    node = _make_node()
    try:
        from robot_autonomous_yolo_slam_nav2.zone_guide_logic import (
            SAY_ASK_CONFIRM,
        )

        now_ns = node.get_clock().now().nanoseconds
        node._say(SAY_ASK_CONFIRM)
        target = PersonTarget(track_id=7, valid=True, identity_confirmed=True, state="tracking")
        target.header.stamp = node.get_clock().now().to_msg()
        node._target_callback(target)
        node._hand_confirm.set_waiting(True, now_ns - 1)
        node._hand_confirm.receive(HandRaise(header=target.header, track_id=7, raised=True))
        assert node._hand_confirm._hold._last_ns >= now_ns
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stale_confirm_expires_before_switching_mode() -> None:
    """A past confirmation must not auto-switch mode on a later violation."""
    import rclpy
    from robot_autonomous_yolo_slam_nav2.hand_raise_logic import GestureSample
    from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
    from robot_autonomous_yolo_slam_nav2.target_state import TrackId

    node = _make_node()
    try:
        now_ns = node.get_clock().now().nanoseconds
        node._phase = "WAIT_CONFIRM"
        target = PersonTarget(track_id=7, valid=True, identity_confirmed=True, state="tracking")
        target.header.stamp = node.get_clock().now().to_msg()
        node._target_callback(target)
        node._hand_confirm.set_waiting(True, now_ns - 65_000_000_000)
        for offset in (61_000_000_000, 60_500_000_000, 60_000_000_000):
            stamp = now_ns - offset
            node._hand_confirm._hold.update(GestureSample(TrackId(7), stamp, True), stamp)
        assert node._consume_confirm() is False
        for offset in (1_000_000_000, 500_000_000, 0):
            stamp = now_ns - offset
            node._hand_confirm._hold.update(GestureSample(TrackId(7), stamp, True), stamp)
        assert node._consume_confirm() is True
        assert node._consume_confirm() is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_initializes_guide_state() -> None:
    import rclpy

    node = _make_node()
    try:
        assert node._guide_fails == 0
        assert node._guide_poly is None
        assert node._guide_active is False
        assert node._guide_handle is None
        assert node._guide_pending is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_during_active_guide_cancels_all_and_idles() -> None:
    """Reset clears zone/UI state but retains cancelled action ownership.

    Supersedes the abandonment variant of this test: refs are no longer
    dropped fire-and-forget. The old guide/verify handoffs stay owned while
    IDLE and block every new motion goal until their terminal results land,
    so reset-then-immediate-guide can never double-command the server.
    """
    import rclpy
    from std_srvs.srv import Trigger

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "GUIDE_NAV"
        node._guide_active = True
        node._guide_done = False
        node._guide_succeeded = False
        node._guide_fails = 0
        node._guide_poly = node._polygon
        node._hand_confirm.set_waiting(True, node.get_clock().now().nanoseconds)

        cancelled = []

        class _Handle:
            def cancel_goal_async(self):
                cancelled.append("guide")
                return None

        class _VerifyHandle:
            def cancel_goal_async(self):
                cancelled.append("verify")
                return None

        guide_handle = _Handle()
        verify_handle = _VerifyHandle()
        node._guide_handle = None
        setattr(node, "_guide_handle", guide_handle)
        setattr(node, "_guide_pending", object())
        setattr(node, "_verify_handle", verify_handle)
        setattr(node, "_verify_pending", object())
        node._saved_max_vel_x = 0.26
        restored = []
        node._set_controller_param = lambda name, value: restored.append((name, value))

        response = node._reset_zone(Trigger.Request(), Trigger.Response())

        assert response.success is True
        assert node._polygon is None
        assert node._phase == "IDLE"
        assert node._guide_active is False
        # Ownership retained: cancelled but still owned, blocking replacements.
        assert node._guide_handle is guide_handle
        assert node._guide_pending is not None
        assert node._verify_handle is verify_handle
        assert node._verify_pending is not None
        assert node._guide_cancel_requested is True
        assert node._guide_cancel_pending is True
        assert node._verify_cancel_requested is True
        assert node._verify_cancel_pending is True
        assert node._guide_done is False
        assert node._guide_succeeded is False
        assert node._guide_fails == 0
        assert node._guide_poly is None
        assert node._hand_confirm._waiting is False
        assert "guide" in cancelled
        assert "verify" in cancelled
        assert node._saved_max_vel_x is None
        assert ("max_vel_x", 0.26) in restored
        node._tick()
        assert node._phase == "IDLE"
        assert node._guide_active is False
        assert node._guide_handle is guide_handle
        assert node._guide_pending is not None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_arrived_verify_outside_releases_guide() -> None:
    """Person outside after guide arrival must not wedge follow suppressed.

    Regression: the ARRIVED_VERIFY outside branch used to re-enter
    VIOLATION while leaving guide_active latched, so the follow node
    yielded forever and every later edge crossing re-announced the
    violation sentence.
    """
    import rclpy

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "ARRIVED_VERIFY"
        node._guide_active = True
        node._saved_max_vel_x = 0.26
        restored = []
        node._set_controller_param = lambda name, value: restored.append((name, value))

        node._verify_spin(node.get_clock().now().nanoseconds, True, (10.0, 10.0))

        assert node._phase == "VIOLATION"
        assert node._guide_active is False
        assert node._saved_max_vel_x is None
        assert ("max_vel_x", 0.26) in restored
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_verify_hold_outside_releases_guide() -> None:
    """VERIFY_HOLD seeing the person outside must hand follow back too."""
    import rclpy

    node = _make_node()
    try:
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "VERIFY_HOLD"
        node._guide_active = True
        node._saved_max_vel_x = 0.26
        restored = []
        node._set_controller_param = lambda name, value: restored.append((name, value))
        node._person_state = lambda now_ns: (True, (10.0, 10.0))

        node._tick()

        assert node._phase == "VIOLATION"
        assert node._guide_active is False
        assert node._saved_max_vel_x is None
        assert ("max_vel_x", 0.26) in restored
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_cancel_requested_verify_pending_cancels_on_acceptance() -> None:
    """A verify cancel raced with a pending send cancels on acceptance.

    Supersedes test_stale_verify_response_installs_no_goal from the
    epoch-only fix: the pending future is no longer dropped, so there is no
    orphan. The owned pending stays until the server answers; an acceptance
    installs the handle and cancels it at once, and no replacement verify
    goal may be sent until the terminal result arrives.
    """
    import rclpy
    from rclpy.task import Future

    node = _make_node()
    try:
        sent: list = []

        class _SendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _Client:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                future = _SendFuture()
                self.futures.append(future)
                return future

        class _AcceptedHandle:
            accepted = True

            def __init__(self):
                self.cancel_calls: list = []
                self.result_callbacks: list = []

            def cancel_goal_async(self):
                self.cancel_calls.append(True)
                done: Future = Future()
                done.set_result(True)
                return done

            def get_result_async(self):
                owner = self

                class _ResultFuture:
                    def add_done_callback(self, callback):
                        owner.result_callbacks.append(callback)
                        return None

                return _ResultFuture()

        node._action_client = _Client()  # type: ignore[assignment]
        node._send_verify_yaw_goal((1.0, 2.0, 0.0), 0.5)
        assert node._verify_pending is not None

        node._cancel_verify_goal()
        assert node._verify_pending is not None
        assert node._verify_cancel_requested is True

        handle = _AcceptedHandle()
        accepted: Future = Future()
        accepted.set_result(handle)
        node._action_client.futures[0]._callback(accepted)  # type: ignore[attr-defined]

        assert node._verify_handle is handle
        assert handle.cancel_calls == [True]
        assert node._verify_cancel_pending is True
        assert node._verify_cancel_requested is False
        assert len(sent) == 1

        terminal: Future = Future()
        terminal.set_result(True)
        assert len(handle.result_callbacks) == 1
        handle.result_callbacks[0](terminal)

        assert node._verify_handle is None
        assert node._verify_cancel_pending is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_guide_replan_queues_new_goal_until_old_terminal() -> None:
    """A mid-guide redraw must not replace the goal before terminal state.

    Supersedes test_guide_replan_orphan_response_keeps_new_goal: the replan
    no longer cancels-and-immediately-resends (two goals racing). It latches
    cancellation, queues the replan, and only sends the new-centroid goal
    once the old goal's terminal result arrives.
    """
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.task import Future
    from types import SimpleNamespace

    node = _make_node()
    try:
        sent: list = []

        class _GuideSendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _GuideClient:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                future = _GuideSendFuture()
                self.futures.append(future)
                return future

        class _GuideHandle:
            accepted = True

            def __init__(self):
                self.cancel_calls: list = []
                self.result_callbacks: list = []

            def cancel_goal_async(self):
                self.cancel_calls.append("guide")
                done: Future = Future()
                done.set_result(True)
                return done

            def get_result_async(self):
                owner = self

                class _ResultFuture:
                    def add_done_callback(self, callback):
                        owner.result_callbacks.append(callback)
                        return None

                return _ResultFuture()

        node._action_client = _GuideClient()  # type: ignore[assignment]
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "GUIDE_NAV"
        node._guide_active = True
        node._guide_done = False
        node._guide_fails = 0
        node._guide_poly = node._polygon
        node._send_guide_goal()
        assert node._guide_pending is not None

        old_handle = _GuideHandle()
        accepted: Future = Future()
        accepted.set_result(old_handle)
        node._action_client.futures[0]._callback(accepted)  # type: ignore[attr-defined]
        assert node._guide_handle is old_handle

        node._polygon = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))
        node._tick()

        assert old_handle.cancel_calls == ["guide"]
        assert node._guide_replan_queued is True
        assert len(sent) == 1
        assert node._guide_handle is old_handle

        node._tick()
        assert len(sent) == 1

        terminal: Future = Future()
        terminal.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        assert len(old_handle.result_callbacks) == 1
        old_handle.result_callbacks[0](terminal)

        assert len(sent) == 2
        assert node._guide_replan_queued is False
        assert node._guide_pending is not None
        new_goal = sent[1]
        assert new_goal.pose.pose.position.x == 1.0
        assert new_goal.pose.pose.position.y == 0.5
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_retains_guide_pending_until_terminal() -> None:
    """A reset raced with a pending guide send keeps ownership, not epochs.

    Supersedes test_reset_guide_epoch_ignores_late_response: abandonment
    with generation epochs is replaced by retained cancellation. The owned
    pending stays until the server answers; an acceptance installs the
    handle and cancels it at once, all while IDLE, with no new sends.
    """
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.task import Future
    from std_srvs.srv import Trigger
    from types import SimpleNamespace

    node = _make_node()
    try:
        sent: list = []

        class _GuideSendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _GuideClient:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                future = _GuideSendFuture()
                self.futures.append(future)
                return future

        class _GuideHandle:
            accepted = True

            def __init__(self):
                self.cancel_calls: list = []
                self.result_callbacks: list = []

            def cancel_goal_async(self):
                self.cancel_calls.append("guide")
                done: Future = Future()
                done.set_result(True)
                return done

            def get_result_async(self):
                owner = self

                class _ResultFuture:
                    def add_done_callback(self, callback):
                        owner.result_callbacks.append(callback)
                        return None

                return _ResultFuture()

        node._action_client = _GuideClient()  # type: ignore[assignment]
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "GUIDE_NAV"
        node._guide_active = True
        node._send_guide_goal()
        assert node._guide_pending is not None
        old_callback = node._action_client.futures[0]._callback  # type: ignore[attr-defined]

        node._reset_zone(Trigger.Request(), Trigger.Response())
        assert node._phase == "IDLE"
        assert node._guide_pending is not None
        assert node._guide_cancel_requested is True

        late_handle = _GuideHandle()
        accepted: Future = Future()
        accepted.set_result(late_handle)
        old_callback(accepted)

        assert node._guide_handle is late_handle
        assert late_handle.cancel_calls == ["guide"]
        assert node._guide_cancel_pending is True
        assert len(sent) == 1
        assert node._phase == "IDLE"
        assert node._guide_active is False

        terminal: Future = Future()
        terminal.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        assert len(late_handle.result_callbacks) == 1
        late_handle.result_callbacks[0](terminal)

        assert node._guide_handle is None
        assert node._guide_pending is None
        assert node._guide_cancel_pending is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_active_guide_blocks_new_guide_until_terminal() -> None:
    """Reset-then-immediate-guide must never double-command the server.

    STOP-condition regression: after reset with an old active goal, a new
    guide request stays blocked (no send, no phase change, no guide_active)
    until the old goal's terminal result lands; only then is a new request
    allowed.
    """
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.task import Future
    from std_srvs.srv import Trigger
    from types import SimpleNamespace

    node = _make_node()
    try:
        sent: list = []

        class _GuideSendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _GuideClient:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                future = _GuideSendFuture()
                self.futures.append(future)
                return future

        class _GuideHandle:
            accepted = True

            def __init__(self):
                self.cancel_calls: list = []
                self.result_callbacks: list = []

            def cancel_goal_async(self):
                self.cancel_calls.append("guide")
                done: Future = Future()
                done.set_result(True)
                return done

            def get_result_async(self):
                owner = self

                class _ResultFuture:
                    def add_done_callback(self, callback):
                        owner.result_callbacks.append(callback)
                        return None

                return _ResultFuture()

        node._action_client = _GuideClient()  # type: ignore[assignment]
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "GUIDE_NAV"
        node._guide_active = True
        node._guide_done = False
        node._guide_fails = 0
        node._guide_poly = node._polygon
        node._send_guide_goal()
        old_handle = _GuideHandle()
        accepted: Future = Future()
        accepted.set_result(old_handle)
        node._action_client.futures[0]._callback(accepted)  # type: ignore[attr-defined]
        assert node._guide_handle is old_handle

        node._reset_zone(Trigger.Request(), Trigger.Response())
        assert node._phase == "IDLE"
        assert old_handle.cancel_calls == ["guide"]

        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        statuses: list = []
        node._publish_status = lambda phase, reason: statuses.append((phase, reason))  # type: ignore[method-assign]
        node._start_guide(node.get_clock().now().nanoseconds)

        assert len(sent) == 1
        assert node._guide_active is False
        assert node._phase == "IDLE"
        assert ("guide_nav", "guide_start_blocked_handoff") in statuses

        terminal: Future = Future()
        terminal.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        assert len(old_handle.result_callbacks) == 1
        old_handle.result_callbacks[0](terminal)
        assert node._guide_handle is None

        node._start_guide(node.get_clock().now().nanoseconds)

        assert len(sent) == 2
        assert node._guide_active is True
        assert node._phase == "GUIDE_NAV"
        assert node._guide_pending is not None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_active_verify_blocks_new_verify_until_terminal() -> None:
    """Reset-then-immediate-verify must never double-command the server.

    Same STOP condition for the verify handoff: the old goal stays owned
    while IDLE, `_send_verify_yaw_goal` refuses while busy, and sends resume
    only after the terminal result.
    """
    import rclpy
    from rclpy.task import Future
    from std_srvs.srv import Trigger

    node = _make_node()
    try:
        sent: list = []

        class _SendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _Client:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self) -> bool:
                return True

            def send_goal_async(self, goal):
                sent.append(goal)
                future = _SendFuture()
                self.futures.append(future)
                return future

        class _AcceptedHandle:
            accepted = True

            def __init__(self):
                self.cancel_calls: list = []
                self.result_callbacks: list = []

            def cancel_goal_async(self):
                self.cancel_calls.append(True)
                done: Future = Future()
                done.set_result(True)
                return done

            def get_result_async(self):
                owner = self

                class _ResultFuture:
                    def add_done_callback(self, callback):
                        owner.result_callbacks.append(callback)
                        return None

                return _ResultFuture()

        node._action_client = _Client()  # type: ignore[assignment]
        node._polygon = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        node._phase = "ARRIVED_VERIFY"
        node._send_verify_yaw_goal((1.0, 2.0, 0.0), 0.5)
        handle = _AcceptedHandle()
        accepted: Future = Future()
        accepted.set_result(handle)
        node._action_client.futures[0]._callback(accepted)  # type: ignore[attr-defined]
        assert node._verify_handle is handle

        node._reset_zone(Trigger.Request(), Trigger.Response())
        assert node._phase == "IDLE"
        assert handle.cancel_calls == [True]
        assert node._verify_handle is handle

        node._send_verify_yaw_goal((1.0, 2.0, 0.0), 0.5)
        assert len(sent) == 1

        terminal: Future = Future()
        terminal.set_result(True)
        assert len(handle.result_callbacks) == 1
        handle.result_callbacks[0](terminal)
        assert node._verify_handle is None

        node._send_verify_yaw_goal((1.0, 2.0, 0.0), 0.5)
        assert len(sent) == 2
        assert node._verify_pending is not None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_late_guide_result_cannot_clear_newer_goal() -> None:
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.task import Future
    from types import SimpleNamespace

    node = _make_node()
    try:
        old = SimpleNamespace()
        new = SimpleNamespace()
        node._guide_handle = old  # type: ignore[assignment]
        terminal_old: Future = Future()
        terminal_old.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        node._guide_result_callback(terminal_old, old)
        assert node._guide_handle is None

        node._guide_handle = new  # type: ignore[assignment]
        late_duplicate: Future = Future()
        late_duplicate.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        node._guide_result_callback(late_duplicate, old)

        assert node._guide_handle is new
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_dropped_click_reports_tf_unavailable() -> None:
    import rclpy
    from geometry_msgs.msg import PointStamped

    node = _make_node()
    try:
        statuses: list = []
        node._publish_status = lambda phase, reason: statuses.append((phase, reason))  # type: ignore[method-assign]
        node._to_map = lambda _message: None  # type: ignore[method-assign]
        message = PointStamped()
        message.header.frame_id = "base_link"
        message.point.x = 1.0
        message.point.y = 2.0
        node._clicked_callback(message)

        assert node._clicks == []
        assert node._polygon is None
        assert ("draw", "draw_click_tf_unavailable") in statuses
    finally:
        node.destroy_node()
        rclpy.shutdown()
