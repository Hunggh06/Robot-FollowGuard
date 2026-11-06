"""Spin-cancellation handoff: follow waits for the terminal result, fail-closed.

Regression for the async handoff race where _cancel_spin cleared
_active_spin immediately after issuing cancel_goal_async, letting
follow goals resume while the old /spin was still live on the server.
A cancel acknowledgement alone never unblocks follow; only the
terminal Spin result does. A cancellation timeout latches an observable
error and keeps holding instead of reopening motion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _make_node():
    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    rclpy.init(args=[])
    return rclpy, Nav2PersonFollowRuntime()


def _confirmed_target(node):
    from robot_autonomous_yolo_slam_nav2.msg import PersonTarget

    target = PersonTarget()
    target.header.stamp = node.get_clock().now().to_msg()
    target.header.frame_id = "base_link"
    target.track_id = 7
    target.confidence = 0.9
    target.state = "tracking"
    target.valid = True
    target.identity_confirmed = True
    target.point_base_link.x = 2.0
    target.point_base_link.y = 0.0
    target.point_base_link.z = 0.0
    target.range_m = 2.0
    target.bearing_rad = 0.0
    return target


def _fresh_obs(node, track_id, now_ns=None):
    from robot_autonomous_yolo_slam_nav2.msg import PersonObservation

    message = PersonObservation()
    if now_ns is None:
        message.header.stamp = node.get_clock().now().to_msg()
    else:
        message.header.stamp.sec = now_ns // 1_000_000_000
        message.header.stamp.nanosec = now_ns % 1_000_000_000
    message.header.frame_id = "phone_camera_back"
    message.track_id = track_id
    message.confidence = 0.9
    message.bbox_x = 10.0
    message.bbox_y = 10.0
    message.bbox_width = 50.0
    message.bbox_height = 100.0
    message.state = "searching"
    message.valid = False
    node._person_observations_callback(message)


class _DelayedCancelFuture:
    """cancel_goal_async future that only completes when fire() is called."""

    def __init__(self):
        self._callback = None

    def add_done_callback(self, callback):
        self._callback = callback
        return None

    def fire_success(self):
        from rclpy.task import Future
        from types import SimpleNamespace

        done: Future = Future()
        done.set_result(SimpleNamespace(goals_canceling=[]))
        assert self._callback is not None
        self._callback(done)

    def fire_error(self):
        from rclpy.task import Future

        done: Future = Future()
        done.set_exception(RuntimeError("cancel transport failed"))
        assert self._callback is not None
        self._callback(done)


def _fire_spin_result(node):
    from rclpy.task import Future
    from types import SimpleNamespace

    done: Future = Future()
    done.set_result(SimpleNamespace(status=4))
    node._spin_result_callback(done)


class FakeSpin:
    def __init__(self):
        self.cancel_calls: list = []
        self.cancel_future = _DelayedCancelFuture()
        self.accepted = True

    def cancel_goal_async(self):
        self.cancel_calls.append(True)
        return self.cancel_future

    def get_result_async(self):
        from rclpy.task import Future

        # Never completes: spin stays live until cancelled.
        return Future()


def test_target_regain_retains_spin_until_cancel_ack(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        sent_goals: list = []
        statuses: list = []
        monkeypatch.setattr(node, "_send_goal", lambda request: sent_goals.append(request))
        monkeypatch.setattr(node, "_obstacle_points", lambda _now: ())
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        # Given a search episode with one live /spin goal.
        node._searching = True
        node._spin_accum_rad = 1.0
        spin = FakeSpin()
        monkeypatch.setattr(node, "_active_spin", spin, raising=False)
        monkeypatch.setattr(node, "_pending_spin", None, raising=False)

        # When the confirmed target reappears.
        node._target_callback(_confirmed_target(node))
        node._publish_current()

        # Then the cancel is issued but the handle is retained (no race),
        # and no follow goal resumes while the old spin is still live.
        assert spin.cancel_calls == [True]
        assert node._active_spin is spin
        assert sent_goals == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_inspection_pause_retains_spin_until_ack(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        monkeypatch.setattr(
            node, "_send_spin", lambda yaw: spin_sent.append(float(yaw)), raising=False
        )
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        spin = FakeSpin()
        monkeypatch.setattr(node, "_active_spin", spin, raising=False)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        # When fresh people arrive mid-search (inspection window).
        node._spin_search(now_ns, True)

        # Then the in-flight spin is cancelled but retained; the
        # acknowledgement alone never releases it and no replacement
        # spin is sent meanwhile.
        assert spin.cancel_calls == [True]
        assert node._active_spin is spin
        assert spin_sent == []
        assert statuses[-1] == ("searching", "search_inspecting")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_cancel_ack_without_result_keeps_follow_blocked(monkeypatch) -> None:
    import math

    rclpy, node = _make_node()
    try:
        sent_goals: list = []
        statuses: list = []
        spin_sent: list = []
        monkeypatch.setattr(node, "_send_goal", lambda request: sent_goals.append(request))
        monkeypatch.setattr(node, "_obstacle_points", lambda _now: ())
        monkeypatch.setattr(
            node, "_send_spin", lambda yaw: spin_sent.append(float(yaw)), raising=False
        )
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)
        node._searching = True
        node._spin_accum_rad = 1.0
        spin = FakeSpin()
        monkeypatch.setattr(node, "_active_spin", spin, raising=False)
        monkeypatch.setattr(node, "_pending_spin", None, raising=False)
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0

        # Given regain while the cancel acknowledgement is still delayed.
        node._target_callback(_confirmed_target(node))
        node._publish_current()
        assert node._active_spin is spin
        assert sent_goals == []

        # When the server acknowledges the cancellation but the terminal
        # Spin result has not arrived yet.
        spin.cancel_future.fire_success()

        # Then follow stays blocked: the stale handle is retained and no
        # follow goal resumes on acknowledgement alone.
        assert node._active_spin is spin
        assert node._spin_cancel_pending is True
        node._publish_current()
        assert sent_goals == []
        assert spin_sent == []
        assert statuses[-1] == ("following", "following_spin_cancelling")

        # When the terminal Spin result lands.
        _fire_spin_result(node)

        # Then the next tick resumes following without a replacement spin.
        assert node._active_spin is None
        assert node._spin_cancel_pending is False
        node._publish_current()
        assert len(sent_goals) == 1
        assert spin_sent == []
        # Revolution budget was reset on regain, never inflated by the handoff.
        assert node._spin_accum_rad == 0.0
        assert node._spin_capped is False
        assert statuses[-1] == ("following", "following_confirmed_target")
        assert all(abs(yaw) <= math.pi + 1e-9 for yaw in spin_sent)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_cancel_response_error_keeps_follow_blocked_until_result(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        sent_goals: list = []
        statuses: list = []
        monkeypatch.setattr(node, "_send_goal", lambda request: sent_goals.append(request))
        monkeypatch.setattr(node, "_obstacle_points", lambda _now: ())
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        node._searching = True
        spin = FakeSpin()
        monkeypatch.setattr(node, "_active_spin", spin, raising=False)
        monkeypatch.setattr(node, "_pending_spin", None, raising=False)

        # Given regain with an in-flight spin cancellation.
        node._target_callback(_confirmed_target(node))
        node._publish_current()
        assert node._active_spin is spin
        assert sent_goals == []

        # When the cancel response itself fails.
        spin.cancel_future.fire_error()

        # Then the handoff stays blocked until the terminal result or
        # the bounded timeout; the error never unblocks follow alone.
        assert node._active_spin is spin
        assert node._spin_cancel_pending is True
        node._publish_current()
        assert sent_goals == []
        assert statuses[-1] == ("following", "following_spin_cancelling")

        _fire_spin_result(node)
        assert node._active_spin is None
        node._publish_current()
        assert len(sent_goals) == 1
        assert statuses[-1] == ("following", "following_confirmed_target")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_cancel_requested_pending_spin_cancels_on_acceptance(monkeypatch) -> None:
    """A cancel raced with a pending send must cancel on acceptance.

    Supersedes test_stale_pending_spin_response_installs_no_goal from the
    epoch-only fix: the pending future is no longer dropped, so there is no
    orphan. The owned pending stays until the server answers; an acceptance
    installs the handle and cancels it at once, and no replacement goal may
    be sent until the terminal result arrives.
    """
    rclpy, node = _make_node()
    try:
        from rclpy.task import Future
        from types import SimpleNamespace

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

            def server_is_ready(self):
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

            def cancel_goal_async(self):
                self.cancel_calls.append(True)
                done: Future = Future()
                done.set_result(SimpleNamespace(goals_canceling=[]))
                return done

            def get_result_async(self):
                return Future()

        client = _Client()
        monkeypatch.setattr(node, "_spin_action_client", client, raising=False)

        assert node._send_spin(0.5) is True
        assert node._pending_spin is not None

        # When cancellation is requested while the send is still pending.
        node._cancel_spin()

        # Then the pending future stays owned, marked for cancel-on-accept.
        assert node._pending_spin is not None
        assert node._spin_cancel_requested is True
        assert node._send_spin(0.5) is False

        # When the server accepts the owned pending send.
        handle = _AcceptedHandle()
        accepted: Future = Future()
        accepted.set_result(handle)
        client.futures[0]._callback(accepted)

        # Then the handle installs and is cancelled at once; the handoff
        # stays blocked with no replacement goal until the terminal result.
        assert node._active_spin is handle
        assert handle.cancel_calls == [True]
        assert node._spin_cancel_pending is True
        assert node._spin_cancel_requested is False
        assert node._send_spin(0.5) is False
        assert len(sent) == 1

        # When the terminal result for the owned handle arrives.
        terminal: Future = Future()
        terminal.set_result(SimpleNamespace(status=4))
        node._spin_result_callback(terminal, handle)  # type: ignore[arg-type]

        # Then the handoff frees and a new goal may be sent.
        assert node._active_spin is None
        assert node._spin_cancel_pending is False
        assert node._send_spin(0.5) is True
        assert len(sent) == 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pending_spin_rejection_frees_handoff_without_cancel(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        from rclpy.task import Future

        class _SendFuture:
            def __init__(self):
                self._callback = None

            def add_done_callback(self, callback):
                self._callback = callback
                return None

        class _Client:
            def __init__(self):
                self.futures: list = []

            def server_is_ready(self):
                return True

            def send_goal_async(self, goal):
                future = _SendFuture()
                self.futures.append(future)
                return future

        class _RejectedHandle:
            accepted = False

        client = _Client()
        monkeypatch.setattr(node, "_spin_action_client", client, raising=False)
        node._spin_accum_rad = 1.0

        assert node._send_spin(0.5) is True
        node._cancel_spin()
        assert node._spin_cancel_requested is True

        rejected: Future = Future()
        rejected.set_result(_RejectedHandle())
        client.futures[0]._callback(rejected)

        assert node._pending_spin is None
        assert node._active_spin is None
        assert node._spin_cancel_requested is False
        assert node._spin_accum_rad == 0.5
        assert node._send_spin(0.5) is True
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_late_spin_result_cannot_clear_newer_goal(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        from rclpy.task import Future
        from types import SimpleNamespace

        old = SimpleNamespace()
        new = SimpleNamespace()
        monkeypatch.setattr(node, "_active_spin", old, raising=False)

        terminal_old: Future = Future()
        terminal_old.set_result(SimpleNamespace(status=4))
        node._spin_result_callback(terminal_old, old)  # type: ignore[arg-type]
        assert node._active_spin is None

        monkeypatch.setattr(node, "_active_spin", new, raising=False)
        late_duplicate: Future = Future()
        late_duplicate.set_result(SimpleNamespace(status=4))
        node._spin_result_callback(late_duplicate, old)  # type: ignore[arg-type]

        assert node._active_spin is new
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_spin_cancel_timeout_holds_follow_blocked_until_result(monkeypatch) -> None:
    """Timeout on a spin cancel must hold, never reopen motion.

    Replaces test_spin_cancel_timeout_unblocks_follow_when_ack_never_arrives,
    whose reopen-on-timeout expectation is obsolete and unsafe: dropping the
    owned handle without terminal evidence lets follow resume while the old
    /spin may still be live on the server. The fail-closed contract keeps
    the handoff blocked with an observable timeout status until the terminal
    Spin result arrives.
    """
    rclpy, node = _make_node()
    try:
        sent_goals: list = []
        statuses: list = []
        monkeypatch.setattr(node, "_send_goal", lambda request: sent_goals.append(request))
        monkeypatch.setattr(node, "_obstacle_points", lambda _now: ())
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        node._searching = True
        spin = FakeSpin()
        monkeypatch.setattr(node, "_active_spin", spin, raising=False)
        monkeypatch.setattr(node, "_pending_spin", None, raising=False)

        # Given regain with a cancel that is never acknowledged.
        node._target_callback(_confirmed_target(node))
        node._publish_current()
        assert node._active_spin is spin
        assert sent_goals == []

        # When the bounded wait expires (force the deadline into the past
        # so the test does not sleep for the real timeout).
        node._spin_cancel_deadline_ns = 0
        node._publish_current()

        # Then the handoff stays blocked with an observable timeout error;
        # motion never resumes on timeout alone.
        assert node._active_spin is spin
        assert node._spin_cancel_pending is True
        assert node._spin_cancel_timed_out is True
        assert sent_goals == []
        assert statuses[-1] == ("following", "following_spin_cancel_timeout")

        # A further tick still holds: the timeout never self-clears.
        node._publish_current()
        assert node._active_spin is spin
        assert sent_goals == []
        assert statuses[-1] == ("following", "following_spin_cancel_timeout")

        # When the terminal Spin result finally lands.
        _fire_spin_result(node)

        # Then the next tick resumes following.
        assert node._active_spin is None
        assert node._spin_cancel_pending is False
        assert node._spin_cancel_timed_out is False
        node._publish_current()
        assert len(sent_goals) == 1
        assert statuses[-1] == ("following", "following_confirmed_target")
    finally:
        node.destroy_node()
        rclpy.shutdown()
