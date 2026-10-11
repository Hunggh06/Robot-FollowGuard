import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.follow_goal_helpers import (  # noqa: E402
    estimate_person_speed,
    search_spin_command,
)


def test_speed_from_two_samples() -> None:
    assert estimate_person_speed(((0.0, 0.0), 1.0), ((1.0, 0.0), 2.0)) == 1.0


def test_speed_static_person_is_zero() -> None:
    assert estimate_person_speed(((1.0, 1.0), 1.0), ((1.0, 1.0), 2.0)) == 0.0


def test_speed_rejects_bad_windows() -> None:
    assert estimate_person_speed(None, ((1.0, 0.0), 2.0)) is None
    assert estimate_person_speed(((0.0, 0.0), 1.0), ((1.0, 0.0), 1.05)) is None
    assert estimate_person_speed(((0.0, 0.0), 1.0), ((1.0, 0.0), 5.0)) is None


def test_spin_matches_person_speed_over_range() -> None:
    assert search_spin_command(0.5, 1.0, 2.0, 0.5, 0.2, 1.0, 0.1) == 0.5
    assert search_spin_command(-0.5, 1.0, 2.0, 0.5, 0.2, 1.0, 0.1) == -0.5


def test_spin_clamps_to_limits() -> None:
    assert search_spin_command(0.5, 5.0, 1.0, 0.5, 0.2, 1.0, 0.1) == 1.0
    assert search_spin_command(0.5, 0.01, 2.0, 0.5, 0.2, 1.0, 0.1) == 0.2


def test_spin_falls_back_without_estimate() -> None:
    assert search_spin_command(0.5, None, 2.0, 0.5, 0.2, 1.0, 0.1) == 0.5
    assert search_spin_command(0.5, 0.0, 2.0, 0.5, 0.2, 1.0, 0.1) == 0.5
    assert search_spin_command(None, 1.0, 2.0, 0.5, 0.2, 1.0, 0.1) is None
    assert search_spin_command(0.05, 1.0, 2.0, 0.5, 0.2, 1.0, 0.1) == 0.0


def test_unconfirmed_fresh_target_refreshes_search_aim() -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        target = PersonTarget()
        target.header.stamp = node.get_clock().now().to_msg()
        target.header.frame_id = "base_link"
        target.track_id = 9
        target.state = "tracking"
        target.valid = True
        target.identity_confirmed = False
        target.point_base_link.x = 2.0
        target.range_m = 2.0
        target.bearing_rad = 0.4
        node._target_callback(target)
        node._publish_current()
        assert node._last_seen_bearing_rad == 0.4
        assert node._searching is True
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_lost_drives_briefly_before_spinning_in_place(monkeypatch) -> None:
    import math

    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person with a usable last-seen bearing, recovered via
    # bounded /spin goals only (no direct Twist, no NavigateToPose recovery).
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None

        def fake_send_goal(request):
            sent.append(request)
            return True

        def fake_send_spin(target_yaw):
            spin_sent.append(float(target_yaw))
            return True

        monkeypatch.setattr(node, "_send_goal", fake_send_goal)
        monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)
        now_ns = node.get_clock().now().nanoseconds

        # When loss starts inside the drive window, then past it.
        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)
        # Then the early legs ride bounded /spin goals with drive status,
        # never a NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 2
        assert all(abs(yaw) <= math.pi + 1e-9 for yaw in spin_sent)
        assert all(abs(yaw) > 0.0 for yaw in spin_sent)
        assert statuses[-1] == ("searching", "search_driving")

        node._spin_search(now_ns + 3_000_000_000, True)
        # Then the late leg stays on bounded /spin goals with spin status,
        # never Twist and never a NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 3
        assert all(abs(yaw) <= math.pi + 1e-9 for yaw in spin_sent)
        assert statuses[-1] == ("searching", "search_spinning")
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_spin_rides_nav2_goal_without_cancel_or_twist(monkeypatch) -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given an active Nav2 goal while the target is lost.
    sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        from types import SimpleNamespace
        monkeypatch.setattr(node, "_pending_goal", SimpleNamespace(), raising=False)
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = 0.5
        assert not hasattr(node, "_spin_pub")
        monkeypatch.setattr(node, "_send_goal", lambda request: sent.append(request))
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        now_ns = node.get_clock().now().nanoseconds

        # When loss ticks arrive with the goal still live.
        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)

        # Then the live goal is ridden: no cancel, no replacement, no Twist.
        assert node._searching is True
        assert node._cancel_requested is False
        assert node._pending_goal is not None
        assert sent == []
        assert statuses[-1] == ("searching", "search_riding_nav2_goal")
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_spin_stops_after_two_revolutions(monkeypatch) -> None:
    import math

    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person and /spin-only recovery, capped at two revolutions.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        assert not hasattr(node, "_spin_pub")

        def fake_send_goal(request):
            sent.append(request)
            return True

        def fake_send_spin(target_yaw):
            spin_sent.append(float(target_yaw))
            return True

        monkeypatch.setattr(node, "_send_goal", fake_send_goal)
        monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)
        now_ns = node.get_clock().now().nanoseconds

        # When loss persists well past the revolution budget.
        for tick in range(300):
            node._spin_search(now_ns + tick * 100_000_000, True, allow_drive=False)

        # Then motion stops at the cap: bounded /spin goals were sent, the
        # budget holds, no NavigateToPose recovery goal was ever used, and
        # no further spins follow once capped.
        assert node._spin_capped is True
        assert node._spin_accum_rad >= 2.0 * 2.0 * math.pi - 1e-9
        assert sent == []
        assert len(spin_sent) > 0
        assert all(abs(yaw) <= math.pi + 1e-9 for yaw in spin_sent)
        assert sum(abs(yaw) for yaw in spin_sent) <= 2.0 * 2.0 * math.pi + 1e-9
        assert statuses[-1] == ("searching", "search_spin_done")
        count_at_cap = len(spin_sent)
        for tick in range(300, 310):
            node._spin_search(now_ns + tick * 100_000_000, True, allow_drive=False)
        assert len(spin_sent) == count_at_cap
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_accumulates_no_yaw_when_goal_cannot_send(monkeypatch) -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person whose /spin goal cannot be sent (server not ready).
    sent = []
    spin_attempts = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None

        def fake_send_goal(request):
            sent.append(request)
            return True

        def fake_send_spin(target_yaw):
            spin_attempts.append(float(target_yaw))
            return False

        monkeypatch.setattr(node, "_send_goal", fake_send_goal)
        monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        now_ns = node.get_clock().now().nanoseconds

        # When recovery ticks while every /spin send fails.
        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)
        node._spin_search(now_ns + 200_000_000, True)

        # Then no yaw budget may accumulate on failed sends, and no
        # NavigateToPose recovery goal is used as a fallback.
        assert len(spin_attempts) == 3
        assert node._spin_accum_rad == 0.0
        assert sent == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_requests_bounded_incremental_yaw(monkeypatch) -> None:
    import math

    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person with a large absolute bearing plus large prior
    # accumulation, recovered via one bounded /spin goal.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._last_seen_bearing_rad = 2.0
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = None
        now_ns = node.get_clock().now().nanoseconds
        node._searching = True
        node._lost_since_ns = now_ns
        node._last_spin_ns = now_ns
        node._spin_accum_rad = 5.0
        node._spin_capped = False

        def fake_send_goal(request):
            sent.append(request)
            return True

        def fake_send_spin(target_yaw):
            spin_sent.append(float(target_yaw))
            return True

        monkeypatch.setattr(node, "_send_goal", fake_send_goal)
        monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)

        # When one recovery tick fires.
        node._spin_search(now_ns + 100_000_000, True)

        # Then recovery uses one bounded incremental /spin turn (no larger
        # than pi), not the unbounded absolute bearing + accumulation
        # (2.0 + 5.0), and never a NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 1
        assert abs(spin_sent[0]) <= math.pi + 1e-9
        assert abs(spin_sent[0]) < abs(2.0 + 5.0) - 1e-9
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_follow_loss_rides_active_follow_goal_and_defers_spin(monkeypatch) -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given an active follow NavigateToPose goal while the target is lost.
    sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        from types import SimpleNamespace
        monkeypatch.setattr(node, "_active_goal", SimpleNamespace(), raising=False)
        node._pending_goal = None
        node._last_goal_mode = "follow"
        node._last_seen_bearing_rad = 0.5
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = 0.5
        assert not hasattr(node, "_spin_pub")
        monkeypatch.setattr(node, "_send_goal", lambda request: sent.append(request))
        monkeypatch.setattr(
            node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
        )
        now_ns = node.get_clock().now().nanoseconds

        # When loss ticks arrive with the follow goal still live.
        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)

        # Then the live follow goal is ridden without cancellation or
        # replacement, and the bounded /spin seam exists but stays deferred
        # while the follow goal is still in flight.
        assert node._searching is True
        assert node._cancel_requested is False
        assert node._active_goal is not None
        assert sent == []
        assert statuses[-1] == ("searching", "search_riding_nav2_goal")
        assert getattr(node, "_pending_spin", None) is None
        assert getattr(node, "_active_spin", None) is None
        assert hasattr(node, "_spin_action_client")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _extract_spin_yaw(entry) -> float | None:
    import math

    args, kwargs = entry
    candidates = list(args) + list(kwargs.values())
    for candidate in candidates:
        target = getattr(candidate, "target_yaw", candidate)
        if isinstance(target, bool):
            continue
        if isinstance(target, (int, float)) and math.isfinite(float(target)):
            return float(target)
    return None


def _completed_nav2_future(status: int):
    from types import SimpleNamespace
    from rclpy.task import Future

    future: Future = Future()
    future.set_result(SimpleNamespace(status=status))
    return future


def _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent):
    import rclpy  # noqa: F401
    from types import SimpleNamespace
    from robot_autonomous_yolo_slam_nav2.goal_update_logic import GoalSnapshot
    import robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime as runtime_mod

    node._last_seen_bearing_rad = 0.5
    node._last_seen_range_m = 2.0
    node._last_person_speed_m_s = None

    def fake_send(request):
        sent.append(request)
        return True

    def fake_send_spin(*args, **kwargs):
        spin_sent.append((args, kwargs))
        return True

    monkeypatch.setattr(node, "_send_goal", fake_send)
    monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
    monkeypatch.setattr(
        node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
    )
    monkeypatch.setattr(
        runtime_mod, "robot_map_pose", lambda *a, **k: (0.0, 0.0, 0.0), raising=False
    )
    monkeypatch.setattr(node._action_client, "server_is_ready", lambda: True)
    node._active_goal = SimpleNamespace()
    node._pending_goal = None
    node._queued_goal = None
    node._last_goal_map = GoalSnapshot((0.0, 0.0), 0.0, "follow")
    node._last_goal_mode = "follow"
    node._cancel_requested = False


def test_terminal_follow_success_enables_bounded_spin(monkeypatch) -> None:
    import math

    import rclpy
    from action_msgs.msg import GoalStatus
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person whose active follow goal just succeeded.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent)
        done = _completed_nav2_future(GoalStatus.STATUS_SUCCEEDED)

        # When the follow result lands and loss ticks once more.
        node._result_callback(done)
        assert node._active_goal is None
        now_ns = node.get_clock().now().nanoseconds
        node._spin_search(now_ns, True)

        # Then recovery uses one bounded /spin goal, never a replacement
        # NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 1
        yaw = _extract_spin_yaw(spin_sent[0])
        assert yaw is not None
        assert abs(yaw) <= math.pi + 1e-9
        assert abs(yaw) > 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_terminal_follow_canceled_enables_bounded_spin(monkeypatch) -> None:
    import math

    import rclpy
    from action_msgs.msg import GoalStatus
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person whose active follow goal just canceled.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent)
        done = _completed_nav2_future(GoalStatus.STATUS_CANCELED)

        # When the follow cancel lands and loss ticks once more.
        node._result_callback(done)
        assert node._active_goal is None
        now_ns = node.get_clock().now().nanoseconds
        node._spin_search(now_ns, True)

        # Then recovery uses one bounded /spin goal, never a replacement
        # NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 1
        yaw = _extract_spin_yaw(spin_sent[0])
        assert yaw is not None
        assert abs(yaw) <= math.pi + 1e-9
        assert abs(yaw) > 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_terminal_follow_aborted_enables_bounded_spin(monkeypatch) -> None:
    import math

    import rclpy
    from action_msgs.msg import GoalStatus
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person whose active follow goal just aborted.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent)
        done = _completed_nav2_future(GoalStatus.STATUS_ABORTED)

        # When the follow abort lands and loss ticks once more.
        node._result_callback(done)
        assert node._active_goal is None
        now_ns = node.get_clock().now().nanoseconds
        node._spin_search(now_ns, True)

        # Then recovery uses one bounded /spin goal, never a replacement
        # NavigateToPose recovery goal.
        assert sent == []
        assert len(spin_sent) == 1
        yaw = _extract_spin_yaw(spin_sent[0])
        assert yaw is not None
        assert abs(yaw) <= math.pi + 1e-9
        assert abs(yaw) > 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_one_in_flight_spin_blocks_second_spin(monkeypatch) -> None:
    import rclpy
    from types import SimpleNamespace
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person with one /spin already in flight and no follow goal.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent)
        node._active_goal = None
        node._pending_goal = None
        node._last_goal_mode = None
        from types import SimpleNamespace
        monkeypatch.setattr(node, "_pending_spin", SimpleNamespace(), raising=False)
        now_ns = node.get_clock().now().nanoseconds

        # When loss ticks with the spin still live.
        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)

        # Then no second spin and no NavigateToPose recovery goal is sent.
        assert sent == []
        assert spin_sent == []
        assert getattr(node, "_pending_spin", None) is not None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_spin_rotation_cap_stops_further_spins(monkeypatch) -> None:
    import math

    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given a lost person with no follow goal and a fresh spin budget.
    sent = []
    spin_sent = []
    statuses = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        _terminal_follow_to_spin_setup(node, monkeypatch, statuses, sent, spin_sent)
        node._active_goal = None
        node._pending_goal = None
        node._last_goal_mode = None
        now_ns = node.get_clock().now().nanoseconds

        # When loss persists well past the two-revolution budget.
        for tick in range(300):
            node._spin_search(now_ns + tick * 100_000_000, True)

        # Then /spin goals were sent, each bounded, the total rotation is
        # capped at two revolutions, and no NavigateToPose recovery goal
        # was ever used for the search.
        assert sent == []
        assert len(spin_sent) > 0
        yaws = [_extract_spin_yaw(entry) for entry in spin_sent]
        assert all(yaw is not None for yaw in yaws)
        yaw_values = [yaw for yaw in yaws if yaw is not None]
        assert all(abs(yaw) <= math.pi + 1e-9 for yaw in yaw_values)
        total = sum(abs(yaw) for yaw in yaw_values)
        assert total <= 2.0 * 2.0 * math.pi + 1e-9
        count_at_cap = len(spin_sent)
        for tick in range(300, 310):
            node._spin_search(now_ns + tick * 100_000_000, True)
        assert len(spin_sent) == count_at_cap
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_guide_active_yields_nav2_without_motion(monkeypatch) -> None:
    import rclpy
    from std_msgs.msg import Bool
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    # Given an active guide override while following.
    sent = []
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        assert not hasattr(node, "_spin_pub")
        monkeypatch.setattr(node, "_send_goal", lambda request: sent.append(request))
        node._guide_callback(Bool(data=True))
        node._publish_current()
        node._publish_current()

        # Then follow yields: one cancel, no Nav2 recovery goals, no Twist.
        assert node._cancel_requested is True
        assert node._guide_yielded is True
        assert sent == []
        assert not hasattr(node, "_spin_pub")
        node._guide_callback(Bool(data=False))
        assert node._guide_active is False
    finally:
        node.destroy_node()
        rclpy.shutdown()
