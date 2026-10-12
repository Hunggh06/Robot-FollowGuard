"""Recovery-search inspection pause: hold spin while people are visible.

When a lost-target spin search sees fresh tracked people, the follower
cancels/holds the Nav2 /spin action for a short anchored window so the
existing ROI-only AprilTag confirmation can lock without the robot
rotating away. Pause = min(10s, person_count * per_person_s).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.follow_goal_helpers import (  # noqa: E402
    inspection_pause_duration_s,
)


def test_pause_is_two_seconds_per_person() -> None:
    assert inspection_pause_duration_s(1, 2.0, 10.0) == 2.0
    assert inspection_pause_duration_s(2, 2.0, 10.0) == 4.0
    assert inspection_pause_duration_s(3, 2.0, 10.0) == 6.0


def test_pause_caps_at_ten_seconds() -> None:
    assert inspection_pause_duration_s(5, 2.0, 10.0) == 10.0
    assert inspection_pause_duration_s(6, 2.0, 10.0) == 10.0
    assert inspection_pause_duration_s(10, 2.0, 10.0) == 10.0


def test_pause_needs_people() -> None:
    assert inspection_pause_duration_s(0, 1.0, 5.0) == 0.0
    assert inspection_pause_duration_s(-1, 1.0, 5.0) == 0.0


def _make_node():
    import rclpy
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    rclpy.init(args=[])
    return rclpy, Nav2PersonFollowRuntime()


def _setup_lost(node, monkeypatch, statuses, spin_sent):
    node._last_seen_bearing_rad = 0.5
    node._last_seen_range_m = 2.0
    node._last_person_speed_m_s = None

    def fake_send_spin(target_yaw):
        spin_sent.append(float(target_yaw))
        return True

    monkeypatch.setattr(node, "_send_spin", fake_send_spin, raising=False)
    monkeypatch.setattr(
        node, "_publish_status", lambda phase, reason: statuses.append((phase, reason))
    )
    monkeypatch.setattr(node._spin_action_client, "server_is_ready", lambda: True)


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


def test_spin_holds_while_people_fresh(monkeypatch) -> None:
    import math

    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)

        assert spin_sent == []
        assert node._inspection_until_ns == now_ns + 2_000_000_000
        assert statuses[-1] == ("searching", "search_inspecting")
        assert not hasattr(node, "_spin_pub")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_hold_cancels_in_flight_spin(monkeypatch) -> None:
    from types import SimpleNamespace

    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        cancelled: list = []

        class FakeSpin:
            def cancel_goal_async(self):
                cancelled.append(True)
                done = SimpleNamespace()

                class Done:
                    def add_done_callback(self, _cb):
                        return None

                return Done()

        monkeypatch.setattr(node, "_active_spin", FakeSpin(), raising=False)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)

        assert cancelled == [True]
        assert node._active_spin is None
        assert spin_sent == []
        assert statuses[-1] == ("searching", "search_inspecting")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pause_scales_with_person_count(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        for track_id in (3, 4, 5):
            _fresh_obs(node, track_id, now_ns)

        node._spin_search(now_ns, True)

        assert spin_sent == []
        assert node._inspection_until_ns == now_ns + 6_000_000_000
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pause_does_not_extend_indefinitely(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)
        anchored = node._inspection_until_ns
        assert anchored == now_ns + 2_000_000_000

        # People stay visible the whole time; the window must not grow.
        for tick_ms in (200, 400, 600, 800):
            tick_ns = now_ns + tick_ms * 1_000_000
            _fresh_obs(node, 7, tick_ns)
            _fresh_obs(node, 8, tick_ns)
            node._spin_search(tick_ns, True)
        assert node._inspection_until_ns == anchored
        assert spin_sent == []

        # Past the anchored window the spin resumes even with people present,
        # and no second window anchors within the same search episode.
        tick_ns = now_ns + 2_100_000_000
        _fresh_obs(node, 7, tick_ns)
        node._spin_search(tick_ns, True, allow_drive=False)
        assert len(spin_sent) == 1
        tick_ns = now_ns + 2_200_000_000
        _fresh_obs(node, 7, tick_ns)
        node._spin_search(tick_ns, True, allow_drive=False)
        assert len(spin_sent) == 2
        assert statuses[-1] == ("searching", "search_spinning_no_position")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stale_people_resume_spin(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)
        assert statuses[-1] == ("searching", "search_inspecting")

        # No new observations: the sighting goes stale and the spin resumes.
        node._spin_search(now_ns + 900_000_000, True, allow_drive=False)
        assert len(spin_sent) == 1
        assert statuses[-1] == ("searching", "search_spinning_no_position")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_zero_people_never_pauses(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds

        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)

        assert len(spin_sent) == 2
        assert node._inspection_until_ns == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_empty_presence_message_is_not_a_person(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 0, now_ns)

        node._spin_search(now_ns, True)

        assert len(spin_sent) == 1
        assert node._inspection_until_ns == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_active_follow_goal_still_ridden_with_people_present(monkeypatch) -> None:
    from types import SimpleNamespace

    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        monkeypatch.setattr(node, "_active_goal", SimpleNamespace(), raising=False)
        cancels: list = []
        monkeypatch.setattr(
            node, "_request_cancel", lambda reason: cancels.append(reason)
        )
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)
        node._spin_search(now_ns + 100_000_000, True)

        assert cancels == []
        assert spin_sent == []
        assert node._inspection_until_ns == 0
        assert statuses[-1] == ("searching", "search_riding_nav2_goal")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_tag_lock_exits_pause_immediately(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        from robot_autonomous_yolo_slam_nav2.msg import PersonTarget

        statuses: list = []
        spin_sent: list = []
        sent_goals: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        monkeypatch.setattr(node, "_send_goal", lambda request: sent_goals.append(request))
        monkeypatch.setattr(node, "_obstacle_points", lambda _now: ())
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)
        assert statuses[-1] == ("searching", "search_inspecting")

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
        node._target_callback(target)
        node._publish_current()

        assert node._searching is False
        assert node._inspection_until_ns == 0
        assert statuses[-1] == ("following", "following_confirmed_target")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_inspection_params_match_config() -> None:
    import yaml

    package_dir = Path(__file__).parents[1]
    person_follow = yaml.safe_load(
        (package_dir / "config" / "person_follow.yaml").read_text()
    )
    nav2_parameters = person_follow["nav2_person_follow"]["ros__parameters"]
    runtime_source = (
        package_dir
        / "robot_autonomous_yolo_slam_nav2"
        / "nav2_person_follow_runtime.py"
    ).read_text()

    assert nav2_parameters["observations_topic"] == "/yolo_target_node/observations"
    assert nav2_parameters["search_inspection_per_person_s"] == 2.0
    assert nav2_parameters["search_inspection_max_s"] == 10.0
    assert nav2_parameters["search_inspection_max_obs_age_s"] == 0.5
    assert 'declare_parameter("observations_topic"' in runtime_source
    assert 'declare_parameter("search_inspection_per_person_s"' in runtime_source
    assert 'declare_parameter("search_inspection_max_s"' in runtime_source
    assert 'declare_parameter("search_inspection_max_obs_age_s"' in runtime_source
    assert "Twist" not in runtime_source
    assert "_spin_pub" not in runtime_source


def test_default_identity_mode_is_apriltag() -> None:
    import yaml

    package_dir = Path(__file__).parents[1]
    person_follow = yaml.safe_load(
        (package_dir / "config" / "person_follow.yaml").read_text()
    )
    assert (
        person_follow["yolo_target_node"]["ros__parameters"]["identity_mode"]
        == "apriltag"
    )
    launch_source = (package_dir / "launch" / "person_follow.launch.py").read_text()
    assert 'default_value="apriltag"' in launch_source


def test_yolo_publishes_per_person_presence() -> None:
    from robot_autonomous_yolo_slam_nav2.target_state import (
        TrackId,
        TrackObservation,
    )
    from robot_autonomous_yolo_slam_nav2.yolo_target_node import (
        presence_observation_messages,
    )

    class FakeHeader:
        stamp = None
        frame_id = ""

    class FakeObs:
        def __init__(self):
            self.header = None
            self.track_id = 0
            self.confidence = 0.0
            self.state = ""
            self.valid = True

    header = FakeHeader()
    observations = (
        TrackObservation(track_id=TrackId(3), confidence=0.8, bbox=(1.0, 2.0, 9.0, 20.0)),
        TrackObservation(track_id=TrackId(4), confidence=0.7, bbox=(5.0, 6.0, 15.0, 30.0)),
    )

    messages = presence_observation_messages(observations, header, "searching", FakeObs)

    assert len(messages) == 2
    assert [message.track_id for message in messages] == [3, 4]
    # Presence must never steer metric identity: valid stays False.
    assert all(message.valid is False for message in messages)
    assert all(message.state == "searching" for message in messages)
    assert messages[0].bbox_x == 1.0
    assert messages[0].bbox_width == 8.0
    assert messages[0].bbox_height == 18.0


def test_new_track_after_pause_anchors_fresh_window(monkeypatch) -> None:
    rclpy, node = _make_node()
    try:
        statuses: list = []
        spin_sent: list = []
        _setup_lost(node, monkeypatch, statuses, spin_sent)
        now_ns = node.get_clock().now().nanoseconds
        _fresh_obs(node, 7, now_ns)

        node._spin_search(now_ns, True)
        assert statuses[-1] == ("searching", "search_inspecting")
        assert node._inspection_until_ns == now_ns + 2_000_000_000

        expired_ns = now_ns + 2_100_000_000
        node._spin_search(expired_ns, True, allow_drive=False)
        assert len(spin_sent) == 1

        fresh_ns = now_ns + 2_200_000_000
        _fresh_obs(node, 9, fresh_ns)
        node._spin_search(fresh_ns, True, allow_drive=False)

        assert len(spin_sent) == 1
        assert statuses[-1] == ("searching", "search_inspecting")
        assert node._inspection_until_ns == fresh_ns + 2_000_000_000
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_yolo_presence_helper_handles_no_people() -> None:
    from robot_autonomous_yolo_slam_nav2.yolo_target_node import (
        presence_observation_messages,
    )

    class FakeObs:
        pass

    assert presence_observation_messages((), object(), "searching", FakeObs) == ()
