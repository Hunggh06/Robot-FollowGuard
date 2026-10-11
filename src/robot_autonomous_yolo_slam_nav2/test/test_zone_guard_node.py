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


def test_confirm_sets_pending_flag() -> None:
    import rclpy
    from std_msgs.msg import String

    node = _make_node()
    try:
        node._confirm_callback(String(data="ok xác nhận nhé"))
        assert node._confirm_pending is True
        assert node._consume_confirm() is True
        assert node._confirm_pending is False
        node._confirm_callback(String(data="đi tiếp đi"))
        assert node._confirm_pending is False
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


def test_confirm_ignored_right_after_robot_speaks() -> None:
    import rclpy
    from std_msgs.msg import String

    node = _make_node()
    try:
        now_ns = node.get_clock().now().nanoseconds
        node._last_say_ns = now_ns
        node._confirm_callback(String(data="xác nhận"))
        assert node._confirm_pending is False
        node._last_say_ns = now_ns - 10_000_000_000
        node._confirm_callback(String(data="xác nhận"))
        assert node._confirm_pending is True
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
        node._confirm_pending = True

        cancelled = []

        class _Handle:
            def cancel_goal_async(self):
                cancelled.append("guide")
                return None

        class _VerifyHandle:
            def cancel_goal_async(self):
                cancelled.append("verify")
                return None

        node._guide_handle = None
        setattr(node, "_guide_handle", _Handle())
        setattr(node, "_guide_pending", object())
        setattr(node, "_verify_handle", _VerifyHandle())
        setattr(node, "_verify_pending", object())
        node._saved_max_vel_x = 0.26
        restored = []
        node._set_controller_param = lambda name, value: restored.append((name, value))

        response = node._reset_zone(Trigger.Request(), Trigger.Response())

        assert response.success is True
        assert node._polygon is None
        assert node._phase == "IDLE"
        assert node._guide_active is False
        assert node._guide_handle is None
        assert node._guide_pending is None
        assert node._verify_handle is None
        assert node._verify_pending is None
        assert node._guide_done is False
        assert node._guide_succeeded is False
        assert node._guide_fails == 0
        assert node._guide_poly is None
        assert node._confirm_pending is False
        assert "guide" in cancelled
        assert "verify" in cancelled
        assert node._saved_max_vel_x is None
        assert ("max_vel_x", 0.26) in restored
        node._tick()
        assert node._phase == "IDLE"
        assert node._guide_active is False
        assert node._guide_handle is None
        assert node._guide_pending is None
    finally:
        node.destroy_node()
        rclpy.shutdown()
