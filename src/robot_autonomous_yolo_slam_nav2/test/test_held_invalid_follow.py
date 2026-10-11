import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.follow_goal_helpers import (  # noqa: E402
    FollowConfig,
    FollowDecisionStatus,
    TargetGeometry,
    decide_follow_goal,
)


def _config() -> FollowConfig:
    return FollowConfig(
        follow_distance_m=1.0,
        max_target_age_s=0.5,
        corridor_half_width_m=0.4,
        goal_update_distance_m=0.2,
        base_frame="base_link",
    )


def test_held_invalid_geometry_never_produces_nav2_goal() -> None:
    target = TargetGeometry(
        state="tracking",
        valid=False,
        point_base_link=(2.0, 0.0, -0.01),
        range_m=2.0,
        bearing_rad=0.0,
        age_s=2.5,
        frame_id="base_link",
    )
    decision = decide_follow_goal(target, (), _config(), None)
    assert decision.status is FollowDecisionStatus.CANCEL
    assert decision.goal_base_link is None
    assert decision.cancel_active_goal


@pytest.mark.parametrize(("source", "confirmed"), [("held_invalid", True), ("none", False)])
def test_held_target_does_not_refresh_search_aim_or_drive(source, confirmed) -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._last_seen_bearing_rad = 0.9
        node._last_seen_range_m = 2.0
        node._last_person_speed_m_s = 0.5
        statuses: list = []
        node._publish_status = lambda phase, reason: statuses.append((phase, reason))  # type: ignore[method-assign]
        now_ns = node.get_clock().now().nanoseconds
        node._scan_points = ((3.0, 0.0),)
        node._scan_stamp_ns = now_ns
        target = PersonTarget()
        target.header.stamp = node.get_clock().now().to_msg()
        target.header.frame_id = "base_link"
        target.track_id = 9
        target.state = "tracking"
        target.valid = False
        target.identity_confirmed = confirmed
        target.point_base_link.x = 2.0
        target.point_base_link.z = -0.01
        target.range_m = 2.0
        target.bearing_rad = 0.4
        target.position_source = source
        node._target_callback(target)
        node._publish_current()
        assert node._last_seen_bearing_rad == 0.9
        assert node._last_person_speed_m_s == 0.5
        assert node._active_goal is None
        assert node._pending_goal is None
        assert not hasattr(node, "_spin_pub")
        if confirmed:
            assert statuses[-1] == ("searching", "search_driving")
        else:
            assert statuses[-1] == ("searching", "search_spinning_no_position")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stale_visual_contact_drives_nothing() -> None:
    import rclpy
    from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
    from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import (
        Nav2PersonFollowRuntime,
    )

    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._last_seen_bearing_rad = 0.9
        statuses: list = []
        node._publish_status = lambda phase, reason: statuses.append((phase, reason))  # type: ignore[method-assign]
        now_ns = node.get_clock().now().nanoseconds
        node._scan_points = ((3.0, 0.0),)
        node._scan_stamp_ns = now_ns
        target = PersonTarget()
        stale_ns = now_ns - 5_000_000_000
        target.header.stamp.sec = stale_ns // 1_000_000_000
        target.header.stamp.nanosec = stale_ns % 1_000_000_000
        target.header.frame_id = "base_link"
        target.track_id = 9
        target.state = "tracking"
        target.valid = False
        target.identity_confirmed = True
        target.point_base_link.x = 2.0
        target.range_m = 2.0
        target.bearing_rad = 0.4
        target.position_source = "held_invalid"
        node._target_callback(target)
        node._publish_current()
        assert node._active_goal is None
        assert node._pending_goal is None
        assert not hasattr(node, "_spin_pub")
        assert statuses[-1] == ("searching", "search_spinning_no_position")
    finally:
        node.destroy_node()
        rclpy.shutdown()
