import rclpy
import pytest
from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
from robot_autonomous_yolo_slam_nav2.metric_association import MetricPersonObservation
from robot_autonomous_yolo_slam_nav2.metric_target_ros import metric_target_message
from robot_autonomous_yolo_slam_nav2.nav2_person_follow_runtime import Nav2PersonFollowRuntime
from robot_autonomous_yolo_slam_nav2.target_state import TrackId


def test_metric_pose_preserves_confirmed_identity_but_invalid_pose_does_not():
    source = PersonTarget()
    source.track_id = 20
    source.state = "tracking"
    source.valid = True
    source.identity_confirmed = True
    metric = MetricPersonObservation(TrackId(20), (2.0, 0.0, 0.0), 1_000_000_000)
    target = metric_target_message(source, metric, "base_link")
    assert target.valid and target.identity_confirmed and target.track_id == 20
    invalid = metric_target_message(source, None, "base_link")
    assert not invalid.valid and not invalid.identity_confirmed


def test_nav2_rejects_unconfirmed_tracking_message():
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        target = PersonTarget()
        target.header.stamp = node.get_clock().now().to_msg()
        target.header.frame_id = "base_link"
        target.track_id = 7
        target.state = "tracking"
        target.valid = True
        target.point_base_link.x = 2.0
        target.range_m = 2.0
        geometry = node._target_geometry(target, node.get_clock().now().nanoseconds)
        assert not node._is_confirmed_target(target, geometry)
        target.identity_confirmed = True
        assert node._is_confirmed_target(target, geometry)
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("confirmed", [False, True])
def test_nav2_track_id_change_requires_reconfirmed_identity(confirmed):
    rclpy.init(args=[])
    node = Nav2PersonFollowRuntime()
    try:
        node._confirmed_track_id = 7
        target = PersonTarget()
        target.header.stamp = node.get_clock().now().to_msg()
        target.header.frame_id = "base_link"
        target.track_id = 20
        target.state = "tracking"
        target.valid = True
        target.identity_confirmed = confirmed
        target.point_base_link.x = 2.0
        target.range_m = 2.0
        node._target_callback(target)
        node._publish_current()
        assert node._confirmed_track_id == (20 if confirmed else 7)
        assert node._active_goal is None
    finally:
        node.destroy_node()
        rclpy.shutdown()
