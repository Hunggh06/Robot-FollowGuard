import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


def _make_msgs(node, track_id, foot_valid, bad_calib=False):
    from robot_autonomous_yolo_slam_nav2.msg import (
        PersonObservation,
        PersonTarget,
    )
    from sensor_msgs.msg import CameraInfo

    header_stamp = node.get_clock().now().to_msg()
    target = PersonTarget()
    target.header.stamp = header_stamp
    target.header.frame_id = "phone_camera_back"
    target.track_id = track_id
    target.confidence = 0.9
    target.bbox_x, target.bbox_y = 100.0, 20.0
    target.bbox_width, target.bbox_height = 120.0, 212.0
    target.state = "tracking"
    target.valid = True
    target.identity_confirmed = True
    target.footpoint_valid = foot_valid
    target.footpoint_x, target.footpoint_y = 160.0, 180.0
    target.footpoint_confidence = 0.8
    obs = PersonObservation()
    obs.header.stamp = header_stamp
    obs.header.frame_id = "phone_camera_back"
    obs.track_id = track_id
    obs.confidence = 0.9
    obs.bbox_x, obs.bbox_y = 100.0, 20.0
    obs.bbox_width, obs.bbox_height = 120.0, 212.0
    obs.state = "tracking"
    obs.valid = True
    info = CameraInfo()
    info.header.stamp = header_stamp
    info.header.frame_id = "phone_camera_back"
    info.width, info.height = 320, 240
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    if bad_calib:
        info.k = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else:
        info.k = [136.92, 0.0, 159.02, 0.0, 136.78, 118.77, 0.0, 0.0, 1.0]
    return target, obs, info


class FakePub:
    def __init__(self) -> None:
        self.msgs = []

    def publish(self, message) -> None:
        self.msgs.append(message)


def _node_with_seed():
    import rclpy
    from robot_autonomous_yolo_slam_nav2.metric_target_runtime import (
        MetricTargetRuntime,
    )

    rclpy.init(args=[])
    node = MetricTargetRuntime()
    node._target_pub = FakePub()
    node._obstacles_pub = FakePub()
    seed_stamp = node.get_clock().now().nanoseconds - 1_000_000_000
    node._cache.update(
        track_id=5, point=(2.0, 0.0, -0.01), range_m=2.0, bearing_rad=0.0,
        stamp_ns=seed_stamp, source="lidar", confidence=0.9,
        provenance="lidar_bbox",
    )
    return node, seed_stamp


def test_tf_failure_retains_held_invalid_for_same_track() -> None:
    import rclpy

    node, seed_stamp = _node_with_seed()
    try:
        target, obs, info = _make_msgs(node, 5, False)
        node._target_callback(target)
        node._observation_callback(obs)
        node._camera_info_callback(info)
        node._scan = None
        node._publish_current()
        assert len(node._target_pub.msgs) == 1
        held = node._target_pub.msgs[0]
        assert not held.valid
        assert held.position_source == "held_invalid"
        assert held.track_id == 5
        assert (held.header.stamp.sec * 1_000_000_000
                + held.header.stamp.nanosec) == seed_stamp
        assert held.range_m == 2.0
        assert 0.0 < held.position_confidence < 0.9
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_bad_calibration_retains_held_invalid_for_same_track() -> None:
    import rclpy

    node, seed_stamp = _node_with_seed()
    try:
        target, obs, info = _make_msgs(node, 5, False, bad_calib=True)
        node._target_callback(target)
        node._observation_callback(obs)
        node._camera_info_callback(info)
        node._scan = None
        node._publish_current()
        held = node._target_pub.msgs[0]
        assert not held.valid
        assert held.position_source == "held_invalid"
        assert (held.header.stamp.sec * 1_000_000_000
                + held.header.stamp.nanosec) == seed_stamp
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_missing_camera_info_retains_held_invalid() -> None:
    import rclpy

    node, seed_stamp = _node_with_seed()
    try:
        target, obs, _info = _make_msgs(node, 5, False)
        node._target_callback(target)
        node._observation_callback(obs)
        node._camera_info = None
        node._scan = None
        node._publish_current()
        held = node._target_pub.msgs[0]
        assert not held.valid
        assert held.position_source == "held_invalid"
        assert (held.header.stamp.sec * 1_000_000_000
                + held.header.stamp.nanosec) == seed_stamp
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_other_track_id_clears_cache_without_reuse() -> None:
    import rclpy

    node, _seed_stamp = _node_with_seed()
    try:
        target, obs, info = _make_msgs(node, 6, False)
        node._target_callback(target)
        node._observation_callback(obs)
        node._camera_info_callback(info)
        node._scan = None
        node._publish_current()
        invalid = node._target_pub.msgs[0]
        assert not invalid.valid
        assert invalid.position_source == "none"
        assert invalid.track_id == 6
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize(("valid", "confirmed"), [(False, True), (True, False)])
def test_invalid_visual_identity_cannot_reuse_or_refresh_geometry(valid, confirmed):
    import rclpy

    node, _seed_stamp = _node_with_seed()
    try:
        target, obs, info = _make_msgs(node, 5, True)
        target.valid = valid
        target.identity_confirmed = confirmed
        node._target_callback(target)
        node._observation_callback(obs)
        node._camera_info_callback(info)
        node._publish_current()
        output = node._target_pub.msgs[-1]
        assert not output.valid
        assert output.position_source == "none"
        assert output.provenance == "visual_target_unconfirmed"
        assert node._cache.held(5, node.get_clock().now().nanoseconds) is None
    finally:
        node.destroy_node()
        rclpy.shutdown()
