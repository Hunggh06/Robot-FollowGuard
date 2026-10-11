from robot_autonomous_yolo_slam_nav2.msg import PersonTarget
from robot_autonomous_yolo_slam_nav2.metric_association import MetricPersonObservation
from robot_autonomous_yolo_slam_nav2.metric_target_ros import held_target_message, metric_target_message
from robot_autonomous_yolo_slam_nav2.target_state import TrackId


def test_lidar_message_keeps_lidar_measurement_stamp_not_newer_visual_stamp():
    source = PersonTarget()
    source.header.stamp.sec = 10
    source.track_id = 5
    source.identity_confirmed = True
    metric = MetricPersonObservation(TrackId(5), (2.0, 0.0, 0.0), 9_500_000_000)
    output = metric_target_message(source, metric, "base_link")
    assert (output.header.stamp.sec, output.header.stamp.nanosec) == (9, 500_000_000)
    assert source.header.stamp.sec == 10


def test_held_message_does_not_mutate_visual_source_timestamp():
    source = PersonTarget()
    source.header.stamp.sec = 10
    source.track_id = 5
    output = held_target_message(source, 5, (2.0, 0.0, 0.0), 2.0, 0.0, 8, 0,
                                 0.5, "retained", "base_link")
    assert output.header.stamp.sec == 8
    assert source.header.stamp.sec == 10
    assert not output.valid
