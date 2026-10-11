import rclpy

from robot_autonomous_yolo_slam_nav2.metric_target_runtime import MetricTargetRuntime


def test_scan_thresholds_cover_measured_4hz_lidar_while_camera_stays_tight() -> None:
    rclpy.init(args=[])
    node = MetricTargetRuntime()
    try:
        assert float(str(node.get_parameter("max_scan_skew_s").value)) == 1.0
        assert float(str(node.get_parameter("max_scan_age_s").value)) == 1.5
        assert float(str(node.get_parameter("max_sensor_skew_s").value)) == 0.8
        assert float(str(node.get_parameter("max_data_age_s").value)) == 1.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
