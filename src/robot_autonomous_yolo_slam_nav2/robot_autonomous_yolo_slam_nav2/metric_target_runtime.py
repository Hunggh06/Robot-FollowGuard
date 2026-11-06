"""ROS runtime shell for metric target positions with provenance.

I/O shell only: subscriptions, parameters, TF lookups, publishers, and
the retained-position cache. One publish cycle lives in
metric_target_cycle, branch decisions in metric_target_branches.
"""

from __future__ import annotations

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from robot_autonomous_yolo_slam_nav2.msg import PersonObservation, PersonTarget

from .metric_association import BaseToCameraTransform, Point3D, RotationMatrix
from .metric_target_cycle import run_cycle
from .metric_target_helpers import LaserToBaseTransform
from .metric_target_ros import transform_values
from .position_cache import PositionCache
from .target_kalman_filter import KalmanFusionConfig, TargetKalmanFilter


class MetricTargetRuntime(Node):
    """Associate locked visual targets with Camera + LiDAR Kalman fusion or held geometry."""

    def __init__(self) -> None:
        super().__init__("metric_target_node")
        self._declare_parameters()
        self._target: PersonTarget | None = None
        self._observations: dict[int, PersonObservation] = {}
        self._scan: LaserScan | None = None
        self._camera_info: CameraInfo | None = None
        self._cache = PositionCache(
            hold_timeout_s=float(str(self.get_parameter("hold_timeout_s").value)),
            decay_per_s=float(str(self.get_parameter("hold_decay_per_s").value)),
        )
        self._kalman = TargetKalmanFilter(
            KalmanFusionConfig(
                process_pos_noise=float(
                    str(self.get_parameter("kf_process_pos_noise").value)
                ),
                process_vel_noise=float(
                    str(self.get_parameter("kf_process_vel_noise").value)
                ),
                velocity_damping=float(
                    str(self.get_parameter("kf_velocity_damping").value)
                ),
                max_speed_m_s=float(
                    str(self.get_parameter("kf_max_speed_m_s").value)
                ),
                max_coast_s=float(
                    str(self.get_parameter("kf_max_coast_s").value)
                ),
                lidar_coast_s=float(
                    str(self.get_parameter("kf_lidar_coast_s").value)
                ),
                camera_bearing_std_rad=float(
                    str(self.get_parameter("kf_camera_bearing_std_rad").value)
                ),
                camera_range_rel_std=float(
                    str(self.get_parameter("kf_camera_range_rel_std").value)
                ),
                expected_person_width_m=float(
                    str(self.get_parameter("kf_expected_person_width_m").value)
                ),
                expected_person_height_m=float(
                    str(self.get_parameter("kf_expected_person_height_m").value)
                ),
                bearing_window_half_angle_rad=float(
                    str(self.get_parameter("kf_bearing_window_half_angle_rad").value)
                ),
                mahalanobis_gate_sq=float(
                    str(self.get_parameter("kf_mahalanobis_gate_sq").value)
                ),
                allow_camera_only=bool(
                    self.get_parameter("kf_allow_camera_only").value
                ),
            )
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._target_pub = self.create_publisher(
            PersonTarget,
            str(self.get_parameter("metric_target_topic").value),
            5,
        )
        self._obstacles_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("obstacles_topic").value),
            5,
        )
        self.create_subscription(
            PersonTarget,
            str(self.get_parameter("target_topic").value),
            self._target_callback,
            5,
        )
        self.create_subscription(
            PersonObservation,
            str(self.get_parameter("observations_topic").value),
            self._observation_callback,
            20,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            5,
        )
        period = float(str(self.get_parameter("publish_period_s").value))
        self.create_timer(period, self._publish_current)

    def _declare_parameters(self) -> None:
        self.declare_parameter("target_topic", "/yolo_target_node/target")
        self.declare_parameter("observations_topic", "/yolo_target_node/observations")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter(
            "camera_info_topic",
            "/phone/camera/back/camera_info",
        )
        self.declare_parameter("obstacles_topic", "/person_follow/obstacles")
        self.declare_parameter("metric_target_topic", "/person_follow/target")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "phone_camera_back")
        self.declare_parameter("max_data_age_s", 1.0)
        self.declare_parameter("max_sensor_skew_s", 0.8)
        self.declare_parameter("max_scan_age_s", 1.5)
        self.declare_parameter("max_scan_skew_s", 1.0)
        self.declare_parameter("association_image_width", 0)
        self.declare_parameter("association_image_height", 0)
        self.declare_parameter("lidar_position_confidence", 0.95)
        self.declare_parameter("hold_timeout_s", 5.0)
        self.declare_parameter("hold_decay_per_s", 0.05)
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("use_kalman_fusion", True)
        self.declare_parameter("kf_process_pos_noise", 0.15)
        self.declare_parameter("kf_process_vel_noise", 0.60)
        self.declare_parameter("kf_velocity_damping", 0.85)
        self.declare_parameter("kf_max_speed_m_s", 2.2)
        self.declare_parameter("kf_max_coast_s", 1.2)
        self.declare_parameter("kf_lidar_coast_s", 1.5)
        self.declare_parameter("kf_camera_bearing_std_rad", 0.035)
        self.declare_parameter("kf_camera_range_rel_std", 0.25)
        self.declare_parameter("kf_expected_person_width_m", 0.45)
        self.declare_parameter("kf_expected_person_height_m", 1.68)
        self.declare_parameter("kf_bearing_window_half_angle_rad", 0.22)
        self.declare_parameter("kf_mahalanobis_gate_sq", 16.0)
        self.declare_parameter("kf_allow_camera_only", True)

    def _target_callback(self, message: PersonTarget) -> None:
        if self._target is not None and message.track_id != self._target.track_id:
            self._cache.reset()
            self._kalman.reset()
        self._target = message

    def _observation_callback(self, message: PersonObservation) -> None:
        self._observations[message.track_id] = message
        if len(self._observations) > 64:
            now_ns = self.get_clock().now().nanoseconds
            max_age_ns = int(float(str(self.get_parameter("max_data_age_s").value)) * 2_000_000_000)
            stale_keys = [
                tid
                for tid, obs in self._observations.items()
                if now_ns - (int(obs.header.stamp.sec) * 1_000_000_000 + int(obs.header.stamp.nanosec)) > max_age_ns
            ]
            for tid in stale_keys:
                self._observations.pop(tid, None)

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan = message

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _publish_current(self) -> None:
        run_cycle(self)

    def _base_frame(self) -> str:
        return str(self.get_parameter("base_frame").value)

    def _fallback_image_size(self) -> tuple[int, int] | None:
        width = int(str(self.get_parameter("association_image_width").value))
        height = int(str(self.get_parameter("association_image_height").value))
        return (width, height) if width > 0 and height > 0 else None

    def _lookup_base_to_camera(
        self, target_stamp_ns: int
    ) -> BaseToCameraTransform | None:
        stamp = Time(
            seconds=target_stamp_ns // 1_000_000_000,
            nanoseconds=target_stamp_ns % 1_000_000_000,
        )
        base_frame = str(self.get_parameter("base_frame").value)
        camera_frame = str(self.get_parameter("camera_frame").value)
        result = self._lookup_transform(camera_frame, base_frame, stamp)
        if result is None:
            return None
        rotation, translation = result
        return BaseToCameraTransform(rotation, translation)

    def _lookup_laser_to_base(self, scan: LaserScan) -> LaserToBaseTransform | None:
        stamp = Time.from_msg(scan.header.stamp)
        base_frame = str(self.get_parameter("base_frame").value)
        result = self._lookup_transform(base_frame, scan.header.frame_id, stamp)
        if result is None:
            return None
        rotation, translation = result
        return LaserToBaseTransform(rotation, translation)

    def _lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        stamp: Time,
    ) -> tuple[RotationMatrix, Point3D] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(nanoseconds=50_000_000),
            )
        except TransformException:
            return None
        return transform_values(transform)
