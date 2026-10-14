"""ROS safety boundary for the final forward command."""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped, Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from .clearance_logic import ClearancePolicy
from .safety_gate_logic import (
    GateConfig,
    ScanDebounceConfig,
    ScanDebounceState,
    TwistCommand,
    WatchdogState,
    debounce_scan_stale,
    gate_command,
    scan_to_base_points,
    watchdog_step,
)


class SafetyGateNode(Node):
    """Gate forward motion using a timestamped scan in base_link."""

    def __init__(self) -> None:
        super().__init__("safety_gate")
        self._declare_parameters()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._scan_points: tuple[tuple[float, float], ...] | None = None
        self._scan_stamp_ns = 0
        self._last_cmd_ns = 0
        self._last_command_linear_x = 0.0
        self._watchdog_state = WatchdogState()
        self._debounce_state = ScanDebounceState()
        self._publisher = self.create_publisher(
            Twist,
            str(self.get_parameter("output_topic").value),
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("input_topic").value),
            self._command_callback,
            10,
        )
        self.create_timer(0.05, self._watchdog)

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/cmd_vel_safe")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("robot_safety_clearance_m", 0.6)
        self.declare_parameter("clearance_origin_offset_m", 0.2)
        self.declare_parameter("slow_delta_m", 0.3)
        self.declare_parameter("corridor_half_width_m", 0.22)
        self.declare_parameter("max_scan_age_s", 0.25)
        self.declare_parameter("scan_recover_age_s", 0.20)
        self.declare_parameter("scan_recover_count", 3)
        self.declare_parameter("command_timeout_s", 0.25)
        self.declare_parameter("tf_timeout_s", 0.1)
        self.declare_parameter("slow_scale", 0.5)
        self.declare_parameter("allow_reverse", True)

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan_stamp_ns = self._stamp_ns(message)
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.get_parameter("base_frame").value),
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(
                    nanoseconds=int(self._float_parameter("tf_timeout_s") * 1_000_000_000),
                ),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as error:
            self._scan_points = None
            self.get_logger().warning(f"scan transform unavailable: {error}")
            return
        self._scan_points = self._transform_scan(message, transform)

    def _command_callback(self, message: Twist) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._last_cmd_ns = now_ns
        command = TwistCommand(float(message.linear.x), float(message.angular.z))
        self._last_command_linear_x = command.linear_x
        points = self._fresh_scan_points(now_ns)
        gated = gate_command(command, points, self._gate_config())
        output = Twist()
        output.linear.x = gated.linear_x
        output.angular.z = gated.angular_z
        self._publisher.publish(output)
        self._watchdog_state = WatchdogState(
            timeout_latched=False,
            scan_latched=gated.linear_x == 0.0 and command.linear_x > 0.0,
        )

    def _watchdog(self) -> None:
        if self._last_cmd_ns == 0:
            return
        now_ns = self.get_clock().now().nanoseconds
        command_timed_out = (
            now_ns - self._last_cmd_ns
        ) > self._float_parameter("command_timeout_s") * 1_000_000_000
        scan_stale = self._fresh_scan_points(now_ns) is None
        forward_scan_stale = scan_stale and self._last_command_linear_x > 0.0
        self._watchdog_state, publish_zero = watchdog_step(
            self._watchdog_state,
            command_timed_out,
            forward_scan_stale,
        )
        if publish_zero:
            self._publisher.publish(Twist())

    def _debounce_config(self) -> ScanDebounceConfig:
        try:
            recover_count = int(str(self.get_parameter("scan_recover_count").value))
        except (TypeError, ValueError, OverflowError):
            recover_count = 3
        try:
            max_age = float(str(self.get_parameter("max_scan_age_s").value))
        except (TypeError, ValueError, OverflowError):
            max_age = 0.0
        try:
            recover_age = float(str(self.get_parameter("scan_recover_age_s").value))
        except (TypeError, ValueError, OverflowError):
            recover_age = 0.0
        return ScanDebounceConfig(
            max_age_s=max_age,
            recover_age_s=recover_age,
            recover_count=recover_count,
        ).normalized()

    def _scan_age_s(self, now_ns: int) -> float | None:
        if self._scan_points is None or self._scan_stamp_ns <= 0:
            return None
        return (now_ns - self._scan_stamp_ns) / 1_000_000_000.0

    def _fresh_scan_points(self, now_ns: int) -> tuple[tuple[float, float], ...] | None:
        # Fail-closed hysteresis: stale latches immediately, recovery needs
        # consecutive distinct clearly-fresh scans; stale always maps to None
        # so the gate publishes exact zero forward velocity, never a reduced
        # one. The scan stamp keys the counter so the command callback plus
        # the watchdog tick cannot double-count one sample.
        age_s = self._scan_age_s(now_ns)
        stamp = self._scan_stamp_ns if self._scan_stamp_ns > 0 else None
        self._debounce_state, is_stale = debounce_scan_stale(
            self._debounce_state, age_s, self._debounce_config(), stamp
        )
        if is_stale:
            return None
        return self._scan_points

    def _gate_config(self) -> GateConfig:
        policy = ClearancePolicy(
            master_m=self._float_parameter("robot_safety_clearance_m"),
            origin_offset_m=self._float_parameter("clearance_origin_offset_m"),
            slow_delta_m=self._float_parameter("slow_delta_m"),
        )
        return GateConfig(
            stop_distance_m=policy.stop_m,
            slow_distance_m=policy.slow_m,
            corridor_half_width_m=self._float_parameter("corridor_half_width_m"),
            slow_scale=self._float_parameter("slow_scale"),
            allow_reverse=bool(self.get_parameter("allow_reverse").value),
        )

    def _transform_scan(
        self,
        message: LaserScan,
        transform: TransformStamped,
    ) -> tuple[tuple[float, float], ...] | None:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return scan_to_base_points(
            float(message.angle_min),
            float(message.angle_increment),
            float(message.range_min),
            float(message.range_max),
            tuple(float(value) for value in message.ranges),
            (float(translation.x), float(translation.y)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    @staticmethod
    def _stamp_ns(message: LaserScan) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def _float_parameter(self, name: str) -> float:
        return float(str(self.get_parameter(name).value))


def main(args: list[str] | None = None) -> None:
    """Run the final command safety gate."""
    import rclpy

    rclpy.init(args=args)
    node = SafetyGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
