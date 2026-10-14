"""ROS runtime for the hand-drawn red safe zone and guide-home flow.

Zone source: RViz Publish-Point clicks on /clicked_point (map frame),
latched after zone_click_points corners, or preloaded from the
zone_polygon_points parameter. The polygon is the SAFE area boundary.

Flow: ARMED -> (person outside for leaving_verify_s) -> say S1 ->
after ask_delay_s say S2 -> every repeat_s re-ask while the normal
follow keeps running -> text confirm -> say S3, take over Nav2
(guide_active=true) and drive to the fixed return point at
guide_max_vel_x -> on arrival say S4, spin-verify the person (max
search_max_spin_revolutions, existing spin behavior) -> person back
inside releases follow mode; outside re-arms the announce cycle.
"""

from __future__ import annotations

import math

from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from robot_autonomous_yolo_slam_nav2.msg import PersonTarget

from .follow_goal_helpers import estimate_person_speed, search_spin_command
from .nav2_person_follow_ros_helpers import (
    fresh_scan_points,
    robot_map_pose,
    stamp_ns,
    target_map_point,
    transform_scan,
)
from .zone_guide_logic import (
    SAY_ASK_CONFIRM,
    SAY_BACK_SAFE,
    SAY_GUIDE_CONFIRMED,
    SAY_VIOLATION,
    confirm_match,
    debounce_outside,
    latch_polygon,
    point_in_polygon,
    polygon_centroid,
)


def _normalize_say(text: str) -> str:
    """Normalize speech text for own-prompt echo comparison.

    Lowercase with collapsed whitespace so STT spacing noise does not
    defeat the filter. Diacritics are preserved.
    """
    return " ".join(text.strip().lower().split())


class ZoneGuardRuntime(Node):
    """Guard the red zone; guide home on confirmed violation."""

    def __init__(self) -> None:
        super().__init__("zone_guard")
        self._declare_parameters()
        self._clicks: list[tuple[float, float]] = []
        self._polygon: tuple[tuple[float, float], ...] | None = None
        self._polygon_yaw = 0.0
        preload: list[float] = []
        try:
            raw_value = self.get_parameter("zone_polygon_points").value or []
            preload = [float(value) for value in raw_value]
        except Exception:  # noqa: BLE001
            preload = []
        if len(preload) >= 6 and len(preload) % 2 == 0:
            pts = tuple(
                (float(preload[index]), float(preload[index + 1]))
                for index in range(0, len(preload), 2)
            )
            if len(pts) >= 3:
                self._polygon = pts
        self._phase = "IDLE" if self._polygon is None else "ARMED"
        self._phase_since_ns = 0
        self._outside_accum_s = 0.0
        self._last_repeat_ns = 0
        self._confirm_pending = False
        self._confirm_ns = 0
        self._last_say_ns = 0
        self._last_say_text = ""
        self._recent_say: list[tuple[str, int]] = []
        self._pending_say: str | None = None
        self._guide_active = False
        self._guide_pending = None
        self._guide_handle: ClientGoalHandle | None = None
        self._guide_done = False
        self._guide_succeeded = False
        self._guide_fails = 0
        self._guide_poly: tuple[tuple[float, float], ...] | None = None
        self._saved_max_vel_x: float | None = None
        self._last_seen_bearing_rad: float | None = None
        self._last_seen_range_m = 0.0
        self._last_map_sample: tuple[tuple[float, float], float] | None = None
        self._last_person_speed_m_s: float | None = None
        self._spin_accum_rad = 0.0
        self._last_spin_ns = 0
        self._verify_pending = None
        self._verify_handle: ClientGoalHandle | None = None
        self._verify_cancel_requested = False
        self._verify_cancel_pending = False
        self._guide_cancel_requested = False
        self._guide_cancel_pending = False
        self._guide_replan_queued = False
        self._target: PersonTarget | None = None
        self._scan_points: tuple[tuple[float, float], ...] | None = None
        self._scan_stamp_ns = 0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        say_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._say_pub = self.create_publisher(
            String, str(self.get_parameter("say_topic").value), say_qos)
        self._guide_pub = self.create_publisher(
            Bool, str(self.get_parameter("guide_active_topic").value), 10)
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("guide_status_topic").value), 10)
        self._zone_pub = self.create_publisher(
            Path, str(self.get_parameter("zone_path_topic").value), 10)
        self._clicks_pub = self.create_publisher(
            Marker, str(self.get_parameter("zone_path_topic").value) + "_clicks", 10)
        self._action_client = ActionClient(
            self, NavigateToPose, str(self.get_parameter("action_name").value))
        self.create_subscription(
            PersonTarget, str(self.get_parameter("target_topic").value),
            self._target_callback, 5)
        self.create_subscription(
            PointStamped, str(self.get_parameter("clicked_point_topic").value),
            self._clicked_callback, 10)
        self.create_subscription(
            String, str(self.get_parameter("confirm_topic").value),
            self._confirm_callback, 10)
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value),
            self._scan_callback, qos_profile_sensor_data)
        self.create_service(Trigger, "~/reset_zone", self._reset_zone)
        period = float(str(self.get_parameter("publish_period_s").value))
        self.create_timer(period, self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("target_topic", "/person_follow/target")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("confirm_topic", "/phone/speech_text")
        self.declare_parameter("say_topic", "/phone/speaker/say")
        self.declare_parameter("guide_active_topic", "/person_follow/guide_active")
        self.declare_parameter("guide_status_topic", "/person_follow/guide_status")
        self.declare_parameter("zone_path_topic", "/person_follow/red_zone")
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("controller_node", "/controller_server")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("zone_click_points", 4)
        self.declare_parameter("zone_polygon_points", [])
        self.declare_parameter("leaving_verify_s", 2.0)
        self.declare_parameter("ask_delay_s", 5.0)
        self.declare_parameter("repeat_s", 15.0)
        self.declare_parameter("max_target_age_s", 1.0)
        self.declare_parameter("max_scan_age_s", 0.25)
        self.declare_parameter("confirm_keyword", "xác nhận")
        self.declare_parameter("confirm_valid_s", 30.0)
        self.declare_parameter("confirm_ignore_after_say_s", 3.0)
        self.declare_parameter("confirm_ignore_per_char_s", 0.12)
        self.declare_parameter("confirm_echo_window_s", 20.0)
        self.declare_parameter("guide_return_x", 0.0)
        self.declare_parameter("guide_return_y", 0.0)
        self.declare_parameter("guide_return_yaw", 0.0)
        self.declare_parameter("guide_return_override", False)
        self.declare_parameter("guide_max_vel_x", 0.25)
        self.declare_parameter("search_spin_default_rate_rad_s", 0.7)
        self.declare_parameter("search_spin_min_rate_rad_s", 0.51)
        self.declare_parameter("search_spin_max_rate_rad_s", 1.4)
        self.declare_parameter("search_spin_deadband_rad", 0.1)
        self.declare_parameter("search_max_spin_revolutions", 2.0)
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("tf_timeout_s", 0.1)

    def _target_callback(self, message: PersonTarget) -> None:
        self._target = message

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan_stamp_ns = stamp_ns(message)
        self._scan_points = transform_scan(
            self._tf_buffer, message, str(self.get_parameter("base_frame").value))

    def _clicked_callback(self, message: PointStamped) -> None:
        if self._polygon is not None:
            return
        point = self._to_map(message)
        if point is None:
            self._publish_status("draw", "draw_click_tf_unavailable")
            return
        self._clicks.append(point)
        self._publish_clicks()
        latched = latch_polygon(
            tuple(self._clicks), int(str(self.get_parameter("zone_click_points").value)))
        if latched is not None:
            self._polygon = latched
            self._polygon_yaw = 0.0
            self._phase = "ARMED"
            self._publish_status("armed", "zone_latched")
        else:
            self._publish_status(
                "draw",
                f"draw_clicks_{len(self._clicks)}_of_"
                f"{int(str(self.get_parameter('zone_click_points').value))}")

    def _confirm_callback(self, message: String) -> None:
        # The phone mic hears our own speaker, and our own questions contain
        # the keyword, so two layers guard the confirmation:
        # 1. a speech-duration-scaled ignore window measured from publish
        #    (TTS speech + STT latency outlast the fixed base window), and
        # 2. an echo filter dropping texts that contain a recently spoken
        #    prompt verbatim (late full-sentence echoes).
        now_ns = self.get_clock().now().nanoseconds
        base_s = float(str(self.get_parameter("confirm_ignore_after_say_s").value))
        per_char_s = float(str(self.get_parameter("confirm_ignore_per_char_s").value))
        window_s = base_s + len(self._last_say_text) * per_char_s
        if (now_ns - self._last_say_ns) / 1_000_000_000.0 < window_s:
            return
        text = str(message.data)
        if self._is_own_echo(text, now_ns):
            return
        if confirm_match(
            text,
            str(self.get_parameter("confirm_keyword").value),
        ):
            self._confirm_pending = True
            self._confirm_ns = now_ns

    def _is_own_echo(self, text: str, now_ns: int) -> bool:
        """True when the text contains a recently spoken prompt verbatim."""
        window_s = float(str(self.get_parameter("confirm_echo_window_s").value))
        incoming = _normalize_say(text)
        if not incoming:
            return False
        for prompt, said_ns in self._recent_say:
            if (now_ns - said_ns) / 1_000_000_000.0 > window_s:
                continue
            prompt_norm = _normalize_say(prompt)
            if prompt_norm and prompt_norm in incoming:
                return True
        return False

    def _reset_zone(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._clicks = []
        self._polygon = None
        self._phase = "IDLE"
        self._outside_accum_s = 0.0
        self._spin_accum_rad = 0.0
        self._confirm_pending = False
        self._confirm_ns = 0
        # Reset clears UI/zone state immediately but retains action ownership:
        # the old guide/verify handoff stays cancelled-and-owned while IDLE
        # and blocks every new motion goal until its terminal result lands,
        # so reset-then-immediate-guide can never double-command the server.
        self._cancel_guide_handoff()
        self._guide_replan_queued = False
        self._cancel_verify_goal()
        self._guide_active = False
        self._guide_done = False
        self._guide_succeeded = False
        self._guide_fails = 0
        self._guide_poly = None
        self._restore_speed()
        self._publish_clicks()
        resp.success = True
        resp.message = "red zone cleared; click 4 corners in RViz to redraw"
        return resp

    def _cancel_guide_handoff(self) -> None:
        """Request guide cancellation without dropping owned futures.

        A pending send stays owned until the server accepts or rejects it;
        the response cancels on acceptance. An active goal gets one cancel
        request and stays owned until its terminal result.
        """
        if self._guide_pending is not None:
            self._guide_cancel_requested = True
        if self._guide_handle is not None and not self._guide_cancel_pending:
            try:
                self._guide_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
            self._guide_cancel_pending = True

    def _guide_handoff_busy(self) -> bool:
        """True while an old guide goal still owns the handoff."""
        return (
            self._guide_pending is not None
            or self._guide_handle is not None
            or self._guide_cancel_requested
            or self._guide_cancel_pending
        )

    def _to_map(self, message: PointStamped) -> tuple[float, float] | None:
        map_frame = str(self.get_parameter("map_frame").value)
        if message.header.frame_id == map_frame:
            return (float(message.point.x), float(message.point.y))
        try:
            out = self._tf_buffer.transform(
                message, map_frame,
                timeout=self._duration(str(self.get_parameter("tf_timeout_s").value)),
            )
        except Exception:  # noqa: BLE001
            return None
        return (float(out.point.x), float(out.point.y))

    def _duration(self, seconds_s: str):  # type: ignore[no-untyped-def]
        from rclpy.duration import Duration
        return Duration(nanoseconds=int(float(seconds_s) * 1_000_000_000))

    def _say(self, text: str) -> None:
        self._pending_say = text

    def _person_state(self, now_ns: int) -> tuple[bool, tuple[float, float] | None]:
        """Visible flag plus map-frame position of the confirmed target."""
        target = self._target
        if target is None or not target.valid or not target.identity_confirmed:
            return False, None
        if str(target.state) != "tracking":
            return False, None
        stamp = target.header.stamp.sec * 1_000_000_000 + target.header.stamp.nanosec
        age_s = (now_ns - stamp) / 1_000_000_000.0
        if age_s < 0.0 or age_s > self._float("max_target_age_s"):
            return False, None
        self._last_seen_bearing_rad = float(target.bearing_rad)
        self._last_seen_range_m = float(target.range_m)
        point = target_map_point(
            self._tf_buffer, target,
            str(self.get_parameter("base_frame").value),
            str(self.get_parameter("map_frame").value))
        if point is not None:
            stamp_s = float(target.header.stamp.sec) + float(target.header.stamp.nanosec) / 1_000_000_000.0
            sample = ((point[0], point[1]), stamp_s)
            self._last_person_speed_m_s = estimate_person_speed(
                self._last_map_sample, sample) or self._last_person_speed_m_s
            self._last_map_sample = sample
        return True, (point[0], point[1]) if point is not None else None

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._guide_pub.publish(Bool(data=self._guide_active))
        self._publish_zone()
        self._publish_clicks()
        if self._pending_say is not None and self._say_pub.get_subscription_count() > 0:
            message = String()
            message.data = self._pending_say
            self._say_pub.publish(message)
            said_ns = self.get_clock().now().nanoseconds
            self._last_say_ns = said_ns
            self._last_say_text = message.data
            self._recent_say.append((_normalize_say(message.data), said_ns))
            del self._recent_say[:-4]
            self._pending_say = None
        if self._polygon is None:
            if self._phase != "IDLE":
                self._phase = "IDLE"
            self._publish_status("idle", "idle_no_zone")
            return
        visible, position = self._person_state(now_ns)
        outside = visible and position is not None and not point_in_polygon(
            position[0], position[1], self._polygon)
        if self._phase == "IDLE":
            self._phase = "ARMED"
        if self._phase == "ARMED":
            self._outside_accum_s, confirmed = debounce_outside(
                outside, self._outside_accum_s,
                float(str(self.get_parameter("publish_period_s").value)),
                float(str(self.get_parameter("leaving_verify_s").value)))
            if confirmed:
                self._enter("VIOLATION", now_ns)
                self._say(SAY_VIOLATION)
            else:
                self._publish_status("armed", "armed_inside_zone")
            return
        if self._phase == "VIOLATION":
            if visible and position is not None and point_in_polygon(
                    position[0], position[1], self._polygon):
                self._outside_accum_s = 0.0
                self._enter("ARMED", now_ns)
                return
            if self._guide_handoff_busy():
                self._publish_status("violation", "violation_handoff_busy")
                return
            if self._consume_confirm():
                self._start_guide(now_ns)
                return
            if (now_ns - self._phase_since_ns) / 1_000_000_000.0 >= float(
                    str(self.get_parameter("ask_delay_s").value)):
                self._enter("WAIT_CONFIRM", now_ns)
                self._last_repeat_ns = now_ns
                self._say(SAY_ASK_CONFIRM)
            else:
                self._publish_status("violation", "violation_announced")
            return
        if self._phase == "WAIT_CONFIRM":
            if visible and position is not None and point_in_polygon(
                    position[0], position[1], self._polygon):
                self._outside_accum_s = 0.0
                self._enter("ARMED", now_ns)
                return
            if self._guide_handoff_busy():
                self._publish_status("wait_confirm", "wait_confirm_handoff_busy")
                return
            if self._consume_confirm():
                self._start_guide(now_ns)
                return
            if (now_ns - self._last_repeat_ns) / 1_000_000_000.0 >= float(
                    str(self.get_parameter("repeat_s").value)):
                self._last_repeat_ns = now_ns
                self._say(SAY_ASK_CONFIRM)
            self._publish_status("wait_confirm", "wait_confirm_repeating")
            return
        if self._phase == "GUIDE_NAV":
            if self._guide_done:
                self._restore_speed()
                self._say(SAY_BACK_SAFE)
                self._enter("ARRIVED_VERIFY", now_ns)
                self._spin_accum_rad = 0.0
                self._last_spin_ns = now_ns
                self._verify_pending = None
                self._verify_handle = None
                self._verify_cancel_requested = False
                self._verify_cancel_pending = False
                return
            if self._guide_fails >= 3:
                self._publish_status("guide_nav", "guide_goal_unreachable_hold")
                return
            if self._polygon is not None and self._polygon != self._guide_poly:
                self._guide_poly = self._polygon
                self._guide_fails = 0
                if self._guide_pending is not None:
                    self._guide_cancel_requested = True
                if self._guide_handle is not None and not self._guide_cancel_pending:
                    try:
                        self._guide_handle.cancel_goal_async()
                    except Exception:  # noqa: BLE001
                        pass
                    self._guide_cancel_pending = True
                self._guide_replan_queued = True
                self._flush_guide_replan()
                self._publish_status("guide_nav", "guide_replanned_zone_moved")
                return
            self._publish_status("guide_nav", "guide_driving_home")
            return
        if self._phase == "ARRIVED_VERIFY":
            self._verify_spin(now_ns, visible, position)
            return
        if self._phase == "VERIFY_HOLD":
            if visible and position is not None:
                if point_in_polygon(position[0], position[1], self._polygon):
                    self._release_guide()
                    self._outside_accum_s = 0.0
                    self._enter("ARMED", now_ns)
                else:
                    self._release_guide()
                    self._outside_accum_s = 0.0
                    self._enter("VIOLATION", now_ns)
                    self._say(SAY_VIOLATION)
            else:
                self._publish_status("verify_hold", "verify_hold_wait_person")
            return

    def _enter(self, phase: str, now_ns: int) -> None:
        self._phase = phase
        self._phase_since_ns = now_ns

    def _consume_confirm(self) -> bool:
        if self._confirm_pending:
            self._confirm_pending = False
            valid_s = float(str(self.get_parameter("confirm_valid_s").value))
            age_s = (self.get_clock().now().nanoseconds - self._confirm_ns) / 1_000_000_000.0
            if 0.0 <= age_s <= valid_s:
                return True
        return False

    def _start_guide(self, now_ns: int) -> None:
        if self._guide_handoff_busy():
            self._publish_status("guide_nav", "guide_start_blocked_handoff")
            return
        self._say(SAY_GUIDE_CONFIRMED)
        self._guide_active = True
        self._guide_done = False
        self._guide_succeeded = False
        self._guide_fails = 0
        self._guide_poly = self._polygon
        self._guide_cancel_requested = False
        self._guide_cancel_pending = False
        self._guide_replan_queued = False
        self._limit_speed()
        self._send_guide_goal()
        self._enter("GUIDE_NAV", now_ns)
        self._publish_status("guide_nav", "guide_confirmed_start")

    def _guide_goal_point(self) -> tuple[float, float, float] | None:
        """Return point: yaml override, else the drawn polygon centroid.

        Falls back to the first corner when the centroid lies outside
        (concave hand-drawn shapes). None without a polygon.
        """
        if bool(self.get_parameter("guide_return_override").value):
            return (
                float(str(self.get_parameter("guide_return_x").value)),
                float(str(self.get_parameter("guide_return_y").value)),
                float(str(self.get_parameter("guide_return_yaw").value)),
            )
        polygon = self._polygon
        if polygon is None:
            return None
        center = polygon_centroid(polygon)
        if center is not None and point_in_polygon(center[0], center[1], polygon):
            return (center[0], center[1], self._polygon_yaw)
        corner = polygon[0]
        return (corner[0], corner[1], self._polygon_yaw)

    def _release_guide(self) -> None:
        self._guide_active = False
        self._restore_speed()
        self._publish_status("armed", "guide_done_person_safe")

    def _send_guide_goal(self) -> None:
        from nav2_msgs.action import NavigateToPose
        if self._guide_fails >= 3:
            return
        if not self._guide_active:
            return
        if (
            self._guide_pending is not None
            or self._guide_handle is not None
            or self._guide_cancel_requested
            or self._guide_cancel_pending
        ):
            return
        point = self._guide_goal_point()
        if point is None:
            return
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter("map_frame").value)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.orientation.z = math.sin(point[2] / 2.0)
        pose.pose.orientation.w = math.cos(point[2] / 2.0)
        goal.pose = pose
        if not self._action_client.server_is_ready():
            return
        pending = self._action_client.send_goal_async(goal)
        self._guide_pending = pending
        pending.add_done_callback(self._guide_response_callback)

    def _flush_guide_replan(self) -> None:
        """Send the queued replan once the old goal reached terminal state."""
        if not self._guide_replan_queued:
            return
        if self._guide_handoff_busy():
            return
        self._guide_replan_queued = False
        self._send_guide_goal()

    def _guide_response_callback(self, future) -> None:  # type: ignore[no-untyped-def]
        """Own the pending guide send until the server accepts or rejects it.

        An acceptance installs the handle and — when a cancel was requested
        meanwhile (replan, reset, new-zone flow) — cancels it at once and
        keeps awaiting the terminal result. The queued replan, if any, is
        only flushed after that terminal result, never before.
        """
        self._guide_pending = None
        try:
            error = future.exception()
        except Exception:  # noqa: BLE001
            error = True
        if error is not None:
            self._guide_fails += 1
            self.get_logger().warning("guide goal request failed; will retry")
            if self._guide_replan_queued:
                self._flush_guide_replan()
            else:
                self._send_guide_goal()
            return
        handle = future.result()
        if handle is None or not handle.accepted:
            self._guide_fails += 1
            self.get_logger().warning("guide goal rejected; will retry")
            if self._guide_replan_queued:
                self._flush_guide_replan()
            else:
                self._send_guide_goal()
            return
        self._guide_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done, owned=handle: self._guide_result_callback(done, owned))
        if self._guide_cancel_requested:
            self._guide_cancel_requested = False
            try:
                handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
            self._guide_cancel_pending = True

    def _guide_result_callback(self, future, handle=None) -> None:  # type: ignore[no-untyped-def]
        from action_msgs.msg import GoalStatus
        if handle is not None and self._guide_handle is not handle:
            self.get_logger().error("Stale guide result ignored; newer goal owns the handoff")
            return
        self._guide_handle = None
        self._guide_cancel_requested = False
        self._guide_cancel_pending = False
        try:
            result = future.result()
        except Exception:  # noqa: BLE001
            result = None
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self._guide_done = True
            self._guide_succeeded = True
            self._guide_replan_queued = False
        else:
            self._guide_fails += 1
            self.get_logger().warning("guide goal ended unsucceeded; retrying")
            if self._guide_replan_queued:
                self._flush_guide_replan()
            else:
                self._send_guide_goal()

    def _set_controller_param(self, name: str, value: float) -> None:
        """Best-effort set a double parameter on the controller server."""
        try:
            from rcl_interfaces.msg import Parameter as ParameterMsg
            from rcl_interfaces.msg import ParameterType, ParameterValue
            from rcl_interfaces.srv import SetParameters
            client = self.create_client(
                SetParameters,
                f"{self.get_parameter('controller_node').value}/set_parameters",
            )
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning(
                    "controller set_parameters unavailable; speed unchanged")
                return
            request = SetParameters.Request()
            request.parameters = [ParameterMsg(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=value))]
            client.call_async(request)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"speed change unavailable: {error}")

    def _limit_speed(self) -> None:
        try:
            from rcl_interfaces.srv import GetParameters
            client = self.create_client(
                GetParameters,
                f"{self.get_parameter('controller_node').value}/get_parameters",
            )
            if client.wait_for_service(timeout_sec=1.0):
                request = GetParameters.Request()
                request.names = ["max_vel_x"]

                def _got(future) -> None:  # type: ignore[no-untyped-def]
                    try:
                        values = future.result().values
                    except Exception:  # noqa: BLE001
                        return
                    if values:
                        value = values[0].double_value
                        if value > 0.0:
                            self._saved_max_vel_x = float(value)

                client.call_async(request).add_done_callback(_got)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f"speed save unavailable: {error}")
        self._set_controller_param(
            "max_vel_x", float(str(self.get_parameter("guide_max_vel_x").value)))

    def _restore_speed(self) -> None:
        if self._saved_max_vel_x is None:
            return
        saved, self._saved_max_vel_x = self._saved_max_vel_x, None
        self._set_controller_param("max_vel_x", saved)

    def _verify_spin(
        self,
        now_ns: int,
        visible: bool,
        position: tuple[float, float] | None,
    ) -> None:
        if visible and position is not None and self._polygon is not None:
            if point_in_polygon(position[0], position[1], self._polygon):
                self._cancel_verify_goal()
                self._release_guide()
                self._outside_accum_s = 0.0
                self._enter("ARMED", now_ns)
                return
            self._cancel_verify_goal()
            self._release_guide()
            self._outside_accum_s = 0.0
            self._enter("VIOLATION", now_ns)
            self._say(SAY_VIOLATION)
            return
        scan = fresh_scan_points(
            self._scan_points, self._scan_stamp_ns, now_ns,
            float(str(self.get_parameter("max_scan_age_s").value)))
        if scan is None:
            self._cancel_verify_goal()
            self._publish_status("verify", "verify_scan_stale")
            return
        yaw_rate = search_spin_command(
            self._last_seen_bearing_rad,
            self._last_person_speed_m_s,
            self._last_seen_range_m,
            float(str(self.get_parameter("search_spin_default_rate_rad_s").value)),
            float(str(self.get_parameter("search_spin_min_rate_rad_s").value)),
            float(str(self.get_parameter("search_spin_max_rate_rad_s").value)),
            float(str(self.get_parameter("search_spin_deadband_rad").value)),
        )
        if yaw_rate is None:
            self._cancel_verify_goal()
            self._publish_status("verify", "verify_no_bearing")
            return
        max_rev = float(str(self.get_parameter("search_max_spin_revolutions").value))
        if max_rev > 0.0 and self._spin_accum_rad >= max_rev * 2.0 * math.pi:
            self._cancel_verify_goal()
            self._enter("VERIFY_HOLD", now_ns)
            self._publish_status("verify_hold", "verify_spin_done")
            return
        if yaw_rate == 0.0:
            self._cancel_verify_goal()
            self._publish_status("verify", "verify_spinning")
            return
        if self._verify_pending is not None or self._verify_handle is not None:
            self._publish_status("verify", "verify_spinning")
            return
        if self._verify_cancel_requested or self._verify_cancel_pending:
            self._publish_status("verify", "verify_cancelling")
            return
        pose = robot_map_pose(
            self._tf_buffer,
            str(self.get_parameter("map_frame").value),
            str(self.get_parameter("base_frame").value),
        )
        if pose is None:
            self._publish_status("verify", "verify_tf_unavailable")
            return
        if not self._action_client.server_is_ready():
            self._publish_status("verify", "verify_nav_unavailable")
            return
        direction = 1.0 if yaw_rate > 0.0 else -1.0
        cap_rad = max_rev * 2.0 * math.pi if max_rev > 0.0 else math.pi / 2.0
        remaining = cap_rad - self._spin_accum_rad if max_rev > 0.0 else math.pi / 2.0
        step = direction * min(math.pi / 2.0, max(0.0, remaining))
        if step == 0.0:
            self._cancel_verify_goal()
            self._enter("VERIFY_HOLD", now_ns)
            self._publish_status("verify_hold", "verify_spin_done")
            return
        self._send_verify_yaw_goal(pose, step)
        self._publish_status("verify", "verify_spinning")

    def _send_verify_yaw_goal(
        self, robot_pose: tuple[float, float, float], step_rad: float) -> None:
        """Send one bounded in-place NavigateToPose yaw goal for verification."""
        from nav2_msgs.action import NavigateToPose
        target_yaw = robot_pose[2] + step_rad
        goal = NavigateToPose.Goal()
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = str(self.get_parameter("map_frame").value)
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = robot_pose[0]
        pose_msg.pose.position.y = robot_pose[1]
        pose_msg.pose.orientation.z = math.sin(target_yaw / 2.0)
        pose_msg.pose.orientation.w = math.cos(target_yaw / 2.0)
        goal.pose = pose_msg
        if self._verify_handoff_busy():
            return
        pending = self._action_client.send_goal_async(goal)
        self._verify_pending = pending
        pending.add_done_callback(self._verify_response_callback)
        self._spin_accum_rad += abs(step_rad)

    def _verify_handoff_busy(self) -> bool:
        """True while an old verify goal still owns the handoff."""
        return (
            self._verify_pending is not None
            or self._verify_handle is not None
            or self._verify_cancel_requested
            or self._verify_cancel_pending
        )

    def _verify_response_callback(self, future) -> None:  # type: ignore[no-untyped-def]
        """Own the pending verify send until the server accepts or rejects it."""
        self._verify_pending = None
        try:
            error = future.exception()
        except Exception:  # noqa: BLE001
            error = True
        if error is not None:
            self._verify_cancel_requested = False
            self._verify_handle = None
            return
        try:
            handle = future.result()
        except Exception:  # noqa: BLE001
            self._verify_cancel_requested = False
            self._verify_handle = None
            return
        if handle is None or not handle.accepted:
            self._verify_cancel_requested = False
            self._verify_handle = None
            return
        self._verify_handle = handle
        try:
            result_future = handle.get_result_async()
            result_future.add_done_callback(
                lambda done, owned=handle: self._verify_result_callback(done, owned))
        except Exception:  # noqa: BLE001
            self._verify_handle = None
            self._verify_cancel_requested = False
            return
        if self._verify_cancel_requested:
            self._verify_cancel_requested = False
            try:
                handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
            self._verify_cancel_pending = True

    def _verify_result_callback(self, future, handle=None) -> None:  # type: ignore[no-untyped-def]
        if handle is not None and self._verify_handle is not handle:
            self.get_logger().error("Stale verify result ignored; newer goal owns the handoff")
            return
        self._verify_handle = None
        self._verify_cancel_requested = False
        self._verify_cancel_pending = False
        try:
            future.result()
        except Exception:  # noqa: BLE001
            return

    def _cancel_verify_goal(self) -> None:
        if self._verify_pending is not None:
            self._verify_cancel_requested = True
        if self._verify_handle is not None and not self._verify_cancel_pending:
            try:
                self._verify_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
            self._verify_cancel_pending = True

    def _publish_clicks(self) -> None:
        """Show each clicked corner as a red sphere while drawing."""
        from geometry_msgs.msg import Point as PointMsg

        marker = Marker()
        marker.header.frame_id = str(self.get_parameter("map_frame").value)
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "red_zone_clicks"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        for x, y in self._clicks:
            marker.points.append(PointMsg(x=x, y=y, z=0.05))
        self._clicks_pub.publish(marker)

    def _publish_zone(self) -> None:
        if self._polygon is None:
            return
        path = Path()
        path.header.frame_id = str(self.get_parameter("map_frame").value)
        path.header.stamp = self.get_clock().now().to_msg()
        corners = list(self._polygon) + [self._polygon[0]]
        for x, y in corners:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._zone_pub.publish(path)

    def _publish_status(self, phase: str, reason: str) -> None:
        message = String()
        message.data = f"phase={phase} reason={reason}"
        self._status_pub.publish(message)

    def _float(self, name: str) -> float:
        return float(str(self.get_parameter(name).value))
