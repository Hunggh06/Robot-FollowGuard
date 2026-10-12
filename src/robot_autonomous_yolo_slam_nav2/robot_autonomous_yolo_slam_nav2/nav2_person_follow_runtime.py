"""ROS runtime boundary for person follow and lost-person recovery.

Visible: follow the confirmed target through Nav2 NavigateToPose.
Lost: keep riding the current Nav2 goal (its trajectory may still lead
to the person); search in place toward the last seen bearing only when
no Nav2 goal is active, through the Nav2 /spin action client, at most
search_max_spin_revolutions, then stop. Recovery rate tracks the
person's last measured speed. The safety gate still guards output.
"""

from __future__ import annotations

import math

from nav2_msgs.action import NavigateToPose, Spin

from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from robot_autonomous_yolo_slam_nav2.msg import PersonObservation, PersonTarget

from .follow_goal_helpers import (
    CancelGrace,
    FollowConfig,
    ObstaclePoint,
    TargetGeometry,
    decide_follow_goal,
    estimate_person_speed,
    grace_advance,
    inspection_pause_duration_s,
    search_spin_command,
    update_close_latch,
)
from .nav2_person_follow_actions import Nav2PersonFollowActions
from .nav2_person_follow_spin_actions import Nav2PersonFollowSpinActions
from .goal_update_logic import GoalMode, GoalRequest, GoalSnapshot
from .nav2_person_follow_ros_helpers import (
    fresh_scan_points,
    parse_obstacle_cloud,
    stamp_ns,
    target_map_point,
    transform_scan,
)


class Nav2PersonFollowRuntime(Nav2PersonFollowActions, Nav2PersonFollowSpinActions, Node):
    """Follow confirmed targets with Nav2; spin toward them when lost."""

    def __init__(self) -> None:
        super().__init__("nav2_person_follow")
        self._declare_parameters()
        self._target: PersonTarget | None = None
        self._obstacles: PointCloud2 | None = None
        self._active_goal: ClientGoalHandle | None = None
        self._active_goal_token: str | None = None
        self._pending_goal: Future | None = None
        self._queued_goal: GoalRequest | None = None
        self._last_goal_base_link: tuple[float, float] | None = None
        self._last_goal_map: GoalSnapshot | None = None
        self._last_goal_mode: GoalMode | None = None
        self._cancel_requested = False
        self._cancel_sent = False
        self._cancel_sent_at_ns = 0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._scan_points: tuple[tuple[float, float], ...] | None = None
        self._scan_stamp_ns = 0
        self._confirmed_track_id: int | None = None
        self._close_latched = False
        self._cancel_grace = CancelGrace()
        self._last_seen_bearing_rad: float | None = None
        self._last_seen_range_m = 0.0
        self._last_map_sample: tuple[tuple[float, float], float] | None = None
        self._last_person_speed_m_s: float | None = None
        self._searching = False
        self._lost_since_ns = 0
        self._spin_accum_rad = 0.0
        self._last_spin_ns = 0
        self._spin_capped = False
        self._pending_spin: Future | None = None
        self._active_spin: ClientGoalHandle | None = None
        self._pending_spin_yaw = 0.0
        self._guide_active = False
        self._guide_yielded = False
        self._person_obs: dict[int, int] = {}
        self._inspection_until_ns = 0
        self._inspection_ids: set[int] = set()
        self._inspection_done_ids: set[int] = set()
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("recovery_status_topic").value),
            10,
        )
        action_name = str(self.get_parameter("action_name").value)
        self._action_client = ActionClient(self, NavigateToPose, action_name)
        spin_action_name = str(self.get_parameter("spin_action_name").value)
        self._spin_action_client: ActionClient = ActionClient(self, Spin, spin_action_name)
        self.create_subscription(
            PersonTarget,
            str(self.get_parameter("target_topic").value),
            self._target_callback,
            5,
        )
        self.create_subscription(
            PersonObservation,
            str(self.get_parameter("observations_topic").value),
            self._person_observations_callback,
            5,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("obstacles_topic").value),
            self._obstacles_callback,
            5,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("guide_active_topic").value),
            self._guide_callback,
            10,
        )
        period = float(str(self.get_parameter("publish_period_s").value))
        self.create_timer(period, self._publish_current)

    def _declare_parameters(self) -> None:
        self.declare_parameter("target_topic", "/person_follow/target")
        self.declare_parameter("observations_topic", "/yolo_target_node/observations")
        self.declare_parameter("obstacles_topic", "/person_follow/obstacles")
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("spin_action_name", "/spin")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("follow_distance_m", 1.0)
        self.declare_parameter("follow_resume_distance_m", 1.2)
        self.declare_parameter("cancel_grace_cycles", 5)
        self.declare_parameter("max_target_age_s", 1.0)
        self.declare_parameter("corridor_half_width_m", 0.22)
        self.declare_parameter("goal_update_distance_m", 0.1)
        self.declare_parameter("goal_update_yaw_rad", 0.1)
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("tf_timeout_s", 0.1)
        self.declare_parameter("max_obstacle_points", 10000)
        self.declare_parameter("cancel_timeout_s", 0.5)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("max_scan_age_s", 0.25)
        self.declare_parameter("robot_safety_clearance_m", 0.6)
        self.declare_parameter("clearance_origin_offset_m", 0.2)
        self.declare_parameter("search_spin_default_rate_rad_s", 0.7)
        self.declare_parameter("search_spin_min_rate_rad_s", 0.51)
        self.declare_parameter("search_spin_max_rate_rad_s", 1.4)
        self.declare_parameter("search_spin_deadband_rad", 0.1)
        self.declare_parameter("search_max_spin_revolutions", 2.0)
        self.declare_parameter("search_timeout_s", 0.0)
        self.declare_parameter("search_drive_duration_s", 2.0)
        self.declare_parameter("search_drive_speed_m_s", 0.15)
        self.declare_parameter("search_inspection_per_person_s", 2.0)
        self.declare_parameter("search_inspection_max_s", 10.0)
        self.declare_parameter("search_inspection_max_obs_age_s", 0.5)
        self.declare_parameter("guide_active_topic", "/person_follow/guide_active")
        self.declare_parameter("recovery_status_topic", "/person_follow/recovery_status")

    def _guide_callback(self, message: Bool) -> None:
        self._guide_active = bool(message.data)

    def _target_callback(self, message: PersonTarget) -> None:
        self._target = message

    def _person_observations_callback(self, message: PersonObservation) -> None:
        """Track fresh per-person presence for the search inspection pause.

        Track id 0 is the YOLO empty message (no person); only real track
        ids count as presence. Stamps key freshness, never motion.
        """
        track_id = int(message.track_id)
        if track_id == 0:
            return
        self._person_obs[track_id] = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _fresh_track_ids(self, now_ns: int) -> set[int]:
        max_age_s = self._float_parameter("search_inspection_max_obs_age_s")
        fresh: set[int] = set()
        for track_id, stamp_ns in self._person_obs.items():
            age_s = (now_ns - stamp_ns) / 1_000_000_000.0
            if 0.0 <= age_s <= max_age_s:
                fresh.add(track_id)
        return fresh

    def _obstacles_callback(self, message: PointCloud2) -> None:
        self._obstacles = message

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan_stamp_ns = stamp_ns(message)
        self._scan_points = transform_scan(
            self._tf_buffer, message, str(self.get_parameter("base_frame").value)
        )

    def _publish_current(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        target = self._target
        if self._guide_active:
            if not self._guide_yielded:
                self._guide_yielded = True
                self._request_cancel("guide_override")
            self._cancel_spin()
            self._publish_status("guide_override", "guide_active_yield_nav2")
            return
        self._guide_yielded = False
        target_geometry = None if target is None else self._target_geometry(target, now_ns)
        target_visible = self._is_confirmed_target(target, target_geometry)
        if target_visible and target is not None and target.identity_confirmed:
            if self._confirmed_track_id != int(target.track_id):
                self._confirmed_track_id = int(target.track_id)
        if (
            target is not None
            and target_geometry is not None
            and target_geometry.valid
            and target_geometry.state == "tracking"
            and 0.0 <= target_geometry.age_s <= self._float_parameter("max_target_age_s")
        ):
            self._last_seen_bearing_rad = target_geometry.bearing_rad
            self._last_seen_range_m = target_geometry.range_m
            target_map = target_map_point(
                self._tf_buffer, target,
                str(self.get_parameter("base_frame").value),
                str(self.get_parameter("map_frame").value),
            )
            if target_map is not None:
                stamp_s = (
                    float(target.header.stamp.sec)
                    + float(target.header.stamp.nanosec) / 1_000_000_000.0
                )
                sample = ((target_map[0], target_map[1]), stamp_s)
                self._last_person_speed_m_s = estimate_person_speed(
                    self._last_map_sample, sample
                ) or self._last_person_speed_m_s
                self._last_map_sample = sample
        scan_points = fresh_scan_points(
            self._scan_points, self._scan_stamp_ns, now_ns, self._float_parameter("max_scan_age_s")
        )
        if not target_visible:
            visual_contact = (
                target is not None
                and target.identity_confirmed
                and target.state == "tracking"
                and target_geometry is not None
                and 0.0 <= target_geometry.age_s <= self._float_parameter("max_target_age_s")
            )
            self._spin_search(
                now_ns, scan_points is not None,
                allow_drive=visual_contact,
            )
            return
        if self._searching:
            self._searching = False
            self._spin_accum_rad = 0.0
            self._last_spin_ns = 0
            self._spin_capped = False
            self._inspection_until_ns = 0
            self._inspection_ids = set()
            self._inspection_done_ids = set()
            self._cancel_spin()
        follow_config = self._follow_config()
        fresh_range: float | None = None
        if (
            target is not None
            and target_geometry is not None
            and target_geometry.valid
            and 0.0 <= target_geometry.age_s <= follow_config.max_target_age_s
        ):
            fresh_range = target_geometry.range_m
        self._close_latched, close_action = update_close_latch(
            self._close_latched, fresh_range, follow_config)
        if close_action == "cancel_now":
            self._cancel_grace = CancelGrace()
            self._request_cancel("target_inside_follow_distance")
            return
        if close_action == "hold":
            return
        if target is None or target_geometry is None:
            self._request_cancel_graced("target_missing")
            return
        obstacles = self._obstacle_points(now_ns)
        if obstacles is None:
            self._request_cancel_graced("obstacles_invalid")
            return
        self._cancel_grace = CancelGrace()
        decision = decide_follow_goal(
            target_geometry,
            obstacles,
            self._follow_config(),
            self._last_goal_base_link,
        )
        self._apply_decision(decision, target.header.stamp)
        self._publish_status("following", "following_confirmed_target")

    def _inspect_people(self, now_ns: int) -> bool:
        """Hold the spin while fresh people may still confirm the tag.

        Anchors one bounded window per fresh track set (never extended),
        cancels only the in-flight Nav2 spin, and sends nothing while the
        window holds. Stale or zero people ends the pause and resumes the
        spin; a tag lock exits through the following path instead. Each
        completed track id is remembered so the same track never retriggers,
        while a new fresh track id may anchor its own bounded window.
        """
        fresh_ids = self._fresh_track_ids(now_ns)
        if self._inspection_until_ns > now_ns:
            if fresh_ids:
                self._inspection_ids |= fresh_ids
                self._cancel_spin()
                self._last_spin_ns = now_ns
                self._publish_status("searching", "search_inspecting")
                return True
            self._inspection_done_ids |= self._inspection_ids
            self._inspection_ids = set()
            self._inspection_until_ns = 0
            return False
        if self._inspection_until_ns != 0:
            self._inspection_done_ids |= self._inspection_ids
            self._inspection_ids = set()
            self._inspection_until_ns = 0
            return False
        novel_ids = fresh_ids - self._inspection_done_ids
        if not novel_ids:
            return False
        window_s = inspection_pause_duration_s(
            len(novel_ids),
            self._float_parameter("search_inspection_per_person_s"),
            self._float_parameter("search_inspection_max_s"),
        )
        if window_s <= 0.0:
            self._inspection_done_ids |= novel_ids
            return False
        self._inspection_ids = set(novel_ids)
        self._inspection_until_ns = now_ns + int(window_s * 1_000_000_000)
        self._cancel_spin()
        self._last_spin_ns = now_ns
        self._publish_status("searching", "search_inspecting")
        return True

    def _spin_search(self, now_ns: int, scan_fresh: bool, allow_drive: bool = True) -> None:
        """Ride the live Nav2 goal; search with bounded Nav2 /spin when free.

        The last follow NavigateToPose goal is never cancelled on loss:
        its trajectory may still drive the robot to the person, and every
        terminal result (success, cancel, abort) flows through the existing
        goal cleanup. While a follow goal is active or pending, send
        nothing and let Nav2 drive. Only when no follow goal remains,
        send at most one bounded Spin.Goal.target_yaw at a time through
        the lifetime /spin action client, accumulating accepted yaw up to
        search_max_spin_revolutions full turns before stopping for good.
        """
        if not self._searching:
            self._searching = True
            self._lost_since_ns = now_ns
            self._spin_accum_rad = 0.0
            self._last_spin_ns = now_ns
            self._spin_capped = False
            self._inspection_until_ns = 0
            self._inspection_ids = set()
            self._inspection_done_ids = set()
        if self._active_goal is not None or self._pending_goal is not None:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_riding_nav2_goal")
            return
        if self._inspect_people(now_ns):
            return
        if self._pending_spin is not None or self._active_spin is not None:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spinning")
            return
        timeout_s = self._float_parameter("search_timeout_s")
        if timeout_s > 0.0 and (now_ns - self._lost_since_ns) / 1_000_000_000.0 > timeout_s:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_timeout_idle")
            return
        if not scan_fresh:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_scan_stale")
            return
        if self._spin_capped:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spin_done")
            return
        bearing_rad = self._last_seen_bearing_rad
        if bearing_rad is None or not math.isfinite(bearing_rad):
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_no_bearing")
            return
        yaw_rate = search_spin_command(
            bearing_rad,
            self._last_person_speed_m_s,
            self._last_seen_range_m,
            self._float_parameter("search_spin_default_rate_rad_s"),
            self._float_parameter("search_spin_min_rate_rad_s"),
            self._float_parameter("search_spin_max_rate_rad_s"),
            self._float_parameter("search_spin_deadband_rad"),
        )
        if yaw_rate is None:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_no_bearing")
            return
        max_rev = self._float_parameter("search_max_spin_revolutions")
        if max_rev > 0.0 and self._spin_accum_rad >= max_rev * 2.0 * math.pi:
            self._spin_capped = True
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spin_done")
            return
        if yaw_rate == 0.0:
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spinning")
            return
        cap_rad = max_rev * 2.0 * math.pi if max_rev > 0.0 else math.pi
        remaining = cap_rad - self._spin_accum_rad if max_rev > 0.0 else math.pi
        if max_rev > 0.0 and remaining <= 0.0:
            self._spin_capped = True
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spin_done")
            return
        bounded = min(math.pi, max(0.0, remaining))
        if bounded <= 0.0:
            self._spin_capped = True
            self._last_spin_ns = now_ns
            self._publish_status("searching", "search_spin_done")
            return
        direction = 1.0 if yaw_rate > 0.0 else -1.0
        target_yaw = direction * bounded
        if self._send_spin(target_yaw):
            self._spin_accum_rad += bounded
        self._last_spin_ns = now_ns
        drive_window_s = self._float_parameter("search_drive_duration_s")
        if (
            allow_drive
            and drive_window_s > 0.0
            and (now_ns - self._lost_since_ns) / 1_000_000_000.0 <= drive_window_s
        ):
            self._publish_status("searching", "search_driving")
        elif allow_drive:
            self._publish_status("searching", "search_spinning")
        else:
            self._publish_status("searching", "search_spinning_no_position")

    def _is_confirmed_target(
        self,
        target: PersonTarget | None,
        geometry: TargetGeometry | None,
    ) -> bool:
        return (
            target is not None
            and geometry is not None
            and target.track_id >= 0
            and geometry.valid
            and target.identity_confirmed
            and geometry.state == "tracking"
            and 0.0 <= geometry.age_s <= self._float_parameter("max_target_age_s")
        )

    def _publish_status(self, phase: str, reason: str) -> None:
        """Publish structured follow state for operators and QA."""
        message = String()
        message.data = f"phase={phase} reason={reason}"
        self._status_pub.publish(message)

    def _request_cancel_graced(self, reason: str) -> None:
        limit = self._int_parameter("cancel_grace_cycles")
        self._cancel_grace, fire = grace_advance(self._cancel_grace, True, limit)
        if fire:
            self._request_cancel(reason)

    def _on_nav2_result(self, succeeded: bool, mode: GoalMode | None, token: str | None = None) -> None:
        return None

    def _target_geometry(self, target: PersonTarget, now_ns: int) -> TargetGeometry:
        stamp_ns = target.header.stamp.sec * 1_000_000_000 + target.header.stamp.nanosec
        return TargetGeometry(
            state=target.state,
            valid=target.valid,
            point_base_link=(
                float(target.point_base_link.x),
                float(target.point_base_link.y),
                float(target.point_base_link.z),
            ),
            range_m=float(target.range_m),
            bearing_rad=float(target.bearing_rad),
            age_s=(now_ns - stamp_ns) / 1_000_000_000.0,
            frame_id=str(target.header.frame_id),
        )

    def _obstacle_points(self, now_ns: int) -> tuple[ObstaclePoint, ...] | None:
        values = parse_obstacle_cloud(
            self._obstacles,
            now_ns,
            self._float_parameter("max_target_age_s"),
            str(self.get_parameter("base_frame").value),
            self._int_parameter("max_obstacle_points"),
        )
        if values is None:
            self._obstacles = None
            return None
        return tuple(ObstaclePoint(point_base_link=point) for point in values)

    def _follow_config(self) -> FollowConfig:
        return FollowConfig(
            follow_distance_m=self._float_parameter("follow_distance_m"),
            follow_resume_distance_m=self._float_parameter("follow_resume_distance_m"),
            max_target_age_s=self._float_parameter("max_target_age_s"),
            corridor_half_width_m=self._float_parameter("corridor_half_width_m"),
            goal_update_distance_m=self._float_parameter("goal_update_distance_m"),
            base_frame=str(self.get_parameter("base_frame").value),
        )

    def _float_parameter(self, name: str) -> float:
        return float(str(self.get_parameter(name).value))

    def _int_parameter(self, name: str) -> int:
        return int(str(self.get_parameter(name).value))
