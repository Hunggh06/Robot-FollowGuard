"""Nav2 action-client lifecycle for the person-follow runtime."""

from __future__ import annotations

import math
from typing import Protocol
from typing_extensions import assert_never

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.parameter import Parameter
from rclpy.task import Future
from tf2_ros import Buffer, TransformException

from .follow_goal_helpers import FollowDecision, FollowDecisionStatus
from .goal_update_logic import (
    GoalMode,
    GoalRequest,
    GoalSnapshot,
    GoalUpdateThresholds,
    goal_requires_update,
)


class _ActionLogger(Protocol):
    def debug(self, msg: str) -> None: ...

    def error(self, msg: str) -> None: ...

    def info(self, msg: str) -> None: ...

    def warning(self, msg: str) -> None: ...

    def fatal(self, msg: str) -> None: ...


class _ActionRuntime(Protocol):
    _action_client: ActionClient
    _active_goal: ClientGoalHandle | None
    _active_goal_token: str | None
    _cancel_requested: bool
    _cancel_sent: bool
    _cancel_sent_at_ns: int
    _last_goal_base_link: tuple[float, float] | None
    _last_goal_map: GoalSnapshot | None
    _last_goal_mode: GoalMode | None
    _pending_goal: Future | None
    _queued_goal: GoalRequest | None
    _tf_buffer: Buffer

    def _float_parameter(self, name: str) -> float: ...

    def get_logger(self) -> RcutilsLogger: ...

    def get_clock(self) -> Clock: ...

    def get_parameter(self, name: str) -> Parameter: ...

    def _cancel_response_callback(self, future: Future) -> None: ...

    def _goal_response_callback(self, future: Future) -> None: ...

    def _request_cancel(self, reason: str, preserve_queued: bool = False) -> None: ...

    def _report_rejected_goal(self) -> None: ...

    def _result_callback(self, future: Future) -> None: ...

    def _send_cancel(self) -> None: ...

    def _send_queued_goal(self) -> None: ...

    def _on_nav2_result(self, succeeded: bool, mode: GoalMode | None, token: str | None = None) -> None: ...

    def _send_goal(
        self,
        request: GoalRequest,
    ) -> None: ...

    def _transform_goal(
        self,
        goal_base_link: tuple[float, float],
        yaw_rad: float,
        stamp: Time,
    ) -> PoseStamped | None: ...


class Nav2PersonFollowActions:
    """Manage one active NavigateToPose goal and its lifecycle callbacks."""

    def _apply_decision(self: _ActionRuntime, decision: FollowDecision, stamp: Time) -> None:
        match decision.status:
            case FollowDecisionStatus.GOAL:
                if decision.goal_base_link is None or decision.goal_yaw_rad is None:
                    return
                self._cancel_requested = False
                self._send_goal(
                    GoalRequest(decision.goal_base_link, decision.goal_yaw_rad, stamp, "follow")
                )
            case FollowDecisionStatus.HOLD:
                if decision.cancel_active_goal or self._last_goal_mode == "recovery":
                    self._last_goal_base_link = None
                    self._last_goal_map = None
                    self._last_goal_mode = None
                    self._request_cancel(decision.reason)
            case FollowDecisionStatus.CANCEL:
                self._request_cancel(decision.reason)
            case unreachable:
                assert_never(unreachable)

    def _send_goal(
        self: _ActionRuntime,
        request: GoalRequest,
    ) -> None:
        goal_base_link = request.goal_base_link
        yaw_rad = request.yaw_rad
        stamp = request.stamp
        mode = request.mode
        pose = self._transform_goal(goal_base_link, yaw_rad, stamp)
        if pose is None:
            return
        current_goal = GoalSnapshot(
            (float(pose.pose.position.x), float(pose.pose.position.y)),
            math.atan2(
                2.0 * (pose.pose.orientation.w * pose.pose.orientation.z),
                1.0 - 2.0 * pose.pose.orientation.z * pose.pose.orientation.z,
            ),
            mode,
        )
        if not goal_requires_update(
            self._last_goal_map,
            current_goal,
            GoalUpdateThresholds(
                self._float_parameter("goal_update_distance_m"),
                self._float_parameter("goal_update_yaw_rad"),
            ),
        ):
            return
        if self._active_goal is not None or self._pending_goal is not None:
            self._queued_goal = request
            self._request_cancel("goal_update", preserve_queued=True)
            return
        if not self._action_client.server_is_ready():
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        pending_goal = self._action_client.send_goal_async(goal)
        self._active_goal_token = request.token
        self._pending_goal = pending_goal
        pending_goal.add_done_callback(self._goal_response_callback)
        self._last_goal_base_link = goal_base_link
        self._last_goal_map = current_goal
        self._last_goal_mode = mode

    def _transform_goal(
        self: _ActionRuntime,
        goal_base_link: tuple[float, float],
        yaw_rad: float,
        stamp: Time,
    ) -> PoseStamped | None:
        source = PoseStamped()
        source.header.frame_id = str(self.get_parameter("base_frame").value)
        source.header.stamp = stamp
        source.pose.position.x = goal_base_link[0]
        source.pose.position.y = goal_base_link[1]
        source.pose.orientation.z = math.sin(yaw_rad / 2.0)
        source.pose.orientation.w = math.cos(yaw_rad / 2.0)
        try:
            transformed = self._tf_buffer.transform(
                source,
                str(self.get_parameter("map_frame").value),
                timeout=Duration(
                    nanoseconds=int(self._float_parameter("tf_timeout_s") * 1_000_000_000),
                ),
            )
        except TransformException as error:
            self.get_logger().warning(f"goal transform unavailable: {error}")
            return None
        transformed.header.frame_id = str(self.get_parameter("map_frame").value)
        transformed.header.stamp = stamp
        return transformed

    def _request_cancel(
        self: _ActionRuntime,
        reason: str,
        preserve_queued: bool = False,
    ) -> None:
        self._cancel_requested = True
        if not preserve_queued:
            self._queued_goal = None
        now_ns = self.get_clock().now().nanoseconds
        goal = self._active_goal
        if goal is None:
            if self._pending_goal is None:
                self._last_goal_base_link = None
                self._last_goal_map = None
                self._last_goal_mode = None
            return
        if self._cancel_sent:
            timeout_ns = int(self._float_parameter("cancel_timeout_s") * 1_000_000_000)
            if now_ns - self._cancel_sent_at_ns < timeout_ns:
                return
            self._cancel_sent = False
        self._cancel_sent = True
        self._cancel_sent_at_ns = now_ns
        self.get_logger().info(f"cancelling Nav2 follow goal: {reason}")
        cancel_future = goal.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_response_callback)

    def _goal_response_callback(self: _ActionRuntime, future: Future) -> None:
        self._pending_goal = None
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"NavigateToPose goal request failed: {error}")
            self._report_rejected_goal()
            return
        goal_handle: ClientGoalHandle | None = future.result()
        if goal_handle is None:
            self.get_logger().error("NavigateToPose goal request returned no handle")
            self._report_rejected_goal()
            return
        if not goal_handle.accepted:
            self.get_logger().warning("NavigateToPose goal rejected")
            self._report_rejected_goal()
            return
        self._active_goal = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)
        if self._cancel_requested:
            self._send_cancel()

    def _report_rejected_goal(self: _ActionRuntime) -> None:
        mode = self._last_goal_mode
        token = self._active_goal_token
        self._active_goal_token = None
        self._active_goal = None
        self._last_goal_base_link = None
        self._last_goal_map = None
        self._last_goal_mode = None
        self._on_nav2_result(False, mode, token)
        self._send_queued_goal()

    def _send_cancel(self: _ActionRuntime) -> None:
        goal = self._active_goal
        if goal is None or self._cancel_sent:
            return
        self._cancel_sent = True
        self._cancel_sent_at_ns = self.get_clock().now().nanoseconds
        cancel_future = goal.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_response_callback)

    def _cancel_response_callback(self: _ActionRuntime, future: Future) -> None:
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"NavigateToPose cancellation failed: {error}")
            self._cancel_sent = False

    def _result_callback(self: _ActionRuntime, future: Future) -> None:
        error = future.exception()
        succeeded = False
        canceled = False
        if error is not None:
            self.get_logger().error(f"NavigateToPose result failed: {error}")
        else:
            result = future.result()
            if result is None:
                self.get_logger().error("NavigateToPose result returned no status")
            elif result.status == GoalStatus.STATUS_SUCCEEDED:
                succeeded = True
                self.get_logger().info("NavigateToPose follow goal succeeded")
            elif result.status == GoalStatus.STATUS_CANCELED:
                canceled = True
            else:
                self.get_logger().warning(
                    f"NavigateToPose follow goal finished with status {result.status}",
                )
        self._active_goal = None
        self._last_goal_base_link = None
        self._last_goal_map = None
        mode = self._last_goal_mode
        token = self._active_goal_token
        self._active_goal_token = None
        self._last_goal_mode = None
        self._cancel_requested = False
        self._cancel_sent = False
        if not canceled:
            self._on_nav2_result(succeeded, mode, token)
        self._send_queued_goal()

    def _send_queued_goal(self: _ActionRuntime) -> None:
        queued_goal = self._queued_goal
        self._queued_goal = None
        if queued_goal is not None:
            self._send_goal(queued_goal)
