"""Nav2 Spin action-client lifecycle for the person-follow runtime."""

from __future__ import annotations

from typing import Protocol

from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future


class _SpinRuntime(Protocol):
    _pending_spin: Future | None
    _active_spin: ClientGoalHandle | None
    _pending_spin_yaw: float
    _spin_accum_rad: float
    _searching: bool
    _spin_action_client: ActionClient

    def get_logger(self) -> RcutilsLogger: ...

    def _spin_goal_response_callback(self, future: Future) -> None: ...

    def _spin_result_callback(self, future: Future) -> None: ...

    def _spin_cancel_response_callback(self, future: Future) -> None: ...


class Nav2PersonFollowSpinActions:
    """Manage one in-flight Spin goal and its lifecycle callbacks."""

    def _send_spin(self: _SpinRuntime, target_yaw: float) -> bool:
        """Send one bounded Spin goal when no spin is in flight."""
        if self._pending_spin is not None or self._active_spin is not None:
            return False
        if not self._spin_action_client.server_is_ready():
            return False
        goal = Spin.Goal()
        goal.target_yaw = target_yaw
        pending = self._spin_action_client.send_goal_async(goal)
        self._pending_spin = pending
        self._pending_spin_yaw = target_yaw
        pending.add_done_callback(self._spin_goal_response_callback)
        return True

    def _spin_goal_response_callback(self: _SpinRuntime, future: Future) -> None:
        yaw = self._pending_spin_yaw
        self._pending_spin = None
        self._pending_spin_yaw = 0.0
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin goal request failed: {error}")
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        handle: ClientGoalHandle | None = future.result()
        if handle is None:
            self.get_logger().error("Spin goal request returned no handle")
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        if not handle.accepted:
            self.get_logger().warning("Spin goal rejected")
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        if not self._searching:
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            cancel_future = handle.cancel_goal_async()
            cancel_future.add_done_callback(self._spin_cancel_response_callback)
            return
        self._active_spin = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._spin_result_callback)

    def _spin_result_callback(self: _SpinRuntime, future: Future) -> None:
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin result failed: {error}")
        else:
            result = future.result()
            if result is None:
                self.get_logger().error("Spin result returned no status")
        self._active_spin = None

    def _spin_cancel_response_callback(self: _SpinRuntime, future: Future) -> None:
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin cancellation failed: {error}")

    def _cancel_spin(self: _SpinRuntime) -> None:
        pending_yaw = self._pending_spin_yaw
        self._pending_spin = None
        self._pending_spin_yaw = 0.0
        if pending_yaw != 0.0:
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(pending_yaw))
        goal = self._active_spin
        self._active_spin = None
        if goal is not None:
            cancel_future = goal.cancel_goal_async()
            cancel_future.add_done_callback(self._spin_cancel_response_callback)
