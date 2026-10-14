"""Nav2 Spin action-client lifecycle for the person-follow runtime."""

from __future__ import annotations

from typing import Protocol

from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.clock import Clock
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future


class _SpinRuntime(Protocol):
    _pending_spin: Future | None
    _active_spin: ClientGoalHandle | None
    _pending_spin_yaw: float
    _spin_accum_rad: float
    _searching: bool
    _spin_action_client: ActionClient
    _spin_cancel_requested: bool
    _spin_cancel_pending: bool
    _spin_cancel_deadline_ns: int
    _spin_cancel_timed_out: bool

    def get_logger(self) -> RcutilsLogger: ...

    def get_clock(self) -> Clock: ...

    def _float_parameter(self, name: str) -> float: ...

    def _spin_goal_response_callback(self, future: Future) -> None: ...

    def _spin_result_callback(self, future: Future, handle: ClientGoalHandle | None = ...) -> None: ...

    def _spin_cancel_response_callback(self, future: Future) -> None: ...

    def _issue_spin_cancel(self, goal: ClientGoalHandle) -> None: ...

    def _resolve_spin_cancel_timeout(self, now_ns: int) -> None: ...

    def _spin_handoff_blocked(self, now_ns: int) -> bool: ...

    def _cancel_spin(self) -> None: ...


class Nav2PersonFollowSpinActions:
    """Manage one in-flight Spin goal and its lifecycle callbacks."""

    def _send_spin(self: _SpinRuntime, target_yaw: float) -> bool:
        """Send one bounded Spin goal when no spin is in flight."""
        if (
            self._pending_spin is not None
            or self._active_spin is not None
            or self._spin_cancel_requested
            or self._spin_cancel_pending
        ):
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
        """Own the pending send until the server accepts or rejects it.

        The pending future is never dropped: a rejection frees the handoff,
        while an acceptance installs the handle and — when a cancel was
        requested meanwhile, or search already ended — cancels it at once
        and keeps awaiting the terminal result. No replacement goal may be
        sent until that terminal result arrives.
        """
        yaw = self._pending_spin_yaw
        self._pending_spin = None
        self._pending_spin_yaw = 0.0
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin goal request failed: {error}")
            self._spin_cancel_requested = False
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        handle: ClientGoalHandle | None = future.result()
        if handle is None:
            self.get_logger().error("Spin goal request returned no handle")
            self._spin_cancel_requested = False
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        if not handle.accepted:
            self.get_logger().warning("Spin goal rejected")
            self._spin_cancel_requested = False
            self._spin_accum_rad = max(0.0, self._spin_accum_rad - abs(yaw))
            return
        self._active_spin = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done, owned=handle: self._spin_result_callback(done, owned))
        if self._spin_cancel_requested or not self._searching:
            self._spin_cancel_requested = False
            self._issue_spin_cancel(handle)
            return

    def _spin_result_callback(
        self: _SpinRuntime, future: Future, handle: ClientGoalHandle | None = None,
    ) -> None:
        """Clear the handoff only on the owned handle's terminal result.

        A late result for a previous handle — arriving after a newer goal
        took ownership — is ignored so it can never free or clear the new
        goal. Callers without handle knowledge pass None and keep the
        legacy unconditional clear.
        """
        if handle is not None and self._active_spin is not handle:
            self.get_logger().error("Stale Spin result ignored; newer goal owns the handoff")
            return
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin result failed: {error}")
        else:
            result = future.result()
            if result is None:
                self.get_logger().error("Spin result returned no status")
        self._active_spin = None
        self._spin_cancel_requested = False
        self._spin_cancel_pending = False
        self._spin_cancel_deadline_ns = 0
        self._spin_cancel_timed_out = False

    def _spin_cancel_response_callback(self: _SpinRuntime, future: Future) -> None:
        error = future.exception()
        if error is not None:
            self.get_logger().error(f"Spin cancellation failed: {error}")
            return
        self.get_logger().debug("Spin cancellation acknowledged; awaiting result")

    def _resolve_spin_cancel_timeout(self: _SpinRuntime, now_ns: int) -> None:
        """Latch a fail-closed error when the terminal result never arrives.

        The handoff stays blocked: motion never resumes on timeout alone,
        only on the terminal result. The timeout only makes the hold
        observable (error log plus timeout status reason) instead of
        looking like an ordinary in-flight cancel.
        """
        if not self._spin_cancel_pending or self._spin_cancel_timed_out:
            return
        if now_ns < self._spin_cancel_deadline_ns:
            return
        self.get_logger().error(
            "Spin cancellation timed out; holding. Motion stays blocked "
            "until the terminal Spin result arrives")
        self._spin_cancel_timed_out = True

    def _spin_handoff_blocked(self: _SpinRuntime, now_ns: int) -> bool:
        """True while a stale spin must still block follow/spin commands."""
        self._resolve_spin_cancel_timeout(now_ns)
        return (
            self._pending_spin is not None
            or self._active_spin is not None
            or self._spin_cancel_requested
            or self._spin_cancel_pending
        )

    def _cancel_spin(self: _SpinRuntime) -> None:
        """Request cancellation without dropping owned futures.

        A pending send stays owned until the server accepts or rejects it;
        the response cancels on acceptance. An active goal gets one cancel
        request and stays owned until its terminal result. Either way no
        replacement goal may be sent meanwhile.
        """
        if self._pending_spin is not None:
            self._spin_cancel_requested = True
        goal = self._active_spin
        if goal is None:
            return
        if self._spin_cancel_pending:
            return
        self._issue_spin_cancel(goal)

    def _issue_spin_cancel(self: _SpinRuntime, goal: ClientGoalHandle) -> None:
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(self._float_parameter("cancel_timeout_s") * 1_000_000_000)
        self._spin_cancel_pending = True
        self._spin_cancel_timed_out = False
        self._spin_cancel_deadline_ns = now_ns + max(0, timeout_ns)
        cancel_future = goal.cancel_goal_async()
        cancel_future.add_done_callback(self._spin_cancel_response_callback)
