"""Pure forward-corridor command gating used by the ROS safety node."""

from __future__ import annotations

import math
from dataclasses import dataclass


Point2D = tuple[float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Gate thresholds after master clearance derivation."""

    stop_distance_m: float = 0.4
    slow_distance_m: float = 0.7
    corridor_half_width_m: float = 0.22
    slow_scale: float = 0.5
    allow_reverse: bool = True


@dataclass(frozen=True, slots=True)
class TwistCommand:
    """The command components relevant to the gate contract."""

    linear_x: float
    angular_z: float


@dataclass(frozen=True, slots=True)
class WatchdogState:
    """Independent latches for command timeout and forward scan stops."""

    timeout_latched: bool = False
    scan_latched: bool = False


def watchdog_step(
    state: WatchdogState,
    command_timed_out: bool,
    forward_scan_stale: bool,
) -> tuple[WatchdogState, bool]:
    """Return the next watchdog latches and whether zero must be published."""
    publish_zero = (
        command_timed_out and not state.timeout_latched
    ) or (
        forward_scan_stale and not state.scan_latched
    )
    return WatchdogState(command_timed_out, forward_scan_stale), publish_zero


def min_forward_corridor_distance(
    points: tuple[Point2D, ...],
    half_width_m: float,
) -> float | None:
    """Return nearest finite point in front corridor, or no scan evidence."""
    distances = tuple(
        x
        for x, y in points
        if math.isfinite(x)
        and math.isfinite(y)
        and x > 0.0
        and abs(y) <= half_width_m
    )
    return min(distances) if distances else None


def scan_to_base_points(
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    ranges: tuple[float, ...],
    translation: Point2D,
    rotation: Quaternion,
) -> tuple[Point2D, ...] | None:
    """Transform finite LaserScan returns into the configured base frame."""
    if (
        not math.isfinite(angle_min)
        or not math.isfinite(angle_increment)
        or abs(angle_increment) <= 1e-9
        or not math.isfinite(range_min)
        or not math.isfinite(range_max)
        or range_min <= 0.0
        or range_max < range_min
        or not all(math.isfinite(value) for value in (*translation, *rotation))
    ):
        return None
    qx, qy, qz, qw = rotation
    points: list[Point2D] = []
    for index, distance in enumerate(ranges):
        if not math.isfinite(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + index * angle_increment
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)
        rotated_x = (
            (1.0 - 2.0 * (qy * qy + qz * qz)) * local_x
            + 2.0 * (qx * qy - qz * qw) * local_y
        )
        rotated_y = (
            2.0 * (qx * qy + qz * qw) * local_x
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * local_y
        )
        points.append((rotated_x + translation[0], rotated_y + translation[1]))
    return tuple(points)


def gate_command(
    command: TwistCommand,
    corridor_points: tuple[Point2D, ...] | None,
    config: GateConfig,
) -> TwistCommand:
    """Stop or slow forward motion while preserving reverse and rotation."""
    if not math.isfinite(command.linear_x) or not math.isfinite(command.angular_z):
        return TwistCommand(0.0, 0.0)
    if command.linear_x <= 0.0:
        if not config.allow_reverse:
            return TwistCommand(0.0, command.angular_z)
        return command
    if corridor_points is None:
        return TwistCommand(0.0, command.angular_z)
    distance = min_forward_corridor_distance(corridor_points, config.corridor_half_width_m)
    if distance is None or distance <= config.stop_distance_m:
        return TwistCommand(0.0, command.angular_z)
    if distance <= config.slow_distance_m:
        fraction = (distance - config.stop_distance_m) / (
            config.slow_distance_m - config.stop_distance_m
        )
        scale = config.slow_scale + (1.0 - config.slow_scale) * fraction
        return TwistCommand(command.linear_x * scale, command.angular_z * scale)
    return command
