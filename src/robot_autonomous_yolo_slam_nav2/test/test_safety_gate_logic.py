import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
    GateConfig,
    TwistCommand,
    WatchdogState,
    gate_command,
    min_forward_corridor_distance,
    scan_to_base_points,
    watchdog_step,
)


def test_missing_or_malformed_scan_stops_forward_but_preserves_rotation() -> None:
    command = TwistCommand(linear_x=0.4, angular_z=0.2)

    result = gate_command(command, None, GateConfig())

    assert result == TwistCommand(linear_x=0.0, angular_z=0.2)


def test_reverse_and_in_place_rotation_are_not_blocked_by_forward_corridor() -> None:
    config = GateConfig()

    reverse = gate_command(TwistCommand(-0.4, 0.2), None, config)
    rotate = gate_command(TwistCommand(0.0, 0.2), None, config)

    assert reverse == TwistCommand(-0.4, 0.2)
    assert rotate == TwistCommand(0.0, 0.2)


def test_disabled_reverse_blocks_backup_without_blocking_rotation() -> None:
    config = GateConfig(allow_reverse=False)
    assert gate_command(TwistCommand(-0.1, 0.2), None, config) == TwistCommand(0.0, 0.2)
    assert gate_command(TwistCommand(0.0, -0.2), None, config) == TwistCommand(0.0, -0.2)


def test_corridor_uses_front_x_and_lateral_width() -> None:
    points = ((0.8, 0.1), (0.5, 0.8), (-0.1, 0.0))

    assert min_forward_corridor_distance(points, half_width_m=0.3) == pytest.approx(0.8)


def test_stop_and_slow_bands_are_fail_closed() -> None:
    config = GateConfig(stop_distance_m=0.5, slow_distance_m=1.0, slow_scale=0.3)
    blocked = gate_command(TwistCommand(0.4, 0.0), ((0.4, 0.0),), config)
    slow = gate_command(TwistCommand(0.4, 0.0), ((0.8, 0.0),), config)
    clear = gate_command(TwistCommand(0.4, 0.0), ((1.5, 0.0),), config)

    assert blocked.linear_x == 0.0
    assert slow.linear_x == pytest.approx(0.288)
    assert clear.linear_x == pytest.approx(0.4)


def test_nonfinite_scan_distance_is_not_treated_as_clear() -> None:
    assert min_forward_corridor_distance(((math.nan, 0.0),), 0.3) is None


def test_scan_points_are_transformed_into_base_frame_before_corridor_check() -> None:
    points = scan_to_base_points(
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        range_min=0.1,
        range_max=5.0,
        ranges=(1.0, math.inf),
        translation=(0.2, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert points is not None
    assert points[0] == pytest.approx((1.2, 0.0))


def test_command_timeout_publishes_zero_after_forward_scan_stop_latch() -> None:
    state = WatchdogState(timeout_latched=False, scan_latched=True)

    next_state, publish_zero = watchdog_step(
        state,
        command_timed_out=True,
        forward_scan_stale=False,
    )

    assert publish_zero
    assert next_state == WatchdogState(timeout_latched=True, scan_latched=False)


@pytest.mark.parametrize("angular_z", [-0.4, 0.4])
def test_slowdown_preserves_nav2_turning_curvature(angular_z: float) -> None:
    command = TwistCommand(0.2, angular_z)

    result = gate_command(command, ((0.6, 0.0),), GateConfig())

    assert 0.0 < result.linear_x < command.linear_x
    assert result.angular_z / result.linear_x == pytest.approx(command.angular_z / command.linear_x)


def test_slowdown_does_not_drop_to_thirty_percent_at_band_entry() -> None:
    command = TwistCommand(0.2, 0.4)

    result = gate_command(command, ((0.6999, 0.0),), GateConfig())

    assert result.linear_x == pytest.approx(0.2, abs=0.0001)
    assert result.angular_z == pytest.approx(0.4, abs=0.0001)
