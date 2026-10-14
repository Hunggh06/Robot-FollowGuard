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


def test_nonfinite_scan_age_is_treated_as_stale() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    _, is_stale = debounce_scan_stale(
        ScanDebounceState(), math.nan, ScanDebounceConfig()
    )

    assert is_stale is True


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


def test_scan_jitter_single_fresh_blip_stays_stopped() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.25, recover_age_s=0.20, recover_count=3)
    state = ScanDebounceState(stale_latched=True, fresh_consec=0)
    new_state, is_stale = debounce_scan_stale(state, 0.10, config)
    assert is_stale is True
    assert new_state.stale_latched is True
    # Fail-closed: stale scan must gate to exact zero forward, never reduced.
    gated = gate_command(TwistCommand(linear_x=0.4, angular_z=0.1), None, GateConfig())
    assert gated.linear_x == 0.0


def test_scan_stale_triggers_immediately_but_recovery_needs_consecutive_fresh() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.25, recover_age_s=0.20, recover_count=3)
    fresh = ScanDebounceState(stale_latched=False, fresh_consec=0)
    latched, is_stale = debounce_scan_stale(fresh, 0.30, config)
    assert is_stale is True
    assert latched.stale_latched is True
    # Marginal age inside hysteresis band must not recover.
    still, is_stale2 = debounce_scan_stale(latched, 0.22, config)
    assert is_stale2 is True
    # N consecutive clearly-fresh scans recover.
    state = still
    recovered = False
    for _ in range(3):
        state, recovered_stale = debounce_scan_stale(state, 0.10, config)
        recovered = not recovered_stale
    assert recovered is True
    assert state.stale_latched is False


def test_debounce_config_normalization_is_fail_closed() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
    )

    assert hasattr(ScanDebounceConfig, "normalized")
    # Non-finite or negative max age collapses to zero (everything stale).
    assert ScanDebounceConfig(float("nan"), 0.20, 3).normalized().max_age_s == 0.0
    assert ScanDebounceConfig(float("inf"), 0.20, 3).normalized().max_age_s == 0.0
    assert ScanDebounceConfig(-0.1, 0.20, 3).normalized().max_age_s == 0.0
    # Non-finite or negative recover age collapses to zero (recovery ~never).
    assert ScanDebounceConfig(0.25, float("nan"), 3).normalized().recover_age_s == 0.0
    assert ScanDebounceConfig(0.25, -0.05, 3).normalized().recover_age_s == 0.0
    # Recover age above max clamps down to max, never widening acceptance.
    clamped = ScanDebounceConfig(0.25, 0.50, 3).normalized()
    assert clamped.recover_age_s == 0.25
    # Recover count below one clamps up to one.
    assert ScanDebounceConfig(0.25, 0.20, 0).normalized().recover_count == 1
    # Valid config is unchanged.
    valid = ScanDebounceConfig(0.25, 0.20, 3).normalized()
    assert (valid.max_age_s, valid.recover_age_s, valid.recover_count) == (0.25, 0.20, 3)


def test_debounce_with_invalid_config_never_opens_forward() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    for bad in [
        ScanDebounceConfig(float("nan"), 0.20, 3),
        ScanDebounceConfig(-0.5, 0.20, 3),
        ScanDebounceConfig(0.25, float("nan"), 3),
        ScanDebounceConfig(0.25, 0.50, 3),
    ]:
        latched = ScanDebounceState(stale_latched=True, fresh_consec=0)
        _, is_stale = debounce_scan_stale(latched, 0.10, bad)
        assert is_stale is True


def test_debounce_nonfinite_age_is_stale() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.25, recover_age_s=0.20, recover_count=3)
    for age in (float("nan"), float("inf"), float("-inf")):
        for state in (
            ScanDebounceState(stale_latched=False, fresh_consec=0),
            ScanDebounceState(stale_latched=True, fresh_consec=0),
        ):
            _, is_stale = debounce_scan_stale(state, age, config)
            assert is_stale is True


def test_safety_gate_node_debounce_config_normalizes_invalid_settings() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_node import SafetyGateNode

    class _FakeParam:
        def __init__(self, value) -> None:
            self.value = value

    class _FakeNode:
        def __init__(self, params: dict) -> None:
            self._params = params

        def get_parameter(self, name: str) -> _FakeParam:
            return _FakeParam(self._params[name])

    def _config_for(params: dict):
        fake = _FakeNode(params)
        # Reuse the real helpers unbound so the test exercises node parsing.
        fake._float_parameter = SafetyGateNode._float_parameter.__get__(fake)  # type: ignore[attr-defined]
        return SafetyGateNode._debounce_config(fake)  # type: ignore[arg-type]

    # Non-finite max age must collapse fail-closed instead of reopening motion.
    cfg = _config_for(
        {"max_scan_age_s": float("nan"), "scan_recover_age_s": 0.20, "scan_recover_count": 3}
    )
    assert cfg.max_age_s == 0.0
    # Garbage strings must not raise; they collapse fail-closed.
    cfg = _config_for(
        {"max_scan_age_s": "garbage", "scan_recover_age_s": 0.20, "scan_recover_count": 3}
    )
    assert cfg.max_age_s == 0.0
    # Recover age above max clamps down, never widening acceptance.
    cfg = _config_for(
        {"max_scan_age_s": 0.25, "scan_recover_age_s": 0.50, "scan_recover_count": 3}
    )
    assert cfg.recover_age_s == 0.25
    # Non-finite recover age collapses to zero.
    cfg = _config_for(
        {"max_scan_age_s": 0.25, "scan_recover_age_s": float("inf"), "scan_recover_count": 3}
    )
    assert cfg.recover_age_s == 0.0
    # Recover count below one clamps up; garbage counts fall back without raising.
    cfg = _config_for(
        {"max_scan_age_s": 0.25, "scan_recover_age_s": 0.20, "scan_recover_count": 0}
    )
    assert cfg.recover_count == 1
    cfg = _config_for(
        {"max_scan_age_s": 0.25, "scan_recover_age_s": 0.20, "scan_recover_count": "bad"}
    )
    assert cfg.recover_count >= 1


def test_safety_gate_yaml_declares_recover_parameters() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "config" / "person_follow.yaml"
    config = yaml.safe_load(config_path.read_text())
    parameters = config["safety_gate"]["ros__parameters"]
    assert parameters["max_scan_age_s"] == 0.60
    assert parameters["scan_recover_age_s"] == 0.45
    assert parameters["scan_recover_count"] == 3


def test_debounce_repeated_evaluations_of_one_scan_do_not_recover() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.60, recover_age_s=0.45, recover_count=3)
    state = ScanDebounceState(stale_latched=True, fresh_consec=0)
    state, _ = debounce_scan_stale(state, 0.70, config, stamp_ns=100)
    assert state.stale_latched is True
    state, is_stale = debounce_scan_stale(state, 0.10, config, stamp_ns=200)
    assert is_stale is True
    for _ in range(5):
        state, is_stale = debounce_scan_stale(state, 0.15, config, stamp_ns=200)
        assert is_stale is True
    assert state.fresh_consec == 1


def test_debounce_recovery_needs_distinct_fresh_scans() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.60, recover_age_s=0.45, recover_count=3)
    state = ScanDebounceState(stale_latched=True, fresh_consec=0)
    state, _ = debounce_scan_stale(state, 0.70, config, stamp_ns=100)
    state, first = debounce_scan_stale(state, 0.10, config, stamp_ns=200)
    assert first is True
    state, second = debounce_scan_stale(state, 0.10, config, stamp_ns=300)
    assert second is True
    state, third = debounce_scan_stale(state, 0.10, config, stamp_ns=400)
    assert third is False
    assert state.stale_latched is False


def test_debounce_stale_age_resets_distinct_sample_count() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.60, recover_age_s=0.45, recover_count=3)
    state = ScanDebounceState(stale_latched=True, fresh_consec=0)
    state, _ = debounce_scan_stale(state, 0.10, config, stamp_ns=200)
    state, _ = debounce_scan_stale(state, 0.10, config, stamp_ns=300)
    assert state.fresh_consec == 2
    state, is_stale = debounce_scan_stale(state, 0.70, config, stamp_ns=400)
    assert is_stale is True
    assert state.fresh_consec == 0


def test_debounce_backward_stamp_cannot_count_as_recovery() -> None:
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceConfig,
        ScanDebounceState,
        debounce_scan_stale,
    )

    config = ScanDebounceConfig(max_age_s=0.60, recover_age_s=0.45, recover_count=3)
    state = ScanDebounceState(stale_latched=True, fresh_consec=0)
    state, _ = debounce_scan_stale(state, 0.10, config, stamp_ns=200)
    state, _ = debounce_scan_stale(state, 0.10, config, stamp_ns=300)
    assert state.fresh_consec == 2
    # A delayed out-of-order sample older than the newest counted one holds.
    state, is_stale = debounce_scan_stale(state, 0.10, config, stamp_ns=250)
    assert is_stale is True
    assert state.fresh_consec == 2
    assert state.last_stamp_ns == 300
    # Only a strictly newer stamp advances; the third in-order sample recovers.
    state, still = debounce_scan_stale(state, 0.10, config, stamp_ns=300)
    assert still is True
    state, recovered = debounce_scan_stale(state, 0.10, config, stamp_ns=400)
    assert recovered is False
    assert state.stale_latched is False


def test_safety_gate_node_counts_distinct_scans_not_callbacks() -> None:
    import rclpy  # noqa: E402
    from robot_autonomous_yolo_slam_nav2.safety_gate_logic import (  # noqa: E402
        ScanDebounceState,
    )
    from robot_autonomous_yolo_slam_nav2.safety_gate_node import (  # noqa: E402
        SafetyGateNode,
    )

    rclpy.init(args=[])
    node = SafetyGateNode()
    try:
        node._scan_points = ((1.0, 0.0),)
        now_ns = node.get_clock().now().nanoseconds
        node._scan_stamp_ns = now_ns - 50_000_000
        node._debounce_state = ScanDebounceState(True, 0)

        assert node._fresh_scan_points(now_ns) is None
        assert node._debounce_state.fresh_consec == 1
        assert node._fresh_scan_points(now_ns + 5_000_000) is None
        assert node._debounce_state.fresh_consec == 1

        node._scan_stamp_ns = now_ns + 5_000_000
        assert node._fresh_scan_points(now_ns + 10_000_000) is None
        assert node._debounce_state.fresh_consec == 2
        node._scan_stamp_ns = now_ns + 10_000_000
        assert node._fresh_scan_points(now_ns + 15_000_000) == ((1.0, 0.0),)
        assert node._debounce_state.stale_latched is False
    finally:
        node.destroy_node()
        rclpy.shutdown()
