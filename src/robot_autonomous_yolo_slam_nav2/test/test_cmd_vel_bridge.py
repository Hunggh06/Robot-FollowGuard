import importlib.util
from pathlib import Path


BRIDGE_PATH = Path(__file__).parents[1] / "scripts" / "cmd_vel_bridge.py"
BRIDGE_SPEC = importlib.util.spec_from_file_location("cmd_vel_bridge_under_test", BRIDGE_PATH)
assert BRIDGE_SPEC is not None
assert BRIDGE_SPEC.loader is not None
BRIDGE_MODULE = importlib.util.module_from_spec(BRIDGE_SPEC)
BRIDGE_SPEC.loader.exec_module(BRIDGE_MODULE)


def test_bridge_targets_firmware_scalar_rpm_topics() -> None:
    source = BRIDGE_PATH.read_text()
    assert "/motor/target_rpm" in source
    assert "Float32MultiArray" in source
    assert BRIDGE_MODULE is not None


def test_bridge_input_is_launch_configurable_without_changing_default_contract() -> None:
    source = BRIDGE_PATH.read_text()

    assert "declare_parameter('input_topic', '/cmd_vel')" in source
    assert "get_parameter('input_topic')" in source


def test_rolling_floor_keeps_any_nonzero_command_at_25_rpm() -> None:
    assert BRIDGE_MODULE.CmdVelBridge._rolling_floor(30.0, 12.0) == 25.0
    assert BRIDGE_MODULE.CmdVelBridge._rolling_floor(-30.0, -12.0) == -25.0
    assert BRIDGE_MODULE.CmdVelBridge._rolling_floor(30.0, 0.0) == 0.0


def test_bridge_timeout_clears_stale_targets_to_hard_zero() -> None:
    # Fail-safe: after CMD_TIMEOUT with no fresh /cmd_vel, stale queued/ramped
    # commands must resolve to exact zero, never a decayed non-zero revival.
    assert hasattr(BRIDGE_MODULE, "resolve_bridge_timeout")
    out = BRIDGE_MODULE.resolve_bridge_timeout(
        target_l=40.0, target_r=40.0, rpm_l=50.0, rpm_r=50.0, expired=True
    )
    assert out == (0.0, 0.0, 0.0, 0.0)


def test_bridge_timeout_preserves_live_commands() -> None:
    assert hasattr(BRIDGE_MODULE, "resolve_bridge_timeout")
    out = BRIDGE_MODULE.resolve_bridge_timeout(
        target_l=40.0, target_r=40.0, rpm_l=10.0, rpm_r=10.0, expired=False
    )
    assert out == (40.0, 40.0, 10.0, 10.0)


def test_bridge_ramp_hard_zeroes_rpm_on_timeout() -> None:
    source = BRIDGE_PATH.read_text()
    # ramp() must consult the fail-safe resolver so timeout hard-zeroes
    # rpm instead of _approach-decaying stale queued commands.
    assert "resolve_bridge_timeout" in source


def test_nonfinite_twist_targets_resolve_to_zero() -> None:
    # Behavioral: NaN/Inf Twist must never command RPM (min/max clamping
    # lets NaN through to full RPM, so conversion must hard-zero first).
    assert hasattr(BRIDGE_MODULE, "twist_to_targets")
    import math as _math

    for v, w in [
        (_math.nan, 0.0),
        (0.5, _math.inf),
        (_math.inf, _math.nan),
        (_math.inf, _math.inf),
        (0.5, -_math.inf),
    ]:
        assert BRIDGE_MODULE.twist_to_targets(v, w) == (0.0, 0.0)


def test_finite_twist_targets_still_convert() -> None:
    assert hasattr(BRIDGE_MODULE, "twist_to_targets")
    target_l, target_r = BRIDGE_MODULE.twist_to_targets(0.2, 0.0)
    assert target_l > 0.0 and target_r > 0.0
    assert target_l == target_r


def test_cb_hard_zeroes_targets_for_nonfinite_twist() -> None:
    import math as _math

    import rclpy
    from geometry_msgs.msg import Twist

    if not rclpy.ok():
        rclpy.init()
    try:
        node = BRIDGE_MODULE.CmdVelBridge()
        try:
            for v, w in [
                (_math.nan, 0.0),
                (0.5, _math.inf),
                (_math.inf, _math.nan),
            ]:
                msg = Twist()
                msg.linear.x = v
                msg.angular.z = w
                node.cb(msg)
                assert (node.target_l, node.target_r) == (0.0, 0.0)
        finally:
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_timeout_ramp_hard_zeroes_and_fresh_command_recovers() -> None:
    import rclpy
    from geometry_msgs.msg import Twist

    if not rclpy.ok():
        rclpy.init()
    try:
        node = BRIDGE_MODULE.CmdVelBridge()
        try:
            published: list[tuple[float, float]] = []
            node.cmd.publish = lambda m: published.append(tuple(m.data))
            node.rpm_l, node.rpm_r = 50.0, 50.0
            node.target_l, node.target_r = 40.0, 40.0
            node.last = node.get_clock().now()
            node.last_cmd = node.get_clock().now() - rclpy.duration.Duration(seconds=2.0)
            node.ramp()
            assert published[-1] == (0.0, 0.0)
            assert (node.rpm_l, node.rpm_r) == (0.0, 0.0)
            assert (node.target_l, node.target_r) == (0.0, 0.0)
            # A fresh finite command resumes motion; stale memory cannot.
            msg = Twist()
            msg.linear.x = 0.2
            msg.angular.z = 0.0
            node.cb(msg)
            assert node.target_l > 0.0 and node.target_r > 0.0
            node.ramp()
            assert published[-1][0] > 0.0 and published[-1][1] > 0.0
        finally:
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
