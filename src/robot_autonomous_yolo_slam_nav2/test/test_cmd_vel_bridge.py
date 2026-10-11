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
