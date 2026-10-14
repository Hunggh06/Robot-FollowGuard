from pathlib import Path
from typing import Final


WORKSPACE_ROOT: Final = Path(__file__).resolve().parents[3]
ACTIVE_FIRMWARE_FILE: Final = (
    WORKSPACE_ROOT / "src/robot_autonomous_yolo_slam_nav2/firmware_esp32/robot.ino"
)


def test_active_firmware_has_per_wheel_pid_and_command_timeout_guards() -> None:
    source = ACTIVE_FIRMWARE_FILE.read_text(encoding="utf-8")

    assert "#define CMD_TIMEOUT_MS    1000UL" in source
    assert "if (now - lastCmdMs > CMD_TIMEOUT_MS" in source
    assert "stopMotors();" in source
    assert "pwmLeft  = calculatePIDLeft(dt);" in source
    assert "pwmRight = calculatePIDRight(dt);" in source
    assert "float error = targetRPMLeft - rpmLeftFiltered;" in source
    assert "float error = targetRPMRight - rpmRightFiltered;" in source
