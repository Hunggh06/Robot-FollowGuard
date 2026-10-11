from pathlib import Path
import shutil
import subprocess

import pytest


def test_firmware_startup_and_reversal_driver(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the firmware host test")
    package = Path(__file__).parents[1]
    firmware = Path(__file__).resolve().parents[3] / "firmware/robot_lowlevel_v2_3.ino"
    source = firmware.read_text()
    assert '#include "motor_pid_control.h"' not in source
    start = source.index("struct WheelPIDConfig {")
    end = source.index("// ================= WIFI", start)
    (tmp_path / "motor_pid_control.h").write_text(
        "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n" + source[start:end]
    )
    binary = tmp_path / "motor_pid_test"
    subprocess.run(
        [compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-I", str(tmp_path),
         str(package / "test" / "firmware_motor_pid_driver.cpp"), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)
    assert result.stdout.count(" OK") == 5
