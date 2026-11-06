import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.clearance_logic import (  # noqa: E402
    ClearancePolicy,
    derive_clearance,
)


def test_master_clearance_derives_stop_and_slow_bands() -> None:
    policy = ClearancePolicy(master_m=0.6, origin_offset_m=0.2, slow_delta_m=0.3)

    assert policy.stop_m == pytest.approx(0.4)
    assert policy.slow_m == pytest.approx(0.7)


def test_negative_clearance_is_fail_closed_at_zero() -> None:
    assert derive_clearance(master_m=0.1, delta_m=0.0, origin_offset_m=0.2) == 0.0
