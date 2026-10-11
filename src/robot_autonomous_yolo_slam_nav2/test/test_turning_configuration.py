from pathlib import Path

import yaml


PACKAGE = Path(__file__).parents[1]


def test_live_slam_keeps_the_approved_turning_and_forward_limits() -> None:
    current = yaml.safe_load((PACKAGE / "config/nav2_params_slam.yaml").read_text())
    follow = current["controller_server"]["ros__parameters"]["FollowPath"]

    assert follow["max_vel_theta"] == 1.0
    assert follow["acc_lim_theta"] == 1.5
    assert follow["decel_lim_theta"] == -1.5
    assert follow["max_vel_x"] == 0.25
    assert follow["min_vel_x"] == -0.10
    assert follow["acc_lim_x"] == 1.5
    assert follow["decel_lim_x"] == -1.5
    smoother = current["velocity_smoother"]["ros__parameters"]
    assert smoother["max_velocity"][2] >= follow["max_vel_theta"]
    assert smoother["min_velocity"][2] <= -follow["max_vel_theta"]
    assert smoother["max_accel"][2] >= follow["acc_lim_theta"]
    assert smoother["max_decel"][2] <= follow["decel_lim_theta"]
