from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).parents[1]


def test_cmd_vel_bridge_uses_safe_incremental_speed_limits() -> None:
    path = PACKAGE_DIR / "scripts" / "cmd_vel_bridge.py"
    source = path.read_text()

    assert "MAX_RPM = 126.0" in source
    assert "ACCEL_LIM = 60.0" in source


def test_slam_nav_disables_reverse_motion() -> None:
    config_path = PACKAGE_DIR / "config" / "nav2_params_slam.yaml"
    config = yaml.safe_load(config_path.read_text())
    follow_path = config["controller_server"]["ros__parameters"]["FollowPath"]

    assert follow_path["min_vel_x"] == -0.10
    assert follow_path["max_vel_x"] == 0.25
    assert follow_path["max_speed_xy"] == 0.25
    assert follow_path["max_vel_theta"] == 1.0
    assert follow_path["acc_lim_theta"] == 1.5
    assert follow_path["decel_lim_theta"] == -1.5
    assert follow_path["min_speed_theta"] == 0.05
    assert follow_path["vtheta_samples"] == 30
    assert follow_path["sim_time"] == 2.0
    assert follow_path["RotateToGoal.slowing_factor"] == 3.0
    assert follow_path["PathAlign.scale"] == 22.0
    assert follow_path["GoalAlign.scale"] == 6.0

    smoother = config["velocity_smoother"]["ros__parameters"]
    assert smoother["scale_velocities"] is False

    assert smoother["max_velocity"][0] == 0.25
    assert smoother["min_velocity"][0] == -0.10
    assert smoother["max_accel"][2] == 3.2
    assert smoother["max_decel"][2] == -3.2
    assert smoother["max_velocity"][2] == 1.0
    assert smoother["min_velocity"][2] == -1.0


def test_saved_map_navigation_disables_reverse_motion() -> None:
    config = yaml.safe_load((PACKAGE_DIR / "config" / "nav2_params.yaml").read_text())
    assert config["controller_server"]["ros__parameters"]["FollowPath"]["min_vel_x"] == -0.10
    assert config["velocity_smoother"]["ros__parameters"]["min_velocity"][0] == -0.10


def test_navigation_turn_limits_reach_first_look_heading_without_early_success() -> None:
    for filename in ("nav2_params_slam.yaml", "nav2_params.yaml"):
        config = yaml.safe_load((PACKAGE_DIR / "config" / filename).read_text())
        controller = config["controller_server"]["ros__parameters"]
        follow = controller["FollowPath"]
        smoother = config["velocity_smoother"]["ros__parameters"]
        assert follow["max_vel_theta"] == 1.0
        assert smoother["max_velocity"][2] >= follow["max_vel_theta"]
        assert smoother["min_velocity"][2] <= -follow["max_vel_theta"]
        tolerance = controller["general_goal_checker"]["yaw_goal_tolerance"]
        assert tolerance == 0.10
        assert abs(0.588 - 0.1) > tolerance
