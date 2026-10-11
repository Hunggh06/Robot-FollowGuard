from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).parents[1]


def test_spin_search_needs_no_map_parameters() -> None:
    person_follow = yaml.safe_load(
        (PACKAGE_DIR / "config" / "person_follow.yaml").read_text()
    )
    nav2_parameters = person_follow["nav2_person_follow"]["ros__parameters"]
    runtime_source = (
        PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "nav2_person_follow_runtime.py"
    ).read_text()

    assert "map_topic" not in nav2_parameters
    assert "recovery_max_map_age_s" not in nav2_parameters
    assert "RecoveryCoordinator" not in runtime_source
    assert "MapGrid" not in runtime_source


def test_spin_search_params_match_defaults() -> None:
    person_follow = yaml.safe_load(
        (PACKAGE_DIR / "config" / "person_follow.yaml").read_text()
    )
    nav2_parameters = person_follow["nav2_person_follow"]["ros__parameters"]
    runtime_source = (
        PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "nav2_person_follow_runtime.py"
    ).read_text()

    assert nav2_parameters["recovery_status_topic"] == "/person_follow/recovery_status"
    assert nav2_parameters["spin_action_name"] == "/spin"
    assert "search_spin_topic" not in nav2_parameters
    assert nav2_parameters["search_spin_default_rate_rad_s"] == 0.7
    assert nav2_parameters["search_spin_min_rate_rad_s"] == 0.51
    assert nav2_parameters["search_spin_max_rate_rad_s"] == 1.4
    assert nav2_parameters["search_spin_deadband_rad"] == 0.1
    assert nav2_parameters["search_max_spin_revolutions"] == 2.0
    assert nav2_parameters["search_timeout_s"] == 0.0
    assert nav2_parameters["search_drive_duration_s"] == 2.0
    assert nav2_parameters["search_drive_speed_m_s"] == 0.15
    assert "RecoveryCoordinator" not in runtime_source
    assert "SEARCH_NAV" not in runtime_source
    assert 'declare_parameter("search_spin_topic"' not in runtime_source
    assert 'declare_parameter("spin_action_name", "/spin")' in runtime_source
    assert 'declare_parameter("search_max_spin_revolutions"' in runtime_source
    assert 'declare_parameter("recovery_status_topic"' in runtime_source
    assert "Twist" not in runtime_source
    assert "_spin_pub" not in runtime_source


def test_spin_action_client_uses_configured_name() -> None:
    person_follow = yaml.safe_load(
        (PACKAGE_DIR / "config" / "person_follow.yaml").read_text()
    )
    nav2_parameters = person_follow["nav2_person_follow"]["ros__parameters"]
    runtime_source = (
        PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "nav2_person_follow_runtime.py"
    ).read_text()
    spin_mixin_source = (
        PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "nav2_person_follow_spin_actions.py"
    ).read_text()

    assert nav2_parameters["spin_action_name"] == "/spin"
    assert 'get_parameter("spin_action_name")' in runtime_source
    assert "Nav2PersonFollowSpinActions" in runtime_source
    assert "def _send_spin" not in runtime_source
    assert "def _cancel_spin" not in runtime_source
    assert "def _send_spin" in spin_mixin_source
    assert "def _spin_goal_response_callback" in spin_mixin_source
    assert "def _spin_result_callback" in spin_mixin_source
    assert "def _spin_cancel_response_callback" in spin_mixin_source
    assert "def _cancel_spin" in spin_mixin_source


def test_person_follow_perception_mode_does_not_duplicate_motion_safety_nodes() -> None:
    launch_source = (PACKAGE_DIR / "launch" / "person_follow.launch.py").read_text()
    readme_source = (PACKAGE_DIR / "README.md").read_text()

    assert 'executable="safety_gate_node.py"' not in launch_source
    assert 'executable="cmd_vel_bridge.py"' not in launch_source
    assert "exactly one" in launch_source
    assert "safety_gate_node.py" in readme_source
    assert "cmd_vel_safe" in readme_source
