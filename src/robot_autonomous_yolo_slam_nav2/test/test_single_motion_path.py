"""Single-motion-path routing contract (TDD RED).

Normal motion path only:
controller_server -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel
-> safety_gate -> /cmd_vel_safe -> cmd_vel_bridge.

Custom runtimes (nav2_person_follow, zone_guard) must never publish
Twist directly to bare /cmd_vel; person_follow.yaml must not point a
custom source at bare /cmd_vel; bridge launch input stays /cmd_vel_safe.
"""

from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).parents[1]
NAV2_RUNTIME = PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "nav2_person_follow_runtime.py"
ZONE_RUNTIME = PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "zone_guard_runtime.py"
FOLLOW_CONFIG = PACKAGE_DIR / "config" / "person_follow.yaml"


def test_nav2_follow_spin_topic_default_is_not_bare_cmd_vel() -> None:
    # Given the custom Nav2 follow runtime source.
    source = NAV2_RUNTIME.read_text()

    # When the spin-topic default is inspected.
    # Then it must not default to the gated bare /cmd_vel topic.
    assert 'declare_parameter("search_spin_topic", "/cmd_vel")' not in source


def test_nav2_follow_runtime_creates_no_twist_publisher() -> None:
    # Given the custom Nav2 follow runtime source.
    source = NAV2_RUNTIME.read_text()

    # When publisher construction is inspected.
    # Then no direct Twist publisher may remain (Nav2 action only).
    assert not ("create_publisher" in source and "Twist" in source)


def test_zone_guard_runtime_publishes_no_bare_cmd_vel() -> None:
    # Given the custom zone-guard runtime source.
    source = ZONE_RUNTIME.read_text()

    # When raw topic strings are inspected.
    # Then bare /cmd_vel must never appear as a direct publish target.
    assert '"/cmd_vel"' not in source


def test_person_follow_yaml_sets_no_bare_cmd_vel_source() -> None:
    # Given the follower YAML config.
    config = yaml.safe_load(FOLLOW_CONFIG.read_text())
    parameters = config["nav2_person_follow"]["ros__parameters"]

    # When the custom spin source is read.
    # Then no custom Twist source may remain (Nav2 action only).
    assert "search_spin_topic" not in parameters


def _node_parameter_dict(node) -> dict:
    parameters: dict = {}
    for parameter_group in node.__dict__["_Node__parameters"]:
        if not isinstance(parameter_group, dict):
            continue
        for key, value in parameter_group.items():
            if isinstance(value, bool):
                parameters[key[0].text] = value
                continue
            if isinstance(value, (float, int)):
                parameters[key[0].text] = value
                continue
            substitution = value[0]
            if isinstance(substitution, list):
                substitution = substitution[0]
            if isinstance(substitution, (bool, float, int, str)):
                parsed_value = substitution
            else:
                parsed_value = yaml.safe_load(substitution.text)
            assert isinstance(parsed_value, (bool, float, int, str))
            parameters[key[0].text] = parsed_value
    return parameters


def test_bridge_launch_input_remains_cmd_vel_safe() -> None:
    # Given the bridge launches wiring the fail-closed tail.
    import importlib.util

    from launch import LaunchDescription
    from launch_ros.actions import Node

    for filename in ("slam.launch.py", "slam_nav.launch.py", "navigation.launch.py"):
        path = PACKAGE_DIR / "launch" / filename
        spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        description = module.generate_launch_description()
        assert isinstance(description, LaunchDescription)
        bridges = [
            entity
            for entity in description.entities
            if isinstance(entity, Node)
            and entity.__dict__["_Node__node_executable"] == "cmd_vel_bridge.py"
        ]
        assert len(bridges) == 1, filename
        assert _node_parameter_dict(bridges[0])["input_topic"] == "/cmd_vel_safe", filename
