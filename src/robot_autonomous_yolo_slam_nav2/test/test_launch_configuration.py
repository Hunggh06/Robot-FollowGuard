import importlib.util
from pathlib import Path
from types import ModuleType

import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node


PACKAGE_DIR = Path(__file__).parents[1]


def _load_launch_module(filename: str) -> ModuleType:
    path = PACKAGE_DIR / "launch" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_nodes(description: LaunchDescription) -> list[Node]:
    return [entity for entity in description.entities if isinstance(entity, Node)]


def _find_node(
    nodes: list[Node],
    package: str,
    executable: str,
    context: str,
) -> Node:
    matches = [
        node
        for node in nodes
        if node.__dict__["_Node__package"] == package
        and node.__dict__["_Node__node_executable"] == executable
    ]
    assert len(matches) == 1, (
        f"{context}: expected one {package}/{executable}, found {len(matches)}"
    )
    return matches[0]


def _node_parameter_dict(node: Node) -> dict[str, bool | float | int | str]:
    parameters: dict[str, bool | float | int | str] = {}
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


def _find_static_lidar_transform(nodes: list[Node], context: str) -> Node:
    matches = [
        node
        for node in nodes
        if node.__dict__["_Node__package"] == "tf2_ros"
        and node.__dict__["_Node__node_executable"] == "static_transform_publisher"
        and node.__dict__.get("_Node__arguments", [])[-2:] == [
            "lidar_link",
            "laser",
        ]
    ]
    assert len(matches) == 1, (
        f"{context}: expected one static lidar transform, found {len(matches)}"
    )
    return matches[0]


def test_ekf_uses_only_yaw_rate_from_remapped_imu() -> None:
    config_path = PACKAGE_DIR / "config" / "ekf.yaml"
    config = yaml.safe_load(config_path.read_text())
    parameters = config["ekf_filter_node"]["ros__parameters"]

    assert parameters["imu0"] == "/imu/yaw"
    assert parameters["imu0_config"] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert parameters["publish_tf"] is False
    assert parameters["odom0"] == "/wheel/odom"


def test_both_live_slam_launches_start_remap_and_ekf() -> None:
    for filename in ("slam.launch.py", "slam_nav.launch.py"):
        module = _load_launch_module(filename)
        nodes = _launch_nodes(module.generate_launch_description())

        remap = _find_node(nodes, "robot_autonomous_yolo_slam_nav2", "imu_remap.py", filename)
        remap_parameters = _node_parameter_dict(remap)
        assert remap_parameters["use_sim_time"] is False, filename
        _find_node(nodes, "robot_localization", "ekf_node", filename)


def test_each_bridge_launch_has_one_explicit_gate_and_safe_bridge_input() -> None:
    for filename in ("slam.launch.py", "slam_nav.launch.py", "navigation.launch.py"):
        module = _load_launch_module(filename)
        nodes = _launch_nodes(module.generate_launch_description())
        gate = _find_node(
            nodes,
            "robot_autonomous_yolo_slam_nav2",
            "safety_gate_node.py",
            filename,
        )
        bridge = _find_node(
            nodes,
            "robot_autonomous_yolo_slam_nav2",
            "cmd_vel_bridge.py",
            filename,
        )
        assert _node_parameter_dict(gate)["output_topic"] == "/cmd_vel_safe"
        assert _node_parameter_dict(bridge)["input_topic"] == "/cmd_vel_safe"
        assert sum(
            node.__dict__["_Node__node_executable"] == "safety_gate_node.py"
            for node in nodes
        ) == 1


def test_both_live_slam_launches_keep_filtered_host_timestamped_scan() -> None:
    for filename in ("slam.launch.py", "slam_nav.launch.py"):
        module = _load_launch_module(filename)
        nodes = _launch_nodes(module.generate_launch_description())
        telemetry = _find_node(nodes, "kaiaai_telemetry", "telem", filename)
        parameters = _node_parameter_dict(telemetry)

        assert parameters["laser_scan.topic_name_pub"] == "scan", filename
        assert parameters["laser_scan.frame_id"] == "laser", filename
        assert parameters["lidar.clockwise"] is False, filename
        assert parameters["use_host_time_stamp"] is True, filename
        assert parameters["laser_scan.discard_broken_scans"] is True, filename
        assert parameters["laser_scan.mask_radius_meters"] == 0.17, filename
        assert parameters["lidar.range_min_meters"] == 0.15, filename
        assert parameters["lidar.range_max_meters"] == 6.0, filename

        transform = _find_static_lidar_transform(nodes, filename)
        assert transform.__dict__["_Node__arguments"][-2:] == [
            "lidar_link",
            "laser",
        ], filename


def test_live_slam_launches_use_matching_laser_range_configuration() -> None:
    config_path = PACKAGE_DIR / "config" / "slam_toolbox.yaml"
    config = yaml.safe_load(config_path.read_text())
    parameters = config["slam_toolbox"]["ros__parameters"]

    assert parameters["scan_topic"] == "/scan"
    assert parameters["base_frame"] == "base_footprint"
    assert parameters["odom_frame"] == "odom"
    assert parameters["map_frame"] == "map"
    assert parameters["min_laser_range"] == 0.15
    assert parameters["max_laser_range"] == 6.0


def test_slam_nav_standalone_does_not_auto_start_person_follow() -> None:
    module = _load_launch_module("slam_nav.launch.py")
    description = module.generate_launch_description()
    launch_source = (PACKAGE_DIR / "launch" / "slam_nav.launch.py").read_text()

    assert "default_value='false'" in launch_source
    assert "person_follow.launch.py" in launch_source
    assert "use_slam" in launch_source
    assert len([entity for entity in description.entities if isinstance(entity, Node)]) >= 1


def test_slam_nav_bridge_reads_filtered_odometry_without_scan_relay() -> None:
    module = _load_launch_module("slam_nav.launch.py")
    nodes = _launch_nodes(module.generate_launch_description())
    bridge = _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "odom_tf_bridge.py",
        "slam_nav.launch.py",
    )
    parameters = _node_parameter_dict(bridge)

    assert parameters["odom_topic"] == "/odometry/filtered"
    assert parameters["relay_scan"] is False


def test_saved_map_navigation_keeps_imu_remap() -> None:
    module = _load_launch_module("navigation.launch.py")
    nodes = _launch_nodes(module.generate_launch_description())

    remap = _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "imu_remap.py",
        "navigation.launch.py",
    )
    assert _node_parameter_dict(remap)["use_sim_time"] is False


def test_person_follow_launch_wires_nodes_and_conditional_slam() -> None:
    # Given the user-facing person-follow launch description.
    module = _load_launch_module("person_follow.launch.py")

    # When the launch description is constructed without starting any nodes.
    description = module.generate_launch_description()
    nodes = _launch_nodes(description)

    # Then YOLO owns conditional tag confirmation and feeds metric and Nav2 nodes.
    for executable in (
        "yolo_target_node.py",
        "metric_target_node.py",
        "nav2_person_follow.py",
        "zone_guard_node.py",
    ):
        node = _find_node(nodes, "robot_autonomous_yolo_slam_nav2", executable, "person_follow.launch.py")
        assert len(node.__dict__["_Node__parameters"]) >= 1
    assert (PACKAGE_DIR / "launch" / "person_follow.launch.py").read_text().count(
        "parameters=[config_path]",
    ) == 4

    # And SLAM is included behind the declared use_slam condition.
    includes = [entity for entity in description.entities if isinstance(entity, IncludeLaunchDescription)]
    assert len(includes) == 1
    assert includes[0].condition.__class__.__name__ == "IfCondition"
    launch_source = (PACKAGE_DIR / "launch" / "person_follow.launch.py").read_text()
    assert 'DeclareLaunchArgument(\n                "use_slam"' in launch_source
    assert "IfCondition(use_slam)" in launch_source
    assert '"start_person_follow": "false"' in launch_source


def test_person_follow_launch_gates_calibrated_camera_info_publisher() -> None:
    module = _load_launch_module("person_follow.launch.py")
    description = module.generate_launch_description()
    nodes = _launch_nodes(description)
    publisher = _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "camera_info_publisher.py",
        "person_follow.launch.py",
    )

    assert publisher.condition.__class__.__name__ == "IfCondition"
    launch_source = (PACKAGE_DIR / "launch" / "person_follow.launch.py").read_text()
    assert 'DeclareLaunchArgument(\n                "calibration_file"' in launch_source
    assert '"phone_camera_back.yaml"' in launch_source
    assert "PythonExpression" in launch_source
    assert "calibration_file, \"' != ''\"" in launch_source


def test_person_follow_launch_exposes_apriltag_contract_without_reid_or_home() -> None:
    module = _load_launch_module("person_follow.launch.py")
    description = module.generate_launch_description()
    nodes = _launch_nodes(description)
    yolo = _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "yolo_target_node.py",
        "person_follow.launch.py",
    )
    assert len(yolo.__dict__["_Node__parameters"]) >= 1
    config = yaml.safe_load((PACKAGE_DIR / "config" / "person_follow.yaml").read_text())
    parameters = config["yolo_target_node"]["ros__parameters"]

    assert parameters["image_topic"] == "/phone/camera/back/image"
    assert parameters["camera_frame"] == "phone_camera_back"
    assert parameters["tag_family"] == "tag36h11"
    assert parameters["target_tag_id"] == 0
    assert parameters["detections_topic"] == "/apriltag/detections"

    launch_source = (PACKAGE_DIR / "launch" / "person_follow.launch.py").read_text()
    config_source = (PACKAGE_DIR / "config" / "person_follow.yaml").read_text()
    assert "reid_target_node" not in launch_source
    assert "reid_target_node" not in config_source
    assert "yolo_target_node" in launch_source
    assert "yolo_target_node:" in config_source
    assert "target_topic: /yolo_target_node/target" in config_source
    assert (
        '"start_person_follow": "false"' in launch_source
        or "'start_person_follow': 'false'" in launch_source
    )
    assert 'camera_frame: phone_camera_back' in config_source
    node_source = (PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2" / "yolo_target_node.py").read_text()
    assert 'get_parameter("observations_topic")' in node_source
    assert 'get_parameter("target_topic")' in node_source
    assert "GUIDE_HOME" not in launch_source
    assert "return_home" not in launch_source


def test_zone_guard_launch_and_zone_params() -> None:
    module = _load_launch_module("person_follow.launch.py")
    nodes = _launch_nodes(module.generate_launch_description())
    _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "zone_guard_node.py",
        "person_follow.launch.py",
    )
    config = yaml.safe_load((PACKAGE_DIR / "config" / "person_follow.yaml").read_text())
    parameters = config["zone_guard"]["ros__parameters"]

    assert parameters["clicked_point_topic"] == "/clicked_point"
    assert parameters["confirm_topic"] == "/phone/speech_text"
    assert parameters["say_topic"] == "/phone/speaker/say"
    assert parameters["guide_active_topic"] == "/person_follow/guide_active"
    assert parameters["zone_click_points"] == 4
    assert parameters["leaving_verify_s"] == 2.0
    assert parameters["ask_delay_s"] == 5.0
    assert parameters["repeat_s"] == 15.0
    assert parameters["guide_max_vel_x"] == 0.15
    assert parameters["confirm_ignore_after_say_s"] == 3.0
    assert parameters["search_max_spin_revolutions"] == 2.0
    assert config["nav2_person_follow"]["ros__parameters"]["guide_active_topic"] == "/person_follow/guide_active"


def test_apriltag_ros_output_is_kept_separate_from_target_input() -> None:
    module = _load_launch_module("person_follow.launch.py")
    nodes = _launch_nodes(module.generate_launch_description())
    executables = [node.__dict__["_Node__node_executable"] for node in nodes]
    assert "apriltag_node" not in executables
    assert "apriltag_debug_image.py" not in executables
    assert "apriltag_target_node.py" not in executables
    assert executables.count("yolo_target_node.py") == 1


def test_person_follow_launch_publishes_debug_image_for_rqt() -> None:
    module = _load_launch_module("person_follow.launch.py")
    nodes = _launch_nodes(module.generate_launch_description())
    _find_node(
        nodes,
        "robot_autonomous_yolo_slam_nav2",
        "yolo_target_node.py",
        "person_follow.launch.py",
    )
    config = yaml.safe_load((PACKAGE_DIR / "config" / "person_follow.yaml").read_text())
    parameters = config["yolo_target_node"]["ros__parameters"]

    assert parameters["image_topic"] == "/phone/camera/back/image"
    assert parameters["annotated_topic"] == "/person_follow/annotated"
    assert parameters["detections_topic"] == "/apriltag/detections"
    assert parameters["tag_family"] == "tag36h11"
    assert parameters["target_tag_id"] == 0


def test_person_follow_detector_preserves_small_tags_in_camera_frames() -> None:
    config = yaml.safe_load((PACKAGE_DIR / "config" / "person_follow.yaml").read_text())
    parameters = config["yolo_target_node"]["ros__parameters"]
    assert parameters["tag_roi_upscale"] == 4.0
    assert parameters["lost_timeout_s"] == 2.0
    assert parameters["ambiguity_iou"] == 0.3


def test_follow_path_contains_no_home_behavior() -> None:
    forbidden = ("GUIDE_HOME", "return_home", "fixed_home_pose", "home_goal")
    paths = (
        *(PACKAGE_DIR / "launch").glob("*.py"),
        *(PACKAGE_DIR / "config").glob("*.yaml"),
        *(PACKAGE_DIR / "robot_autonomous_yolo_slam_nav2").glob("*.py"),
    )
    source = "\n".join(path.read_text() for path in paths)
    assert not any(term.casefold() in source.casefold() for term in forbidden)


def _find_phone_camera_transform(nodes: list[Node], context: str) -> Node:
    matches = [
        node
        for node in nodes
        if node.__dict__["_Node__package"] == "tf2_ros"
        and node.__dict__["_Node__node_executable"] == "static_transform_publisher"
        and node.__dict__.get("_Node__arguments", [])[-2:] == [
            "base_link",
            "phone_camera_back",
        ]
    ]
    assert len(matches) == 1, (
        f"{context}: expected one static phone camera transform, found {len(matches)}"
    )
    return matches[0]


def _quaternion_to_columns(x: float, y: float, z: float, w: float) -> tuple:
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rows = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    return tuple(tuple(rows[r][c] for r in range(3)) for c in range(3))


def test_motor_launches_have_single_gated_motion_path() -> None:
    # Given the motor-owning launch descriptions.
    for filename in ("slam.launch.py", "slam_nav.launch.py", "navigation.launch.py"):
        module = _load_launch_module(filename)

        # When the launch description is constructed without starting any nodes.
        nodes = _launch_nodes(module.generate_launch_description())

        # Then exactly one gate maps /cmd_vel to /cmd_vel_safe.
        gate = _find_node(
            nodes,
            "robot_autonomous_yolo_slam_nav2",
            "safety_gate_node.py",
            filename,
        )
        gate_parameters = _node_parameter_dict(gate)
        assert gate_parameters["input_topic"] == "/cmd_vel", filename
        assert gate_parameters["output_topic"] == "/cmd_vel_safe", filename
        assert sum(
            node.__dict__["_Node__node_executable"] == "safety_gate_node.py"
            for node in nodes
        ) == 1, filename

        # And exactly one bridge consumes /cmd_vel_safe.
        bridge = _find_node(
            nodes,
            "robot_autonomous_yolo_slam_nav2",
            "cmd_vel_bridge.py",
            filename,
        )
        assert _node_parameter_dict(bridge)["input_topic"] == "/cmd_vel_safe", filename
        assert sum(
            node.__dict__["_Node__node_executable"] == "cmd_vel_bridge.py"
            for node in nodes
        ) == 1, filename


def test_person_follow_launch_starts_neither_gate_nor_bridge() -> None:
    # Given the user-facing person-follow launch description.
    module = _load_launch_module("person_follow.launch.py")

    # When the launch description is constructed without starting any nodes.
    executables = [
        node.__dict__["_Node__node_executable"]
        for node in _launch_nodes(module.generate_launch_description())
    ]

    # Then it owns no motion-path endpoint.
    assert executables.count("safety_gate_node.py") == 0
    assert executables.count("cmd_vel_bridge.py") == 0


def test_bench_teleop_and_rpm_scripts_absent_from_all_launches() -> None:
    # Given every launch description shipped by this package.
    filenames = sorted(path.name for path in (PACKAGE_DIR / "launch").glob("*.py"))
    assert len(filenames) >= 1

    # When each description is constructed without starting any nodes.
    # Then bench-only scripts are never launched.
    for filename in filenames:
        executables = [
            node.__dict__["_Node__node_executable"]
            for node in _launch_nodes(_load_launch_module(filename).generate_launch_description())
        ]
        assert "awsd_teleop.py" not in executables, filename
        assert "drive_circle.py" not in executables, filename


def test_phone_camera_optical_axis_faces_robot_forward() -> None:
    for filename in ("person_follow.launch.py", "slam_nav.launch.py"):
        module = _load_launch_module(filename)
        nodes = _launch_nodes(module.generate_launch_description())
        transform = _find_phone_camera_transform(nodes, filename)
        arguments = transform.__dict__.get("_Node__arguments", [])
        assert arguments[:3] == ["-0.055", "0.063", "0.177"], filename
        qx, qy, qz, qw = (float(value) for value in arguments[3:7])
        cam_x, cam_y, cam_z = _quaternion_to_columns(qx, qy, qz, qw)
        for actual, expected in zip(cam_z, (1.0, 0.0, 0.0)):
            assert actual == expected or abs(actual - expected) < 1e-9, filename
        for actual, expected in zip(cam_x, (0.0, -1.0, 0.0)):
            assert actual == expected or abs(actual - expected) < 1e-9, filename
