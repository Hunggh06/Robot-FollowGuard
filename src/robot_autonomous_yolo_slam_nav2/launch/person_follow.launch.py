"""Launch YOLO person-follow perception and Nav2 goal client.

This launch intentionally does not start a second safety gate or motor bridge.
When ``use_slam:=false``, an external Nav2 stack must still route ``/cmd_vel``
through exactly one ``safety_gate_node.py`` and ``cmd_vel_bridge.py`` before
motor output is enabled.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_dir = get_package_share_directory("robot_autonomous_yolo_slam_nav2")
    config_path = os.path.join(package_dir, "config", "person_follow.yaml")
    calibration_default = os.path.join(
        package_dir, "calibration", "phone_camera_back.yaml"
    )
    slam_launch = os.path.join(package_dir, "launch", "slam_nav.launch.py")
    use_slam = LaunchConfiguration("use_slam")
    calibration_file = LaunchConfiguration("calibration_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_slam",
                default_value="true",
                description="Start live SLAM and Nav2 bringup.",
            ),
            DeclareLaunchArgument(
                "calibration_file",
                default_value=calibration_default,
                description="Measured ROS CameraInfo YAML.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={"start_person_follow": "false"}.items(),
                condition=IfCondition(use_slam),
            ),
            Node(
                package="robot_autonomous_yolo_slam_nav2",
                executable="camera_info_publisher.py",
                name="phone_camera_info",
                output="screen",
                condition=IfCondition(PythonExpression(["'", calibration_file, "' != ''"])),
                parameters=[
                    config_path,
                    {"calibration_file": calibration_file},
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="phone_camera_tf",
                arguments=[
                     # base_link (x fwd, y left, z up) -> optical frame
                     # (x right, y down, z fwd): q=(0.5, -0.5, 0.5, -0.5).
                     # Identity rotation here projects base-X onto image-x and
                    # base-Z onto depth, pushing everything in front of the
                    # robot off the image so metric association finds nothing.
                     "-0.055", "0.063", "0.177",
                     "0.5", "-0.5", "0.5", "-0.5",
                     "base_link", "phone_camera_back",
                ],
            ),
            Node(
                package="robot_autonomous_yolo_slam_nav2",
                executable="yolo_target_node.py",
                name="yolo_target_node",
                output="screen",
                parameters=[config_path],
            ),
            Node(
                package="robot_autonomous_yolo_slam_nav2",
                executable="metric_target_node.py",
                name="metric_target_node",
                output="screen",
                parameters=[config_path],
            ),
            Node(
                package="robot_autonomous_yolo_slam_nav2",
                executable="nav2_person_follow.py",
                name="nav2_person_follow",
                output="screen",
                parameters=[config_path],
            ),
            Node(
                package="robot_autonomous_yolo_slam_nav2",
                executable="zone_guard_node.py",
                name="zone_guard",
                output="screen",
                parameters=[config_path],
            ),
        ],
    )
