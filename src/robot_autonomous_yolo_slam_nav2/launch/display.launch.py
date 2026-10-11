"""Hien thi URDF xebamnguoi tren RViz (khong can ESP32)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('robot_autonomous_yolo_slam_nav2')
    with open(os.path.join(pkg, 'urdf', 'xebamnguoi.urdf')) as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc, 'use_sim_time': False}]),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             parameters=[{'use_sim_time': False}]),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', os.path.join(pkg, 'rviz', 'display.rviz')]),
    ])
