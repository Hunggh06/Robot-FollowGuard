"""Test Nav2 voi map co san (sau khi da luu map tu SLAM).

Chay: ros2 launch robot_autonomous_yolo_slam_nav2 navigation.launch.py map:=/duong/dan/map.yaml
Luu map tu SLAM: ros2 run nav2_map_server map_saver_cli -f ~/maps/phong1
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('robot_autonomous_yolo_slam_nav2')
    with open(os.path.join(pkg, 'urdf', 'xebamnguoi.urdf')) as f:
        robot_desc = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Duong dan file map .yaml'),
        Node(package='kaiaai_telemetry', executable='telem',
             name='kaiaai_telemetry_node',
             parameters=[{
                 'lidar.model': ['XIAOMI-LDS02RR'],
                 'laser_scan.lidar_model': 'XIAOMI-LDS02RR',
                 'laser_scan.frame_id': 'laser',
                 'lidar.pub_scan_size': 360,
                 'lidar.clockwise': False,
                 'lidar.range_min_meters': 0.15,
                 'lidar.range_max_meters': 6.0,
                 'telemetry.topic_name_sub': 'telemetry',
                 'laser_scan.topic_name_pub': 'scan',
                 'laser_scan.discard_broken_scans': True,
                 'laser_scan.mask_radius_meters': 0.17,
                 'laser_scan.min_quality': 500.0,
                 'use_host_time_stamp': True,
             }]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc, 'use_sim_time': False}]),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             parameters=[{'use_sim_time': False}]),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'laser']),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='motor_odom.py',
              parameters=[{'use_sim_time': False}]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='safety_gate_node.py',
             name='safety_gate', output='screen',
             parameters=[{
                 'input_topic': '/cmd_vel',
                 'output_topic': '/cmd_vel_safe',
                 'scan_topic': '/scan',
                 'base_frame': 'base_footprint',
                 'use_sim_time': False,
             }]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='cmd_vel_bridge.py',
              name='cmd_vel_bridge',
              parameters=[{'use_sim_time': False, 'input_topic': '/cmd_vel_safe'}]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='imu_remap.py',
             parameters=[{'use_sim_time': False}]),
        Node(package='robot_localization', executable='ekf_node',
             name='ekf_filter_node', output='screen',
             parameters=[os.path.join(pkg, 'config', 'ekf.yaml'),
                         {'use_sim_time': False}]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='odom_tf_bridge.py',
             parameters=[{'use_sim_time': False,
                          'odom_topic': '/odometry/filtered',
                          'relay_scan': False}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('nav2_bringup'),
                             'launch', 'bringup_launch.py')),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'params_file': os.path.join(pkg, 'config', 'nav2_params.yaml'),
                'use_sim_time': 'false',
            }.items()),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', os.path.join(pkg, 'rviz', 'slam.rviz')]),
    ])
