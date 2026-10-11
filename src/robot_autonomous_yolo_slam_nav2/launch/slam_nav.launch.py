"""SLAM live + Nav2 cung luc: vua quet ban do vua dieu huong, khong can map co san.

Khac navigation.launch.py (dung AMCL + map_server voi file da luu), file nay
chay slam_toolbox de sinh /map + TF map->odom real-time; Nav2 chi chay phan
dieu huong (khong co localization), costmap doc /map tu slam_toolbox.

Full person-follow entrypoint (khuyen dung):
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=true
File nay dung doc lap (standalone) mac dinh KHONG start person-follow nodes
de tranh circular include / duplicate TF voi person_follow.launch.py.
Chay: ros2 launch robot_autonomous_yolo_slam_nav2 slam_nav.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('robot_autonomous_yolo_slam_nav2')
    start_person_follow = LaunchConfiguration('start_person_follow')
    with open(os.path.join(pkg, 'urdf', 'xebamnguoi.urdf')) as f:
        robot_desc = f.read()

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'),
                         'launch', 'online_async_launch.py')),
        launch_arguments={
            'slam_params_file': os.path.join(pkg, 'config', 'slam_toolbox.yaml'),
            'use_sim_time': 'false',
        }.items())

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'),
                         'launch', 'navigation_launch.py')),
        launch_arguments={
            'params_file': os.path.join(pkg, 'config', 'nav2_params_slam.yaml'),
            'use_sim_time': 'false',
            'autostart': 'true',
        }.items())

    # Costmap khong co trong lifecycle_nodes mac dinh cua navigation_launch.py
    # nen lifecycle manager khong kich hoat chung -> inactive -> khong co du lieu
    # -> planner khong lap ke hoach duoc. Them vao day.
    costmap_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_costmap', output='screen',
        parameters=[{'use_sim_time': False},
                    {'autostart': True},
                    {'node_names': ['local_costmap/local_costmap',
                                    'global_costmap/global_costmap']}])

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_person_follow',
            default_value='false',
            description='Start the AprilTag and metric person-follow nodes. '
                        'Keep false when included from person_follow.launch.py '
                        '(which is the full entrypoint) to avoid duplicates.',
        ),
        # Firmware moi day raw byte LDS02RR len /telemetry, khong publish
        # /scan truc tiep. Node nay bien thanh /scan cho odom_tf_bridge.
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
                  'use_host_time_stamp': True,
                   'telemetry.topic_name_sub': 'telemetry',
                  'laser_scan.topic_name_pub': 'scan',
                  'laser_scan.discard_broken_scans': True,
                  'laser_scan.mask_radius_meters': 0.17,
              }]),

        # /cmd_vel (Twist) -> /motor/target_rpm (Float32MultiArray [L, R]). Nav2 va
        # firmware dung hai kieu tin khac nhau, khong co node nao noi
        # giua chung thi /cmd_vel co publisher nhung 0 subscriber.
         Node(package='robot_autonomous_yolo_slam_nav2', executable='safety_gate_node.py',
              name='safety_gate', output='screen',
              parameters=[os.path.join(pkg, 'config', 'person_follow.yaml'),
                          {
                  'input_topic': '/cmd_vel',
                  'output_topic': '/cmd_vel_safe',
                  'scan_topic': '/scan',
                  'base_frame': 'base_footprint',
                  'corridor_half_width_m': 0.22,
                  'use_sim_time': False,
              }]),
         Node(package='robot_autonomous_yolo_slam_nav2', executable='cmd_vel_bridge.py',
              name='cmd_vel_bridge', output='log',
              parameters=[{'input_topic': '/cmd_vel_safe', 'use_sim_time': False}]),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc, 'use_sim_time': False}]),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             parameters=[{'use_sim_time': False}]),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'laser']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='phone_camera_tf',
             arguments=['-0.055', '0.063', '0.177',
                        '0.5', '-0.5', '0.5', '-0.5',
                        'base_link', 'phone_camera_back'],
             condition=IfCondition(start_person_follow)),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='apriltag_target_node.py',
             name='apriltag_target_node', output='screen',
             condition=IfCondition(start_person_follow),
             parameters=[os.path.join(pkg, 'config', 'person_follow.yaml')]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='metric_target_node.py',
             name='metric_target_node', output='screen',
             condition=IfCondition(start_person_follow),
             parameters=[os.path.join(pkg, 'config', 'person_follow.yaml')]),
        Node(package='robot_autonomous_yolo_slam_nav2', executable='motor_odom.py',
             name='motor_odom', output='log',
             parameters=[{'use_sim_time': False}]),
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
        slam,
        nav2,
        costmap_lifecycle,
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', os.path.join(pkg, 'rviz', 'slam.rviz')]),
    ])
