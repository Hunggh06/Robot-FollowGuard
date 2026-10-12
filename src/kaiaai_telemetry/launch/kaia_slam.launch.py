# ============================================================
# KAIA SLAM — Humble, khong can kaiaai_bringup
# ============================================================
# Chuoi TF day du:
#
#   map --> odom --> base_footprint --> laser_frame
#     ^        ^         static          static
#     |        |
#   slam_    static_tf
#   toolbox   identity
#
# Node kaiaai_telemetry (ban port) CHI publish /scan — khong publish TF,
# khong publish /odom. Nen phai tu dat static transform cho phan odom.
#
# CHAY DU KHONG CO ODOM THAT: odom -> base_footprint la identity. Du de ve
# ban do dau tien. Khi odom cua ESP32 chinh xac roi, thay static bang o
# do cua node telem.
#
# Cach chay:
#   ros2 launch kaiaai_telemetry kaia_slam.launch.py
#   ros2 launch kaiaai_telemetry kaia_slam.launch.py rviz:=false
# ============================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # File nam trong share sau khi cai, dung duong dan share cua goi
    # (ament_index_cpp) chu khong hard-code — neu hard-code duong dan
    # workspace thi lan build lai se tro sai.
    share = get_package_share_directory("kaiaai_telemetry")

    slam_cfg = os.path.join(
        get_package_share_directory("slam_toolbox"), "config",
        "mapper_params_online_async.yaml")

    cfg_yaml = os.path.join(share, "config", "slam_kaia.yaml")

    return LaunchDescription([

        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Mo RViz hay khong"),

        # ---------------- NODE GIAI MA LIDAR ----------------
        # ESP32 day raw byte LDS02RR len /telemetry, node nay giai ma
        # thanh /scan 360 diem.
        Node(
            package="kaiaai_telemetry",
            executable="telem",
            name="kaiaai_telemetry_node",
            output="screen",
            parameters=[{
                "lidar.model": ["XIAOMI-LDS02RR"],
                "laser_scan.lidar_model": "XIAOMI-LDS02RR",
                "laser_scan.frame_id": "laser",
                "lidar.pub_scan_size": 360,
                "lidar.range_min_meters": 0.15,
                "lidar.range_max_meters": 6.0,
                "telemetry.topic_name_sub": "telemetry",
                "laser_scan.topic_name_pub": "scan",
            }],
        ),

        # odom phat lien tuc, khong phai static_transform_publisher: robot
        # dung yen thi cache TF khong co du lieu moi, /scan bi loai.
        Node(
            package="kaiaai_telemetry",
            executable="dummy_odom.py",
            name="dummy_odom",
            output="log",
        ),

        # base_footprint -> laser_frame: LiDAR cao 0,10 m tren san.
        # Do lai bang thuoc roi sua — sai 5 cm thi duong bang lech.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_laser",
            output="log",
            arguments=["0", "0", "0.10", "0", "0", "0",
                       "base_footprint", "laser_frame"],
        ),

        # ---------------- SLAM TOOLBOX ----------------
        # Hai file: slam_config_file la tham so cua slam_toolbox, cfg_file
        # la tham so cua ROS. Truyen ca hai moi dung tham so mac dinh.
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[
                slam_cfg,
                cfg_yaml,
                {"use_sim_time": False},
            ],
        ),

        # ---------------- RVIZ ----------------
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
