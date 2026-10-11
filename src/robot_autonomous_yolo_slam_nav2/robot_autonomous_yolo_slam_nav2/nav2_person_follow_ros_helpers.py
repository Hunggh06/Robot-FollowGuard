"""ROS message and TF adapters used by the person-follow runtime."""

from __future__ import annotations

import math
from typing import cast

from builtin_interfaces.msg import Time as MessageTime
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer
from tf2_geometry_msgs import do_transform_pose

from robot_autonomous_yolo_slam_nav2.msg import PersonTarget

from .safety_gate_logic import scan_to_base_points
from .map_recovery_logic import MapGrid


def occupancy_grid_snapshot(message: OccupancyGrid) -> MapGrid:
    """Parse a ROS occupancy grid into the immutable map geometry boundary."""
    orientation = message.info.origin.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    stamp = message.header.stamp
    return MapGrid(
        frame_id=str(message.header.frame_id),
        stamp_s=float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0,
        resolution_m=float(message.info.resolution),
        width=int(message.info.width),
        height=int(message.info.height),
        origin_x=float(message.info.origin.position.x),
        origin_y=float(message.info.origin.position.y),
        origin_yaw_rad=yaw,
        data=tuple(int(value) for value in message.data),
    )


def stamp_ns(message: LaserScan | PointCloud2) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def transform_scan(tf_buffer: Buffer, message: LaserScan, base_frame: str) -> tuple[tuple[float, float], ...] | None:
    try:
        transform = tf_buffer.lookup_transform(base_frame, message.header.frame_id, Time.from_msg(message.header.stamp))
    except Exception:  # noqa: BLE001
        return None
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return scan_to_base_points(
        float(message.angle_min), float(message.angle_increment), float(message.range_min),
        float(message.range_max), tuple(float(value) for value in message.ranges),
        (float(translation.x), float(translation.y)),
        (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
    )


def fresh_scan_points(points: tuple[tuple[float, float], ...] | None, scan_stamp_ns: int, now_ns: int, max_age_s: float) -> tuple[tuple[float, float], ...] | None:
    if points is None or scan_stamp_ns <= 0:
        return None
    age_ns = now_ns - scan_stamp_ns
    if age_ns < 0 or age_ns > max_age_s * 1_000_000_000:
        return None
    return points


def robot_map_pose(tf_buffer: Buffer, map_frame: str, base_frame: str) -> tuple[float, float, float] | None:
    try:
        transform = tf_buffer.lookup_transform(map_frame, base_frame, Time())
    except Exception:  # noqa: BLE001
        return None
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    yaw = math.atan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y), 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z))
    return float(translation.x), float(translation.y), yaw


def target_map_point(tf_buffer: Buffer, target: PersonTarget, base_frame: str, map_frame: str) -> tuple[float, float] | None:
    source = PoseStamped()
    source.header = target.header
    source.header.frame_id = base_frame
    source.pose.position.x = float(target.point_base_link.x)
    source.pose.position.y = float(target.point_base_link.y)
    source.pose.orientation.w = 1.0
    try:
        transform = tf_buffer.lookup_transform(
            map_frame,
            source.header.frame_id,
            Time.from_msg(source.header.stamp),
        )
        transformed = cast(PoseStamped, PoseStamped(pose=do_transform_pose(source.pose, transform)))
    except Exception:  # noqa: BLE001
        return None
    return float(transformed.pose.position.x), float(transformed.pose.position.y)


def map_goal_to_base(tf_buffer: Buffer, goal: tuple[float, float, float], map_frame: str, base_frame: str, stamp: MessageTime) -> tuple[float, float, float] | None:
    source = PoseStamped()
    source.header.frame_id = map_frame
    source.header.stamp = stamp
    source.pose.position.x, source.pose.position.y = goal[0], goal[1]
    source.pose.orientation.z = math.sin(goal[2] / 2.0)
    source.pose.orientation.w = math.cos(goal[2] / 2.0)
    try:
        transform = tf_buffer.lookup_transform(
            base_frame,
            source.header.frame_id,
            Time.from_msg(source.header.stamp),
        )
        transformed = cast(PoseStamped, PoseStamped(pose=do_transform_pose(source.pose, transform)))
    except Exception:  # noqa: BLE001
        return None
    rotation = transformed.pose.orientation
    yaw = math.atan2(2.0 * rotation.w * rotation.z, 1.0 - 2.0 * rotation.z * rotation.z)
    return float(transformed.pose.position.x), float(transformed.pose.position.y), yaw


def parse_obstacle_cloud(cloud: PointCloud2 | None, now_ns: int, max_age_s: float, base_frame: str, max_points: int) -> tuple[tuple[float, float, float], ...] | None:
    if cloud is None:
        return None
    age_s = (now_ns - stamp_ns(cloud)) / 1_000_000_000.0
    if age_s < 0.0 or age_s > max_age_s or cloud.header.frame_id != base_frame:
        return None
    points: list[tuple[float, float, float]] = []
    try:
        for index, point in enumerate(read_points(cloud, field_names=["x", "y", "z"], skip_nans=False)):
            if index >= max_points:
                return None
            values = (float(point[0]), float(point[1]), float(point[2]))
            if not all(math.isfinite(value) for value in values):
                return None
            points.append(values)
    except (IndexError, TypeError, ValueError):
        return None
    return tuple(points)
