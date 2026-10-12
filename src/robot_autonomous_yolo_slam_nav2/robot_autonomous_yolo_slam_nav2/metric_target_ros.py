"""ROS message conversion functions for the metric association runtime."""

from __future__ import annotations

import math
from collections.abc import Sequence

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from .metric_association import MetricPersonObservation, Point3D, RotationMatrix
from .metric_target_fields import PersonObservationFields, PersonTargetFields
from .metric_target_helpers import CameraInfoFields, LaserScanFields, quaternion_to_rotation, is_fresh, timestamps_within_skew
from robot_autonomous_yolo_slam_nav2.msg import PersonObservation, PersonTarget


def stamp_ns(header: Header) -> int:
    """Convert a ROS header stamp into nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def fresh_observation_messages(
    messages: Sequence[PersonObservation],
    target_stamp_ns: int,
    camera_info_stamp_ns: int,
    now_ns: int,
    max_age_s: float,
    max_sensor_skew_s: float,
) -> tuple[PersonObservation, ...]:
    return tuple(
        message for message in messages
        if is_fresh(stamp_ns(message.header), now_ns, max_age_s)
        and timestamps_within_skew(stamp_ns(message.header), target_stamp_ns, max_sensor_skew_s)
        and timestamps_within_skew(stamp_ns(message.header), camera_info_stamp_ns, max_sensor_skew_s)
    )


def target_fields(message: PersonTarget) -> PersonTargetFields[Header]:
    """Project a ROS target into the pure helper contract."""
    return PersonTargetFields(
        header=message.header,
        track_id=int(message.track_id),
        confidence=float(message.confidence),
        bbox_x=float(message.bbox_x),
        bbox_y=float(message.bbox_y),
        bbox_width=float(message.bbox_width),
        bbox_height=float(message.bbox_height),
        state=str(message.state),
        valid=bool(message.valid),
        point_base_link=(
            float(message.point_base_link.x),
            float(message.point_base_link.y),
            float(message.point_base_link.z),
        ),
        range_m=float(message.range_m),
        bearing_rad=float(message.bearing_rad),
    )


def observation_fields(message: PersonObservation) -> PersonObservationFields[Header]:
    """Project a ROS observation into the pure helper contract."""
    return PersonObservationFields(
        header=message.header,
        track_id=int(message.track_id),
        confidence=float(message.confidence),
        bbox_x=float(message.bbox_x),
        bbox_y=float(message.bbox_y),
        bbox_width=float(message.bbox_width),
        bbox_height=float(message.bbox_height),
        state=str(message.state),
        valid=bool(message.valid),
    )


def camera_info_fields(message: CameraInfo) -> CameraInfoFields:
    """Project a ROS CameraInfo into the pure parser contract."""
    return CameraInfoFields(
        width=int(message.width),
        height=int(message.height),
        k=tuple(float(value) for value in message.k),
        d=tuple(float(value) for value in message.d),
        distortion_model=str(message.distortion_model),
    )


def scan_fields(message: LaserScan) -> LaserScanFields:
    """Project a ROS LaserScan into the pure polar conversion contract."""
    return LaserScanFields(
        angle_min=float(message.angle_min),
        angle_increment=float(message.angle_increment),
        range_min=float(message.range_min),
        range_max=float(message.range_max),
        ranges=tuple(float(value) for value in message.ranges),
    )


def invalid_target_message(
    source: PersonTarget | None,
    base_frame: str,
    provenance: str = "no_geometry",
) -> PersonTarget:
    """Copy raw target identity fields while clearing metric validity."""
    target = PersonTarget()
    if source is not None:
        target.header.stamp.sec = source.header.stamp.sec
        target.header.stamp.nanosec = source.header.stamp.nanosec
        target.track_id = source.track_id
        target.confidence = source.confidence
        target.bbox_x = source.bbox_x
        target.bbox_y = source.bbox_y
        target.bbox_width = source.bbox_width
        target.bbox_height = source.bbox_height
        target.state = source.state
    target.header.frame_id = base_frame
    target.position_source = "none"
    target.position_confidence = 0.0
    target.provenance = provenance
    return target


def held_target_message(
    source: PersonTarget,
    track_id: int,
    point: Point3D,
    range_m: float,
    bearing_rad: float,
    stamp_sec: int,
    stamp_nanosec: int,
    position_confidence: float,
    provenance: str,
    base_frame: str,
) -> PersonTarget:
    """Retain the last valid point with valid=false and its original stamp."""
    target = invalid_target_message(source, base_frame, provenance)
    target.identity_confirmed = source.identity_confirmed
    target.footpoint_valid = source.footpoint_valid
    target.footpoint_x = source.footpoint_x
    target.footpoint_y = source.footpoint_y
    target.footpoint_confidence = source.footpoint_confidence
    target.track_id = track_id
    target.header.stamp.sec = stamp_sec
    target.header.stamp.nanosec = stamp_nanosec
    target.point_base_link.x = point[0]
    target.point_base_link.y = point[1]
    target.point_base_link.z = point[2]
    target.range_m = range_m
    target.bearing_rad = bearing_rad
    target.position_source = "held_invalid"
    target.position_confidence = position_confidence
    return target


def metric_target_message(
    source: PersonTarget,
    metric: MetricPersonObservation | None,
    base_frame: str,
    position_source: str = "lidar",
    position_confidence: float = 0.95,
    provenance: str = "lidar_bbox",
) -> PersonTarget:
    """Create a base-link target message from an optional metric point."""
    target = invalid_target_message(source, base_frame, provenance)
    if metric is None:
        return target
    target.valid = True
    target.header.stamp.sec = metric.source_stamp_ns // 1_000_000_000
    target.header.stamp.nanosec = metric.source_stamp_ns % 1_000_000_000
    target.identity_confirmed = source.identity_confirmed
    target.footpoint_valid = source.footpoint_valid
    target.footpoint_x = source.footpoint_x
    target.footpoint_y = source.footpoint_y
    target.footpoint_confidence = source.footpoint_confidence
    target.point_base_link.x = metric.point[0]
    target.point_base_link.y = metric.point[1]
    target.point_base_link.z = metric.point[2]
    target.range_m = metric.range_m
    target.bearing_rad = metric.bearing_rad
    target.position_source = position_source
    target.position_confidence = position_confidence
    target.provenance = provenance
    return target


def empty_obstacle_cloud(
    source_header: Header,
    base_frame: str,
    points: Sequence[Point3D],
) -> PointCloud2:
    """Create a fresh base-link PointCloud2 with explicit xyz float32 fields."""
    header = Header()
    header.stamp = source_header.stamp
    header.frame_id = base_frame
    fields = (
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    )
    return point_cloud2.create_cloud(header, fields, points)


def transform_values(
    transform: TransformStamped,
) -> tuple[RotationMatrix, Point3D] | None:
    """Extract finite rotation and translation values from a TF message."""
    rotation = quaternion_to_rotation(
        float(transform.transform.rotation.x),
        float(transform.transform.rotation.y),
        float(transform.transform.rotation.z),
        float(transform.transform.rotation.w),
    )
    translation = (
        float(transform.transform.translation.x),
        float(transform.transform.translation.y),
        float(transform.transform.translation.z),
    )
    if rotation is None or not all(math.isfinite(value) for value in translation):
        return None
    return rotation, translation
