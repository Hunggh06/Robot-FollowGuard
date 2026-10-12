"""One publish cycle for the metric target runtime (ROS adapter).

Cache contract: calibration, TF, skew, staleness, and missing-geometry
failures RETAIN the last valid point for the same confirmed track as
held_invalid (valid=false, original stamp, decayed confidence). Only an
incompatible frame, absent vision identity (no fresh obs / no selection),
or a confirmed track change/reset/timeout clears the cache. The held
path never produces a fresh goal, and the two publish helpers call in
one direction only (held -> invalid), so they cannot recurse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from robot_autonomous_yolo_slam_nav2.msg import PersonTarget

from .metric_target_branches import (
    LidarInputs,
    decide_branch,
    usable_scan_stamp,
)
from .metric_target_helpers import (
    derive_target_selection,
    is_fresh,
    laser_scan_to_lidar_points,
    parse_camera_info,
    timestamps_within_skew,
)
from .metric_target_ros import (
    camera_info_fields,
    empty_obstacle_cloud,
    fresh_observation_messages,
    held_target_message,
    invalid_target_message,
    metric_target_message,
    observation_fields,
    scan_fields,
    stamp_ns,
    target_fields,
)

if TYPE_CHECKING:
    from .metric_target_runtime import MetricTargetRuntime


def publish_invalid_and_empty(
    host: MetricTargetRuntime,
    target: PersonTarget | None,
    source_header: Header,
    reason: str,
) -> None:
    host.get_logger().warning(
        f"metric invalid: {reason} track="
        f"{target.track_id if target is not None else 'none'}",
        throttle_duration_sec=2.0,
    )
    host._target_pub.publish(
        invalid_target_message(target, host._base_frame(), reason))
    host._obstacles_pub.publish(
        empty_obstacle_cloud(source_header, host._base_frame(), ()),
    )


def publish_held_or_invalid(
    host: MetricTargetRuntime,
    target: PersonTarget,
    now_ns: int,
) -> None:
    held = host._cache.held(int(target.track_id), now_ns)
    if held is None:
        publish_invalid_and_empty(host, target, target.header, "no_geometry")
        return
    age_s = (now_ns - held.stamp_ns) / 1_000_000_000.0
    host.get_logger().warning(
        f"metric held_invalid: track={held.track_id} age={age_s:.1f}s",
        throttle_duration_sec=2.0,
    )
    host._target_pub.publish(held_target_message(
        target,
        held.track_id,
        held.point,
        held.range_m,
        held.bearing_rad,
        held.stamp_ns // 1_000_000_000,
        held.stamp_ns % 1_000_000_000,
        held.confidence,
        f"held track={held.track_id} age={age_s:.1f}s",
        host._base_frame(),
    ))
    host._obstacles_pub.publish(
        empty_obstacle_cloud(target.header, host._base_frame(), ()),
    )


def run_cycle(host: MetricTargetRuntime) -> None:
    """Run one association cycle and publish target plus person obstacles."""
    target = host._target
    camera_info = host._camera_info
    if target is None:
        publish_invalid_and_empty(host, None, Header(), "missing_input")
        return
    if not target.valid or not target.identity_confirmed:
        host._cache.reset()
        publish_invalid_and_empty(host, target, target.header, "visual_target_unconfirmed")
        return
    now_ns = host.get_clock().now().nanoseconds
    if camera_info is None:
        publish_held_or_invalid(host, target, now_ns)
        return
    max_age_s = float(str(host.get_parameter("max_data_age_s").value))
    max_sensor_skew_s = float(str(host.get_parameter("max_sensor_skew_s").value))
    max_scan_age_s = float(str(host.get_parameter("max_scan_age_s").value))
    max_scan_skew_s = float(str(host.get_parameter("max_scan_skew_s").value))
    if not all(
        is_fresh(stamp_ns(message.header), now_ns, max_age_s)
        for message in (target, camera_info)
    ):
        publish_held_or_invalid(host, target, now_ns)
        return
    target_stamp_ns = stamp_ns(target.header)
    camera_info_stamp_ns = stamp_ns(camera_info.header)
    if not timestamps_within_skew(
        target_stamp_ns, camera_info_stamp_ns, max_sensor_skew_s
    ):
        publish_held_or_invalid(host, target, now_ns)
        return
    camera_frame = str(host.get_parameter("camera_frame").value)
    if (
        target.header.frame_id != camera_frame
        or camera_info.header.frame_id != camera_frame
    ):
        host._cache.reset()
        publish_invalid_and_empty(host, target, target.header, "frame_mismatch")
        return

    scan: LaserScan | None = host._scan
    usable_stamp: int | None = None
    if scan is not None:
        usable_stamp = usable_scan_stamp(
            stamp_ns(scan.header), target_stamp_ns, now_ns,
            max_scan_age_s, max_scan_skew_s,
        )
    fresh = fresh_observation_messages(
        tuple(host._observations.values()), target_stamp_ns, camera_info_stamp_ns,
        now_ns, max_age_s, max_sensor_skew_s,
    )
    if not any(
        message.valid and message.track_id == target.track_id for message in fresh
    ):
        host._cache.reset()
        publish_invalid_and_empty(
            host, target, target.header,
            f"no_fresh_obs track={target.track_id} obs={len(fresh)}")
        return
    selection = derive_target_selection(
        target_fields(target),
        tuple(observation_fields(message) for message in fresh),
    )
    if selection is None:
        host._cache.reset()
        publish_invalid_and_empty(host, target, target.header, "selection_empty")
        return

    parsed_info = parse_camera_info(
        camera_info_fields(camera_info),
        host._fallback_image_size(),
    )
    if parsed_info is None:
        publish_held_or_invalid(host, target, now_ns)
        return
    camera_transform = host._lookup_base_to_camera(target_stamp_ns)
    if camera_transform is None:
        publish_held_or_invalid(host, target, now_ns)
        return

    lidar_inputs: LidarInputs | None = None
    if scan is not None and usable_stamp is not None:
        laser_transform = host._lookup_laser_to_base(scan)
        if laser_transform is not None:
            lidar_inputs = LidarInputs(
                points=laser_scan_to_lidar_points(
                    scan_fields(scan), laser_transform),
                calibration=parsed_info.calibration,
                camera_transform=camera_transform,
                image_size=parsed_info.image_size,
                stamp_ns=usable_stamp,
                position_confidence=float(
                    str(host.get_parameter("lidar_position_confidence").value)),
            )
    decision = decide_branch(selection, lidar_inputs)
    if lidar_inputs is not None and scan is not None:
        host._obstacles_pub.publish(empty_obstacle_cloud(
            scan.header, host._base_frame(), decision.obstacles))
    else:
        host._obstacles_pub.publish(empty_obstacle_cloud(
            target.header, host._base_frame(), decision.obstacles))
    if decision.observation is not None:
        host._cache.update(
            track_id=int(target.track_id),
            point=decision.observation.point,
            range_m=decision.observation.range_m,
            bearing_rad=decision.observation.bearing_rad,
            stamp_ns=decision.observation.source_stamp_ns,
            source=decision.source,
            confidence=decision.confidence,
            provenance=decision.provenance,
        )
        host._target_pub.publish(metric_target_message(
            target, decision.observation, host._base_frame(),
            position_source=decision.source,
            position_confidence=decision.confidence,
            provenance=decision.provenance,
        ))
        return
    publish_held_or_invalid(host, target, now_ns)
