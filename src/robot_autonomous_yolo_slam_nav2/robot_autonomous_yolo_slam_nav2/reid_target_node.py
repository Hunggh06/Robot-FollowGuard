#!/usr/bin/env python3
"""ROS 2 camera person tracking with an explicit first-target lock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .reid_target_helpers import (
    PersonObservationFields,
    PersonTargetFields,
    selection_to_message_fields,
    selection_to_ros_messages,
    tracker_rows_to_observations,
)

if TYPE_CHECKING:
    from rclpy.node import Node


class ReidTargetNode:
    """Lazy facade that keeps ROS and model imports out of pure helper imports."""

    def __new__(cls) -> Node:
        from .reid_target_runtime import ReidTargetNode as RuntimeReidTargetNode

        return RuntimeReidTargetNode()


def main(args: list[str] | None = None) -> None:
    """Run the ROS 2 target node."""
    import rclpy

    rclpy.init(args=args)
    node = ReidTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
