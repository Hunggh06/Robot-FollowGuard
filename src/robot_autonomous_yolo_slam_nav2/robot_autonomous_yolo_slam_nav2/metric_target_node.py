"""Lazy ROS entrypoint for metric target association."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class MetricTargetNode:
    """Keep ROS imports out of pure helper and test imports."""

    def __new__(cls) -> Node:
        from .metric_target_runtime import MetricTargetRuntime

        return MetricTargetRuntime()


def main(args: list[str] | None = None) -> None:
    """Run the metric association ROS node."""
    import rclpy

    rclpy.init(args=args)
    node = MetricTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
