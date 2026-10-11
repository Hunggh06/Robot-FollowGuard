"""Lazy ROS entrypoint for the red-zone guard and guide home."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class ZoneGuardNode:
    """Keep ROS action and message imports out of pure helper imports."""

    def __new__(cls) -> Node:
        from .zone_guard_runtime import ZoneGuardRuntime

        return ZoneGuardRuntime()


def main(args: list[str] | None = None) -> None:
    """Run the red-zone guard node."""
    import rclpy

    rclpy.init(args=args)
    node = ZoneGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
