"""Lazy ROS entrypoint for Nav2 person following."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class Nav2PersonFollowNode:
    """Keep ROS action and message imports out of pure helper imports."""

    def __new__(cls) -> Node:
        from .nav2_person_follow_runtime import Nav2PersonFollowRuntime

        return Nav2PersonFollowRuntime()


def main(args: list[str] | None = None) -> None:
    """Run the Nav2 person-follow action client."""
    import rclpy

    rclpy.init(args=args)
    node = Nav2PersonFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
