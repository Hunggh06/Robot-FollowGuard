from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pytest
import rclpy
from std_msgs.msg import Float32MultiArray


MOTOR_ODOM_PATH = Path(__file__).parents[1] / "scripts" / "motor_odom.py"
MOTOR_ODOM_SPEC = importlib.util.spec_from_file_location("motor_odom_under_test", MOTOR_ODOM_PATH)
assert MOTOR_ODOM_SPEC is not None
assert MOTOR_ODOM_SPEC.loader is not None
MOTOR_ODOM_MODULE = importlib.util.module_from_spec(MOTOR_ODOM_SPEC)
MOTOR_ODOM_SPEC.loader.exec_module(MOTOR_ODOM_MODULE)


class MotorOdomNode(Protocol):
    x: float
    y: float
    th: float
    prev_enc: tuple[float, float] | None

    def state_cb(self, msg: Float32MultiArray) -> None: ...

    def destroy_node(self) -> None: ...


@pytest.fixture
def odom_node() -> Iterator[MotorOdomNode]:
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=[])
    node: MotorOdomNode = MOTOR_ODOM_MODULE.MotorOdom()
    try:
        yield node
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def motor_state(left_ticks: float, right_ticks: float) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.data = [0.0, 0.0, 0.0, 0.0, left_ticks, right_ticks]
    return msg


def test_initial_encoder_sample_sets_baseline_without_moving(odom_node: MotorOdomNode) -> None:
    odom_node.state_cb(motor_state(123.0, 456.0))

    assert odom_node.prev_enc == (123.0, 456.0)
    assert odom_node.x == pytest.approx(0.0)
    assert odom_node.y == pytest.approx(0.0)
    assert odom_node.th == pytest.approx(0.0)


def test_straight_encoder_motion_translates_along_current_heading(
    odom_node: MotorOdomNode,
) -> None:
    odom_node.state_cb(motor_state(0.0, 0.0))
    odom_node.state_cb(motor_state(100.0, 100.0))

    expected_distance = 100.0 * math.pi * 0.065 / 227.0

    assert odom_node.x == pytest.approx(expected_distance)
    assert odom_node.y == pytest.approx(0.0)
    assert odom_node.th == pytest.approx(0.0)


def test_pure_spin_changes_heading_without_translation(odom_node: MotorOdomNode) -> None:
    odom_node.state_cb(motor_state(0.0, 0.0))
    odom_node.state_cb(motor_state(-100.0, 100.0))

    m_per_tick = math.pi * 0.065 / 227.0
    expected_heading = 200.0 * m_per_tick / 0.2385

    assert odom_node.x == pytest.approx(0.0)
    assert odom_node.y == pytest.approx(0.0)
    assert odom_node.th == pytest.approx(expected_heading)


@pytest.mark.parametrize(
    ("left_ticks", "right_ticks"),
    [(100.0, 200.0), (200.0, 100.0)],
    ids=("positive_curve", "negative_curve"),
)
def test_pose_translation_uses_midpoint_heading_for_encoder_arc(
    odom_node: MotorOdomNode,
    left_ticks: float,
    right_ticks: float,
) -> None:
    odom_node.state_cb(motor_state(0.0, 0.0))
    odom_node.state_cb(motor_state(left_ticks, right_ticks))

    m_per_tick = math.pi * 0.065 / 227.0
    d_l = left_ticks * m_per_tick
    d_r = right_ticks * m_per_tick
    d_c = (d_l + d_r) * 0.5
    d_th = (d_r - d_l) / 0.2385
    expected_midpoint = d_th * 0.5

    assert odom_node.x == pytest.approx(d_c * math.cos(expected_midpoint))
    assert odom_node.y == pytest.approx(d_c * math.sin(expected_midpoint))
    assert odom_node.th == pytest.approx(d_th)


def test_encoder_arc_uses_midpoint_from_nonzero_heading(odom_node: MotorOdomNode) -> None:
    m_per_tick = math.pi * 0.065 / 227.0
    odom_node.state_cb(motor_state(100.0, 100.0))
    odom_node.state_cb(motor_state(0.0, 200.0))
    odom_node.state_cb(motor_state(100.0, 400.0))

    heading_before_arc = 200.0 * m_per_tick / 0.2385
    d_l = 100.0 * m_per_tick
    d_r = 200.0 * m_per_tick
    d_c = (d_l + d_r) * 0.5
    d_th = (d_r - d_l) / 0.2385
    expected_heading = heading_before_arc + d_th
    expected_midpoint = heading_before_arc + d_th * 0.5

    assert odom_node.x == pytest.approx(d_c * math.cos(expected_midpoint))
    assert odom_node.y == pytest.approx(d_c * math.sin(expected_midpoint))
    assert odom_node.th == pytest.approx(expected_heading)


def test_encoder_jump_is_rejected_and_next_sample_resumes_from_jump(
    odom_node: MotorOdomNode,
) -> None:
    m_per_tick = math.pi * 0.065 / 227.0
    odom_node.state_cb(motor_state(0.0, 0.0))
    odom_node.state_cb(motor_state(100.0, 100.0))
    pose_before_jump = (odom_node.x, odom_node.y, odom_node.th)

    odom_node.state_cb(motor_state(2101.0, 2101.0))

    assert odom_node.prev_enc == (2101.0, 2101.0)
    assert (odom_node.x, odom_node.y, odom_node.th) == pytest.approx(pose_before_jump)

    odom_node.state_cb(motor_state(2201.0, 2201.0))

    assert odom_node.x == pytest.approx(pose_before_jump[0] + 100.0 * m_per_tick)
    assert odom_node.y == pytest.approx(0.0)
    assert odom_node.th == pytest.approx(0.0)
