import math
import sys
from pathlib import Path

import pytest
from sensor_msgs.msg import Imu


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import imu_remap  # noqa: E402


FALLBACK_YAW_VARIANCE = 0.05
UNAVAILABLE_AXIS_VARIANCE = 1e6


@pytest.mark.parametrize(
    ("input_x", "bias", "expected_z"),
    ((1.75, 0.25, 1.5), (-1.75, 0.25, -2.0)),
)
def test_remap_uses_phone_x_minus_bias_as_yaw_rate(
    input_x: float,
    bias: float,
    expected_z: float,
) -> None:
    message = Imu()
    message.angular_velocity.x = input_x

    result = imu_remap.remap_imu(message, bias)

    assert result.angular_velocity.z == pytest.approx(expected_z)


def test_remap_preserves_stamp_and_publishes_in_base_link() -> None:
    message = Imu()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 345678901
    message.header.frame_id = "phone_link"

    result = imu_remap.remap_imu(message, 0.0)

    assert result.header.stamp == message.header.stamp
    assert result.header.frame_id == "base_link"


def test_remap_marks_orientation_and_acceleration_unavailable() -> None:
    result = imu_remap.remap_imu(Imu(), 0.0)

    assert result.orientation_covariance[0] == -1.0
    assert result.linear_acceleration_covariance[0] == -1.0


def test_remap_does_not_advertise_phone_x_or_y_as_usable_gyro() -> None:
    result = imu_remap.remap_imu(Imu(), 0.0)

    assert result.angular_velocity.x == 0.0
    assert result.angular_velocity.y == 0.0
    assert result.angular_velocity_covariance[0] == UNAVAILABLE_AXIS_VARIANCE
    assert result.angular_velocity_covariance[4] == UNAVAILABLE_AXIS_VARIANCE
    assert math.isfinite(result.angular_velocity_covariance[0])
    assert math.isfinite(result.angular_velocity_covariance[4])


@pytest.mark.parametrize(
    ("source_x_variance", "expected_yaw_variance", "expected_marker"),
    (
        (0.125, 0.125, UNAVAILABLE_AXIS_VARIANCE),
        (0.0, FALLBACK_YAW_VARIANCE, UNAVAILABLE_AXIS_VARIANCE),
        (-1.0, FALLBACK_YAW_VARIANCE, -1.0),
    ),
)
def test_remap_handles_known_unknown_and_unavailable_x_covariance(
    source_x_variance: float,
    expected_yaw_variance: float,
    expected_marker: float,
) -> None:
    message = Imu()
    message.angular_velocity_covariance[0] = source_x_variance

    result = imu_remap.remap_imu(message, 0.0)

    assert result.angular_velocity_covariance[8] == pytest.approx(
        expected_yaw_variance,
    )
    assert result.angular_velocity_covariance[0] == expected_marker
