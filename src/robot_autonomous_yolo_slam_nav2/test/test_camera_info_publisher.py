import math
import sys
from pathlib import Path

import pytest
import rclpy
import yaml
from rclpy.qos import ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

PACKAGE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from robot_autonomous_yolo_slam_nav2.camera_calibration import CalibrationError  # noqa: E402
from robot_autonomous_yolo_slam_nav2.camera_info_calibration import (  # noqa: E402
    CameraCalibration,
    load_camera_calibration,
)
from robot_autonomous_yolo_slam_nav2.camera_info_publisher import (  # noqa: E402
    CameraInfoPublisherNode,
)


CALIBRATION_PATH = PACKAGE_DIR / "calibration" / "phone_camera_back.yaml"


class _PublisherSpy:
    def __init__(self) -> None:
        self.messages: list[CameraInfo] = []

    def publish(self, message: CameraInfo) -> None:
        self.messages.append(message)


def test_generated_calibration_loads_with_validated_ros_shapes() -> None:
    calibration = load_camera_calibration(CALIBRATION_PATH)

    assert isinstance(calibration, CameraCalibration)
    assert (calibration.image_width, calibration.image_height) == (320, 240)
    assert len(calibration.camera_matrix) == 9
    assert len(calibration.distortion_coefficients) == 5
    assert len(calibration.rectification_matrix) == 9
    assert len(calibration.projection_matrix) == 12
    assert calibration.camera_matrix[0] != 0.0
    assert calibration.camera_matrix[4] != 0.0
    assert all(math.isfinite(value) for value in calibration.camera_matrix)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload.__setitem__("image_width", 0),
        lambda payload: payload["camera_matrix"]["data"].pop(),
        lambda payload: payload["distortion_coefficients"]["data"].__setitem__(0, float("nan")),
        lambda payload: payload["camera_matrix"]["data"].__setitem__(0, 0.0),
        lambda payload: payload.__setitem__("distortion_model", "unknown"),
    ),
)
def test_invalid_calibration_is_rejected(tmp_path: Path, mutator) -> None:
    payload = yaml.safe_load(CALIBRATION_PATH.read_text())
    mutator(payload)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(CalibrationError):
        load_camera_calibration(path)


def test_missing_or_empty_calibration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError):
        load_camera_calibration("")
    with pytest.raises(CalibrationError):
        load_camera_calibration(tmp_path / "missing.yaml")


def test_image_callback_stamps_host_clock_and_uses_camera_frame() -> None:
    rclpy.init(args=["--ros-args", "-p", f"calibration_file:={CALIBRATION_PATH}"])
    node = CameraInfoPublisherNode()
    try:
        publisher = _PublisherSpy()
        setattr(node, "_camera_info_publisher", publisher)
        before_ns = node.get_clock().now().nanoseconds

        image = Image()
        image.header.stamp.sec = 123
        image.header.stamp.nanosec = 456789

        node._image_callback(image)

        after_ns = node.get_clock().now().nanoseconds
        assert len(publisher.messages) == 1
        camera_info = publisher.messages[0]
        stamp_ns = camera_info.header.stamp.sec * 1_000_000_000 + camera_info.header.stamp.nanosec
        assert before_ns <= stamp_ns <= after_ns
        assert (stamp_ns != 123 * 1_000_000_000 + 456789)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    calibration = load_camera_calibration(CALIBRATION_PATH)
    assert calibration.image_width == 320
    assert calibration.image_height == 240
    assert camera_info.header.frame_id == "phone_camera_back"
    assert camera_info.width == 320
    assert camera_info.height == 240
    assert camera_info.k[0] == pytest.approx(136.9219816928559)
    assert camera_info.d == pytest.approx(calibration.distortion_coefficients)


def test_empty_default_node_does_not_create_publisher() -> None:
    rclpy.init(args=[])
    node = CameraInfoPublisherNode()
    try:
        assert node._camera_info_publisher is None
        assert node._image_subscription is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_camera_info_publisher_uses_reliable_qos_for_detector() -> None:
    rclpy.init(args=[
        "--ros-args",
        "-p",
        f"calibration_file:={CALIBRATION_PATH}",
    ])
    node = CameraInfoPublisherNode()
    try:
        assert node._camera_info_publisher is not None
        assert node._camera_info_publisher.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_camera_info_subscriber_uses_sensor_data_qos_for_camera_stream() -> None:
    rclpy.init(args=[
        "--ros-args",
        "-p",
        f"calibration_file:={CALIBRATION_PATH}",
    ])
    node = CameraInfoPublisherNode()
    try:
        assert node._image_subscription is not None
        assert node._image_subscription.qos_profile == qos_profile_sensor_data
    finally:
        node.destroy_node()
        rclpy.shutdown()
