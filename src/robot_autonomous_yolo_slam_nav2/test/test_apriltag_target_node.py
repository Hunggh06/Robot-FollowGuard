import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.apriltag_target_node import (  # noqa: E402
    _stamp_ns,
    _tag_id,
    detection_frame_matches,
    detection_message_to_observation,
)
from robot_autonomous_yolo_slam_nav2.apriltag_target_state import TAG_FAMILY  # noqa: E402


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class DetectionMessage:
    id: object
    corners: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class Stamp:
    sec: object
    nanosec: object


@dataclass(frozen=True, slots=True)
class Header:
    stamp: Stamp
    frame_id: str


def test_runtime_boundary_maps_verified_detection_shape_without_importing_detector() -> None:
    message = DetectionMessage(
        id=(0,),
        corners=(
            Point(10.0, 20.0),
            Point(30.0, 20.0),
            Point(30.0, 60.0),
            Point(10.0, 60.0),
        ),
    )

    observation = detection_message_to_observation(message, TAG_FAMILY, 42)

    assert observation is not None
    assert observation.family == TAG_FAMILY
    assert observation.tag_id == 0
    assert observation.stamp_ns == 42
    assert observation.corners == ((10.0, 20.0), (30.0, 20.0), (30.0, 60.0), (10.0, 60.0))


def test_runtime_boundary_rejects_malformed_ids_and_corners() -> None:
    malformed_id = DetectionMessage(id=(0, 1), corners=())
    malformed_corners = DetectionMessage(id=(0,), corners=(Point(1.0, 2.0),))

    assert detection_message_to_observation(malformed_id, TAG_FAMILY, 42) is None
    assert detection_message_to_observation(malformed_corners, TAG_FAMILY, 42) is None


def test_runtime_boundary_rejects_non_finite_corners() -> None:
    malformed = DetectionMessage(
        id=(0,),
        corners=(
            Point(float("nan"), 20.0),
            Point(30.0, 20.0),
            Point(30.0, 60.0),
            Point(10.0, 60.0),
        ),
    )

    assert detection_message_to_observation(malformed, TAG_FAMILY, 42) is None


def test_runtime_boundary_rejects_coercible_non_integral_ids() -> None:
    assert _tag_id(DetectionMessage(id=(0.9,), corners=())) is None
    assert _tag_id(DetectionMessage(id=(True,), corners=())) is None


def test_runtime_boundary_rejects_malformed_timestamps() -> None:
    malformed = Header(Stamp(1, 1_000_000_000), "phone_camera_back")

    try:
        _stamp_ns(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed timestamp was accepted")


def test_runtime_boundary_requires_configured_detection_frame() -> None:
    assert detection_frame_matches(Header(Stamp(1, 0), "phone_camera_back"), "phone_camera_back")
    assert not detection_frame_matches(Header(Stamp(1, 0), "camera_link"), "phone_camera_back")


def test_ros_detector_dependency_is_not_imported_at_module_import_time() -> None:
    source = (Path(__file__).parents[1] / "robot_autonomous_yolo_slam_nav2" / "apriltag_target_node.py").read_text()
    assert "from apriltag_msgs.msg import" not in source.split("def main", 1)[0]


def test_ros_executable_wrapper_imports_package_entrypoint() -> None:
    wrapper = Path(__file__).parents[1] / "scripts" / "apriltag_target_node.py"
    namespace = runpy.run_path(str(wrapper), run_name="apriltag_target_node_wrapper")
    assert callable(namespace["main"])
