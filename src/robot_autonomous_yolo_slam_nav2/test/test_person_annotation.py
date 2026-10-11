import cv2
import numpy as np
import pytest
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
from std_msgs.msg import Header

from robot_autonomous_yolo_slam_nav2 import person_annotation
from robot_autonomous_yolo_slam_nav2.target_state import (
    TargetSelectionResult, TargetState, TrackId, TrackObservation,
)


def tag_message() -> AprilTagDetectionArray:
    message = AprilTagDetectionArray()
    message.header.frame_id = "phone_camera_back"
    message.header.stamp.sec = 10
    message.detections = [AprilTagDetection(
        family="tag36h11", id=0,
        corners=[Point(x=x, y=y) for x, y in ((80., 90.), (100., 90.), (100., 110.), (80., 110.))],
    )]
    return message


def selection() -> TargetSelectionResult:
    target = TrackObservation(TrackId(7), 0.9, (40., 50., 140., 210.))
    other = TrackObservation(TrackId(12), 0.8, (180., 50., 280., 210.))
    return TargetSelectionResult(TargetState.TRACKING, target, (other,), True)


def test_overlay_draws_all_people_selected_target_and_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    header = Header(frame_id="phone_camera_back")
    header.stamp.sec = 10
    tags = person_annotation.fresh_tags(tag_message(), header, 0.1)
    labels: list[str] = []
    original = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    tracks = [(7, (40., 50., 140., 210.), 0.9), (12, (180., 50., 280., 210.), 0.8)]
    output = person_annotation.annotate_people(frame, tracks, selection(), tags)

    assert any("TARGET ID 7" in label for label in labels)
    assert any("person ID 12" in label for label in labels)
    assert any("tag36h11 ID 0" in label for label in labels)
    assert tuple(output[150, 40]) == (0, 255, 0)
    assert tuple(output[150, 180]) == (255, 180, 0)
    assert tuple(output[100, 80]) == (0, 255, 255)
    assert not frame.any()


@pytest.mark.parametrize("offset_ns,frame_id", [(200_000_000, "phone_camera_back"), (0, "wrong_camera")])
def test_overlay_rejects_stale_or_wrong_frame_tags(offset_ns: int, frame_id: str) -> None:
    header = Header(frame_id=frame_id)
    header.stamp.sec = 10
    header.stamp.nanosec = offset_ns

    assert person_annotation.fresh_tags(tag_message(), header, 0.1) == ()


def test_overlay_does_not_mark_a_target_when_selection_is_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[str] = []
    original = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    lost = TargetSelectionResult(TargetState.LOST, None, (), False)
    person_annotation.annotate_people(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [(12, (180., 50., 280., 210.), 0.8)], lost, (),
    )
    assert any("target=none" in label and "lost" in label for label in labels)
    assert not any("TARGET ID" in label for label in labels)
