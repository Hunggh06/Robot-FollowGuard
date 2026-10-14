from pathlib import Path

import cv2
import numpy as np
import pytest

from robot_autonomous_yolo_slam_nav2 import person_annotation
from robot_autonomous_yolo_slam_nav2.person_annotation import BandOverlay, TagOverlay
from robot_autonomous_yolo_slam_nav2.target_state import (
    TargetSelectionResult, TargetState, TrackId, TrackObservation,
)


def tag(
    family: str = "tag36h11",
    tag_id: int = 0,
    corners: tuple[tuple[float, float], ...] = ((80., 90.), (100., 90.), (100., 110.), (80., 110.)),
) -> TagOverlay:
    return TagOverlay(family, tag_id, corners)


def selection() -> TargetSelectionResult:
    target = TrackObservation(TrackId(7), 0.9, (40., 50., 140., 210.))
    other = TrackObservation(TrackId(12), 0.8, (180., 50., 280., 210.))
    return TargetSelectionResult(TargetState.TRACKING, target, (other,), True)


def labels_of(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    labels: list[str] = []
    original = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    return labels


def test_overlay_draws_all_people_selected_target_and_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    labels = labels_of(monkeypatch)
    tracks = [(7, (40., 50., 140., 210.), 0.9), (12, (180., 50., 280., 210.), 0.8)]
    output = person_annotation.annotate_people(frame, tracks, selection(), (tag(),))

    assert any("TARGET ID 7" in label for label in labels)
    assert any("person ID 12" in label for label in labels)
    assert any("tag36h11 ID 0" in label for label in labels)
    assert any("tag=CONFIRM" in label for label in labels)
    assert any("tags=1" in label for label in labels)
    assert tuple(output[150, 40]) == (0, 255, 0)
    assert tuple(output[150, 180]) == (255, 180, 0)
    assert tuple(output[100, 80]) == (0, 255, 255)
    assert not frame.any()


@pytest.mark.parametrize(
    ("tags", "result", "expected"),
    [
        ((tag(),), "locked", "tag=CONFIRM"),
        ((), "locked", "tag=SKIP"),
        ((), "lost", "tag=SEARCH"),
    ],
)
def test_overlay_status_strings(
    monkeypatch: pytest.MonkeyPatch,
    tags: tuple[TagOverlay, ...],
    result: str,
    expected: str,
) -> None:
    labels = labels_of(monkeypatch)
    sel = (
        selection()
        if result == "locked"
        else TargetSelectionResult(TargetState.LOST, None, (), False)
    )
    person_annotation.annotate_people(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [(7, (40., 50., 140., 210.), 0.9)],
        sel,
        tags,
    )
    assert any(expected in label for label in labels)


def test_overlay_does_not_mark_a_target_when_selection_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = labels_of(monkeypatch)
    lost = TargetSelectionResult(TargetState.LOST, None, (), False)
    person_annotation.annotate_people(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [(12, (180., 50., 280., 210.), 0.8)], lost, (),
    )
    assert any("target=none" in label and "lost" in label for label in labels)
    assert not any("TARGET ID" in label for label in labels)


def band(
    track_id: int = 7,
    red_center: tuple[float, float] = (90.0, 170.0),
    blue_center: tuple[float, float] = (90.0, 195.0),
) -> BandOverlay:
    return BandOverlay(TrackId(track_id), red_center, blue_center)


def test_overlay_draws_band_boxes_and_confirm_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    labels = labels_of(monkeypatch)
    tracks = [(7, (40., 50., 140., 210.), 0.9)]
    output = person_annotation.annotate_people(frame, tracks, selection(), (), (band(),))

    assert any("bands=CONFIRM" in label for label in labels)
    assert any("tags=0" in label for label in labels)
    assert tuple(output[164, 90]) == (0, 0, 255)
    assert tuple(output[189, 90]) == (255, 0, 0)
    assert not frame.any()


@pytest.mark.parametrize(
    ("bands", "result", "expected"),
    [
        ((band(),), "locked", "bands=CONFIRM"),
        ((), "locked", "bands=SKIP"),
        ((), "lost", "bands=SEARCH"),
    ],
)
def test_overlay_band_status_strings(
    monkeypatch: pytest.MonkeyPatch,
    bands: tuple[BandOverlay, ...],
    result: str,
    expected: str,
) -> None:
    labels = labels_of(monkeypatch)
    sel = (
        selection()
        if result == "locked"
        else TargetSelectionResult(TargetState.LOST, None, (), False)
    )
    person_annotation.annotate_people(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [(7, (40., 50., 140., 210.), 0.9)],
        sel,
        (),
        bands,
    )
    assert any(expected in label for label in labels)


def test_overlay_preserves_tag_path_alongside_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = labels_of(monkeypatch)
    output = person_annotation.annotate_people(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [(7, (40., 50., 140., 210.), 0.9)],
        selection(),
        (tag(),),
        (band(),),
    )
    assert any("tag=CONFIRM" in label for label in labels)
    assert any("bands=CONFIRM" in label for label in labels)
    assert tuple(output[100, 80]) == (0, 255, 255)


def test_annotation_module_has_no_legacy_marker_references() -> None:
    source = (
        Path(person_annotation.__file__).read_text()
    )
    lowered = source.lower()
    assert "scarf" not in lowered
    assert "apriltag" not in lowered
    assert "fresh_tags" not in source
    assert "detections_topic" not in source
    assert "annotation_tag_max_skew" not in source
    assert " scarves=" not in source
    assert " scarf=" not in source
    assert "TagOverlay" in source
    assert "(0, 255, 255)" in source
