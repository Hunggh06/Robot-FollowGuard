from pathlib import Path

SOURCE = (
    Path(__file__).parents[1]
    / "robot_autonomous_yolo_slam_nav2"
    / "yolo_target_node.py"
).read_text()

TAG_PARAMS = (
    "tag_family",
    "target_tag_id",
    "tag_roi_upscale",
)

def test_yolo_node_uses_tag_identity_wiring() -> None:
    assert "from .person_tag_detector import PersonTagDetector, tagged_tracks" in SOURCE
    assert "from .tag_person_identity import TagPersonIdentity" in SOURCE
    assert "from .person_annotation import PersonAnnotationPublisher, TagOverlay" in SOURCE
    assert "PersonTagDetector(" in SOURCE
    assert "TagPersonIdentity(" in SOURCE
    assert "needs_tag = " in SOURCE
    assert "tags: tuple[TagOverlay, ...]" in SOURCE
    assert "tagged_tracks(observations, tags, self._tag_id)" in SOURCE
    assert "tags={len(tags)} tag_search={needs_tag}" in SOURCE
    for param in TAG_PARAMS:
        assert f'"{param}"' in SOURCE, param
    assert '"lost_timeout_s"' in SOURCE
    assert "image_stamp_ns" in SOURCE
    assert "identity_confirmed = True" in SOURCE


def test_yolo_node_has_no_legacy_marker_wiring() -> None:
    lowered = SOURCE.lower()
    assert "scarf" not in lowered
    assert "PersonScarfDetector" not in SOURCE
    assert "ScarfPersonIdentity" not in SOURCE
    assert "ScarfOverlay" not in SOURCE
    assert "scarf_tracks" not in SOURCE
    assert "detections_topic" not in SOURCE
    assert "_detections_pub" not in SOURCE
    assert "_detections_sub" not in SOURCE
    assert "test_only_lock_without_tag" not in SOURCE
    assert "AprilTagDetectionArray" not in SOURCE
    assert "needs_scarf" not in SOURCE
    assert " scarves=" not in SOURCE
    assert " scarf=" not in SOURCE
    assert "scarf_search" not in SOURCE


def test_yolo_wrapper_is_absolute_import_entrypoint() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "yolo_target_node.py").read_text()
    assert "from robot_autonomous_yolo_slam_nav2.yolo_target_node import main" in source
