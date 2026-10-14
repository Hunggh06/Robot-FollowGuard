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

BAND_PARAMS = (
    "band_leg_roi_fraction",
    "band_confirm_frames",
    "band_min_area_px",
    "band_max_color_ratio",
    "band_min_sep_ratio",
    "band_max_dx_ratio",
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


def test_yolo_node_selects_identity_mode_with_band_defaults() -> None:
    assert '"identity_mode"' in SOURCE
    assert '"color_bands"' in SOURCE
    assert "from .color_band_person_identity import ColorBandPersonIdentity" in SOURCE
    assert "PersonColorBandsDetector" in SOURCE
    assert "banded_tracks" in SOURCE
    assert "from .person_annotation import PersonAnnotationPublisher, TagOverlay, BandOverlay" in SOURCE
    assert "ColorBandPersonIdentity(" in SOURCE
    assert "PersonColorBandsDetector(" in SOURCE
    assert "_resolve_identity_mode" in SOURCE
    assert "_bands_config_from_params" in SOURCE
    assert "ColorBandsConfig(" in SOURCE
    assert "self._band_selector.reset()" in SOURCE
    for param in BAND_PARAMS:
        assert f'"{param}"' in SOURCE, param
    assert '"band_red_hue_low_a"' in SOURCE
    assert '"band_blue_hue_low"' in SOURCE
    assert "mode = self._resolve_identity_mode()" in SOURCE
    assert 'if mode == "color_bands":' in SOURCE
    assert "banded_tracks(observations, band_overlays)" in SOURCE
    assert "self._band_selector.update(observations, banded_ids, image_stamp_ns)" in SOURCE
    assert "bands={len(bands)}" in SOURCE


def test_yolo_node_color_mode_never_scans_tags() -> None:
    color_branch = SOURCE.index('if mode == "color_bands":')
    else_branch = SOURCE.index("        else:", color_branch)
    tag_detect = SOURCE.index("self._tag_detector.detect")
    bands_detect = SOURCE.index("self._bands_detector.detect")
    assert SOURCE.count("self._tag_detector.detect") == 1
    assert SOURCE.count("self._bands_detector.detect") == 1
    assert color_branch < bands_detect < else_branch < tag_detect
    assert "self._annotation.publish(frame, tracked, result, msg.header, tags, bands, reid_state)" in SOURCE


def test_yolo_node_invalid_mode_falls_back_to_apriltag() -> None:
    assert "unknown identity_mode=" in SOURCE
    assert "falling back to apriltag" in SOURCE
    assert 'return "apriltag"' in SOURCE
    assert '("color_bands", "apriltag")' in SOURCE
