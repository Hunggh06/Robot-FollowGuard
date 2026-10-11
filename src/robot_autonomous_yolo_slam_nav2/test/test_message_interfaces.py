from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]


def _fields(message_name: str) -> set[str]:
    message_path = PACKAGE_ROOT / "msg" / message_name
    fields: set[str] = set()
    for line in message_path.read_text(encoding="utf-8").splitlines():
        declaration = line.split("#", 1)[0].strip()
        if not declaration:
            continue
        _, field_name = declaration.split()
        field_name = field_name.split("=", 1)[0]
        # Skip UPPER_CASE message constants (e.g. POSITION_SOURCE_*).
        if field_name.upper() == field_name:
            continue
        fields.add(field_name)
    return fields


def test_person_target_declares_follow_adapter_fields() -> None:
    assert _fields("PersonTarget.msg") == {
        "header",
        "track_id",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "state",
        "valid",
        "identity_confirmed",
        "point_base_link",
        "range_m",
        "bearing_rad",
        "footpoint_valid",
        "footpoint_x",
        "footpoint_y",
        "footpoint_confidence",
        "position_source",
        "position_confidence",
        "provenance",
    }


def test_person_observation_declares_raw_tracking_fields() -> None:
    assert _fields("PersonObservation.msg") == {
        "header",
        "track_id",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "state",
        "valid",
        "footpoint_valid",
        "footpoint_x",
        "footpoint_y",
        "footpoint_confidence",
    }


def test_person_obstacle_declares_costmap_freshness_fields() -> None:
    assert _fields("PersonObstacle.msg") == {
        "header",
        "track_id",
        "confidence",
        "position",
        "valid",
        "age_s",
        "fresh",
    }
