from pathlib import Path

from robot_autonomous_yolo_slam_nav2.tag_person_identity import bbox_iou as _iou


def test_yolo_node_uses_light_tracker_and_tag_confirm() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0
    source = (Path(__file__).parents[1] / "robot_autonomous_yolo_slam_nav2" / "yolo_target_node.py").read_text()
    assert "tag_roi_upscale" in source
    assert "image_stamp_ns" in source


def test_yolo_wrapper_is_absolute_import_entrypoint() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "yolo_target_node.py").read_text()
    assert "from robot_autonomous_yolo_slam_nav2.yolo_target_node import main" in source
