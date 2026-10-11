import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.person_tracking import (  # noqa: E402
    PersonTracker,
)


def _box(x1, y1, x2, y2, conf=0.9):
    return (x1, y1, x2, y2, conf)


def test_stable_id_for_static_person() -> None:
    tracker = PersonTracker()
    first = tracker.update([_box(10.0, 10.0, 50.0, 100.0)])
    second = tracker.update([_box(11.0, 10.0, 51.0, 100.0)])
    assert first[0][0] == second[0][0]


def test_reappearing_person_keeps_id_across_short_gap() -> None:
    tracker = PersonTracker()
    first = tracker.update([_box(10.0, 10.0, 50.0, 100.0)])
    for _ in range(5):
        assert tracker.update([]) == []
    second = tracker.update([_box(14.0, 10.0, 54.0, 100.0)])
    assert second[0][0] == first[0][0]


def test_distant_detection_spawns_new_id() -> None:
    tracker = PersonTracker()
    first = tracker.update([_box(10.0, 10.0, 50.0, 100.0)])
    second = tracker.update([_box(200.0, 10.0, 240.0, 100.0)])
    assert second[0][0] != first[0][0]


def test_hungarian_prefers_global_best_match() -> None:
    tracker = PersonTracker()
    tracker.update([_box(0.0, 0.0, 10.0, 10.0), _box(5.0, 0.0, 15.0, 10.0)])
    tracks = tracker.update([_box(2.0, 0.0, 12.0, 10.0), _box(0.0, 0.0, 10.0, 10.0)])
    by_id = {track_id: bbox for track_id, bbox, _conf in tracks}
    assert by_id[1] == (0.0, 0.0, 10.0, 10.0)
    assert by_id[2] == (2.0, 0.0, 12.0, 10.0)


def test_expired_track_gets_fresh_id() -> None:
    tracker = PersonTracker()
    first = tracker.update([_box(10.0, 10.0, 50.0, 100.0)])
    for _ in range(25):
        tracker.update([])
    second = tracker.update([_box(10.0, 10.0, 50.0, 100.0)])
    assert second[0][0] != first[0][0]
