import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.zone_guide_logic import (  # noqa: E402
    SAY_ASK_CONFIRM,
    SAY_BACK_SAFE,
    SAY_GUIDE_CONFIRMED,
    SAY_VIOLATION,
    confirm_match,
    debounce_outside,
    latch_polygon,
    point_in_polygon,
    polygon_centroid,
)


SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def test_point_inside_square() -> None:
    assert point_in_polygon(2.0, 1.5, SQUARE) is True


def test_point_outside_square() -> None:
    assert point_in_polygon(5.0, 1.5, SQUARE) is False
    assert point_in_polygon(2.0, -0.5, SQUARE) is False


def test_point_on_edge_counts_inside() -> None:
    assert point_in_polygon(0.0, 1.5, SQUARE) is True


def test_degenerate_polygon_never_contains() -> None:
    assert point_in_polygon(0.0, 0.0, ()) is False
    assert point_in_polygon(0.0, 0.0, ((0.0, 0.0), (1.0, 1.0))) is False


def test_latch_polygon_after_enough_clicks() -> None:
    clicks = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0))
    assert latch_polygon(clicks, 4) is None
    latched = latch_polygon(clicks + ((0.0, 3.0),), 4)
    assert latched == ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def test_latch_polygon_sorts_shuffled_clicks_no_bowtie() -> None:
    shuffled = ((4.0, 3.0), (0.0, 0.0), (0.0, 3.0), (4.0, 0.0))
    latched = latch_polygon(shuffled, 4)
    assert latched == ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
    assert latched is not None
    assert point_in_polygon(2.0, 1.5, latched) is True
    assert point_in_polygon(5.0, 1.5, latched) is False


def test_debounce_confirms_after_continuous_outside() -> None:
    accum, confirmed = debounce_outside(True, 0.0, 1.0, 2.0)
    assert (accum, confirmed) == (1.0, False)
    accum, confirmed = debounce_outside(True, accum, 1.0, 2.0)
    assert (accum, confirmed) == (2.0, True)


def test_debounce_resets_when_back_inside() -> None:
    accum, confirmed = debounce_outside(False, 1.5, 0.5, 2.0)
    assert (accum, confirmed) == (0.0, False)


def test_confirm_match_is_case_insensitive_substring() -> None:
    assert confirm_match("Dạ XÁC NHẬN ạ", "xác nhận") is True
    assert confirm_match("đi tiếp đi", "xác nhận") is False
    assert confirm_match("xác nhận", "") is False


def test_speech_lines_are_exact() -> None:
    assert SAY_VIOLATION == "Bạn đã đi vào vùng cấm, quay đầu ngay lập tức"
    assert SAY_ASK_CONFIRM == "Hãy nói xác nhận để chuyển chế độ dẫn đường"
    assert SAY_GUIDE_CONFIRMED == "Đã xác nhận chuyển chế độ dẫn đường"
    assert SAY_BACK_SAFE == "Bạn đã quay trở lại vùng an toàn"


def test_polygon_centroid_of_square() -> None:
    assert polygon_centroid(SQUARE) == (2.0, 1.5)


def test_polygon_centroid_rejects_degenerate() -> None:
    assert polygon_centroid(()) is None
    assert polygon_centroid(((0.0, 0.0), (1.0, 1.0))) is None
