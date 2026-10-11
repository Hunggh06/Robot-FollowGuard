"""Pure helpers for the red-zone guard: polygon math and verify timing.

The red polygon is the SAFE zone boundary drawn by hand in RViz
(/clicked_point corners, latched after zone_click_points clicks).
A confirmed target outside it for leaving_verify_s straight becomes a
violation. Speech lines and prompt/repeat timing live here so the ROS
shell stays thin.
"""

from __future__ import annotations

import math


SAY_VIOLATION = "Bạn đã đi vào vùng cấm, quay đầu ngay lập tức"
SAY_ASK_CONFIRM = "Hãy nói xác nhận để chuyển chế độ dẫn đường"
SAY_GUIDE_CONFIRMED = "Đã xác nhận chuyển chế độ dẫn đường"
SAY_BACK_SAFE = "Bạn đã quay trở lại vùng an toàn"


def point_in_polygon(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    """Ray-cast containment; the polygon closes implicitly.

    Points exactly on an edge count as inside. Fewer than 3 corners
    never contains anything.
    """
    if len(polygon) < 3:
        return False
    inside = False
    count = len(polygon)
    for index in range(count):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % count]
        if (y0 > y) == (y1 > y):
            continue
        intersect_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        if x < intersect_x:
            inside = not inside
        elif x == intersect_x:
            return True
    return inside


def latch_polygon(
    clicks: tuple[tuple[float, float], ...],
    num_points: int,
) -> tuple[tuple[float, float], ...] | None:
    """Return the polygon once enough corners were clicked, else None.

    Corners are ordered by angle around their centroid, so clicks in any
    order form a proper convex quad instead of a bowtie (hourglass).
    """
    if num_points >= 3 and len(clicks) >= num_points:
        corners = list(clicks[:num_points])
        center_x = sum(corner[0] for corner in corners) / len(corners)
        center_y = sum(corner[1] for corner in corners) / len(corners)
        corners.sort(
            key=lambda corner: math.atan2(
                corner[1] - center_y, corner[0] - center_x))
        return tuple(corners)
    return None


def polygon_centroid(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    """Centroid of the corners, or None when the polygon is degenerate.

    For concave shapes the centroid can fall outside; callers must
    verify containment and fall back to a corner.
    """
    if len(polygon) < 3:
        return None
    return (
        sum(corner[0] for corner in polygon) / len(polygon),
        sum(corner[1] for corner in polygon) / len(polygon),
    )


def debounce_outside(
    outside_now: bool,
    accum_s: float,
    dt_s: float,
    require_s: float,
) -> tuple[float, bool]:
    """Accumulate continuous outside time; inside resets the accumulator.

    Returns (new_accum_s, confirmed) where confirmed latches once the
    accumulated time reaches require_s.
    """
    if not outside_now:
        return 0.0, False
    accum_s = max(0.0, accum_s + max(0.0, dt_s))
    return accum_s, accum_s >= require_s


def confirm_match(text: str, keyword: str) -> bool:
    """Case-insensitive substring match for the spoken confirmation."""
    wanted = keyword.strip().lower()
    if not wanted:
        return False
    return wanted in text.lower()
