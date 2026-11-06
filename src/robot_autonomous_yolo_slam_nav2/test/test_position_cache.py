import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.position_cache import (  # noqa: E402
    PositionCache,
)


def test_returns_none_without_update() -> None:
    cache = PositionCache(hold_timeout_s=5.0)
    assert cache.held(track_id=1, now_ns=2_000_000_000) is None


def test_holds_same_track_with_original_stamp_and_decay() -> None:
    cache = PositionCache(hold_timeout_s=5.0, decay_per_s=0.05)
    cache.update(
        track_id=3, point=(2.0, 0.0, -0.01), range_m=2.0,
        bearing_rad=0.0, stamp_ns=1_000_000_000,
        source="lidar", confidence=0.9, provenance="lidar_bbox",
    )
    held = cache.held(track_id=3, now_ns=3_000_000_000)
    assert held is not None
    assert held.point == (2.0, 0.0, -0.01)
    assert held.stamp_ns == 1_000_000_000
    assert held.confidence < 0.9
    assert held.confidence > 0.0


def test_clears_on_different_track_id() -> None:
    cache = PositionCache(hold_timeout_s=5.0)
    cache.update(
        track_id=3, point=(2.0, 0.0, -0.01), range_m=2.0,
        bearing_rad=0.0, stamp_ns=1_000_000_000,
        source="lidar", confidence=0.9, provenance="lidar_bbox",
    )
    assert cache.held(track_id=4, now_ns=2_000_000_000) is None
    assert cache.held(track_id=3, now_ns=2_000_000_000) is None


def test_expires_after_timeout() -> None:
    cache = PositionCache(hold_timeout_s=5.0)
    cache.update(
        track_id=3, point=(2.0, 0.0, -0.01), range_m=2.0,
        bearing_rad=0.0, stamp_ns=1_000_000_000,
        source="lidar", confidence=0.9, provenance="lidar_bbox",
    )
    assert cache.held(track_id=3, now_ns=7_000_000_000) is None


def test_reset_clears_cache() -> None:
    cache = PositionCache(hold_timeout_s=5.0)
    cache.update(
        track_id=3, point=(2.0, 0.0, -0.01), range_m=2.0,
        bearing_rad=0.0, stamp_ns=1_000_000_000,
        source="lidar", confidence=0.9, provenance="lidar_bbox",
    )
    cache.reset()
    assert cache.held(track_id=3, now_ns=2_000_000_000) is None


def test_accepts_fused_source() -> None:
    cache = PositionCache(hold_timeout_s=5.0, decay_per_s=0.05)
    cache.update(
        track_id=5, point=(1.8, 0.1, 0.0), range_m=1.8027756,
        bearing_rad=0.0554985, stamp_ns=1_000_000_000,
        source="fused", confidence=0.92, provenance="kalman_fused_lidar_bbox+camera_bbox_wh",
    )
    held = cache.held(track_id=5, now_ns=1_500_000_000)
    assert held is not None
    assert held.source == "fused"
