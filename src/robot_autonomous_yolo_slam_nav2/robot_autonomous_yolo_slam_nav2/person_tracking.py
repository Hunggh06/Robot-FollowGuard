"""SORT-style person tracker: Kalman prediction + Hungarian assignment.

Same public API as the previous greedy IoU tracker
(``PersonTracker(iou_threshold)``, ``reset()``, ``update(boxes)``),
so callers are untouched. Each track runs a constant-velocity Kalman
filter over ``[cx, cy, w, h, vx, vy]``; detections are matched to
*predicted* boxes with a global Hungarian assignment gated by IoU.
Coasted (predicted) tracks keep their velocity through short gaps, so
a person reappearing near the predicted box keeps its ID instead of
spawning a new one. Unmatched tracks survive 20 missed updates,
exactly like before.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tag_person_identity import bbox_iou

_LARGE_COST = 1e9


class _KalmanBox:
    """Constant-velocity Kalman filter for one person box."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox
        self._x = np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self._p = np.diag([100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0])
        self._f = np.eye(6)
        self._f[0, 4] = 1.0
        self._f[1, 5] = 1.0
        self._h = np.zeros((4, 6))
        self._h[0, 0] = 1.0
        self._h[1, 1] = 1.0
        self._h[2, 2] = 1.0
        self._h[3, 3] = 1.0
        self._q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
        self._r = np.diag([4.0, 4.0, 4.0, 4.0])
        self.misses = 0

    def predict(self) -> None:
        """Advance the state one frame without a measurement."""
        self._x = self._f @ self._x
        self._p = self._f @ self._p @ self._f.T + self._q
        self._x[2] = max(1.0, self._x[2])
        self._x[3] = max(1.0, self._x[3])

    def correct(self, bbox: tuple[float, float, float, float]) -> None:
        """Fuse one detection into the state."""
        x1, y1, x2, y2 = bbox
        z = np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)],
            dtype=float,
        )
        innovation = z - self._h @ self._x
        s = self._h @ self._p @ self._h.T + self._r
        gain = self._p @ self._h.T @ np.linalg.inv(s)
        self._x = self._x + gain @ innovation
        self._x[2] = max(1.0, self._x[2])
        self._x[3] = max(1.0, self._x[3])
        self._p = (np.eye(6) - gain @ self._h) @ self._p
        self.misses = 0

    def predicted_bbox(self) -> tuple[float, float, float, float]:
        """Return the current box estimate in corner form."""
        cx, cy, w, h = (float(value) for value in self._x[:4])
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class PersonTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._threshold = iou_threshold
        self._tracks: dict[int, _KalmanBox] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Forget every track."""
        self._tracks.clear()

    def update(
        self, boxes: list[tuple[float, float, float, float, float]]
    ) -> list[tuple[int, tuple[float, float, float, float], float]]:
        """Match detections to predicted tracks, creating IDs as needed."""
        for track in self._tracks.values():
            track.predict()
        track_ids = list(self._tracks.keys())
        assigned_det: dict[int, int] = {}
        assigned_track: dict[int, int] = {}
        if track_ids and boxes:
            cost = np.full((len(track_ids), len(boxes)), _LARGE_COST)
            for row, track_id in enumerate(track_ids):
                predicted = self._tracks[track_id].predicted_bbox()
                for col, (x1, y1, x2, y2, _confidence) in enumerate(boxes):
                    overlap = bbox_iou(predicted, (x1, y1, x2, y2))
                    if overlap > self._threshold and math.isfinite(overlap):
                        cost[row, col] = 1.0 - overlap
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if cost[row, col] >= _LARGE_COST:
                    continue
                assigned_track[track_ids[row]] = col
                assigned_det[col] = track_ids[row]
        output: list[tuple[int, tuple[float, float, float, float], float]] = []
        for col, (x1, y1, x2, y2, confidence) in enumerate(boxes):
            bbox = (x1, y1, x2, y2)
            if col in assigned_det:
                track_id = assigned_det[col]
                self._tracks[track_id].correct(bbox)
            else:
                track_id = self._next_id
                self._next_id += 1
                fresh = _KalmanBox(bbox)
                self._tracks[track_id] = fresh
            output.append((track_id, bbox, confidence))
        for track_id, track in tuple(self._tracks.items()):
            if track_id not in assigned_track:
                track.misses += 1
                if track.misses > 20:
                    del self._tracks[track_id]
        return output
