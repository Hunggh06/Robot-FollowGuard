"""Hold the last valid position for the same confirmed track only.

A held entry keeps its ORIGINAL measurement stamp so its age keeps
growing, carries a decaying confidence, and is always published with
valid=false. A different track ID, an explicit reset, or the age bound
clears the cache. Held geometry never becomes a fresh Nav2 goal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeldPosition:
    """The retained point with its original stamp and decayed confidence."""

    track_id: int
    point: tuple[float, float, float]
    range_m: float
    bearing_rad: float
    stamp_ns: int
    source: str
    confidence: float
    provenance: str


class PositionCache:
    """Single-slot cache keyed by confirmed track identity."""

    __slots__ = ("_decay_per_s", "_held", "_timeout_s")

    def __init__(self, hold_timeout_s: float = 5.0, decay_per_s: float = 0.05) -> None:
        self._timeout_s = hold_timeout_s
        self._decay_per_s = decay_per_s
        self._held: HeldPosition | None = None

    def update(
        self,
        track_id: int,
        point: tuple[float, float, float],
        range_m: float,
        bearing_rad: float,
        stamp_ns: int,
        source: str,
        confidence: float,
        provenance: str,
    ) -> None:
        """Store one valid geometry sample, ignoring malformed input."""
        if (
            track_id < 0
            or stamp_ns <= 0
            or not all(math.isfinite(value) for value in (*point, range_m, bearing_rad))
            or range_m <= 0.0
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or source not in ("lidar", "camera")
        ):
            return
        self._held = HeldPosition(
            track_id, point, range_m, bearing_rad,
            stamp_ns, source, confidence, provenance,
        )

    def held(self, track_id: int, now_ns: int) -> HeldPosition | None:
        """Return the decayed entry for the same track, else clear and None."""
        current = self._held
        if current is None or current.track_id != track_id:
            self._held = None
            return None
        age_s = (now_ns - current.stamp_ns) / 1_000_000_000.0
        if not math.isfinite(age_s) or age_s < 0.0 or age_s > self._timeout_s:
            self._held = None
            return None
        confidence = current.confidence * max(0.0, 1.0 - self._decay_per_s * age_s)
        if confidence <= 0.0:
            self._held = None
            return None
        return HeldPosition(
            current.track_id, current.point, current.range_m,
            current.bearing_rad, current.stamp_ns, current.source,
            confidence, current.provenance,
        )

    def reset(self) -> None:
        """Forget any retained position."""
        self._held = None
