"""Centralized clearance ladder shared by gate and recovery planning."""

from __future__ import annotations

from dataclasses import dataclass


def derive_clearance(master_m: float, delta_m: float, origin_offset_m: float) -> float:
    """Derive one clearance band from the single master safety value."""
    return max(0.0, float(master_m) + float(delta_m) - float(origin_offset_m))


def recovery_clearance(requested_m: float, guard_m: float) -> float:
    """Keep the larger margin: never silently reduce a requested clearance.

    The guard is a floor, not a ceiling. Recovery plans with at least the
    guard distance and honors any larger operator-requested margin.
    """
    return max(0.0, max(float(requested_m), float(guard_m)))


@dataclass(frozen=True, slots=True)
class ClearancePolicy:
    """Stop and slow distances derived from one master clearance."""

    master_m: float = 0.6
    origin_offset_m: float = 0.2
    slow_delta_m: float = 0.3

    @property
    def stop_m(self) -> float:
        """Return the forward stop distance."""
        return derive_clearance(self.master_m, 0.0, self.origin_offset_m)

    @property
    def slow_m(self) -> float:
        """Return the forward slow-down distance."""
        return derive_clearance(self.master_m, self.slow_delta_m, self.origin_offset_m)
