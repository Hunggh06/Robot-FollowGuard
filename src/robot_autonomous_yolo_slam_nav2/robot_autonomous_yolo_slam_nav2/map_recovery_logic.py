"""Pure timestamped occupancy-grid geometry for bounded recovery."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion


Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MapGrid:
    """Immutable map snapshot in one world frame."""

    frame_id: str
    stamp_s: float
    resolution_m: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw_rad: float
    data: tuple[int, ...]

    @property
    def known_free_cells(self) -> int:
        """Count cells explicitly observed as free."""
        return sum(value == 0 for value in self.data)


def world_to_cell(grid: MapGrid, point: Point2D) -> tuple[int, int] | None:
    """Convert a world point into a bounded grid index, respecting origin yaw."""
    dx = point[0] - grid.origin_x
    dy = point[1] - grid.origin_y
    cos_yaw = math.cos(grid.origin_yaw_rad)
    sin_yaw = math.sin(grid.origin_yaw_rad)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    cell = (math.floor(local_x / grid.resolution_m), math.floor(local_y / grid.resolution_m))
    if 0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height:
        return cell
    return None


def path_is_free(
    grid: MapGrid,
    start: Point2D,
    goal: Point2D,
    footprint_radius_m: float,
) -> bool:
    """Require every sampled path cell and circular footprint neighborhood to be free."""
    if grid.resolution_m <= 0.0 or grid.width <= 0 or grid.height <= 0:
        return False
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    samples = max(1, math.ceil(distance / (grid.resolution_m * 0.5)))
    for index in range(samples + 1):
        fraction = index / samples
        point = (
            start[0] + (goal[0] - start[0]) * fraction,
            start[1] + (goal[1] - start[1]) * fraction,
        )
        if not _footprint_is_free(grid, point, footprint_radius_m):
            return False
    return True


def _footprint_is_free(grid: MapGrid, center: Point2D, radius_m: float) -> bool:
    center_cell = world_to_cell(grid, center)
    if center_cell is None:
        return False
    radius_cells = math.ceil(max(0.0, radius_m) / grid.resolution_m) + 1
    diagonal = grid.resolution_m * math.sqrt(2.0) / 2.0
    for row in range(center_cell[1] - radius_cells, center_cell[1] + radius_cells + 1):
        for column in range(center_cell[0] - radius_cells, center_cell[0] + radius_cells + 1):
            cell_center = _cell_center(grid, column, row)
            if math.hypot(cell_center[0] - center[0], cell_center[1] - center[1]) <= radius_m + diagonal:
                if not _cell_is_free(grid, column, row):
                    return False
    return True


def _cell_center(grid: MapGrid, column: int, row: int) -> Point2D:
    local_x = (column + 0.5) * grid.resolution_m
    local_y = (row + 0.5) * grid.resolution_m
    cos_yaw = math.cos(grid.origin_yaw_rad)
    sin_yaw = math.sin(grid.origin_yaw_rad)
    return (
        grid.origin_x + cos_yaw * local_x - sin_yaw * local_y,
        grid.origin_y + sin_yaw * local_x + cos_yaw * local_y,
    )


def _cell_is_free(grid: MapGrid, column: int, row: int) -> bool:
    if not 0 <= column < grid.width or not 0 <= row < grid.height:
        return False
    index = row * grid.width + column
    if index >= len(grid.data):
        return False
    value = grid.data[index]
    return value == 0


def reachable_cells(
    grid: MapGrid,
    start: Point2D,
    footprint_radius_m: float,
    max_expansions: int = 4000,
) -> set[tuple[int, int]] | None:
    """Flood known-free cells from start; unknown counts as blocked.

    Returns the connected cell set, or None when the start itself has no
    footprint clearance. Expansion is bounded so a 10 Hz caller stays
    responsive; callers cache the result per map stamp.
    """
    if grid.resolution_m <= 0.0 or grid.width <= 0 or grid.height <= 0:
        return None
    origin = world_to_cell(grid, start)
    if origin is None:
        return None
    if not _footprint_is_free(grid, start, footprint_radius_m):
        return None
    if len(grid.data) != grid.width * grid.height:
        return None
    margin = max(0.0, footprint_radius_m) + grid.resolution_m * math.sqrt(2.0) / 2.0
    radius_cells = math.ceil(margin / grid.resolution_m)
    offsets = np.arange(-radius_cells, radius_cells + 1) * grid.resolution_m
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= margin ** 2
    free = np.asarray(grid.data).reshape(grid.height, grid.width) == 0
    traversable = binary_erosion(free, structure=disk, border_value=0)
    seen: set[tuple[int, int]] = {origin}
    frontier = deque([origin])
    expansions = 0
    while frontier and expansions < max(0, max_expansions):
        column, row = frontier.popleft()
        expansions += 1
        for next_column, next_row in (
            (column + 1, row),
            (column - 1, row),
            (column, row + 1),
            (column, row - 1),
        ):
            if (next_column, next_row) in seen:
                continue
            if not (0 <= next_column < grid.width and 0 <= next_row < grid.height):
                continue
            if not traversable[next_row, next_column]:
                continue
            seen.add((next_column, next_row))
            frontier.append((next_column, next_row))
    return seen
