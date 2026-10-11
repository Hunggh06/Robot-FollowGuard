import math

from robot_autonomous_yolo_slam_nav2.map_recovery_logic import (
    MapGrid,
    path_is_free,
    reachable_cells,
    world_to_cell,
)


def _grid(data: tuple[int, ...], width: int = 4, yaw: float = 0.0) -> MapGrid:
    return MapGrid("map", 10.0, 1.0, width, len(data) // width, 0.0, 0.0, yaw, data)


def test_rotated_origin_maps_world_point_to_expected_cell() -> None:
    grid = _grid((0,) * 16, yaw=math.pi / 2.0)

    assert world_to_cell(grid, (-0.5, 0.5)) == (0, 0)


def test_unknown_occupied_and_out_of_bounds_paths_are_rejected() -> None:
    unknown = _grid((0, -1, 0, 0) + (0,) * 12)
    occupied = _grid((0, 100, 0, 0) + (0,) * 12)
    free = _grid((0,) * 16)

    assert not path_is_free(unknown, (0.5, 0.5), (2.5, 0.5), 0.0)
    assert not path_is_free(occupied, (0.5, 0.5), (2.5, 0.5), 0.0)
    assert not path_is_free(free, (0.5, 0.5), (4.5, 0.5), 0.0)


def test_footprint_clearance_rejects_adjacent_obstacle() -> None:
    grid = _grid((0, 100, 0, 0) + (0,) * 12)

    assert not path_is_free(grid, (0.5, 1.5), (2.5, 1.5), 0.6)


def test_free_map_accepts_reachable_path() -> None:
    grid = _grid((0,) * 16)

    assert path_is_free(grid, (0.5, 0.5), (2.5, 0.5), 0.2)


def _corridor_grid() -> MapGrid:
    # 12x12 cells at 0.5 m: vertical wall at column 5, rows 0..10, with a
    # 1-cell gap only at row 11. Straight lines from the left side to the
    # right side all cross the wall; only the bottom corridor connects.
    width, height = 12, 12
    values = [0] * (width * height)
    for row in range(11):
        values[row * width + 5] = 100
    return MapGrid("map", 0.0, 0.5, width, height, 0.0, 0.0, 0.0, tuple(values))


def test_bfs_reaches_around_wall_through_free_corridor() -> None:
    grid = _corridor_grid()
    start, goal = (1.0, 1.0), (4.5, 1.0)

    assert not path_is_free(grid, start, goal, 0.1)

    connected = reachable_cells(grid, start, 0.1, max_expansions=5000)

    assert connected is not None
    assert world_to_cell(grid, goal) in connected


def test_bfs_does_not_treat_unknown_as_free() -> None:
    grid = _corridor_grid()
    unknown_values = list(grid.data)
    # Fill the only corridor gap with unknown cells: no detour remains.
    unknown_values[11 * 12 + 5] = -1
    unknown = MapGrid(
        grid.frame_id, grid.stamp_s, grid.resolution_m, grid.width,
        grid.height, grid.origin_x, grid.origin_y, grid.origin_yaw_rad,
        tuple(unknown_values),
    )

    connected = reachable_cells(unknown, (1.0, 1.0), 0.1, max_expansions=5000)

    assert connected is not None
    assert world_to_cell(unknown, (4.5, 1.0)) not in connected


def test_bfs_returns_none_when_start_has_no_footprint_clearance() -> None:
    values = [100] + [0] * 15
    grid = _grid(tuple(values))

    assert reachable_cells(grid, (0.5, 0.5), 0.2) is None


def test_bfs_expansion_is_bounded() -> None:
    grid = _grid((0,) * 16)

    connected = reachable_cells(grid, (0.5, 0.5), 0.0, max_expansions=1)

    assert connected is not None
    assert len(connected) <= 5 < 16
