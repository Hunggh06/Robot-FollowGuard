from pathlib import Path

import pytest
import yaml


PACKAGE_DIR = Path(__file__).parents[1]
CONFIGURATIONS = (
    ("nav2_params.yaml", 0.20),
    ("nav2_params_slam.yaml", 0.25),
)
COSTMAP_NAMES = ("local_costmap", "global_costmap")


def _load_configuration(filename: str):
    return yaml.safe_load((PACKAGE_DIR / "config" / filename).read_text())


@pytest.mark.parametrize(("filename", "max_forward_speed"), CONFIGURATIONS)
def test_costmaps_include_fresh_people_obstacles_and_preserve_scan(
    filename: str, max_forward_speed: float
) -> None:
    config = _load_configuration(filename)

    for costmap_name in COSTMAP_NAMES:
        parameters = config[costmap_name][costmap_name]["ros__parameters"]
        voxel_layer = parameters["voxel_layer"]
        sources = voxel_layer["observation_sources"].split()

        assert sources == ["scan", "people"]
        assert parameters["inflation_layer"]["plugin"] == (
            "nav2_costmap_2d::InflationLayer"
        )

        scan = voxel_layer["scan"]
        assert scan["topic"] == "/scan"
        assert scan["data_type"] == "LaserScan"
        assert scan["marking"] is True
        assert scan["clearing"] is True

        people = voxel_layer["people"]
        assert people["topic"] == "/person_follow/obstacles"
        assert people["data_type"] == "PointCloud2"
        assert people["marking"] is True
        assert people["clearing"] is False
        assert people["observation_persistence"] <= 0.5
        assert people["expected_update_rate"] <= 1.0
        assert people["min_obstacle_height"] == 0.05
        assert people["max_obstacle_height"] == 2.0
        assert people["obstacle_min_range"] == 0.2
        assert people["obstacle_max_range"] == 4.0

    follow_path = config["controller_server"]["ros__parameters"]["FollowPath"]
    assert follow_path["max_vel_x"] == max_forward_speed
    assert follow_path["max_speed_xy"] == max_forward_speed
