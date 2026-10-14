from pathlib import Path

from ament_index_python.packages import get_package_prefix


PACKAGE_DIR = Path(__file__).parents[1]


def test_runtime_assets_are_present() -> None:
    expected_files = (
        PACKAGE_DIR / "config" / "ekf.yaml",
        PACKAGE_DIR / "config" / "nav2_params.yaml",
        PACKAGE_DIR / "config" / "nav2_params_slam.yaml",
        PACKAGE_DIR / "config" / "slam_toolbox.yaml",
        PACKAGE_DIR / "config" / "person_follow.yaml",
        PACKAGE_DIR / "launch" / "display.launch.py",
        PACKAGE_DIR / "launch" / "navigation.launch.py",
        PACKAGE_DIR / "launch" / "slam.launch.py",
        PACKAGE_DIR / "launch" / "slam_nav.launch.py",
        PACKAGE_DIR / "launch" / "person_follow.launch.py",
        PACKAGE_DIR / "README.md",
        PACKAGE_DIR / "urdf" / "xebamnguoi.urdf",
        PACKAGE_DIR / "rviz" / "display.rviz",
        PACKAGE_DIR / "rviz" / "slam.rviz",
        PACKAGE_DIR / "meshes" / "base_link.STL",
    )

    assert all(path.is_file() for path in expected_files)


def test_follow_wrapper_is_an_absolute_import_entrypoint() -> None:
    source = (PACKAGE_DIR / "scripts" / "nav2_person_follow.py").read_text()

    assert "from robot_autonomous_yolo_slam_nav2.nav2_person_follow import main" in source
    assert "from .nav2_person_follow" not in source


def test_follow_executable_is_installed_by_cmake() -> None:
    cmake_source = (PACKAGE_DIR / "CMakeLists.txt").read_text()

    assert "nav2_person_follow.py" in cmake_source


def test_reid_executable_is_not_installed_by_cmake() -> None:
    cmake_source = (PACKAGE_DIR / "CMakeLists.txt").read_text()

    assert "reid_target_node.py" not in cmake_source


def test_tag_detection_has_no_standalone_executable_in_cmake() -> None:
    cmake_source = (PACKAGE_DIR / "CMakeLists.txt").read_text()

    assert "apriltag_target_node.py" not in cmake_source
    assert "apriltag_debug_image.py" not in cmake_source


def test_yolo_executable_is_installed_by_cmake() -> None:
    cmake_source = (PACKAGE_DIR / "CMakeLists.txt").read_text()

    assert "yolo_target_node.py" in cmake_source


def test_follow_wrapper_is_installed_as_a_package_executable() -> None:
    executable = (
        Path(get_package_prefix("robot_autonomous_yolo_slam_nav2"))
        / "lib"
        / "robot_autonomous_yolo_slam_nav2"
        / "nav2_person_follow.py"
    )

    assert executable.is_file()
