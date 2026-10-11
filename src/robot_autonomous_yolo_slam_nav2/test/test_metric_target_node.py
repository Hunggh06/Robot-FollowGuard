import runpy
from pathlib import Path


def test_ros_executable_wrapper_imports_package_entrypoint() -> None:
    # Given the installed ROS executable is launched as a standalone script.
    wrapper = Path(__file__).parents[1] / "scripts" / "metric_target_node.py"

    # When the wrapper is loaded without invoking its main function.
    namespace = runpy.run_path(str(wrapper), run_name="metric_target_node_wrapper")

    # Then it exposes the package entrypoint without relative-import failure.
    assert callable(namespace["main"])
