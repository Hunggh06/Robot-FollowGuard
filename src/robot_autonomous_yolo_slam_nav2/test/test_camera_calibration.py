import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

PACKAGE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from robot_autonomous_yolo_slam_nav2.camera_calibration import (  # noqa: E402
    CalibrationError,
    CalibrationResult,
    calibrate_images,
    camera_info_dict,
    write_camera_info_yaml,
)


def _result() -> CalibrationResult:
    return CalibrationResult(
        image_width=640,
        image_height=480,
        camera_matrix=(
            (500.0, 0.0, 320.0),
            (0.0, 500.0, 240.0),
            (0.0, 0.0, 1.0),
        ),
        distortion_coefficients=(0.1, -0.02, 0.001, 0.002, 0.0),
        reprojection_error=0.25,
        detection_count=3,
    )


def test_camera_info_dict_has_ros_camera_info_shape_and_finite_values() -> None:
    payload = camera_info_dict(_result(), "phone_camera_back")

    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert payload["camera_name"] == "phone_camera_back"
    assert payload["distortion_model"] == "plumb_bob"
    assert payload["camera_matrix"] == {
        "rows": 3,
        "cols": 3,
        "data": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
    }
    assert payload["distortion_coefficients"] == {
        "rows": 1,
        "cols": 5,
        "data": [0.1, -0.02, 0.001, 0.002, 0.0],
    }
    assert payload["rectification_matrix"] == {
        "rows": 3,
        "cols": 3,
        "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    assert payload["projection_matrix"]["rows"] == 3
    assert payload["projection_matrix"]["cols"] == 4
    assert payload["binning_x"] == 0
    assert payload["binning_y"] == 0
    assert payload["roi"] == {
        "x_offset": 0,
        "y_offset": 0,
        "height": 0,
        "width": 0,
        "do_rectify": False,
    }


def test_camera_info_yaml_round_trip_is_ros_compatible(tmp_path: Path) -> None:
    output_path = tmp_path / "phone_camera.yaml"

    write_camera_info_yaml(_result(), "phone_camera_back", output_path)

    payload = yaml.safe_load(output_path.read_text())
    assert payload["camera_name"] == "phone_camera_back"
    assert payload["camera_matrix"]["data"][0] == 500.0
    assert payload["projection_matrix"]["data"] == [
        500.0,
        0.0,
        320.0,
        0.0,
        0.0,
        500.0,
        240.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def test_nonfinite_calibration_is_rejected_without_output(tmp_path: Path) -> None:
    invalid = CalibrationResult(
        image_width=640,
        image_height=480,
        camera_matrix=((np.nan, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0,),
        reprojection_error=0.25,
        detection_count=3,
    )
    output_path = tmp_path / "invalid.yaml"

    with pytest.raises(CalibrationError):
        write_camera_info_yaml(invalid, "phone_camera_back", output_path)

    assert not output_path.exists()


def test_calibration_uses_sb_and_passes_meter_object_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_paths = tuple(tmp_path / f"image_{index}.jpg" for index in range(3))
    for image_path in image_paths:
        image_path.touch()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.zeros((54, 1, 2), dtype=np.float32)
    captured_object_points: list[np.ndarray] | None = None
    captured_image_size: tuple[int, int] | None = None

    monkeypatch.setattr(cv2, "imread", lambda *_args: image)
    monkeypatch.setattr(
        cv2,
        "findChessboardCornersSB",
        lambda _image, pattern_size, flags=0: (pattern_size == (9, 6), corners),
    )

    def fake_calibrate(
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        image_size: tuple[int, int],
        _camera_matrix: None,
        _distortion_coefficients: None,
    ) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        nonlocal captured_object_points, captured_image_size
        captured_object_points = object_points
        captured_image_size = image_size
        return 0.25, _result_matrix(), np.zeros((1, 5), dtype=np.float64), (), ()

    monkeypatch.setattr(cv2, "calibrateCamera", fake_calibrate)

    result = calibrate_images(image_paths, (9, 6), 0.025)

    assert captured_object_points is not None
    assert captured_image_size == (640, 480)
    object_points = captured_object_points
    assert object_points[0].shape == (54, 3)
    assert object_points[0][1].tolist() == pytest.approx([0.025, 0.0, 0.0])
    assert object_points[0][9].tolist() == pytest.approx([0.0, 0.025, 0.0])
    assert result.detection_count == 3


def test_calibration_falls_back_when_sb_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_paths = tuple(tmp_path / f"image_{index}.jpg" for index in range(3))
    for image_path in image_paths:
        image_path.touch()
    image = np.zeros((480, 640), dtype=np.uint8)
    corners = np.zeros((54, 1, 2), dtype=np.float32)

    monkeypatch.delattr(cv2, "findChessboardCornersSB", raising=False)
    monkeypatch.setattr(cv2, "imread", lambda *_args: image)
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args, **_kwargs: (True, corners))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda *_args: corners)
    monkeypatch.setattr(
        cv2,
        "calibrateCamera",
        lambda *_args: (0.25, _result_matrix(), np.zeros((1, 5)), (), ()),
    )

    result = calibrate_images(image_paths, (9, 6), 0.025)

    assert result.image_width == 640


def test_calibration_rejects_too_few_detections_without_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_paths = tuple(tmp_path / f"image_{index}.jpg" for index in range(2))
    for image_path in image_paths:
        image_path.touch()
    image = np.zeros((480, 640), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imread", lambda *_args: image)
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *_args, **_kwargs: (False, None))
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args, **_kwargs: (False, None))

    with pytest.raises(CalibrationError):
        calibrate_images(image_paths, (9, 6), 0.025)


def test_calibration_rejects_mismatched_input_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_paths = tuple(tmp_path / f"image_{index}.jpg" for index in range(3))
    for image_path in image_paths:
        image_path.touch()
    images = iter(
        (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),
        )
    )
    monkeypatch.setattr(cv2, "imread", lambda *_args: next(images))

    with pytest.raises(CalibrationError, match="identical dimensions"):
        calibrate_images(image_paths, (9, 6), 0.025)


def test_cli_fails_without_creating_output_for_empty_glob(tmp_path: Path) -> None:
    output_path = tmp_path / "phone_camera.yaml"
    environment = os.environ | {"PYTHONPATH": str(PACKAGE_DIR)}

    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_DIR / "scripts" / "calibrate_phone_camera.py"),
            "--input-glob",
            str(tmp_path / "missing" / "*.jpg"),
            "--output",
            str(output_path),
            "--camera-name",
            "phone_camera_back",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not output_path.exists()


def _result_matrix() -> np.ndarray:
    return np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
