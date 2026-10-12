"""Offline OpenCV chessboard calibration and ROS CameraInfo YAML output."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import yaml

PatternSize: TypeAlias = tuple[int, int]
Matrix3: TypeAlias = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vector: TypeAlias = tuple[float, ...]
FloatArray: TypeAlias = npt.NDArray[np.float32]

DEFAULT_PATTERN_SIZE: Final[PatternSize] = (9, 6)
DEFAULT_SQUARE_SIZE_M: Final[float] = 0.025
MIN_DETECTION_COUNT: Final[int] = 3


class MatrixPayload(TypedDict):
    rows: int
    cols: int
    data: list[float]


class RoiPayload(TypedDict):
    x_offset: int
    y_offset: int
    height: int
    width: int
    do_rectify: bool


class CameraInfoPayload(TypedDict):
    image_width: int
    image_height: int
    camera_name: str
    camera_matrix: MatrixPayload
    distortion_model: str
    distortion_coefficients: MatrixPayload
    rectification_matrix: MatrixPayload
    projection_matrix: MatrixPayload
    binning_x: int
    binning_y: int
    roi: RoiPayload


@dataclass(frozen=True, slots=True)
class CalibrationError(Exception):
    """A calibration input, detection, or output validation failure."""

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Validated monocular calibration values in ROS CameraInfo units."""

    image_width: int
    image_height: int
    camera_matrix: Matrix3
    distortion_coefficients: Vector
    reprojection_error: float
    detection_count: int


def _validate_calibration_inputs(pattern_size: PatternSize, square_size_m: float) -> None:
    cols, rows = pattern_size
    if cols <= 0 or rows <= 0:
        raise CalibrationError("pattern dimensions must be positive")
    if not math.isfinite(square_size_m) or square_size_m <= 0.0:
        raise CalibrationError("square-size-m must be a finite positive length")


def _object_points(pattern_size: PatternSize, square_size_m: float) -> FloatArray:
    cols, rows = pattern_size
    points = [
        (column * square_size_m, row * square_size_m, 0.0)
        for row in range(rows)
        for column in range(cols)
    ]
    return np.asarray(points, dtype=np.float32)


def _valid_corners(corners: np.ndarray | None, expected_count: int) -> FloatArray | None:
    if corners is None:
        return None
    normalized = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if normalized.shape != (expected_count, 2) or not np.isfinite(normalized).all():
        return None
    return normalized.reshape(expected_count, 1, 2)


def _find_corners(image: np.ndarray, pattern_size: PatternSize) -> FloatArray | None:
    expected_count = pattern_size[0] * pattern_size[1]
    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    if find_sb is not None:
        found, corners = find_sb(
            image,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found:
            valid = _valid_corners(corners, expected_count)
            if valid is not None:
                return valid

    found, corners = cv2.findChessboardCorners(
        image,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return _valid_corners(refined, expected_count)


def _as_matrix3(matrix: np.ndarray) -> Matrix3:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise CalibrationError("OpenCV returned a non-finite or malformed camera matrix")
    return ((float(values[0, 0]), float(values[0, 1]), float(values[0, 2])), (float(values[1, 0]), float(values[1, 1]), float(values[1, 2])), (float(values[2, 0]), float(values[2, 1]), float(values[2, 2])))


def calibrate_images(
    image_paths: Sequence[Path],
    pattern_size: PatternSize = DEFAULT_PATTERN_SIZE,
    square_size_m: float = DEFAULT_SQUARE_SIZE_M,
) -> CalibrationResult:
    """Calibrate one camera from readable chessboard images."""
    _validate_calibration_inputs(pattern_size, square_size_m)
    if not image_paths:
        raise CalibrationError("input glob matched no images")

    object_template = _object_points(pattern_size, square_size_m)
    object_points: list[FloatArray] = []
    image_points: list[FloatArray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise CalibrationError(f"image has invalid dimensions: {image_path}")
        current_size = (width, height)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise CalibrationError("all calibration images must have identical dimensions")

        corners = _find_corners(image, pattern_size)
        if corners is not None:
            object_points.append(object_template.copy())
            image_points.append(corners)

    if image_size is None:
        raise CalibrationError("input glob contained no readable images")
    if len(image_points) < MIN_DETECTION_COUNT:
        raise CalibrationError(
            f"need at least {MIN_DETECTION_COUNT} detected chessboards; found {len(image_points)}"
        )

    try:
        rms, camera_matrix, distortion_coefficients, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )
    except cv2.error as error:
        raise CalibrationError(f"OpenCV calibration failed: {error}") from error

    if not math.isfinite(float(rms)) or rms < 0.0:
        raise CalibrationError("OpenCV returned an invalid reprojection error")
    coefficients = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size == 0 or not np.isfinite(coefficients).all():
        raise CalibrationError("OpenCV returned invalid distortion coefficients")
    return CalibrationResult(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=_as_matrix3(camera_matrix),
        distortion_coefficients=tuple(float(value) for value in coefficients),
        reprojection_error=float(rms),
        detection_count=len(image_points),
    )


def _finite_values(values: Sequence[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise CalibrationError(f"{label} contains a non-finite value")
    return result


def _matrix_payload(rows: int, cols: int, data: Sequence[float], label: str) -> MatrixPayload:
    values = _finite_values(data, label)
    if len(values) != rows * cols:
        raise CalibrationError(f"{label} has the wrong number of values")
    return {"rows": rows, "cols": cols, "data": values}


def camera_info_dict(result: CalibrationResult, camera_name: str) -> CameraInfoPayload:
    """Build a finite ROS camera_info_manager-compatible YAML mapping."""
    if result.image_width <= 0 or result.image_height <= 0:
        raise CalibrationError("image dimensions must be positive")
    if not camera_name.strip():
        raise CalibrationError("camera-name must not be empty")
    if result.detection_count < MIN_DETECTION_COUNT:
        raise CalibrationError("calibration has too few detected chessboards")
    if not math.isfinite(result.reprojection_error) or result.reprojection_error < 0.0:
        raise CalibrationError("reprojection error must be finite and non-negative")

    matrix_values = [value for row in result.camera_matrix for value in row]
    camera_matrix = _matrix_payload(3, 3, matrix_values, "camera matrix")
    distortion_coefficients = _matrix_payload(
        1,
        len(result.distortion_coefficients),
        result.distortion_coefficients,
        "distortion coefficients",
    )
    k = camera_matrix["data"]
    projection = (k[0], k[1], k[2], 0.0, k[3], k[4], k[5], 0.0, k[6], k[7], k[8], 0.0)
    return {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "camera_name": camera_name,
        "camera_matrix": camera_matrix,
        "distortion_model": "plumb_bob",
        "distortion_coefficients": distortion_coefficients,
        "rectification_matrix": _matrix_payload(
            3,
            3,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "rectification matrix",
        ),
        "projection_matrix": _matrix_payload(3, 4, projection, "projection matrix"),
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


def write_camera_info_yaml(result: CalibrationResult, camera_name: str, output_path: Path) -> None:
    """Atomically write a validated ROS CameraInfo YAML file."""
    payload = camera_info_dict(result, camera_name)
    parent = output_path.parent
    if not parent.is_dir():
        raise CalibrationError(f"output directory does not exist: {parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yaml.safe_dump(payload, temporary_file, sort_keys=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise CalibrationError(f"could not write calibration YAML: {output_path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
