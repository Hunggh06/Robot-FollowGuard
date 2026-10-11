"""Validated loading of ROS CameraInfo calibration YAML files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .camera_calibration import CalibrationError


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    image_width: int
    image_height: int
    camera_matrix: tuple[float, ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    rectification_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    binning_x: int
    binning_y: int
    roi: dict[str, Any]


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{key} must be a positive integer")
    return value


def _matrix(payload: Any, rows: int, cols: int, label: str) -> tuple[float, ...]:
    if not isinstance(payload, dict):
        raise CalibrationError(f"{label} must be a matrix mapping")
    if payload.get("rows") != rows or payload.get("cols") != cols:
        raise CalibrationError(f"{label} must have shape {rows}x{cols}")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != rows * cols:
        raise CalibrationError(f"{label} has the wrong number of values")
    try:
        values = tuple(float(value) for value in data)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{label} contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise CalibrationError(f"{label} contains a non-finite value")
    return values


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """Load and validate a ROS camera_info_manager-compatible YAML file."""
    if not path:
        raise CalibrationError("calibration_file must not be empty")
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise CalibrationError(f"calibration file does not exist: {calibration_path}")
    try:
        payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CalibrationError(f"could not read calibration file: {calibration_path}") from error
    if not isinstance(payload, dict):
        raise CalibrationError("calibration YAML must contain a mapping")

    width = _positive_int(payload, "image_width")
    height = _positive_int(payload, "image_height")
    model = payload.get("distortion_model")
    if model not in {"plumb_bob", "rational_polynomial", "equidistant"}:
        raise CalibrationError(f"unsupported distortion model: {model!r}")
    camera_matrix = _matrix(payload.get("camera_matrix"), 3, 3, "camera_matrix")
    if camera_matrix[0] <= 0.0 or camera_matrix[4] <= 0.0:
        raise CalibrationError("camera_matrix focal lengths must be positive")
    distortion_payload = payload.get("distortion_coefficients")
    if not isinstance(distortion_payload, dict):
        raise CalibrationError("distortion_coefficients must be a matrix mapping")
    distortion_data = distortion_payload.get("data")
    if not isinstance(distortion_data, list):
        raise CalibrationError("distortion_coefficients must contain a data list")
    distortion = _matrix(
        distortion_payload,
        1,
        len(distortion_data),
        "distortion_coefficients",
    )
    if not distortion:
        raise CalibrationError("distortion_coefficients must not be empty")
    rectification = _matrix(payload.get("rectification_matrix"), 3, 3, "rectification_matrix")
    projection = _matrix(payload.get("projection_matrix"), 3, 4, "projection_matrix")
    binning_x = int(payload.get("binning_x", 0))
    binning_y = int(payload.get("binning_y", 0))
    if binning_x < 0 or binning_y < 0:
        raise CalibrationError("binning values must be non-negative")
    roi = payload.get("roi", {})
    if not isinstance(roi, dict):
        raise CalibrationError("roi must be a mapping")
    return CameraCalibration(
        image_width=width,
        image_height=height,
        camera_matrix=camera_matrix,
        distortion_model=str(model),
        distortion_coefficients=distortion,
        rectification_matrix=rectification,
        projection_matrix=projection,
        binning_x=binning_x,
        binning_y=binning_y,
        roi=dict(roi),
    )
