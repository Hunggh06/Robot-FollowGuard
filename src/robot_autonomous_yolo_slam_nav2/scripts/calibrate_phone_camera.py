#!/usr/bin/env python3
"""Calibrate the phone camera from a glob of chessboard images."""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="glob matching calibration images")
    parser.add_argument("--output", required=True, type=Path, help="output ROS CameraInfo YAML")
    parser.add_argument(
        "--pattern-cols",
        "--cols",
        dest="cols",
        type=_positive_int,
        default=9,
        help="inner corners across the board",
    )
    parser.add_argument(
        "--pattern-rows",
        "--rows",
        dest="rows",
        type=_positive_int,
        default=6,
        help="inner corners down the board",
    )
    parser.add_argument(
        "--square-size-m",
        type=_positive_float,
        default=0.025,
        help="measured square edge length in metres",
    )
    parser.add_argument("--camera-name", required=True, help="ROS CameraInfo camera_name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline calibration command."""
    from robot_autonomous_yolo_slam_nav2.camera_calibration import (
        CalibrationError,
        calibrate_images,
        write_camera_info_yaml,
    )

    args = _parser().parse_args(argv)
    image_paths = tuple(Path(path) for path in sorted(glob.glob(args.input_glob)))
    try:
        result = calibrate_images(image_paths, (args.cols, args.rows), args.square_size_m)
        write_camera_info_yaml(result, args.camera_name, args.output)
    except CalibrationError as error:
        print(f"calibration failed: {error}", file=sys.stderr)
        return 1
    print(
        f"calibrated {result.detection_count} images; "
        f"reprojection error {result.reprojection_error:.6f} px; wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
