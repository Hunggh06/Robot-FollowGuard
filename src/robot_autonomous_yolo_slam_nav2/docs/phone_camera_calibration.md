# Phone Camera Calibration

This package provides an offline OpenCV calibration utility for the phone image
topic. It writes a ROS `camera_info_manager`-compatible YAML file. It does not
publish `sensor_msgs/msg/CameraInfo` and it does not add launch wiring; a
separate runtime publisher will consume the generated file later.

## 1. Prepare the chessboard

The OpenCV pattern is **9 by 6 inner corners**. The printed board therefore
has **10 by 7 visible squares**. Keep the board flat and measure the edge of a
printed square with a ruler. Use `0.025` metres only when the measured square
is actually 25 mm; otherwise pass the measured value to `--square-size-m`.

Capture at least 10 to 20 usable views. Vary distance, image position, and
angle so that the board reaches the corners and edges of the phone image. Keep
the whole board visible, avoid motion blur and glare, and do not crop or resize
the images after capture. The utility requires three detected views at
minimum, but three views are not a good calibration dataset.

## 2. Capture real images

Start the phone image publisher and capture individual frames with the ROS 2
skill CLI. The exact topic is `/phone/camera/back/image`:

```bash
python3 /home/hungdo/.agents/skills/ros2-skill/scripts/ros2_cli.py topics capture-image \
  --topic /phone/camera/back/image \
  --output /home/hungdo/.agents/skills/ros2-skill/.artifacts/phone_camera_01.jpg
```

Repeat the command with distinct output names, for example
`phone_camera_02.jpg` through `phone_camera_20.jpg`. Copy only the real captured
images to the package calibration directory if desired:

```bash
mkdir -p /home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/images
cp /home/hungdo/.agents/skills/ros2-skill/.artifacts/phone_camera_*.jpg \
  /home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/images/
```

Do not create placeholder images or use guessed intrinsics when the camera is
not publishing. The capture command must receive real `sensor_msgs/msg/Image`
frames.

## 3. Run offline calibration

Run the utility with the package virtual environment. Quote the glob so the
utility, rather than the shell, expands it:

```bash
/home/hungdo/ros2_ws/.venv/bin/python \
  /home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/scripts/calibrate_phone_camera.py \
  --input-glob '/home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/images/*.jpg' \
  --output /home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/phone_camera_back.yaml \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size-m 0.025 \
  --camera-name phone_camera_back
```

The defaults are `--pattern-cols 9`, `--pattern-rows 6`, and `--square-size-m 0.025`, so the
options may be omitted only when they match the measured board. OpenCV's
`findChessboardCornersSB` is used when available, with the classic corner
detector and sub-pixel refinement as a fallback.

The output contains image dimensions, `camera_matrix` (`K`),
`distortion_coefficients` (`D`), `distortion_model: plumb_bob`, an identity
`rectification_matrix` (`R`), `projection_matrix` (`P`), zero binning, and an
empty ROI. Values are checked for finiteness before writing, and the file is
written atomically only after calibration succeeds.

If the glob matches no files, images cannot be read, dimensions differ, or too
few chessboards are detected, the command exits non-zero and does not create a
new YAML output. Existing output files are not replaced on those failures.

## 4. Keep the result separate from runtime wiring

Inspect the generated YAML before using it with a future `CameraInfo`
publisher. Keep the calibration tied to the same phone camera, resolution,
crop, and orientation used during capture. A resolution or crop change needs a
new calibration dataset. This utility does not start a publisher, rectify
images, or modify the package's launch files.
