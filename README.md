# Robot FollowGuard (AprilTag-only)

ROS 2 Humble person-following robot: a mobile base with 2D LiDAR follows one
person identified by YOLO detection plus an AprilTag (`tag36h11` ID `0`)
identity check, using SLAM Toolbox for mapping and Nav2 for motion.

This is the clean, AprilTag-only public release. It contains no ReID, color-band,
or experimental identity code paths.

## Layout

```text
.
  src/
    robot_autonomous_yolo_slam_nav2/  Follower stack (YOLO + AprilTag perception,
                                      metric association, Nav2 follow/recovery
                                      client, safety gate, zone guard,
                                      SLAM/Nav2 launch and configs, ESP32
                                      firmware under firmware_esp32/, URDF,
                                      meshes, RViz)
    kaiaai_msgs/                      Telemetry message definitions
                                      (JointPosVel, KaiaaiTelemetry2, WifiState)
    kaiaai_telemetry/                 LDS decoder that turns raw ESP32 lidar
                                      bytes into LaserScan messages on /scan
```

Everything under `build/`, `install/`, and `log/` is generated locally and
stays out of version control.

## Requirements

- ROS 2 Humble with colcon
- SLAM Toolbox and Nav2 bringup packages (`slam_toolbox`, `nav2_bringup`)
- Python: Ultralytics (YOLO), OpenCV with `aruco`/`apriltag` support, NumPy,
  SciPy, PyYAML, `cv_bridge`

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select kaiaai_msgs kaiaai_telemetry robot_autonomous_yolo_slam_nav2
source install/setup.bash
```

Rebuild, then stop and restart all package nodes before running again. Mixed
old and new nodes can disagree on follow and search behavior.

## Run

The main entry point is `person_follow.launch.py`. With SLAM enabled it
brings up SLAM/Nav2 through `slam_nav.launch.py` plus the full follower
chain (YOLO + AprilTag tracker, metric association, Nav2 follow and recovery
client, zone guard, camera calibration publisher, and camera TF):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=true
```

`use_slam` defaults to `true`. Set `use_slam:=false` for perception wiring
without SLAM/Nav2 bringup, which suits a stationary camera check with the
annotated overlay on `/person_follow/annotated`.

Motor output stays behind a fail-closed path. Nav2 `/cmd_vel` must flow
through exactly one `safety_gate_node.py` into `/cmd_vel_safe`, and only
`/cmd_vel_safe` may reach the motor bridge. Don't enable drive output
without that chain in place.

## How following works

Camera frames give detection boxes and identity only. Every distance comes
from LiDAR points projected into the target box. Fresh scans, valid TF, and
a fresh map gate each Nav2 goal, and the safety gate needs its own fresh
scan (0.25 s) before any forward motion.

Identity is AprilTag-only: one valid `tag36h11` ID `0` inside exactly one
person bounding box locks that person; the tag detector scans person ROIs
only, never the full frame. While the same track remains unambiguous the tag
detector is skipped; after confirmed loss a fresh tag sighting is required to
lock again. See `src/robot_autonomous_yolo_slam_nav2/README.md` for detail.

When the confirmed target is lost, the follower rides the live Nav2 goal
first, then searches in place through the Nav2 `/spin` action, capped at
two full revolutions. A confirmed sighting returns to following at once.

## YOLO weights

Model weights are intentionally not committed. `*.pt`/`*.pth` files are
ignored. Fetch `yolov8n.pt` through Ultralytics yourself and place it where
your config expects it (default
`src/robot_autonomous_yolo_slam_nav2/config/person_follow.yaml`
`yolo_target_node.model`). If you move the weights, update that path.

## Firmware

Low-level control lives in
`src/robot_autonomous_yolo_slam_nav2/firmware_esp32/robot.ino` (ESP32 +
LDS02RR lidar parser, wheel PID motor control, micro-ROS transport).

## Package guide

- [`src/robot_autonomous_yolo_slam_nav2/README.md`](src/robot_autonomous_yolo_slam_nav2/README.md)
