# ROS 2 Person-Following Robot

This workspace holds a selectively published ROS 2 robot project. A mobile
base with 2D LiDAR follows one person, identified by YOLO detection plus a
selectable identity check (default red-above-blue leg bands, AprilTag
rollback), using SLAM Toolbox for mapping and Nav2 for motion.

It runs on ROS 2 Humble. The stack was cleaned for public sharing, so only
the packages needed to build and run the follower remain here.

## Retained layout

```text
ros2_ws/
  src/
    robot_autonomous_yolo_slam_nav2/  Follower stack (perception, metric
                                      association, Nav2 client, safety gate,
                                      SLAM/Nav2 launch and configs)
    kaiaai_msgs/                      Telemetry message definitions
                                      (JointPosVel, KaiaaiTelemetry2, WifiState)
    kaiaai_telemetry/                 LDS decoder that turns raw ESP32 lidar
                                      bytes into LaserScan messages on /scan
  firmware/
    robot_lowlevel_v2_3.ino           ESP32 low-level firmware (v2.3, lidar
                                      parser plus wheel PID motor control)
    wifi_credentials.example.h        Template for local WiFi credentials
  maps/
    kaia_map.pgm / .yaml
    stationary_map.pgm / .yaml
    wifi_map.pgm / .yaml
```

Everything under `build/`, `install/`, and `log/` is generated locally and
stays out of version control.

## Requirements

You need a working ROS 2 Humble install with colcon, plus SLAM Toolbox and
the Nav2 bringup packages.

Python nodes use the workspace virtualenv that contains Ultralytics for YOLO
inference.

## Build

Build from the workspace root. Package order matters, so build the message
package before the telemetry and follower packages:

```bash
cd /home/hungdo/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select kaiaai_msgs kaiaai_telemetry robot_autonomous_yolo_slam_nav2
source install/setup.bash
```

Rebuild, then stop and restart all package nodes before running again. Mixed
old and new nodes can disagree on follow and search behavior.

## Run

The main entry point is `person_follow.launch.py`. With SLAM enabled it
brings up SLAM/Nav2 through `slam_nav.launch.py` plus the full follower
chain (YOLO tracker, metric association, Nav2 follow and recovery client,
zone guard, camera calibration publisher, and camera TF):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=true
```

`use_slam` defaults to `true`. Set `use_slam:=false` for perception wiring
without SLAM/Nav2 bringup, which suits a stationary camera check with the
annotated overlay on `/person_follow/annotated`.

The launch ships a measured camera calibration and a static transform from
`base_link` to `phone_camera_back` by default. You can point at another
calibration file when needed:

```bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py \
  use_slam:=true \
  calibration_file:=/home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/phone_camera_back.yaml
```

Motor output stays behind a fail-closed path. Nav2 `/cmd_vel` must flow
through exactly one `safety_gate_node.py` into `/cmd_vel_safe`, and only
`/cmd_vel_safe` may reach the motor bridge. Don't enable drive output
without that chain in place.

## How following works

Camera frames give detection boxes and identity only. Every distance comes
from LiDAR points projected into the target box. Fresh scans, valid TF, and
a fresh map gate each Nav2 goal, and the safety gate needs its own fresh
scan (0.25 s) before any forward motion.

When the confirmed target is lost, the follower rides the live Nav2 goal
first, then searches in place through the Nav2 `/spin` action, capped at
two full revolutions. A confirmed sighting returns to following at once.

Identity selection lives in `yolo_target_node.identity_mode` (default
`color_bands`: three consecutive red-above-blue lower-leg frames lock the
target; tune `band_*` HSV/geometry in
`src/robot_autonomous_yolo_slam_nav2/config/person_follow.yaml` for the room
lighting). The AprilTag path stays fully wired for rollback with no behavior
change:

```bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py \
  use_slam:=true identity_mode:=apriltag
```

For full detail on topics, position sources (`lidar`, `held_invalid`,
`none`), target policy, recovery search, and the zone guard, read the
package guide:

- [`src/robot_autonomous_yolo_slam_nav2/README.md`](src/robot_autonomous_yolo_slam_nav2/README.md)

## Maps

`maps/` ships three saved occupancy maps in ROS map YAML plus PGM pairs:
`kaia_map`, `stationary_map`, and `wifi_map`. They load through the standard
map server configuration used by the SLAM/Nav2 launch. Record fresh maps
for new rooms instead of reusing these blindly, since resolution and origin
differ per file.

## YOLO weights

Model weights are intentionally not committed. `*.pt` files are ignored, so
`yolov8n.pt` won't appear in this repo even though the default config reads
it from the workspace root.

Fetch `yolov8n.pt` through Ultralytics yourself and place it where your
config expects it (default `/home/hungdo/ros2_ws/yolov8n.pt`). The tracker
runs a single detect-only model with boxes and no pose keypoints. If you
move the weights, update the `model` path in
`src/robot_autonomous_yolo_slam_nav2/config/person_follow.yaml` to match.

## Firmware setup

Low-level control lives in `firmware/robot_lowlevel_v2_3.ino`. It parses the
LDS02RR lidar stream, publishes telemetry for the `kaiaai_telemetry` bridge,
and drives both wheels with per-wheel PID plus feed-forward.

Real credentials are never committed. `firmware/wifi_credentials.h` is
ignored by git, and only the template ships. Set it up locally:

```bash
cp firmware/wifi_credentials.example.h firmware/wifi_credentials.h
```

Then edit `firmware/wifi_credentials.h` and fill in your SSID, password,
micro-ROS agent IP, and port. The `.ino` includes that header directly, so
the firmware won't compile until the copy exists.

## Tests

Each package carries a pytest suite. The follower's tests live under
`src/robot_autonomous_yolo_slam_nav2/test/` and cover metric association,
identity handling, the safety gate, spin search, zone logic, and launch
configuration. `kaiaai_telemetry` ships tests under its own `test/`
directory as well.

Run them through colcon after a build:

```bash
colcon test --packages-select kaiaai_msgs kaiaai_telemetry robot_autonomous_yolo_slam_nav2
colcon test-result --all
```

Keep tests green before sharing changes. They guard the safety behavior
that real hardware depends on.
