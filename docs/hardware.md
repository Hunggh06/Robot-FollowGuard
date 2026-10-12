# Hardware and Setup

## Host and workspace setup

Install ROS 2 Humble with `colcon`, SLAM Toolbox, Nav2, and the dependencies
declared in `src/robot_autonomous_yolo_slam_nav2/package.xml`. Build from the
repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select kaiaai_msgs kaiaai_telemetry robot_autonomous_yolo_slam_nav2
source install/setup.bash
```

Set `yolo_target_node.model` in `config/person_follow.yaml` to a local
`yolov8n.pt`. Model weights are not distributed by this repository.

## Phone bridge

The configured input is `/phone/camera/back/image`. The companion Android app
is [jros2_cellphone_interface](https://github.com/Hunggh06/jros2_cellphone_interface).
That project documents camera, IMU, microphone-related, and other available
phone capabilities; FollowGuard source directly configures the back camera and
the optional speech topics only. Check its current branch/release before
assuming another phone sensor is connected here.

## ESP32 and LiDAR

`firmware_esp32/robot.ino` contains ESP32 firmware for LDS02RR UART telemetry,
micro-ROS communication, motor PID, and encoder state. It contains placeholder
Wi-Fi and Agent constants. Replace them locally; never commit credentials.
Start an Agent compatible with the firmware's configured micro-ROS transport
before bringing up ROS nodes.

## AprilTag

Print and mount a readable `tag36h11` marker with ID `0` on the intended
person. The marker must remain within the detected person crop for initial or
reacquisition confirmation.
