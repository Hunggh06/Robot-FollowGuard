# Architecture

## Runtime boundaries

`person_follow.launch.py` starts the camera calibration publisher, a static
`base_link` to `phone_camera_back` transform, YOLO/AprilTag perception, metric
association, Nav2 follow/recovery, and zone guarding. With `use_slam:=true`,
it also includes `slam_nav.launch.py` for SLAM Toolbox and Nav2.

The perception node consumes `/phone/camera/back/image`. It detects people,
tracks boxes, and accepts only one `tag36h11:0` wholly attributable to one
person box. It publishes image-space target observations; the metric node
combines them with `/scan`, camera calibration, and TF. It publishes the
metric `/person_follow/target` used by Nav2 and the zone guard.

## Safety boundaries

The follow node does not publish direct motor commands. Nav2 output must pass
through `safety_gate_node.py`, which accepts `/cmd_vel`, publishes
`/cmd_vel_safe`, and applies its own fresh-scan check. The launch does not
start a motor bridge; deployments must provide exactly one safe downstream
bridge.

## Safe-zone boundary

The zone guard receives target position, scan data, and four user-selected map
points. It publishes zone/status visualizations and uses `/phone/speech_text`
and `/phone/speaker/say` only as configured interaction topics. It does not
derive person distance from the camera.
