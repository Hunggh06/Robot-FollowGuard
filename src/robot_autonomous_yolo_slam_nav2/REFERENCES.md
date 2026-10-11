# AprilTag Migration References

## Verified GitHub sources

- `christianrauch/apriltag_ros`: `README.md`, `cfg/tags_36h11.yaml`, `launch/camera_36h11.launch.yml`, `src/AprilTagNode.cpp`. The detector consumes rectified image plus `CameraInfo` and publishes `apriltag_msgs/msg/AprilTagDetectionArray` and tag TF.
- `SinfonIAUniandes/jros2_cellphone_interface`: `SensorBridge.kt`, `DualCameraSensor.kt`, `ImuSensor.kt`, `README.md`. The phone publishes `/phone/camera/back/image` as `sensor_msgs/msg/Image`, but does not publish `CameraInfo`.
- `ihmcrobotics/jros2`: `README.md`, `android/`, `examples/ros2-opencv-webcam/`, `scripts/run_webcam_publisher.sh`, Android Support wiki. This is Android/DDS transport infrastructure, not a camera-calibration or AprilTag API.
- `toyof/edge-ai-robotics-showcase`: `docs/architecture_detail.md`, `docs/robot_architecture.md`, `docs/mode_details.md`, `docs/development_guide.md`. These document camera rectification -> AprilTag -> TF/Nav2 patterns; implementation code is private and the docs disagree on the camera-info topic.
- Toyof article: `https://note.com/toyof/n/n6b55c89cf808?hl=en`. It is architecture evidence only and does not define local message or launch contracts.

## Local contract

The full entrypoint is `person_follow.launch.py` (default `use_slam:=true`
includes `slam_nav.launch.py` for SLAM/Nav2 plus the follower chain;
`slam_nav.launch.py` standalone keeps `start_person_follow:=false`):

```text
/phone/camera/back/image
        + /phone/camera/back/camera_info
        -> yolo_target_node (person ROI + conditional tag36h11 ID 0)
        -> /yolo_target_node/target
        -> metric_target_node (LiDAR + CameraInfo + TF)
        -> /person_follow/target
        -> nav2_person_follow (NavigateToPose + active search)
```

The phone image was verified as `sensor_msgs/msg/Image`, about 20 Hz, frame `phone_camera_back`. The launch uses the measured calibration shipped in this package; no fake calibration is generated here.

The local launch consumes an externally started detector and requires TF connectivity between `phone_camera_back` and `base_link`.

## Recovery search inspirations (principles only, no copied code)

- `D-Robotics/tros_person_following` (Apache-2.0): lost-target behavior that
  keeps a short hold, then re-searches likely positions instead of blindly
  spinning. Only the hold-then-directed-search structure informed
  `active_search_logic.py` / `follow_recovery_logic.py`; no source was copied.
- `erib001/autonomous_robot_following` (license unverified — treated as
  all-rights-reserved): person-following search-state structure. Only the
  general idea of distinct navigate/observe/search states informed the
  GRACE/SEARCH_NAV/SEARCH_SCAN phase split; no source was copied or derived.

All recovery code in this package is original. The AprilTag identity policy
(single `tag36h11` ID 0, confirmed-identity gate, no nearest-stranger
fallback) is unchanged by the recovery replacement.

## Scope

The adapter accepts only `tag36h11` ID `0`, rejects stale/ambiguous detections, and fails closed on missing calibration or TF. GUIDE_HOME, return-home, TTS, gestures, QR, and search-to-home are not part of this phase.

`apriltag_ros` and `apriltag_msgs` were not installed in the current Humble environment when this migration was prepared. The local package therefore keeps the detector message import at the runtime boundary and does not invent a replacement detector API.
