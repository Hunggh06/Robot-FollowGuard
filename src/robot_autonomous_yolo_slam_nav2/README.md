# YOLO Person Follow

This package follows a single person confirmed by a `tag36h11` ID `0` AprilTag
inside that person's YOLO bounding box. It does not implement GUIDE_HOME or
return-home behavior.

## Inputs

- `/phone/camera/back/image` (`sensor_msgs/msg/Image`)
- `/phone/camera/back/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/person_follow/annotated` (`sensor_msgs/msg/Image`, for `rqt_image_view`)
- `/scan` and the required TF chain to `base_link`

The detector and CameraInfo must use frame `phone_camera_back`; metric association also requires a valid TF chain between `base_link` and `phone_camera_back`.

The phone bridge supplies the image but not CameraInfo. The launch uses the measured calibration shipped in this package by default. Do not use guessed intrinsics for LiDAR association.

## Position sources and provenance

`/person_follow/target` (`PersonTarget`) carries `position_source`
(`none` | `lidar` | `held_invalid`), a geometry-only
`position_confidence` (independent of person identity confidence), and a
machine-readable `provenance` string. `header.stamp` is the measurement
stamp of a live source, or the ORIGINAL stamp for `held_invalid`.
LiDAR positions retain the scan timestamp. Camera images are only used
for detection boxes and identity (YOLO + AprilTag); no distance is ever
estimated from the camera.

- `lidar`: fresh scan points projected inside the target box (highest trust).
- `held_invalid`: last valid point for the SAME confirmed track, decaying
  confidence, `valid=false`. Retained (not refreshed) across calibration,
  TF, skew, and staleness errors; cleared on track change, reset, timeout,
  or incompatible frames. Never a Nav2 goal, never reused for another ID.
- `none`: no geometry (also after ID change, reset, or cache timeout).

Person-in-box LiDAR missing (empty box, scan otherwise fine) reports no
geometry (`none`) while the people obstacle output still holds projected
non-target people (raw wall and leg returns excluded; the raw `/scan`
costmap path is unchanged), and the safety
gate still needs its own fresh scan (0.25 s) before forward motion. The
existing gate's reverse/in-place rotation policy is unchanged.

The old bearing-window fallback (nearest scan return within +-20 deg) is
no longer used for goals: a near wall is not person geometry.

After rebuilding this version, stop and restart all package nodes.

Following uses the existing detect-only model (`yolov8n.pt`). An additional
`yolov8n-pose.pt` model runs only while the zone guard requests foot-lift
confirmation. Pose never supplies distance or navigation geometry. The
`footpoint_*` message fields remain invalid and unused; all geometry is LiDAR-only.

## Launch

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=false
```

With the measured calibration:

```bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py \
  use_slam:=false \
  calibration_file:=/home/hungdo/ros2_ws/src/robot_autonomous_yolo_slam_nav2/calibration/phone_camera_back.yaml
```

The publisher stamps receipt time (host clock) and publishes frame `phone_camera_back`.
Phone and host clocks were measured ~1.3 s apart over WiFi, so copying the
phone image stamp would break every downstream timestamp comparison.
The corresponding TF from `base_link` to `phone_camera_back` must exist before
metric association can use the camera.

The launch starts the YOLO person tracker with conditional OpenCV AprilTag
detection inside person bounding boxes, metric target association, and the
Nav2 follow/recovery client.
`use_slam:=false` is the hardware-independent perception wiring mode; it does not start SLAM/Nav2 bringup.
`use_slam:=true` (the default) is the full entrypoint: it includes
`slam_nav.launch.py` for SLAM/Nav2 bringup plus the follower chain.
`slam_nav.launch.py` standalone defaults `start_perception_nodes:=false` so it
never duplicates the perception nodes or TF when included from here.
It also does not start a safety gate or motor bridge: any external Nav2 bringup
must provide exactly one `safety_gate_node.py` and `cmd_vel_bridge.py`, with
Nav2 `/cmd_vel` connected to the gate input and only `/cmd_vel_safe` connected
to the motor bridge. Do not enable motor output without that fail-closed path.

## Camera-only visual test

Enable the phone's back camera and run `person_follow.launch.py use_slam:=false`.
Stop any separately running motor/Nav2 stack before a stationary camera test.
Use the workspace Python environment containing Ultralytics:

```bash
cd /home/hungdo/ros2_ws
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=false
```

Then open the viewer:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select **`/person_follow/annotated`** for the combined overlay:

- Green bbox: selected **TARGET**, track ID and YOLO confidence.
- Blue/cyan bbox: other detected people, each with a track ID and confidence.
- Yellow polygon: AprilTag outline, family and tag ID.
- Top status: tracker state, selected target ID, person count, and
  `tag=CONFIRM/SKIP/SEARCH`.

All displayed person boxes come from the same YOLO inference/tracker used by following;
the conditional gesture model is not used for visualization. The output preserves the source
image timestamp and frame. `target=none` means no person was selected.
The overlay reports visual tracking, not permission to move: LiDAR, TF and
Nav2 safety checks still determine whether following is possible.

The default launch publishes only the combined `/person_follow/annotated` image.
The YOLO node always confirms identity with the AprilTag detector, run
in-process on person ROIs only; there is no identity topic and no full-frame
scan. ROI evidence search runs only when the identity needs confirmation; a
locked, unambiguous person is tracked without it.
The raw camera image remains available at `/phone/camera/back/image`.

Test with `tag36h11` ID 0 inside one person's bbox, then hide the tag while the
person remains visible, add a second person, and finally move the target out of
view. Check the green target box, `tag=SKIP` while locked, track IDs and state
in the combined image.

## Recovery search

On confirmed-target loss the follower keeps riding the live Nav2 follow
goal (its trajectory may still lead to the person; every terminal result
— success, cancel, abort — flows through the existing goal cleanup).
Only when no follow goal remains, it searches in place toward the last
seen bearing through the Nav2 `/spin` action client (no direct `/cmd_vel`,
no recovery `NavigateToPose`): at most one bounded `Spin.Goal.target_yaw`
in flight at a time, accumulating accepted yaw up to
`search_max_spin_revolutions` full turns, then stopping. A confirmed
sighting exits back to `following` at once and cancels any spin.

While a confirmed target is visible and inside the follow stand-off, the
follower sends a same-position Nav2 goal to turn and face it (no `/spin`,
no direct `/cmd_vel`).

While searching, fresh tracked people pause the spin so the existing
ROI-only AprilTag confirmation can run without the robot rotating away:
the in-flight `/spin` goal is cancelled and no new spin is sent for one
anchored window of `min(search_inspection_max_s,
person_count * search_inspection_per_person_s)` (2 s for one person, up to
10 s). The window never extends — not even while people stay visible — and
stale or zero people resume the spin at once. A tag lock returns to
`following` immediately and clears the pause. The pause never cancels the
live Nav2 follow goal, never bypasses the scan-freshness or revolution
cap, and never changes tag ROI or identity selection. Presence comes from
per-person `PersonObservation` stamps on `observations_topic`
(`reason=search_inspecting` on `/person_follow/recovery_status`).

Watch the state live:

```bash
ros2 topic echo /person_follow/recovery_status
```

Key config (`config/person_follow.yaml`, `nav2_person_follow`):

| Parameter | Default | Meaning |
|---|---|---|
| `spin_action_name` | `/spin` | Nav2 Spin action server name |
| `search_spin_default_rate_rad_s` | 0.7 | Spin rate without a speed estimate |
| `search_spin_min_rate_rad_s` | 0.51 | Spin rate floor |
| `search_spin_max_rate_rad_s` | 1.4 | Spin rate ceiling |
| `search_spin_deadband_rad` | 0.1 | Bearing deadband holding still |
| `search_max_spin_revolutions` | 2.0 | Total spin budget before stopping |
| `search_timeout_s` | 0.0 | Loss timeout before idling (0 disables) |
| `search_drive_duration_s` | 2.0 | Drive-status window after loss |
| `search_drive_speed_m_s` | 0.15 | Reserved drive speed (status only) |
| `observations_topic` | `/yolo_target_node/observations` | Fresh person presence for the inspection pause |
| `search_inspection_per_person_s` | 2.0 | Pause seconds per visible person |
| `search_inspection_max_s` | 10.0 | Pause cap (never extended) |
| `search_inspection_max_obs_age_s` | 0.5 | Presence freshness for the pause |

Isolated QA (unique domain, fake Nav2 server, motors disabled):

```bash
ROS_QA_DOMAIN_ID=208 ROS_LOCALHOST_ONLY=1 python3 /tmp/opencode/recovery_ros_qa.py
```

Floor-fallback QA (unique domain, synthetic camera + TF + scan, no motion):

```bash
ROS_QA_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1 python3 /tmp/opencode/floor_fallback_ros_qa.py
```

It steps one live `metric_target_node` through
`lidar -> camera -> camera_no_scan -> corridor_people_cloud ->
held_invalid -> other_id -> none`,
asserts `position_source` / `valid` / stamp semantics at each step, checks
the people cloud holds only the projected second person (raw wall returns
excluded), and
fails if anything is published on `/cmd_vel` or `/cmd_vel_safe`. It never
touches the default domain or any user ROS process. Use an isolated
domain >= 218 for verification runs:

```bash
ROS_QA_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 python3 /tmp/opencode/floor_fallback_ros_qa.py
```

## Red zone guard

`zone_guard_node.py` watches a hand-drawn red polygon (the SAFE area
boundary) and guides the person back on foot-lift confirmation. Geometry is
LiDAR-only; the camera only detects boxes/identity.

- Draw by clicking 4 corners straight on the map: RViz Publish-Point
  tool, click 4 points on `/clicked_point` (map frame, any order — they
  are auto-sorted around the center so no bowtie). Each click shows
  as a red sphere on `/person_follow/red_zone_clicks` with progress on
  `/person_follow/guide_status` (  `draw_clicks_N_of_4`); the 4th click
  auto-connects into the red polygon on `/person_follow/red_zone`. Reset: `ros2 service call
  /zone_guard/reset_zone std_srvs/srv/Trigger`. Preload via
  `zone_polygon_points: [x1, y1, ...]`.
- Person outside for `leaving_verify_s` (2 s) → loa
  "Bạn đã đi vào vùng cấm, quay đầu ngay lập tức". After `ask_delay_s`
  (5 s) → "Hãy nhấc một bàn chân lên khỏi mặt đất và giữ một giây để chuyển chế độ dẫn đường", repeated every
  `repeat_s` (15 s) while normal follow keeps running. Back inside
  cancels to armed.
- While waiting, the followed person lifts one foot clearly off the ground and
  holds it for at least **1 second**. Turn toward the camera so hips, knees, and
  both ankles are visible. The lower ankle is treated as the planted foot, and
  the other ankle must rise by a fraction of that leg's length; with only one
  leg visible, the ankle must rise above its knee. Low-confidence or ambiguous
  poses do not confirm. Other people's gestures are ignored, and overlapping
  people are rejected conservatively. Lowering the foot, losing identity/geometry,
  stale evidence, repeated/out-of-order camera timestamps, or a gap longer than
  0.5 s resets the hold.
- A completed hold → "Đã xác nhận, đi theo tôi để quay về vùng an toàn", `guide_active=true` (follow
  node yields: cancels once, publishes nothing), controller `max_vel_x`
  set to `guide_max_vel_x` (0.25, same as the follow speed), Nav2 drives to the polygon centroid
  (first corner if concave; `guide_return_override: true` uses the fixed
  `guide_return_*` point instead) regardless of the person. Speed is
  restored after. Redrawing mid-guide replans to the new centroid; three
  straight rejections hold position with `guide_goal_unreachable_hold`.
- On arrival → "Bạn đã quay trở lại vùng an toàn", then spin-verify
  (existing 2-revolution cap): person visible and inside releases follow
  mode; outside re-arms the announce cycle; unseen after the cap waits
  stopped until the person reappears.
- Live state: `/person_follow/guide_status` (`phase=... reason=...`).

Voice input on `/phone/speech_text` no longer confirms or changes mode.
Speaker announcements still work. Gestures during ordinary following or the
initial warning delay are ignored. Returning inside the zone cancels the request.
The gesture publisher reuses the exact image and selected track from the main
YOLO callback, runs pose at up to 5 Hz only while requested, and never selects
a new target. Missing model or uncertain evidence leaves normal following active
without switching to guide mode.

Gesture configuration in `config/person_follow.yaml`:

| Node / parameter | Default | Meaning |
|---|---|---|
| `yolo_target_node.hand_raise_model` | `/home/hungdo/ros2_ws/yolov8n-pose.pt` | Local COCO-17 pose weights; no automatic download |
| `yolo_target_node.hand_raise_hz` | `5.0` | Maximum gesture inference frequency |
| `yolo_target_node.hand_raise_keypoint_confidence` | `0.5` | Minimum leg joint confidence |
| `zone_guard.hand_raise_hold_s` | `1.0` | Continuous lifted-foot duration |
| `zone_guard.hand_raise_max_gap_s` | `0.5` | Maximum sample age and interval |

The local pose model must be present before testing. After changing the model
path or replacing weights, restart the nodes. To inspect the request and samples:

```bash
ros2 topic echo /person_follow/confirm_requested
ros2 topic echo /person_follow/hand_raise
```

`HandRaise` contains `header`, `track_id`, and `raised`; it is an observation,
not a motion command. Keep the lifted foot and both legs inside the camera image.
Rebuild and restart all package nodes after this interface change. Start with
a stationary, motors-disabled camera trial before supervised navigation.

## Safety QA

The package `slam_toolbox` config publishes `/map` every 2 s
(`map_update_interval: 2.0`), so recovery accepts snapshots for five intervals
(`recovery_max_map_age_s: 10.0`) without changing this package's SLAM tuning.

The slowdown band scales both forward speed and yaw rate by the same factor
to preserve the Nav2 turning radius. The factor varies continuously from 1 at
the outer band to `slow_scale` just above the stop distance; it is not a fixed
30% speed reduction throughout the band. The stop band still blocks forward
motion, and missing/stale scans are not treated as clear. Controller tuning,
wheel geometry, and the bridge RPM ramp are covered by this package's tests.

## Target policy

### Short scan gaps during navigation

`nav2_person_follow.scan_gap_hold_s` (0.8 s) retains an existing or pending
Nav2 goal, or its unconsumed result, across a brief scan gap. It does not issue
new goals or advance search using stale scan points. Fresh TF, the action server,
and a fresh map remain required; a longer gap cancels navigation. The elapsed
gap still counts toward the search budget when processing resumes.

This is separate from `safety_gate.max_scan_age_s` (0.25 s), which is unchanged:
the motor path still stops forward motion when scan evidence is stale. A short
gap can therefore pause physical movement without canceling and replanning the
whole search goal. `reason=scan_gap_hold_goal` identifies this state on
`/person_follow/recovery_status`.

`yolo_target_node` confirms identity only with AprilTag `tag36h11:0`; it has no
selectable identity mode. Downstream `PersonTarget` / `PersonObservation`,
metric association, Nav2, and zone guard are unchanged.

AprilTag identity policy:

1. YOLO detects people and the tracker assigns IDs.
2. While searching, OpenCV searches each person crop, enlarged up to 4x and
   capped at a 640-pixel longest side. Corners are mapped back to source pixels.
3. One valid `tag36h11:0` inside exactly one person bbox locks that person
   immediately; three consecutive visible-tag frames are no longer required.
4. While the same track remains unambiguous, the tag detector is skipped.
5. Tag search resumes for a missing track, a gap of 2 s or longer, abrupt
   bbox discontinuity, or substantial person-bbox overlap (IoU >= 0.3).
6. The combined image shows `tag=CONFIRM`, `tag=SKIP`, or `tag=SEARCH`;
   `tags=0` with `tag=SKIP` while locked is normal. A tag inside two
   overlapping boxes, duplicated target tags, a wrong ID, or a tag outside
   person boxes does not select a target.

```bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py use_slam:=false
```

Missing CameraInfo, TF, or all geometry still prevents metric follow goals;
visual identity confirmation does not bypass navigation safety checks.
Fresh LiDAR is strictly required for a metric position,
and forward motion still requires the safety gate's independent fresh scan.
Upscaling helps detector sampling but cannot recover unreadable tag details
or motion blur. Use a larger, sharply printed marker or a closer camera view
when the marker occupies only a few pixels. Changing phone resolution requires
matching camera calibration, not just increasing YOLO `imgsz`.

After rebuilding this version, stop and restart all package nodes because
the follow/search behavior changed (lost targets no longer cancel the
Nav2 goal; spin search is capped at two revolutions).
