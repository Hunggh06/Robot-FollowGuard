# YOLO Person Follow

This package follows a single person selected by `identity_mode` (default
`color_bands`: red-above-blue leg bands; rollback `apriltag` keeps
`tag36h11` ID `0`). It does not implement GUIDE_HOME or return-home behavior.

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

YOLO runs a single detect-only model (`yolov8n.pt`, boxes only, no pose
keypoints). The `footpoint_*` message fields are always invalid and unused;
all geometry is LiDAR-only.

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
`slam_nav.launch.py` standalone defaults `start_person_follow:=false` so it
never duplicates the follower nodes or TF when included from here.
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
- Red box + blue box: accepted red-above-blue leg-band components (`color_bands` mode).
- Yellow polygon: AprilTag outline, family and tag ID (`apriltag` mode).
- Top status: tracker state, selected target ID, person count, `bands=CONFIRM/SKIP/SEARCH`
  and `tag=CONFIRM/SKIP/SEARCH` (only the selected mode reports `CONFIRM`).

All person boxes come from the same YOLO inference/tracker used by following;
no second YOLO model runs for visualization. The output preserves the source
image timestamp and frame. `target=none` means no person was selected.
The overlay reports visual tracking, not permission to move: LiDAR, TF and
Nav2 safety checks still determine whether following is possible.

The default launch publishes only the combined `/person_follow/annotated` image.
The YOLO node runs exactly one identity mode per frame (`identity_mode`:
`color_bands` by default, `apriltag` for rollback). Both detectors run
in-process on person ROIs only; there is no identity topic and no full-frame
scan. ROI evidence search runs only when the selected identity needs
confirmation; a locked, unambiguous person is tracked without it.
The raw camera image remains available at `/phone/camera/back/image`.

Test with red-above-blue bands on one lower leg, then hide the bands while the
person remains visible, add a second person, and finally move the target out of
view. Check the green target box, the red/blue component boxes, `bands=SKIP`
while locked, track IDs and state in the combined image. For the tag rollback,
launch with `identity_mode:=apriltag` and test with ID 0 inside a person's bbox
the same way (`tag=SKIP` while locked).

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
boundary) and guides the person home on text confirmation. Geometry is
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
  (5 s) → "Hãy nói xác nhận để chuyển chế độ dẫn đường", repeated every
  `repeat_s` (15 s) while normal follow keeps running. Back inside
  cancels to armed.
- Text on `confirm_topic` (`/phone/speech_text` from the phone app, keyword "xác nhận")
  → "Đã xác nhận chuyển chế độ dẫn đường",   `guide_active=true` (follow
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

`yolo_target_node.identity_mode` selects one detector/identity per frame
(`color_bands` default; unknown values log and fall back to `apriltag`).
Only the selected mode runs its evidence detector; `color_bands` never scans
tags and `apriltag` never scans color bands. Downstream `PersonTarget` /
`PersonObservation`, metric association, Nav2, and zone guard are unchanged.

1. YOLO detects people and the tracker assigns IDs.
2. `color_bands`: OpenCV searches the lower `band_leg_roi_fraction` (0.5) of
   each person bbox for red above blue on the same leg (horizontal centers
   within `band_max_dx_ratio` of bbox width, vertical separation above
   `band_min_sep_ratio` of leg ROI height, each component above
   `band_min_area_px` and each color ratio below `band_max_color_ratio`).
3. `color_bands`: exactly one person with valid ordered bands over
   `band_confirm_frames` (3) consecutive frames locks that person.
4. While the same track remains unambiguous, the band detector is skipped.
   Turning away or covering the bands alone does not drop the target.
5. Band search resumes for a missing track, a gap of 2 s or longer, abrupt
   bbox discontinuity, or substantial person-bbox overlap (IoU >= 0.3).
   Ambiguous or lost identity never falls back to the nearest person.
6. After confirmed loss, a new tracker ID needs fresh three-frame band
   confirmation before it can be accepted as the target. Metric association
   preserves the `PersonTarget.identity_confirmed` flag for the Nav2 consumer.

The combined image shows `bands=CONFIRM`, `bands=SKIP`, or `bands=SEARCH`.
`bands=0` with a green TARGET and `bands=SKIP` is normal: the identity is already
locked. Blue above red, a missing color, excessive horizontal separation, or a
floor-sized color region does not select a target.

Tuning (`config/person_follow.yaml`, `yolo_target_node`): start with the
conservative HSV/geometry defaults (`band_leg_roi_fraction: 0.5`,
`band_confirm_frames: 3`, `band_min_area_px: 100.0`,
`band_max_color_ratio: 0.4`, `band_min_sep_ratio: 0.25`,
`band_max_dx_ratio: 0.3`, red dual-hue `[0, 10]` + `[170, 179]` sat/val
`70/50`, blue hue `[90, 130]` sat/val `70/50`) and adjust the HSV bounds and
`band_min_area_px` for the actual camera lighting and band size first; widen
`band_max_dx_ratio` only when both bands are confirmed on one leg but rejected
for horizontal offset.

Rollback (`apriltag` mode, behavior unchanged):

1. While searching, OpenCV searches each person crop, enlarged up to 4x and
   capped at a 640-pixel longest side. Corners are mapped back to source pixels.
2. One valid `tag36h11:0` inside exactly one person bbox locks that person
   immediately; three consecutive visible-tag frames are no longer required.
3. While the same track remains unambiguous, the tag detector is skipped.
4. Tag search resumes for a missing track, a gap of 2 s or longer, abrupt
   bbox discontinuity, or substantial person-bbox overlap (IoU >= 0.3).
5. The combined image shows `tag=CONFIRM`, `tag=SKIP`, or `tag=SEARCH`;
   `tags=0` with `tag=SKIP` while locked is normal. A tag inside two
   overlapping boxes, duplicated target tags, a wrong ID, or a tag outside
   person boxes does not select a target.

```bash
ros2 launch robot_autonomous_yolo_slam_nav2 person_follow.launch.py \
  use_slam:=false identity_mode:=apriltag
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
