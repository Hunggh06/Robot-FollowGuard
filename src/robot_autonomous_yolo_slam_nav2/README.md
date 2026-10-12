# YOLO Person Follow (AprilTag-only)

This package follows a single person identified by YOLO detection plus an
AprilTag check (`tag36h11` ID `0`). AprilTag is the only identity method in
this release. It does not implement GUIDE_HOME or return-home behavior.

## Inputs

- `/phone/camera/back/image` (`sensor_msgs/msg/Image`)
- `/phone/camera/back/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/person_follow/annotated` (`sensor_msgs/msg/Image`, for `rqt_image_view`)
- `/scan` and the required TF chain to `base_link`

The detector and CameraInfo must use frame `phone_camera_back`; metric
association also requires a valid TF chain between `base_link` and
`phone_camera_back`.

The phone bridge supplies the image but not CameraInfo. The launch uses the
measured calibration shipped in this package by default. Do not use guessed
intrinsics for LiDAR association.

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
  calibration_file:=<workspace>/src/robot_autonomous_yolo_slam_nav2/calibration/phone_camera_back.yaml
```

The publisher stamps receipt time (host clock) and publishes frame
`phone_camera_back`. The corresponding TF from `base_link` to
`phone_camera_back` must exist before metric association can use the camera.

The launch starts the YOLO person tracker with conditional OpenCV AprilTag
detection inside person bounding boxes, metric target association, and the
Nav2 follow/recovery client.
`use_slam:=false` is the hardware-independent perception wiring mode; it does
not start SLAM/Nav2 bringup.
`use_slam:=true` (the default) is the full entrypoint: it includes
`slam_nav.launch.py` for SLAM/Nav2 bringup plus the follower chain.
`slam_nav.launch.py` standalone defaults `start_person_follow:=false` so it
never duplicates the follower nodes or TF when included from here.
It also does not start a safety gate or motor bridge: any external Nav2
bringup must provide exactly one `safety_gate_node.py` and
`cmd_vel_bridge.py`, with Nav2 `/cmd_vel` connected to the gate input and
only `/cmd_vel_safe` connected to the motor bridge. Do not enable motor
output without that fail-closed path.

## Camera-only visual test

Enable the phone's back camera and run
`person_follow.launch.py use_slam:=false`. Stop any separately running
motor/Nav2 stack before a stationary camera test. Use a Python environment
containing Ultralytics:

```bash
source /opt/ros/humble/setup.bash
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

All person boxes come from the same YOLO inference/tracker used by
following; no second YOLO model runs for visualization. The output preserves
the source image timestamp and frame. `target=none` means no person was
selected. The overlay reports visual tracking, not permission to move: LiDAR,
TF and Nav2 safety checks still determine whether following is possible.

The tag detector runs in-process on person ROIs only; there is no identity
topic and no full-frame scan. ROI evidence search runs only when the
identity needs confirmation; a locked, unambiguous person is tracked without
it. The raw camera image remains available at `/phone/camera/back/image`.

Test with tag ID 0 inside a person's bbox, then hide the tag while the
person remains visible, add a second person, and finally move the target out
of view. Check the green target box, `tag=SKIP` while locked, track IDs and
state in the combined image.

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
person_count * search_inspection_per_person_s)` (1 s for one person, up to
5 s). The window never extends — not even while people stay visible — and
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
| `search_inspection_per_person_s` | 1.0 | Pause seconds per visible person |
| `search_inspection_max_s` | 5.0 | Pause cap (never extended) |
| `search_inspection_max_obs_age_s` | 0.5 | Presence freshness for the pause |

## Red zone guard

`zone_guard_node.py` watches a hand-drawn red polygon (the SAFE area
boundary) and guides the person home on text confirmation. Geometry is
LiDAR-only; the camera only detects boxes/identity.

- Draw by clicking 4 corners straight on the map: RViz Publish-Point
  tool, click 4 points on `/clicked_point` (map frame, any order — they
  are auto-sorted around the center so no bowtie). Each click shows
  as a red sphere on `/person_follow/red_zone_clicks` with progress on
  `/person_follow/guide_status` (`draw_clicks_N_of_4`); the 4th click
  auto-connects into the red polygon on `/person_follow/red_zone`. Reset:
  `ros2 service call /zone_guard/reset_zone std_srvs/srv/Trigger`.
  Preload via `zone_polygon_points: [x1, y1, ...]`.
- Person outside for `leaving_verify_s` (2 s) → loa
  "Bạn đã đi vào vùng cấm, quay đầu ngay lập tức". After `ask_delay_s`
  (5 s) → "Hãy nói xác nhận để chuyển chế độ dẫn đường", repeated every
  `repeat_s` (15 s) while normal follow keeps running. Back inside
  cancels to armed.
- Text on `confirm_topic` (`/phone/speech_text` from the phone app, keyword
  "xác nhận") → "Đã xác nhận chuyển chế độ dẫn đường", `guide_active=true`
  (follow node yields: cancels once, publishes nothing), controller
  `max_vel_x` set to `guide_max_vel_x` (0.25, same as the follow speed),
  Nav2 drives to the polygon centroid (first corner if concave;
  `guide_return_override: true` uses the fixed `guide_return_*` point
  instead) regardless of the person. Speed is restored after. Redrawing
  mid-guide replans to the new centroid; three straight rejections hold
  position with `guide_goal_unreachable_hold`.
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
motion, and missing/stale scans are not treated as clear.

## Target policy

YOLO detects people and the tracker assigns IDs. AprilTag is the only
identity signal:

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
