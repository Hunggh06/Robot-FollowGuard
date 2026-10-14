# Troubleshooting

## No confirmed target

Verify the phone publishes `/phone/camera/back/image`, the marker is
`tag36h11:0`, it is sharp and fully inside one detected person box, and the
annotated image shows the expected tracker state. Open
`/person_follow/annotated` with `rqt_image_view`.

## Target has no metric position

Check `/scan`, `/phone/camera/back/camera_info`, `/tf`, the calibration image
resolution, and the `base_link` to `phone_camera_back` transform. A visible
tag alone intentionally cannot generate a navigation goal.

## Robot does not move

Inspect the Nav2 action servers, `/person_follow/target`, and safety-gate
inputs. Confirm the required downstream path is Nav2 `/cmd_vel` → safety gate
→ `/cmd_vel_safe` → motor bridge. Stale scan data intentionally blocks forward
motion.

## Recovery keeps searching

Inspect `/person_follow/recovery_status` and `/yolo_target_node/observations`.
During an inspection pause, the status reason is `search_inspecting`. A tag
that is too small, blurred, outside a person crop, or ambiguous cannot lock a
target.
