from robot_autonomous_yolo_slam_nav2.metric_association import (
    BaseToCameraTransform,
    CameraCalibration,
    project_point,
)


OPTICAL_ROTATION = ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
CAMERA_POSITION = (-0.055, 0.063, 0.177)


def test_person_in_front_of_robot_projects_inside_phone_image() -> None:
    transform = BaseToCameraTransform(OPTICAL_ROTATION, CAMERA_POSITION)
    calibration = CameraCalibration(fx=136.92, fy=136.78, cx=159.02, cy=118.77)

    pixel = project_point((2.0, 0.0, 0.5), calibration, transform, (320, 240))

    assert pixel is not None
    assert 0 <= pixel[0] < 320
    assert 0 <= pixel[1] < 240


def test_identity_rotation_would_push_frontal_person_off_image() -> None:
    identity = BaseToCameraTransform(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), CAMERA_POSITION)
    calibration = CameraCalibration(fx=136.92, fy=136.78, cx=159.02, cy=118.77)

    assert project_point((2.0, 0.0, 0.5), calibration, identity, (320, 240)) is None
