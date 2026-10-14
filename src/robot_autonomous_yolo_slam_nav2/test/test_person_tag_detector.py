import cv2
import numpy as np

from robot_autonomous_yolo_slam_nav2.person_tag_detector import PersonTagDetector, tagged_tracks
from robot_autonomous_yolo_slam_nav2.target_state import TrackId, TrackObservation


def tagged_image(tag_id=0):
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 24)
    image[100:124, 80:104] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def test_upscaled_person_crop_detects_tag_and_restores_original_pixel_coordinates():
    people = (TrackObservation(TrackId(7), 0.9, (30., 30., 140., 230.)),)
    tags = PersonTagDetector(upscale=4.0).detect(tagged_image(), people)
    assert len(tags) == 1 and tags[0].tag_id == 0
    xs = [point[0] for point in tags[0].corners]
    ys = [point[1] for point in tags[0].corners]
    assert 78. <= min(xs) <= 82. and 102. <= max(xs) <= 106.
    assert 98. <= min(ys) <= 102. and 122. <= max(ys) <= 126.
    assert tagged_tracks(people, tags, 0) == (TrackId(7),)


def test_tag_outside_people_is_not_used_for_confirmation():
    # Given a tag placed fully outside the only person bbox.
    people = (TrackObservation(TrackId(7), 0.9, (180., 30., 280., 230.)),)

    # When ROI-only detection runs over the frame.
    tags = PersonTagDetector().detect(tagged_image(), people)

    # Then no tag is reported and nothing can confirm a target.
    assert tags == ()
    assert tagged_tracks(people, tags, 0) == ()


def test_wrong_tag_id_cannot_confirm_target():
    people = (TrackObservation(TrackId(7), 0.9, (30., 30., 140., 230.)),)
    tags = PersonTagDetector().detect(tagged_image(5), people)
    assert tagged_tracks(people, tags, 0) == ()


def test_tag_inside_two_overlapping_people_is_rejected():
    people = (TrackObservation(TrackId(7), 0.9, (30., 30., 140., 230.)),
              TrackObservation(TrackId(12), 0.9, (50., 40., 160., 220.)))
    tags = PersonTagDetector().detect(tagged_image(), people)
    assert len(tags) == 1
    assert tagged_tracks(people, tags, 0) == ()


def test_detector_never_scans_without_people():
    # Given a frame holding a tag but zero person observations.
    # When detection runs with no person ROIs.
    tags = PersonTagDetector().detect(tagged_image(), ())

    # Then no tag is reported because there is no person ROI to search.
    assert tags == ()
