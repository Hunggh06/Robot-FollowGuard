from robot_autonomous_yolo_slam_nav2.tag_person_identity import TagPersonIdentity
from robot_autonomous_yolo_slam_nav2.target_state import TargetState, TrackId, TrackObservation


def person(track_id=7, bbox=(20., 30., 120., 220.)):
    return TrackObservation(TrackId(track_id), 0.9, bbox)


def test_single_tag_confirmation_locks_immediately():
    selector = TagPersonIdentity()
    result = selector.update((person(),), (TrackId(7),), 1_000_000_000)
    assert result.state is TargetState.TRACKING
    assert result.target_observation == person()
    assert result.forward_motion_allowed


def test_no_tag_never_selects_first_person():
    selector = TagPersonIdentity()
    result = selector.update((person(),), (), 1_000_000_000)
    assert result.target_observation is None
    assert result.state is TargetState.SEARCHING


def test_hidden_tag_does_not_drop_confirmed_person_or_trigger_tag_search():
    selector = TagPersonIdentity()
    selector.update((person(),), (TrackId(7),), 1_000_000_000)
    for index in range(1, 100):
        now = 1_000_000_000 + index * 100_000_000
        assert not selector.needs_confirmation((person(),), now)
        result = selector.update((person(),), (), now)
        assert result.target_observation == person()


def test_separate_second_person_does_not_trigger_tag_search_or_switch():
    selector = TagPersonIdentity()
    selector.update((person(),), (TrackId(7),), 1_000_000_000)
    people = (person(12, (180., 30., 280., 220.)), person())
    assert not selector.needs_confirmation(people, 1_100_000_000)
    assert selector.update(people, (), 1_100_000_000).target_observation == person()


def test_overlapping_people_require_tag_reconfirmation_until_resolved():
    selector = TagPersonIdentity()
    selector.update((person(),), (TrackId(7),), 1_000_000_000)
    people = (person(), person(12, (40., 30., 140., 220.)))
    assert selector.needs_confirmation(people, 1_100_000_000)
    assert selector.update(people, (), 1_100_000_000).target_observation is None
    assert selector.needs_confirmation((person(),), 1_200_000_000)
    assert selector.update((person(),), (TrackId(7),), 1_200_000_000).target_observation == person()


def test_lost_track_can_relock_new_track_only_with_tag():
    selector = TagPersonIdentity(lost_frames=2)
    selector.update((person(),), (TrackId(7),), 1_000_000_000)
    selector.update((), (), 1_100_000_000)
    assert selector.update((), (), 1_200_000_000).state is TargetState.LOST
    assert selector.update((person(20),), (), 1_300_000_000).target_observation is None
    assert selector.update((person(20),), (TrackId(20),), 1_400_000_000).target_observation == person(20)


def test_long_camera_gap_requires_confirmation_even_if_track_id_is_same():
    selector = TagPersonIdentity(lost_timeout_s=2.0)
    selector.update((person(),), (TrackId(7),), 1_000_000_000)
    assert selector.needs_confirmation((person(),), 4_000_000_000)
    assert selector.update((person(),), (), 4_000_000_000).target_observation is None


def test_multiple_tag_candidates_do_not_guess_target():
    selector = TagPersonIdentity()
    assert selector.update((person(), person(12)), (TrackId(7), TrackId(12)), 1_000_000_000).target_observation is None


def test_sub_two_second_absence_retains_identity_without_tag():
    # Given a tag-locked target at t0 with lost_timeout_s=2.0.
    selector = TagPersonIdentity(lost_timeout_s=2.0)
    selector.update((person(),), (TrackId(7),), 1_000_000_000)

    # When the target is absent for 1.9s and the same track reappears untagged.
    selector.update((), (), 2_900_000_000)
    assert not selector.needs_confirmation((person(),), 2_900_000_000)
    result = selector.update((person(),), (), 2_900_000_000)

    # Then the lock is retained without any tag evidence.
    assert result.state is TargetState.TRACKING
    assert result.target_observation == person()


def test_two_second_absence_clears_identity_until_fresh_tag():
    # Given a tag-locked target at t0 with lost_timeout_s=2.0.
    selector = TagPersonIdentity(lost_timeout_s=2.0)
    selector.update((person(),), (TrackId(7),), 1_000_000_000)

    # When the target is absent for exactly 2.0s and reappears untagged.
    selector.update((), (), 3_000_000_000)
    assert selector.needs_confirmation((person(),), 3_000_000_000)
    cleared = selector.update((person(),), (), 3_000_000_000)

    # Then the lock is cleared and no untagged person relocks it.
    assert cleared.target_observation is None

    # When a fresh unique tag ID 0 association arrives for the same track.
    relocked = selector.update((person(),), (TrackId(7),), 3_100_000_000)

    # Then the target relocks immediately.
    assert relocked.state is TargetState.TRACKING
    assert relocked.target_observation == person()
    assert relocked.forward_motion_allowed
