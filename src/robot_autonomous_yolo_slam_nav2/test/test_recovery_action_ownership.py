from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from rclpy.task import Future

from robot_autonomous_yolo_slam_nav2.nav2_person_follow_actions import Nav2PersonFollowActions


def test_result_keeps_active_token_when_replacement_is_queued():
    events = []
    logger = SimpleNamespace(info=lambda message: None, warning=lambda message: None)
    owner = SimpleNamespace(
        _active_goal=None,
        _active_goal_token="nav:old",
        _owned_token="nav:replacement",
        _last_goal_mode="recovery",
        get_logger=lambda: logger,
        _on_nav2_result=lambda *event: events.append(event),
        _send_queued_goal=lambda: None,
    )
    future = Future()
    future.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
    Nav2PersonFollowActions._result_callback(owner, future)
    assert events == [(False, "recovery", "nav:old")]


def test_expected_cancellation_does_not_count_as_search_failure():
    events = []
    logger = SimpleNamespace(info=lambda message: None, warning=lambda message: None)
    owner = SimpleNamespace(
        _active_goal=None,
        _active_goal_token="nav:old",
        _last_goal_mode="recovery",
        get_logger=lambda: logger,
        _on_nav2_result=lambda *event: events.append(event),
        _send_queued_goal=lambda: None,
    )
    future = Future()
    future.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    Nav2PersonFollowActions._result_callback(owner, future)
    assert events == []
