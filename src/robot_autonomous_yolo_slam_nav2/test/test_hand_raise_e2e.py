from pathlib import Path
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.task import Future
from std_msgs.msg import Bool, String
from tf2_ros import StaticTransformBroadcaster

from robot_autonomous_yolo_slam_nav2.msg import HandRaise, PersonTarget

sys.path.insert(0, str(Path(__file__).parents[1]))

from robot_autonomous_yolo_slam_nav2.zone_guard_runtime import ZoneGuardRuntime


def test_ros_gesture_confirmation_reaches_nav2_without_voice_or_motor_output() -> None:
    rclpy.init(args=[])
    driver = Node("hand_raise_surface_qa")
    zone = ZoneGuardRuntime()
    executor = SingleThreadedExecutor()
    executor.add_node(driver)
    executor.add_node(zone)
    goal_release: Future = Future()
    goals: list[NavigateToPose.Goal] = []
    states: list[str] = []
    requests: list[bool] = []
    active: list[bool] = []
    motion: list[Twist] = []

    async def execute(goal_handle):
        goals.append(goal_handle.request)
        await goal_release
        goal_handle.succeed()
        return NavigateToPose.Result()

    def get_speed(request, response):
        response.values = [ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.25)
                           for _ in request.names]
        return response

    def set_speed(request, response):
        response.results = [SetParametersResult(successful=True) for _ in request.parameters]
        return response

    def on_status(message: String) -> None:
        states.append(message.data)

    def on_request(message: Bool) -> None:
        requests.append(message.data)

    def on_active(message: Bool) -> None:
        active.append(message.data)

    server = ActionServer(driver, NavigateToPose, "/navigate_to_pose", execute)
    driver.create_service(GetParameters, "/controller_server/get_parameters", get_speed)
    driver.create_service(SetParameters, "/controller_server/set_parameters", set_speed)
    driver.create_subscription(String, "/person_follow/guide_status", on_status, 10)
    driver.create_subscription(Bool, "/person_follow/confirm_requested", on_request, 10)
    driver.create_subscription(Bool, "/person_follow/guide_active", on_active, 10)
    driver.create_subscription(Twist, "/cmd_vel", motion.append, 10)
    driver.create_subscription(Twist, "/cmd_vel_safe", motion.append, 10)
    target_pub = driver.create_publisher(PersonTarget, "/person_follow/target", 5)
    gesture_pub = driver.create_publisher(HandRaise, "/person_follow/hand_raise", 5)
    speech_pub = driver.create_publisher(String, "/phone/speech_text", 5)
    clicks_pub = driver.create_publisher(PointStamped, "/clicked_point", 10)
    broadcaster = StaticTransformBroadcaster(driver)
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "base_link"
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)
    zone.set_parameters([
        Parameter("leaving_verify_s", value=0.1),
        Parameter("ask_delay_s", value=0.1),
    ])

    def spin_until(predicate, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not predicate():
            assert time.monotonic() < deadline, f"Timed out; latest states: {states[-5:]}"
            executor.spin_once(timeout_sec=0.05)

    trial = "inactive"
    clicks_sent = False

    def publish_inputs() -> None:
        nonlocal clicks_sent
        if not clicks_sent and clicks_pub.get_subscription_count() > 0:
            for x, y in ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)):
                point = PointStamped()
                point.header.frame_id = "map"
                point.point.x, point.point.y = x, y
                clicks_pub.publish(point)
            clicks_sent = True
        target = PersonTarget(track_id=7, valid=True, identity_confirmed=True, state="tracking")
        target.header.frame_id = "base_link"
        target.header.stamp = driver.get_clock().now().to_msg()
        target.point_base_link.x = 6.0
        target.point_base_link.y = 1.0
        target.range_m = 6.08
        target.bearing_rad = 0.16
        target_pub.publish(target)
        message = HandRaise(header=target.header, track_id=7, raised=trial != "lowered")
        if trial == "wrong_person":
            message.track_id = 8
        if trial == "stale":
            message.header.stamp.sec -= 3
        gesture_pub.publish(message)
        speech_pub.publish(String(data="xác nhận"))

    timer = driver.create_timer(0.05, publish_inputs)
    try:
        spin_until(lambda: any("phase=wait_confirm" in state for state in states))
        spin_until(lambda: bool(requests) and requests[-1])
        for trial, duration in (("wrong_person", 1.2), ("short_hold", 0.6),
                                ("lowered", 0.2), ("stale", 1.2)):
            deadline = time.monotonic() + duration
            spin_until(lambda: time.monotonic() >= deadline)
            assert goals == [], trial
            assert not any(active), trial
        trial = "valid_hold"
        spin_until(lambda: bool(goals) and any(active))
        assert len(goals) == 1
        goal = goals[0].pose
        assert goal.header.frame_id == "map"
        assert (goal.pose.position.x, goal.pose.position.y) == (2.0, 1.5)
        assert motion == []
        print("ROS QA: wrong person, brief hold, lowered foot, stale input and speech rejected; "
              "valid hold -> one centroid Nav2 goal; no cmd_vel output")
    finally:
        timer.cancel()
        goal_release.set_result(True)
        for _ in range(10):
            executor.spin_once(timeout_sec=0.01)
        server.destroy()
        executor.remove_node(zone)
        executor.remove_node(driver)
        executor.shutdown()
        zone.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()
