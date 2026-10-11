from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.time import Time

from robot_autonomous_yolo_slam_nav2.yolo_target_node import YoloTargetNode


def test_inference_latency_is_not_hidden_by_a_new_output_stamp(monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")

    clock = {"ns": 1_000_000_000}

    class Model:
        def predict(self, *args, **kwargs):
            clock["ns"] += 1_000_000_000
            return []

    monkeypatch.setattr(ultralytics, "YOLO", lambda name: Model())
    rclpy.init(args=[])
    node = YoloTargetNode()
    outputs = []
    try:
        clock["ns"] = 1_000_000_000
        monkeypatch.setattr(node, "get_clock", lambda: SimpleNamespace(now=lambda: Time(nanoseconds=clock["ns"])))
        monkeypatch.setattr(node, "_target_pub", SimpleNamespace(publish=outputs.append))
        monkeypatch.setattr(node._annotation, "publish", lambda *args: None)
        message = node._bridge.cv2_to_imgmsg(np.zeros((240, 320, 3), dtype=np.uint8), "bgr8")
        message.header.frame_id = "phone_camera_back"
        node._image_callback(message)
        assert outputs[-1].header.stamp.sec == 1
        assert clock["ns"] == 2_000_000_000
    finally:
        node.destroy_node()
        rclpy.shutdown()
