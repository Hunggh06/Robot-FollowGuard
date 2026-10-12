#!/usr/bin/env python3
"""Drive a circle with guaranteed breakaway: forward until the right
wheel actually rolls, then blend into the turn with no stop in between.
One continuous 20 Hz command stream (no gaps for the wheel to re-stick).

Usage: drive_circle.py --left 60 --right 19 --circle-time 10
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


class CircleDriver(Node):
    def __init__(self, left, right, circle_time):
        super().__init__('drive_circle')
        self.left = left
        self.right = right
        self.circle_time = circle_time
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Float32MultiArray, '/motor/state',
                                 self.state_cb, qos)
        self.pub = self.create_publisher(Float32MultiArray,
                                         '/motor/target_rpm', 10)
        self.enc = None
        self.start_enc = None

    def state_cb(self, msg):
        if len(msg.data) >= 6:
            self.enc = (msg.data[4], msg.data[5])

    def send(self, l, r):
        m = Float32MultiArray()
        m.data = [float(l), float(r)]
        self.pub.publish(m)

    def spin_wait(self, secs, rate=20.0):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=1.0 / rate)

    def run(self):
        while self.enc is None:
            self.get_logger().info('waiting for /motor/state ...')
            self.spin_wait(0.5)

        # Phase 1: forward until right wheel proves it rolls.
        self.start_enc = self.enc
        t0 = time.time()
        while time.time() - t0 < 4.0:
            self.send(50.0, 50.0)
            self.spin_wait(0.05)
            if self.enc[1] - self.start_enc[1] > 80:
                break
        freed = self.enc[1] - self.start_enc[1] > 80
        self.get_logger().info(f'breakaway: right moved '
                               f'{self.enc[1] - self.start_enc[1]:.0f} ticks')

        # Phase 2+3: ramp into the circle, hold (never stop in between).
        steps = 10
        for i in range(1, steps + 1):
            l = 50.0 + (self.left - 50.0) * i / steps
            r = 50.0 + (self.right - 50.0) * i / steps
            t1 = time.time()
            while time.time() - t1 < 0.15:
                self.send(l, r)
                self.spin_wait(0.05)
        t1 = time.time()
        while time.time() - t1 < self.circle_time:
            self.send(self.left, self.right)
            self.spin_wait(0.05)

        # Phase 4: stop.
        t1 = time.time()
        while time.time() - t1 < 0.8:
            self.send(0.0, 0.0)
            self.spin_wait(0.05)
        self.get_logger().info(f'done (breakaway ok: {freed})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left', type=float, default=60.0)
    ap.add_argument('--right', type=float, default=19.0)
    ap.add_argument('--circle-time', type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    node = CircleDriver(args.left, args.right, args.circle_time)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
