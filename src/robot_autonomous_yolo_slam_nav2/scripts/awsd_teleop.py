#!/usr/bin/env python3
"""AWSD teleop: giu phim chay, nha phim dung (kieu game).

W tien, S lui, A trai, D phai, Space dung gap, Q thoat.
Terminal chi gui ky tu khi phim duoc nhan (lap lai khi giu) nen node
xem phim nao co tin hieu trong 0.25 s qua la "dang giu". Het tin hieu
-> ve 0 ngay. Toc do nhe de mapping: 0.2 m/s, 0.5 rad/s.
"""
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LINEAR = 0.15
ANGULAR = 0.35
HOLD = 0.25


class AwsdTeleop(Node):
    def __init__(self):
        super().__init__('awsd_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pressed = {}
        self.settings = termios.tcgetattr(sys.stdin)
        print('AWSD: W/S tien/lui, A/D trai/phai, Space dung, Q thoat. '
              'Giu phim de chay, nha phim de dung.')

    def key(self):
        tty.setraw(sys.stdin.fileno())
        out = None
        try:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                out = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return out

    def run(self):
        try:
            while rclpy.ok():
                k = self.key()
                now = time.time()
                if k == 'q':
                    break
                if k:
                    self.pressed[k] = now
                hold = {c: t for c, t in self.pressed.items()
                        if now - t < HOLD}
                self.pressed = hold
                t = Twist()
                if ' ' in hold:
                    pass
                else:
                    if 'w' in hold:
                        t.linear.x += LINEAR
                    if 's' in hold:
                        t.linear.x -= LINEAR
                    if 'a' in hold:
                        t.angular.z += ANGULAR
                    if 'd' in hold:
                        t.angular.z -= ANGULAR
                self.pub.publish(t)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.pub.publish(Twist())
            print('\nawsd_teleop dung.')


def main():
    rclpy.init()
    node = AwsdTeleop()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
