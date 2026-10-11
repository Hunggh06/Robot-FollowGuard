#!/usr/bin/env python3
"""Kiem tra node kaiaai_telemetry bang du lieu LDS02RR gia lap.

ESP32 that chua nap duoc, nen bo cac byte LDS02RR raw ( dung protocol Neato
XV11 ) vao /telemetry roi xac nhan node giai ma ra /scan dung goc.

Kiem tra 4 thu:
  1. CRC dung — neu cong thuc sai, node bo het packet, /scan khong co du lieu
  2. Goc khop     — dat moi tia o goc 0..359 do, doi 1 mm theo goc
  3. INFINITY     — tia khong co du lieu phai ra inf chu khong phai 0.0
  4. Timestamp    — lech bao nhieu giay so voi host

Chay:  source install/setup.bash && python3 kiem_tra_telem.py
"""
import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from kaiaai_msgs.msg import KaiaaiTelemetry2
from sensor_msgs.msg import LaserScan

N_BIN = 360
BYTES_PER_PKT = 22
N_PKT = 90
MAX_LDS = 511


def make_packet(index, distances_mm, quality=255):
    """Dung 1 goi Neato XV11 22 byte, kem CRC 15-bit dung cong thuc trong
    src/lds_neato_xv11.h (isValidPacket)."""
    p = bytearray(BYTES_PER_PKT)
    p[0] = 0xFA
    p[1] = 0xA0 + index
    p[2] = 0x00
    p[3] = 0x00
    for i, d in enumerate(distances_mm):
        base = 4 + i * 4
        p[base + 0] = d & 0xFF
        p[base + 1] = (d >> 8) & 0x3F      # bit7/bit6 = 0: khong invalid
        p[base + 2] = quality & 0xFF
        p[base + 3] = (quality >> 8) & 0xFF
    words = [p[2 * i] + (p[2 * i + 1] << 8) for i in range(10)]
    chk = 0
    for w in words:
        chk = (chk << 1) + w
    s = ((chk & 0x7FFF) + (chk >> 15)) & 0x7FFF
    p[20] = s & 0xFF
    p[21] = (s >> 8) & 0xFF
    return bytes(p)


def build_revolution(marks=()):
    """90 goi, moi goi 4 tia. dist = 1000 + goc (mm) nen doi 1 mm theo goc.
    `marks` la cac goc bi bo trong (gia lap tia khong do duoc)."""
    out = bytearray()
    for idx in range(N_PKT):
        base = idx * 4
        quad = []
        for k in range(4):
            ang = base + k
            quad.append(0 if ang in marks else 1000 + ang)
        out += make_packet(idx, quad)
    return bytes(out)


def main():
    rclpy.init()
    node = Node('kiem_tra_telem')

    pub = node.create_publisher(KaiaaiTelemetry2, '/telemetry',
                                QoSProfile(depth=10,
                                           reliability=ReliabilityPolicy.RELIABLE))
    scans = []
    node.create_subscription(LaserScan, '/scan', scans.append, 10)

    # goc 40, 41, 200, 201 bo de kiem tra INFINITY
    marks = {40, 41, 200, 201}
    data = build_revolution(marks) + build_revolution(marks)
    print(f"goc lap: {len(data)} byte, chia thanh "
          f"{-(-len(data)//MAX_LDS)} goi telemetry (<= {MAX_LDS} byte moi goi)")

    deadline = time.time() + 12
    seq = 0
    while time.time() < deadline:
        for off in range(0, len(data), MAX_LDS):
            m = KaiaaiTelemetry2()
            now = time.time()
            m.stamp.sec = int(now)
            m.stamp.nanosec = int((now - int(now)) * 1e9)
            m.seq = seq
            seq += 1
            chunk = data[off:off + MAX_LDS]
            m.lds = list(chunk)
            pub.publish(m)
            time.sleep(0.02)
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.1)

    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.2)
        if scans:
            break

    print(f"\nnhan duoc {len(scans)} scan tren /scan")

    if not scans:
        print("KET QUA: FAIL — khong co /scan nao")
        rclpy.shutdown()
        return 1

    s = scans[-1]
    age = time.time() - (s.header.stamp.sec + s.header.stamp.nanosec * 1e-9)
    fin = [r for r in s.ranges if not math.isinf(r)]
    zeros = [r for r in s.ranges if r == 0.0]

    print(f"  frame_id  = {s.header.frame_id}")
    print(f"  so diem   = {len(s.ranges)}")
    print(f"  increment = {s.angle_increment:.6f}  (2pi/360 = {2*math.pi/360:.6f})")
    print(f"  range     = [{s.range_min}, {s.range_max}]")
    print(f"  stamp lech= {age:+.3f} s")
    print(f"  co gia tri= {len(fin)}   inf= {len(s.ranges)-len(fin)-len(zeros)}"
          f"   =0.0 -> {len(zeros)}")

    ok = True
    print("\nkiem tra:")
    if len(s.ranges) != N_BIN:
        print(f"  [SAI] so diem {len(s.ranges)} != {N_BIN}")
        ok = False
    else:
        print(f"  [OK] {N_BIN} diem")

    if not fin:
        print("  [SAI] khong co diem nao -> CRC hoac goc khong khop")
        ok = False
    else:
        print(f"  [OK] {len(fin)} diem co gia tri (CRC dung)")

    # gac 5, 100, 300, 350 — bo cac goc da danh dau
    errs = []
    for ang in (5, 100, 300, 350):
        want = (1000 + ang) / 1000.0
        got = s.ranges[ang]
        if not math.isinf(got) and abs(got - want) > 1e-6:
            errs.append(f"goc {ang}: can {want:.3f}, co {got:.3f}")
    if errs:
        print(f"  [SAI] gac sai: {errs}")
        ok = False
    else:
        print("  [OK] gac khop: r[5]=1.005 r[100]=1.100 r[300]=1.300 r[350]=1.350")

    bad_inf = [a for a in marks if not math.isinf(s.ranges[a])]
    if bad_inf:
        print(f"  [SAI] tia bo khong ra inf tai goc {sorted(bad_inf)}")
        ok = False
    else:
        print(f"  [OK] tia khong do duoc ra INFINITY tai goc {sorted(marks)}")

    if zeros:
        print(f"  [SAI] con {len(zeros)} tia = 0.0 — phai la inf")
        ok = False
    else:
        print("  [OK] khong co tia 0.0")

    if abs(age) > 5.0:
        print(f"  [SAI] timestamp lech {age:+.1f} s — qua 5 s")
        ok = False
    else:
        print(f"  [OK] timestamp lech {age:+.3f} s")

    print(f"\nKET QUA: {'PASS' if ok else 'FAIL'}")
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
