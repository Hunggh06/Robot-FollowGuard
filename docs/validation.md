# Validation Status

## Implemented evidence

The public source includes runtime validity gates: AprilTag attribution,
sensor/TF freshness checks, bounded spin search, and a safety-gate scan check.
The repository was built as ROS 2 packages before publication.

## Pending quantitative validation

No hardware trial dataset, reproducible benchmark suite, accuracy statistic,
tracking-success rate, FPS measurement, latency measurement, or obstacle
avoidance certification is committed here. Do not infer any of these from a
successful launch or visualization.

## Recommended evaluation record

For each hardware run, record lighting, tag size and range, camera resolution,
LiDAR health, Wi-Fi/DDS topology, target-loss/reacquisition outcomes, zone
transition behavior, and any safety-gate stops. Publish measurements and a
protocol before making performance claims.
