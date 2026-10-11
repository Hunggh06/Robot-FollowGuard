#!/bin/bash
# 1 lenh mo tat ca: agent (tab rieng) + slam+rviz (tab rieng) + ban phim (terminal hien tai).
# Dieu kien: ESP32 da nap firmware, chung WiFi voi laptop.
WS=/home/hungdo/ros2_ws

# Agent chay roi thi thoi (tranh 2 agent danh nhau port 8888)
if ! docker ps --format '{{.Image}}' | grep -q micro-ros-agent; then
  gnome-terminal --tab --title="agent" -- bash -c \
    "docker run --rm --net=host microros/micro-ros-agent:humble udp4 --port 8888; exec bash"
else
  echo "Agent dang chay roi, bo qua."
fi

gnome-terminal --tab --title="slam+rviz" -- bash -c \
  "export DISPLAY=:0; source /opt/ros/humble/setup.bash && source $WS/install/setup.bash && sleep 4 && ros2 launch robot_autonomous_yolo_slam_nav2 slam.launch.py; exec bash"

sleep 6
source /opt/ros/humble/setup.bash
echo "Ban phim san sang: i=ten, ,=lui, j/trai, l=phai, k=dung. Mo RViz xem map."
ros2 run teleop_twist_keyboard teleop_twist_keyboard
