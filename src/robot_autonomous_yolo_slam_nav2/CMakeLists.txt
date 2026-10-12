cmake_minimum_required(VERSION 3.8)
project(robot_autonomous_yolo_slam_nav2)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonTarget.msg"
  "msg/PersonObservation.msg"
  "msg/PersonObstacle.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

file(GLOB PYTHON_PACKAGE_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/*.py"
)
install(FILES ${PYTHON_PACKAGE_FILES}
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/${PROJECT_NAME}
)

install(DIRECTORY urdf meshes launch config calibration rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES calibration/board-a4-ultrawide.pdf calibration/board-a4-ultrawide-preview.png
  DESTINATION share/${PROJECT_NAME}/calibration
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)
install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

install(PROGRAMS scripts/odom_tf_bridge.py scripts/cmd_vel_bridge.py scripts/motor_odom.py scripts/awsd_teleop.py scripts/drive_circle.py scripts/imu_remap.py scripts/map_room.sh
  scripts/metric_target_node.py scripts/nav2_person_follow.py
  scripts/zone_guard_node.py
  scripts/phone_camera_info.py scripts/camera_info_publisher.py
  scripts/yolo_target_node.py scripts/safety_gate_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  file(GLOB PYTHON_TEST_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.py"
  )
  foreach(TEST_FILE IN LISTS PYTHON_TEST_FILES)
    get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
    ament_add_pytest_test("${TEST_NAME}" "${TEST_FILE}")
  endforeach()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
