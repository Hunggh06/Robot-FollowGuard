cmake_minimum_required(VERSION 3.8)
project(kaiaai_telemetry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(kaiaai_msgs REQUIRED)

add_executable(telem src/telem.cpp)
target_include_directories(telem PRIVATE src include)
target_link_libraries(telem kaiaai_msgs::kaiaai_msgs__rosidl_typesupport_cpp)
ament_target_dependencies(telem rclcpp sensor_msgs)

install(TARGETS telem DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY src/ DESTINATION include/${PROJECT_NAME}/)
install(DIRECTORY include/ DESTINATION include/)

install(PROGRAMS dummy_odom.py DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME}/)

ament_package()
