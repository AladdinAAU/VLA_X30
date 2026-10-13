cmake_minimum_required(VERSION 3.8)
project(x30_nav_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rcl_interfaces REQUIRED)

set(NB_DEPS rclcpp sensor_msgs nav_msgs geometry_msgs diagnostic_msgs std_msgs std_srvs tf2 tf2_ros rcl_interfaces)

add_executable(nav_bridge_node
  src/nav_bridge_node.cpp
  src/action_executor.cpp
  src/udp_transport.cpp
  src/x30_nav_bridge.cpp
  src/x30_nav_bridge_core.cpp
  src/x30_nav_bridge_control.cpp
  src/x30_nav_bridge_io.cpp
)
target_include_directories(nav_bridge_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(nav_bridge_node ${NB_DEPS})

install(TARGETS nav_bridge_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME}/)
install(FILES LICENSE NOTICE CHANGES.md DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_protocol_layout test/test_protocol_layout.cpp)
  target_include_directories(test_protocol_layout PRIVATE include)
  ament_add_gtest(test_velocity_mapping test/test_velocity_mapping.cpp)
  target_include_directories(test_velocity_mapping PRIVATE include)
endif()

ament_package()
