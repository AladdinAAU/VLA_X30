cmake_minimum_required(VERSION 3.8)
project(x30_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection3D.msg"
  "msg/Detection3DArray.msg"
  "msg/SemanticObject.msg"
  "msg/SemanticObjectArray.msg"
  "msg/SafetyStatus.msg"
  "srv/Json.srv"
  "srv/SetMode.srv"
  "action/Navigate.action"
  "action/Skill.action"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs)
ament_package()
