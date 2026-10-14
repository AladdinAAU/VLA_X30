cmake_minimum_required(VERSION 3.8)
project(x30_worlds)

find_package(ament_cmake REQUIRED)

# worlds/<id>/{world.sdf,objects.json,rooms.json,map.yaml,map.png}, models/<name>/..., config/labels.yaml
install(DIRECTORY worlds config DESTINATION share/${PROJECT_NAME})
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
else()
  message(STATUS "x30_worlds: optional models directory missing; imported worlds require generated/downloaded assets")
endif()

# GZ_SIM_RESOURCE_PATH gets <share>/x30_worlds/models so model:// URIs resolve after `source install/setup.bash`
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/x30_worlds.dsv.in")

ament_package()
