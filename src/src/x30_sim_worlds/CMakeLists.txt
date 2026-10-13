cmake_minimum_required(VERSION 3.8)
project(x30_sim_worlds)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds DESTINATION share/${PROJECT_NAME})

ament_package()
