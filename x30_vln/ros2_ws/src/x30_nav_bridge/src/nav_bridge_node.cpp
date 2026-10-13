/// @file nav_bridge_node.cpp
/// @brief Nav Bridge entry point (x30 fork: multi-threaded executor so ~/soft_estop is never queued behind a
///        blocking ~/stand; signal handling is left to rclcpp, which installs SIGINT/SIGTERM handlers).

#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "nav_bridge/x30_nav_bridge.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<nav_bridge::X30NavBridge>(rclcpp::NodeOptions());
    if (!node->initialize()) {
        RCLCPP_FATAL(node->get_logger(), "x30_nav_bridge initialisation failed, exiting");
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();
    node->shutdown();
    node.reset();
    rclcpp::shutdown();
    return 0;
}
