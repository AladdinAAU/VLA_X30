#pragma once

/// @file nav_bridge_base.hpp
/// @brief Protocol independent navigation bridge interface (upstream class, extended in the x30 fork:
///        /x30 topic names, DiagnosticArray + BatteryState outputs, latched e-stop output, callback groups,
///        explicit acquire/enable services; rviz_2d_overlay_msgs dependency removed).

#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <chrono>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <functional>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/battery_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>

namespace nav_bridge {

/// Generic robot state
enum class RobotState {
    DISCONNECTED,
    LYING_DOWN,
    STANDING_UP,
    INITIAL_STAND,
    FORCE_STAND,
    STEPPING,
    GOING_DOWN,
    SOFT_ESTOP,
    RL_MODE,
};

class NavBridgeBase : public rclcpp::Node {
public:
    explicit NavBridgeBase(const std::string &node_name,
                           const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    virtual ~NavBridgeBase() = default;

    virtual bool initialize() = 0;
    virtual void shutdown()   = 0;

protected:
    /// Send an already limited body velocity (m/s, rad/s; ROS convention +y left, +wz CCW) to the robot.
    virtual void sendVelocityCommand(double vx, double vy, double vyaw) = 0;
    virtual void sendGaitCommand(uint32_t gait_cmd_code)               = 0;
    virtual void processIncomingData()                                  = 0;

    // ---- subscriptions
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;

    // ---- publishers (names are built from the `topic_prefix` parameter, default /x30)
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr robot_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr gait_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr body_height_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr charge_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr battery_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_active_pub_;  ///< transient local (latched)
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr ros_heartbeat_pub_;

    // ---- services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stand_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr lie_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr soft_estop_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr release_control_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr acquire_control_srv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_motion_srv_;
    rclcpp::Service<rcl_interfaces::srv::SetParameters>::SharedPtr set_gait_srv_;
    rclcpp::Service<rcl_interfaces::srv::SetParameters>::SharedPtr set_body_height_srv_;
    rclcpp::Service<rcl_interfaces::srv::SetParameters>::SharedPtr charge_command_srv_;

    virtual void handleStandRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                    std::shared_ptr<std_srvs::srv::Trigger::Response> res) = 0;
    virtual void handleLieRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> res) = 0;
    virtual void handleSoftEstopRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                        std::shared_ptr<std_srvs::srv::Trigger::Response> res) = 0;
    virtual void handleReleaseControlRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) = 0;
    virtual void handleAcquireControlRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) = 0;
    virtual void handleEnableMotionRequest(const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
                                           std::shared_ptr<std_srvs::srv::SetBool::Response> res) = 0;
    virtual void handleSetGaitRequest(const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
                                      std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) = 0;
    virtual void handleSetBodyHeightRequest(const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
                                            std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) = 0;
    virtual void handleChargeCommandRequest(const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
                                            std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) = 0;

    // ---- TF (publish_tf=false by default)
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ---- timers and callback groups. The executor is multi-threaded: the slow service group serialises
    //      stand/lie/gait services, so soft_estop (fast group) is never queued behind them.
    rclcpp::TimerBase::SharedPtr cmd_vel_timer_;
    rclcpp::TimerBase::SharedPtr diag_timer_;
    rclcpp::TimerBase::SharedPtr ros_heartbeat_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_realtime_;
    rclcpp::CallbackGroup::SharedPtr cb_fast_;
    rclcpp::CallbackGroup::SharedPtr cb_slow_;
    rclcpp::CallbackGroup::SharedPtr cb_diag_;

    std::atomic<RobotState> robot_state_{RobotState::DISCONNECTED};

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    virtual void onCmdVelReceived() {}
    void publishRosHeartbeat();

    // latest /cmd_vel input
    std::atomic<double> target_vx_{0.0};
    std::atomic<double> target_vy_{0.0};
    std::atomic<double> target_vyaw_{0.0};
    std::atomic<std::chrono::steady_clock::time_point> last_active_time_{
        std::chrono::steady_clock::time_point::min()};  // min = never received

    // parameters shared with the base
    std::string imu_frame_id_{"imu_link"};
    std::string odom_frame_id_{"odom"};
    std::string base_frame_id_{"base_link"};
    bool publish_tf_{false};
    std::string ros_heartbeat_topic_{"/inspection_task_hub/heartbeat/nav_bridge"};
    double ros_heartbeat_rate_hz_{1.0};
};

}  // namespace nav_bridge
