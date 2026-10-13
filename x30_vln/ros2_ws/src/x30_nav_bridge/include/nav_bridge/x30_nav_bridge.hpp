#pragma once

/// @file x30_nav_bridge.hpp
/// @brief 绝影X30导航桥接子类 — 基于UDP模拟手柄轴指令

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nav_bridge/action_executor.hpp"
#include "nav_bridge/nav_bridge_base.hpp"
#include "nav_bridge/robot_state_store.hpp"
#include "nav_bridge/telemetry_utils.hpp"
#include "nav_bridge/udp_transport.hpp"
#include "nav_bridge/velocity_mapping.hpp"
#include "nav_bridge/x30_protocol.hpp"
#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>

namespace nav_bridge {

class X30NavBridge : public NavBridgeBase {
public:
    explicit X30NavBridge(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~X30NavBridge() override;

    bool initialize() override;
    void shutdown() override;

protected:
    void sendVelocityCommand(double vx, double vy, double vyaw) override;
    void sendGaitCommand(uint32_t gait_cmd_code) override;
    void processIncomingData() override;
    void onCmdVelReceived() override;
    void handleStandRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> res) override;
    void handleLieRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> res) override;
    void handleSoftEstopRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res) override;
    void handleReleaseControlRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res) override;
    void handleAcquireControlRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res) override;
    void handleEnableMotionRequest(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
        std::shared_ptr<std_srvs::srv::SetBool::Response> res) override;
    void handleSetGaitRequest(const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
                              std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) override;
    void handleSetBodyHeightRequest(
        const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
        std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) override;
    void handleChargeCommandRequest(
        const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
        std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) override;

private:
    struct ControlPulse {
        bool session_started{false};
        bool session_stopped{false};
        bool send_heartbeat{false};
        bool send_query{false};
        bool send_zero_velocity{false};
        bool active{false};
    };

    struct ChargeCommandResult {
        bool success{false};
        uint16_t charge_state{x30_protocol::CHARGE_STATE_IDLE};
        std::string state_name{"idle"};
        std::string message;
    };

    // ===================== UDP 通信 =====================
    UdpTransport motion_udp_;   ///< 与103运动主机通信
    UdpTransport percept_udp_;  ///< 与105感知主机通信 (可选)

    std::thread recv_thread_;
    std::thread charge_recv_thread_;
    std::thread charge_query_thread_;
    std::thread control_heartbeat_thread_;
    std::atomic<bool> running_{false};
    void receiveLoop();

    // ===================== 数据解析 & 发布 =====================
    void applyControlActions(const ControlPulse &actions);
    void acquireControl();
    bool releaseControlOwnership();
    bool hasActiveControlSession() const;
    void warmupControl(int warmup_ms, int pulse_ms);
    bool isControlInputFresh(std::chrono::steady_clock::time_point now) const;
    ControlPulse evaluateControlPulse(std::chrono::steady_clock::time_point now, bool force_heartbeat);
    ActionResult executeActionWithCmdVelSuppressed(const char *action_name,
                                                   const std::function<ActionResult()> &action);
    void enterCmdVelSuppression(const char *action_name);
    void exitCmdVelSuppression(const char *action_name);
    bool isCmdVelSuppressed() const;
    bool isCmdVelForwardingAllowed() const;
    bool isSupportedNavigationGait(uint8_t gait) const;
    bool isCrawlCompatibleGait(uint8_t gait) const;
    bool isBodyHeightSwitchAllowedGait(uint8_t gait) const;
    bool isCmdVelCompatibleState(uint8_t basic_state, uint8_t gait_state) const;
    bool waitForCmdVelCompatibleState(int timeout_ms) const;
    ActionResult setNavigationGait(uint8_t gait);
    ActionResult setBodyHeight(int32_t requested_height_value);
    bool sendChargeVelocitySource(int32_t source);
    bool sendChargeCommand(uint32_t code, int32_t value);
    bool waitForChargeResponse(uint64_t previous_seq, uint16_t &out_state);
    bool waitForChargeResponseFor(uint64_t previous_seq, std::chrono::milliseconds timeout, uint16_t &out_state);
    bool ensureManualMode(const char *reason);
    bool switchToManualModeAfterCharge();
    bool isFailureChargeState(uint16_t state) const;
    bool isExpectedChargeStateForCommand(uint8_t command, uint16_t state) const;
    bool parseChargeCommandParameter(const rcl_interfaces::msg::Parameter &param,
                                     uint8_t &command,
                                     std::string &error) const;
    ChargeCommandResult executeChargeCommand(uint8_t command);
    std::string encodeChargeCommandResult(const ChargeCommandResult &result) const;
    void chargeReceiveLoop();
    void chargeQueryLoop();
    void handleRcsData(const x30_protocol::RcsData &data);
    void handleMotionState(const x30_protocol::MotionStateData &data);
    void handleBodyHeightState(int32_t body_height_state);
    void handleControllerSensor(const x30_protocol::ControllerSensorData &data);
    void handleBattery(const x30_protocol::BatterySensorData &data);

    // ===================== 定频发送轴指令 =====================
    void controlHeartbeatLoop();
    void sendCmdVelTick();


    // ===================== x30 fork additions =====================
    struct LatestTelemetry {
        x30_protocol::RcsData rcs{};
        x30_protocol::ControllerSafeData safe{};
        x30_protocol::BatterySensorData battery{};
        bool have_rcs{false};
        bool have_safe{false};
        bool have_battery{false};
    };

    void declareAndReadParameters();
    void createInterfaces();
    // velocity path (x30_nav_bridge_control.cpp)
    velmap::GaitSpeedLimit gaitLimitOrZero(uint8_t gait, bool &known) const;
    void performSoftEstop(const char *reason);
    void setForwardingEnabled(bool enabled, const char *reason);
    bool telemetryFresh(std::chrono::steady_clock::time_point now) const;
    void updateEstopState(std::chrono::steady_clock::time_point now);
    void publishEstopActive();
    void onEstopMessage(const std_msgs::msg::Bool::SharedPtr msg);
    // telemetry / diagnostics (x30_nav_bridge_io.cpp)
    void publishDiagnostics();
    void publishBattery(const x30_protocol::BatterySensorData &data);
    void publishJointStates(const x30_protocol::ControllerSensorData &data, const rclcpp::Time &stamp);

    bool use_motion_{false};
    velmap::GaitSpeedTable gait_table_;
    velmap::HardLimits hard_limits_;
    velmap::LimiterConfig limiter_config_;
    velmap::VelocityLimiter limiter_;  ///< touched only from the 50 Hz tick (cb_realtime_ group)
    std::atomic<bool> limiter_reset_requested_{false};
    std::atomic<bool> forwarding_enabled_{false};
    std::atomic<bool> soft_estop_latched_{false};
    std::atomic<bool> robot_estop_{false};
    std::atomic<bool> estop_active_{false};
    std::atomic<int> estop_resends_left_{0};
    std::atomic<uint8_t> last_basic_state_{0};
    std::atomic<uint8_t> last_emergency_source_{0};
    std::atomic<std::chrono::steady_clock::time_point> soft_estop_sent_time_{std::chrono::steady_clock::time_point::min()};
    std::atomic<std::chrono::steady_clock::time_point> last_robot_estop_seen_{std::chrono::steady_clock::time_point::min()};
    std::atomic<std::chrono::steady_clock::time_point> last_motion_rx_{std::chrono::steady_clock::time_point::min()};
    std::atomic<std::chrono::steady_clock::time_point> last_rcs_rx_{std::chrono::steady_clock::time_point::min()};
    std::atomic<std::chrono::steady_clock::time_point> last_sensor_rx_{std::chrono::steady_clock::time_point::min()};
    std::atomic<uint64_t> hard_clamp_events_{0};
    std::atomic<uint64_t> gait_clamp_events_{0};
    std::atomic<uint64_t> rx_rcs_{0}, rx_motion_{0}, rx_sensor_{0}, rx_safe_{0}, rx_battery_{0};
    std::atomic<bool> telemetry_was_stale_{true};
    std::atomic<uint8_t> last_gait_for_log_{255};
    mutable std::mutex telemetry_mutex_;
    LatestTelemetry latest_;
    std::chrono::steady_clock::time_point last_joint_pub_{std::chrono::steady_clock::time_point::min()};
    std::chrono::steady_clock::time_point last_cmd_tick_{std::chrono::steady_clock::time_point::min()};
    std::chrono::steady_clock::time_point last_estop_msg_action_{std::chrono::steady_clock::time_point::min()};

    // parameters (fork)
    std::string topic_prefix_{"/x30"};
    std::string cmd_vel_topic_{"/x30/cmd_vel_hw"};
    std::string estop_topic_{"/x30/estop"};
    int telemetry_stale_ms_{500};
    bool enable_charge_interface_{false};
    double max_vx_fwd_{0.2}, max_vx_back_{0.1}, max_vy_{0.1}, max_wz_{0.3};
    double odom_pose_cov_xy_{0.05}, odom_pose_cov_yaw_{0.1}, odom_twist_cov_v_{0.01}, odom_twist_cov_wz_{0.02};
    double odom_unobserved_cov_{1.0e6};
    double imu_orientation_cov_{1.0e-3}, imu_angvel_cov_{1.0e-4}, imu_linacc_cov_{1.0e-2};
    bool odom_flip_vy_{false}, odom_flip_yaw_{false};
    double battery_current_unit_a_{1.0};
    double battery_current_sign_{-1.0};  ///< BatteryState.current: + = charging; robot sign convention not documented
    double joint_states_rate_hz_{20.0};
    int estop_clear_ms_{1500};

    // ===================== 参数 =====================
    std::string motion_host_ip_;
    std::string charge_host_ip_;
    int motion_host_port_;
    int charge_host_port_;
    int charge_config_port_;
    int charge_local_port_;
    int local_recv_port_;
    int heartbeat_interval_ms_;
    int cmd_vel_rate_hz_;
    int cmd_vel_timeout_ms_;
    bool enable_charge_state_query_;
    int charge_state_query_interval_ms_;
    int charge_state_query_timeout_ms_;
    bool startup_acquire_control_;
    int stand_target_gait_;

    // ===================== 控制权 & 心跳 =====================
    mutable std::mutex control_mutex_;
    bool control_latched_{false};
    bool control_session_started_{false};
    bool control_query_sent_{false};
    std::chrono::steady_clock::time_point last_heartbeat_sent_{
        std::chrono::steady_clock::time_point::min()};
    std::atomic<int> cmd_vel_suppression_depth_{0};
    std::mutex charge_service_mutex_;
    mutable std::mutex charge_state_mutex_;
    std::condition_variable charge_state_cv_;
    uint16_t latest_charge_response_state_{x30_protocol::CHARGE_STATE_IDLE};
    uint64_t charge_response_seq_{0};
    std::atomic<bool> is_nav_mode_{false};
    std::atomic<int> last_logged_nav_mode_{-1};

    // ===================== 业务模块 =====================
    RobotStateStore state_store_;
    std::unique_ptr<ActionExecutor> action_executor_;
};

}  // namespace nav_bridge
