/// @file x30_nav_bridge_core.cpp
/// @brief x30 fork: parameters, interface creation, lifecycle. Replaces the upstream constructor/initialize/shutdown.

#include <algorithm>
#include <chrono>
#include <cmath>

#include "nav_bridge/x30_nav_bridge.hpp"

namespace nav_bridge {

using namespace x30_protocol;

namespace {
constexpr double kCeilingLinear = 0.6;   ///< refuse hard limits above this unless allow_high_speed (digest 5.1: stage D max)
constexpr double kCeilingYaw    = 0.8;

std::string gaitParamName(uint8_t gait) { return "gait_limits.g" + std::to_string(static_cast<int>(gait)); }
}  // namespace

NavBridgeBase::NavBridgeBase(const std::string &node_name, const rclcpp::NodeOptions &options)
    : rclcpp::Node(node_name, options) {}

void NavBridgeBase::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    target_vx_.store(msg->linear.x);
    target_vy_.store(msg->linear.y);
    target_vyaw_.store(msg->angular.z);
    last_active_time_.store(std::chrono::steady_clock::now());
    onCmdVelReceived();
}

void NavBridgeBase::publishRosHeartbeat() {
    if (!ros_heartbeat_pub_) return;
    diagnostic_msgs::msg::DiagnosticArray msg;
    msg.header.stamp = this->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name        = this->get_name();
    status.hardware_id = "nav_bridge";
    status.level       = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message     = "ok";
    msg.status.push_back(status);
    ros_heartbeat_pub_->publish(msg);
}

// ============================================================================
// Parameters
// ============================================================================

X30NavBridge::X30NavBridge(const rclcpp::NodeOptions &options) : NavBridgeBase("nav_bridge_node", options) {
    declareAndReadParameters();
}

X30NavBridge::~X30NavBridge() { shutdown(); }

void X30NavBridge::declareAndReadParameters() {
    // --- network
    this->declare_parameter("motion_host_ip", std::string(MOTION_HOST_IP));
    this->declare_parameter("motion_host_port", MOTION_HOST_PORT);
    this->declare_parameter("local_recv_port", 43897);
    // --- charge interface (disabled by default: it talks to the PERCEPTION host, not needed for VLN)
    this->declare_parameter("enable_charge_interface", false);
    this->declare_parameter("charge_host_ip", std::string(PERCEPT_HOST_IP));
    this->declare_parameter("charge_host_port", PERCEPT_CHARGE_PORT);
    this->declare_parameter("charge_config_port", PERCEPT_HOST_PORT);
    this->declare_parameter("charge_local_port", PERCEPT_CHARGE_LOCAL_PORT);
    this->declare_parameter("enable_charge_state_query", true);
    this->declare_parameter("charge_state_query_interval_ms", 1000);
    this->declare_parameter("charge_state_query_timeout_ms", 1500);
    // --- timing
    this->declare_parameter("heartbeat_interval_ms", HEARTBEAT_INTERVAL_MS);
    this->declare_parameter("cmd_vel_rate_hz", CMD_VEL_RATE_HZ);
    this->declare_parameter("cmd_vel_timeout_ms", 400);
    this->declare_parameter("telemetry_stale_ms", 500);
    this->declare_parameter("startup_acquire_control", false);
    this->declare_parameter("stand_target_gait", static_cast<int>(GaitState::MOUNTAIN));
    // --- frames / tf
    this->declare_parameter("imu_frame_id", std::string("imu_link"));
    this->declare_parameter("odom_frame_id", std::string("odom"));
    this->declare_parameter("base_frame_id", std::string("base_link"));
    this->declare_parameter("publish_tf", false);
    this->declare_parameter("ros_heartbeat_topic", std::string(""));
    this->declare_parameter("ros_heartbeat_rate_hz", 1.0);
    // --- topics
    this->declare_parameter("topic_prefix", std::string("/x30"));
    this->declare_parameter("cmd_vel_topic", std::string("/x30/cmd_vel_hw"));
    this->declare_parameter("estop_topic", std::string("/x30/estop"));
    // --- hard limits (independent of gait) and shaping
    this->declare_parameter("allow_high_speed", false);
    this->declare_parameter("max_vx_forward", 0.2);
    this->declare_parameter("max_vx_backward", 0.1);
    this->declare_parameter("max_vy", 0.1);
    this->declare_parameter("max_wz", 0.3);
    this->declare_parameter("max_accel_x", 0.5);
    this->declare_parameter("max_accel_y", 0.5);
    this->declare_parameter("max_accel_yaw", 1.0);
    this->declare_parameter("max_decel_x", 0.5);
    this->declare_parameter("max_decel_y", 0.5);
    this->declare_parameter("max_decel_yaw", 1.0);
    this->declare_parameter("max_jerk_x", 2.0);
    this->declare_parameter("max_jerk_y", 2.0);
    this->declare_parameter("max_jerk_yaw", 4.0);
    // --- gait speed table (API Appendix B defaults; override per gait as [fwd, back, lat, yaw])
    const auto defaults = velmap::GaitSpeedTable::apiAppendixB();
    for (uint8_t g : velmap::GaitSpeedTable::knownGaits()) {
        velmap::GaitSpeedLimit l;
        defaults.find(g, l);
        this->declare_parameter(gaitParamName(g),
                                std::vector<double>{l.max_forward, l.max_backward, l.max_lateral, l.max_yaw});
    }
    // --- covariances (placeholders: to be measured, digest gaps 8 and 9)
    this->declare_parameter("odom_pose_cov_xy", 0.05);
    this->declare_parameter("odom_pose_cov_yaw", 0.1);
    this->declare_parameter("odom_twist_cov_v", 0.01);
    this->declare_parameter("odom_twist_cov_wz", 0.02);
    this->declare_parameter("odom_unobserved_cov", 1.0e6);
    this->declare_parameter("imu_orientation_cov", 1.0e-3);
    this->declare_parameter("imu_angvel_cov", 1.0e-4);
    this->declare_parameter("imu_linacc_cov", 1.0e-2);
    this->declare_parameter("odom_flip_vy", false);
    this->declare_parameter("odom_flip_yaw", false);
    // --- battery / misc
    this->declare_parameter("battery_current_unit_a", 1.0);
    this->declare_parameter("battery_current_sign", -1.0);
    this->declare_parameter("joint_states_rate_hz", 20.0);
    this->declare_parameter("estop_clear_ms", 1500);

    auto geti = [this](const char *n) { return static_cast<int>(this->get_parameter(n).as_int()); };
    auto getd = [this](const char *n) { return this->get_parameter(n).as_double(); };
    auto getb = [this](const char *n) { return this->get_parameter(n).as_bool(); };
    auto gets = [this](const char *n) { return this->get_parameter(n).as_string(); };

    motion_host_ip_        = gets("motion_host_ip");
    motion_host_port_      = geti("motion_host_port");
    local_recv_port_       = geti("local_recv_port");
    enable_charge_interface_ = getb("enable_charge_interface");
    charge_host_ip_        = gets("charge_host_ip");
    charge_host_port_      = geti("charge_host_port");
    charge_config_port_    = geti("charge_config_port");
    charge_local_port_     = geti("charge_local_port");
    enable_charge_state_query_      = getb("enable_charge_state_query");
    charge_state_query_interval_ms_ = geti("charge_state_query_interval_ms");
    charge_state_query_timeout_ms_  = geti("charge_state_query_timeout_ms");
    heartbeat_interval_ms_ = geti("heartbeat_interval_ms");
    cmd_vel_rate_hz_       = std::clamp(geti("cmd_vel_rate_hz"), 10, 100);
    cmd_vel_timeout_ms_    = geti("cmd_vel_timeout_ms");
    telemetry_stale_ms_    = geti("telemetry_stale_ms");
    startup_acquire_control_ = getb("startup_acquire_control");
    stand_target_gait_     = geti("stand_target_gait");
    imu_frame_id_          = gets("imu_frame_id");
    odom_frame_id_         = gets("odom_frame_id");
    base_frame_id_         = gets("base_frame_id");
    publish_tf_            = getb("publish_tf");
    ros_heartbeat_topic_   = gets("ros_heartbeat_topic");
    ros_heartbeat_rate_hz_ = getd("ros_heartbeat_rate_hz");
    topic_prefix_          = gets("topic_prefix");
    cmd_vel_topic_         = gets("cmd_vel_topic");
    estop_topic_           = gets("estop_topic");

    hard_limits_.max_vx_forward  = getd("max_vx_forward");
    hard_limits_.max_vx_backward = getd("max_vx_backward");
    hard_limits_.max_vy          = getd("max_vy");
    hard_limits_.max_wz          = getd("max_wz");
    const bool allow_high        = getb("allow_high_speed");
    auto sane = [&](double &v, double ceiling, const char *name) {
        if (!(v >= 0.0)) {
            RCLCPP_ERROR(get_logger(), "%s must be >= 0 (got %.3f); using 0 (no motion on that axis)", name, v);
            v = 0.0;
        } else if (v > ceiling && !allow_high) {
            RCLCPP_ERROR(get_logger(), "%s=%.2f exceeds the safety ceiling %.2f; clamped (set allow_high_speed:=true to override)",
                         name, v, ceiling);
            v = ceiling;
        }
    };
    sane(hard_limits_.max_vx_forward, kCeilingLinear, "max_vx_forward");
    sane(hard_limits_.max_vx_backward, kCeilingLinear, "max_vx_backward");
    sane(hard_limits_.max_vy, kCeilingLinear, "max_vy");
    sane(hard_limits_.max_wz, kCeilingYaw, "max_wz");
    max_vx_fwd_ = hard_limits_.max_vx_forward;
    max_vx_back_ = hard_limits_.max_vx_backward;
    max_vy_ = hard_limits_.max_vy;
    max_wz_ = hard_limits_.max_wz;

    limiter_config_.accel_x   = getd("max_accel_x");
    limiter_config_.accel_y   = getd("max_accel_y");
    limiter_config_.accel_yaw = getd("max_accel_yaw");
    limiter_config_.decel_x   = getd("max_decel_x");
    limiter_config_.decel_y   = getd("max_decel_y");
    limiter_config_.decel_yaw = getd("max_decel_yaw");
    limiter_config_.jerk_x    = getd("max_jerk_x");
    limiter_config_.jerk_y    = getd("max_jerk_y");
    limiter_config_.jerk_yaw  = getd("max_jerk_yaw");
    limiter_.configure(limiter_config_);

    gait_table_ = velmap::GaitSpeedTable::apiAppendixB();
    for (uint8_t g : velmap::GaitSpeedTable::knownGaits()) {
        const auto v = this->get_parameter(gaitParamName(g)).as_double_array();
        if (v.size() != 4 || std::any_of(v.begin(), v.end(), [](double x) { return !(x > 0.0); })) {
            RCLCPP_ERROR(get_logger(), "%s must be 4 positive numbers [fwd, back, lat, yaw]; keeping the API Appendix B default",
                         gaitParamName(g).c_str());
            continue;
        }
        gait_table_.set(g, {v[0], v[1], v[2], v[3]});
    }

    odom_pose_cov_xy_    = getd("odom_pose_cov_xy");
    odom_pose_cov_yaw_   = getd("odom_pose_cov_yaw");
    odom_twist_cov_v_    = getd("odom_twist_cov_v");
    odom_twist_cov_wz_   = getd("odom_twist_cov_wz");
    odom_unobserved_cov_ = getd("odom_unobserved_cov");
    imu_orientation_cov_ = getd("imu_orientation_cov");
    imu_angvel_cov_      = getd("imu_angvel_cov");
    imu_linacc_cov_      = getd("imu_linacc_cov");
    odom_flip_vy_        = getb("odom_flip_vy");
    odom_flip_yaw_       = getb("odom_flip_yaw");
    battery_current_unit_a_ = getd("battery_current_unit_a");
    battery_current_sign_   = getd("battery_current_sign");
    joint_states_rate_hz_   = getd("joint_states_rate_hz");
    estop_clear_ms_         = geti("estop_clear_ms");
}

// ============================================================================
// Lifecycle
// ============================================================================

bool X30NavBridge::initialize() {
    RCLCPP_INFO(get_logger(), "x30_nav_bridge: motion host %s:%d, telemetry bind :%d", motion_host_ip_.c_str(),
                motion_host_port_, local_recv_port_);
    RCLCPP_INFO(get_logger(),
                "hard limits: vx +%.2f/-%.2f m/s, vy %.2f m/s, wz %.2f rad/s; cmd_vel timeout %d ms; control acquired at start: %s",
                hard_limits_.max_vx_forward, hard_limits_.max_vx_backward, hard_limits_.max_vy, hard_limits_.max_wz,
                cmd_vel_timeout_ms_, startup_acquire_control_ ? "yes" : "no");
    RCLCPP_DEBUG(get_logger(), "struct sizes: Rcs=%zu Motion=%zu Sensor=%zu Battery=%zu", sizeof(RcsData),
                 sizeof(MotionStateData), sizeof(ControllerSensorData), sizeof(BatterySensorData));

    if (!motion_udp_.open(motion_host_ip_, motion_host_port_, local_recv_port_)) {
        RCLCPP_ERROR(get_logger(), "cannot open the UDP socket for the motion host (port %d busy?)", local_recv_port_);
        return false;
    }
    if (enable_charge_interface_) {
        if (!percept_udp_.open(charge_host_ip_, charge_host_port_, charge_local_port_)) {
            RCLCPP_ERROR(get_logger(), "cannot open the UDP socket for the perception host (charge interface)");
            return false;
        }
        RCLCPP_WARN(get_logger(), "charge interface ENABLED: this node will send UDP to the perception host %s",
                    charge_host_ip_.c_str());
    }

    createInterfaces();

    action_executor_ = std::make_unique<ActionExecutor>(
        this->get_logger(), state_store_,
        [this](int warmup_ms, int pulse_ms) { this->warmupControl(warmup_ms, pulse_ms); },
        [this](uint32_t code) { this->sendGaitCommand(code); });

    if (startup_acquire_control_) {
        acquireControl();
        RCLCPP_WARN(get_logger(), "startup_acquire_control=true: heartbeats start now (this competes with the handheld controller)");
    }

    running_ = true;
    control_heartbeat_thread_ = std::thread(&X30NavBridge::controlHeartbeatLoop, this);
    const int cmd_period_ms   = std::max(1, 1000 / cmd_vel_rate_hz_);
    cmd_vel_timer_ = this->create_wall_timer(std::chrono::milliseconds(cmd_period_ms),
                                             [this]() { this->sendCmdVelTick(); }, cb_realtime_);
    recv_thread_ = std::thread(&X30NavBridge::receiveLoop, this);
    if (enable_charge_interface_) {
        charge_recv_thread_ = std::thread(&X30NavBridge::chargeReceiveLoop, this);
        if (enable_charge_state_query_) charge_query_thread_ = std::thread(&X30NavBridge::chargeQueryLoop, this);
    }

    RCLCPP_INFO(get_logger(), "x30_nav_bridge ready; waiting for telemetry (the dev PC must be registered in network.toml on the motion host)");
    return true;
}

void X30NavBridge::createInterfaces() {
    using namespace std::placeholders;
    const std::string p = topic_prefix_;
    cb_realtime_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_fast_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_slow_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_diag_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions rt_opts;
    rt_opts.callback_group = cb_realtime_;
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        cmd_vel_topic_, rclcpp::QoS(10),
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) { this->cmdVelCallback(msg); }, rt_opts);
    rclcpp::SubscriptionOptions fast_opts;
    fast_opts.callback_group = cb_fast_;
    estop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        estop_topic_, rclcpp::QoS(10), std::bind(&X30NavBridge::onEstopMessage, this, _1), fast_opts);

    imu_pub_  = this->create_publisher<sensor_msgs::msg::Imu>(p + "/imu/data", rclcpp::SensorDataQoS());
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(p + "/leg_odom", rclcpp::SensorDataQoS());
    robot_state_pub_       = this->create_publisher<std_msgs::msg::Int32>(p + "/basic_state", 10);
    gait_state_pub_        = this->create_publisher<std_msgs::msg::Int32>(p + "/gait_state", 10);
    body_height_state_pub_ = this->create_publisher<std_msgs::msg::Int32>(p + "/body_height_state", 10);
    charge_state_pub_      = this->create_publisher<std_msgs::msg::Int32>(p + "/charge_manager_state", 10);
    battery_pub_           = this->create_publisher<sensor_msgs::msg::BatteryState>(p + "/battery_state", 10);
    joint_pub_             = this->create_publisher<sensor_msgs::msg::JointState>(p + "/joint_states", rclcpp::SensorDataQoS());
    diag_pub_              = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(p + "/diagnostics", 10);
    estop_active_pub_      = this->create_publisher<std_msgs::msg::Bool>(p + "/estop_active", rclcpp::QoS(1).reliable().transient_local());
    if (!ros_heartbeat_topic_.empty() && ros_heartbeat_rate_hz_ > 0.0) {
        ros_heartbeat_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(ros_heartbeat_topic_, 10);
        ros_heartbeat_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / ros_heartbeat_rate_hz_)),
            [this]() { this->publishRosHeartbeat(); }, cb_diag_);
    }
    diag_timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {
        this->updateEstopState(std::chrono::steady_clock::now());
        this->publishDiagnostics();
        this->publishEstopActive();
    }, cb_diag_);
    publishEstopActive();  // latched initial value (false)

    if (publish_tf_) tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    auto svc = rmw_qos_profile_services_default;
    stand_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "~/stand", std::bind(&X30NavBridge::handleStandRequest, this, _1, _2), svc, cb_slow_);
    lie_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "~/lie", std::bind(&X30NavBridge::handleLieRequest, this, _1, _2), svc, cb_slow_);
    set_gait_srv_ = this->create_service<rcl_interfaces::srv::SetParameters>(
        "~/set_gait", std::bind(&X30NavBridge::handleSetGaitRequest, this, _1, _2), svc, cb_slow_);
    set_body_height_srv_ = this->create_service<rcl_interfaces::srv::SetParameters>(
        "~/set_body_height", std::bind(&X30NavBridge::handleSetBodyHeightRequest, this, _1, _2), svc, cb_slow_);
    charge_command_srv_ = this->create_service<rcl_interfaces::srv::SetParameters>(
        "~/charge_command", std::bind(&X30NavBridge::handleChargeCommandRequest, this, _1, _2), svc, cb_slow_);
    soft_estop_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "~/soft_estop", std::bind(&X30NavBridge::handleSoftEstopRequest, this, _1, _2), svc, cb_fast_);
    release_control_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "~/release_control", std::bind(&X30NavBridge::handleReleaseControlRequest, this, _1, _2), svc, cb_fast_);
    acquire_control_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "~/acquire_control", std::bind(&X30NavBridge::handleAcquireControlRequest, this, _1, _2), svc, cb_fast_);
    enable_motion_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "~/enable_motion", std::bind(&X30NavBridge::handleEnableMotionRequest, this, _1, _2), svc, cb_fast_);
}

void X30NavBridge::shutdown() {
    if (!running_.load()) return;
    RCLCPP_INFO(get_logger(), "x30_nav_bridge shutting down (zero velocity, heartbeat stops)");
    running_ = false;
    charge_state_cv_.notify_all();
    if (control_heartbeat_thread_.joinable()) control_heartbeat_thread_.join();
    if (recv_thread_.joinable()) recv_thread_.join();
    if (charge_recv_thread_.joinable()) charge_recv_thread_.join();
    if (charge_query_thread_.joinable()) charge_query_thread_.join();

    if (motion_udp_.isOpen()) {
        for (uint32_t code : {CMD_VEL_FORWARD, CMD_VEL_LATERAL, CMD_VEL_YAW}) {
            auto zero_cmd = makeSimpleCommand(code, 0);
            motion_udp_.send(&zero_cmd, sizeof(zero_cmd));
        }
    }
    motion_udp_.close();
    percept_udp_.close();
}

}  // namespace nav_bridge
