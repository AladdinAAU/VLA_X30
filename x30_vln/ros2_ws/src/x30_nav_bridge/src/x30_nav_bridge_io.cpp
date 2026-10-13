/// @file x30_nav_bridge_io.cpp
/// @brief x30 fork: UDP receive loop and telemetry decoding/publishing (odometry, IMU, joint states, battery,
///        diagnostics). Replaces the upstream receiveLoop and handle* functions.

#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <limits>
#include <sstream>

#include "nav_bridge/x30_nav_bridge.hpp"

namespace nav_bridge {

using namespace x30_protocol;
using Clock = std::chrono::steady_clock;

namespace {
using diagnostic_msgs::msg::DiagnosticStatus;
using diagnostic_msgs::msg::KeyValue;

void addKv(DiagnosticStatus &s, const std::string &k, const std::string &v) {
    KeyValue kv;
    kv.key   = k;
    kv.value = v;
    s.values.push_back(kv);
}
std::string num(double v, int prec = 2) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(prec);
    o << v;
    return o.str();
}
double ageMs(Clock::time_point now, Clock::time_point t) {
    if (t == Clock::time_point::min()) return -1.0;
    return std::chrono::duration<double, std::milli>(now - t).count();
}
}  // namespace

// ============================================================================
// Receive loop
// ============================================================================

void X30NavBridge::receiveLoop() {
    CommandMessage msg;
    uint64_t total_recv_count = 0, unknown_code_count = 0;

    while (running_) {
        const int received = motion_udp_.receive(&msg, sizeof(msg), 50);
        if (received < static_cast<int>(sizeof(CommandHead))) continue;
        ++total_recv_count;

        const bool was_connected = state_store_.connected();
        state_store_.markPacketReceived(Clock::now());
        if (!was_connected) {
            RCLCPP_INFO(get_logger(), "robot link up (first packet code=0x%04X, %d bytes)", msg.head.code, received);
        }
        const int data_size = received - static_cast<int>(sizeof(CommandHead));

        // memcpy into aligned locals: casting the packed buffer to structs with 8-byte members is undefined behaviour
        auto decode = [&](auto &out, const char *name) {
            if (msg.head.type == 1 && data_size >= static_cast<int>(sizeof(out)) &&
                msg.head.paramters_size >= sizeof(out)) {
                std::memcpy(&out, msg.data_buffer, sizeof(out));
                return true;
            }
            RCLCPP_WARN_ONCE(get_logger(), "%s size mismatch: got %d payload bytes (header says %u), expected %zu", name,
                             data_size, msg.head.paramters_size, sizeof(out));
            return false;
        };

        switch (msg.head.code) {
            case RECV_RCS_DATA: {
                RcsData d;
                if (decode(d, "RcsData")) handleRcsData(d);
                break;
            }
            case RECV_MOTION_STATE: {
                MotionStateData d;
                if (decode(d, "MotionStateData")) handleMotionState(d);
                break;
            }
            case RECV_CONTROLLER_SENSOR: {
                ControllerSensorData d;
                if (decode(d, "ControllerSensorData")) handleControllerSensor(d);
                break;
            }
            case RECV_CONTROLLER_SAFE: {
                ControllerSafeData d;
                if (decode(d, "ControllerSafeData")) {
                    std::lock_guard<std::mutex> lock(telemetry_mutex_);
                    latest_.safe      = d;
                    latest_.have_safe = true;
                    ++rx_safe_;
                }
                break;
            }
            case RECV_BATTERY: {
                BatterySensorData d;
                if (decode(d, "BatterySensorData")) handleBattery(d);
                break;
            }
            case RECV_BODY_HEIGHT: {
                if (msg.head.type == 0) {
                    int32_t v = 0;
                    std::memcpy(&v, &msg.head.paramters_size, sizeof(v));
                    handleBodyHeightState(v);
                }
                break;
            }
            default:
                if (++unknown_code_count <= 5) {
                    RCLCPP_DEBUG(get_logger(), "unhandled code 0x%04X type=%u size=%d", msg.head.code, msg.head.type, data_size);
                }
                break;
        }
    }
    RCLCPP_DEBUG(get_logger(), "receive thread exit (%lu packets, %lu unknown)", total_recv_count, unknown_code_count);
}

void X30NavBridge::processIncomingData() {}

// ============================================================================
// 0x1008 RcsData
// ============================================================================

void X30NavBridge::handleRcsData(const RcsData &data) {
    const auto now = Clock::now();
    is_nav_mode_.store(data.rcs_state_list.is_nav_mode != 0);
    last_emergency_source_.store(data.rcs_state_list.emergency_source);
    last_rcs_rx_.store(now);
    ++rx_rcs_;
    {
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        latest_.rcs      = data;
        latest_.have_rcs = true;
    }

    const int mode = data.rcs_state_list.is_nav_mode ? 1 : 0;
    const int prev = last_logged_nav_mode_.exchange(mode);
    if (prev >= 0 && prev != mode) {
        RCLCPP_INFO(get_logger(), "control mode: %s -> %s", prev ? "non-manual" : "manual", mode ? "non-manual" : "manual");
    }
    if (state_store_.markRcsReceived()) {
        char name[16] = {};
        std::memcpy(name, data.robot_name, 15);
        RCLCPP_INFO(get_logger(), "robot name '%s', control mode %s, total mileage %.1f m", name,
                    mode ? "non-manual" : "manual", data.total_mileage / 100.0);
    }
    if (data.error_state & telemetry::kCriticalErrorMask) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "robot error bits: %s",
                              telemetry::errorBitsToString(data.error_state).c_str());
    } else if (data.error_state) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "robot warning bits: %s",
                             telemetry::errorBitsToString(data.error_state).c_str());
    }
    updateEstopState(now);
}

// ============================================================================
// 0x1009 MotionStateData -> state topics and leg odometry
// ============================================================================

void X30NavBridge::handleMotionState(const MotionStateData &data) {
    const auto steady_now = Clock::now();
    const rclcpp::Time now = this->now();
    last_basic_state_.store(data.basic_state);
    last_motion_rx_.store(steady_now);
    ++rx_motion_;

    const auto transition = state_store_.updateMotionState(data.basic_state, data.gait_state);
    if (data.basic_state != transition.previous_basic_state || data.gait_state != transition.previous_gait_state) {
        RCLCPP_INFO(get_logger(), "state: %s/%s -> %s/%s", telemetry::basicStateName(transition.previous_basic_state),
                    telemetry::gaitStateName(transition.previous_gait_state), telemetry::basicStateName(data.basic_state),
                    telemetry::gaitStateName(data.gait_state));
    }

    std_msgs::msg::Int32 m;
    m.data = data.basic_state;
    robot_state_pub_->publish(m);
    m.data = data.gait_state;
    gait_state_pub_->publish(m);

    switch (static_cast<BasicState>(data.basic_state)) {
        case BasicState::LYING_DOWN: robot_state_.store(RobotState::LYING_DOWN); break;
        case BasicState::STANDING_UP: robot_state_.store(RobotState::STANDING_UP); break;
        case BasicState::INITIAL_STAND: robot_state_.store(RobotState::INITIAL_STAND); break;
        case BasicState::FORCE_STAND: robot_state_.store(RobotState::FORCE_STAND); break;
        case BasicState::STEPPING: robot_state_.store(RobotState::STEPPING); break;
        case BasicState::GOING_DOWN: robot_state_.store(RobotState::GOING_DOWN); break;
        case BasicState::SOFT_ESTOP: robot_state_.store(RobotState::SOFT_ESTOP); break;
        case BasicState::RL_MODE: robot_state_.store(RobotState::RL_MODE); break;
        default: break;
    }

    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = now;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id  = base_frame_id_;
    odom.pose.pose.position.x = data.leg_odom_pos[0];
    odom.pose.pose.position.y = data.leg_odom_pos[1];
    odom.pose.pose.position.z = 0.0;
    const double yaw = (odom_flip_yaw_ ? -1.0 : 1.0) * data.leg_odom_pos[2];
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.twist.twist.linear.x  = data.leg_odom_vel[0];
    odom.twist.twist.linear.y  = (odom_flip_vy_ ? -1.0 : 1.0) * data.leg_odom_vel[1];
    odom.twist.twist.angular.z = (odom_flip_yaw_ ? -1.0 : 1.0) * data.leg_odom_vel[2];
    // planar odometry: z / roll / pitch are not provided, flag them as unobserved with a huge variance
    odom.pose.covariance[0]  = odom_pose_cov_xy_;
    odom.pose.covariance[7]  = odom_pose_cov_xy_;
    odom.pose.covariance[14] = odom_unobserved_cov_;
    odom.pose.covariance[21] = odom_unobserved_cov_;
    odom.pose.covariance[28] = odom_unobserved_cov_;
    odom.pose.covariance[35] = odom_pose_cov_yaw_;
    odom.twist.covariance[0]  = odom_twist_cov_v_;
    odom.twist.covariance[7]  = odom_twist_cov_v_;
    odom.twist.covariance[14] = odom_unobserved_cov_;
    odom.twist.covariance[21] = odom_unobserved_cov_;
    odom.twist.covariance[28] = odom_unobserved_cov_;
    odom.twist.covariance[35] = odom_twist_cov_wz_;
    odom_pub_->publish(odom);

    if (publish_tf_ && tf_broadcaster_) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp            = now;
        tf.header.frame_id         = odom_frame_id_;
        tf.child_frame_id          = base_frame_id_;
        tf.transform.translation.x = data.leg_odom_pos[0];
        tf.transform.translation.y = data.leg_odom_pos[1];
        tf.transform.rotation      = odom.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }
    updateEstopState(steady_now);
}

void X30NavBridge::handleBodyHeightState(int32_t body_height_state) {
    const int8_t height_state = static_cast<int8_t>(body_height_state);
    const auto transition     = state_store_.updateBodyHeightState(height_state);
    if (height_state != transition.previous_body_height_state) {
        RCLCPP_INFO(get_logger(), "body height: %s", height_state == -1 ? "crawl" : "normal");
    }
    std_msgs::msg::Int32 m;
    m.data = body_height_state;
    body_height_state_pub_->publish(m);
}

// ============================================================================
// 0x100A ControllerSensorData -> IMU, joint states
// ============================================================================

void X30NavBridge::handleControllerSensor(const ControllerSensorData &data) {
    const rclcpp::Time now = this->now();
    last_sensor_rx_.store(Clock::now());
    ++rx_sensor_;
    const auto &imu = data.imu_data;

    sensor_msgs::msg::Imu msg;
    msg.header.stamp    = now;
    msg.header.frame_id = imu_frame_id_;
    constexpr double DEG2RAD = M_PI / 180.0;  // vendor spec: roll/pitch/yaw in degrees
    tf2::Quaternion q;
    q.setRPY(imu.roll * DEG2RAD, imu.pitch * DEG2RAD, imu.yaw * DEG2RAD);
    msg.orientation.x = q.x();
    msg.orientation.y = q.y();
    msg.orientation.z = q.z();
    msg.orientation.w = q.w();
    msg.angular_velocity.x    = imu.omega_x;
    msg.angular_velocity.y    = imu.omega_y;
    msg.angular_velocity.z    = imu.omega_z;
    msg.linear_acceleration.x = imu.acc_x;
    msg.linear_acceleration.y = imu.acc_y;
    msg.linear_acceleration.z = imu.acc_z;
    for (int i : {0, 4, 8}) {
        msg.orientation_covariance[i]         = imu_orientation_cov_;
        msg.angular_velocity_covariance[i]    = imu_angvel_cov_;
        msg.linear_acceleration_covariance[i] = imu_linacc_cov_;
    }
    imu_pub_->publish(msg);
    publishJointStates(data, now);
}

void X30NavBridge::publishJointStates(const ControllerSensorData &data, const rclcpp::Time &stamp) {
    if (joint_states_rate_hz_ <= 0.0) return;
    const auto t = Clock::now();
    if (last_joint_pub_ != Clock::time_point::min() &&
        std::chrono::duration<double>(t - last_joint_pub_).count() < 1.0 / joint_states_rate_hz_) {
        return;
    }
    last_joint_pub_ = t;
    static const char *kNames[12] = {"fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fr_hipx_joint",
                                     "fr_hipy_joint", "fr_knee_joint", "hl_hipx_joint", "hl_hipy_joint",
                                     "hl_knee_joint", "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint"};
    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    for (int i = 0; i < 12; ++i) {
        js.name.emplace_back(kNames[i]);
        js.position.push_back(data.joint_pos.data[i]);
        js.velocity.push_back(data.joint_vel.data[i]);
        js.effort.push_back(data.joint_tau.data[i]);
    }
    joint_pub_->publish(js);
}

// ============================================================================
// 0x21050F0A BatterySensorData
// ============================================================================

void X30NavBridge::handleBattery(const BatterySensorData &data) {
    ++rx_battery_;
    {
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        latest_.battery      = data;
        latest_.have_battery = true;
    }
    publishBattery(data);
}

void X30NavBridge::publishBattery(const BatterySensorData &d) {
    sensor_msgs::msg::BatteryState b;
    b.header.stamp = this->now();
    b.voltage      = static_cast<float>(d.voltage);
    b.current      = static_cast<float>(battery_current_sign_ * telemetry::batteryCurrentAmps(d.current, battery_current_unit_a_));
    b.charge       = std::numeric_limits<float>::quiet_NaN();
    b.capacity     = static_cast<float>(telemetry::capacityAh(d.remaining_capacity));
    b.design_capacity = static_cast<float>(telemetry::capacityAh(d.nominal_capacity));
    b.percentage   = std::min(1.0f, static_cast<float>(d.battery_level) / 100.0f);
    b.power_supply_status = b.current > 0.1f ? sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_CHARGING
                                             : sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_DISCHARGING;
    b.power_supply_health     = sensor_msgs::msg::BatteryState::POWER_SUPPLY_HEALTH_UNKNOWN;
    b.power_supply_technology = sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LION;
    b.present  = true;
    b.location = "main";
    float tsum = 0.f;
    for (float t : d.battery_temperature) tsum += t;
    b.temperature = tsum / 4.0f;
    battery_pub_->publish(b);
}

// ============================================================================
// Diagnostics (1 Hz)
// ============================================================================

void X30NavBridge::publishDiagnostics() {
    const auto now = Clock::now();
    LatestTelemetry t;
    {
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        t = latest_;
    }
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = this->now();

    // --- link
    DiagnosticStatus link;
    link.name        = "x30_nav_bridge/link";
    link.hardware_id = "x30_motion_host";
    const bool fresh = telemetryFresh(now);
    if (last_motion_rx_.load() == Clock::time_point::min()) {
        link.level   = DiagnosticStatus::ERROR;
        link.message = "no telemetry received (register this PC in network.toml on the motion host and restart jy_exe)";
    } else if (!fresh) {
        link.level   = DiagnosticStatus::ERROR;
        link.message = "telemetry stale: velocity forwarding forced to zero";
    } else {
        link.level   = DiagnosticStatus::OK;
        link.message = "telemetry alive";
    }
    addKv(link, "motion_state_age_ms", num(ageMs(now, last_motion_rx_.load()), 0));
    addKv(link, "rcs_age_ms", num(ageMs(now, last_rcs_rx_.load()), 0));
    addKv(link, "sensor_age_ms", num(ageMs(now, last_sensor_rx_.load()), 0));
    addKv(link, "packets_0x1008", std::to_string(rx_rcs_.load()));
    addKv(link, "packets_0x1009", std::to_string(rx_motion_.load()));
    addKv(link, "packets_0x100A", std::to_string(rx_sensor_.load()));
    addKv(link, "packets_0x100B", std::to_string(rx_safe_.load()));
    addKv(link, "packets_battery", std::to_string(rx_battery_.load()));
    arr.status.push_back(link);

    // --- motion
    DiagnosticStatus mot;
    mot.name        = "x30_nav_bridge/motion";
    mot.hardware_id = "x30_motion_host";
    const auto snap = state_store_.snapshot();
    mot.level       = DiagnosticStatus::OK;
    mot.message     = std::string(telemetry::basicStateName(snap.basic_state)) + "/" + telemetry::gaitStateName(snap.gait_state);
    if (estop_active_.load()) {
        mot.level   = DiagnosticStatus::ERROR;
        mot.message = "E-STOP ACTIVE";
    } else if (is_nav_mode_.load() && hasActiveControlSession()) {
        mot.level   = DiagnosticStatus::WARN;
        mot.message += " (non-manual mode: joystick axes are ignored)";
    }
    addKv(mot, "basic_state", std::to_string(snap.basic_state));
    addKv(mot, "gait_state", std::to_string(snap.gait_state));
    addKv(mot, "body_height", snap.body_height_state == -1 ? "crawl" : "normal");
    addKv(mot, "is_nav_mode", is_nav_mode_.load() ? "1" : "0");
    addKv(mot, "emergency_source", std::to_string(last_emergency_source_.load()));
    addKv(mot, "emergency_source_name", telemetry::emergencySourceName(last_emergency_source_.load()));
    addKv(mot, "estop_active", estop_active_.load() ? "true" : "false");
    addKv(mot, "soft_estop_latched", soft_estop_latched_.load() ? "true" : "false");
    addKv(mot, "control_owned", hasActiveControlSession() ? "true" : "false");
    addKv(mot, "forwarding_enabled", forwarding_enabled_.load() ? "true" : "false");
    addKv(mot, "hard_clamp_events", std::to_string(hard_clamp_events_.load()));
    addKv(mot, "gait_clamp_events", std::to_string(gait_clamp_events_.load()));
    arr.status.push_back(mot);

    // --- error bits
    DiagnosticStatus err;
    err.name        = "x30_nav_bridge/errors";
    err.hardware_id = "x30_motion_host";
    if (!t.have_rcs) {
        err.level   = DiagnosticStatus::STALE;
        err.message = "no RcsData yet";
    } else {
        const uint32_t e = t.rcs.error_state;
        err.level   = (e & telemetry::kCriticalErrorMask) ? DiagnosticStatus::ERROR : (e ? DiagnosticStatus::WARN : DiagnosticStatus::OK);
        err.message = e ? telemetry::errorBitsToString(e) : "none";
        addKv(err, "error_state", std::to_string(e));
        for (size_t i = 0; i < telemetry::errorBitNames().size(); ++i) {
            addKv(err, telemetry::errorBitNames()[i], (e & (1u << i)) ? "1" : "0");
        }
    }
    arr.status.push_back(err);

    // --- battery
    DiagnosticStatus bat;
    bat.name        = "x30_nav_bridge/battery";
    bat.hardware_id = "x30_battery";
    if (!t.have_battery) {
        bat.level   = DiagnosticStatus::STALE;
        bat.message = "no battery packet yet (0.5 Hz)";
    } else {
        const auto &b = t.battery;
        bat.level   = b.battery_level <= 15 ? DiagnosticStatus::ERROR : (b.battery_level <= 25 ? DiagnosticStatus::WARN : DiagnosticStatus::OK);
        bat.message = std::to_string(b.battery_level) + " %";
        addKv(bat, "level_percent", std::to_string(b.battery_level));
        addKv(bat, "voltage_v", std::to_string(b.voltage));
        addKv(bat, "current_raw", std::to_string(b.current));
        addKv(bat, "cycles", std::to_string(b.cycles));
        addKv(bat, "remaining_capacity_ah", num(telemetry::capacityAh(b.remaining_capacity)));
        addKv(bat, "nominal_capacity_ah", num(telemetry::capacityAh(b.nominal_capacity)));
        for (int i = 0; i < 4; ++i) addKv(bat, "temperature_" + std::to_string(i), num(b.battery_temperature[i], 1));
    }
    arr.status.push_back(bat);

    // --- temperatures (0x100B)
    DiagnosticStatus tmp;
    tmp.name        = "x30_nav_bridge/temperatures";
    tmp.hardware_id = "x30_motion_host";
    if (!t.have_safe) {
        tmp.level   = DiagnosticStatus::STALE;
        tmp.message = "no ControllerSafeData yet (1 Hz)";
    } else {
        float mmax = 0.f;
        int dmax   = 0;
        for (float v : t.safe.motor_temperature) mmax = std::max(mmax, v);
        for (uint8_t v : t.safe.driver_temperture) dmax = std::max<int>(dmax, v);
        tmp.level   = DiagnosticStatus::OK;
        tmp.message = "motor max " + num(mmax, 1) + " C, driver max " + std::to_string(dmax) + " C";
        addKv(tmp, "motor_temp_max_c", num(mmax, 1));
        addKv(tmp, "driver_temp_max_c", std::to_string(dmax));
        addKv(tmp, "cpu_temp_c", num(t.safe.cpu_info_.temperature, 1));
        addKv(tmp, "cpu_freq_mhz", num(t.safe.cpu_info_.frequency, 0));
    }
    arr.status.push_back(tmp);

    diag_pub_->publish(arr);
}

}  // namespace nav_bridge
