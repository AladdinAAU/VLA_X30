/// @file x30_nav_bridge_control.cpp
/// @brief x30 fork: the velocity path (hard clamp, accel/jerk limiter, per-gait axis mapping), the 50 Hz tick,
///        the heartbeat loop, explicit acquire/enable services and the latched e-stop logic.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "nav_bridge/x30_nav_bridge.hpp"

namespace nav_bridge {

using namespace x30_protocol;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int kEstopBurstGapMs = 20;
constexpr int kEstopResends    = 5;  // extra repeats, one per heartbeat-loop poll (about 100 ms each)
}  // namespace

// ============================================================================
// Velocity mapping
// ============================================================================

velmap::GaitSpeedLimit X30NavBridge::gaitLimitOrZero(uint8_t gait, bool &known) const {
    velmap::GaitSpeedLimit lim;
    known = gait_table_.find(gait, lim);
    return known ? lim : velmap::GaitSpeedLimit{};
}

void X30NavBridge::sendVelocityCommand(double vx, double vy, double vyaw) {
    const uint8_t gait = state_store_.gaitState();
    bool known         = false;
    const auto limits  = gaitLimitOrZero(gait, known);
    velmap::AxisTriple axes;
    if (!known) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "gait_state %u has no entry in the speed table: sending zero velocity", gait);
    } else {
        bool gait_saturated = false;
        axes = velmap::twistToAxes({vx, vy, vyaw}, limits, &gait_saturated);
        if (gait_saturated) {
            ++gait_clamp_events_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "command exceeds the full-scale speed of gait %u (%s): saturated at +-32767", gait,
                                 telemetry::gaitStateName(gait));
        }
    }
    // signs: velocityToAxis already negates vy and wz (robot: + = right, + = turn right)
    auto cmd_fwd = makeSimpleCommand(CMD_VEL_FORWARD, axes.fwd);
    auto cmd_lat = makeSimpleCommand(CMD_VEL_LATERAL, axes.lat);
    auto cmd_yaw = makeSimpleCommand(CMD_VEL_YAW, axes.yaw);
    motion_udp_.send(&cmd_fwd, sizeof(cmd_fwd));
    motion_udp_.send(&cmd_lat, sizeof(cmd_lat));
    motion_udp_.send(&cmd_yaw, sizeof(cmd_yaw));
}

bool X30NavBridge::telemetryFresh(Clock::time_point now) const {
    const auto limit = std::chrono::milliseconds(telemetry_stale_ms_);
    const auto m     = last_motion_rx_.load();
    const auto r     = last_rcs_rx_.load();
    if (m == Clock::time_point::min() || r == Clock::time_point::min()) return false;
    return (now - m) < limit && (now - r) < limit;
}

void X30NavBridge::sendCmdVelTick() {
    const auto now = Clock::now();
    const auto last_recv = state_store_.lastReceiveTime();
    if (state_store_.connected()) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_recv).count();
        if (elapsed > 2000) {
            state_store_.markDisconnected();
            robot_state_.store(RobotState::DISCONNECTED);
            RCLCPP_WARN(get_logger(), "robot link lost: no UDP data for %.1f s", elapsed / 1000.0);
        }
    }

    double dt = 1.0 / cmd_vel_rate_hz_;
    if (last_cmd_tick_ != Clock::time_point::min()) {
        dt = std::clamp(std::chrono::duration<double>(now - last_cmd_tick_).count(), 0.001, 0.1);
    }
    last_cmd_tick_ = now;
    if (limiter_reset_requested_.exchange(false)) limiter_.reset();

    updateEstopState(now);

    if (!hasActiveControlSession() || !isCmdVelForwardingAllowed()) {
        limiter_.reset();  // no axes in incompatible states (stand/lie/e-stop): dogx30 rule, digest 3.5
        return;
    }

    const bool fresh_telemetry = telemetryFresh(now);
    const bool was_stale       = telemetry_was_stale_.exchange(!fresh_telemetry);
    if (!fresh_telemetry && !was_stale) {
        RCLCPP_ERROR(get_logger(), "telemetry stale (> %d ms without 0x1008/0x1009): sending zero velocity", telemetry_stale_ms_);
    } else if (fresh_telemetry && was_stale) {
        RCLCPP_INFO(get_logger(), "telemetry alive again");
    }

    const bool allow = forwarding_enabled_.load() && !estop_active_.load() && fresh_telemetry;
    velmap::Twist2D out;
    if (allow && isControlInputFresh(now)) {
        velmap::Twist2D in{target_vx_.load(), target_vy_.load(), target_vyaw_.load()};
        bool saturated = false, non_finite = false;
        const auto clamped = velmap::clampHard(in, hard_limits_, saturated, non_finite);
        if (non_finite) RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "non-finite /cmd_vel component replaced by 0");
        if (saturated) {
            ++hard_clamp_events_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "cmd_vel (%.2f, %.2f, %.2f) exceeds the hard limits (+%.2f/-%.2f, %.2f, %.2f): clamped",
                                 in.vx, in.vy, in.wz, hard_limits_.max_vx_forward, hard_limits_.max_vx_backward,
                                 hard_limits_.max_vy, hard_limits_.max_wz);
        }
        out = limiter_.step(clamped, dt);
    } else {
        // not allowed, or the input went stale: hard zero (the limiter state is cleared; the robot's own controller
        // decelerates)
        limiter_.reset();
    }
    sendVelocityCommand(out.vx, out.vy, out.wz);
}

void X30NavBridge::controlHeartbeatLoop() {
    // Poll more often than the heartbeat period so due heartbeats go out promptly; evaluateControlPulse enforces the interval.
    const int poll_ms = std::max(10, std::min(100, heartbeat_interval_ms_ / 2));
    const auto poll_interval = std::chrono::milliseconds(poll_ms);
    while (running_) {
        const auto actions = evaluateControlPulse(Clock::now(), false);
        applyControlActions(actions);
        if (estop_resends_left_.load() > 0) {
            --estop_resends_left_;
            sendGaitCommand(CMD_SOFT_ESTOP);
        }
        std::this_thread::sleep_for(poll_interval);
    }
}

void X30NavBridge::onCmdVelReceived() {
    // Upstream latched control ownership here (any /cmd_vel started the heartbeat). The fork never does that:
    // control is taken only by ~/acquire_control, by the stand/lie/gait services, or by startup_acquire_control.
}

// ============================================================================
// Safety: e-stop, enable, acquire
// ============================================================================

void X30NavBridge::setForwardingEnabled(bool enabled, const char *reason) {
    const bool prev = forwarding_enabled_.exchange(enabled);
    limiter_reset_requested_ = true;
    if (prev != enabled) {
        RCLCPP_INFO(get_logger(), "velocity forwarding %s (%s)", enabled ? "ENABLED" : "disabled", reason);
    }
}

void X30NavBridge::publishEstopActive() {
    std_msgs::msg::Bool msg;
    msg.data = estop_active_.load();
    estop_active_pub_->publish(msg);
}

void X30NavBridge::updateEstopState(Clock::time_point now) {
    const bool have_telemetry = last_rcs_rx_.load() != Clock::time_point::min();
    const bool robot =
        have_telemetry && telemetry::estopCondition(last_basic_state_.load(), last_emergency_source_.load());
    robot_estop_.store(robot);
    if (robot) last_robot_estop_seen_.store(now);

    if (soft_estop_latched_.load()) {
        const auto clear = std::chrono::milliseconds(estop_clear_ms_);
        if (!robot && telemetryFresh(now) && (now - soft_estop_sent_time_.load()) > clear &&
            (now - last_robot_estop_seen_.load()) > clear) {
            soft_estop_latched_.store(false);
            RCLCPP_INFO(get_logger(), "soft e-stop latch cleared: the robot reports no e-stop condition any more");
        }
    }

    const bool active = robot || soft_estop_latched_.load();
    const bool prev   = estop_active_.exchange(active);
    if (active && !prev) {
        setForwardingEnabled(false, "e-stop active");
        RCLCPP_ERROR(get_logger(), "E-STOP ACTIVE (basic_state=%u emergency_source=%u: %s)", last_basic_state_.load(),
                     last_emergency_source_.load(), telemetry::emergencySourceName(last_emergency_source_.load()));
    } else if (!active && prev) {
        RCLCPP_WARN(get_logger(), "e-stop condition cleared; velocity forwarding stays disabled until ~/enable_motion true");
    }
    if (active != prev && estop_active_pub_) publishEstopActive();
}

void X30NavBridge::performSoftEstop(const char *reason) {
    const auto now = Clock::now();
    soft_estop_latched_.store(true);
    soft_estop_sent_time_.store(now);
    setForwardingEnabled(false, reason);
    estop_active_.store(true);
    publishEstopActive();
    acquireControl();  // the robot only obeys a live heartbeat session
    auto hb = makeSimpleCommand(CMD_HEARTBEAT, 0);
    motion_udp_.send(&hb, sizeof(hb));
    for (int i = 0; i < 3; ++i) {
        sendVelocityCommand(0.0, 0.0, 0.0);
        sendGaitCommand(CMD_SOFT_ESTOP);
        std::this_thread::sleep_for(std::chrono::milliseconds(kEstopBurstGapMs));
    }
    estop_resends_left_ = kEstopResends;
    RCLCPP_ERROR(get_logger(), "SOFT E-STOP sent (0x%08X) - reason: %s", CMD_SOFT_ESTOP, reason);
}

void X30NavBridge::handleSoftEstopRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                          std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    performSoftEstop("service ~/soft_estop");
    res->success = true;
    res->message = "Soft e-stop sent (burst + 5 repeats). Recover with ~/lie (toggle 0x21010202), then ~/stand.";
}

void X30NavBridge::onEstopMessage(const std_msgs::msg::Bool::SharedPtr msg) {
    if (!msg->data) return;
    const auto now = Clock::now();
    if (last_estop_msg_action_ != Clock::time_point::min() &&
        now - last_estop_msg_action_ < std::chrono::seconds(1)) {
        return;  // a persistent true is re-sent once per second at most
    }
    last_estop_msg_action_ = now;
    performSoftEstop("topic /x30/estop");
}

void X30NavBridge::handleReleaseControlRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                               std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    setForwardingEnabled(false, "release_control");
    const bool released = releaseControlOwnership();
    sendVelocityCommand(0.0, 0.0, 0.0);
    res->success = true;
    res->message = released ? "Control released, heartbeat stopped (the handheld controller can be used again; the robot's "
                              "link-loss chain will start)."
                            : "No active control session to release.";
    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
}

void X30NavBridge::handleAcquireControlRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                               std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    acquireControl();
    for (int i = 0; i < 30 && !hasActiveControlSession(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool fresh = telemetryFresh(Clock::now());
    res->success     = hasActiveControlSession();
    res->message = std::string("Heartbeat running (this competes with the handheld controller: single 0x21 source rule). "
                               "Telemetry ") + (fresh ? "alive." : "NOT alive: is this PC registered in network.toml?");
    RCLCPP_WARN(get_logger(), "control acquired by service: %s", res->message.c_str());
}

void X30NavBridge::handleEnableMotionRequest(const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
                                             std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
    if (!req->data) {
        setForwardingEnabled(false, "service ~/enable_motion false");
        res->success = true;
        res->message = "Velocity forwarding disabled; zero axes are sent.";
        return;
    }
    const auto now = Clock::now();
    updateEstopState(now);
    std::string why;
    if (!hasActiveControlSession()) why = "control not acquired (call ~/acquire_control or ~/stand first)";
    else if (!telemetryFresh(now)) why = "telemetry is stale";
    else if (estop_active_.load()) why = "e-stop is active";
    else if (!isCmdVelForwardingAllowed()) {
        why = std::string("robot state ") + telemetry::basicStateName(state_store_.basicState()) + "/" +
              telemetry::gaitStateName(state_store_.gaitState()) + " cannot take velocity (run ~/stand first)";
    }
    if (!why.empty()) {
        res->success = false;
        res->message = "Refused: " + why;
        RCLCPP_WARN(get_logger(), "enable_motion refused: %s", why.c_str());
        return;
    }
    setForwardingEnabled(true, "service ~/enable_motion true");
    res->success = true;
    res->message = "Velocity forwarding enabled (cmd_vel timeout " + std::to_string(cmd_vel_timeout_ms_) + " ms).";
}

}  // namespace nav_bridge
