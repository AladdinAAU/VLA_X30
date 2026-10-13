/// @file x30_nav_bridge.cpp
/// @brief 绝影X30导航桥接实现 — UDP模拟手柄轴指令, 接收IMU/里程计/状态数据
/// x30 fork: constructor, lifecycle, velocity path, telemetry and e-stop moved to x30_nav_bridge_core/control/io.cpp;
/// this file keeps the upstream stand/lie, gait, body height and charge logic (see CHANGES.md).

#include "nav_bridge/x30_nav_bridge.hpp"

#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <unordered_map>

namespace nav_bridge {

using namespace x30_protocol;

namespace {

constexpr int kControlWarmupMs        = 400;
constexpr int kControlWarmupPulseMs   = 100;
constexpr int kGaitSwitchTimeoutMs    = 8000;
constexpr int kNonRlGaitSwitchTimeoutMs = 12000;  ///< 从 RL_MODE 切到 MPC 步态需经过 FORCE_STAND 过渡，留足时间
constexpr int kBodyHeightSwitchTimeoutMs = 4000;
constexpr int kCmdVelStateWaitMs      = 4000;
constexpr int kMotionStartRetryMs     = 1500;
constexpr int kChargePrepSettleMs        = 200;
constexpr int kChargeReceivePollMs       = 100;
constexpr int kChargeQueryPollMs         = 1000;
constexpr int kChargeVelSourceNavigation = 2;
constexpr int kChargeManualModeRetryMs   = 200;
constexpr int kChargeManualModeTimeoutMs = 20000;
constexpr uint8_t kChargeCommandStart    = 0;
constexpr uint8_t kChargeCommandStop     = 1;
constexpr uint8_t kChargeCommandReset    = 2;
constexpr uint8_t kChargeCommandQuery    = 3;

std::string escapeJsonString(const std::string &input) {
    std::string output;
    output.reserve(input.size());
    for (const char c : input) {
        switch (c) {
        case '\\':
            output += "\\\\";
            break;
        case '"':
            output += "\\\"";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            output += c;
            break;
        }
    }
    return output;
}


}  // namespace



void X30NavBridge::sendGaitCommand(uint32_t gait_cmd_code) {
    auto cmd = makeSimpleCommand(gait_cmd_code, 0);
    if (!motion_udp_.send(&cmd, sizeof(cmd))) {
        RCLCPP_ERROR(this->get_logger(), "UDP send of command 0x%08X failed (errno %d)", gait_cmd_code, errno);
    }
    RCLCPP_DEBUG(this->get_logger(), "发送控制指令: 0x%08X", gait_cmd_code);
}

// ============================================================================
// 定频发送轴指令
// ============================================================================

void X30NavBridge::applyControlActions(const ControlPulse &actions) {
    if (actions.send_heartbeat) {
        auto hb = makeSimpleCommand(CMD_HEARTBEAT, 0);
        if (!motion_udp_.send(&hb, sizeof(hb))) {
            std::lock_guard<std::mutex> lock(control_mutex_);
            last_heartbeat_sent_ = std::chrono::steady_clock::time_point::min();
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "控制心跳发送失败");
        }
    }

    if (actions.send_query) {
        auto query = makeSimpleCommand(CMD_QUERY_103, 0);
        if (!motion_udp_.send(&query, sizeof(query))) {
            std::lock_guard<std::mutex> lock(control_mutex_);
            control_query_sent_ = false;
            RCLCPP_WARN(this->get_logger(), "控制会话查询发送失败");
            return;
        }
        RCLCPP_DEBUG(this->get_logger(), "心跳已启动, 并已发送连接确认查询(0x21020001)");
    }

    if (actions.send_zero_velocity) {
        sendVelocityCommand(0.0, 0.0, 0.0);
    }
}

void X30NavBridge::acquireControl() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    control_latched_ = true;
}

bool X30NavBridge::releaseControlOwnership() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    bool had_control          = control_latched_ || control_session_started_;
    control_latched_          = false;
    control_session_started_  = false;
    control_query_sent_       = false;
    last_heartbeat_sent_      = std::chrono::steady_clock::time_point::min();
    return had_control;
}

bool X30NavBridge::hasActiveControlSession() const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return control_latched_ && control_session_started_;
}

void X30NavBridge::warmupControl(int warmup_ms, int pulse_ms) {
    acquireControl();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        applyControlActions(evaluateControlPulse(now, true));
        if (now >= deadline) {
            break;
        }
        // x30 fork: upstream computed the remaining time in whole milliseconds, so the last sleep could be 0 ms and the
        // loop then sent one heartbeat per iteration until the deadline (about 1000 packets in 1 ms), which fills the
        // receive queue of the motion host and drops the command sent right after the warm-up. Sleep at least 1 ms.
        const auto remaining_us =
            std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
        const auto sleep_us = std::max<int64_t>(1000, std::min<int64_t>(pulse_ms * 1000LL, remaining_us));
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
}

bool X30NavBridge::isControlInputFresh(std::chrono::steady_clock::time_point now) const {
    auto last_input = last_active_time_.load();
    return last_input != std::chrono::steady_clock::time_point::min() &&
           now - last_input < std::chrono::milliseconds(cmd_vel_timeout_ms_);
}

ActionResult X30NavBridge::executeActionWithCmdVelSuppressed(
    const char *action_name, const std::function<ActionResult()> &action) {
    enterCmdVelSuppression(action_name);
    struct ScopeExit {
        X30NavBridge *self;
        const char *name;
        ~ScopeExit() { self->exitCmdVelSuppression(name); }
    } scope_exit{this, action_name};

    return action();
}

void X30NavBridge::enterCmdVelSuppression(const char *action_name) {
    int previous_depth = cmd_vel_suppression_depth_.fetch_add(1);
    if (previous_depth == 0) {
        RCLCPP_DEBUG(this->get_logger(), "⏸️ %s 执行期间暂停转发 /cmd_vel UDP 轴指令", action_name);
    }
}

void X30NavBridge::exitCmdVelSuppression(const char *action_name) {
    int previous_depth = cmd_vel_suppression_depth_.fetch_sub(1);
    if (previous_depth <= 0) {
        cmd_vel_suppression_depth_.store(0);
        RCLCPP_WARN(this->get_logger(), "cmd_vel suppression depth underflow after %s", action_name);
        return;
    }

    if (previous_depth == 1) {
        RCLCPP_DEBUG(this->get_logger(), "▶️ %s 执行结束，恢复 /cmd_vel UDP 轴指令转发", action_name);
    }
}

bool X30NavBridge::isCmdVelSuppressed() const {
    return cmd_vel_suppression_depth_.load() > 0;
}

bool X30NavBridge::isSupportedNavigationGait(uint8_t gait) const {
    switch (static_cast<GaitState>(gait)) {
        case GaitState::WALK:
        case GaitState::OBSTACLE:
        case GaitState::SLOPE:
        case GaitState::RUN:
        case GaitState::STAIR_SOLID:
        case GaitState::STAIR_ACC:
        case GaitState::STAIR45_ACC:
        case GaitState::L_WALK:
        case GaitState::MOUNTAIN:
        case GaitState::SILENT:
        case GaitState::L_STAIR:
            return true;
        default:
            return false;
    }
}

bool X30NavBridge::isCrawlCompatibleGait(uint8_t gait) const
{
    return gait == static_cast<uint8_t>(GaitState::WALK) || gait == static_cast<uint8_t>(GaitState::SLOPE);
}

bool X30NavBridge::isBodyHeightSwitchAllowedGait(uint8_t gait) const
{
    return gait != static_cast<uint8_t>(GaitState::L_WALK) &&
           gait != static_cast<uint8_t>(GaitState::MOUNTAIN) &&
           gait != static_cast<uint8_t>(GaitState::SILENT) &&
           gait != static_cast<uint8_t>(GaitState::L_STAIR);
}

bool X30NavBridge::isCmdVelCompatibleState(uint8_t basic_state, uint8_t gait_state) const {
    // 按手册 1.2.3.2: 踏步状态(STEPPING)下可发轴指令控制速度, 适用于所有步态.
    // RL 类步态(L_WALK/MOUNTAIN/SILENT/L_STAIR)切换后进入 RL_MODE, 同样允许速度转发.
    if (basic_state == static_cast<uint8_t>(BasicState::STEPPING) &&
        isSupportedNavigationGait(gait_state)) {
        return true;
    }
    auto is_rl_gait = [](uint8_t g) {
        return g == static_cast<uint8_t>(GaitState::L_WALK) ||
               g == static_cast<uint8_t>(GaitState::MOUNTAIN) ||
               g == static_cast<uint8_t>(GaitState::SILENT) ||
               g == static_cast<uint8_t>(GaitState::L_STAIR);
    };
    return basic_state == static_cast<uint8_t>(BasicState::RL_MODE) && is_rl_gait(gait_state);
}

bool X30NavBridge::isCmdVelForwardingAllowed() const {
    if (isCmdVelSuppressed()) {
        return false;
    }

    auto snapshot = state_store_.snapshot();
    return isCmdVelCompatibleState(snapshot.basic_state, snapshot.gait_state);
}

bool X30NavBridge::waitForCmdVelCompatibleState(int timeout_ms) const {
    return state_store_.waitForState(
        [this](uint8_t basic_state, uint8_t gait_state) {
            return isCmdVelCompatibleState(basic_state, gait_state);
        },
        timeout_ms);
}

ActionResult X30NavBridge::setNavigationGait(uint8_t gait) {
    if (!isSupportedNavigationGait(gait)) {
        return {false, "Unsupported navigation gait."};
    }

    auto snapshot = state_store_.snapshot();
    if (!snapshot.connected) {
        return {false, "Robot is not connected."};
    }

    if (snapshot.body_height_state == static_cast<int8_t>(BodyHeightState::CRAWL) && !isCrawlCompatibleGait(gait))
    {
        return {false, "只有「行走WALK」和「斜坡SLOPE」模式允许机身高度为「匍匐CRAWL」。"};
    }

    // 按手册 1.2.6: RL 类步态 (L_WALK/MOUNTAIN/SILENT/L_STAIR) 切换后机器人进入 RL_MODE,
    // 普通步态 (WALK/SLOPE/OBSTACLE/STAIR*) 切换后机器人处于 STEPPING, 不会进入 RL_MODE.
    auto isRlGait = [](uint8_t g) {
        return g == static_cast<uint8_t>(GaitState::L_WALK) ||
               g == static_cast<uint8_t>(GaitState::MOUNTAIN) ||
               g == static_cast<uint8_t>(GaitState::SILENT) ||
               g == static_cast<uint8_t>(GaitState::L_STAIR);
    };

    const bool starts_from_force_stand =
        snapshot.basic_state == static_cast<uint8_t>(BasicState::FORCE_STAND);
    if (snapshot.basic_state != static_cast<uint8_t>(BasicState::RL_MODE) &&
        snapshot.basic_state != static_cast<uint8_t>(BasicState::STEPPING) &&
        !starts_from_force_stand) {
        return {false, "Gait switching is only allowed in RL_MODE, STEPPING, or FORCE_STAND."};
    }

    // 快速检查: 是否已经处于目标步态且基本状态与步态类型一致
    const uint8_t expected_basic_state = isRlGait(gait)
                                             ? static_cast<uint8_t>(BasicState::RL_MODE)
                                             : static_cast<uint8_t>(BasicState::STEPPING);
    if (snapshot.gait_state == gait && snapshot.basic_state == expected_basic_state) {
        return {true, "Robot is already in the requested navigation gait."};
    }

    uint32_t command = 0;
    const char *gait_name = "UNKNOWN";
    switch (static_cast<GaitState>(gait)) {
        case GaitState::WALK:
            command   = CMD_GAIT_WALK;
            gait_name = "WALK";
            break;
        case GaitState::OBSTACLE:
            command   = CMD_GAIT_OBSTACLE;
            gait_name = "OBSTACLE";
            break;
        case GaitState::SLOPE:
            command   = CMD_GAIT_SLOPE;
            gait_name = "SLOPE";
            break;
        case GaitState::RUN:
            command   = CMD_GAIT_RUN;
            gait_name = "RUN";
            break;
        case GaitState::STAIR_SOLID:
            command   = CMD_GAIT_STAIR;
            gait_name = "STAIR_SOLID";
            break;
        case GaitState::STAIR_ACC:
            command   = CMD_GAIT_STAIR_ACC;
            gait_name = "STAIR_ACC";
            break;
        case GaitState::STAIR45_ACC:
            command   = CMD_GAIT_STAIR45_ACC;
            gait_name = "STAIR45_ACC";
            break;
        case GaitState::L_WALK:
            command   = CMD_GAIT_L_WALK;
            gait_name = "L_WALK";
            break;
        case GaitState::MOUNTAIN:
            command   = CMD_GAIT_MOUNTAIN;
            gait_name = "MOUNTAIN";
            break;
        case GaitState::SILENT:
            command   = CMD_GAIT_SILENT;
            gait_name = "SILENT";
            break;
        case GaitState::L_STAIR:
            command   = CMD_GAIT_L_STAIR;
            gait_name = "L_STAIR";
            break;
        default:
            return {false, "Unsupported navigation gait."};
    }

    RCLCPP_DEBUG(this->get_logger(), "📌 请求切换步态到 %s(%u)", gait_name, gait);
    warmupControl(kControlWarmupMs, kControlWarmupPulseMs);

    if (starts_from_force_stand) {
        bool entered_stepping = false;
        int motion_retry_count = 0;
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kCmdVelStateWaitMs);

        while (std::chrono::steady_clock::now() < deadline) {
            auto snap = state_store_.snapshot();
            if (snap.basic_state == static_cast<uint8_t>(BasicState::STEPPING)) {
                entered_stepping = true;
                break;
            }

            if (snap.basic_state == static_cast<uint8_t>(BasicState::FORCE_STAND)) {
                ++motion_retry_count;
                RCLCPP_DEBUG(this->get_logger(),
                            "⏳ 步态 %s 从力控站立启动, 先发送开始运动指令 (attempt=%d)...",
                            gait_name, motion_retry_count);
                sendGaitCommand(CMD_MOTION);
            }

            auto now = std::chrono::steady_clock::now();
            auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            if (remaining <= 0) {
                break;
            }

            int wait_ms = std::min<int>(kMotionStartRetryMs, static_cast<int>(remaining));
            if (state_store_.waitForBasicState({BasicState::STEPPING}, wait_ms)) {
                entered_stepping = true;
                break;
            }
        }

        if (!entered_stepping) {
            auto final_snap = state_store_.snapshot();
            RCLCPP_WARN(this->get_logger(),
                        "⚠️ 步态 %s 从力控站立启动失败 (basic=%u, gait=%u)",
                        gait_name, final_snap.basic_state, final_snap.gait_state);
            return {false, std::string("Failed to start motion from FORCE_STAND for gait ") +
                              gait_name + "."};
        }

        auto snap = state_store_.snapshot();
        if (!isRlGait(gait) && snap.gait_state == gait) {
            RCLCPP_DEBUG(this->get_logger(), "✅ 机器人已进入踏步状态, 步态: %s", gait_name);
            return {true, std::string("Navigation gait switched to ") + gait_name + "."};
        }
    }

    sendGaitCommand(command);

    // 根据步态类型等待最终状态.
    // - RL 类步态: 直接等待 RL_MODE + 目标步态, 避免 gait 已经相同而过早进入短等待.
    // - 非 RL 步态: 先等步态反馈进入目标步态, 再确保 STEPPING + 目标步态.
    if (isRlGait(gait)) {
        if (!state_store_.waitForState(
                [gait](uint8_t basic_state, uint8_t gait_state) {
                    return basic_state == static_cast<uint8_t>(BasicState::RL_MODE) &&
                           gait_state == gait;
                },
                kGaitSwitchTimeoutMs)) {
            auto final_snap = state_store_.snapshot();
            RCLCPP_WARN(this->get_logger(),
                        "⚠️ 步态 %s 未进入RL最终状态 (basic=%u, gait=%u)",
                        gait_name, final_snap.basic_state, final_snap.gait_state);
            return {false, std::string("Timeout waiting for RL_MODE + ") + gait_name + "."};
        }
    } else {
        // 从 RL_MODE 切到 MPC 步态需经过 FORCE_STAND 过渡, 步态变化链为 RL_GAIT -> WALK -> target,
        // 因此使用更长的超时时间.
        if (!state_store_.waitForGaitState({static_cast<GaitState>(gait)}, kNonRlGaitSwitchTimeoutMs)) {
            return {false, std::string("Timeout waiting for gait state: ") + gait_name + "."};
        }

        auto is_target_stepping = [gait](uint8_t basic_state, uint8_t gait_state) {
            return basic_state == static_cast<uint8_t>(BasicState::STEPPING) &&
                   gait_state == gait;
        };

        bool entered_stepping = false;
        int motion_retry_count = 0;
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kCmdVelStateWaitMs);

        while (std::chrono::steady_clock::now() < deadline) {
            auto snap = state_store_.snapshot();
            if (is_target_stepping(snap.basic_state, snap.gait_state)) {
                entered_stepping = true;
                break;
            }

            if (snap.basic_state == static_cast<uint8_t>(BasicState::FORCE_STAND)) {
                ++motion_retry_count;
                RCLCPP_DEBUG(this->get_logger(),
                            "⏳ 步态已切换到 %s, 机器人处于力控站立, 发送开始运动指令 (attempt=%d)...",
                            gait_name, motion_retry_count);
                sendGaitCommand(CMD_MOTION);
            }

            auto now = std::chrono::steady_clock::now();
            auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            if (remaining <= 0) {
                break;
            }

            int wait_ms = std::min<int>(kMotionStartRetryMs, static_cast<int>(remaining));
            if (state_store_.waitForState(is_target_stepping, wait_ms)) {
                entered_stepping = true;
                break;
            }
        }

        if (!entered_stepping) {
            auto final_snap = state_store_.snapshot();
            RCLCPP_WARN(this->get_logger(),
                        "⚠️ 步态已切换到 %s, 但机器人未进入踏步状态 (basic=%u, gait=%u)",
                        gait_name, final_snap.basic_state, final_snap.gait_state);
            return {false, std::string("Gait switched to ") + gait_name +
                              ", but robot did not enter STEPPING in time."};
        }
        RCLCPP_DEBUG(this->get_logger(), "✅ 机器人已进入踏步状态, 步态: %s", gait_name);
    }

    return {true, std::string("Navigation gait switched to ") + gait_name + "."};
}

ActionResult X30NavBridge::setBodyHeight(int32_t requested_height_value)
{
    BodyHeightState target_state;
    const char *target_name = "UNKNOWN";
    switch (requested_height_value)
    {
    case 0:
        target_state = BodyHeightState::CRAWL;
        target_name = "CRAWL";
        break;
    case 2:
        target_state = BodyHeightState::NORMAL;
        target_name = "NORMAL";
        break;
    default:
        return {false, "Unsupported body height value. Use 0 (CRAWL) or 2 (NORMAL)."};
    }

    const auto snapshot = state_store_.snapshot();
    if (!snapshot.connected)
    {
        return {false, "Robot is not connected."};
    }

    if (snapshot.body_height_state == static_cast<int8_t>(target_state))
    {
        return {true, "Robot is already at the requested body height."};
    }

    if (target_state == BodyHeightState::CRAWL && !isCrawlCompatibleGait(snapshot.gait_state))
    {
        return {false, std::string("Crawl height is only allowed when current gait is WALK or SLOPE. Current gait=") + std::to_string(snapshot.gait_state) + "."};
    }
    if (!isBodyHeightSwitchAllowedGait(snapshot.gait_state))
    {
        return {false, "Body height switch is not supported in L_WALK, MOUNTAIN, SILENT, or L_STAIR gait."};
    }

    RCLCPP_DEBUG(this->get_logger(), "📌 请求切换身体高度到 %s", target_name);
    warmupControl(kControlWarmupMs, kControlWarmupPulseMs);

    auto cmd = makeSimpleCommand(CMD_HEIGHT_SWITCH, requested_height_value);
    if (!motion_udp_.send(&cmd, sizeof(cmd)))
    {
        return {false, "Failed to send body height switch command."};
    }

    RCLCPP_DEBUG(this->get_logger(), "发送身体高度切换指令: 0x%08X value=%d", CMD_HEIGHT_SWITCH, requested_height_value);

    if (!state_store_.waitForBodyHeightState({target_state}, kBodyHeightSwitchTimeoutMs))
    {
        const auto final_snapshot = state_store_.snapshot();
        RCLCPP_WARN(this->get_logger(),
                    "⚠️ 身体高度 %s 切换超时 (basic=%u, gait=%u, body_height=%d)",
                    target_name, final_snapshot.basic_state, final_snapshot.gait_state,
                    final_snapshot.body_height_state);
        return {false, std::string("Timeout waiting for body height state: ") + target_name + "."};
    }

    return {true, std::string("Body height switched to ") + target_name + "."};
}

bool X30NavBridge::sendChargeCommand(uint32_t code, int32_t value)
{
    auto cmd = makeSimpleCommand(code, value);
    return percept_udp_.send(&cmd, sizeof(cmd));
}

bool X30NavBridge::sendChargeVelocitySource(int32_t source)
{
    auto cmd = makeSimpleCommand(CMD_VEL_SOURCE, source);
    return percept_udp_.sendTo(&cmd, sizeof(cmd), charge_host_ip_, charge_config_port_);
}

bool X30NavBridge::waitForChargeResponse(uint64_t previous_seq, uint16_t &out_state)
{
    std::unique_lock<std::mutex> lock(charge_state_mutex_);

    while (running_)
    {
        if (charge_response_seq_ > previous_seq)
        {
            out_state = latest_charge_response_state_;
            return true;
        }

        charge_state_cv_.wait(lock);
    }
    return false;
}

bool X30NavBridge::waitForChargeResponseFor(uint64_t previous_seq, std::chrono::milliseconds timeout, uint16_t &out_state)
{
    std::unique_lock<std::mutex> lock(charge_state_mutex_);
    const bool ready = charge_state_cv_.wait_for(lock, timeout, [this, previous_seq]()
                                                 { return !running_ || charge_response_seq_ > previous_seq; });
    if (!ready || !running_ || charge_response_seq_ <= previous_seq)
    {
        return false;
    }

    out_state = latest_charge_response_state_;
    return true;
}

bool X30NavBridge::ensureManualMode(const char *reason)
{
    if (!is_nav_mode_.load())
    {
        return true;
    }

    RCLCPP_DEBUG(this->get_logger(), "🔁 %s后当前仍为非手动模式，开始切回手动", reason);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kChargeManualModeTimeoutMs);
    warmupControl(kControlWarmupMs, kControlWarmupPulseMs);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (!is_nav_mode_.load())
        {
            RCLCPP_INFO(this->get_logger(), "%s后已切回手动模式", reason);
            return true;
        }

        auto manual_cmd = makeSimpleCommand(CMD_MANUAL_MODE, 0);
        if (!motion_udp_.send(&manual_cmd, sizeof(manual_cmd)))
        {
            RCLCPP_ERROR(this->get_logger(), "切回手动模式失败：发送指令失败");
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kChargeManualModeRetryMs));
    }

    if (is_nav_mode_.load())
    {
        RCLCPP_ERROR(this->get_logger(), "切回手动模式超时：%s", reason);
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "%s后已切回手动模式", reason);
    return true;
}

bool X30NavBridge::switchToManualModeAfterCharge()
{
    return ensureManualMode("充电停止");
}

bool X30NavBridge::isFailureChargeState(uint16_t state) const
{
    switch (state)
    {
    case CHARGE_STATE_PILE_ERROR:
    case CHARGE_STATE_SAFETY_WARNING:
    case CHARGE_STATE_TAG_RECV_TIMEOUT:
    case CHARGE_STATE_MARK_LAUNCH_FAILED:
    case CHARGE_STATE_TAG_NO_VALUE:
    case CHARGE_STATE_GOTO_STACK_FAILED:
    case CHARGE_STATE_TAG_POSE_JUMP:
    case CHARGE_STATE_NO_CHARGE_PLUG:
    case CHARGE_STATE_NO_CHARGE_PLUG_STEP_BACK:
        return true;
    default:
        return false;
    }
}

bool X30NavBridge::isExpectedChargeStateForCommand(uint8_t command, uint16_t state) const
{
    switch (command)
    {
    case kChargeCommandStart:
        return state == CHARGE_STATE_DO_CHARGE_TASK || state == CHARGE_STATE_CHARGING;
    case kChargeCommandStop:
        return state == CHARGE_STATE_IDLE;
    case kChargeCommandReset:
        return state == CHARGE_STATE_IDLE;
    case kChargeCommandQuery:
        return true;
    default:
        return false;
    }
}

void X30NavBridge::chargeReceiveLoop()
{
    while (running_)
    {
        CommandHead response{};
        int received = percept_udp_.receive(&response, sizeof(response), kChargeReceivePollMs);
        if (received <= 0)
        {
            continue;
        }

        if (received < static_cast<int>(sizeof(CommandHead)))
        {
            continue;
        }

        if (response.type != 0)
        {
            continue;
        }

        if (response.code != CMD_CHARGE_MANAGER && response.code != CMD_CHARGE_MANAGER_QUERY)
        {
            continue;
        }

        uint16_t state = static_cast<uint16_t>(response.paramters_size);
        {
            std::lock_guard<std::mutex> lock(charge_state_mutex_);
            latest_charge_response_state_ = state;
            ++charge_response_seq_;
        }
        if (charge_state_pub_)
        {
            auto msg = std_msgs::msg::Int32();
            msg.data = static_cast<int32_t>(state);
            charge_state_pub_->publish(msg);
        }
        charge_state_cv_.notify_all();
    }
}

void X30NavBridge::chargeQueryLoop()
{
    const auto interval = std::chrono::milliseconds(std::max(100, charge_state_query_interval_ms_));
    const auto timeout = std::chrono::milliseconds(std::max(100, charge_state_query_timeout_ms_));

    while (running_)
    {
        std::this_thread::sleep_for(interval);
        if (!running_)
        {
            break;
        }

        std::unique_lock<std::mutex> service_lock(charge_service_mutex_, std::try_to_lock);
        if (!service_lock.owns_lock())
        {
            continue;
        }

        uint64_t before_seq = 0;
        {
            std::lock_guard<std::mutex> state_lock(charge_state_mutex_);
            before_seq = charge_response_seq_;
        }

        if (!sendChargeCommand(CMD_CHARGE_MANAGER_QUERY, CHARGE_COMMAND_QUERY_VALUE))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "自主充电状态周期查询发送失败");
            continue;
        }

        uint16_t state = CHARGE_STATE_IDLE;
        if (!waitForChargeResponseFor(before_seq, timeout, state))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000, "自主充电状态周期查询超时");
        }
    }
}

X30NavBridge::ControlPulse X30NavBridge::evaluateControlPulse(
    std::chrono::steady_clock::time_point now, bool force_heartbeat) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    ControlPulse actions;

    if (!control_latched_) {
        return actions;
    }

    if (!control_session_started_) {
        control_session_started_ = true;
        actions.session_started  = true;
        force_heartbeat          = true;
    }

    if (force_heartbeat ||
        last_heartbeat_sent_ == std::chrono::steady_clock::time_point::min() ||
        now - last_heartbeat_sent_ >= std::chrono::milliseconds(heartbeat_interval_ms_)) {
        last_heartbeat_sent_   = now;
        actions.send_heartbeat = true;
    }

    if (!control_query_sent_) {
        control_query_sent_  = true;
        actions.send_query   = true;
    }

    actions.active = true;
    return actions;
}




// ============================================================================
// 服务: 动作控制 (阻塞等待状态反馈)
// ============================================================================

void X30NavBridge::handleStandRequest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
    auto result = executeActionWithCmdVelSuppressed("stand", [this]() {
        auto action_result = action_executor_->stand(stand_target_gait_);
        if (action_result.success && !ensureManualMode("stand"))
        {
            action_result.success = false;
            action_result.message = "Stand succeeded, but robot did not confirm manual mode.";
        }
        return action_result; });
    res->success = result.success;
    res->message = result.message;
    if (result.success) {
        RCLCPP_INFO(this->get_logger(), "Stand 完成: %s", result.message.c_str());
    }
}

void X30NavBridge::handleLieRequest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    auto result  = executeActionWithCmdVelSuppressed("lie", [this]() {
        return action_executor_->lieDown();
    });
    res->success = result.success;
    res->message = result.message;
    if (result.success) {
        RCLCPP_INFO(this->get_logger(), "Lie 完成: 机器人已趴下");
    }
}


void X30NavBridge::handleSetGaitRequest(
    const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
    std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res) {

    // 字符串步态名称 → GaitState 数值的映射表 (大写, 与协议保持一致)
    static const std::unordered_map<std::string, uint8_t> kGaitNameMap = {
        {"WALK",        static_cast<uint8_t>(GaitState::WALK)},
        {"OBSTACLE",    static_cast<uint8_t>(GaitState::OBSTACLE)},
        {"SLOPE",       static_cast<uint8_t>(GaitState::SLOPE)},
        {"RUN",         static_cast<uint8_t>(GaitState::RUN)},
        {"STAIR_SOLID", static_cast<uint8_t>(GaitState::STAIR_SOLID)},
        {"STAIR_ACC",   static_cast<uint8_t>(GaitState::STAIR_ACC)},
        {"STAIR45_ACC", static_cast<uint8_t>(GaitState::STAIR45_ACC)},
        {"L_WALK",      static_cast<uint8_t>(GaitState::L_WALK)},
        {"MOUNTAIN",    static_cast<uint8_t>(GaitState::MOUNTAIN)},
        {"SILENT",      static_cast<uint8_t>(GaitState::SILENT)},
        {"L_STAIR",     static_cast<uint8_t>(GaitState::L_STAIR)},
    };

    if (req->parameters.size() != 1) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = false;
        result.reason     = "set_gait expects exactly one parameter named 'gait'.";

        if (req->parameters.empty()) {
            res->results.push_back(result);
        } else {
            res->results.assign(req->parameters.size(), result);
        }
        return;
    }

    for (const auto &param : req->parameters) {
        rcl_interfaces::msg::SetParametersResult result;

        if (param.name != "gait") {
            result.successful = false;
            result.reason     = "Unknown parameter '" + param.name + "'. Only 'gait' is supported.";
            res->results.push_back(result);
            continue;
        }

        // 解析步态值: 支持整数或字符串两种方式
        // rcl_interfaces::msg::ParameterType: BOOL=1, INT=2, DOUBLE=3, STRING=4, ...
        uint8_t gait_value = 0;
        bool parse_ok      = false;

        if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER) {
            int64_t raw = param.value.integer_value;
            if (raw >= 0 && raw <= 255) {
                gait_value = static_cast<uint8_t>(raw);
                parse_ok   = true;
            } else {
                result.successful = false;
                result.reason     = "Integer value " + std::to_string(raw) + " out of range [0, 255].";
            }
        } else if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_STRING) {
            // 转为大写以实现不区分大小写匹配
            std::string upper = param.value.string_value;
            std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            auto it = kGaitNameMap.find(upper);
            if (it != kGaitNameMap.end()) {
                gait_value = it->second;
                parse_ok   = true;
            } else {
                result.successful = false;
                result.reason     = "Unknown gait name '" + param.value.string_value +
                                    "'. Supported: WALK, OBSTACLE, SLOPE, RUN, STAIR_SOLID, "
                                    "STAIR_ACC, STAIR45_ACC, L_WALK, MOUNTAIN, SILENT, L_STAIR.";
            }
        } else {
            result.successful = false;
            result.reason     = "Unsupported parameter type " +
                                std::to_string(static_cast<int>(param.value.type)) +
                                ". Use INTEGER (type=2) or STRING (type=4).";
        }

        if (parse_ok) {
            auto action_result = executeActionWithCmdVelSuppressed("set_gait", [this, gait_value]() {
                return setNavigationGait(gait_value);
            });
            result.successful = action_result.success;
            result.reason     = action_result.message;
            if (action_result.success) {
                RCLCPP_INFO(this->get_logger(), "步态切换完成: %s", action_result.message.c_str());
            }
        }

        res->results.push_back(result);
    }
}

void X30NavBridge::handleSetBodyHeightRequest(
    const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
    std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res)
{

    static const std::unordered_map<std::string, int32_t> kBodyHeightNameMap = {
        {"CRAWL", 0},
        {"NORMAL", 2},
    };

    if (req->parameters.size() != 1)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = false;
        result.reason = "set_body_height expects exactly one parameter named 'body_height'.";

        if (req->parameters.empty())
        {
            res->results.push_back(result);
        }
        else
        {
            res->results.assign(req->parameters.size(), result);
        }
        return;
    }

    for (const auto &param : req->parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;

        if (param.name != "body_height")
        {
            result.successful = false;
            result.reason = "Unknown parameter '" + param.name + "'. Only 'body_height' is supported.";
            res->results.push_back(result);
            continue;
        }

        int32_t target_value = 0;
        bool parse_ok = false;

        if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER)
        {
            const int64_t raw = param.value.integer_value;
            if (raw == 0 || raw == 2)
            {
                target_value = static_cast<int32_t>(raw);
                parse_ok = true;
            }
            else
            {
                result.successful = false;
                result.reason = "Unsupported body_height integer " + std::to_string(raw) + ". Use 0 (CRAWL) or 2 (NORMAL).";
            }
        }
        else if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_STRING)
        {
            std::string upper = param.value.string_value;
            std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c)
                           { return static_cast<char>(std::toupper(c)); });
            auto it = kBodyHeightNameMap.find(upper);
            if (it != kBodyHeightNameMap.end())
            {
                target_value = it->second;
                parse_ok = true;
            }
            else
            {
                result.successful = false;
                result.reason = "Unknown body_height name '" + param.value.string_value + "'. Supported: CRAWL, NORMAL.";
            }
        }
        else
        {
            result.successful = false;
            result.reason = "Unsupported parameter type " + std::to_string(static_cast<int>(param.value.type)) + ". Use INTEGER (type=2) or STRING (type=4).";
        }

        if (parse_ok)
        {
            auto action_result =
                executeActionWithCmdVelSuppressed("set_body_height", [this, target_value]()
                                                  { return setBodyHeight(target_value); });
            result.successful = action_result.success;
            result.reason = action_result.message;
            if (action_result.success) {
                RCLCPP_INFO(this->get_logger(), "身体高度切换完成: %s", action_result.message.c_str());
            }
        }

        res->results.push_back(result);
    }
}

bool X30NavBridge::parseChargeCommandParameter(const rcl_interfaces::msg::Parameter &param,
                                               uint8_t &command,
                                               std::string &error) const
{
    if (param.name != "charge_command") {
        error = "Unknown parameter '" + param.name + "'. Only 'charge_command' is supported.";
        return false;
    }

    if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER) {
        const int64_t raw = param.value.integer_value;
        if (raw >= kChargeCommandStart && raw <= kChargeCommandQuery) {
            command = static_cast<uint8_t>(raw);
            return true;
        }
        error = "Integer charge_command " + std::to_string(raw) + " out of range [0, 3].";
        return false;
    }

    if (param.value.type == rcl_interfaces::msg::ParameterType::PARAMETER_STRING) {
        std::string upper = param.value.string_value;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

        if (upper == "START") {
            command = kChargeCommandStart;
            return true;
        }
        if (upper == "STOP") {
            command = kChargeCommandStop;
            return true;
        }
        if (upper == "RESET") {
            command = kChargeCommandReset;
            return true;
        }
        if (upper == "QUERY") {
            command = kChargeCommandQuery;
            return true;
        }

        error = "Unknown charge_command '" + param.value.string_value +
                "'. Supported: start, stop, reset, query, or integer 0..3.";
        return false;
    }

    error = "Unsupported charge_command parameter type " +
            std::to_string(static_cast<int>(param.value.type)) +
            ". Use INTEGER (type=2) or STRING (type=4).";
    return false;
}

std::string X30NavBridge::encodeChargeCommandResult(const ChargeCommandResult &result) const
{
    return "{\"charge_state\":" + std::to_string(result.charge_state) +
           ",\"state_name\":\"" + escapeJsonString(result.state_name) +
           "\",\"message\":\"" + escapeJsonString(result.message) + "\"}";
}

void X30NavBridge::handleChargeCommandRequest(
    const std::shared_ptr<rcl_interfaces::srv::SetParameters::Request> req,
    std::shared_ptr<rcl_interfaces::srv::SetParameters::Response> res)
{
    rcl_interfaces::msg::SetParametersResult result;

    if (!enable_charge_interface_) {
        result.successful = false;
        result.reason = "Charge interface disabled (parameter enable_charge_interface=false); it would contact the perception host.";
        res->results.push_back(result);
        return;
    }

    if (req->parameters.size() != 1) {
        result.successful = false;
        result.reason = "charge_command expects exactly one parameter named 'charge_command'.";
        if (req->parameters.empty()) {
            res->results.push_back(result);
        } else {
            res->results.assign(req->parameters.size(), result);
        }
        return;
    }

    uint8_t command = kChargeCommandStart;
    std::string error;
    if (!parseChargeCommandParameter(req->parameters.front(), command, error)) {
        result.successful = false;
        result.reason = error;
        res->results.push_back(result);
        return;
    }

    const ChargeCommandResult charge_result = executeChargeCommand(command);
    result.successful = charge_result.success;
    result.reason = encodeChargeCommandResult(charge_result);
    res->results.push_back(result);
}

X30NavBridge::ChargeCommandResult X30NavBridge::executeChargeCommand(uint8_t command)
{
    ChargeCommandResult result;
    uint32_t cmd_code = CMD_CHARGE_MANAGER;
    int32_t cmd_value = CHARGE_COMMAND_START_VALUE;

    switch (command)
    {
    case kChargeCommandStart:
        cmd_code = CMD_CHARGE_MANAGER;
        cmd_value = CHARGE_COMMAND_START_VALUE;
        break;
    case kChargeCommandStop:
        cmd_code = CMD_CHARGE_MANAGER;
        cmd_value = CHARGE_COMMAND_STOP_VALUE;
        break;
    case kChargeCommandReset:
        cmd_code = CMD_CHARGE_MANAGER;
        cmd_value = CHARGE_COMMAND_RESET_VALUE;
        break;
    case kChargeCommandQuery:
        cmd_code = CMD_CHARGE_MANAGER_QUERY;
        cmd_value = CHARGE_COMMAND_QUERY_VALUE;
        break;
    default:
        result.success = false;
        result.charge_state = CHARGE_STATE_IDLE;
        result.state_name = "invalid_command";
        result.message = "Invalid charge command. Allowed: 0(start),1(stop),2(reset),3(query).";
        return result;
    }

    const bool should_wait_until_target = command == kChargeCommandStart || command == kChargeCommandStop;

    RCLCPP_DEBUG(this->get_logger(), "⚡ 收到充电服务请求 command=%u", static_cast<unsigned int>(command));

    if (command == kChargeCommandStart)
    {
        auto snapshot = state_store_.snapshot();
        if (snapshot.basic_state == static_cast<uint8_t>(BasicState::RL_MODE))
        {
            RCLCPP_DEBUG(this->get_logger(), "⚡ START前检测到RL_MODE，先自动切换到可充电状态");
            auto exit_result = executeActionWithCmdVelSuppressed("charge_prepare_exit_rl", [this](){ return action_executor_->forceStand(); });
            if (!exit_result.success)
            {
                result.success = false;
                result.charge_state = CHARGE_STATE_IDLE;
                result.state_name = "failed_to_exit_rl";
                result.message = std::string("Failed to exit RL mode before charge START: ") + exit_result.message;
                return result;
            }
        }

        if (!sendChargeVelocitySource(kChargeVelSourceNavigation))
        {
            result.success = false;
            result.charge_state = CHARGE_STATE_IDLE;
            result.state_name = "vel_source_send_failed";
            result.message = "Failed to switch velocity source to navigation before charge START.";
            return result;
        }
        RCLCPP_DEBUG(this->get_logger(), "⚡ START前已切换速度源到导航模式");
        std::this_thread::sleep_for(std::chrono::milliseconds(kChargePrepSettleMs));
    }

    std::lock_guard<std::mutex> lock(charge_service_mutex_);
    uint64_t before_seq = 0;
    {
        std::lock_guard<std::mutex> state_lock(charge_state_mutex_);
        before_seq = charge_response_seq_;
    }

    RCLCPP_DEBUG(this->get_logger(), "⚡ 发送充电命令 command=%u code=0x%08X value=%d", static_cast<unsigned int>(command), cmd_code, cmd_value);
    if (!sendChargeCommand(cmd_code, cmd_value))
    {
        result.success = false;
        result.charge_state = CHARGE_STATE_IDLE;
        result.state_name = "send_failed";
        result.message = "Failed to send charge command.";
        return result;
    }

    uint16_t final_state = CHARGE_STATE_IDLE;
    while (rclcpp::ok() && running_)
    {
        if (!waitForChargeResponse(before_seq, final_state))
        {
            result.success = false;
            result.charge_state = CHARGE_STATE_IDLE;
            result.state_name = "interrupted";
            result.message = "Charge wait was interrupted by node shutdown.";
            return result;
        }

        result.charge_state = final_state;
        result.state_name = chargeStateToString(final_state);
        if (isExpectedChargeStateForCommand(command, final_state))
        {
            if (command == kChargeCommandStop)
            {
                if (!switchToManualModeAfterCharge())
                {
                    result.success = false;
                    result.message = "Charge STOP succeeded, but robot did not confirm manual mode.";
                    return result;
                }
                RCLCPP_INFO(this->get_logger(), "⚡ STOP完成后已切换到手动模式");
            }
            RCLCPP_INFO(this->get_logger(), "✅ 充电服务 command=%u 完成: state=0x%04X(%s)",
                        static_cast<unsigned int>(command), final_state, chargeStateToString(final_state));
            result.success = true;
            result.message = "Charge command accepted.";
            return result;
        }

        if (isFailureChargeState(final_state))
        {
            RCLCPP_WARN(this->get_logger(), "⚠️ 充电服务 command=%u 返回错误状态: 0x%04X(%s)",
                        static_cast<unsigned int>(command), final_state, chargeStateToString(final_state));
            result.success = false;
            result.message = "Charge manager reported a failure state.";
            return result;
        }

        if (!should_wait_until_target)
        {
            result.success = false;
            result.message = "Charge manager responded, but state did not match the command.";
            return result;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kChargeQueryPollMs));
        {
            std::lock_guard<std::mutex> state_lock(charge_state_mutex_);
            before_seq = charge_response_seq_;
        }
        if (!sendChargeCommand(CMD_CHARGE_MANAGER_QUERY, CHARGE_COMMAND_QUERY_VALUE))
        {
            result.success = false;
            result.message = "Failed to query charge state after command.";
            return result;
        }
    }

    result.success = false;
    result.charge_state = final_state;
    result.state_name = chargeStateToString(final_state);
    result.message = "Charge wait was interrupted.";
    return result;
}

}  // namespace nav_bridge
