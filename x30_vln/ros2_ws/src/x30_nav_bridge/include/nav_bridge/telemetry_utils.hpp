#pragma once

/// @file telemetry_utils.hpp
/// @brief ROS-free helpers for the fork: error bits, e-stop rule, state names, battery unit conversion.

#include <cstdint>
#include <string>
#include <vector>

#include "nav_bridge/x30_protocol.hpp"

namespace nav_bridge {
namespace telemetry {

inline const std::vector<const char *> &errorBitNames() {
    static const std::vector<const char *> names = {
        "imu_error",        "wifi_error",     "driver_heat_warn", "driver_error", "motor_heat_warn",
        "battery_low_warn", "battery_heat_warn", "gpio_error",    "cpu_heat_warn", "cpu_freq_warn"};
    return names;
}

/// Comma separated names of the set bits in RcsData.error_state ("" when none).
inline std::string errorBitsToString(uint32_t error_state) {
    std::string out;
    const auto &names = errorBitNames();
    for (size_t i = 0; i < names.size(); ++i) {
        if (error_state & (1u << i)) {
            if (!out.empty()) out += ",";
            out += names[i];
        }
    }
    return out;
}

/// Bits that must stop autonomy (heartbeat timeout, IMU timeout, driver fault); the warn bits only raise a diagnostic WARN.
constexpr uint32_t kCriticalErrorMask = (1u << 0) | (1u << 1) | (1u << 3) | (1u << 7);

inline const char *emergencySourceName(uint8_t source) {
    switch (source) {
        case 0: return "none";
        case 1: return "tail soft e-stop";
        case 2: return "autonomous during motion";
        case 3: return "autonomous during navigation";
        case 4: return "autonomous during charging";
        case 5: return "triggered by client";
        case 6: return "triggered by RAS";
        default: return "unknown";
    }
}

/// Real joint locks: sources 1, 4, 5, 6. Sources 2 and 3 show up during gait switches without a lock (dogx30 field note).
inline bool isLockEmergencySource(uint8_t source) { return source == 1 || source == 4 || source == 5 || source == 6; }

/// Latched-safety input rule (digest 4.3 item 8): basic_state 6, or a locking emergency source.
inline bool estopCondition(uint8_t basic_state, uint8_t emergency_source) {
    return basic_state == static_cast<uint8_t>(x30_protocol::BasicState::SOFT_ESTOP) ||
           isLockEmergencySource(emergency_source);
}

inline const char *basicStateName(uint8_t state) {
    using x30_protocol::BasicState;
    switch (static_cast<BasicState>(state)) {
        case BasicState::LYING_DOWN: return "LYING_DOWN";
        case BasicState::STANDING_UP: return "STANDING_UP";
        case BasicState::INITIAL_STAND: return "INITIAL_STAND";
        case BasicState::FORCE_STAND: return "FORCE_STAND";
        case BasicState::STEPPING: return "STEPPING";
        case BasicState::GOING_DOWN: return "GOING_DOWN";
        case BasicState::SOFT_ESTOP: return "SOFT_ESTOP";
        case BasicState::RL_MODE: return "RL_MODE";
        default: return "UNKNOWN";
    }
}

inline const char *gaitStateName(uint8_t state) {
    using x30_protocol::GaitState;
    switch (static_cast<GaitState>(state)) {
        case GaitState::WALK: return "WALK";
        case GaitState::OBSTACLE: return "OFFROAD";
        case GaitState::SLOPE: return "SLOPE";
        case GaitState::RUN: return "RUN";
        case GaitState::STAIR_SOLID: return "STAIR";
        case GaitState::STAIR_ACC: return "STAIR_MULTI";
        case GaitState::STAIR45_ACC: return "STAIR45_MULTI";
        case GaitState::L_WALK: return "L_WALK";
        case GaitState::MOUNTAIN: return "MOUNTAIN";
        case GaitState::SILENT: return "SILENT";
        case GaitState::L_STAIR: return "L_STAIR";
        default: return "UNKNOWN";
    }
}

/// Battery current: amperes before deeprcs 1.10.19, units of 10 mA from that version on (api p.25-26). `unit_a` converts.
inline double batteryCurrentAmps(int16_t raw, double unit_a) { return static_cast<double>(raw) * unit_a; }

/// remaining / nominal capacity are in units of 10 mAh -> Ah
inline double capacityAh(uint16_t raw_10mah) { return static_cast<double>(raw_10mah) * 0.01; }

}  // namespace telemetry
}  // namespace nav_bridge
