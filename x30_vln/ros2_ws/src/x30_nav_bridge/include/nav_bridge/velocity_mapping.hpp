#pragma once

/// @file velocity_mapping.hpp
/// @brief ROS-free velocity path of x30_nav_bridge (added in the fork, not in upstream):
///        configurable per-gait speed table, hard clamp, ROS twist -> joystick axes, acceleration/jerk limiter.
///
/// Conventions (docs/robot/x30-interfaces.md 3.4): ROS twist is x forward, y LEFT, wz counter-clockwise. The robot's
/// joystick axes are + forward, + RIGHT (lateral) and + turn RIGHT (yaw), so lateral and yaw axes are negated.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace nav_bridge {
namespace velmap {

constexpr int32_t kAxisMax      = 32767;
constexpr int32_t kAxisDeadZone = 655;  ///< |axis| < 655 is treated as 0 by the robot

struct Twist2D {
    double vx{0.0};
    double vy{0.0};
    double wz{0.0};
};

struct AxisTriple {
    int32_t fwd{0};
    int32_t lat{0};
    int32_t yaw{0};
};

/// Full-scale (axis = +-32767) speed of one gait, normal body height.
struct GaitSpeedLimit {
    double max_forward{0.0};   ///< m/s
    double max_backward{0.0};  ///< m/s
    double max_lateral{0.0};   ///< m/s
    double max_yaw{0.0};       ///< rad/s
};

/// Gait code (basic gait_state values) -> full-scale limits.
/// Defaults are API Appendix B; gaits that Appendix B does not list (34 Silent, 36 L-Stair, 3 Run) take the LARGEST value
/// found in the three sources (api / nav_bridge upstream / dogx30): a larger full-scale value makes the robot slower than
/// requested, which is the safe error direction.
class GaitSpeedTable {
public:
    static GaitSpeedTable apiAppendixB() {
        GaitSpeedTable t;
        t.set(0, {1.2, 0.75, 0.15, 0.45});   // Walk
        t.set(1, {0.3, 0.4, 0.1, 0.5});      // Off-road
        t.set(2, {0.7, 0.75, 0.25, 0.5});    // Slope
        t.set(6, {0.3, 0.3, 0.2, 0.8});      // Stair
        t.set(7, {0.6, 0.3, 0.2, 0.8});      // Stair multi-frame
        t.set(8, {0.3, 0.2, 0.2, 0.8});      // 45 deg stair multi-frame
        t.set(32, {1.0, 1.0, 0.5, 1.2});     // L-Walk (RL)
        t.set(33, {1.0, 1.0, 0.8, 1.2});     // Mountain (RL)
        t.set(34, {1.0, 1.0, 0.8, 1.2});     // Silent (not in api): max(nb, dx)
        t.set(36, {1.0, 1.0, 0.8, 1.2});     // L-Stair (not in api): max(nb, dx)
        t.set(3, {2.5, 0.25, 0.8, 1.2});     // Run (not in api): max(nb, dx)
        return t;
    }

    void set(uint8_t gait, const GaitSpeedLimit &limit) { table_[gait] = limit; }

    bool find(uint8_t gait, GaitSpeedLimit &out) const {
        auto it = table_.find(gait);
        if (it == table_.end()) return false;
        out = it->second;
        return true;
    }

    static std::vector<uint8_t> knownGaits() { return {0, 1, 2, 3, 6, 7, 8, 32, 33, 34, 36}; }

private:
    std::map<uint8_t, GaitSpeedLimit> table_;
};

/// Absolute limits applied to every command regardless of gait.
struct HardLimits {
    double max_vx_forward{0.2};
    double max_vx_backward{0.1};
    double max_vy{0.1};
    double max_wz{0.3};
};

inline double clampAbs(double v, double limit, bool &saturated) {
    if (v > limit) {
        saturated = true;
        return limit;
    }
    if (v < -limit) {
        saturated = true;
        return -limit;
    }
    return v;
}

/// Replace non-finite components by 0 (returns true if any was replaced) and clamp to the hard limits.
inline Twist2D clampHard(const Twist2D &in, const HardLimits &lim, bool &saturated, bool &non_finite) {
    Twist2D out = in;
    non_finite = !(std::isfinite(in.vx) && std::isfinite(in.vy) && std::isfinite(in.wz));
    if (!std::isfinite(out.vx)) out.vx = 0.0;
    if (!std::isfinite(out.vy)) out.vy = 0.0;
    if (!std::isfinite(out.wz)) out.wz = 0.0;
    out.vx = out.vx >= 0.0 ? clampAbs(out.vx, lim.max_vx_forward, saturated)
                           : clampAbs(out.vx, lim.max_vx_backward, saturated);
    out.vy = clampAbs(out.vy, lim.max_vy, saturated);
    out.wz = clampAbs(out.wz, lim.max_wz, saturated);
    return out;
}

/// One axis: physical speed -> axis counts, dead zone applied (mirror of the upstream velocityToAxisValue).
inline int32_t velocityToAxisValue(double velocity, double max_vel, bool *saturated = nullptr) {
    if (!(max_vel > 0.0)) return 0;
    double ratio = velocity / max_vel;
    if (ratio > 1.0 || ratio < -1.0) {
        if (saturated) *saturated = true;
        ratio = ratio > 0 ? 1.0 : -1.0;
    }
    const int32_t raw = static_cast<int32_t>(ratio * kAxisMax);  // truncates toward zero
    if (std::abs(raw) < kAxisDeadZone) return 0;
    return raw;
}

/// ROS twist -> (forward, lateral, yaw) axis counts for the gait with the given limits.
inline AxisTriple twistToAxes(const Twist2D &t, const GaitSpeedLimit &g, bool *gait_saturated = nullptr) {
    AxisTriple a;
    a.fwd = velocityToAxisValue(t.vx, t.vx >= 0.0 ? g.max_forward : g.max_backward, gait_saturated);
    a.lat = velocityToAxisValue(-t.vy, g.max_lateral, gait_saturated);  // + axis = right, ROS +y = left
    a.yaw = velocityToAxisValue(-t.wz, g.max_yaw, gait_saturated);      // + axis = turn right, ROS +wz = CCW
    return a;
}

/// Per-axis slew limiter: acceleration limit (separate limit for decelerating), optional jerk limit.
/// jerk <= 0 disables the jerk limit. The jerk-limited profile never overshoots the target: the commanded
/// acceleration is also bounded by sqrt(2 * jerk * remaining_dv), so it can always ramp back to zero in time.
class AxisLimiter {
public:
    void configure(double max_accel, double max_decel, double max_jerk) {
        max_accel_ = max_accel;
        max_decel_ = max_decel;
        max_jerk_  = max_jerk;
    }

    void reset() {
        v_ = 0.0;
        a_ = 0.0;
    }

    double velocity() const { return v_; }
    double acceleration() const { return a_; }

    double step(double target, double dt) {
        if (dt <= 0.0) return v_;
        const double dv = target - v_;
        const bool reducing = (v_ != 0.0) && (dv * v_ < 0.0);
        const double a_limit = reducing ? max_decel_ : max_accel_;
        double a_mag = std::min(a_limit, std::abs(dv) / dt);
        if (max_jerk_ > 0.0) a_mag = std::min(a_mag, std::sqrt(2.0 * max_jerk_ * std::abs(dv)));
        double a_des = dv >= 0.0 ? a_mag : -a_mag;
        if (max_jerk_ > 0.0) {
            const double da = std::clamp(a_des - a_, -max_jerk_ * dt, max_jerk_ * dt);
            a_ += da;
        } else {
            a_ = a_des;
        }
        double v_next = v_ + a_ * dt;
        if ((v_next - target) * (v_ - target) < 0.0) {  // crossed the target within this step
            v_next = target;
            a_     = 0.0;
        }
        v_ = v_next;
        if (std::abs(v_ - target) < 1e-9) {
            v_ = target;
            a_ = 0.0;
        }
        return v_;
    }

private:
    double max_accel_{0.5};
    double max_decel_{0.5};
    double max_jerk_{0.0};
    double v_{0.0};
    double a_{0.0};
};

struct LimiterConfig {
    double accel_x{0.5};
    double accel_y{0.5};
    double accel_yaw{1.0};
    double decel_x{0.5};
    double decel_y{0.5};
    double decel_yaw{1.0};
    double jerk_x{2.0};
    double jerk_y{2.0};
    double jerk_yaw{4.0};
};

class VelocityLimiter {
public:
    void configure(const LimiterConfig &c) {
        x_.configure(c.accel_x, c.decel_x, c.jerk_x);
        y_.configure(c.accel_y, c.decel_y, c.jerk_y);
        w_.configure(c.accel_yaw, c.decel_yaw, c.jerk_yaw);
    }
    void reset() {
        x_.reset();
        y_.reset();
        w_.reset();
    }
    Twist2D step(const Twist2D &target, double dt) { return {x_.step(target.vx, dt), y_.step(target.vy, dt), w_.step(target.wz, dt)}; }
    Twist2D current() const { return {x_.velocity(), y_.velocity(), w_.velocity()}; }

private:
    AxisLimiter x_, y_, w_;
};

}  // namespace velmap
}  // namespace nav_bridge
