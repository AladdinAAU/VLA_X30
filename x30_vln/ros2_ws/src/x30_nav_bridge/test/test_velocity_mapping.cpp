#include <gtest/gtest.h>

#include <cmath>

#include "nav_bridge/velocity_mapping.hpp"

using namespace nav_bridge::velmap;

static GaitSpeedLimit mountain() {
    GaitSpeedLimit l;
    GaitSpeedTable::apiAppendixB().find(33, l);
    return l;
}

TEST(Signs, PlusVyLeftAndPlusWzCcwGiveNegativeAxes) {
    auto a = twistToAxes({0.5, 0.4, 0.6}, mountain());
    EXPECT_GT(a.fwd, 0);
    EXPECT_LT(a.lat, 0);
    EXPECT_LT(a.yaw, 0);
    a = twistToAxes({-0.5, -0.4, -0.6}, mountain());
    EXPECT_LT(a.fwd, 0);
    EXPECT_GT(a.lat, 0);
    EXPECT_GT(a.yaw, 0);
}

TEST(Mapping, MountainValuesMatchTheDigest) {
    auto a = twistToAxes({0.5, 0.4, 0.6}, mountain());
    EXPECT_EQ(a.fwd, static_cast<int32_t>(0.5 * 32767));
    EXPECT_EQ(a.lat, -static_cast<int32_t>(0.5 * 32767));
    EXPECT_EQ(a.yaw, -static_cast<int32_t>(0.5 * 32767));
    bool sat = false;
    a = twistToAxes({5.0, -5.0, 5.0}, mountain(), &sat);
    EXPECT_TRUE(sat);
    EXPECT_EQ(a.fwd, 32767);
    EXPECT_EQ(a.lat, 32767);
    EXPECT_EQ(a.yaw, -32767);
}

TEST(Mapping, DeadZone) {
    EXPECT_EQ(velocityToAxisValue(0.019, 1.0), 0);
    EXPECT_NE(velocityToAxisValue(0.0205, 1.0), 0);
    EXPECT_EQ(velocityToAxisValue(-0.019, 1.0), 0);
}

TEST(Table, BackwardUsesBackwardLimitAndTableIsAppendixB) {
    GaitSpeedLimit walk;
    ASSERT_TRUE(GaitSpeedTable::apiAppendixB().find(0, walk));
    EXPECT_DOUBLE_EQ(walk.max_forward, 1.2);
    EXPECT_DOUBLE_EQ(walk.max_backward, 0.75);
    EXPECT_DOUBLE_EQ(walk.max_lateral, 0.15);
    auto a = twistToAxes({-0.375, 0, 0}, walk);
    EXPECT_EQ(a.fwd, -static_cast<int32_t>(0.375 / 0.75 * 32767));
    GaitSpeedLimit slope;
    ASSERT_TRUE(GaitSpeedTable::apiAppendixB().find(2, slope));
    EXPECT_DOUBLE_EQ(slope.max_forward, 0.7);  // upstream nav_bridge had 0.3
    GaitSpeedLimit none;
    EXPECT_FALSE(GaitSpeedTable::apiAppendixB().find(99, none));
}

TEST(HardClamp, ClampsAndFlags) {
    HardLimits h{0.2, 0.1, 0.1, 0.3};
    bool sat = false, nf = false;
    auto c = clampHard({1.0, -1.0, 2.0}, h, sat, nf);
    EXPECT_TRUE(sat);
    EXPECT_FALSE(nf);
    EXPECT_DOUBLE_EQ(c.vx, 0.2);
    EXPECT_DOUBLE_EQ(c.vy, -0.1);
    EXPECT_DOUBLE_EQ(c.wz, 0.3);
    sat = false;
    c = clampHard({-0.5, 0, 0}, h, sat, nf);
    EXPECT_DOUBLE_EQ(c.vx, -0.1);
    sat = false;
    c = clampHard({0.1, 0.05, 0.1}, h, sat, nf);
    EXPECT_FALSE(sat);
    c = clampHard({std::nan(""), 0.0, INFINITY}, h, sat, nf);
    EXPECT_TRUE(nf);
    EXPECT_DOUBLE_EQ(c.vx, 0.0);
    EXPECT_DOUBLE_EQ(c.wz, 0.0);
}

TEST(Limiter, AccelerationBoundAndConvergence) {
    VelocityLimiter lim;
    LimiterConfig c;
    c.jerk_x = 0.0;
    lim.configure(c);
    const double dt = 0.02;
    double prev = 0.0;
    double t    = 0.0;
    for (int i = 0; i < 200; ++i) {
        auto v = lim.step({0.4, 0, 0}, dt);
        EXPECT_LE((v.vx - prev) / dt, 0.5 + 1e-9);
        prev = v.vx;
        t += dt;
        if (v.vx >= 0.4) break;
    }
    EXPECT_NEAR(prev, 0.4, 1e-9);
    EXPECT_NEAR(t, 0.8, 0.05);  // 0.4 / 0.5 m/s2
}

TEST(Limiter, JerkBoundAndNoOvershoot) {
    VelocityLimiter lim;
    LimiterConfig c;
    c.jerk_x = 2.0;
    lim.configure(c);
    const double dt = 0.02;
    double prev_v = 0.0, prev_a = 0.0, max_v = 0.0;
    for (int i = 0; i < 400; ++i) {
        auto v         = lim.step({0.4, 0, 0}, dt);
        const double a = (v.vx - prev_v) / dt;
        // the jerk bound holds on every step except the last one, where the profile snaps onto the target
        if (std::abs(v.vx - 0.4) > 1e-12) {
            EXPECT_LE(std::abs(a - prev_a), 2.0 * dt + 1e-6) << "step " << i;
        } else {
            EXPECT_LE(std::abs(a - prev_a), 0.25) << "final snap step " << i;
        }
        EXPECT_LE(a, 0.5 + 1e-9);
        prev_v = v.vx;
        prev_a = a;
        max_v  = std::max(max_v, v.vx);
    }
    EXPECT_LE(max_v, 0.4 + 1e-9);
    EXPECT_NEAR(prev_v, 0.4, 1e-6);
}

TEST(Limiter, DecelAndReset) {
    VelocityLimiter lim;
    LimiterConfig c;
    c.jerk_x = 0.0;
    lim.configure(c);
    for (int i = 0; i < 100; ++i) lim.step({0.4, 0, 0}, 0.02);
    EXPECT_NEAR(lim.current().vx, 0.4, 1e-9);
    auto v = lim.step({0, 0, 0}, 0.02);
    EXPECT_NEAR(v.vx, 0.4 - 0.5 * 0.02, 1e-9);
    lim.reset();
    EXPECT_DOUBLE_EQ(lim.current().vx, 0.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
