// Decodes byte buffers assembled at the digest offsets (docs/robot/x30-interfaces.md 3.7) into the native structs.
#include <gtest/gtest.h>

#include <cstring>

#include "nav_bridge/telemetry_utils.hpp"
#include "nav_bridge/x30_protocol.hpp"

using namespace nav_bridge::x30_protocol;
namespace tu = nav_bridge::telemetry;

template <typename T>
static void put(uint8_t *b, size_t off, T v) {
    std::memcpy(b + off, &v, sizeof(v));
}

TEST(Layout, Sizes) {
    EXPECT_EQ(sizeof(RcsData), 88u);
    EXPECT_EQ(sizeof(MotionStateData), 60u);
    EXPECT_EQ(sizeof(ControllerSensorData), 184u);
    EXPECT_EQ(sizeof(ControllerSafeData), 68u);
    EXPECT_EQ(sizeof(BatterySensorData), 40u);  // upstream: packed, 39
}

TEST(Layout, Rcs) {
    uint8_t b[88] = {};
    std::memcpy(b, "X30", 3);
    put<int32_t>(b, 16, 1234);
    put<int64_t>(b, 24, 100);
    put<float>(b, 56, 0.5f);
    b[72] = 1;
    b[73] = 5;
    put<uint32_t>(b, 84, 0b100010);
    RcsData d;
    std::memcpy(&d, b, sizeof(d));
    EXPECT_EQ(d.current_mileage, 1234);
    EXPECT_EQ(d.current_run_time, 100);
    EXPECT_FLOAT_EQ(d.joystick[0], 0.5f);
    EXPECT_EQ(d.rcs_state_list.is_nav_mode, 1);
    EXPECT_EQ(d.rcs_state_list.emergency_source, 5);
    EXPECT_EQ(d.error_state, 0b100010u);
    EXPECT_EQ(tu::errorBitsToString(d.error_state), "wifi_error,battery_low_warn");
}

TEST(Layout, Motion) {
    uint8_t b[60] = {};
    b[0] = 16;
    b[1] = 33;
    put<float>(b, 12, 1.5f);
    put<float>(b, 16, -0.5f);
    put<float>(b, 20, 0.25f);
    put<float>(b, 24, 0.3f);
    put<float>(b, 36, 321.f);
    b[49] = 2;
    MotionStateData d;
    std::memcpy(&d, b, sizeof(d));
    EXPECT_EQ(d.basic_state, 16);
    EXPECT_EQ(d.gait_state, 33);
    EXPECT_FLOAT_EQ(d.leg_odom_pos[1], -0.5f);
    EXPECT_FLOAT_EQ(d.leg_odom_pos[2], 0.25f);
    EXPECT_FLOAT_EQ(d.leg_odom_vel[0], 0.3f);
    EXPECT_FLOAT_EQ(d.robot_distance, 321.f);
    EXPECT_EQ(d.task_state[1], 2);
}

TEST(Layout, SensorAndSafe) {
    uint8_t b[184] = {};
    put<int32_t>(b, 0, 123456);
    put<float>(b, 4, 1.0f);
    put<float>(b, 12, 90.0f);
    put<float>(b, 36, 9.8f);  // acc_z is the 9th float after the timestamp
    put<float>(b, 40 + 4 * 11, 0.11f);
    put<float>(b, 136 + 4 * 11, 0.33f);
    ControllerSensorData d;
    std::memcpy(&d, b, sizeof(d));
    EXPECT_EQ(d.imu_data.timestamp, 123456);
    EXPECT_FLOAT_EQ(d.imu_data.roll, 1.0f);
    EXPECT_FLOAT_EQ(d.imu_data.yaw, 90.0f);
    EXPECT_FLOAT_EQ(d.imu_data.acc_z, 9.8f);
    EXPECT_FLOAT_EQ(d.joint_pos.data[11], 0.11f);
    EXPECT_FLOAT_EQ(d.joint_tau.data[11], 0.33f);
    uint8_t s[68] = {};
    put<float>(s, 4 * 11, 41.f);
    s[48 + 11] = 51;
    put<float>(s, 60, 55.5f);
    put<float>(s, 64, 1992.f);
    ControllerSafeData sd;
    std::memcpy(&sd, s, sizeof(sd));
    EXPECT_FLOAT_EQ(sd.motor_temperature[11], 41.f);
    EXPECT_EQ(sd.driver_temperture[11], 51);
    EXPECT_FLOAT_EQ(sd.cpu_info_.temperature, 55.5f);
    EXPECT_FLOAT_EQ(sd.cpu_info_.frequency, 1992.f);
}

TEST(Layout, BatteryTemperaturesAtOffset24) {
    uint8_t b[40] = {};
    put<uint16_t>(b, 0, 76);
    put<int16_t>(b, 2, -12);
    put<uint16_t>(b, 8, 33);
    b[19] = 81;
    put<float>(b, 24, 25.f);
    put<float>(b, 36, 28.f);
    BatterySensorData d;
    std::memcpy(&d, b, sizeof(d));
    EXPECT_EQ(d.voltage, 76);
    EXPECT_EQ(d.current, -12);
    EXPECT_EQ(d.cycles, 33);
    EXPECT_EQ(d.battery_level, 81);
    EXPECT_FLOAT_EQ(d.battery_temperature[0], 25.f);
    EXPECT_FLOAT_EQ(d.battery_temperature[3], 28.f);
}

TEST(Command, SimpleCommandBytes) {
    auto c                   = makeSimpleCommand(CMD_HEARTBEAT, 0);
    const uint8_t expect[12] = {0x01, 0x00, 0x04, 0x21, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(std::memcmp(&c, expect, 12), 0);
    auto n                    = makeSimpleCommand(CMD_VEL_LATERAL, -32767);
    const uint8_t expect2[12] = {0x31, 0x01, 0x01, 0x21, 0x01, 0x80, 0xff, 0xff, 0, 0, 0, 0};
    EXPECT_EQ(std::memcmp(&n, expect2, 12), 0);
}

TEST(Safety, EstopRule) {
    EXPECT_TRUE(tu::estopCondition(6, 0));
    EXPECT_TRUE(tu::estopCondition(0, 1));
    EXPECT_TRUE(tu::estopCondition(16, 5));
    EXPECT_FALSE(tu::estopCondition(16, 2));
    EXPECT_FALSE(tu::estopCondition(16, 3));
    EXPECT_FALSE(tu::estopCondition(16, 0));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
