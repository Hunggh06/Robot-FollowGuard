#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include "motor_pid_control.h"

int main() {
    const WheelPIDConfig config{1.9f, 2.3f, 0.0f, 1.45f, 12.0f,
                                90.0f, 300u, 2.0f, 0.5f, 255.0f, 300.0f};
    WheelPIDState wheel{};
    const float start = stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 0u);
    assert(start >= 90.0f);
    const float repeated = stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 100u);
    assert(repeated >= 90.0f);
    const float expired = stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 400u);
    assert(expired < 90.0f);
    std::cout << "bounded_startup_no_rearm OK\n";

    wheel = WheelPIDState{};
    stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 100u);
    const float moving = stepWheelPID(wheel, config, 13.0f, 5.0f, 5.0f, 0.1f, 200u);
    assert(moving < 90.0f);
    const float later_stall = stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 250u);
    assert(later_stall < 90.0f);
    std::cout << "encoder_motion_ends_startup OK\n";

    wheel = WheelPIDState{};
    stepWheelPID(wheel, config, 13.0f, 13.0f, 13.0f, 0.1f, 100u);
    wheel.integral = 100.0f;
    const float reversing = stepWheelPID(wheel, config, -13.0f, 5.0f, 10.0f, 0.1f, 200u);
    assert(wheel.integral < 0.0f && reversing < 0.0f);
    assert(std::fabs(reversing) < 90.0f);
    const float restart = stepWheelPID(wheel, config, -13.0f, 5.0f, 0.0f, 0.1f, 500u);
    assert(restart <= -90.0f);
    std::cout << "reversal_resets_integral_waits_for_encoder_stop OK\n";

    const float stopped = stepWheelPID(wheel, config, 0.0f, 5.0f, 0.0f, 0.1f, 550u);
    assert(stopped == 0.0f && wheel.integral == 0.0f);
    assert(stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.1f, 600u) >= 90.0f);
    std::cout << "zero_stops_and_rearms_next_start OK\n";

    assert(stepWheelPID(wheel, config, std::numeric_limits<float>::infinity(),
                        0.0f, 0.0f, 0.1f, 700u) == 0.0f);
    assert(stepWheelPID(wheel, config, 13.0f, 0.0f, 0.0f, 0.0f, 800u) == 0.0f);
    wheel = WheelPIDState{};
    const float saturated = stepWheelPID(wheel, config, 1000.0f, 0.0f, 0.0f, 0.1f, 900u);
    assert(saturated <= 255.0f && wheel.integral == 0.0f);
    std::cout << "invalid_input_and_saturation_safe OK\n";
}
