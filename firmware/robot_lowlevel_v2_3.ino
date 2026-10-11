// ============================================================
// ESP32 + LDS02RR + MOTOR/ENCODER + WiFi micro-ROS  (v2.3)
// v2.3: lower-control PID timing/filter tune based on v2.2.
//
// PHẦN LIDAR: parser chính thức kaiaai/LDS, target 5 Hz
//   LDS02RR -> UART -> parser -> packet queue -> /telemetry
//
// PHẦN MOTOR: mới thêm
//   SUB /motor/target_rpm  (Float32MultiArray: [left, right])
//   PUB /motor/state       (Float32MultiArray: 8 phần tử)
//        [0] target_left   [1] target_right
//        [2] rpm_left      [3] rpm_right
//        [4] encoder_left  [5] encoder_right
//        [6] pwm_left      [7] pwm_right
//
// KHÔNG có /odom trên ESP32.
//
// v2: FF rieng 2 banh + breakaway boost chong ket de-pa.
// v2.3: PID dt clamp, RPM filter response, and gentler startup kick.
// Startup is encoder-gated and bounded; reversal resets each wheel integrator.
// ============================================================

#include <Arduino.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <micro_ros_kaia.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <kaiaai_msgs/msg/kaiaai_telemetry2.h>
#include <std_msgs/msg/float32_multi_array.h>   // MOTOR: thêm mới

#include <rmw_microros/rmw_microros.h>

struct WheelPIDConfig {
    float kp, ki, kd;
    float ff_gain, ff_offset;
    float kick_pwm;
    uint32_t kick_duration_ms;
    float moving_rpm, min_target_rpm;
    float max_pwm, integral_limit;
};

struct WheelPIDState {
    float integral = 0.0f;
    float previous_error = 0.0f;
    int direction = 0;
    bool startup_pending = false;
    bool kick_active = false;
    uint32_t kick_started_ms = 0;
};

inline float stepWheelPID(WheelPIDState& state, const WheelPIDConfig& config,
                          float target, float filtered_rpm, float raw_rpm,
                          float dt, uint32_t now_ms);

inline float stepWheelPID(WheelPIDState& state, const WheelPIDConfig& config,
                          float target, float filtered_rpm, float raw_rpm,
                          float dt, uint32_t now_ms) {
    if (!std::isfinite(target) || !std::isfinite(filtered_rpm) ||
        !std::isfinite(raw_rpm) || !std::isfinite(dt) || dt <= 0.0f ||
        std::fabs(target) < config.min_target_rpm) {
        state = WheelPIDState{};
        return 0.0f;
    }
    const int direction = target > 0.0f ? 1 : -1;
    const float error = target - filtered_rpm;
    if (state.direction != direction) {
        state.integral = 0.0f;
        state.previous_error = error;
        state.direction = direction;
        state.startup_pending = true;
        state.kick_active = false;
    }
    const float control_dt = std::min(dt, 0.25f);
    const float derivative = config.kd == 0.0f ? 0.0f
        : (error - state.previous_error) / control_dt;
    state.previous_error = error;
    const float ff = config.ff_gain * target + config.ff_offset * direction;
    const float pd = config.kp * error + config.kd * derivative;
    const float candidate = state.integral + config.ki * error * control_dt;
    const float candidate_output = pd + ff + candidate;
    if ((candidate_output < config.max_pwm && candidate_output > -config.max_pwm) ||
        (candidate_output >= config.max_pwm && error < 0.0f) ||
        (candidate_output <= -config.max_pwm && error > 0.0f)) {
        state.integral = candidate;
    }
    state.integral = std::max(-config.integral_limit,
                             std::min(config.integral_limit, state.integral));
    float output = pd + ff + state.integral;
    const bool moving_as_requested = raw_rpm * direction >= config.moving_rpm;
    if (moving_as_requested) {
        state.startup_pending = false;
        state.kick_active = false;
    } else if (state.startup_pending && std::fabs(raw_rpm) < config.moving_rpm) {
        state.startup_pending = false;
        state.kick_active = true;
        state.kick_started_ms = now_ms;
    }
    if (state.kick_active) {
        if (uint32_t(now_ms - state.kick_started_ms) >= config.kick_duration_ms) {
            state.kick_active = false;
        } else if (std::fabs(raw_rpm) < config.moving_rpm &&
                   output * direction < config.kick_pwm) {
            output = config.kick_pwm * direction;
        }
    }
    return std::max(-config.max_pwm, std::min(config.max_pwm, output));
}

// ================= WIFI =================
#include "wifi_credentials.h"

// ================= LDS02RR =================
HardwareSerial LIDAR(2);
#define LIDAR_RX        17
#define LIDAR_TX        16
#define LIDAR_MOT_EN    15
#define LIDAR_BAUD          115200
#define LIDAR_PWM_GPIO      LIDAR_MOT_EN
#define LIDAR_PWM_CHANNEL   2
#define LIDAR_PWM_FREQ      10000
#define LIDAR_PWM_BITS      8
#define LIDAR_PWM_DUTY      150
#define LIDAR_STARTUP_DUTY  255
#define LIDAR_STARTUP_MS    1000UL

// ================= TELEMETRY =================
#define LDS_BUF_MAX         400
uint8_t  ldsBuf[LDS_BUF_MAX];
uint16_t ldsSize = 0;
uint32_t telemSeq = 0;

// ================= MICRO-ROS =================
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;
rcl_publisher_t telem_pub;
rclc_executor_t executor;
kaiaai_msgs__msg__KaiaaiTelemetry2 telem_msg;

// ================= STATISTICS =================
uint32_t totalBytes         = 0;
uint32_t totalPublished     = 0;
uint32_t totalPublishError  = 0;
uint32_t lastStats          = 0;

// ############################################################
// ##  MOTOR + ENCODER
// ############################################################

#define IN1 22
#define IN2 23
#define ENA 32
#define IN3 2
#define IN4 4
#define ENB 33

#define ENC_L_A 27
#define ENC_L_B 14
#define ENC_R_A 26
#define ENC_R_B 25

// Kênh LEDC cố định 4,5 (timer 2) -> không đụng timer PWM LiDAR (kênh thấp, timer 0)
#define MOTOR_PWM_FREQ   1000
#define MOTOR_PWM_RES    8
#define MOTOR_PWM_CH_L   4
#define MOTOR_PWM_CH_R   5
#define MAX_PWM          255

#define ENCODER_TPR       227.0f   // do tay 4 lan: 222/225/227/228 (234 thieu ~3%)
#define WHEEL_DIAMETER_MM 65.0f

#define SWAP_MOTOR_SIDES true

float KpLeft  = 1.9f, KiLeft  = 2.3f, KdLeft  = 0.0f;
float KpRight = 1.9f, KiRight = 2.3f, KdRight = 0.0f;

#define PID_PERIOD_MS     100
#define INTEGRAL_LIMIT    300.0f
#define RPM_FILTER_ALPHA  0.25f
#define MOTOR_FF_LEFT     1.275f
#define MOTOR_FF_RIGHT    1.450f
#define MOTOR_FF_INTERCEPT 12.0f
// Feed-forward and 90 PWM are inherited from v2.2, not newly calibrated.
#define MOTOR_STARTUP_KICK_PWM 120.0f
#define MOTOR_STARTUP_KICK_MS  500UL
#define MOTOR_STARTUP_RPM_THRESHOLD 2.0f

// Quá thời gian này không có /motor/target_rpm -> tự dừng. 0 = tắt.
#define CMD_TIMEOUT_MS    1000UL
#define STATE_PERIOD_MS   100

rcl_publisher_t    state_pub;
rcl_subscription_t target_sub;

std_msgs__msg__Float32MultiArray state_msg;
std_msgs__msg__Float32MultiArray target_msg;

float stateBuf[8];
float targetBuf[2];
std_msgs__msg__MultiArrayDimension targetDimBuf[1];
char  targetDimLabel[16];

bool rosReady = false;

volatile long leftCount  = 0;
volatile long rightCount = 0;

float targetRPMLeft = 0, targetRPMRight = 0;
float rpmLeftFiltered = 0, rpmRightFiltered = 0;
float rpmLeftRaw = 0, rpmRightRaw = 0;
WheelPIDState leftPIDState, rightPIDState;
float pwmLeft = 0, pwmRight = 0;

unsigned long lastPIDTime = 0;
long previousLeftCount = 0, previousRightCount = 0;
unsigned long lastCmdMs = 0, lastStateMs = 0;

void IRAM_ATTR leftEncoderISR()
{
    bool A = digitalRead(ENC_L_A);
    bool B = digitalRead(ENC_L_B);
    if (A == B) leftCount++; else leftCount--;
}

void IRAM_ATTR rightEncoderISR()
{
    bool A = digitalRead(ENC_R_A);
    bool B = digitalRead(ENC_R_B);
    if (A == B) rightCount--; else rightCount++;
}

void setLeftMotor(float pwm)
{
    pwm = constrain(pwm, -MAX_PWM, MAX_PWM);
    if (pwm > 0) {
        digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
        ledcWrite(ENA, (uint32_t)constrain((int)pwm, 0, MAX_PWM));
    } else if (pwm < 0) {
        digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
        ledcWrite(ENA, (uint32_t)constrain((int)(-pwm), 0, MAX_PWM));
    } else {
        digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
        ledcWrite(ENA, 0);
    }
}

void setRightMotor(float pwm)
{
    pwm = constrain(pwm, -MAX_PWM, MAX_PWM);
    if (pwm > 0) {
        digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
        ledcWrite(ENB, (uint32_t)constrain((int)pwm, 0, MAX_PWM));
    } else if (pwm < 0) {
        digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
        ledcWrite(ENB, (uint32_t)constrain((int)(-pwm), 0, MAX_PWM));
    } else {
        digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
        ledcWrite(ENB, 0);
    }
}

void resetPID()
{
    leftPIDState = WheelPIDState{};
    rightPIDState = WheelPIDState{};
}

void stopMotors()
{
    setLeftMotor(0);
    setRightMotor(0);
    pwmLeft = pwmRight = 0;
    targetRPMLeft = targetRPMRight = 0;
    resetPID();
}

void calculateRPM(float dt)
{
    long pl, pr;
    noInterrupts();
    pl = leftCount;
    pr = rightCount;
    interrupts();

#if SWAP_MOTOR_SIDES
    long currentLeft = pr, currentRight = pl;
#else
    long currentLeft = pl, currentRight = pr;
#endif

    long deltaLeft  = currentLeft  - previousLeftCount;
    long deltaRight = currentRight - previousRightCount;
    previousLeftCount  = currentLeft;
    previousRightCount = currentRight;

    if (dt <= 0.001f) return;

    rpmLeftRaw  = ((float)deltaLeft  / ENCODER_TPR) * 60.0f / dt;
    rpmRightRaw = ((float)deltaRight / ENCODER_TPR) * 60.0f / dt;

    rpmLeftFiltered  = RPM_FILTER_ALPHA * rpmLeftRaw  + (1.0f - RPM_FILTER_ALPHA) * rpmLeftFiltered;
    rpmRightFiltered = RPM_FILTER_ALPHA * rpmRightRaw + (1.0f - RPM_FILTER_ALPHA) * rpmRightFiltered;
}

float calculatePIDLeft(float dt)
{
    const WheelPIDConfig config{KpLeft, KiLeft, KdLeft, MOTOR_FF_LEFT,
        MOTOR_FF_INTERCEPT, MOTOR_STARTUP_KICK_PWM, MOTOR_STARTUP_KICK_MS,
        MOTOR_STARTUP_RPM_THRESHOLD, 0.5f, MAX_PWM, INTEGRAL_LIMIT};
    return stepWheelPID(leftPIDState, config, targetRPMLeft,
                        rpmLeftFiltered, rpmLeftRaw, dt, millis());
}

float calculatePIDRight(float dt)
{
    const WheelPIDConfig config{KpRight, KiRight, KdRight, MOTOR_FF_RIGHT,
        MOTOR_FF_INTERCEPT, MOTOR_STARTUP_KICK_PWM, MOTOR_STARTUP_KICK_MS,
        MOTOR_STARTUP_RPM_THRESHOLD, 0.5f, MAX_PWM, INTEGRAL_LIMIT};
    return stepWheelPID(rightPIDState, config, targetRPMRight,
                        rpmRightFiltered, rpmRightRaw, dt, millis());
}

void updateMotorPID()
{
    unsigned long now = millis();
    if (now - lastPIDTime < PID_PERIOD_MS) return;

    float dt = (now - lastPIDTime) / 1000.0f;
    lastPIDTime = now;
    if (dt <= 0.001f) dt = 0.1f;

#if CMD_TIMEOUT_MS > 0
    if (now - lastCmdMs > CMD_TIMEOUT_MS &&
        (targetRPMLeft != 0.0f || targetRPMRight != 0.0f))
    {
        Serial.println("[MOTOR] cmd timeout -> STOP");
        stopMotors();
    }
#endif

    calculateRPM(dt);   // luôn cập nhật để /motor/state đúng khi đứng yên

    if (fabs(targetRPMLeft) < 0.1f && fabs(targetRPMRight) < 0.1f) {
        stopMotors();
        return;
    }

    pwmLeft  = calculatePIDLeft(dt);
    pwmRight = calculatePIDRight(dt);

#if SWAP_MOTOR_SIDES
    setLeftMotor(pwmRight);
    setRightMotor(pwmLeft);
#else
    setLeftMotor(pwmLeft);
    setRightMotor(pwmRight);
#endif
}

// ---- SUB /motor/target_rpm ----
void targetCallback(const void * msgin)
{
    const std_msgs__msg__Float32MultiArray * m =
        (const std_msgs__msg__Float32MultiArray *)msgin;

    if (m->data.size < 2) return;

    float l = m->data.data[0];
    float r = m->data.data[1];
    if (!isfinite(l) || !isfinite(r)) {
        stopMotors();
        return;
    }

    targetRPMLeft  = l;
    targetRPMRight = r;
    lastCmdMs      = millis();
    if (fabs(l) < 0.1f && fabs(r) < 0.1f) stopMotors();
}

// ---- PUB /motor/state ----
void publishMotorState()
{
    if (!rosReady) return;

    unsigned long now = millis();
    if (now - lastStateMs < STATE_PERIOD_MS) return;
    lastStateMs = now;

    long pL, pR;
    noInterrupts();
    pL = leftCount;
    pR = rightCount;
    interrupts();

#if SWAP_MOTOR_SIDES
    long encL = pR, encR = pL;
#else
    long encL = pL, encR = pR;
#endif

    stateBuf[0] = targetRPMLeft;
    stateBuf[1] = targetRPMRight;
    stateBuf[2] = rpmLeftFiltered;
    stateBuf[3] = rpmRightFiltered;
    stateBuf[4] = (float)encL;
    stateBuf[5] = (float)encR;
    stateBuf[6] = pwmLeft;
    stateBuf[7] = pwmRight;
    state_msg.data.size = 8;

    rcl_publish(&state_pub, &state_msg, NULL);
}

void setupMotors()
{
    pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
    pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

    if (!ledcAttachChannel(ENA, MOTOR_PWM_FREQ, MOTOR_PWM_RES, MOTOR_PWM_CH_L))
        Serial.println("[MOTOR] LEDC LEFT attach failed");
    if (!ledcAttachChannel(ENB, MOTOR_PWM_FREQ, MOTOR_PWM_RES, MOTOR_PWM_CH_R))
        Serial.println("[MOTOR] LEDC RIGHT attach failed");

    stopMotors();
    Serial.println("[MOTOR] ready");
}

void setupEncoders()
{
    pinMode(ENC_L_A, INPUT_PULLUP);
    pinMode(ENC_L_B, INPUT_PULLUP);
    pinMode(ENC_R_A, INPUT_PULLUP);
    pinMode(ENC_R_B, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(ENC_L_A), leftEncoderISR,  RISING);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), rightEncoderISR, RISING);
    Serial.println("[ENC] ready");
}

void initMotorMessages()
{
    memset(&state_msg, 0, sizeof(state_msg));
    state_msg.data.data     = stateBuf;
    state_msg.data.size     = 8;
    state_msg.data.capacity = 8;

    memset(&target_msg, 0, sizeof(target_msg));
    memset(targetDimBuf, 0, sizeof(targetDimBuf));
    targetDimBuf[0].label.data     = targetDimLabel;
    targetDimBuf[0].label.size     = 0;
    targetDimBuf[0].label.capacity = sizeof(targetDimLabel);

    target_msg.layout.dim.data     = targetDimBuf;
    target_msg.layout.dim.size     = 0;
    target_msg.layout.dim.capacity = 1;

    target_msg.data.data     = targetBuf;
    target_msg.data.size     = 0;
    target_msg.data.capacity = 2;
}

// ============================================================
// WIFI
// ============================================================
bool setupWiFi()
{
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.print("[WIFI] Connecting");
    uint32_t start = millis();

    while (WiFi.status() != WL_CONNECTED && millis() - start < 15000)
    {
        delay(300);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("[WIFI] FAILED");
        return false;
    }

    Serial.println("[WIFI] CONNECTED");
    Serial.print("[WIFI] IP: ");   Serial.println(WiFi.localIP());
    Serial.print("[WIFI] RSSI: "); Serial.print(WiFi.RSSI()); Serial.println(" dBm");
    return true;
}

// ============================================================
// MICRO-ROS
// ============================================================
bool setupMicroROS()
{
    Serial.println("[ROS] Starting micro-ROS");

    set_microros_wifi_transports(AGENT_IP, AGENT_PORT);
    delay(2000);

    if (rmw_uros_ping_agent(1000, 3) != RMW_RET_OK)
    {
        Serial.println("[ROS] Agent PING FAILED");
        return false;
    }
    Serial.println("[ROS] Agent PING OK");

    allocator = rcl_get_default_allocator();

    if (rclc_support_init(&support, 0, NULL, &allocator) != RCL_RET_OK)
    {
        Serial.println("[ROS] support FAILED");
        return false;
    }

    if (rclc_node_init_default(&node, "lds02rr_wifi", "", &support) != RCL_RET_OK)
    {
        Serial.println("[ROS] node FAILED");
        return false;
    }

    // Sensor chunks are time-sensitive; do not block the LDS parser on ACKs.
    if (rclc_publisher_init_best_effort(
            &telem_pub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(kaiaai_msgs, msg, KaiaaiTelemetry2),
            "/telemetry") != RCL_RET_OK)
    {
        Serial.println("[ROS] /telemetry FAILED");
        return false;
    }

    // ---- MOTOR: /motor/state ----
    initMotorMessages();

    if (rclc_publisher_init_best_effort(
            &state_pub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
            "/motor/state") != RCL_RET_OK)
    {
        Serial.println("[ROS] /motor/state FAILED");
        return false;
    }

    // ---- MOTOR: /motor/target_rpm ----
    if (rclc_subscription_init_best_effort(
            &target_sub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
            "/motor/target_rpm") != RCL_RET_OK)
    {
        Serial.println("[ROS] /motor/target_rpm FAILED");
        return false;
    }

    // Executor: 1 handle (subscription)
    if (rclc_executor_init(&executor, &support.context, 1, &allocator) != RCL_RET_OK)
    {
        Serial.println("[ROS] executor FAILED");
        return false;
    }

    if (rclc_executor_add_subscription(
            &executor, &target_sub, &target_msg,
            &targetCallback, ON_NEW_DATA) != RCL_RET_OK)
    {
        Serial.println("[ROS] executor add sub FAILED");
        return false;
    }

    if (!kaiaai_msgs__msg__KaiaaiTelemetry2__init(&telem_msg))
    {
        Serial.println("[ROS] message init FAILED");
        return false;
    }

    if (rmw_uros_sync_session(1000) == RCL_RET_OK)
        Serial.println("[ROS] Time sync OK");
    else
        Serial.println("[ROS] Time sync FAILED");

    Serial.println("[ROS] READY");
    rosReady = true;
    return true;
}

// ============================================================
// LIDAR SETUP
// ============================================================
void setupLidar()
{
    ledcAttach(LIDAR_MOT_EN, LIDAR_PWM_FREQ, LIDAR_PWM_BITS);
    ledcWrite(LIDAR_MOT_EN, LIDAR_STARTUP_DUTY);
    delay(LIDAR_STARTUP_MS);
    ledcWrite(LIDAR_MOT_EN, LIDAR_PWM_DUTY);

    LIDAR.setRxBufferSize(16384);  // must be before begin()
    LIDAR.begin(LIDAR_BAUD, SERIAL_8N1, LIDAR_RX, LIDAR_TX);

    Serial.println("[LDS] UART ready");
}

// ============================================================
// PUBLISH RAW TELEMETRY
// ============================================================
void publishTelemetry()
{
    if (!rosReady || ldsSize == 0)
        return;

    memset(&telem_msg, 0, sizeof(telem_msg));

    int64_t ns = rmw_uros_epoch_nanos();
    if (ns > 0)
    {
        telem_msg.stamp.sec     = (uint32_t)(ns / 1000000000LL);
        telem_msg.stamp.nanosec = (uint32_t)(ns % 1000000000LL);
    }

    telem_msg.seq = ++telemSeq;

    telem_msg.lds.data     = ldsBuf;
    telem_msg.lds.size     = ldsSize;
    telem_msg.lds.capacity = LDS_BUF_MAX;

    telem_msg.wifi_rssi_dbm = (int8_t)WiFi.RSSI();

    telem_msg.battery_mv = 0;
    telem_msg.scan_start_hint = false;

    rcl_ret_t rc = rcl_publish(&telem_pub, &telem_msg, NULL);

    if (rc == RCL_RET_OK)
    {
        totalPublished++;

        if (totalPublished <= 3 || totalPublished % 25 == 0)
        {
            Serial.print("[ROS] telemetry #");
            Serial.print(totalPublished);
            Serial.print(" seq=");
            Serial.print(telemSeq);
            Serial.print(" bytes=");
            Serial.println(ldsSize);
        }
    }
    else
    {
        totalPublishError++;
        Serial.print("[ROS][ERROR] rc=");
        Serial.print(rc);
        Serial.println();
        return;
    }

    ldsSize = 0;
}

// ============================================================
// READ LDS02RR - nơi DUY NHẤT gọi LIDAR.read()
// ============================================================
void processLidar()
{
    while (LIDAR.available() > 0)
    {
        if (ldsSize >= LDS_BUF_MAX)
            publishTelemetry();

        ldsBuf[ldsSize++] = (uint8_t)LIDAR.read();
        totalBytes++;
    }

    if (ldsSize >= LDS_BUF_MAX)
        publishTelemetry();
}

// ============================================================
// STATISTICS
// ============================================================
void printStatistics()
{
    if (millis() - lastStats < 2000)
        return;

    static uint32_t previousBytes = 0;

    uint32_t currentBytes = totalBytes;
    uint32_t bytesPerSecond = (currentBytes - previousBytes) / 2;

    previousBytes = currentBytes;
    lastStats = millis();

    Serial.print("[LDS] ");
    Serial.print(bytesPerSecond);
    Serial.print(" B/s");
    Serial.print(" | RSSI ");
    Serial.print(WiFi.RSSI());
    Serial.print(" dBm");
    Serial.print(" | pub=");
    Serial.print(totalPublished);
    Serial.print(" | err=");
    Serial.print(totalPublishError);
    Serial.print(" | buffered=");
    Serial.println(ldsSize);
}

// ============================================================
// SETUP
// ============================================================
void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("================================");
    Serial.println(" LDS02RR + MOTOR + WIFI + MICRO-ROS");
    Serial.println(" LDS PARSER + RAW TELEMETRY + /motor/*");
    Serial.println("================================");

    // Motor: đưa chân về trạng thái dừng sớm nhất
    setupMotors();
    setupEncoders();
    lastPIDTime = millis();
    lastCmdMs   = millis();

    setupLidar();

    if (!setupWiFi())
    {
        Serial.println("[ERROR] WiFi FAILED");
        return;
    }

    if (!setupMicroROS())
    {
        Serial.println("[ERROR] micro-ROS FAILED");
        return;
    }

    Serial.println();
    Serial.println("[SYSTEM] READY");
}

// ============================================================
// LOOP
// ============================================================
void loop()
{
    processLidar();
    publishTelemetry();

    if (rosReady)
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

    updateMotorPID();      // PID 100 ms + watchdog lệnh
    publishMotorState();   // /motor/state 10 Hz

    printStatistics();
}
