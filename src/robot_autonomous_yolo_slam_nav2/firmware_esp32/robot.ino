// ============================================================
// ESP32 + LDS02RR + MOTOR/ENCODER + WiFi micro-ROS  (v2.3 Clean Fix)
// - Trái: if (A == B) leftCount--; else leftCount++; (giữ nguyên)
// - Phải: if (A == B) rightCount++; else rightCount--; (đã đảo lại)
// ============================================================

#include <Arduino.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <math.h>

#include <micro_ros_kaia.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <kaiaai_msgs/msg/kaiaai_telemetry2.h>
#include <std_msgs/msg/float32_multi_array.h>

#include <rmw_microros/rmw_microros.h>

// ================= WIFI =================
// Copy wifi_credentials.example.h to wifi_credentials.h and fill in your values.
#include "wifi_credentials.h"
#define AGENT_PORT 8888

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
#define LDS_PACKET_SIZE     22
#define LDS_BUF_MAX         (LDS_PACKET_SIZE * 18)
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

#define IN1 2
#define IN2 4
#define ENA 33
#define IN3 22
#define IN4 23
#define ENB 32

#define ENC_L_A 26
#define ENC_L_B 25
#define ENC_R_A 27
#define ENC_R_B 14

#define MOTOR_PWM_FREQ   1000
#define MOTOR_PWM_RES    8
#define MOTOR_PWM_CH_L   4
#define MOTOR_PWM_CH_R   5
#define MAX_PWM          255

#define ENCODER_TPR       227.0f
#define WHEEL_DIAMETER_MM 65.0f

float KpLeft  = 0.85f, KiLeft  = 1.00f, KdLeft  = 0.03f;
float KpRight = 0.85f, KiRight = 1.00f, KdRight = 0.03f;

#define PID_PERIOD_MS     25
#define INTEGRAL_LIMIT    120.0f
#define RPM_FILTER_ALPHA  0.50f
#define MOTOR_DEADZONE    28
#define MOTOR_FF_GAIN     0.50f

#define CMD_TIMEOUT_MS    1000UL
#define STATE_PERIOD_MS   100

rcl_publisher_t    state_pub;
rcl_subscription_t target_sub;

std_msgs__msg__Float32MultiArray state_msg;
std_msgs__msg__Float32MultiArray target_msg;

float stateBuf[8];
#define TARGET_BUF_MAX      8
float targetBuf[TARGET_BUF_MAX];
std_msgs__msg__MultiArrayDimension targetDimBuf[1];
char  targetDimLabel[16];

bool rosReady = false;

volatile long leftCount  = 0;
volatile long rightCount = 0;

float targetRPMLeft = 0, targetRPMRight = 0;
float rpmLeftFiltered = 0, rpmRightFiltered = 0;
float integralLeft = 0, integralRight = 0;
float previousErrorLeft = 0, previousErrorRight = 0;
float pwmLeft = 0, pwmRight = 0;

unsigned long lastPIDTime = 0;
long previousLeftCount = 0, previousRightCount = 0;
unsigned long lastCmdMs = 0, lastStateMs = 0;

// TRÁI: Giữ nguyên (đã khỏi)
void IRAM_ATTR leftEncoderISR()
{
    bool A = digitalRead(ENC_L_A);
    bool B = digitalRead(ENC_L_B);
    if (A == B) leftCount--; else leftCount++;
}

// PHẢI: Đảo lại thành ++ / --
void IRAM_ATTR rightEncoderISR()
{
    bool A = digitalRead(ENC_R_A);
    bool B = digitalRead(ENC_R_B);
    if (A == B) rightCount++; else rightCount--;
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
    integralLeft = integralRight = 0;
    previousErrorLeft = previousErrorRight = 0;
}

void stopMotors()
{
    setLeftMotor(0);
    setRightMotor(0);
    pwmLeft = pwmRight = 0;
    targetRPMLeft = targetRPMRight = 0;
    rpmLeftFiltered = rpmRightFiltered = 0;
    resetPID();
}

void calculateRPM(float dt)
{
    long pl, pr;
    noInterrupts();
    pl = leftCount;
    pr = rightCount;
    interrupts();

    long currentLeft = pl, currentRight = pr;

    long deltaLeft  = currentLeft  - previousLeftCount;
    long deltaRight = currentRight - previousRightCount;
    previousLeftCount  = currentLeft;
    previousRightCount = currentRight;

    if (dt <= 0.001f) return;

    float rawLeft  = ((float)deltaLeft  / ENCODER_TPR) * 60.0f / dt;
    float rawRight = ((float)deltaRight / ENCODER_TPR) * 60.0f / dt;

    rpmLeftFiltered  = RPM_FILTER_ALPHA * rawLeft  + (1.0f - RPM_FILTER_ALPHA) * rpmLeftFiltered;
    rpmRightFiltered = RPM_FILTER_ALPHA * rawRight + (1.0f - RPM_FILTER_ALPHA) * rpmRightFiltered;
}

float calculatePIDLeft(float dt)
{
    float error = targetRPMLeft - rpmLeftFiltered;
    float derivative = (error - previousErrorLeft) / dt;
    previousErrorLeft = error;

    float P = KpLeft * error;
    float D = KdLeft * derivative;
    float FF = MOTOR_FF_GAIN * targetRPMLeft;

    float I_candidate = integralLeft + KiLeft * error * dt;
    float u_candidate = P + I_candidate + D + FF;

    bool satHigh = (u_candidate >=  MAX_PWM);
    bool satLow  = (u_candidate <= -MAX_PWM);

    if (!satHigh && !satLow)            integralLeft = I_candidate;
    else if (satHigh && error < 0.0f)   integralLeft = I_candidate;
    else if (satLow  && error > 0.0f)   integralLeft = I_candidate;

    integralLeft = constrain(integralLeft, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

    float output = P + integralLeft + D + FF;
    if (output > 0) output += MOTOR_DEADZONE;
    else if (output < 0) output -= MOTOR_DEADZONE;

    if (targetRPMLeft > 0.0f && output < 0.0f) output = 0.0f;
    if (targetRPMLeft < 0.0f && output > 0.0f) output = 0.0f;

    return constrain(output, -MAX_PWM, MAX_PWM);
}

float calculatePIDRight(float dt)
{
    float error = targetRPMRight - rpmRightFiltered;
    float derivative = (error - previousErrorRight) / dt;
    previousErrorRight = error;

    float P = KpRight * error;
    float D = KdRight * derivative;
    float FF = MOTOR_FF_GAIN * targetRPMRight;

    float I_candidate = integralRight + KiRight * error * dt;
    float u_candidate = P + I_candidate + D + FF;

    bool satHigh = (u_candidate >=  MAX_PWM);
    bool satLow  = (u_candidate <= -MAX_PWM);

    if (!satHigh && !satLow)            integralRight = I_candidate;
    else if (satHigh && error < 0.0f)   integralRight = I_candidate;
    else if (satLow  && error > 0.0f)   integralRight = I_candidate;

    integralRight = constrain(integralRight, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

    float output = P + integralRight + D + FF;
    if (output > 0) output += MOTOR_DEADZONE;
    else if (output < 0) output -= MOTOR_DEADZONE;

    if (targetRPMRight > 0.0f && output < 0.0f) output = 0.0f;
    if (targetRPMRight < 0.0f && output > 0.0f) output = 0.0f;

    return constrain(output, -MAX_PWM, MAX_PWM);
}

void updateMotorPID()
{
    unsigned long now = millis();
    if (now - lastPIDTime < PID_PERIOD_MS) return;

    float dt = (now - lastPIDTime) / 1000.0f;
    lastPIDTime = now;
    if (dt <= 0.001f || dt > 0.25f) dt = (float)PID_PERIOD_MS / 1000.0f;

#if CMD_TIMEOUT_MS > 0
    if (now - lastCmdMs > CMD_TIMEOUT_MS &&
        (targetRPMLeft != 0.0f || targetRPMRight != 0.0f))
    {
        Serial.println("[MOTOR] cmd timeout -> STOP");
        stopMotors();
        calculateRPM(dt);
        return;
    }
#endif

    calculateRPM(dt);

    if (fabs(targetRPMLeft) < 0.1f && fabs(targetRPMRight) < 0.1f) {
        stopMotors();
        return;
    }

    if (fabs(targetRPMLeft) < 0.1f) {
        integralLeft = 0;
        previousErrorLeft = 0;
        pwmLeft = 0;
        setLeftMotor(0);
    } else {
        pwmLeft = calculatePIDLeft(dt);
        setLeftMotor(pwmLeft);
    }

    if (fabs(targetRPMRight) < 0.1f) {
        integralRight = 0;
        previousErrorRight = 0;
        pwmRight = 0;
        setRightMotor(0);
    } else {
        pwmRight = calculatePIDRight(dt);
        setRightMotor(pwmRight);
    }
}

// ---- SUB /motor/target_rpm ----
void targetCallback(const void * msgin)
{
    const std_msgs__msg__Float32MultiArray * m =
        (const std_msgs__msg__Float32MultiArray *)msgin;

    if (m->data.size < 2) return;

    float l = m->data.data[0];
    float r = m->data.data[1];
    if (isnan(l) || isnan(r)) return;

    if (l * targetRPMLeft <= 0.0f) {
        integralLeft = 0;
        previousErrorLeft = 0;
    }
    if (r * targetRPMRight <= 0.0f) {
        integralRight = 0;
        previousErrorRight = 0;
    }

    targetRPMLeft  = l;
    targetRPMRight = r;
    lastCmdMs      = millis();
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

    long encL = pL, encR = pR;

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
    target_msg.data.capacity = TARGET_BUF_MAX;
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

    if (rclc_publisher_init_best_effort(
            &telem_pub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(kaiaai_msgs, msg, KaiaaiTelemetry2),
            "/telemetry") != RCL_RET_OK)
    {
        Serial.println("[ROS] /telemetry FAILED");
        return false;
    }

    initMotorMessages();

    if (rclc_publisher_init_best_effort(
            &state_pub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
            "/motor/state") != RCL_RET_OK)
    {
        Serial.println("[ROS] /motor/state FAILED");
        return false;
    }

    if (rclc_subscription_init_best_effort(
            &target_sub, &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
            "/motor/target_rpm") != RCL_RET_OK)
    {
        Serial.println("[ROS] /motor/target_rpm FAILED");
        return false;
    }

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

    LIDAR.setRxBufferSize(16384);
    LIDAR.begin(LIDAR_BAUD, SERIAL_8N1, LIDAR_RX, LIDAR_TX);

    Serial.println("[LDS] UART ready");
}

// ============================================================
// PUBLISH RAW TELEMETRY
// ============================================================
void publishTelemetry()
{
    if (ldsSize == 0)
        return;

    if (!rosReady)
    {
        ldsSize = 0;
        return;
    }

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
        Serial.println(rc);
    }

    ldsSize = 0;
}

// ============================================================
// READ LDS02RR
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
    Serial.println("================================");

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

    if (rosReady)
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

    updateMotorPID();      // PID 25 ms + watchdog lệnh
    publishMotorState();   // /motor/state 10 Hz

    printStatistics();
}