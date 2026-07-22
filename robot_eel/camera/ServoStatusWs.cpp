#include "ServoStatusWs.h"

#include <ArduinoJson.h>
#include <cstdlib>
#include <esp_heap_caps.h>

#include "config.h"

namespace {

WebSocketsServer *g_ws = nullptr;

unsigned long lastBroadcast = 0;
constexpr unsigned long BROADCAST_INTERVAL_MS = 25;

struct ServoStatusPacketLogEntry {
    uint32_t seq;
    uint32_t millisValue;
    uint8_t count;
    float targetDeg[SERVO_STATUS_MAX];
    float actualDeg[SERVO_STATUS_MAX];
    float errorDeg[SERVO_STATUS_MAX];
};

constexpr size_t LOG_CAPACITY = 1200;
ServoStatusPacketLogEntry *logBuffer = nullptr;
size_t logHead = 0;
size_t logSize = 0;

struct ImuPacketLogEntry {
    uint32_t seq;
    uint32_t millisValue;
    uint32_t tMs;
    float accel[3];
    float gyro[3];
    float tempC;
};

constexpr size_t IMU_LOG_CAPACITY = 3000;
ImuPacketLogEntry *imuLogBuffer = nullptr;
size_t imuLogHead = 0;
size_t imuLogSize = 0;

template <typename T>
T *allocateLogBuffer(size_t count)
{
    void *ptr = heap_caps_malloc(sizeof(T) * count, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ptr) {
        ptr = heap_caps_malloc(sizeof(T) * count, MALLOC_CAP_8BIT);
    }
    return static_cast<T *>(ptr);
}

void appendLog(
    uint8_t count,
    uint32_t seq,
    const float *target,
    const float *actual,
    const float *error)
{
    if (!logBuffer) return;

    ServoStatusPacketLogEntry &entry = logBuffer[logHead];
    entry.seq = seq;
    entry.millisValue = millis();
    entry.count = count < SERVO_STATUS_MAX ? count : SERVO_STATUS_MAX;

    for (uint8_t i = 0; i < entry.count; i++) {
        entry.targetDeg[i] = target[i];
        entry.actualDeg[i] = actual[i];
        entry.errorDeg[i] = error[i];
    }

    logHead = (logHead + 1) % LOG_CAPACITY;
    if (logSize < LOG_CAPACITY) {
        logSize++;
    }
}

void appendImuLog(const ImuPacket &imu)
{
    if (!imuLogBuffer) return;

    ImuPacketLogEntry &entry = imuLogBuffer[imuLogHead];
    entry.seq = imu.seq;
    entry.millisValue = millis();
    entry.tMs = imu.t_ms;
    entry.tempC = imu.tempC;

    for (uint8_t i = 0; i < 3; i++) {
        entry.accel[i] = imu.accel[i];
        entry.gyro[i] = imu.gyro[i];
    }

    imuLogHead = (imuLogHead + 1) % IMU_LOG_CAPACITY;
    if (imuLogSize < IMU_LOG_CAPACITY) {
        imuLogSize++;
    }
}

} // namespace

void ServoStatusWs::begin(WebSocketsServer &ws)
{
    g_ws = &ws;

    if (!logBuffer) {
        logBuffer = allocateLogBuffer<ServoStatusPacketLogEntry>(LOG_CAPACITY);
        if (!logBuffer) {
            Serial.println("[SERVO CSV] log buffer allocation failed");
        }
    }

    if (!imuLogBuffer) {
        imuLogBuffer = allocateLogBuffer<ImuPacketLogEntry>(IMU_LOG_CAPACITY);
        if (!imuLogBuffer) {
            Serial.println("[IMU CSV] log buffer allocation failed");
        }
    }

    CtrlUartBridge::onServoStatusPacket =
        [](const ServoStatusPacket &s)
        {
            ServoStatusWs::broadcast(
                s.count,
                s.seq,
                s.targetDeg,
                s.actualDeg,
                s.errorDeg
            );
        };

    CtrlUartBridge::onImuPacket =
        [](const ImuPacket &imu)
        {
            ServoStatusWs::broadcastImu(imu);
        };
}

void ServoStatusWs::broadcast(
    uint8_t count,
    uint32_t seq,
    const float *target,
    const float *actual,
    const float *error)
{
    appendLog(count, seq, target, actual, error);

    if (!g_ws) return;

    unsigned long now = millis();
    if (now - lastBroadcast < BROADCAST_INTERVAL_MS) return;
    lastBroadcast = now;

    StaticJsonDocument<512> doc;
    doc["type"] = "servo_status";
    doc["seq"] = seq;

    JsonArray t = doc.createNestedArray("target");
    JsonArray a = doc.createNestedArray("actual");
    JsonArray e = doc.createNestedArray("error");

    const uint8_t safeCount = count < SERVO_STATUS_MAX ? count : SERVO_STATUS_MAX;
    for (uint8_t i = 0; i < safeCount; i++) {
        t.add(target[i]);
        a.add(actual[i]);
        e.add(error[i]);
    }

    String out;
    serializeJson(doc, out);
    g_ws->broadcastTXT(out);
}

void ServoStatusWs::broadcastImu(const ImuPacket &imu)
{
    appendImuLog(imu);

    if (!g_ws) return;

    StaticJsonDocument<384> doc;
    doc["type"] = "imu_status";
    doc["seq"] = imu.seq;
    doc["t_ms"] = imu.t_ms;
    doc["tempC"] = imu.tempC;

    JsonArray accel = doc.createNestedArray("accel");
    JsonArray gyro = doc.createNestedArray("gyro");

    for (uint8_t i = 0; i < 3; i++) {
        accel.add(imu.accel[i]);
        gyro.add(imu.gyro[i]);
    }

    String out;
    serializeJson(doc, out);
    g_ws->broadcastTXT(out);
}

void ServoStatusWs::clearLog()
{
    logHead = 0;
    logSize = 0;
}

void ServoStatusWs::clearImuLog()
{
    imuLogHead = 0;
    imuLogSize = 0;
}

size_t ServoStatusWs::logCount()
{
    if (!logBuffer) return 0;
    return logSize;
}

size_t ServoStatusWs::imuLogCount()
{
    if (!imuLogBuffer) return 0;
    return imuLogSize;
}

void ServoStatusWs::sendCsv(WebServer &server)
{
    if (!logBuffer) {
        server.send(503, "text/plain", "servo log buffer unavailable");
        return;
    }

    const size_t countSnapshot = logSize;
    const size_t headSnapshot = logHead;
    const size_t start = (headSnapshot + LOG_CAPACITY - countSnapshot) % LOG_CAPACITY;

    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/csv", "");
    server.sendContent("seq,millis,channel,target_deg,actual_deg,error_deg\n");

    char line[128];
    for (size_t i = 0; i < countSnapshot; i++) {
        const ServoStatusPacketLogEntry &entry = logBuffer[(start + i) % LOG_CAPACITY];
        for (uint8_t ch = 0; ch < entry.count; ch++) {
            snprintf(
                line,
                sizeof(line),
                "%lu,%lu,CH%u,%.3f,%.3f,%.3f\n",
                static_cast<unsigned long>(entry.seq),
                static_cast<unsigned long>(entry.millisValue),
                static_cast<unsigned int>(ch + 1),
                entry.targetDeg[ch],
                entry.actualDeg[ch],
                entry.errorDeg[ch]
            );
            server.sendContent(line);
        }
    }

    server.sendContent("");
}

void ServoStatusWs::sendImuCsv(WebServer &server)
{
    if (!imuLogBuffer) {
        server.send(503, "text/plain", "imu log buffer unavailable");
        return;
    }

    const size_t countSnapshot = imuLogSize;
    const size_t headSnapshot = imuLogHead;
    const size_t start = (headSnapshot + IMU_LOG_CAPACITY - countSnapshot) % IMU_LOG_CAPACITY;

    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/csv", "");
    server.sendContent("seq,millis,t_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,temp_c\n");

    char line[192];
    for (size_t i = 0; i < countSnapshot; i++) {
        const ImuPacketLogEntry &entry = imuLogBuffer[(start + i) % IMU_LOG_CAPACITY];
        snprintf(
            line,
            sizeof(line),
            "%lu,%lu,%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n",
            static_cast<unsigned long>(entry.seq),
            static_cast<unsigned long>(entry.millisValue),
            static_cast<unsigned long>(entry.tMs),
            entry.accel[0],
            entry.accel[1],
            entry.accel[2],
            entry.gyro[0],
            entry.gyro[1],
            entry.gyro[2],
            entry.tempC
        );
        server.sendContent(line);
    }

    server.sendContent("");
}
