#include "ServoStatusWs.h"

#include <ArduinoJson.h>

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
ServoStatusPacketLogEntry logBuffer[LOG_CAPACITY];
size_t logHead = 0;
size_t logSize = 0;

void appendLog(
    uint8_t count,
    uint32_t seq,
    const float *target,
    const float *actual,
    const float *error)
{
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

} // namespace

void ServoStatusWs::begin(WebSocketsServer &ws)
{
    g_ws = &ws;

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

void ServoStatusWs::clearLog()
{
    logHead = 0;
    logSize = 0;
}

size_t ServoStatusWs::logCount()
{
    return logSize;
}

void ServoStatusWs::sendCsv(WebServer &server)
{
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
