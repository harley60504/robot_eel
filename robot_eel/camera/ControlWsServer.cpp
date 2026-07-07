#include "ControlWsServer.h"

#include <ArduinoJson.h>
#include <esp_camera.h>
#include <WiFi.h>

#include "wifi_manager.h"
#include "CtrlUartBridge.h"
#include "config.h"

namespace {

WebSocketsServer* g_wsServers[2] = {nullptr, nullptr};
uint8_t g_wsServerCount = 0;
ControlParamsPacket g_pkt = {
    CONTROL_PARAMS_PACKET_HEADER,
    15.0f,
    1.0f,
    1.6275f,
    1.0f,
    {1.24f, 1.08f, 1.0f, 1.05f, 1.1f, 1.2f},
    {0.614439f, 0.614439f, 0.614439f, 0.614439f, 0.614439f},
    {0, 0, 0, 0, 0, 0},
    {120, 120, 120, 120, 120, 120},
    false,
    2,
    0
};

uint32_t g_lastSeq = 0;

unsigned long lastSnapshot = 0;
constexpr unsigned long SNAPSHOT_INTERVAL_MS = 2000;

void broadcastControlText(String &out)
{
    for (uint8_t i = 0; i < g_wsServerCount; i++) {
        if (g_wsServers[i]) {
            g_wsServers[i]->broadcastTXT(out);
        }
    }
}

void sendAck(WebSocketsServer &ws, uint8_t client, const char *type, uint32_t seq)
{
    StaticJsonDocument<160> ack;
    ack["type"] = type;
    ack["seq"] = seq;
    ack["esp_rx_millis"] = millis();

    String out;
    serializeJson(ack, out);
    ws.sendTXT(client, out);
}

void handleSetParam(JsonDocument &doc)
{
    if (doc.containsKey("Ajoint"))     g_pkt.Ajoint      = doc["Ajoint"];
    if (doc.containsKey("frequency"))  g_pkt.frequency   = doc["frequency"];
    if (doc.containsKey("lambda"))     g_pkt.lambda      = doc["lambda"];
    if (doc.containsKey("L"))          g_pkt.L           = doc["L"];
    if (doc.containsKey("paused"))     g_pkt.isPaused    = doc["paused"];
    if (doc.containsKey("mode"))       g_pkt.controlMode = doc["mode"];

    if (doc["ampScales"].is<JsonArray>()) {
        JsonArray arr = doc["ampScales"].as<JsonArray>();
        for (int i = 0; i < bodyNum && i < arr.size(); i++) {
            g_pkt.ampScales[i] = arr[i].as<float>();
        }
    }

    if (doc["phaseLags"].is<JsonArray>()) {
        JsonArray arr = doc["phaseLags"].as<JsonArray>();
        for (int i = 0; i < bodyNum - 1 && i < arr.size(); i++) {
            g_pkt.phaseLags[i] = arr[i].as<float>();
        }
    }

    if (doc["jointBiasDeg"].is<JsonArray>()) {
        JsonArray arr = doc["jointBiasDeg"].as<JsonArray>();
        for (int i = 0; i < bodyNum && i < arr.size(); i++) {
            g_pkt.jointBiasDeg[i] = arr[i].as<float>();
        }
    }

    CtrlUartBridge::sendCtrlParams(g_pkt);
}

void handleAngles(JsonArray arr)
{
    if (arr.isNull()) return;

    float tmp[bodyNum] = {0};
    uint8_t count = 0;

    for (JsonVariant v : arr) {
        if (count >= bodyNum) break;
        tmp[count++] = v.as<float>();
    }

    if (count == 0) return;
    CtrlUartBridge::sendAngle(tmp, count);
}

void handleServoCenter(WebSocketsServer &ws, uint8_t client, JsonDocument &doc)
{
    uint32_t seq = doc["seq"] | 0;
    bool save = doc["save"] | false;

    StaticJsonDocument<160> ack;
    ack["type"] = "servo_center_ack";
    ack["seq"] = seq;
    ack["save"] = save;
    ack["esp_rx_millis"] = millis();

    String out;
    serializeJson(ack, out);
    ws.sendTXT(client, out);

    if (!doc.containsKey("angles")) return;

    JsonArray arr = doc["angles"].as<JsonArray>();
    if (arr.isNull()) return;

    float tmp[bodyNum] = {0};
    uint8_t count = 0;

    for (JsonVariant v : arr) {
        if (count >= bodyNum) break;
        tmp[count++] = v.as<float>();
    }

    if (count == 0) return;
    CtrlUartBridge::sendServoCenter(tmp, count, save);
}

void handleCameraParam(JsonDocument &doc)
{
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return;

    if (doc.containsKey("quality")) {
        s->set_quality(s, doc["quality"]);
    }

    if (doc.containsKey("framesize")) {
        s->set_framesize(s, static_cast<framesize_t>(doc["framesize"].as<int>()));
    }
}

void handleWsText(WebSocketsServer &ws, uint8_t client, uint8_t *payload, size_t len)
{
    StaticJsonDocument<2048> doc;
    if (deserializeJson(doc, payload, len)) return;

    const char* cmd = doc["cmd"] | "";

    if (!strcmp(cmd, "set_param")) {
        handleSetParam(doc);
        return;
    }

    if (!strcmp(cmd, "set_angle")) {
        uint32_t seq = doc["seq"] | 0;
        g_lastSeq = seq;
        sendAck(ws, client, "angle_ack", seq);

        if (!doc.containsKey("angles")) return;
        handleAngles(doc["angles"].as<JsonArray>());
        return;
    }

    if (!strcmp(cmd, "set_servo_center")) {
        handleServoCenter(ws, client, doc);
        return;
    }

    if (!strcmp(cmd, "camera_param")) {
        handleCameraParam(doc);
        return;
    }
}

} // namespace


/* =========================================================
 * ctrl_params snapshot
 * ========================================================= */
void ControlWsServer::tick()
{
    if (g_wsServerCount == 0) return;

    unsigned long now = millis();
    if (now - lastSnapshot < SNAPSHOT_INTERVAL_MS) return;
    lastSnapshot = now;

    StaticJsonDocument<1536> doc;
    doc["type"]      = "ctrl_params";
    doc["Ajoint"]    = g_pkt.Ajoint;
    doc["frequency"] = g_pkt.frequency;
    doc["lambda"]    = g_pkt.lambda;
    doc["L"]         = g_pkt.L;
    doc["paused"]    = g_pkt.isPaused;
    doc["mode"]      = g_pkt.controlMode;
    JsonArray amps = doc.createNestedArray("ampScales");
    JsonArray phases = doc.createNestedArray("phaseLags");
    JsonArray biases = doc.createNestedArray("jointBiasDeg");
    JsonArray centers = doc.createNestedArray("servoDefaultAngles");
    for (int i = 0; i < bodyNum; i++) {
        amps.add(g_pkt.ampScales[i]);
        biases.add(g_pkt.jointBiasDeg[i]);
        centers.add(g_pkt.servoDefaultAngles[i]);
    }
    for (int i = 0; i < bodyNum - 1; i++) {
        phases.add(g_pkt.phaseLags[i]);
    }

    String out;
    serializeJson(doc, out);
    broadcastControlText(out);
}


/* =========================================================
 * INIT
 * ========================================================= */
void ControlWsServer::begin(WebSocketsServer &ws)
{
    if (g_wsServerCount < 2) {
        g_wsServers[g_wsServerCount++] = &ws;
    }

    CtrlUartBridge::onCtrlParams =
        [](const ControlParamsPacket &p)
        {
            g_pkt = p;
        };

    WebSocketsServer *server = &ws;
    ws.onEvent([server](uint8_t num,
                        WStype_t type,
                        uint8_t *payload,
                        size_t len)
    {
        if (type != WStype_TEXT) return;
        handleWsText(*server, num, payload, len);
    });
}
