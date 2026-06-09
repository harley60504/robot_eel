#include "CtrlWsServer.h"

#include <ArduinoJson.h>
#include <esp_camera.h>
#include <WiFi.h>

#include "wifi_manager.h"
#include "CtrlUartBridge.h"
#include "config.h"
#include "servo_csv_log.h"

namespace {

WebSocketsServer* g_ws = nullptr;
ControlPacket g_pkt = {
    CONTROL_PACKET_HEADER,
    15.0f,
    1.0f,
    1.6275f,
    1.0f,
    {1.24f, 1.08f, 1.0f, 1.05f, 1.1f, 1.2f},
    {0.614439f, 0.614439f, 0.614439f, 0.614439f, 0.614439f},
    {0, 0, 0, 0, 0, 0},
    false,
    2,
    false,
    1.0f,
    0
};

uint32_t g_lastSeq = 0;   // 只給 angle_ack 用

unsigned long lastServoBroadcast = 0;
constexpr unsigned long SERVO_INTERVAL_MS = 25;

unsigned long lastSnapshot = 0;
constexpr unsigned long SNAPSHOT_INTERVAL_MS = 2000;

} // namespace


/* =========================================================
 * Servo Status
 * ========================================================= */
void CtrlWsServer::broadcastServoStatus(
    uint8_t count,
    uint32_t seq,
    const float *target,
    const float *actual,
    const float *error)
{
    if (!g_ws) return;

    unsigned long now = millis();
    if (now - lastServoBroadcast < SERVO_INTERVAL_MS) return;
    lastServoBroadcast = now;

    StaticJsonDocument<512> doc;
    doc["type"] = "servo_status";
    doc["seq"]  = seq;   // ✅ 用控制板送來的 ServoStatus.seq

    auto t = doc.createNestedArray("target");
    auto a = doc.createNestedArray("actual");
    auto e = doc.createNestedArray("error");

    for (int i = 0; i < count; i++) {
        t.add(target[i]);
        a.add(actual[i]);
        e.add(error[i]);
    }

    String out;
    serializeJson(doc, out);
    g_ws->broadcastTXT(out);
}


/* =========================================================
 * ctrl_params snapshot
 * ========================================================= */
void CtrlWsServer::tick()
{
    if (!g_ws) return;

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
    doc["feedback"]  = g_pkt.feedbackGain;
    auto amps = doc.createNestedArray("ampScales");
    auto phases = doc.createNestedArray("phaseLags");
    auto biases = doc.createNestedArray("jointBiasDeg");
    for (int i = 0; i < bodyNum; i++) {
        amps.add(g_pkt.ampScales[i]);
        biases.add(g_pkt.jointBiasDeg[i]);
    }
    for (int i = 0; i < bodyNum - 1; i++) {
        phases.add(g_pkt.phaseLags[i]);
    }

    String out;
    serializeJson(doc, out);
    g_ws->broadcastTXT(out);
}


/* =========================================================
 * INIT
 * ========================================================= */
void CtrlWsServer::begin(WebSocketsServer &ws)
{
    g_ws = &ws;

    CtrlUartBridge::onServoStatus =
        [](const ServoStatus &s)
        {
            appendServoCsvLog(s);
            CtrlWsServer::broadcastServoStatus(
                s.count,
                s.seq,
                s.target,
                s.actual,
                s.error
            );
        };

    CtrlUartBridge::onCtrlParams =
        [](const ControlPacket &p)
        {
            g_pkt = p;
        };

    ws.onEvent([](uint8_t num,
                  WStype_t type,
                  uint8_t *payload,
                  size_t len)
    {
        if (type != WStype_TEXT) return;

        StaticJsonDocument<2048> doc;
        if (deserializeJson(doc, payload, len)) return;

        const char* cmd = doc["cmd"] | "";

        /* ================= set_param ================= */
        if (!strcmp(cmd, "set_param")) {

            if (doc.containsKey("Ajoint"))     g_pkt.Ajoint       = doc["Ajoint"];
            if (doc.containsKey("frequency"))  g_pkt.frequency    = doc["frequency"];
            if (doc.containsKey("lambda"))     g_pkt.lambda       = doc["lambda"];
            if (doc.containsKey("L"))          g_pkt.L            = doc["L"];
            if (doc.containsKey("paused"))     g_pkt.isPaused     = doc["paused"];
            if (doc.containsKey("mode"))       g_pkt.controlMode  = doc["mode"];
            if (doc.containsKey("feedback"))   g_pkt.feedbackGain = doc["feedback"];
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
            return;
        }

        /* ================= set_angle（RTT） ================= */
        if (!strcmp(cmd, "set_angle")) {

            uint32_t seq = doc["seq"] | 0;
            g_lastSeq = seq;

            unsigned long now = millis();

            if (g_ws) {
                StaticJsonDocument<128> ack;
                ack["type"] = "angle_ack";
                ack["seq"]  = seq;   // ✅ RTT 仍然用 Python 送進來的 seq
                ack["esp_rx_millis"] = now;

                String out;
                serializeJson(ack, out);
                g_ws->sendTXT(num, out);
            }

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

            CtrlUartBridge::sendAngle(tmp, count);
            return;
        }

        /* ================= camera_param ================= */
        if (!strcmp(cmd, "camera_param")) {
            sensor_t *s = esp_camera_sensor_get();
            if (!s) return;

            if (doc.containsKey("quality"))
                s->set_quality(s, doc["quality"]);

            if (doc.containsKey("framesize"))
                s->set_framesize(s, (framesize_t)doc["framesize"]);

            return;
        }
    });
}
