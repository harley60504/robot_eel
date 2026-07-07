#include "HttpApi.h"

#include <ArduinoJson.h>
#include <WiFi.h>
#include <esp_wifi.h>

#include "ServoStatusWs.h"
#include "wifi_manager.h"

namespace {

void prepareForScan()
{
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.disconnect(false, false);
        esp_wifi_disconnect();
        delay(50);
    }
}

void sendJson(int statusCode, JsonDocument &doc)
{
    String out;
    serializeJson(doc, out);
    server.send(statusCode, "application/json", out);
}

void handleRoot()
{
    server.send(200, "text/plain", "ESP32 HTTP OK");
}

void handleWifiScan()
{
    prepareForScan();

    int n = WiFi.scanNetworks(false, true);

    DynamicJsonDocument doc(2048);
    JsonArray arr = doc.createNestedArray("list");

    for (int i = 0; i < n; i++) {
        JsonObject o = arr.createNestedObject();
        o["ssid"] = WiFi.SSID(i);
        o["rssi"] = WiFi.RSSI(i);
    }

    WiFi.scanDelete();
    sendJson(200, doc);
}

void handleWifiSaved()
{
    auto list = loadWiFiList();

    DynamicJsonDocument doc(1024);
    JsonArray arr = doc.createNestedArray("list");

    for (auto &w : list) {
        JsonObject o = arr.createNestedObject();
        o["ssid"] = w.first;
    }

    sendJson(200, doc);
}

void handleWifiCurrent()
{
    DynamicJsonDocument doc(256);

    if (WiFi.status() == WL_CONNECTED) {
        doc["connected"] = true;
        doc["ssid"] = WiFi.SSID();
        doc["ip"] = WiFi.localIP().toString();
        doc["rssi"] = WiFi.RSSI();
    } else {
        doc["connected"] = false;
    }

    sendJson(200, doc);
}

void handleWifiConnect()
{
    if (!server.hasArg("ssid")) {
        server.send(400, "text/plain", "missing ssid");
        return;
    }

    String ssid = server.arg("ssid");
    String pass = server.hasArg("pass") ? server.arg("pass") : "";

    addOrUpdateWifi(ssid, pass);
    bool ok = wifiConnectOnce(ssid, pass);

    server.send(200, "text/plain", ok ? "OK" : "FAIL");
}

void handleWifiDelete()
{
    if (!server.hasArg("ssid")) {
        server.send(400, "text/plain", "missing ssid");
        return;
    }

    deleteWifi(server.arg("ssid"));
    server.send(200, "text/plain", "OK");
}

void handleServoLogCsv()
{
    ServoStatusWs::sendCsv(server);
}

void handleServoLogClear()
{
    ServoStatusWs::clearLog();
    server.send(200, "text/plain", "OK");
}

void handleServoLogStatus()
{
    DynamicJsonDocument doc(128);
    doc["samples"] = ServoStatusWs::logCount();
    sendJson(200, doc);
}

} // namespace

void HttpApi::begin()
{
    server.on("/", handleRoot);

    server.on("/wifi_scan", handleWifiScan);
    server.on("/wifi_saved", handleWifiSaved);
    server.on("/wifi_current", handleWifiCurrent);
    server.on("/wifi_connect", handleWifiConnect);
    server.on("/wifi_delete", handleWifiDelete);

    server.on("/servo_log.csv", handleServoLogCsv);
    server.on("/servo_log_clear", handleServoLogClear);
    server.on("/servo_log_status", handleServoLogStatus);
}
