#pragma once
#include <WebServer.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include "wifi_manager.h"
#include "servo_csv_log.h"

extern WebServer server;

// =====================================================
// 確保 scan 前 STA 是 idle（不影響 AP）
// =====================================================
inline void prepareForScan()
{
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.disconnect(false, false);
    esp_wifi_disconnect();
    delay(50);
  }
}

// =====================================================
// HTTP API
// =====================================================
inline void setupWifiHttpApi()
{
  server.on("/", []() {
    server.send(200, "text/plain", "ESP32 HTTP OK");
  });

  // ---------- scan ----------
  server.on("/wifi_scan", []() {
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
    String out;
    serializeJson(doc, out);
    server.send(200, "application/json", out);
  });

  // ---------- saved ----------
  server.on("/wifi_saved", []() {
    auto list = loadWiFiList();

    DynamicJsonDocument doc(1024);
    JsonArray arr = doc.createNestedArray("list");

    for (auto &w : list) {
      JsonObject o = arr.createNestedObject();
      o["ssid"] = w.first;
    }

    String out;
    serializeJson(doc, out);
    server.send(200, "application/json", out);
  });

  // ---------- current ----------
  server.on("/wifi_current", []() {
    DynamicJsonDocument doc(256);

    if (WiFi.status() == WL_CONNECTED) {
      doc["connected"] = true;
      doc["ssid"] = WiFi.SSID();
      doc["ip"]   = WiFi.localIP().toString();
      doc["rssi"] = WiFi.RSSI();
    } else {
      doc["connected"] = false;
    }

    String out;
    serializeJson(doc, out);
    server.send(200, "application/json", out);
  });

  // ---------- connect ----------
  server.on("/wifi_connect", []() {
    if (!server.hasArg("ssid")) {
      server.send(400, "text/plain", "missing ssid");
      return;
    }

    String ssid = server.arg("ssid");
    String pass = server.hasArg("pass") ? server.arg("pass") : "";

    addOrUpdateWifi(ssid, pass);
    bool ok = wifiConnectOnce(ssid, pass);

    server.send(200, "text/plain", ok ? "OK" : "FAIL");
  });

  // ---------- delete ----------
  server.on("/wifi_delete", []() {
    if (!server.hasArg("ssid")) {
      server.send(400, "text/plain", "missing ssid");
      return;
    }

    deleteWifi(server.arg("ssid"));
    server.send(200, "text/plain", "OK");
  });

  // ---------- download camera-side servo CSV ----------
  server.on("/download", []() {
    if (!SPIFFS.exists(SERVO_CSV_PATH)) {
      server.send(404, "text/plain", "data.csv not found");
      return;
    }

    File f = SPIFFS.open(SERVO_CSV_PATH, "r");
    if (!f) {
      server.send(500, "text/plain", "failed to open data.csv");
      return;
    }

    server.sendHeader("Content-Disposition", "attachment; filename=data.csv");
    server.streamFile(f, "text/csv");
    f.close();
  });
}
