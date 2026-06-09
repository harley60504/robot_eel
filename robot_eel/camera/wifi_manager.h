#pragma once
#include <WiFi.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <vector>
#include <algorithm>
#include "esp_wifi.h"
#include "config.h"

// =====================================================
// Global
// =====================================================
static Preferences wifiPrefs;
static bool mdnsStarted = false;

// =====================================================
// STA：只嘗試一次，失敗就回 idle
// =====================================================
inline bool wifiConnectOnce(const String& ssid, const String& pass)
{
  Serial.printf("[WiFi] Try STA: %s\n", ssid.c_str());

  WiFi.begin(ssid.c_str(), pass.c_str());

  unsigned long t0 = millis();
  while (millis() - t0 < 8000) {
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("[WiFi] STA connected");
      return true;
    }
    delay(200);
  }

  // ⭐關鍵：把 STA 拉回 idle（這就是你之前穩的原因）
  Serial.println("[WiFi] STA failed -> idle");
  WiFi.disconnect(false, false);
  esp_wifi_disconnect();

  return false;
}

// =====================================================
// NVS load
// =====================================================
inline std::vector<std::pair<String,String>> loadWiFiList()
{
  wifiPrefs.begin("wifi", true);
  String raw = wifiPrefs.getString("list", "[]");
  wifiPrefs.end();

  std::vector<std::pair<String,String>> list;
  DynamicJsonDocument doc(2048);

  if (deserializeJson(doc, raw)) return list;

  for (JsonObject o : doc.as<JsonArray>()) {
    list.push_back({
      o["ssid"].as<String>(),
      o["pass"].as<String>()
    });
  }
  return list;
}

// =====================================================
// NVS save
// =====================================================
inline void saveWiFiList(const std::vector<std::pair<String,String>>& list)
{
  DynamicJsonDocument doc(2048);
  JsonArray arr = doc.to<JsonArray>();

  for (auto &w : list) {
    JsonObject o = arr.createNestedObject();
    o["ssid"] = w.first;
    o["pass"] = w.second;
  }

  String out;
  serializeJson(arr, out);

  wifiPrefs.begin("wifi", false);
  wifiPrefs.putString("list", out);
  wifiPrefs.end();
}

// =====================================================
// add / update
// =====================================================
inline void addOrUpdateWifi(const String& ssid, const String& pass)
{
  auto list = loadWiFiList();
  bool found = false;

  for (auto &w : list) {
    if (w.first == ssid) {
      w.second = pass;
      found = true;
      break;
    }
  }

  if (!found)
    list.push_back({ssid, pass});

  saveWiFiList(list);
}

// =====================================================
// delete
// =====================================================
inline void deleteWifi(const String& ssid)
{
  auto list = loadWiFiList();

  list.erase(
    std::remove_if(
      list.begin(),
      list.end(),
      [&](auto &w){ return w.first == ssid; }
    ),
    list.end()
  );

  saveWiFiList(list);
}

// =====================================================
// Wi-Fi 啟動（只呼叫一次）
// 行為 = 你之前那份
// =====================================================
inline void startWifiApSta()
{
  Serial.println("\n=== WiFi START (AP+STA) ===");

  WiFi.mode(WIFI_AP_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  WiFi.setAutoReconnect(false);   // ⭐非常重要
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setHostname(HOSTNAME);

  Serial.printf("[AP] %s  IP=%s\n",
                AP_SSID,
                WiFi.softAPIP().toString().c_str());

  if (!mdnsStarted && MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    mdnsStarted = true;
  }

  // ⭐只嘗試一次已儲存 WiFi
  auto list = loadWiFiList();
  for (auto &w : list) {
    if (wifiConnectOnce(w.first, w.second)) {
      break;
    }
  }

  Serial.println("=== WiFi READY ===\n");
}
