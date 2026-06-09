#pragma once

#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>

#include "CtrlUartBridge.h"

static const char* SERVO_CSV_PATH = "/data.csv";

inline bool initServoCsvLog()
{
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS init failed for servo CSV");
    return false;
  }

  if (!SPIFFS.exists(SERVO_CSV_PATH)) {
    File f = SPIFFS.open(SERVO_CSV_PATH, FILE_WRITE);
    if (!f) return false;

    f.print("millis,seq,channel,target_deg,actual_deg,error_deg\n");
    f.close();
  }

  return true;
}

inline void appendServoCsvLog(const ServoStatus& status)
{
  File f = SPIFFS.open(SERVO_CSV_PATH, FILE_APPEND);
  if (!f) return;

  const uint32_t now = millis();
  const uint8_t count = min<uint8_t>(status.count, SERVO_MAX);
  for (uint8_t i = 0; i < count; i++) {
    f.printf(
      "%lu,%lu,%u,%.3f,%.3f,%.3f\n",
      static_cast<unsigned long>(now),
      static_cast<unsigned long>(status.seq),
      static_cast<unsigned>(i + 1),
      status.target[i],
      status.actual[i],
      status.error[i]
    );
  }

  f.close();
}
