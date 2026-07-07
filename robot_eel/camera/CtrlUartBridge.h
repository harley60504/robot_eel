#pragma once
#include <Arduino.h>
#include <functional>

#include "config.h"           // ✅ 一定要有 bodyNum
#include "ControlParamsPacket.h"
#include "ServoTargetPacket.h"
#include "ServoCenterPacket.h"
#include "ServoStatusPacket.h"

namespace CtrlUartBridge {

  void begin(HardwareSerial& ser,
             uint32_t baud,
             int rxPin,
             int txPin);

  // UART TX：把控制參數送回控制板
  void sendCtrlParams(const ControlParamsPacket &pkt);

  // ✅ UART TX：ServoTargetPacket（Flutter 控制 servo 用）
  void sendAngle(const float* targetDeg, uint8_t count);

  void sendServoCenter(const float* centerDeg, uint8_t count, bool save);

  // callbacks（UART RX → 上層）
  extern std::function<void(const ControlParamsPacket&)> onCtrlParams;
  extern std::function<void(const ServoStatusPacket&)>   onServoStatusPacket;
}
