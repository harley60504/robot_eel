#pragma once
#include <Arduino.h>
#include <functional>

#include "config.h"           // ✅ 一定要有 bodyNum
#include "ControltoCamera.h"
#include "AnglePacket.h"

// ==== Servo 回報封包定義（需跟控制板一致）====
#define SERVO_STATUS_HEADER 0xBB
#define SERVO_MAX bodyNum     // ✅ 直接跟機器人段數同步

#pragma pack(push,1)
struct ServoStatus {
  uint8_t  header;          // 固定 = 0xBB
  uint8_t  count;           // servo 數量
  uint32_t seq;             // 序號
  float    target[SERVO_MAX];
  float    actual[SERVO_MAX];
  float    error[SERVO_MAX];
  uint8_t  checksum;        // XOR checksum（用 calcControlChecksum）
};
#pragma pack(pop)

namespace CtrlUartBridge {

  void begin(HardwareSerial& ser,
             uint32_t baud,
             int rxPin,
             int txPin);

  // UART TX：把控制參數送回控制板
  void sendCtrlParams(const ControlPacket &pkt);

  // ✅ UART TX：AnglePacket（Flutter 控制 servo 用）
  void sendAngle(const float* targetDeg, uint8_t count);

  // callbacks（UART RX → 上層）
  extern std::function<void(const ControlPacket&)> onCtrlParams;
  extern std::function<void(const ServoStatus&)>   onServoStatus;
}
