#pragma once

#include <Arduino.h>

#include "UartPacketChecksum.h"
#include "config.h"

#define SERVO_STATUS_PACKET_HEADER 0xBB
#define SERVO_STATUS_MAX bodyNum

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  uint8_t  count;
  uint32_t seq;
  float    targetDeg[SERVO_STATUS_MAX];
  float    actualDeg[SERVO_STATUS_MAX];
  float    errorDeg[SERVO_STATUS_MAX];
  uint8_t  checksum;
} ServoStatusPacket;
#pragma pack(pop)
