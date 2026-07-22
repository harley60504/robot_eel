#ifndef IMU_PACKET_H
#define IMU_PACKET_H

#include <Arduino.h>
#include "UartPacketChecksum.h"

#define IMU_PACKET_HEADER 0xBC

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  uint32_t seq;
  uint32_t t_ms;
  float    accel[3];
  float    gyro[3];
  float    tempC;
  uint8_t  checksum;
} ImuPacket;
#pragma pack(pop)

#endif
