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

static inline void sendImuPacketUART(
  Print& serial,
  uint32_t seq,
  uint32_t t_ms,
  float ax,
  float ay,
  float az,
  float gx,
  float gy,
  float gz,
  float tempC
) {
  ImuPacket pkt;
  pkt.header = IMU_PACKET_HEADER;
  pkt.seq = seq;
  pkt.t_ms = t_ms;
  pkt.accel[0] = ax;
  pkt.accel[1] = ay;
  pkt.accel[2] = az;
  pkt.gyro[0] = gx;
  pkt.gyro[1] = gy;
  pkt.gyro[2] = gz;
  pkt.tempC = tempC;

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<const uint8_t*>(&pkt),
    sizeof(ImuPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(ImuPacket));
}

#endif
