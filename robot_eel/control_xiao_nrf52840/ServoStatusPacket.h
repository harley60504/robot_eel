#ifndef SERVO_STATUS_PACKET_H
#define SERVO_STATUS_PACKET_H

#include <Arduino.h>
#include <mbed.h>
#include "config.h"
#include "UartPacketChecksum.h"

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

extern ServoState servoState[];
extern float angleDeg[];

extern ServoStatusPacket g_status;
extern rtos::Mutex statusMutex;
extern volatile uint32_t g_servoStatusSeq;

static inline void sendServoStatusUART(Print& serial)
{
  if (!statusMutex.trylock()) {
    return;
  }

  g_status.header = SERVO_STATUS_PACKET_HEADER;
  g_status.count  = SERVO_STATUS_MAX;
  g_status.seq    = g_servoStatusSeq++;

  g_status.checksum = calcPacketChecksum(
    reinterpret_cast<uint8_t*>(&g_status),
    sizeof(ServoStatusPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&g_status), sizeof(ServoStatusPacket));
  statusMutex.unlock();
}

#endif
