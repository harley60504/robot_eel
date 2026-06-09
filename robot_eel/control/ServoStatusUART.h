#ifndef SERVO_STATUS_UART_H
#define SERVO_STATUS_UART_H

#include <Arduino.h>
#include "config.h"
#include "PacketChecksum.h"

#define SERVO_STATUS_HEADER 0xBB
#define SERVO_MAX bodyNum

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  uint8_t  count;
  uint32_t seq;
  float    targetDeg[SERVO_MAX];
  float    actualDeg[SERVO_MAX];
  float    errorDeg[SERVO_MAX];
  uint8_t  checksum;
} ServoStatusPacket;
#pragma pack(pop)

extern ServoState servoState[];
extern float angleDeg[];

extern ServoStatusPacket g_status;
extern SemaphoreHandle_t statusMutex;
extern volatile uint32_t g_servoStatusSeq;

static inline void sendServoStatusUART(HardwareSerial& serial)
{
  if (!statusMutex) return;

  if (!xSemaphoreTake(statusMutex, 0))
    return;

  g_status.header = SERVO_STATUS_HEADER;
  g_status.count  = SERVO_MAX;
  g_status.seq    = g_servoStatusSeq++;

  g_status.checksum = calcPacketChecksum(
    reinterpret_cast<uint8_t*>(&g_status),
    sizeof(ServoStatusPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&g_status), sizeof(ServoStatusPacket));

  xSemaphoreGive(statusMutex);
}

static inline void servoStatusTxTask(void *pv)
{
  TickType_t lastWake = xTaskGetTickCount();

  while (true)
  {
    sendServoStatusUART(Serial2);
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(80));
  }
}

#endif