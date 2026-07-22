#pragma once
#include <math.h>
#include <mbed.h>
#include "config.h"
#include "utils.h"
#include "cpg.h"
#include "ServoStatusPacket.h"
#include "ServoTargetPacket.h"
#include "ControlParamsPacket.h"

extern volatile bool g_haveAngleCmd;
extern float g_uartTargetDeg[bodyNum];
extern rtos::Mutex statusMutex;
extern rtos::Mutex angleMutex;

static inline void servoTask()
{
  const uint16_t MOVE_TIME_MS = 100;
  const float dt = MOVE_TIME_MS / 1000.0f;
  static uint32_t seq = 0;

  while (true) {
    if (!isPaused) {
      float t = millis() / 1000.0f;

      if (controlMode == MODE_CPG) {
        updateCPGAll(dt);
      }

      for (int j = 0; j < bodyNum; j++) {
        float targetDeg = servoDefaultAngles[j];

        switch (controlMode) {
          case MODE_SIN:
          {
            float outDeg =
              Ajoint *
              ampScales[j] *
              sinf(2 * PI * frequency * t + getPhaseOffset(j)) +
              jointBiasDeg[j];
            targetDeg = servoDefaultAngles[j] + outDeg;
          }
          break;

          case MODE_CPG:
            targetDeg = servoDefaultAngles[j] + getCPGOutput(j);
            break;

          case MODE_UART_ANGLE:
            if (g_haveAngleCmd && angleMutex.trylock()) {
              targetDeg = g_uartTargetDeg[j];
              angleMutex.unlock();
            }
            break;

          case MODE_OFFSET:
          default:
            targetDeg = servoDefaultAngles[j];
            break;
        }

        servoState[j].targetDeg = targetDeg;
        angleDeg[j] = targetDeg;
        moveLX224(j + 1, degreeToLX224(targetDeg), MOVE_TIME_MS);
      }

      rtos::ThisThread::sleep_for(std::chrono::milliseconds(MOVE_TIME_MS));

      for (int j = 0; j < bodyNum; j++) {
        int actualPos = readPositionLX224(j + 1);

        if (actualPos >= 0) {
          servoState[j].actualPos = actualPos;
          servoState[j].actualDeg = lx224ToDegree(actualPos);
          servoState[j].errorDeg = servoState[j].targetDeg - servoState[j].actualDeg;
        }
      }

      statusMutex.lock();
      g_status.header = SERVO_STATUS_PACKET_HEADER;
      g_status.count  = bodyNum;
      g_status.seq    = seq++;

      for (int i = 0; i < bodyNum; i++) {
        g_status.targetDeg[i] = servoState[i].targetDeg;
        g_status.actualDeg[i] = servoState[i].actualDeg;
        g_status.errorDeg[i]  = servoState[i].errorDeg;
      }

      g_status.checksum = calcPacketChecksum(
        reinterpret_cast<uint8_t*>(&g_status),
        sizeof(ServoStatusPacket) - 1
      );
      statusMutex.unlock();
    } else {
      rtos::ThisThread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}
