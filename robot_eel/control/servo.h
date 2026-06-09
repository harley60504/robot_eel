#pragma once
#include <math.h>
#include "config.h"
#include "utils.h"
#include "cpg.h"
#include "ServoStatusUART.h"

// ✅ 新增：UART Angle Packet
#include "AnglePacket.h"
#include "ControltoCamera.h"
// ✅ UART 角度快取（由 RX Task 更新）
extern volatile bool g_haveAngleCmd;
extern float g_uartTargetDeg[bodyNum];
extern SemaphoreHandle_t angleMutex;

void servoTask(void *pv)
{
  const uint16_t MOVE_TIME_MS = 100;
  const float    dt = MOVE_TIME_MS / 1000.0f;
  static uint32_t seq = 0;
  TickType_t lastWake = xTaskGetTickCount();

  for (;;)
  {
    if (!isPaused)
    {
      float t = millis() / 1000.0f;

      /* ========= 1. 計算 target 並輸出 MOVE ========= */
      for (int j = 0; j < bodyNum; j++)
      {
        float targetDeg = servoDefaultAngles[j]; // ✅ 預設保底

        switch (controlMode)
        {
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
          {
            float fb_phase = 0, fb_amp = 0;
            updateCPG(t, dt, j, fb_phase, fb_amp);

            float outDeg = getCPGOutput(j);
            targetDeg = servoDefaultAngles[j] + outDeg;
          }
          break;

          case MODE_OFFSET:
          {
            // 全部回到 default
            targetDeg = servoDefaultAngles[j];
          }
          break;

          case MODE_UART_ANGLE:
          {
            // ✅ UART Angle Mode：直接吃 UART 傳來的角度
            if (g_haveAngleCmd)
            {
              // 用 mutex 保護 g_uartTargetDeg
              if (xSemaphoreTake(angleMutex, 0) == pdTRUE)
              {
                targetDeg = g_uartTargetDeg[j];
                xSemaphoreGive(angleMutex);
              }
            }
            else
            {
              // 如果還沒收到 UART 指令 → 保持預設角度
              targetDeg = servoDefaultAngles[j];
            }
          }
          break;

          default:
          {
            // 未知模式：回預設角度
            targetDeg = servoDefaultAngles[j];
          }
          break;
        }

        // ✅ 寫入 state
        servoState[j].targetDeg = targetDeg;
        angleDeg[j] = targetDeg;

        // ✅ 下發 move
        int pos = degreeToLX224(targetDeg);
        moveLX224(j + 1, pos, MOVE_TIME_MS);
      }

      /* ========= 2. 等待 servo 完成運動 ========= */
      vTaskDelay(pdMS_TO_TICKS(MOVE_TIME_MS));

      /* ========= 3. 同步讀回授 ========= */
      for (int j = 0; j < bodyNum; j++)
      {
        int actualPos = readPositionLX224(j + 1);

        if (actualPos >= 0)
        {
          servoState[j].actualPos = actualPos;

          float actualDeg = lx224ToDegree(actualPos);
          servoState[j].actualDeg = actualDeg;

          servoState[j].errorDeg =
            servoState[j].targetDeg - actualDeg;
        }
      }

      /* ========= 4. 建立封包 SNAPSHOT ========= */
      if (xSemaphoreTake(statusMutex, portMAX_DELAY))
      {
        g_status.header = SERVO_STATUS_HEADER;
        g_status.count  = bodyNum;
        g_status.seq    = seq++;

        for(int i=0;i<bodyNum;i++)
        {
          g_status.targetDeg[i] = servoState[i].targetDeg;
          g_status.actualDeg[i] = servoState[i].actualDeg;
          g_status.errorDeg[i]  = servoState[i].errorDeg;
        }

        g_status.checksum = calcControlChecksum(
          (uint8_t*)&g_status,
          sizeof(ServoStatusPacket) - 1
        );

        xSemaphoreGive(statusMutex);
      }
    }
    else
    {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}
