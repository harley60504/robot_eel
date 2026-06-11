#include <Arduino.h>
#include "driver/uart.h"
#include "freertos/semphr.h"

#include "config.h"
#include "utils.h"
#include "cpg.h"
#include "servo.h"
#include "ServoStatusUART.h"
#include "ControltoCamera.h"
#include "AnglePacket.h"




// ServoStatusPacket
ServoStatusPacket g_status;
SemaphoreHandle_t statusMutex = NULL;
volatile uint32_t g_servoStatusSeq = 0;
// ==========================
//  Servo defaults
// ==========================
float servoDefaultAngles[bodyNum] = {120,120,120,120,120,120};
float angleDeg[bodyNum];

// ==========================
//  Control Parameters
// ==========================
// RL exported params: mujoco_simulation/gaits/rl_straight.json
float Ajoint       = 15.0f;  // deg
float frequency    = 1.0f;
float lambda       = 1.6275f;
float L            = 1.0f;
float ampScales[bodyNum] = {
  1.1f,
  0.95f,
  0.9f,
  1.071703f,
  1.161346f,
  1.273484f
};
float phaseLags[bodyNum - 1] = {
  0.614385f,
  0.622822f,
  0.615807f,
  0.615359f,
  0.608868f
};
float jointBiasDeg[bodyNum] = {0, 0, 0, 0, 0, 0};

bool  isPaused     = false;
int   controlMode  = 2;
bool  useFeedback  = false;
float feedbackGain = 1.0f;

// ==========================
HopfOscillator cpg[bodyNum];
unsigned long g_lastLogTime = 0;

// ==========================
// RX state (ControlPacket)
// ==========================
static ControlRxState camCtrlRx;

// ==========================
// RX state (AnglePacket)
// ==========================
static AngleRxState camAngleRx;

// ==========================
// UART Angle cache (shared with servoTask)
// ==========================
volatile bool g_haveAngleCmd = false;
float g_uartTargetDeg[bodyNum] = {0};
volatile uint32_t g_lastAngleSeq = 0;

SemaphoreHandle_t angleMutex = NULL;

// ==========================
// UART TX Task  (→ Camera): send control params
// ==========================
void cameraTxTask(void* pv)
{
  TickType_t lastWake = xTaskGetTickCount();

  while(true)
  {
    sendControlParamsUART(
      Serial2,
      Ajoint,
      frequency,
      lambda,
      L,
      ampScales,
      phaseLags,
      jointBiasDeg,
      isPaused,
      (uint8_t)controlMode,
      useFeedback,
      feedbackGain
    );

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(100));   // 10Hz
  }
}

// ==========================
// UART RX Task (← Camera): demux by header 0xAA / 0xAB
// ==========================
void cameraRxTask(void* pv)
{
  Serial.println("Camera RX Task started");

  while(true)
  {
    while(Serial2.available())
    {
      uint8_t b = Serial2.read();

      // =====================================================
      // 優先處理：如果 Angle parser 正在接收，就只餵 Angle
      // =====================================================
      if (camAngleRx.receiving)
      {
        if (feedAngleRx(camAngleRx, b))
        {
          AnglePacket &pkt = camAngleRx.pkt;

          if (pkt.count == bodyNum)
          {
            if (pkt.seq != g_lastAngleSeq)
            {
              if (angleMutex &&
                  xSemaphoreTake(angleMutex, portMAX_DELAY) == pdTRUE)
              {
                for (int i = 0; i < bodyNum; i++)
                {
                  g_uartTargetDeg[i] = pkt.targetDeg[i];
                }

                g_lastAngleSeq = pkt.seq;
                g_haveAngleCmd = true;

                xSemaphoreGive(angleMutex);
              }
            }
          }

          // Debug（可關）
          Serial.printf("[UART] AnglePacket OK seq=%lu count=%u\n",
                        (unsigned long)pkt.seq, (unsigned)pkt.count);
        }

        // ✅ 已在 Angle 狀態下，這個 byte 不要再給其他 parser
        continue;
      }

      // =====================================================
      // 優先處理：如果 Control parser 正在接收，就只餵 Control
      // =====================================================
      if (camCtrlRx.receiving)
      {
        if (feedControlRx(camCtrlRx, b))
        {
          ControlPacket &pkt = camCtrlRx.pkt;
          int previousMode = controlMode;

          Ajoint       = pkt.Ajoint;
          frequency    = pkt.frequency;
          lambda       = pkt.lambda;
          L            = pkt.L;
          for (int i = 0; i < bodyNum; i++) {
            ampScales[i] = pkt.ampScales[i];
            jointBiasDeg[i] = pkt.jointBiasDeg[i];
          }
          for (int i = 0; i < bodyNum - 1; i++) {
            phaseLags[i] = pkt.phaseLags[i];
          }
          isPaused     = pkt.isPaused;
          controlMode  = pkt.controlMode;

          // ✅ 如果切到非 Angle 模式，把角度命令標記清掉（避免殘留）
          if (controlMode != MODE_UART_ANGLE) {
            g_haveAngleCmd = false;
          }

          if (previousMode != MODE_CPG && controlMode == MODE_CPG) {
            initCPG();
          }

          useFeedback  = pkt.useFeedback;
          feedbackGain = pkt.feedbackGain;

          Serial.println("==== UART ← Camera (ControlPacket) ====");
          Serial.printf("mode=%d pause=%d A=%.2f f=%.2f lambda=%.2f L=%.2f fb=%d gain=%.2f\n",
                        controlMode, (int)isPaused,
                        Ajoint, frequency, lambda, L,
                        (int)useFeedback, feedbackGain);
        }

        // ✅ 已在 Control 狀態下，這個 byte 不要再給其他 parser
        continue;
      }

      // =====================================================
      // Idle 狀態：只認 header，才開始接收
      // =====================================================
      if (b == CONTROL_PACKET_HEADER)
      {
        // 啟動 Control parser（把 header 也放進去）
        feedControlRx(camCtrlRx, b);
        continue;
      }

      if (b == ANGLE_PACKET_HEADER)
      {
        // 啟動 Angle parser（把 header 也放進去）
        feedAngleRx(camAngleRx, b);
        continue;
      }

      // 其他 byte：丟棄
    }

    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ==========================
// SETUP
// ==========================
void setup()
{
  Serial.begin(115200);
  delay(300);
  statusMutex = xSemaphoreCreateMutex();
  // ✅ 建立 angleMutex（很重要）
  angleMutex = xSemaphoreCreateMutex();

  if (!statusMutex || !angleMutex) {
    Serial.println("ERROR: Mutex create failed!");
    while (1) delay(1000);
  }
  // Servo UART (RS485)
  Serial1.begin(115200, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
  uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);

  // Camera UART
  Serial2.begin(115200, SERIAL_8N1, CAMERA_RX_PIN, CAMERA_TX_PIN);

  Serial.println("Control Board Ready");

  initCPG();
  // Servo Task
  xTaskCreatePinnedToCore(
    servoTask,
    "servoTask",
    4096,
    nullptr,
    2,
    nullptr,
    1
  );

  // UART TX Task
  xTaskCreatePinnedToCore(
    cameraTxTask,
    "cameraTxTask",
    4096,
    nullptr,
    1,
    nullptr,
    0
  );

  // UART RX Task (demux Control + Angle)
  xTaskCreatePinnedToCore(
    cameraRxTask,
    "cameraRxTask",
    4096,
    nullptr,
    2,       // 建議比 TX 高一點點，避免 RX 積壓
    nullptr,
    0
  );

  // Servo status TX Task (回傳 target/actual/error)
  xTaskCreatePinnedToCore(
    servoStatusTxTask,
    "servoStatusTxTask",
    4096,
    nullptr,
    1,
    nullptr,
    0
  );
}

// ==========================
// MAIN LOOP
// ==========================
void loop()
{
}
