#include "CtrlUartBridge.h"
#include <cstring>

// ==== UART & 解包狀態 ====
static HardwareSerial* g_ser = nullptr;

// ControlPacket RX parser
static ControlRxState g_ctrlRx;

// ServoStatus RX state
static uint8_t buf[sizeof(ServoStatus)];
static size_t idx = 0;
static bool receivingServo = false;

static const size_t SERVO_PKT_SIZE = sizeof(ServoStatus);

// callbacks
std::function<void(const ControlPacket&)> CtrlUartBridge::onCtrlParams = nullptr;
std::function<void(const ServoStatus&)> CtrlUartBridge::onServoStatus = nullptr;

// ==================================================
// UART RX Task
// ==================================================
static void uartRxTask(void *pv)
{
  while (true)
  {
    while (g_ser && g_ser->available())
    {
      uint8_t b = g_ser->read();

      // =====================================================
      // 1) ServoStatus (0xBB)
      // =====================================================
      if (receivingServo)
      {
        buf[idx++] = b;

        if (idx >= SERVO_PKT_SIZE)
        {
          receivingServo = false;

          if (buf[0] == SERVO_STATUS_HEADER)
          {
            ServoStatus ss;
            memcpy(&ss, buf, SERVO_PKT_SIZE);

            uint8_t cs = calcControlChecksum(
              reinterpret_cast<uint8_t*>(&ss),
              SERVO_PKT_SIZE - 1
            );

            if (cs == ss.checksum)
            {
              if (CtrlUartBridge::onServoStatus)
              {
                CtrlUartBridge::onServoStatus(ss);
              }
            }
          }

          idx = 0;
        }
        continue;
      }

      // =====================================================
      // 2) ControlPacket (0xAA)
      // =====================================================
      if (g_ctrlRx.receiving)
      {
        if (feedControlRx(g_ctrlRx, b))
        {
          if (CtrlUartBridge::onCtrlParams)
          {
            CtrlUartBridge::onCtrlParams(g_ctrlRx.pkt);
          }
        }
        continue;
      }

      // =====================================================
      // 3) Idle：只認 header
      // =====================================================
      if (b == SERVO_STATUS_HEADER)
      {
        receivingServo = true;
        idx = 0;
        buf[idx++] = b;
        continue;
      }

      if (b == CONTROL_PACKET_HEADER)
      {
        feedControlRx(g_ctrlRx, b);
        continue;
      }

      // 其他 byte 丟掉
    }

    vTaskDelay(1);
  }
}

// ==================================================
// TX：控制參數（camera → 控制板）
// ==================================================
void CtrlUartBridge::sendCtrlParams(const ControlPacket &pkt)
{
  if (!g_ser) return;

  sendControlParamsUART(
    *g_ser,
    pkt.Ajoint,
    pkt.frequency,
    pkt.lambda,
    pkt.L,
    pkt.ampScales,
    pkt.phaseLags,
    pkt.jointBiasDeg,
    pkt.isPaused,
    pkt.controlMode,
    pkt.useFeedback,
    pkt.feedbackGain
  );
}

// ==================================================
// TX：AnglePacket（camera → 控制板）
// ==================================================
void CtrlUartBridge::sendAngle(const float* targetDeg, uint8_t count)
{
  if (!g_ser) return;

  if (count == 0) return;
  if (count > bodyNum) count = bodyNum;

  static uint32_t seq = 0;

  sendAnglePacketUART(
    *g_ser,
    targetDeg,
    count,
    seq++
  );
}

// ==================================================
// INIT
// ==================================================
void CtrlUartBridge::begin(HardwareSerial& ser,
                           uint32_t baud,
                           int rxPin,
                           int txPin)
{
  g_ser = &ser;

  ser.begin(
    baud,
    SERIAL_8N1,
    rxPin,
    txPin
  );

  xTaskCreatePinnedToCore(
    uartRxTask,
    "uartRxTask",
    4096,
    nullptr,
    1,
    nullptr,
    1
  );
}
