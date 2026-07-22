#include "CtrlUartBridge.h"
#include <cstring>

// ==== UART & 解包狀態 ====
static HardwareSerial* g_ser = nullptr;

// ControlParamsPacket RX parser
static ControlParamsRxState g_ctrlRx;

// ServoStatusPacket RX state
static uint8_t buf[sizeof(ServoStatusPacket)];
static size_t idx = 0;
static bool receivingServo = false;

static const size_t SERVO_PKT_SIZE = sizeof(ServoStatusPacket);

static uint8_t imuBuf[sizeof(ImuPacket)];
static size_t imuIdx = 0;
static bool receivingImu = false;
static const size_t IMU_PKT_SIZE = sizeof(ImuPacket);

// callbacks
std::function<void(const ControlParamsPacket&)> CtrlUartBridge::onCtrlParams = nullptr;
std::function<void(const ServoStatusPacket&)> CtrlUartBridge::onServoStatusPacket = nullptr;
std::function<void(const ImuPacket&)> CtrlUartBridge::onImuPacket = nullptr;

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
      // 1) ServoStatusPacket (0xBB)
      // =====================================================
      if (receivingServo)
      {
        buf[idx++] = b;

        if (idx >= SERVO_PKT_SIZE)
        {
          receivingServo = false;

          if (buf[0] == SERVO_STATUS_PACKET_HEADER)
          {
            ServoStatusPacket ss;
            memcpy(&ss, buf, SERVO_PKT_SIZE);

            uint8_t cs = calcPacketChecksum(
              reinterpret_cast<uint8_t*>(&ss),
              SERVO_PKT_SIZE - 1
            );

            if (cs == ss.checksum)
            {
              if (CtrlUartBridge::onServoStatusPacket)
              {
                CtrlUartBridge::onServoStatusPacket(ss);
              }
            }
          }

          idx = 0;
        }
        continue;
      }

      if (receivingImu)
      {
        imuBuf[imuIdx++] = b;

        if (imuIdx >= IMU_PKT_SIZE)
        {
          receivingImu = false;

          if (imuBuf[0] == IMU_PACKET_HEADER)
          {
            ImuPacket imu;
            memcpy(&imu, imuBuf, IMU_PKT_SIZE);

            uint8_t cs = calcPacketChecksum(
              reinterpret_cast<uint8_t*>(&imu),
              IMU_PKT_SIZE - 1
            );

            if (cs == imu.checksum)
            {
              if (CtrlUartBridge::onImuPacket)
              {
                CtrlUartBridge::onImuPacket(imu);
              }
            }
          }

          imuIdx = 0;
        }
        continue;
      }

      // =====================================================
      // 2) ControlParamsPacket (0xAA)
      // =====================================================
      if (g_ctrlRx.receiving)
      {
        if (feedControlParamsRx(g_ctrlRx, b))
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
      if (b == SERVO_STATUS_PACKET_HEADER)
      {
        receivingServo = true;
        idx = 0;
        buf[idx++] = b;
        continue;
      }

      if (b == IMU_PACKET_HEADER)
      {
        receivingImu = true;
        imuIdx = 0;
        imuBuf[imuIdx++] = b;
        continue;
      }

      if (b == CONTROL_PARAMS_PACKET_HEADER)
      {
        feedControlParamsRx(g_ctrlRx, b);
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
void CtrlUartBridge::sendCtrlParams(const ControlParamsPacket &pkt)
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
    pkt.servoDefaultAngles
  );
}

// ==================================================
// TX：ServoTargetPacket（camera → 控制板）
// ==================================================
void CtrlUartBridge::sendAngle(const float* targetDeg, uint8_t count)
{
  if (!g_ser) return;

  if (count == 0) return;
  if (count > bodyNum) count = bodyNum;

  static uint32_t seq = 0;

  sendServoTargetPacketUART(
    *g_ser,
    targetDeg,
    count,
    seq++
  );
}

void CtrlUartBridge::sendServoCenter(const float* centerDeg, uint8_t count, bool save)
{
  if (!g_ser) return;

  if (count == 0) return;
  if (count > bodyNum) count = bodyNum;

  static uint32_t seq = 0;

  sendServoCenterPacketUART(
    *g_ser,
    centerDeg,
    count,
    save,
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
