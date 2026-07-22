#include <Arduino.h>
#include <mbed.h>

#include "config.h"
#include "utils.h"
#include "cpg.h"
#include "servo.h"
#include "ServoStatusPacket.h"
#include "ControlParamsPacket.h"
#include "ServoTargetPacket.h"
#include "ServoCenterPacket.h"

#if defined(ARDUINO_ARCH_MBED)
UART CameraSerial(
  digitalPinToPinName(CAMERA_TX_PIN),
  digitalPinToPinName(CAMERA_RX_PIN),
  NC,
  NC
);
#else
#define CameraSerial Serial
#endif

ServoStatusPacket g_status;
rtos::Mutex statusMutex;
volatile uint32_t g_servoStatusSeq = 0;

float servoDefaultAngles[bodyNum] = {120, 120, 120, 120, 120, 120};
float angleDeg[bodyNum];

float Ajoint       = 20.0f;
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

bool isPaused = false;
int controlMode = MODE_OFFSET;

HopfOscillator cpg[bodyNum];

static ControlParamsRxState camCtrlRx;
static ServoTargetRxState camTargetRx;
static ServoCenterRxState camCenterRx;

volatile bool g_haveAngleCmd = false;
float g_uartTargetDeg[bodyNum] = {0};
volatile uint32_t g_lastAngleSeq = 0;
rtos::Mutex angleMutex;

static rtos::Thread servoThread(osPriorityAboveNormal, 4096, nullptr, "servoTask");
static rtos::Thread cameraTxThread(osPriorityNormal, 4096, nullptr, "cameraTxTask");
static rtos::Thread cameraRxThread(osPriorityAboveNormal, 4096, nullptr, "cameraRxTask");
static rtos::Thread servoStatusThread(osPriorityNormal, 4096, nullptr, "servoStatusTxTask");

void loadServoCenters()
{
  Serial.println("[CENTER] nRF52840 version uses built-in defaults.");
}

void saveServoCenters()
{
  Serial.println("[CENTER] save requested, but persistent storage is not enabled in this nRF52840 sketch.");
}

void applyServoCenters(const ServoCenterPacket &pkt)
{
  if (pkt.count != bodyNum) {
    return;
  }

  for (int i = 0; i < bodyNum; i++) {
    servoDefaultAngles[i] = constrain(pkt.centerDeg[i], 0.0f, 240.0f);
  }

  if (pkt.save) {
    saveServoCenters();
  }

  Serial.print("[CENTER] packet OK seq=");
  Serial.print((unsigned long)pkt.seq);
  Serial.print(" save=");
  Serial.println((unsigned)pkt.save);
}

void sendCameraControlParams()
{
  sendControlParamsUART(
    CameraSerial,
    Ajoint,
    frequency,
    lambda,
    L,
    ampScales,
    phaseLags,
    jointBiasDeg,
    isPaused,
    (uint8_t)controlMode,
    servoDefaultAngles
  );
}

void cameraTxTask()
{
  while (true) {
    sendCameraControlParams();
    rtos::ThisThread::sleep_for(std::chrono::milliseconds(100));
  }
}

void handleControlPacket(const ControlParamsPacket &pkt)
{
  int previousMode = controlMode;

  Ajoint = pkt.Ajoint;
  frequency = pkt.frequency;
  lambda = pkt.lambda;
  L = pkt.L;

  for (int i = 0; i < bodyNum; i++) {
    ampScales[i] = pkt.ampScales[i];
    jointBiasDeg[i] = pkt.jointBiasDeg[i];
  }

  for (int i = 0; i < bodyNum - 1; i++) {
    phaseLags[i] = pkt.phaseLags[i];
  }

  isPaused = pkt.isPaused;
  controlMode = pkt.controlMode;

  if (controlMode != MODE_UART_ANGLE) {
    g_haveAngleCmd = false;
  }

  if (previousMode != MODE_CPG && controlMode == MODE_CPG) {
    initCPG();
  }

  Serial.print("[UART] ControlParams OK mode=");
  Serial.print(controlMode);
  Serial.print(" pause=");
  Serial.println((int)isPaused);
}

void serviceCameraRx()
{
  while (CameraSerial.available()) {
    uint8_t b = CameraSerial.read();

    if (camTargetRx.receiving) {
      if (feedServoTargetRx(camTargetRx, b)) {
        ServoTargetPacket &pkt = camTargetRx.pkt;

        if (pkt.count == bodyNum && pkt.seq != g_lastAngleSeq) {
          angleMutex.lock();
          for (int i = 0; i < bodyNum; i++) {
            g_uartTargetDeg[i] = pkt.targetDeg[i];
          }
          angleMutex.unlock();

          g_lastAngleSeq = pkt.seq;
          g_haveAngleCmd = true;
        }

        Serial.print("[UART] ServoTarget OK seq=");
        Serial.println((unsigned long)pkt.seq);
      }
      continue;
    }

    if (camCenterRx.receiving) {
      if (feedServoCenterRx(camCenterRx, b)) {
        applyServoCenters(camCenterRx.pkt);
      }
      continue;
    }

    if (camCtrlRx.receiving) {
      if (feedControlParamsRx(camCtrlRx, b)) {
        handleControlPacket(camCtrlRx.pkt);
      }
      continue;
    }

    if (b == CONTROL_PARAMS_PACKET_HEADER) {
      feedControlParamsRx(camCtrlRx, b);
      continue;
    }

    if (b == SERVO_TARGET_PACKET_HEADER) {
      feedServoTargetRx(camTargetRx, b);
      continue;
    }

    if (b == SERVO_CENTER_PACKET_HEADER) {
      feedServoCenterRx(camCenterRx, b);
      continue;
    }
  }
}

void cameraRxTask()
{
  while (true) {
    serviceCameraRx();
    rtos::ThisThread::sleep_for(std::chrono::milliseconds(1));
  }
}

void servoStatusTxTask()
{
  while (true) {
    sendServoStatusUART(CameraSerial);
    rtos::ThisThread::sleep_for(std::chrono::milliseconds(80));
  }
}

void setup()
{
  Serial.begin(115200);
  uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 5000) {
    delay(10);
  }
  delay(300);

  Serial1.begin(115200);
  CameraSerial.begin(115200);

  Serial.println("XIAO nRF52840 Control Board Ready");
  loadServoCenters();
  initCPG();

  servoThread.start(servoTask);
  cameraTxThread.start(cameraTxTask);
  cameraRxThread.start(cameraRxTask);
  servoStatusThread.start(servoStatusTxTask);
}

void loop()
{
  static uint32_t lastHeartbeatMs = 0;
  uint32_t now = millis();

  if (now - lastHeartbeatMs >= 1000) {
    lastHeartbeatMs = now;
    Serial.println("[HB] nRF52840 control alive");
  }

  delay(1000);
}
