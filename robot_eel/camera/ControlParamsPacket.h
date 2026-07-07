#pragma once

#include <Arduino.h>

#include "UartPacketChecksum.h"
#include "config.h"

#define CONTROL_PARAMS_PACKET_HEADER 0xAA

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  float    Ajoint;
  float    frequency;
  float    lambda;
  float    L;
  float    ampScales[bodyNum];
  float    phaseLags[bodyNum - 1];
  float    jointBiasDeg[bodyNum];
  float    servoDefaultAngles[bodyNum];
  bool     isPaused;
  uint8_t  controlMode;
  uint8_t  checksum;
} ControlParamsPacket;
#pragma pack(pop)

static inline void sendControlParamsUART(
  HardwareSerial& serial,
  float  Ajoint,
  float  frequency,
  float  lambda,
  float  L,
  const float* ampScales,
  const float* phaseLags,
  const float* jointBiasDeg,
  bool   isPaused,
  uint8_t controlMode,
  const float* servoDefaultAngles = nullptr
) {
  ControlParamsPacket pkt;

  pkt.header = CONTROL_PARAMS_PACKET_HEADER;
  pkt.Ajoint = Ajoint;
  pkt.frequency = frequency;
  pkt.lambda = lambda;
  pkt.L = L;
  for (int i = 0; i < bodyNum; i++) {
    pkt.ampScales[i] = ampScales[i];
    pkt.jointBiasDeg[i] = jointBiasDeg[i];
    pkt.servoDefaultAngles[i] = servoDefaultAngles ? servoDefaultAngles[i] : 120.0f;
  }
  for (int i = 0; i < bodyNum - 1; i++) {
    pkt.phaseLags[i] = phaseLags[i];
  }
  pkt.isPaused = isPaused;
  pkt.controlMode = controlMode;

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<const uint8_t*>(&pkt),
    sizeof(ControlParamsPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(ControlParamsPacket));
}

typedef struct {
  ControlParamsPacket pkt;
  size_t index = 0;
  bool receiving = false;
} ControlParamsRxState;

static inline bool feedControlParamsRx(ControlParamsRxState &st, uint8_t b) {
  uint8_t* buf = reinterpret_cast<uint8_t*>(&st.pkt);

  if (!st.receiving) {
    if (b == CONTROL_PARAMS_PACKET_HEADER) {
      st.receiving = true;
      st.index = 0;
      buf[st.index++] = b;
    }
    return false;
  }

  buf[st.index++] = b;

  if (st.index >= sizeof(ControlParamsPacket)) {
    st.receiving = false;

    uint8_t cs = calcPacketChecksum(
      reinterpret_cast<uint8_t*>(&st.pkt),
      sizeof(ControlParamsPacket) - 1
    );

    return (cs == st.pkt.checksum);
  }

  return false;
}
