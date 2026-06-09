#ifndef CONTROL_TO_CAMERA_H
#define CONTROL_TO_CAMERA_H

#include <Arduino.h>
#include "PacketChecksum.h"
#include "config.h"

/* ===============================
 *  UART Control Packet Definition
 * =============================== */

#define CONTROL_PACKET_HEADER 0xAA

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;        // 0xAA
  float    Ajoint;
  float    frequency;
  float    lambda;
  float    L;
  float    ampScales[bodyNum];
  float    phaseLags[bodyNum - 1];
  float    jointBiasDeg[bodyNum];
  bool     isPaused;
  uint8_t  controlMode;
  bool     useFeedback;
  float    feedbackGain;
  uint8_t  checksum;      // XOR checksum
} ControlPacket;
#pragma pack(pop)


/* ===============================
 *  舊名稱相容：仍保留這個名字
 * =============================== */
static inline uint8_t calcControlChecksum(const uint8_t* data, size_t len) {
  return calcPacketChecksum(data, len);
}


/* ===============================
 *  UART Send Interface (TX)
 * =============================== */
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
  bool   useFeedback,
  float  feedbackGain
) {
  ControlPacket pkt;

  pkt.header        = CONTROL_PACKET_HEADER;
  pkt.Ajoint        = Ajoint;
  pkt.frequency     = frequency;
  pkt.lambda        = lambda;
  pkt.L             = L;
  for (int i = 0; i < bodyNum; i++) {
    pkt.ampScales[i] = ampScales[i];
    pkt.jointBiasDeg[i] = jointBiasDeg[i];
  }
  for (int i = 0; i < bodyNum - 1; i++) {
    pkt.phaseLags[i] = phaseLags[i];
  }
  pkt.isPaused      = isPaused;
  pkt.controlMode   = controlMode;
  pkt.useFeedback   = useFeedback;
  pkt.feedbackGain  = feedbackGain;

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<const uint8_t*>(&pkt),
    sizeof(ControlPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(ControlPacket));
}


/* ===============================
 *  UART Receive State (RX)
 * =============================== */
typedef struct {
  ControlPacket pkt;
  size_t index = 0;
  bool receiving = false;
} ControlRxState;


/* ===============================
 *  Feed byte (return true = ok)
 * =============================== */
static inline bool feedControlRx(ControlRxState &st, uint8_t b) {
  uint8_t* buf = reinterpret_cast<uint8_t*>(&st.pkt);

  if (!st.receiving) {
    if (b == CONTROL_PACKET_HEADER) {
      st.receiving = true;
      st.index = 0;
      buf[st.index++] = b;
    }
    return false;
  }

  buf[st.index++] = b;

  if (st.index >= sizeof(ControlPacket)) {
    st.receiving = false;

    uint8_t cs = calcPacketChecksum(
      reinterpret_cast<uint8_t*>(&st.pkt),
      sizeof(ControlPacket) - 1
    );

    return (cs == st.pkt.checksum);
  }

  return false;
}

#endif
