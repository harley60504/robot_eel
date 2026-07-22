#pragma once

#include <Arduino.h>

#include "UartPacketChecksum.h"
#include "config.h"

#define SERVO_TARGET_PACKET_HEADER 0xAB

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  uint8_t  count;
  uint32_t seq;
  float    targetDeg[bodyNum];
  uint8_t  checksum;
} ServoTargetPacket;
#pragma pack(pop)

static inline void sendServoTargetPacketUART(
  Print& serial,
  const float* targetDeg,
  uint8_t count,
  uint32_t seq
) {
  ServoTargetPacket pkt;
  pkt.header = SERVO_TARGET_PACKET_HEADER;
  pkt.count  = count;
  pkt.seq    = seq;

  for (int i = 0; i < bodyNum; i++) {
    pkt.targetDeg[i] = (i < count) ? targetDeg[i] : 0.0f;
  }

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<uint8_t*>(&pkt),
    sizeof(ServoTargetPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(ServoTargetPacket));
}

typedef struct {
  ServoTargetPacket pkt;
  size_t index = 0;
  bool receiving = false;
} ServoTargetRxState;

static inline bool feedServoTargetRx(ServoTargetRxState &st, uint8_t b) {
  uint8_t* buf = reinterpret_cast<uint8_t*>(&st.pkt);

  if (!st.receiving) {
    if (b == SERVO_TARGET_PACKET_HEADER) {
      st.receiving = true;
      st.index = 0;
      buf[st.index++] = b;
    }
    return false;
  }

  buf[st.index++] = b;

  if (st.index >= sizeof(ServoTargetPacket)) {
    st.receiving = false;

    uint8_t cs = calcPacketChecksum(
      reinterpret_cast<uint8_t*>(&st.pkt),
      sizeof(ServoTargetPacket) - 1
    );

    return (cs == st.pkt.checksum);
  }

  return false;
}
