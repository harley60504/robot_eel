#pragma once

#include <Arduino.h>

#include "UartPacketChecksum.h"
#include "config.h"

#define SERVO_CENTER_PACKET_HEADER 0xAC

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;
  uint8_t  count;
  uint8_t  save;
  uint32_t seq;
  float    centerDeg[bodyNum];
  uint8_t  checksum;
} ServoCenterPacket;
#pragma pack(pop)

static inline void sendServoCenterPacketUART(
  HardwareSerial& serial,
  const float* centerDeg,
  uint8_t count,
  bool save,
  uint32_t seq
) {
  ServoCenterPacket pkt;
  pkt.header = SERVO_CENTER_PACKET_HEADER;
  pkt.count  = count;
  pkt.save   = save ? 1 : 0;
  pkt.seq    = seq;

  for (int i = 0; i < bodyNum; i++) {
    pkt.centerDeg[i] = (i < count) ? centerDeg[i] : 120.0f;
  }

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<uint8_t*>(&pkt),
    sizeof(ServoCenterPacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(ServoCenterPacket));
}

typedef struct {
  ServoCenterPacket pkt;
  size_t index = 0;
  bool receiving = false;
} ServoCenterRxState;

static inline bool feedServoCenterRx(ServoCenterRxState &st, uint8_t b) {
  uint8_t* buf = reinterpret_cast<uint8_t*>(&st.pkt);

  if (!st.receiving) {
    if (b == SERVO_CENTER_PACKET_HEADER) {
      st.receiving = true;
      st.index = 0;
      buf[st.index++] = b;
    }
    return false;
  }

  buf[st.index++] = b;

  if (st.index >= sizeof(ServoCenterPacket)) {
    st.receiving = false;

    uint8_t cs = calcPacketChecksum(
      reinterpret_cast<uint8_t*>(&st.pkt),
      sizeof(ServoCenterPacket) - 1
    );

    return (cs == st.pkt.checksum);
  }

  return false;
}
