#pragma once
#include <Arduino.h>
#include "config.h"
#include "PacketChecksum.h"

#define ANGLE_PACKET_HEADER 0xAB

#pragma pack(push, 1)
typedef struct {
  uint8_t  header;                  // 0xAB
  uint8_t  count;                   // servo 數量
  uint32_t seq;                     // sequence number
  float    targetDeg[bodyNum];
  uint8_t  checksum;                // XOR checksum
} AnglePacket;
#pragma pack(pop)

// 保留舊名稱相容
static inline uint8_t calcXorChecksum(const uint8_t* data, size_t len) {
  return calcPacketChecksum(data, len);
}

// TX
static inline void sendAnglePacketUART(
  HardwareSerial& serial,
  const float* targetDeg,
  uint8_t count,
  uint32_t seq
) {
  AnglePacket pkt;
  pkt.header = ANGLE_PACKET_HEADER;
  pkt.count  = count;
  pkt.seq    = seq;

  for (int i = 0; i < bodyNum; i++) {
    pkt.targetDeg[i] = (i < count) ? targetDeg[i] : 0.0f;
  }

  pkt.checksum = calcPacketChecksum(
    reinterpret_cast<uint8_t*>(&pkt),
    sizeof(AnglePacket) - 1
  );

  serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(AnglePacket));
}

// RX state
typedef struct {
  AnglePacket pkt;
  size_t index = 0;
  bool receiving = false;
} AngleRxState;

// feed RX
static inline bool feedAngleRx(AngleRxState &st, uint8_t b) {
  uint8_t* buf = reinterpret_cast<uint8_t*>(&st.pkt);

  if (!st.receiving) {
    if (b == ANGLE_PACKET_HEADER) {
      st.receiving = true;
      st.index = 0;
      buf[st.index++] = b;
    }
    return false;
  }

  buf[st.index++] = b;

  if (st.index >= sizeof(AnglePacket)) {
    st.receiving = false;

    uint8_t cs = calcPacketChecksum(
      reinterpret_cast<uint8_t*>(&st.pkt),
      sizeof(AnglePacket) - 1
    );

    return (cs == st.pkt.checksum);
  }

  return false;
}