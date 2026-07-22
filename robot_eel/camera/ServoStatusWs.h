#pragma once

#include <WebServer.h>
#include <WebSocketsServer.h>

#include "CtrlUartBridge.h"

namespace ServoStatusWs {

    void begin(WebSocketsServer &ws);

    void broadcast(
        uint8_t count,
        uint32_t seq,
        const float *target,
        const float *actual,
        const float *error
    );

    void broadcastImu(const ImuPacket &imu);

    void clearImuLog();

    void sendImuCsv(WebServer &server);

    size_t imuLogCount();

    void clearLog();

    void sendCsv(WebServer &server);

    size_t logCount();
}
