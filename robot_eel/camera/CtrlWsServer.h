#pragma once
#include <WebSocketsServer.h>

#include "CtrlUartBridge.h"
#include "ControltoCamera.h"
#include "config.h"   // ✅ bodyNum

namespace CtrlWsServer {

    void begin(WebSocketsServer &ws);

    void tick();

    void broadcastServoStatus(
        uint8_t count,
        uint32_t seq,
        const float *target,
        const float *actual,
        const float *error
    );
}
