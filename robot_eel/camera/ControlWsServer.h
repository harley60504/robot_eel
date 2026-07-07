#pragma once
#include <WebSocketsServer.h>

#include "CtrlUartBridge.h"
#include "ControlParamsPacket.h"
#include "config.h"   // ✅ bodyNum

namespace ControlWsServer {

    void begin(WebSocketsServer &ws);

    void tick();
}
