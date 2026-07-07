#pragma once
#include <WebSocketsServer.h>

namespace CameraStreamWs {

    void begin(WebSocketsServer &ws);

    void sendFrame(WebSocketsServer &ws);
}
