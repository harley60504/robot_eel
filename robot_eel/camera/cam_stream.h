#pragma once
#include <WebSocketsServer.h>

void initStreamWS(WebSocketsServer &ws);
void sendCameraFrame(WebSocketsServer &ws);
