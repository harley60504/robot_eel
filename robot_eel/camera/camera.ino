#include <Arduino.h>
#include <WebSocketsServer.h>
#include <WebServer.h>
#include <Preferences.h>

#include "camera_init.h"
#include "CameraStreamWs.h"
#include "HttpApi.h"
#include "CtrlUartBridge.h"
#include "ControlWsServer.h"
#include "ServoStatusWs.h"
#include "wifi_manager.h"

// WebSocket ports:
//   81 = camera stream
//   82 = Flutter/manual control
//   83 = Python backend control
//   84 = servo status telemetry
WebSocketsServer wsCam(81);
WebSocketsServer wsCtrlFlutter(82);
WebSocketsServer wsCtrlPython(83);
WebSocketsServer wsServoStatus(84);
WebServer server(80);

bool cameraReady = false;

void setup()
{
    Serial.begin(115200);

    // Wi-Fi AP/STA and HTTP API.
    startWifiApSta();
    HttpApi::begin();

    // Keep control/WebSocket alive even if the camera module fails to init.
    cameraReady = initCamera();

    // UART bridge to the control board.
    CtrlUartBridge::begin(
        Serial2,
        115200,
        UART_RX,
        UART_TX
    );

    // WebSocket servers.
    CameraStreamWs::begin(wsCam);
    ControlWsServer::begin(wsCtrlFlutter);
    ControlWsServer::begin(wsCtrlPython);
    ServoStatusWs::begin(wsServoStatus);

    wsCam.begin();
    wsCtrlFlutter.begin();
    wsCtrlPython.begin();
    wsServoStatus.begin();
    server.begin();

    Serial.println("System Ready.");
}

void loop()
{
    wsCam.loop();
    wsCtrlFlutter.loop();
    wsCtrlPython.loop();
    wsServoStatus.loop();
    server.handleClient();
    ControlWsServer::tick();

    if (cameraReady) {
        CameraStreamWs::sendFrame(wsCam);
    }
}
