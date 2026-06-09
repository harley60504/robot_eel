#include <Arduino.h>
#include <WebSocketsServer.h>
#include <WebServer.h>
#include <Preferences.h>

#include "camera_init.h"
#include "cam_stream.h"
#include "wifi_http.h"
#include "CtrlUartBridge.h"
#include "CtrlWsServer.h"
#include "servo_csv_log.h"
#include "wifi_manager.h"

// WebSocket ports:
//   81 = camera stream
//   82 = control
WebSocketsServer wsCam(81);
WebSocketsServer wsCtrl(82);
WebServer server(80);

bool cameraReady = false;

void setup()
{
    Serial.begin(115200);

    // Wi-Fi AP/STA and HTTP API.
    startWifiApSta();
    initServoCsvLog();
    setupWifiHttpApi();

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
    initStreamWS(wsCam);
    CtrlWsServer::begin(wsCtrl);

    wsCam.begin();
    wsCtrl.begin();
    server.begin();

    Serial.println("System Ready.");
}

void loop()
{
    wsCam.loop();
    wsCtrl.loop();
    server.handleClient();
    CtrlWsServer::tick();

    if (cameraReady) {
        sendCameraFrame(wsCam);
    }
}
