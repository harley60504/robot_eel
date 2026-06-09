#include <esp_camera.h>
#include "cam_stream.h"

void initStreamWS(WebSocketsServer &ws)
{
    ws.onEvent([](uint8_t num, WStype_t type, uint8_t *payload, size_t len){
        switch(type)
        {
            case WStype_CONNECTED:
                Serial.printf("[WS] Client %u connected\n", num);
                break;

            case WStype_DISCONNECTED:
                Serial.printf("[WS] Client %u disconnected\n", num);
                break;

            default:
                break;
        }
    });

    Serial.println("WebSocket stream enabled at ws://<IP>:81");
}


void sendCameraFrame(WebSocketsServer &ws)
{
    if (!ws.connectedClients()) return;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return;

    ws.broadcastBIN(fb->buf, fb->len);

    esp_camera_fb_return(fb);
}
