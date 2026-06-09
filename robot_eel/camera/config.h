#ifndef CONFIG_H
#define CONFIG_H


// Camera defaults
#define CAMERA_FRAME_SIZE    FRAMESIZE_SVGA
#define CAMERA_JPEG_QUALITY  15
#define CAMERA_FB_COUNT      1

// AP 模式的 SSID / 密碼
#define AP_SSID   "Robot"
#define AP_PASS   "12345678"

// mDNS 名稱 → http://robot.local
#define HOSTNAME  "robot"

// Servos
#define bodyNum 6


#define  UART_RX     D10 
#define  UART_TX     D9
#endif
