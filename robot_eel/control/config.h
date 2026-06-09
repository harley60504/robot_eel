#pragma once
#include <Arduino.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// =====================================================
// LX-224 Servo
// =====================================================
#define SERVO_TX_PIN 43
#define SERVO_RX_PIN 44
#define HEADER 0x55
#define CMD_MOVE_TIME_WRITE 0x01
#define CMD_READ_POS        0x1C
#define CMD_ID_WRITE        0x13
#define BROADCAST_ID 0xFE
#define bodyNum 6
// =====================================================
// Control modes (keep in sync with Flutter mode buttons)
// =====================================================
#define MODE_SIN         0
#define MODE_CPG         1
#define MODE_OFFSET      2
#define MODE_UART_ANGLE  3
// =====================================================
// CPG oscillator
// =====================================================
struct HopfOscillator {
  float r;
  float theta;
  float alpha;
  float mu;
};
// =====================================================
// Servo / Control parameters
// =====================================================
extern float servoDefaultAngles[bodyNum];
extern float angleDeg[bodyNum];
extern float Ajoint;
extern float frequency;
extern float lambda;
extern float L;
extern float ampScales[bodyNum];
extern float phaseLags[bodyNum - 1];
extern float jointBiasDeg[bodyNum];
extern bool  isPaused;
extern int   controlMode;
extern bool  useFeedback;
extern float feedbackGain;
// =====================================================
// CPG state
// =====================================================
extern HopfOscillator cpg[bodyNum];
// ==========================
//  UART Pins
// ==========================
#define CAMERA_RX_PIN   D9
#define CAMERA_TX_PIN   D10
