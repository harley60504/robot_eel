#include <Arduino.h>
#include "config.h"
#include "cpg.h"

// ==========================
// Python HopfCPGParams
// ==========================
float frequency  = 1.0f;
float wavelength = 1.6275f;
float bodyLength = 1.0f;

// Python ajoint is radians.
float ajointRad = 15.0f * ((float)M_PI / 180.0f);

float alphaHopf = 4.0f;
float muBase    = 1.0f;

float kCouple  = 0.35f;
float kAnchor  = 0.10f;
float kFbPhase = 0.8f;
float kFbAmp   = 0.25f;

float fbPhase = 0.0f;
float fbAmp   = 0.0f;

// ==========================
// Optional arrays
// ==========================
bool useAmpScales    = true;
bool usePhaseLags    = true;
bool useJointBiasDeg = true;

// RL exported params: rl_straight
float ampScales[bodyNum] = {
  1.1f,
  0.95f,
  0.9f,
  1.071703f,
  1.161346f,
  1.273484f
};

float phaseLags[bodyNum - 1] = {
  0.614385f,
  0.622822f,
  0.615807f,
  0.615359f,
  0.608868f
};

// Stored in degrees and converted to radians by getCPGOutputRad().
float jointBiasDeg[bodyNum] = {
  0.0f,
  0.0f,
  0.0f,
  0.0f,
  0.0f,
  0.0f
};

HopfOscillator cpg[bodyNum];

// ==========================
// Timing
// ==========================
const unsigned long intervalMs = 20;  // 50 Hz
unsigned long lastUpdateMs = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  resetCPG();
  lastUpdateMs = millis();

  // Arduino Serial Plotter labels.
  Serial.println("J0,J1,J2,J3,J4,J5");
}

void loop() {
  unsigned long now = millis();

  if (now - lastUpdateMs >= intervalMs) {
    float dt = (now - lastUpdateMs) / 1000.0f;
    lastUpdateMs = now;

    float t = now / 1000.0f;

    updateCPGAll(t, dt);

    // Print degrees for easier Serial Plotter inspection.
    for (int j = 0; j < bodyNum; j++) {
      float outDeg = getCPGOutputDeg(j);

      Serial.print(outDeg, 4);

      if (j < bodyNum - 1) {
        Serial.print(",");
      } else {
        Serial.println();
      }
    }
  }
}
