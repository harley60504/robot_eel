#include <Arduino.h>
#include "config.h"
#include "cpg.h"

// ==========================
// Control Parameters
// Same naming and units as robot_eel/control/control.ino.
// ==========================
// RL exported params: mujoco_simulation/gaits/rl_straight.json
float Ajoint       = 15.0f;  // deg
float frequency    = 1.0f;
float lambda       = 1.6275f;
float L            = 1.0f;
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
float jointBiasDeg[bodyNum] = {0, 0, 0, 0, 0, 0};

HopfOscillator cpg[bodyNum];

// ==========================
// Standalone CPG test timing
// ==========================
const unsigned long intervalMs = 20;  // 50 Hz
unsigned long lastUpdateMs = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  initCPG();
  lastUpdateMs = millis();

  // Arduino Serial Plotter labels.
  Serial.println("J0,J1,J2,J3,J4,J5");
}

void loop() {
  unsigned long now = millis();

  if (now - lastUpdateMs >= intervalMs) {
    float dt = (now - lastUpdateMs) / 1000.0f;
    lastUpdateMs = now;

    // No sensor feedback in standalone test.
    updateCPGAll(dt);

    for (int j = 0; j < bodyNum; j++) {
      float outDeg = getCPGOutput(j);

      Serial.print(outDeg, 4);

      if (j < bodyNum - 1) {
        Serial.print(",");
      } else {
        Serial.println();
      }
    }
  }
}
