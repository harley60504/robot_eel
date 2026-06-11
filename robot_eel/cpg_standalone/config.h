#pragma once
#include <Arduino.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define bodyNum 6

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
// CPG control parameters
// Same naming and units as robot_eel/control/config.h.
// =====================================================
extern float Ajoint;                 // degree
extern float frequency;              // Hz
extern float lambda;
extern float L;
extern float ampScales[bodyNum];
extern float phaseLags[bodyNum - 1]; // rad
extern float jointBiasDeg[bodyNum];  // degree

// =====================================================
// CPG state
// =====================================================
extern HopfOscillator cpg[bodyNum];
