#pragma once
#include <Arduino.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define bodyNum 6

struct HopfOscillator {
  float r;
  float theta;
  float alpha;
  float mu;
};

// ==========================
// Python HopfCPGParams mapping
// ==========================
extern float frequency;
extern float wavelength;
extern float bodyLength;

// Python uses radians internally for ajoint and output.
extern float ajointRad;
extern float alphaHopf;
extern float muBase;

extern float kCouple;
extern float kAnchor;
extern float kFbPhase;
extern float kFbAmp;

extern float fbPhase;
extern float fbAmp;

// ==========================
// Optional parameter arrays
// ==========================
extern bool useAmpScales;
extern bool usePhaseLags;
extern bool useJointBiasDeg;

extern float ampScales[bodyNum];
extern float phaseLags[bodyNum - 1];
extern float jointBiasDeg[bodyNum];

// ==========================
// CPG state
// ==========================
extern HopfOscillator cpg[bodyNum];
