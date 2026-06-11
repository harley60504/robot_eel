#pragma once
#include <math.h>
#include "config.h"
#include "utils.h"

inline float wrap_pi(float x) {
  while (x >  M_PI) x -= 2*M_PI;
  while (x < -M_PI) x += 2*M_PI;
  return x;
}

inline float getPhaseOffset(int j) {
  float offset = 0.0f;
  for (int i = 0; i < j && i < bodyNum - 1; i++) {
    offset -= phaseLags[i];
  }
  return offset;
}

inline float getNeighborDesiredDelta(int leftJoint, int rightJoint) {
  return getPhaseOffset(leftJoint) - getPhaseOffset(rightJoint);
}

inline void initCPG() {
  for (int j = 0; j < bodyNum; j++) {
    cpg[j].r = fmaxf(0.0f, ampScales[j]);
    cpg[j].theta = getPhaseOffset(j);
    cpg[j].alpha = 4.0f;
    cpg[j].mu = ampScales[j] * ampScales[j];
  }
}

inline float getCPGOutput(int j) {
  return Ajoint * cpg[j].r * cosf(cpg[j].theta) + jointBiasDeg[j];
}


inline float getLambdaInput() { return fmaxf(lambda * L, 1e-6f); }
inline float getTargetDelta() { return 1.0f / getLambdaInput(); }

inline void updateCPG(float t, float dt, int j, float fb_phase, float fb_amp) {
  HopfOscillator &o = cpg[j];
  float omega = 2.0f * M_PI * frequency;
  o.mu = ampScales[j] * ampScales[j];
  float dr = o.alpha * (o.mu - o.r * o.r) * o.r;
  float dtheta = omega;

  const float K_couple   = 0.35f;
  const float K_anchor   = 0.10f;
  const float k_fb_phase = 0.8f;
  const float k_fb_amp   = 0.25f;



  float th_ref = omega * t + getPhaseOffset(j);
  float e_ref = wrap_pi(th_ref - o.theta);
  dtheta += K_anchor * sinf(e_ref);

  dtheta += k_fb_phase * fb_phase;
  dr     += k_fb_amp   * fb_amp;

  o.r      = fmaxf(0.0f, o.r + dr * dt);
  o.theta  = wrap_pi(o.theta + dtheta * dt);
}
