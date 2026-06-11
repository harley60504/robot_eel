#pragma once
#include <math.h>
#include "config.h"

inline float wrap_pi(float x) {
  while (x >  M_PI) x -= 2.0f * M_PI;
  while (x < -M_PI) x += 2.0f * M_PI;
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

inline void computePhaseOffsets(float offsets[bodyNum]) {
  offsets[0] = 0.0f;
  for (int j = 1; j < bodyNum; j++) {
    offsets[j] = offsets[j - 1] - phaseLags[j - 1];
  }
}

inline void initCPG() {
  float phaseOffsets[bodyNum];
  computePhaseOffsets(phaseOffsets);

  // Match Python HopfCPG.reset(): r starts at 0.25 and theta starts at phase offsets.
  for (int j = 0; j < bodyNum; j++) {
    cpg[j].r = 0.25f;
    cpg[j].theta = phaseOffsets[j];
    cpg[j].alpha = 4.0f;
    cpg[j].mu = ampScales[j] * ampScales[j];
  }
}

inline float getCPGOutput(int j) {
  return Ajoint * cpg[j].r * cosf(cpg[j].theta) + jointBiasDeg[j];
}

inline float getLambdaInput() { return fmaxf(lambda * L, 1e-6f); }
inline float getTargetDelta() { return 1.0f / getLambdaInput(); }

inline void updateCPGAll(float t, float dt, float fb_phase, float fb_amp) {
  const float K_couple   = 0.35f;
  const float K_anchor   = 0.10f;
  const float k_fb_phase = 0.8f;
  const float k_fb_amp   = 0.25f;

  float omega = 2.0f * M_PI * frequency;

  float phaseOffsets[bodyNum];
  float oldR[bodyNum];
  float oldTheta[bodyNum];
  float dr[bodyNum];
  float dtheta[bodyNum];
  float muTargets[bodyNum];

  computePhaseOffsets(phaseOffsets);

  // Python equivalent:
  // old_r = self.r.copy()
  // old_theta = self.theta.copy()
  for (int j = 0; j < bodyNum; j++) {
    oldR[j] = cpg[j].r;
    oldTheta[j] = cpg[j].theta;
    muTargets[j] = ampScales[j] * ampScales[j];
  }

  for (int j = 0; j < bodyNum; j++) {
    dr[j] = cpg[j].alpha * (muTargets[j] - oldR[j] * oldR[j]) * oldR[j];
    dtheta[j] = omega;
  }

  for (int j = 0; j < bodyNum; j++) {
    if (j - 1 >= 0) {
      float desiredL = phaseOffsets[j - 1] - phaseOffsets[j];
      float errL = wrap_pi((oldTheta[j - 1] - oldTheta[j]) - desiredL);
      dtheta[j] += K_couple * sinf(errL);
    }

    if (j + 1 < bodyNum) {
      float desiredR = phaseOffsets[j + 1] - phaseOffsets[j];
      float errR = wrap_pi((oldTheta[j + 1] - oldTheta[j]) - desiredR);
      dtheta[j] += K_couple * sinf(errR);
    }

    float th_ref = omega * t + phaseOffsets[j];
    float e_ref = wrap_pi(th_ref - oldTheta[j]);
    dtheta[j] += K_anchor * sinf(e_ref);
  }

  for (int j = 0; j < bodyNum; j++) {
    dtheta[j] += k_fb_phase * fb_phase;
    dr[j]     += k_fb_amp   * fb_amp;
  }

  for (int j = 0; j < bodyNum; j++) {
    cpg[j].r = fmaxf(0.0f, oldR[j] + dr[j] * dt);
    cpg[j].theta = wrap_pi(oldTheta[j] + dtheta[j] * dt);
    cpg[j].mu = muTargets[j];
  }
}

// Backward-compatible single-joint updater. Prefer updateCPGAll() for CPG mode.
inline void updateCPG(float t, float dt, int j, float fb_phase, float fb_amp) {
  (void)j;
  updateCPGAll(t, dt, fb_phase, fb_amp);
}
