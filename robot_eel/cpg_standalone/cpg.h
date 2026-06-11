#pragma once
#include <math.h>
#include "config.h"

inline float wrap_pi(float x) {
  while (x >  M_PI) x -= 2.0f * M_PI;
  while (x < -M_PI) x += 2.0f * M_PI;
  return x;
}

inline float degToRad(float deg) {
  return deg * ((float)M_PI / 180.0f);
}

inline float radToDeg(float rad) {
  return rad * (180.0f / (float)M_PI);
}

// Python: _target_delta()
// target_delta = 1.0 / (wavelength * body_length)
inline float getTargetDelta() {
  float lambdaInput = wavelength * bodyLength;
  if (lambdaInput < 1e-6f) lambdaInput = 1e-6f;
  return 1.0f / lambdaInput;
}

// Python: _phase_offsets()
// If phase_lags exists, it takes priority over wavelength.
inline void computePhaseOffsets(float offsets[bodyNum]) {
  offsets[0] = 0.0f;

  if (usePhaseLags) {
    float sum = 0.0f;
    for (int j = 1; j < bodyNum; j++) {
      sum += phaseLags[j - 1];
      offsets[j] = -sum;
    }
  } else {
    float targetDelta = getTargetDelta();
    for (int j = 1; j < bodyNum; j++) {
      offsets[j] = -((float)j) * targetDelta;
    }
  }
}

// Python: _mu_targets()
// amp_scales are converted to mu scales by squaring.
inline void computeMuTargets(float muTargets[bodyNum]) {
  for (int j = 0; j < bodyNum; j++) {
    if (useAmpScales) {
      float s = ampScales[j];
      if (s < 0.0f) s = 0.0f;
      muTargets[j] = muBase * s * s;
    } else {
      muTargets[j] = muBase;
    }
  }
}

// Python: reset()
// r[:] = 0.25
// theta[:] = phase_offsets
inline void resetCPG() {
  float phaseOffsets[bodyNum];
  computePhaseOffsets(phaseOffsets);

  for (int j = 0; j < bodyNum; j++) {
    cpg[j].r     = 0.25f;
    cpg[j].theta = phaseOffsets[j];
    cpg[j].alpha = alphaHopf;
    cpg[j].mu    = muBase;
  }
}

inline void initCPG() {
  resetCPG();
}

// Python: output()
// return ajoint * r * cos(theta) + joint_bias
inline float getCPGOutputRad(int j) {
  float biasRad = 0.0f;

  if (useJointBiasDeg) {
    biasRad = degToRad(jointBiasDeg[j]);
  }

  return ajointRad * cpg[j].r * cosf(cpg[j].theta) + biasRad;
}

inline float getCPGOutputDeg(int j) {
  return radToDeg(getCPGOutputRad(j));
}

// Python: step()
// This uses old_r / old_theta snapshots, so all joints update synchronously.
inline void updateCPGAll(float t, float dt) {
  float omega = 2.0f * (float)M_PI * frequency;

  float phaseOffsets[bodyNum];
  float muTargets[bodyNum];

  float oldR[bodyNum];
  float oldTheta[bodyNum];

  float dr[bodyNum];
  float dtheta[bodyNum];

  computePhaseOffsets(phaseOffsets);
  computeMuTargets(muTargets);

  for (int j = 0; j < bodyNum; j++) {
    oldR[j] = cpg[j].r;
    oldTheta[j] = cpg[j].theta;
  }

  for (int j = 0; j < bodyNum; j++) {
    dr[j] = alphaHopf * (muTargets[j] - oldR[j] * oldR[j]) * oldR[j];
    dtheta[j] = omega;
  }

  for (int j = 0; j < bodyNum; j++) {
    if (j - 1 >= 0) {
      float desiredL = phaseOffsets[j - 1] - phaseOffsets[j];
      float errL = wrap_pi((oldTheta[j - 1] - oldTheta[j]) - desiredL);
      dtheta[j] += kCouple * sinf(errL);
    }

    if (j + 1 < bodyNum) {
      float desiredR = phaseOffsets[j + 1] - phaseOffsets[j];
      float errR = wrap_pi((oldTheta[j + 1] - oldTheta[j]) - desiredR);
      dtheta[j] += kCouple * sinf(errR);
    }

    float thRef = omega * t + phaseOffsets[j];
    float eRef = wrap_pi(thRef - oldTheta[j]);
    dtheta[j] += kAnchor * sinf(eRef);
  }

  for (int j = 0; j < bodyNum; j++) {
    dtheta[j] += kFbPhase * fbPhase;
    dr[j]     += kFbAmp   * fbAmp;
  }

  for (int j = 0; j < bodyNum; j++) {
    cpg[j].r = oldR[j] + dr[j] * dt;
    if (cpg[j].r < 0.0f) cpg[j].r = 0.0f;

    cpg[j].theta = wrap_pi(oldTheta[j] + dtheta[j] * dt);

    cpg[j].alpha = alphaHopf;
    cpg[j].mu = muTargets[j];
  }
}
