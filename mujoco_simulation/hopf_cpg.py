from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_AJOINT_DEG = 15.0


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def degrees_to_radians(value: float) -> float:
    return float(np.deg2rad(float(value)))


def amp_scales_to_mu_scales(amp_scales: tuple[float, ...] | np.ndarray | None) -> tuple[float, ...] | None:
    if amp_scales is None:
        return None
    values = np.asarray(amp_scales, dtype=np.float64)
    return tuple(float(value * value) for value in values)


@dataclass
class HopfCPGParams:
    frequency: float = 1.0
    wavelength: float = 1.5
    body_length: float = 1.0
    ajoint: float = degrees_to_radians(DEFAULT_AJOINT_DEG)
    alpha: float = 4.0
    mu: float = 1.0
    mu_scales: tuple[float, ...] | None = None
    k_couple: float = 0.35
    k_anchor: float = 0.10
    k_fb_phase: float = 0.8
    k_fb_amp: float = 0.25
    fb_phase: float = 0.0
    fb_amp: float = 0.0
    amp_scales: tuple[float, ...] | None = None
    phase_lags: tuple[float, ...] | None = None
    joint_bias: tuple[float, ...] | None = None


class HopfCPG:
    def __init__(self, num_joints: int, params: HopfCPGParams | None = None):
        self.num_joints = int(num_joints)
        self.params = params or HopfCPGParams()
        self.r = np.zeros(self.num_joints, dtype=np.float64)
        self.theta = np.zeros(self.num_joints, dtype=np.float64)
        self.reset()

    def reset(self):
        self.r[:] = 0.25
        self.theta[:] = self._phase_offsets(self.params, self.num_joints)

    def step(self, t: float, dt: float, params: HopfCPGParams | None = None) -> np.ndarray:
        if params is not None:
            self.params = params

        p = self.params
        omega = 2.0 * np.pi * p.frequency
        phase_offsets = self._phase_offsets(p, self.num_joints)

        old_r = self.r.copy()
        old_theta = self.theta.copy()
        mu_targets = self._mu_targets(p, self.num_joints)
        dr = p.alpha * (mu_targets - old_r * old_r) * old_r
        dtheta = np.full(self.num_joints, omega, dtype=np.float64)

        for j in range(self.num_joints):
            if j - 1 >= 0:
                desired_l = phase_offsets[j - 1] - phase_offsets[j]
                err_l = wrap_pi((old_theta[j - 1] - old_theta[j]) - desired_l)
                dtheta[j] += p.k_couple * np.sin(err_l)
            if j + 1 < self.num_joints:
                desired_r = phase_offsets[j + 1] - phase_offsets[j]
                err_r = wrap_pi((old_theta[j + 1] - old_theta[j]) - desired_r)
                dtheta[j] += p.k_couple * np.sin(err_r)

            th_ref = omega * t + phase_offsets[j]
            e_ref = wrap_pi(th_ref - old_theta[j])
            dtheta[j] += p.k_anchor * np.sin(e_ref)

        dtheta += p.k_fb_phase * p.fb_phase
        dr += p.k_fb_amp * p.fb_amp

        self.r = np.maximum(0.0, old_r + dr * dt)
        self.theta = wrap_pi(old_theta + dtheta * dt)
        return self.output()

    def output(self) -> np.ndarray:
        joint_bias = self._joint_bias(self.params, self.num_joints)
        return self.params.ajoint * self.r * np.cos(self.theta) + joint_bias

    @staticmethod
    def _target_delta(params: HopfCPGParams) -> float:
        lambda_input = max(1e-6, params.wavelength * params.body_length)
        return 1.0 / lambda_input

    @classmethod
    def _phase_offsets(cls, params: HopfCPGParams, num_joints: int) -> np.ndarray:
        if params.phase_lags is None:
            target_delta = cls._target_delta(params)
            return -np.arange(num_joints, dtype=np.float64) * target_delta

        lags = np.asarray(params.phase_lags, dtype=np.float64)
        if lags.size != num_joints - 1:
            raise ValueError(f"phase_lags must have {num_joints - 1} values, got {lags.size}")

        offsets = np.zeros(num_joints, dtype=np.float64)
        offsets[1:] = -np.cumsum(lags)
        return offsets

    @staticmethod
    def _mu_targets(params: HopfCPGParams, num_joints: int) -> np.ndarray:
        mu_scales_source = params.mu_scales
        if mu_scales_source is None and params.amp_scales is not None:
            mu_scales_source = amp_scales_to_mu_scales(params.amp_scales)

        if mu_scales_source is None:
            return np.full(num_joints, params.mu, dtype=np.float64)

        mu_scales = np.asarray(mu_scales_source, dtype=np.float64)
        if mu_scales.size != num_joints:
            raise ValueError(f"mu_scales must have {num_joints} values, got {mu_scales.size}")
        return params.mu * np.maximum(0.0, mu_scales)

    @staticmethod
    def _joint_bias(params: HopfCPGParams, num_joints: int) -> np.ndarray:
        if params.joint_bias is None:
            return np.zeros(num_joints, dtype=np.float64)

        joint_bias = np.asarray(params.joint_bias, dtype=np.float64)
        if joint_bias.size != num_joints:
            raise ValueError(f"joint_bias must have {num_joints} values, got {joint_bias.size}")
        return joint_bias
