import numpy as np
from .base import SwimmerBase, SwimParams

# Keep MuJoCo Legacy swimmer numerically aligned with
# Release/python_backend/angle_generator.py.
RL_VXHARD_AMP_SCALES = np.array([1.24, 1.08, 1.0, 1.05, 1.1, 1.2], dtype=np.float64)
TURN_SOFT_BIAS_RAD = np.array([0.08, 0.10, 0.12, 0.14, 0.16, 0.18], dtype=np.float64)


class LegacySwimmer(SwimmerBase):
    name = "Legacy"

    def __init__(self, steer_front_n: int = 6, steer_gain: float = 1.0):
        self.steer_front_n = int(steer_front_n)
        self.steer_gain = float(steer_gain)

    def compute_ctrl(self, t: float, num_joints: int, p: SwimParams) -> np.ndarray:
        # Match Release/python_backend/angle_generator.py:
        # theta_j = Ajoint * amp_scale[j] * cos(2*pi*f*t - sum(phase_lags[:j])) + joint_bias[j]
        steer = 0.0 if p.auto_mode else p.turn
        ctrl = np.zeros(num_joints, dtype=np.float64)

        phase_lags = np.full(max(0, num_joints - 1), float(p.step), dtype=np.float64)
        amp_scales = np.ones(num_joints, dtype=np.float64)
        amp_scales[:min(num_joints, len(RL_VXHARD_AMP_SCALES))] = RL_VXHARD_AMP_SCALES[:min(num_joints, len(RL_VXHARD_AMP_SCALES))]

        soft_bias = np.zeros(num_joints, dtype=np.float64)
        soft_bias[:min(num_joints, len(TURN_SOFT_BIAS_RAD))] = TURN_SOFT_BIAS_RAD[:min(num_joints, len(TURN_SOFT_BIAS_RAD))]

        # p.turn is interpreted as direction only:
        #   p.turn < 0 -> left_turn_rl  -> negative joint bias
        #   p.turn > 0 -> right_turn_rl -> positive joint bias
        if abs(steer) > 1e-12:
            joint_bias = np.sign(steer) * soft_bias * self.steer_gain
        else:
            joint_bias = np.zeros(num_joints, dtype=np.float64)

        theta = 2.0 * np.pi * p.freq * t
        for i in range(num_joints):
            phase_offset = -float(np.sum(phase_lags[:i]))

            if p.wave_type == "Standing":
                u = p.amp * amp_scales[i] * np.sin(theta) * np.sin(i * p.step) + joint_bias[i]
            else:
                u = p.amp * amp_scales[i] * np.cos(theta + phase_offset) + joint_bias[i]

            ctrl[i] = u

        return ctrl
