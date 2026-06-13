from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians
from sim_config import EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


def direction_sign(value: str | float | int) -> float:
    if isinstance(value, str):
        text = value.lower().strip()
        if text in {"left", "ccw", "+", "positive"}:
            return 1.0
        if text in {"right", "cw", "-", "negative"}:
            return -1.0
        raise ValueError("turn_direction must be left or right")
    sign = float(value)
    return 1.0 if sign >= 0.0 else -1.0


@dataclass
class TurningConfig:
    xml_path: str = EEL_MODEL_XML
    episode_seconds: float = 10.0
    warmup_seconds: float = 2.0
    control_dt: float = 0.02
    fixed_frequency: float = 1.0
    fixed_wavelength: float = 1.6275
    fixed_ajoint: float = degrees_to_radians(DEFAULT_AJOINT_DEG)
    turn_direction: str = "left"
    target_yaw_rate: float = 0.45
    target_radius: float | None = None
    normalized_actions: bool = True

    # Turning PPO now learns only static joint bias.  The forward-wave shape is
    # fixed so PPO cannot destroy the coherent CPG traveling wave while learning
    # to bend the body for turning.
    fixed_amp_scales: tuple[float, ...] = (1.225, 1.075, 1.000, 1.075, 1.150, 1.225)
    fixed_phase_lags: tuple[float, ...] = (0.614439, 0.614439, 0.614439, 0.614439, 0.614439)
    joint_bias_low: float = -0.30
    joint_bias_high: float = 0.30

    reward_average_seconds: float = 0.6
    speed_weight: float = 0.60
    yaw_rate_weight: float = 1.20
    radius_weight: float = 0.5
    turn_direction_weight: float = 0.30
    lateral_speed_weight: float = 0.05
    energy_weight: float = 0.02
    smoothness_weight: float = 0.02
    bias_smoothness_weight: float = 0.02
    boundary_x_min: float = RESET_X_MIN
    boundary_x_max: float = RESET_X_MAX
    boundary_y: float = RESET_Y


class EelTurningRLEnv(gym.Env if gym is not None else object):
    """Train open-loop turning gaits by learning only static joint bias.

    Action layout:
        0:6  joint_bias in radians

    Amplitude and inter-joint phase lags are fixed by TurningConfig.  Positive
    target yaw rate is treated as left/CCW turning.  Negative target yaw rate is
    treated as right/CW turning.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: TurningConfig | None = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium first: python -m pip install gymnasium")

        self.cfg = config or TurningConfig()
        self._validate_fixed_gait()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = (0, 0, 0)

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.tail_ctrl_slice = slice(0, 6)
        self.tail_joint_names = [f"servo{i}" for i in range(1, 7)]
        self.tail_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.tail_joint_names
        ]
        self.tail_qpos_addr = np.array([self.model.jnt_qposadr[jid] for jid in self.tail_joint_ids])
        self.tail_dof_addr = np.array([self.model.jnt_dofadr[jid] for jid in self.tail_joint_ids])

        self.sim_steps_per_control = max(1, int(round(self.cfg.control_dt / self.model.opt.timestep)))
        self.max_steps = max(1, int(round(self.cfg.episode_seconds / self.cfg.control_dt)))
        self.warmup_steps = max(0, int(round(self.cfg.warmup_seconds / self.cfg.control_dt)))
        self.step_count = 0
        self.action_dim = 6
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.cpg = HopfCPG(num_joints=6)
        self.metric_window = deque(
            maxlen=max(1, int(round(self.cfg.reward_average_seconds / self.cfg.control_dt)))
        )

        if self.cfg.normalized_actions:
            self.action_space = spaces.Box(
                low=-np.ones(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            lows, highs = self._action_bounds()
            self.action_space = spaces.Box(lows.astype(np.float32), highs.astype(np.float32), dtype=np.float32)

        # q(6), qd(6), cpg features(4), root features(9), previous summary(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

    @property
    def signed_target_yaw_rate(self) -> float:
        return direction_sign(self.cfg.turn_direction) * abs(float(self.cfg.target_yaw_rate))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.metric_window.clear()
        self.cpg.reset()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        physical_action = self._physical_action(action)
        joint_bias = tuple(float(value) for value in physical_action[:6])
        amp_scales = tuple(float(value) for value in self.cfg.fixed_amp_scales)
        phase_lags = tuple(float(value) for value in self.cfg.fixed_phase_lags)

        params = HopfCPGParams(
            frequency=self.cfg.fixed_frequency,
            wavelength=self.cfg.fixed_wavelength,
            ajoint=self.cfg.fixed_ajoint,
            mu_scales=amp_scales_to_mu_scales(amp_scales),
            phase_lags=phase_lags,
            joint_bias=joint_bias,
            fb_phase=0.0,
            fb_amp=0.0,
        )

        for _ in range(self.sim_steps_per_control):
            targets = self.cpg.step(self.data.time, self.model.opt.timestep, params)
            self.data.ctrl[self.tail_ctrl_slice] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        base_pos = self.data.xpos[self.base_body_id]
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        yaw = float(self.data.qpos[2])
        yaw_rate = float(self.data.qvel[2])
        speed = float(np.hypot(vx, vy))
        self.metric_window.append((speed, vx, vy, yaw_rate))
        metrics = np.mean(np.asarray(self.metric_window, dtype=np.float64), axis=0)
        avg_speed, avg_vx, avg_vy, avg_yaw_rate = (float(value) for value in metrics)

        target_yaw_rate = self.signed_target_yaw_rate
        yaw_rate_error = abs(avg_yaw_rate - target_yaw_rate)
        signed_turn = np.sign(target_yaw_rate) * avg_yaw_rate
        wrong_direction_error = max(0.0, -signed_turn)
        energy = float(np.mean(np.square(self.data.ctrl[self.tail_ctrl_slice])))
        action_delta = float(np.linalg.norm(action - self.prev_action))
        prev_bias = self._physical_action(self.prev_action)
        bias_delta = float(np.linalg.norm(np.asarray(joint_bias, dtype=np.float64) - prev_bias))
        self.prev_action = action.copy()

        if abs(avg_yaw_rate) < 1e-6:
            turn_radius = np.inf
            radius_error = 1.0 if self.cfg.target_radius is not None else 0.0
        else:
            turn_radius = abs(avg_speed / avg_yaw_rate)
            if self.cfg.target_radius is None:
                radius_error = 0.0
            else:
                radius_error = abs(turn_radius - self.cfg.target_radius) / max(self.cfg.target_radius, 1e-6)
                radius_error = min(radius_error, 5.0)

        steady_state = self.step_count > self.warmup_steps
        reward_speed = self.cfg.speed_weight * avg_speed
        reward_yaw_rate = -self.cfg.yaw_rate_weight * yaw_rate_error
        reward_radius = -self.cfg.radius_weight * radius_error
        reward_direction = -self.cfg.turn_direction_weight * wrong_direction_error
        reward_lateral_speed = -self.cfg.lateral_speed_weight * abs(avg_vy)
        reward_energy = -self.cfg.energy_weight * energy
        reward_smooth = -self.cfg.smoothness_weight * action_delta
        reward_bias_smooth = -self.cfg.bias_smoothness_weight * bias_delta
        reward = 0.0
        if steady_state:
            reward = (
                # reward_speed
                + reward_yaw_rate
                #+ reward_radius
                # + reward_direction
                # + reward_lateral_speed
                # + reward_energy
                # + reward_smooth
                # + reward_bias_smooth
            )

        out_of_bounds = (
            float(base_pos[0]) < self.cfg.boundary_x_min
            or float(base_pos[0]) > self.cfg.boundary_x_max
            or abs(float(base_pos[1])) > self.cfg.boundary_y
        )
        terminated = bool(out_of_bounds)
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 1.0

        info = {
            "x": float(base_pos[0]),
            "y": float(base_pos[1]),
            "yaw": yaw,
            "velocity_x": avg_vx,
            "velocity_y": avg_vy,
            "speed": avg_speed,
            "yaw_rate": avg_yaw_rate,
            "target_yaw_rate": target_yaw_rate,
            "yaw_rate_error": yaw_rate_error,
            "turn_radius": float(turn_radius) if np.isfinite(turn_radius) else np.inf,
            "radius_error": radius_error,
            "energy_proxy": energy,
            "action_delta": action_delta,
            "bias_delta": bias_delta,
            "steady_state": steady_state,
            "physical_action": physical_action.astype(np.float32),
            "joint_bias": np.asarray(joint_bias, dtype=np.float32),
            "fixed_amp_scales": np.asarray(amp_scales, dtype=np.float32),
            "fixed_phase_lags": np.asarray(phase_lags, dtype=np.float32),
            "reward_speed": reward_speed,
            "reward_yaw_rate": reward_yaw_rate,
            "reward_radius": reward_radius,
            "reward_direction": reward_direction,
            "reward_lateral_speed": reward_lateral_speed,
            "reward_energy": reward_energy,
            "reward_smooth": reward_smooth,
            "reward_bias_smooth": reward_bias_smooth,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _validate_fixed_gait(self) -> None:
        if len(self.cfg.fixed_amp_scales) != 6:
            raise ValueError("fixed_amp_scales must have 6 values")
        if len(self.cfg.fixed_phase_lags) != 5:
            raise ValueError("fixed_phase_lags must have 5 values")
        if self.cfg.joint_bias_low > self.cfg.joint_bias_high:
            raise ValueError("joint_bias_low cannot be greater than joint_bias_high")

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bias_lows = np.full(6, float(self.cfg.joint_bias_low), dtype=np.float64)
        bias_highs = np.full(6, float(self.cfg.joint_bias_high), dtype=np.float64)
        return bias_lows, bias_highs

    def _physical_action(self, action: np.ndarray) -> np.ndarray:
        if not self.cfg.normalized_actions:
            return action.astype(np.float64)
        lows, highs = self._action_bounds()
        unit = 0.5 * (action + 1.0)
        return lows + unit * (highs - lows)

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self.tail_qpos_addr]
        qd = self.data.qvel[self.tail_dof_addr]
        base_pos = self.data.xpos[self.base_body_id]
        phase_features = np.array(
            [
                np.sin(self.cpg.theta[0]),
                np.cos(self.cpg.theta[0]),
                np.mean(self.cpg.r),
                np.std(self.cpg.r),
            ],
            dtype=np.float64,
        )
        root = np.array(
            [
                base_pos[0],
                base_pos[1],
                self.data.qpos[2],
                self.data.qvel[0],
                self.data.qvel[1],
                self.data.qvel[2],
                self.signed_target_yaw_rate,
                self.data.time / max(self.cfg.episode_seconds, 1e-6),
                float(self.step_count > self.warmup_steps),
            ],
            dtype=np.float64,
        )
        prev_summary = np.array(
            [
                float(np.mean(self.cfg.fixed_amp_scales)),
                float(np.mean(self.cfg.fixed_phase_lags)),
                float(np.mean(self.prev_action)),
            ],
            dtype=np.float64,
        )
        return np.concatenate((q, qd, phase_features, root, prev_summary)).astype(np.float32)


if __name__ == "__main__":
    env = EelTurningRLEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    info = {}
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print("turning RL smoke test OK")
    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)
    print("last info:", info)
    print("total reward:", round(total_reward, 3))
