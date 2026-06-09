from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from sim_config import EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y
from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


@dataclass
class FreeSwimConfig:
    xml_path: str = EEL_MODEL_XML
    episode_seconds: float = 8.0
    warmup_seconds: float = 2.0
    control_dt: float = 0.02
    fixed_frequency: float = 1.0
    fixed_wavelength: float = 1.6275
    fixed_ajoint: float = degrees_to_radians(DEFAULT_AJOINT_DEG)
    normalized_actions: bool = True
    amp_scale_lows: tuple[float, ...] = (1.10, 0.95, 0.90, 0.95, 1.00, 1.05)
    amp_scale_highs: tuple[float, ...] = (1.35, 1.20, 1.10, 1.20, 1.30, 1.40)
    phase_lag_lows: tuple[float, ...] = (0.57, 0.57, 0.57, 0.57, 0.57)
    phase_lag_highs: tuple[float, ...] = (0.66, 0.66, 0.66, 0.66, 0.66)
    reward_average_seconds: float = 0.5
    lateral_velocity_weight: float = 0.2
    lateral_position_weight: float = 0.05
    yaw_weight: float = 0.05
    yaw_rate_weight: float = 0.02
    energy_weight: float = 0.02
    smoothness_weight: float = 0.02
    boundary_x_min: float = RESET_X_MIN
    boundary_x_max: float = RESET_X_MAX
    boundary_y: float = RESET_Y


class EelFreeSwimRLEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, config: FreeSwimConfig | None = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium first: python -m pip install gymnasium")

        self.cfg = config or FreeSwimConfig()
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
        self.action_dim = 11
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.cpg = HopfCPG(num_joints=6)
        self.velocity_window = deque(
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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.velocity_window.clear()
        self.cpg.reset()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        physical_action = self._physical_action(action)
        amp_scales = tuple(float(value) for value in physical_action[:6])
        phase_lags = tuple(float(value) for value in physical_action[6:])

        params = HopfCPGParams(
            frequency=self.cfg.fixed_frequency,
            wavelength=self.cfg.fixed_wavelength,
            ajoint=self.cfg.fixed_ajoint,
            mu_scales=amp_scales_to_mu_scales(amp_scales),
            phase_lags=phase_lags,
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
        self.velocity_window.append((vx, vy))
        avg_velocity = np.mean(np.asarray(self.velocity_window, dtype=np.float64), axis=0)
        avg_vx, avg_vy = float(avg_velocity[0]), float(avg_velocity[1])

        energy = float(np.mean(np.square(self.data.ctrl[self.tail_ctrl_slice])))
        action_delta = float(np.linalg.norm(action - self.prev_action))
        self.prev_action = action.copy()

        steady_state = self.step_count > self.warmup_steps
        reward_forward = avg_vx
        reward_lateral_velocity = -self.cfg.lateral_velocity_weight * abs(avg_vy)
        reward_lateral_position = -self.cfg.lateral_position_weight * abs(float(base_pos[1]))
        reward_yaw = -self.cfg.yaw_weight * abs(yaw)
        reward_yaw_rate = -self.cfg.yaw_rate_weight * abs(yaw_rate)
        reward_energy = -self.cfg.energy_weight * energy
        reward_smooth = -self.cfg.smoothness_weight * action_delta
        reward = 0.0
        if steady_state:
            reward = (
                reward_forward
                + reward_lateral_velocity
                + reward_lateral_position
                + reward_yaw
                + reward_yaw_rate
                + reward_energy
                + reward_smooth
            )

        out_of_bounds = float(base_pos[0]) < self.cfg.boundary_x_min or float(base_pos[0]) > self.cfg.boundary_x_max or abs(float(base_pos[1])) > self.cfg.boundary_y
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
            "energy_proxy": energy,
            "action_delta": action_delta,
            "steady_state": steady_state,
            "physical_action": physical_action.astype(np.float32),
            "reward_forward": reward_forward,
            "reward_lateral_velocity": reward_lateral_velocity,
            "reward_lateral_position": reward_lateral_position,
            "reward_yaw": reward_yaw,
            "reward_yaw_rate": reward_yaw_rate,
            "reward_energy": reward_energy,
            "reward_smooth": reward_smooth,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        amp_lows = np.asarray(self.cfg.amp_scale_lows, dtype=np.float64)
        amp_highs = np.asarray(self.cfg.amp_scale_highs, dtype=np.float64)
        phase_lows = np.asarray(self.cfg.phase_lag_lows, dtype=np.float64)
        phase_highs = np.asarray(self.cfg.phase_lag_highs, dtype=np.float64)
        if amp_lows.size != 6 or amp_highs.size != 6:
            raise ValueError("amp bounds must have 6 values")
        if phase_lows.size != 5 or phase_highs.size != 5:
            raise ValueError("phase bounds must have 5 values")
        return np.concatenate((amp_lows, phase_lows)), np.concatenate((amp_highs, phase_highs))

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
                self.data.time / max(self.cfg.episode_seconds, 1e-6),
                float(self.step_count > self.warmup_steps),
            ],
            dtype=np.float64,
        )
        return np.concatenate((q, qd, phase_features, root)).astype(np.float32)


if __name__ == "__main__":
    env = EelFreeSwimRLEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    info = {}
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print("free-swim smoke test OK")
    print("obs shape:", obs.shape)
    print("last info:", info)
    print("total reward:", round(total_reward, 3))
