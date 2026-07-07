from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from sim_config import DEFAULT_START_X, DEFAULT_START_Y, EEL_MODEL_XML, RESET_X_MAX, RESET_X_MIN, RESET_Y
from hopf_cpg import HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


DEFAULT_FREE_SWIM_AJOINT_DEG = 20.0


@dataclass
class FreeSwimConfig:
    xml_path: str = EEL_MODEL_XML
    episode_seconds: float = 10.0
    warmup_seconds: float = 2.0
    control_dt: float = 0.02
    fixed_frequency: float = 1.0
    fixed_wavelength: float = 1.6275
    fixed_ajoint: float = degrees_to_radians(DEFAULT_FREE_SWIM_AJOINT_DEG)
    fixed_amp_scales: tuple[float, ...] = (1.225, 1.075, 1.000, 1.075, 1.150, 1.225)
    start_x: float = DEFAULT_START_X
    start_y: float = DEFAULT_START_Y
    normalized_actions: bool = True
    frequency_low: float = 1.0
    frequency_high: float = 1.2
    phase_lag_low: float = 0.5
    phase_lag_high: float = 0.8
    reward_average_seconds: float = 1.0
    target_speed: float = 0.17
    speed_error_weight: float = 100.0
    energy_weight: float = 0.08
    boundary_x_min: float = RESET_X_MIN
    boundary_x_max: float = RESET_X_MAX
    boundary_y: float = RESET_Y


class EelFreeSwimRLEnv(gym.Env if gym is not None else object):
    """Free-swim PPO environment.

    Action layout:
        0    frequency
        1:6  phase_lags for the 5 inter-joint phase gaps

    Joint bias is fixed at zero.  Amplitude is fixed through cfg.fixed_ajoint
    and cfg.fixed_amp_scales.  The reward targets a desired forward velocity
    and penalizes squared frequency as a simple energy proxy.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: FreeSwimConfig | None = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium first: python -m pip install gymnasium")

        self.cfg = config or FreeSwimConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = (0, 0, -9.81)

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
        base_xml_pos = self.model.body_pos[self.base_body_id]
        self.data.qpos[0] = float(self.cfg.start_x) - float(base_xml_pos[0])
        self.data.qpos[1] = float(self.cfg.start_y) - float(base_xml_pos[1])
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
        frequency = float(physical_action[0])
        phase_lags = tuple(float(value) for value in physical_action[1:6])
        amp_scales = self.cfg.fixed_amp_scales
        joint_bias = (0.0,) * 6

        params = HopfCPGParams(
            frequency=frequency,
            wavelength=self.cfg.fixed_wavelength,
            ajoint=self.cfg.fixed_ajoint,
            mu_scales=amp_scales_to_mu_scales(amp_scales),
            phase_lags=phase_lags,
            joint_bias=joint_bias,
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

        control_energy = float(np.mean(np.square(self.data.ctrl[self.tail_ctrl_slice])))
        frequency_energy = frequency * frequency
        action_delta = float(np.linalg.norm(action - self.prev_action))
        self.prev_action = action.copy()

        steady_state = self.step_count > self.warmup_steps
        speed_error = avg_vx - self.cfg.target_speed
        speed_error_weight = max(float(self.cfg.speed_error_weight), 0.0)
        reward_forward = float(-speed_error_weight * (speed_error ** 2))
        reward_energy = -self.cfg.energy_weight * frequency_energy
        reward = 0.0
        if steady_state:
            reward = (
                reward_forward
                + reward_energy
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
            "energy_proxy": frequency_energy,
            "control_energy_proxy": control_energy,
            "frequency_energy_proxy": frequency_energy,
            "action_delta": action_delta,
            "steady_state": steady_state,
            "physical_action": physical_action.astype(np.float32),
            "frequency": frequency,
            "amp_scales": np.asarray(amp_scales, dtype=np.float32),
            "phase_lags": np.asarray(phase_lags, dtype=np.float32),
            "joint_bias": np.asarray(joint_bias, dtype=np.float32),
            "reward_forward": reward_forward,
            "reward_energy": reward_energy,
            "target_speed": float(self.cfg.target_speed),
            "speed_error": float(speed_error),
            "speed_error_weight": float(speed_error_weight),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.cfg.fixed_amp_scales) != 6:
            raise ValueError("fixed_amp_scales must have 6 values")
        if self.cfg.frequency_low > self.cfg.frequency_high:
            raise ValueError("frequency_low cannot be greater than frequency_high")
        if self.cfg.phase_lag_low > self.cfg.phase_lag_high:
            raise ValueError("phase_lag_low cannot be greater than phase_lag_high")
        phase_lows = np.full(5, float(self.cfg.phase_lag_low), dtype=np.float64)
        phase_highs = np.full(5, float(self.cfg.phase_lag_high), dtype=np.float64)
        return (
            np.concatenate(([float(self.cfg.frequency_low)], phase_lows)),
            np.concatenate(([float(self.cfg.frequency_high)], phase_highs)),
        )

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
    print("action shape:", env.action_space.shape)
    print("last info:", info)
    print("total reward:", round(total_reward, 3))
