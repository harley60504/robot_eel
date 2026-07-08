from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import mujoco
import numpy as np

from hopf_cpg import DEFAULT_AJOINT_DEG, HopfCPG, HopfCPGParams, amp_scales_to_mu_scales, degrees_to_radians

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Allows importing this module before installing RL deps.
    gym = None
    spaces = None


@dataclass
class TetheredConfig:
    xml_path: str = "eel_tethered.xml"
    episode_seconds: float = 4.0
    warmup_seconds: float = 1.0
    control_dt: float = 0.02
    frequency_min: float = 0.8
    frequency_max: float = 1.3
    fixed_frequency: float | None = None
    wavelength_min: float = 1.2
    wavelength_max: float = 1.8
    fixed_wavelength: float = 1.5
    per_joint_action: bool = False
    amp_scale_min: float = 0.4
    amp_scale_max: float = 1.2
    phase_lag_min: float = 0.45
    phase_lag_max: float = 0.95
    amp_scale_lows: tuple[float, ...] | None = None
    amp_scale_highs: tuple[float, ...] | None = None
    phase_lag_lows: tuple[float, ...] | None = None
    phase_lag_highs: tuple[float, ...] | None = None
    normalized_actions: bool = True
    fixed_ajoint: float = degrees_to_radians(DEFAULT_AJOINT_DEG)
    fixed_fb_phase: float = 0.0
    fixed_fb_amp: float = 0.0
    reward_mode: str = "maximize_fx"
    target_force: float = 4.0
    target_metric: str = "fx"
    target_force_x: float = 0.4
    forward_force_weight: float = 0.0
    lateral_force_weight: float = 0.02
    energy_weight: float = 0.01
    frequency_weight: float = 0.0
    action_smoothness_weight: float = 0.02
    reward_average_seconds: float = 1.0
    target_error_weight: float = 1.0


class EelTetheredRLEnv(gym.Env if gym is not None else object):
    """
    Paper-style tethered thrust environment.

    The robot is held near the origin by root position actuators. The actuator
    effort required to hold root_x/root_y is used as a force-sensor proxy. The
    RL action modulates a Hopf CPG instead of commanding raw motors:

      action = [frequency, wavelength]

    If per_joint_action is enabled:

      action = [frequency, amp_scale_1..6, phase_lag_1..5]

    If fixed_frequency is set with per_joint_action:

      action = [amp_scale_1..6, phase_lag_1..5]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: TetheredConfig | None = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium first: python -m pip install gymnasium")

        self.cfg = config or TetheredConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = (0, 0, 0)

        self.tail_ctrl_slice = slice(3, 9)
        self.tail_joint_names = [f"servo{i}" for i in range(1, 7)]
        self.tail_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.tail_joint_names
        ]
        self.tail_qpos_addr = np.array([self.model.jnt_qposadr[jid] for jid in self.tail_joint_ids])
        self.tail_dof_addr = np.array([self.model.jnt_dofadr[jid] for jid in self.tail_joint_ids])

        self.root_dof_x = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
        ]
        self.root_dof_y = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_y")
        ]

        self.sim_steps_per_control = max(1, int(round(self.cfg.control_dt / self.model.opt.timestep)))
        self.max_steps = max(1, int(round(self.cfg.episode_seconds / self.cfg.control_dt)))
        self.warmup_steps = max(0, int(round(self.cfg.warmup_seconds / self.cfg.control_dt)))
        self.step_count = 0
        self.last_force = np.zeros(2, dtype=np.float32)
        self.action_dim = self._action_dim()
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.cpg = HopfCPG(num_joints=6)
        self.force_window = deque(
            maxlen=max(1, int(round(self.cfg.reward_average_seconds / self.cfg.control_dt)))
        )

        if self.cfg.normalized_actions:
            self.action_space = spaces.Box(
                low=-np.ones(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32),
                dtype=np.float32,
            )
        elif self.cfg.per_joint_action:
            lows, highs = self._per_joint_action_bounds()
            self.action_space = spaces.Box(
                low=lows.astype(np.float32),
                high=highs.astype(np.float32),
                dtype=np.float32,
            )
        else:
            if self.cfg.fixed_frequency is None:
                low = np.array([self.cfg.frequency_min, self.cfg.wavelength_min], dtype=np.float32)
                high = np.array([self.cfg.frequency_max, self.cfg.wavelength_max], dtype=np.float32)
            else:
                low = np.array([self.cfg.wavelength_min], dtype=np.float32)
                high = np.array([self.cfg.wavelength_max], dtype=np.float32)
            self.action_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.step_count = 0
        self.last_force[:] = 0.0
        self.prev_action[:] = 0.0
        self.force_window.clear()
        self.cpg.reset()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def _action_dim(self) -> int:
        if self.cfg.per_joint_action:
            return 11 if self.cfg.fixed_frequency is not None else 12
        return 1 if self.cfg.fixed_frequency is not None else 2

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        physical_action = self._physical_action(action)
        if self.cfg.per_joint_action:
            if self.cfg.fixed_frequency is None:
                frequency = float(physical_action[0])
                shape_action = physical_action[1:]
            else:
                frequency = float(self.cfg.fixed_frequency)
                shape_action = physical_action
            wavelength = float(self.cfg.fixed_wavelength)
            amp_scales = tuple(float(value) for value in shape_action[:6])
            phase_lags = tuple(float(value) for value in shape_action[6:11])
        else:
            if self.cfg.fixed_frequency is None:
                frequency, wavelength = physical_action
            else:
                frequency = float(self.cfg.fixed_frequency)
                wavelength = float(physical_action[0])
            amp_scales = None
            phase_lags = None

        params = HopfCPGParams(
            frequency=float(frequency),
            wavelength=float(wavelength),
            ajoint=float(self.cfg.fixed_ajoint),
            mu_scales=amp_scales_to_mu_scales(amp_scales),
            phase_lags=phase_lags,
            fb_phase=float(self.cfg.fixed_fb_phase),
            fb_amp=float(self.cfg.fixed_fb_amp),
        )

        self.data.ctrl[0:3] = 0.0

        forces = []
        for _ in range(self.sim_steps_per_control):
            targets = self.cpg.step(self.data.time, self.model.opt.timestep, params)
            self.data.ctrl[self.tail_ctrl_slice] = np.clip(targets, -1.2, 1.2)
            mujoco.mj_step(self.model, self.data)
            forces.append(self._holding_force())

        self.step_count += 1

        force = np.mean(forces, axis=0)
        self.last_force = force.astype(np.float32)
        self.force_window.append(force)
        avg_force = np.mean(np.asarray(self.force_window, dtype=np.float64), axis=0)
        fx, fy = float(avg_force[0]), float(avg_force[1])
        energy = float(np.mean(np.square(self.data.ctrl[self.tail_ctrl_slice])))
        action_delta = float(np.linalg.norm(action - self.prev_action))
        self.prev_action = action.copy()

        steady_state = self.step_count > self.warmup_steps
        reward_fx = fx
        force_metric = self._force_metric(fx, fy)
        if self.cfg.reward_mode == "target_force":
            reward_forward = (
                -self.cfg.target_error_weight * abs(force_metric - self.cfg.target_force)
                + self.cfg.forward_force_weight * fx
            )
        elif self.cfg.reward_mode == "maximize_fx":
            reward_forward = self.cfg.target_error_weight * fx
        else:
            raise ValueError(f"unknown reward_mode: {self.cfg.reward_mode}")
        reward_lateral = -self.cfg.lateral_force_weight * abs(fy)
        reward_energy = -self.cfg.energy_weight * energy
        reward_frequency = -self.cfg.frequency_weight * float(frequency * frequency)
        reward_smooth = -self.cfg.action_smoothness_weight * action_delta
        reward = 0.0
        if steady_state:
            reward = (
                reward_forward
                + reward_lateral
                + reward_energy
                + reward_frequency
                + reward_smooth
            )

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            "force_x": fx,
            "force_y": fy,
            "force_metric": force_metric,
            "target_force": self.cfg.target_force,
            "target_metric": self.cfg.target_metric,
            "target_force_x": self.cfg.target_force_x,
            "energy_proxy": energy,
            "action_delta": action_delta,
            "steady_state": steady_state,
            "reward_fx": reward_fx,
            "reward_forward": reward_forward,
            "reward_lateral": reward_lateral,
            "reward_energy": reward_energy,
            "reward_frequency": reward_frequency,
            "reward_smooth": reward_smooth,
            "physical_action": physical_action.astype(np.float32),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _physical_action(self, action: np.ndarray) -> np.ndarray:
        if not self.cfg.normalized_actions:
            return action.astype(np.float64)

        unit = 0.5 * (action + 1.0)
        if self.cfg.per_joint_action:
            lows, highs = self._per_joint_action_bounds()
        else:
            if self.cfg.fixed_frequency is None:
                lows = np.array([self.cfg.frequency_min, self.cfg.wavelength_min], dtype=np.float64)
                highs = np.array([self.cfg.frequency_max, self.cfg.wavelength_max], dtype=np.float64)
            else:
                lows = np.array([self.cfg.wavelength_min], dtype=np.float64)
                highs = np.array([self.cfg.wavelength_max], dtype=np.float64)
        return lows + unit * (highs - lows)

    def _per_joint_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        amp_lows = (
            np.asarray(self.cfg.amp_scale_lows, dtype=np.float64)
            if self.cfg.amp_scale_lows is not None
            else np.full(6, self.cfg.amp_scale_min, dtype=np.float64)
        )
        amp_highs = (
            np.asarray(self.cfg.amp_scale_highs, dtype=np.float64)
            if self.cfg.amp_scale_highs is not None
            else np.full(6, self.cfg.amp_scale_max, dtype=np.float64)
        )
        phase_lows = (
            np.asarray(self.cfg.phase_lag_lows, dtype=np.float64)
            if self.cfg.phase_lag_lows is not None
            else np.full(5, self.cfg.phase_lag_min, dtype=np.float64)
        )
        phase_highs = (
            np.asarray(self.cfg.phase_lag_highs, dtype=np.float64)
            if self.cfg.phase_lag_highs is not None
            else np.full(5, self.cfg.phase_lag_max, dtype=np.float64)
        )
        if amp_lows.size != 6 or amp_highs.size != 6:
            raise ValueError("amp_scale_lows/highs must each have 6 values")
        if phase_lows.size != 5 or phase_highs.size != 5:
            raise ValueError("phase_lag_lows/highs must each have 5 values")
        if self.cfg.fixed_frequency is None:
            lows = np.concatenate(([self.cfg.frequency_min], amp_lows, phase_lows))
            highs = np.concatenate(([self.cfg.frequency_max], amp_highs, phase_highs))
        else:
            lows = np.concatenate((amp_lows, phase_lows))
            highs = np.concatenate((amp_highs, phase_highs))
        return lows, highs

    def _force_metric(self, fx: float, fy: float) -> float:
        if self.cfg.target_metric == "fx":
            return fx
        if self.cfg.target_metric == "resultant":
            return float(np.hypot(fx, fy))
        raise ValueError(f"unknown target_metric: {self.cfg.target_metric}")

    def _holding_force(self) -> np.ndarray:
        # Tether actuator force is the force needed to hold the root in place.
        # The negative is the swimmer-generated force estimate.
        return np.array([
            -self.data.qfrc_actuator[self.root_dof_x],
            -self.data.qfrc_actuator[self.root_dof_y],
        ], dtype=np.float64)

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self.tail_qpos_addr]
        qd = self.data.qvel[self.tail_dof_addr]
        phase_features = np.array([
            np.sin(self.cpg.theta[0]),
            np.cos(self.cpg.theta[0]),
            np.mean(self.cpg.r),
            np.std(self.cpg.r),
        ], dtype=np.float64)
        root = np.array([
            self.data.qpos[0],
            self.data.qpos[1],
            self.data.qpos[2],
            self.data.qvel[0],
            self.data.qvel[1],
            self.data.qvel[2],
        ], dtype=np.float64)
        return np.concatenate([q, qd, phase_features, root, self.last_force]).astype(np.float32)


if __name__ == "__main__":
    env = EelTetheredRLEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print("smoke test OK")
    print("obs shape:", obs.shape)
    print("last info:", info)
    print("total reward:", round(total_reward, 3))
