from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from rl_free_swim_env import EelFreeSwimRLEnv, FreeSwimConfig, spaces
from rl_turning_env import EelTurningRLEnv, TurningConfig


GRAVITY_MPS2 = 9.80665


@dataclass
class RobotImuObsConfig:
    """Options for observations that can plausibly exist on the robot."""

    robot_imu_features: str = "basic"
    servo_center_deg: float = 120.0
    servo_feedback_delay_steps: int = 3
    servo_feedback_noise_std_deg: float = 0.0
    servo_feedback_quantization_deg: float = 0.24
    imu_roll_mount_deg: float = 0.0
    imu_pitch_mount_deg: float = 0.0
    imu_yaw_mount_deg: float = 0.0
    imu_random_roll_deg: float = 0.0
    imu_random_pitch_deg: float = 0.0
    imu_random_yaw_deg: float = 0.0
    imu_accel_noise_std_g: float = 0.0
    imu_gyro_noise_std_rps: float = 0.0
    imu_delay_steps: int = 0


@dataclass
class FreeSwimRobotImuConfig(FreeSwimConfig, RobotImuObsConfig):
    pass


@dataclass
class TurningRobotImuConfig(TurningConfig, RobotImuObsConfig):
    pass


class RobotImuObsMixin:
    """Build policy observations from servo, CPG, command, and IMU-like signals.

    MuJoCo-only ground truth such as world x/y/yaw is still available for rewards
    and info dicts in the parent envs, but it is intentionally not exposed here.
    """

    cfg: FreeSwimRobotImuConfig | TurningRobotImuConfig

    def _init_robot_imu_obs(self) -> None:
        self._imu_velocity_world = np.zeros(2, dtype=np.float64)
        self._imu_samples: deque[np.ndarray] = deque()
        self._imu_feature_samples: deque[np.ndarray] = deque()
        self._imu_rotation = np.eye(3, dtype=np.float64)
        self._servo_samples: deque[np.ndarray] = deque()
        self._prev_servo_q_obs = np.zeros(6, dtype=np.float64)
        self._prev_feedback_for_velocity = np.zeros(6, dtype=np.float64)
        self._reset_imu_filter()
        self._reset_servo_feedback_filter()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._robot_obs_dim(),),
            dtype=np.float32,
        )

    def _reset_imu_filter(self) -> None:
        self._imu_velocity_world = self._current_world_velocity()
        self._imu_samples.clear()
        self._imu_feature_samples.clear()
        self._imu_rotation = self._make_imu_rotation()

    def _make_imu_rotation(self) -> np.ndarray:
        roll = float(self.cfg.imu_roll_mount_deg)
        pitch = float(self.cfg.imu_pitch_mount_deg)
        yaw = float(self.cfg.imu_yaw_mount_deg)
        if float(self.cfg.imu_random_roll_deg) > 0.0:
            roll += float(self.np_random.uniform(-self.cfg.imu_random_roll_deg, self.cfg.imu_random_roll_deg))
        if float(self.cfg.imu_random_pitch_deg) > 0.0:
            pitch += float(self.np_random.uniform(-self.cfg.imu_random_pitch_deg, self.cfg.imu_random_pitch_deg))
        if float(self.cfg.imu_random_yaw_deg) > 0.0:
            yaw += float(self.np_random.uniform(-self.cfg.imu_random_yaw_deg, self.cfg.imu_random_yaw_deg))

        cr, sr = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
        cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz @ ry @ rx

    def _reset_servo_feedback_filter(self) -> None:
        q = np.asarray(self.data.qpos[self.tail_qpos_addr], dtype=np.float64).copy()
        self._servo_samples.clear()
        self._servo_samples.append(q)
        self._prev_servo_q_obs = q.copy()
        self._prev_feedback_for_velocity = q.copy()

    def _robot_obs_dim(self) -> int:
        extra = 0
        if self.cfg.robot_imu_features == "servo_qd":
            extra = 6
        elif self.cfg.robot_imu_features == "filtered":
            extra = 12
        return 6 + 6 + 6 + 4 + int(self.action_dim) + 6 + extra + self._command_dim()

    def _command_dim(self) -> int:
        return 1

    def _current_world_velocity(self) -> np.ndarray:
        return np.array([self.data.qvel[0], self.data.qvel[1]], dtype=np.float64)

    def _robot_command_obs(self) -> np.ndarray:
        if hasattr(self, "signed_target_yaw_rate"):
            return np.array([float(self.signed_target_yaw_rate)], dtype=np.float64)
        return np.array([float(self.cfg.target_speed)], dtype=np.float64)

    def _phase_obs(self) -> np.ndarray:
        return np.array(
            [
                np.sin(self.cpg.theta[0]),
                np.cos(self.cpg.theta[0]),
                np.mean(self.cpg.r),
                np.std(self.cpg.r),
            ],
            dtype=np.float64,
        )

    def _imu_obs(self) -> np.ndarray:
        velocity_world = self._current_world_velocity()
        dt = max(float(self.cfg.control_dt), 1e-6)
        accel_world = (velocity_world - self._imu_velocity_world) / dt
        self._imu_velocity_world = velocity_world

        yaw = float(self.data.qpos[2])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        accel_body_x = cos_yaw * accel_world[0] + sin_yaw * accel_world[1]
        accel_body_y = -sin_yaw * accel_world[0] + cos_yaw * accel_world[1]

        accel_body_g = np.array([accel_body_x / GRAVITY_MPS2, accel_body_y / GRAVITY_MPS2, 1.0], dtype=np.float64)
        gyro_body_rps = np.array([0.0, 0.0, float(self.data.qvel[2])], dtype=np.float64)
        accel_g = self._imu_rotation @ accel_body_g
        gyro_rps = self._imu_rotation @ gyro_body_rps

        if float(self.cfg.imu_accel_noise_std_g) > 0.0:
            accel_g += self.np_random.normal(0.0, float(self.cfg.imu_accel_noise_std_g), size=3)
        if float(self.cfg.imu_gyro_noise_std_rps) > 0.0:
            gyro_rps += self.np_random.normal(0.0, float(self.cfg.imu_gyro_noise_std_rps), size=3)

        sample = np.concatenate((accel_g, gyro_rps)).astype(np.float64)
        delay_steps = max(0, int(self.cfg.imu_delay_steps))
        self._imu_samples.append(sample)
        while len(self._imu_samples) > delay_steps + 1:
            self._imu_samples.popleft()
        return self._imu_samples[0]

    def _filtered_imu_features(self, imu_obs: np.ndarray, feedback: np.ndarray) -> np.ndarray:
        if self.cfg.robot_imu_features == "basic":
            return np.empty(0, dtype=np.float64)
        qd_est = (feedback - self._prev_feedback_for_velocity) / max(float(self.cfg.control_dt), 1e-6)
        self._prev_feedback_for_velocity = feedback.copy()
        if self.cfg.robot_imu_features == "servo_qd":
            return qd_est.astype(np.float64)
        if self.cfg.robot_imu_features != "filtered":
            raise ValueError("robot_imu_features must be basic, servo_qd, or filtered")

        self._imu_feature_samples.append(np.asarray(imu_obs, dtype=np.float64).copy())
        window = max(1, int(round(1.0 / max(float(self.cfg.control_dt), 1e-6))))
        while len(self._imu_feature_samples) > window:
            self._imu_feature_samples.popleft()

        arr = np.asarray(self._imu_feature_samples, dtype=np.float64)
        filtered = np.mean(arr, axis=0)
        accel_x = float(filtered[0])
        accel_y = float(filtered[1])
        gyro_z = float(filtered[5])
        accel_xy = float(np.hypot(filtered[0], filtered[1]))
        radius_proxy = accel_y * GRAVITY_MPS2 / (gyro_z * gyro_z + 1e-4)
        speed_proxy = accel_y * GRAVITY_MPS2 / (abs(gyro_z) + 1e-3)
        return np.concatenate(
            (
                np.array([accel_x, accel_y, accel_xy, gyro_z, radius_proxy, speed_proxy], dtype=np.float64),
                qd_est.astype(np.float64),
            )
        )

    def _servo_target_obs(self) -> np.ndarray:
        return np.asarray(self.data.ctrl[self.tail_ctrl_slice], dtype=np.float64).copy()

    def _servo_rad_to_centered_deg(self, value_rad: np.ndarray) -> np.ndarray:
        return float(self.cfg.servo_center_deg) + np.rad2deg(value_rad)

    def _servo_centered_deg_to_rad(self, value_deg: np.ndarray) -> np.ndarray:
        return np.deg2rad(value_deg - float(self.cfg.servo_center_deg))

    def _servo_feedback_obs(self) -> np.ndarray:
        q = np.asarray(self.data.qpos[self.tail_qpos_addr], dtype=np.float64).copy()
        if float(self.cfg.servo_feedback_noise_std_deg) > 0.0:
            noise_rad = np.deg2rad(float(self.cfg.servo_feedback_noise_std_deg))
            q += self.np_random.normal(0.0, noise_rad, size=q.shape)
        if float(self.cfg.servo_feedback_quantization_deg) > 0.0:
            quantum_rad = np.deg2rad(float(self.cfg.servo_feedback_quantization_deg))
            q = np.round(q / quantum_rad) * quantum_rad

        delay_steps = max(0, int(self.cfg.servo_feedback_delay_steps))
        self._servo_samples.append(q)
        while len(self._servo_samples) > delay_steps + 1:
            self._servo_samples.popleft()

        q_obs = self._servo_samples[0].copy()
        self._prev_servo_q_obs = q_obs.copy()
        return q_obs

    def _obs(self) -> np.ndarray:
        target = self._servo_target_obs()
        feedback = self._servo_feedback_obs()
        error = target - feedback
        imu = self._imu_obs()
        extra = self._filtered_imu_features(imu, feedback)
        return np.concatenate(
            (
                target,
                feedback,
                error,
                self._phase_obs(),
                self.prev_action,
                imu,
                extra,
                self._robot_command_obs(),
            )
        ).astype(np.float32)


class EelFreeSwimRobotImuEnv(RobotImuObsMixin, EelFreeSwimRLEnv):
    """Free-swim env with robot-deployable IMU feedback observations."""

    def __init__(self, cfg: FreeSwimRobotImuConfig | None = None):
        super().__init__(cfg or FreeSwimRobotImuConfig())
        self._init_robot_imu_obs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_imu_filter()
        self._reset_servo_feedback_filter()
        return self._obs(), info


class EelTurningRobotImuEnv(RobotImuObsMixin, EelTurningRLEnv):
    """Turning env with robot-deployable IMU feedback observations."""

    def __init__(self, cfg: TurningRobotImuConfig | None = None):
        super().__init__(cfg or TurningRobotImuConfig())
        self._init_robot_imu_obs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_imu_filter()
        self._reset_servo_feedback_filter()
        return self._obs(), info


if __name__ == "__main__":
    for env in (EelFreeSwimRobotImuEnv(), EelTurningRobotImuEnv()):
        obs, _ = env.reset()
        print(type(env).__name__, "obs shape:", obs.shape, "action shape:", env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print("step:", obs.shape, reward, terminated, truncated, sorted(info)[:5])
