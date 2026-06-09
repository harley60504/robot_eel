import numpy as np
from .base import SwimmerBase, SwimParams

class CPGSwimmer(SwimmerBase):
    name = "CPG"

    def __init__(
        self,
        alpha: float = 20.0,
        coupling: float = 8.0,
        substeps: int = 5,
        taper_head: float = 0.4,
        taper_tail: float = 1.0,
        steer_gain: float = 0.25,
        steer_front_n: int = 4
    ):
        self.alpha = float(alpha)
        self.coupling = float(coupling)
        self.substeps = int(max(1, substeps))
        self.taper_head = float(taper_head)
        self.taper_tail = float(taper_tail)
        self.steer_gain = float(steer_gain)
        self.steer_front_n = int(steer_front_n)

        self.x = None
        self.y = None

    def reset(self, num_joints: int):
        self.x = 0.01 * np.random.randn(num_joints)
        self.y = 0.01 * np.random.randn(num_joints)

    def _step(self, dt, omega, phase_lag, amp_profile, steer_phase_bias):
        x = self.x
        y = self.y

        phi = np.arctan2(y, x)
        r2 = x*x + y*y

        dphi = np.zeros_like(phi)
        n = len(phi)
        for i in range(n):
            if i > 0:
                target = phase_lag + steer_phase_bias[i]
                dphi[i] += np.sin((phi[i-1] + target) - phi[i])
            if i < n - 1:
                target = phase_lag + steer_phase_bias[i+1]
                dphi[i] += np.sin((phi[i+1] - target) - phi[i])

        A2 = amp_profile * amp_profile
        xdot = self.alpha * (A2 - r2) * x - omega * y
        ydot = self.alpha * (A2 - r2) * y + omega * x + self.coupling * dphi

        self.x = x + dt * xdot
        self.y = y + dt * ydot

    def compute_ctrl(self, t: float, num_joints: int, p: SwimParams) -> np.ndarray:
        if self.x is None or len(self.x) != num_joints:
            self.reset(num_joints)

        steer = 0.0 if p.auto_mode else p.turn

        # amp profile：沿身體 taper
        amp_profile = np.zeros(num_joints, dtype=np.float64)
        for i in range(num_joints):
            taper = self.taper_head + (self.taper_tail - self.taper_head) * (i / (num_joints - 1))
            amp_profile[i] = p.amp * taper

        # steering：相位差偏置（只在前段）
        steer_phase_bias = np.zeros(num_joints, dtype=np.float64)
        n = min(self.steer_front_n, num_joints)
        for i in range(n):
            w = 1.0 - i / max(1, n - 1)
            steer_phase_bias[i] = self.steer_gain * steer * w

        omega = 2 * np.pi * p.freq

        # 這裡 dt 由主程式的 timestep 決定，所以 compute_ctrl 只負責「一個大步」內部 substeps
        # dt 會在 main.py 呼叫時給進來（見 main.py 的用法）
        # 但為了保持介面簡單，我們把 dt = 1.0/ (p.freq*something) 不好
        # => 在 main.py 會呼叫 set_dt() 或直接在這裡用外部提供
        # 這裡先假設 main.py 會在每次呼叫前，先把 self._dt 設好
        dt_big = getattr(self, "_dt", 0.001)
        dt = dt_big / self.substeps

        for _ in range(self.substeps):
            self._step(dt, omega, p.step, amp_profile, steer_phase_bias)

        # output：x 當 position target
        return self.x.copy()

    def set_dt(self, dt: float):
        self._dt = float(dt)
