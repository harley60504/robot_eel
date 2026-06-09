import numpy as np
from .base import SwimmerBase, SwimParams

class KuramotoSwimmer(SwimmerBase):
    """
    Kuramoto phase oscillator chain:
      dphi_i = omega + K * [ sin(phi_{i-1} - phi_i - lag) + sin(phi_{i+1} - phi_i + lag) ]
      theta_i = A_i * sin(phi_i)

    轉向（Steering）做法：只在前 N 節對「目標相位差 lag」加偏置，形成前段曲率
    -> 比 DC bias 穩很多，也比較不會原地自轉。

    wave_type 支援：
      - Traveling: 使用 lag = step (你原本 step)
      - Standing : 讓相位差趨近 0 + 用 sin(i*step) 當空間包絡（比較像駐波的感覺）
    """
    name = "Kuramoto"

    def __init__(
        self,
        coupling: float = 10.0,     # K：越大越「黏」成穩定波形，但太大會變硬
        substeps: int = 5,          # 每個 mj_step 內部積分次數，越大越穩但更慢
        taper_head: float = 0.35,   # 頭部振幅比例
        taper_tail: float = 1.00,   # 尾部振幅比例
        steer_gain: float = 0.70,   # 轉向強度（相位差偏置）
        steer_front_n: int = 4,     # 只影響前 N 節（你想要的）
        steer_sign: float = 1.0,    # 如果你覺得左右反了，把它改 -1.0
    ):
        self.K = float(coupling)
        self.substeps = int(max(1, substeps))
        self.taper_head = float(taper_head)
        self.taper_tail = float(taper_tail)
        self.steer_gain = float(steer_gain)
        self.steer_front_n = int(steer_front_n)
        self.steer_sign = float(steer_sign)

        self.phi = None
        self._dt = 0.001

    def set_dt(self, dt: float):
        self._dt = float(dt)

    def reset(self, num_joints: int):
        # 初始化相位：給一個平滑斜坡，避免一開始亂跳
        self.phi = np.linspace(0.0, 0.5 * num_joints, num_joints, dtype=np.float64)

    def _amp_profile(self, num_joints: int, amp: float) -> np.ndarray:
        # A_i: 從頭到尾逐漸增加（比較像魚）
        A = np.zeros(num_joints, dtype=np.float64)
        for i in range(num_joints):
            taper = self.taper_head + (self.taper_tail - self.taper_head) * (i / (num_joints - 1))
            A[i] = amp * taper
        return A

    def compute_ctrl(self, t: float, num_joints: int, p: SwimParams) -> np.ndarray:
        if self.phi is None or len(self.phi) != num_joints:
            self.reset(num_joints)

        dt_big = getattr(self, "_dt", 0.001)
        dt = dt_big / self.substeps

        omega = 2.0 * np.pi * p.freq
        A = self._amp_profile(num_joints, p.amp)

        # 轉向：只在前 N 節加「相位差偏置」
        steer = 0.0 if p.auto_mode else p.turn
        steer *= self.steer_sign  # 左右反了就改 steer_sign=-1

        steer_bias = np.zeros(num_joints, dtype=np.float64)
        n = min(self.steer_front_n, num_joints)
        for i in range(n):
            w = 1.0 - (i / max(1, n - 1))   # 越靠近頭越大
            steer_bias[i] = self.steer_gain * steer * w

        # 目標相位差（Traveling）
        lag = float(p.step)

        # ===== 內部積分（Euler，多 substeps）=====
        for _ in range(self.substeps):
            phi = self.phi

            # neighbor coupling term
            dphi = np.full(num_joints, omega, dtype=np.float64)

            for i in range(num_joints):
                if p.wave_type == "Standing":
                    # 駐波：相位差趨近 0（更像整體同相震盪）
                    lag_i = 0.0
                else:
                    # 行進波：相位差 = step + steer_bias（只前段有）
                    lag_i = lag + steer_bias[i]

                if i > 0:
                    dphi[i] += self.K * np.sin(phi[i - 1] - phi[i] - lag_i)
                if i < num_joints - 1:
                    dphi[i] += self.K * np.sin(phi[i + 1] - phi[i] + lag_i)

            self.phi = phi + dt * dphi

        # ===== 產生輸出角度 =====
        if p.wave_type == "Standing":
            # 駐波輸出：同相振盪 * 空間包絡
            # （這是很常見的“站波近似”，比硬做 forward+back wave 簡單而且穩）
            global_phase = self.phi[0]
            env = np.sin(np.arange(num_joints) * p.step)
            theta = A * np.sin(global_phase) * env
        else:
            theta = A * np.sin(self.phi)

        return theta
