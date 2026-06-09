from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SwimParams:
    amp: float
    freq: float
    step: float
    turn: float
    wave_type: str     # "Traveling" / "Standing"
    auto_mode: bool

class SwimmerBase:
    """所有游法共用介面：回傳 ctrl 向量（len = num_joints）"""
    name: str = "Base"

    def reset(self, num_joints: int):
        """重置內部狀態（例如 CPG state）"""
        pass

    def compute_ctrl(self, t: float, num_joints: int, p: SwimParams) -> np.ndarray:
        raise NotImplementedError
