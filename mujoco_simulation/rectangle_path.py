from __future__ import annotations

import numpy as np


class RectanglePath:
    def __init__(self, half_x: float, half_y: float):
        self.half_x = float(half_x)
        self.half_y = float(half_y)
        self.points = np.array(
            [
                [self.half_x, -self.half_y],
                [self.half_x, self.half_y],
                [-self.half_x, self.half_y],
                [-self.half_x, -self.half_y],
            ],
            dtype=np.float64,
        )
        self.starts = np.vstack([self.points[-1], self.points[:-1]])
        self.ends = self.points
        self.segments = self.ends - self.starts
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.total_length = float(self.cumulative[-1])

    def closest_s(self, xy: np.ndarray) -> float:
        xy = np.asarray(xy, dtype=np.float64)
        best_distance = np.inf
        best_s = 0.0
        for i, (start, segment, length) in enumerate(zip(self.starts, self.segments, self.lengths)):
            if length < 1e-9:
                continue
            u = float(np.clip(np.dot(xy - start, segment) / (length * length), 0.0, 1.0))
            projection = start + u * segment
            distance = float(np.linalg.norm(xy - projection))
            if distance < best_distance:
                best_distance = distance
                best_s = float(self.cumulative[i] + u * length)
        return best_s

    def point_at(self, s: float) -> np.ndarray:
        s = float(s % self.total_length)
        index = int(np.searchsorted(self.cumulative, s, side="right") - 1)
        index = min(max(index, 0), len(self.lengths) - 1)
        local = s - self.cumulative[index]
        u = 0.0 if self.lengths[index] < 1e-9 else local / self.lengths[index]
        return self.starts[index] + u * self.segments[index]

    def progress_info(self, s: float) -> tuple[int, int]:
        s = float(s % self.total_length)
        index = int(np.searchsorted(self.cumulative, s, side="right") - 1)
        index = min(max(index, 0), len(self.lengths) - 1)
        return index, index + 1
