import numpy as np


class WedgeSensor:
    def __init__(self, fov_angle: float, fov_range: float):
        self.fov_angle = fov_angle
        self.fov_range = fov_range
        self.half_angle = fov_angle / 2.0

    def is_visible(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        opponent_pos: np.ndarray,
    ) -> bool:
        rel = opponent_pos - agent_pos
        dist = float(np.linalg.norm(rel))
        if dist > self.fov_range:
            return False
        angle_to_opponent = float(np.arctan2(rel[1], rel[0]))
        angle_diff = self._normalize_angle(angle_to_opponent - agent_yaw)
        return abs(angle_diff) <= self.half_angle

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
