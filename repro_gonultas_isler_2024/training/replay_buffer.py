import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0
        self.ptr = 0

        self.obs_p = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs_e = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_p = np.zeros((capacity, action_dim), dtype=np.float32)
        self.actions_e = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards_p = np.zeros(capacity, dtype=np.float32)
        self.rewards_e = np.zeros(capacity, dtype=np.float32)
        self.next_obs_p = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_e = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs_p: np.ndarray,
        obs_e: np.ndarray,
        action_p: np.ndarray,
        action_e: np.ndarray,
        reward_p: float,
        reward_e: float,
        next_obs_p: np.ndarray,
        next_obs_e: np.ndarray,
        done: bool,
    ):
        idx = self.ptr
        self.obs_p[idx] = obs_p
        self.obs_e[idx] = obs_e
        self.actions_p[idx] = action_p
        self.actions_e[idx] = action_e
        self.rewards_p[idx] = reward_p
        self.rewards_e[idx] = reward_e
        self.next_obs_p[idx] = next_obs_p
        self.next_obs_e[idx] = next_obs_e
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs_p": self.obs_p[indices],
            "obs_e": self.obs_e[indices],
            "actions_p": self.actions_p[indices],
            "actions_e": self.actions_e[indices],
            "rewards_p": self.rewards_p[indices],
            "rewards_e": self.rewards_e[indices],
            "next_obs_p": self.next_obs_p[indices],
            "next_obs_e": self.next_obs_e[indices],
            "dones": self.dones[indices],
        }
