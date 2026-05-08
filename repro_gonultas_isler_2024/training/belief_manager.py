from typing import Optional
import numpy as np
import torch
from agents.bimdn import BiMDN


class BeliefStateManager:
    """Manages observation history, belief computation, and BiMDN training."""

    def __init__(
        self,
        obs_dim: int = 11,
        buffer_size: int = 25000,
        history_length: int = 200,
        downsample: int = 10,
        lr: float = 0.002,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.downsample = downsample
        self.sequence_length = history_length // downsample
        self.device = torch.device(device)

        self.bimdn = BiMDN(obs_dim=obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.bimdn.parameters(), lr=lr)

        # Rolling history of raw observations (at sim rate)
        self.history = []

        # Training buffer: (sequence, target_pos) pairs
        self.training_buffer = []
        self.buffer_size = buffer_size

    def add_observation(self, obs: np.ndarray):
        """Add a raw observation to the rolling history."""
        self.history.append(obs.copy())
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def _get_sequence(self) -> Optional[np.ndarray]:
        """Extract downsampled sequence from history."""
        if len(self.history) < self.sequence_length * self.downsample:
            return None
        indices = list(range(
            len(self.history) - self.sequence_length * self.downsample,
            len(self.history),
            self.downsample,
        ))
        return np.array([self.history[i] for i in indices], dtype=np.float32)

    def get_belief(self) -> np.ndarray:
        """Compute weighted-mean belief from current history."""
        sequence = self._get_sequence()
        if sequence is None:
            return np.zeros(2, dtype=np.float32)

        with torch.no_grad():
            seq_t = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            belief = self.bimdn.get_weighted_mean(seq_t)
            return belief.cpu().numpy()[0]

    def store_training_sample(self, target_pos: np.ndarray):
        """Store a (sequence, target) pair for BiMDN training."""
        sequence = self._get_sequence()
        if sequence is None:
            return
        self.training_buffer.append((sequence, target_pos.copy()))
        if len(self.training_buffer) > self.buffer_size:
            self.training_buffer.pop(0)

    def train_step(self, batch_size: int = 256) -> float:
        """Train BiMDN on a random batch from the training buffer."""
        if len(self.training_buffer) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.training_buffer), batch_size, replace=False)
        sequences = np.array([self.training_buffer[i][0] for i in indices], dtype=np.float32)
        targets = np.array([self.training_buffer[i][1] for i in indices], dtype=np.float32)

        sequences_t = torch.FloatTensor(sequences).to(self.device)
        targets_t = torch.FloatTensor(targets).to(self.device)

        loss = self.bimdn.compute_nll_loss(sequences_t, targets_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save(self.bimdn.state_dict(), path)

    def load(self, path: str):
        self.bimdn.load_state_dict(torch.load(path, map_location=self.device))
