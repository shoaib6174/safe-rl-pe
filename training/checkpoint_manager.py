"""Checkpoint management for AMS-DRL self-play training.

Manages rolling + milestone checkpoints with rollback support.
Each checkpoint saves BOTH the PPO model AND the BiMDN encoder state_dict
to maintain consistency between policy and belief encoder.

Checkpoint directory structure:
    checkpoints/
    ├── rolling_10000/
    │   ├── ppo.zip          (SB3 PPO model)
    │   ├── encoder.pt       (BiMDN/encoder state_dict)
    │   └── meta.json        (step, phase, metrics, curriculum_level)
    ├── rolling_20000/
    │   └── ...
    ├── milestone_phase3/
    │   └── ...
    └── best/
        └── ...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


class CheckpointManager:
    """Manage rolling + milestone checkpoints with rollback support.

    Rolling checkpoints are saved periodically and cleaned up when the
    max count is exceeded. Milestone checkpoints are saved at phase
    boundaries and never cleaned up. Best checkpoints track the best
    metric seen so far.

    Args:
        checkpoint_dir: Root directory for checkpoints.
        max_rolling: Maximum number of rolling checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "./checkpoints",
        max_rolling: int = 15,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_rolling = max_rolling
        self.rolling: list[tuple[int, str]] = []  # (step, path)
        self.best_metric = -float("inf")

    def save_rolling(
        self,
        model,
        encoder=None,
        step: int = 0,
        meta: dict | None = None,
    ):
        """Save a rolling checkpoint.

        Args:
            model: SB3 PPO model.
            encoder: BiMDN encoder (nn.Module) or None.
            step: Current training step.
            meta: Additional metadata (phase, metrics, etc.).
        """
        ckpt_dir = self.checkpoint_dir / f"rolling_{step}"
        ckpt_dir.mkdir(exist_ok=True)

        model.save(str(ckpt_dir / "ppo.zip"))
        if encoder is not None:
            torch.save(encoder.state_dict(), str(ckpt_dir / "encoder.pt"))

        meta_data = meta or {}
        meta_data["step"] = step
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta_data, f, indent=2)

        self.rolling.append((step, str(ckpt_dir)))

        # Clean old rolling checkpoints
        while len(self.rolling) > self.max_rolling:
            _, old_path = self.rolling.pop(0)
            if Path(old_path).exists():
                shutil.rmtree(old_path)

    def save_milestone(
        self,
        model,
        encoder=None,
        phase: int = 0,
        role: str = "",
        meta: dict | None = None,
    ):
        """Save a milestone checkpoint (never auto-cleaned).

        Args:
            model: SB3 PPO model.
            encoder: BiMDN encoder (nn.Module) or None.
            phase: Current self-play phase number.
            role: 'pursuer' or 'evader' (who was training this phase).
            meta: Additional metadata.
        """
        ckpt_dir = self.checkpoint_dir / f"milestone_phase{phase}_{role}"
        ckpt_dir.mkdir(exist_ok=True)

        model.save(str(ckpt_dir / "ppo.zip"))
        if encoder is not None:
            torch.save(encoder.state_dict(), str(ckpt_dir / "encoder.pt"))

        meta_data = meta or {}
        meta_data["phase"] = phase
        meta_data["role"] = role
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta_data, f, indent=2)

    def save_best(
        self,
        model,
        encoder=None,
        metric_value: float = 0.0,
        meta: dict | None = None,
    ) -> bool:
        """Save checkpoint if metric improves over best seen.

        Args:
            model: SB3 PPO model.
            encoder: BiMDN encoder (nn.Module) or None.
            metric_value: Current metric value (higher = better).
            meta: Additional metadata.

        Returns:
            True if this was a new best, False otherwise.
        """
        if metric_value <= self.best_metric:
            return False

        self.best_metric = metric_value
        ckpt_dir = self.checkpoint_dir / "best"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True)

        model.save(str(ckpt_dir / "ppo.zip"))
        if encoder is not None:
            torch.save(encoder.state_dict(), str(ckpt_dir / "encoder.pt"))

        meta_data = meta or {}
        meta_data["metric_value"] = metric_value
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta_data, f, indent=2)

        return True

    def perform_rollback(
        self,
        model_class,
        encoder=None,
        rollback_steps: int = 3,
    ) -> tuple:
        """Rollback to an earlier rolling checkpoint.

        Loads the checkpoint from `rollback_steps` back in the rolling history,
        removes all newer checkpoints, and returns the restored state.

        Args:
            model_class: SB3 model class (e.g., PPO) for loading.
            encoder: BiMDN encoder module to load state_dict into (or None).
            rollback_steps: How many checkpoints to go back.

        Returns:
            (model, step) if encoder is None, else (model, encoder, step).

        Raises:
            ValueError: If not enough rolling checkpoints for rollback.
        """
        if len(self.rolling) < rollback_steps:
            raise ValueError(
                f"Cannot rollback {rollback_steps} steps: "
                f"only {len(self.rolling)} rolling checkpoints available"
            )

        target_idx = -(rollback_steps)
        step, ckpt_dir = self.rolling[target_idx]
        ckpt_path = Path(ckpt_dir)

        # Load model
        model = model_class.load(str(ckpt_path / "ppo.zip"))

        # Load encoder if provided
        if encoder is not None:
            encoder_path = ckpt_path / "encoder.pt"
            if encoder_path.exists():
                encoder.load_state_dict(
                    torch.load(str(encoder_path), weights_only=True)
                )

        # Remove newer checkpoints
        while self.rolling and self.rolling[-1][0] > step:
            _, old_path = self.rolling.pop()
            if Path(old_path).exists():
                shutil.rmtree(old_path)

        if encoder is not None:
            return model, encoder, step
        return model, step

    def get_latest_checkpoint(self) -> tuple[str, int] | None:
        """Get path and step of the latest rolling checkpoint.

        Returns:
            (path, step) or None if no checkpoints exist.
        """
        if not self.rolling:
            return None
        step, path = self.rolling[-1]
        return path, step

    def load_checkpoint(
        self,
        model_class,
        ckpt_dir: str | Path,
        encoder=None,
    ) -> tuple:
        """Load a specific checkpoint by directory path.

        Args:
            model_class: SB3 model class for loading.
            ckpt_dir: Path to checkpoint directory.
            encoder: Optional encoder to load state into.

        Returns:
            (model, meta_dict) if encoder is None,
            else (model, encoder, meta_dict).
        """
        ckpt_path = Path(ckpt_dir)
        model = model_class.load(str(ckpt_path / "ppo.zip"))

        meta = {}
        meta_path = ckpt_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        if encoder is not None:
            encoder_path = ckpt_path / "encoder.pt"
            if encoder_path.exists():
                encoder.load_state_dict(
                    torch.load(str(encoder_path), weights_only=True)
                )
            return model, encoder, meta

        return model, meta
