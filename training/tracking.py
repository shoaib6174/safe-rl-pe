"""Experiment tracking: wandb + TensorBoard integration with PE-specific metrics.

SB3 logs natively to TensorBoard. With sync_tensorboard=True, wandb mirrors
everything automatically. Custom callbacks log domain-specific PE metrics.
"""

from collections import deque

import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


def init_tracking(cfg: DictConfig):
    """Initialize wandb run with Hydra config as hyperparameters.

    Returns wandb run object, or None if wandb is disabled.
    """
    import wandb

    if cfg.wandb.mode == "disabled":
        return None

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=f"PPO_seed{cfg.seed}",
        group=cfg.experiment_group,
        tags=list(cfg.wandb.tags) + [f"seed_{cfg.seed}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        save_code=cfg.wandb.save_code,
        mode=cfg.wandb.mode,
    )
    return run


class PursuitEvasionMetricsCallback(BaseCallback):
    """Log domain-specific PE metrics to TensorBoard/wandb.

    Reads episode_metrics from the info dict at episode boundaries.
    Logs every log_frequency steps to avoid overhead.
    """

    def __init__(self, log_frequency: int = 1024, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.capture_times = deque(maxlen=100)
        self.min_distances = deque(maxlen=100)
        self.capture_rates = deque(maxlen=100)
        self.episode_durations = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_metrics" in info:
                m = info["episode_metrics"]
                self.capture_rates.append(float(m.get("captured", False)))
                self.min_distances.append(m.get("min_distance", 0))
                if m.get("captured"):
                    self.capture_times.append(m["capture_time"])
                self.episode_durations.append(m.get("episode_length", 0))

        if self.n_calls % self.log_frequency == 0 and len(self.capture_rates) > 0:
            self.logger.record("pursuit/capture_rate", np.mean(self.capture_rates))
            self.logger.record(
                "pursuit/min_distance_mean", np.mean(self.min_distances),
            )
            self.logger.record(
                "pursuit/episode_duration_mean", np.mean(self.episode_durations),
            )
            if len(self.capture_times) > 0:
                self.logger.record(
                    "pursuit/capture_time_mean", np.mean(self.capture_times),
                )

        return True


class HParamCallback(BaseCallback):
    """Log hyperparameters to TensorBoard HPARAMS tab."""

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "ent_coef": self.model.ent_coef,
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "pursuit/capture_rate": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
