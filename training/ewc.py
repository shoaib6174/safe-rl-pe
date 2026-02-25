"""Elastic Weight Consolidation (EWC) for curriculum transitions.

Prevents catastrophic forgetting by penalizing parameter drift from a saved
anchor point, weighted by the diagonal Fisher information matrix.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017.

Integration with SB3 PPO is done via PyTorch `register_hook()` on parameters,
injecting the EWC gradient term `lambda * F * (theta - theta_star)` during
backpropagation. This avoids subclassing PPO or modifying the training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    Usage:
        ewc = EWCRegularizer(lambda_=1000.0)

        # On curriculum advancement — snapshot current policy
        ewc.snapshot(model, obs_batch)

        # During training — register hooks before .learn(), remove after
        hooks = ewc.register_hooks(model)
        model.learn(total_timesteps=N)
        ewc.remove_hooks(hooks)

        # For logging
        loss = ewc.penalty(model)
    """

    def __init__(
        self,
        lambda_: float = 1000.0,
        fisher_samples: int = 1024,
    ):
        """Initialize EWC regularizer.

        Args:
            lambda_: Regularization strength. Higher = more conservative.
            fisher_samples: Number of observations to estimate Fisher info.
        """
        self.lambda_ = lambda_
        self.fisher_samples = fisher_samples

        # Anchor parameters (set by snapshot())
        self._theta_star: dict[str, torch.Tensor] = {}
        # Diagonal Fisher information (set by snapshot())
        self._fisher: dict[str, torch.Tensor] = {}
        self._has_snapshot = False

    @property
    def has_snapshot(self) -> bool:
        """Whether a parameter snapshot has been taken."""
        return self._has_snapshot

    def snapshot(self, model: PPO, obs_batch: torch.Tensor) -> None:
        """Save current parameters and compute diagonal Fisher information.

        Should be called right before curriculum advancement, using observations
        from the current (mastered) level.

        Args:
            model: SB3 PPO model whose policy we want to anchor.
            obs_batch: Tensor of observations [N, obs_dim] from the current level.
                       Should have at least `fisher_samples` observations.
        """
        policy = model.policy

        # 1. Save theta_star = current parameter values
        self._theta_star = {
            name: param.detach().clone()
            for name, param in policy.named_parameters()
            if param.requires_grad
        }

        # 2. Compute diagonal Fisher information via policy log-prob gradients
        self._fisher = {
            name: torch.zeros_like(param)
            for name, param in policy.named_parameters()
            if param.requires_grad
        }

        policy.eval()
        n_samples = min(len(obs_batch), self.fisher_samples)
        obs = obs_batch[:n_samples]

        # Process in mini-batches to avoid OOM
        batch_size = min(128, n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = obs[start:end]

            # Get action distribution from policy
            with torch.enable_grad():
                dist = policy.get_distribution(batch)
                # Sample actions and compute log probabilities
                actions = dist.get_actions(deterministic=False)
                log_probs = dist.log_prob(actions)

                # Sum log probs across batch for gradient computation
                loss = log_probs.sum()

                policy.zero_grad()
                loss.backward()

            # Accumulate squared gradients (diagonal Fisher)
            for name, param in policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.detach().pow(2)

        # Normalize by number of samples
        for name in self._fisher:
            self._fisher[name] /= n_samples

        policy.train()
        self._has_snapshot = True

    def register_hooks(self, model: PPO) -> list:
        """Register gradient hooks that inject EWC regularization.

        Adds `lambda * F * (theta - theta_star)` to each parameter's gradient
        during backpropagation.

        Args:
            model: SB3 PPO model to regularize.

        Returns:
            List of hook handles. Pass to remove_hooks() after training.
        """
        if not self._has_snapshot:
            return []

        handles = []
        for name, param in model.policy.named_parameters():
            if param.requires_grad and name in self._fisher:
                # Capture name in closure
                _name = name
                _fisher = self._fisher[name]
                _theta_star = self._theta_star[name]
                _lambda = self.lambda_

                def hook(grad, f=_fisher, ts=_theta_star, lam=_lambda, p=param):
                    return grad + lam * f * (p.data - ts)

                h = param.register_hook(hook)
                handles.append(h)

        return handles

    @staticmethod
    def remove_hooks(handles: list) -> None:
        """Remove gradient hooks after training.

        Args:
            handles: List of hook handles from register_hooks().
        """
        for h in handles:
            h.remove()

    def penalty(self, model: PPO) -> float:
        """Compute scalar EWC penalty for logging.

        Returns:
            0.5 * lambda * sum_i F_i * (theta_i - theta_star_i)^2
        """
        if not self._has_snapshot:
            return 0.0

        total = 0.0
        for name, param in model.policy.named_parameters():
            if param.requires_grad and name in self._fisher:
                diff = param.data - self._theta_star[name]
                total += (self._fisher[name] * diff.pow(2)).sum().item()

        return 0.5 * self.lambda_ * total
