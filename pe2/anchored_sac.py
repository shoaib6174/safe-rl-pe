"""Pilot P-B (derisk plan v2, T0.3): SAC with a KL anchor to a frozen clone.

D82 root-caused the fine-tune collapse: after the critic warm-up the actor
maximizes E[Q(s, a ~ pi)] against a landscape whose action-dependence is
drift-noise-dominated at gamma=0.999 over a 1200-step horizon, so it leaves
the clone manifold and never comes back (0.185 -> ~0.02, four ways). This is
the RLHF-style remedy: add beta * KL(pi_theta(.|s) || pi_clone(.|s)) to the
actor loss so leaving the manifold has an explicit price.

The KL is computed between the PRE-TANH diagonal Gaussians. tanh is a fixed
bijection applied identically to both policies, and KL is invariant under a
bijective reparameterization, so this is EXACTLY the KL of the squashed
policies — not an approximation.

DELIBERATE, BOUNDED LIBRARY OVERRIDE
------------------------------------
`KLAnchoredSAC.train` is `stable_baselines3.sac.sac.SAC.train` (SB3 2.7.1)
copied from the installed source, with the additions marked `# ANCHOR`.
The project rule is that RL libraries are not rewritten; this is a
documented exception rather than a violation, because there is no seam in
SB3 for adding a term to the actor loss — the loss is assembled inside
`train` and never exposed. The exception is bounded: one method, one new
module, deletable in a single `rm`. To re-verify the copy against a future
SB3, diff this method against `sac.py` and check that only the `# ANCHOR`
lines differ. (The copy is verbatim modulo `ruff format` line wrapping.)
That check is AUTOMATED and enforced on every test run — see
`TestLibraryCopyIsFaithful` in tests/pe2/test_anchored_sac.py, which strips
the marked lines and asserts what remains parses to the installed SAC.train.
pyproject pins no SB3 upper bound, so that test is what stops an upgrade
from silently confounding the pilot race.

At `beta == 0.0` the KL is still MEASURED and logged (the control arm's real
drift is the whole point of the diagnosis) but not added to the loss, and the
actor's forward pass for it runs under `no_grad`. So the beta=0 arm remains
bit-identical to SB3's own SAC — verified by
`test_beta_zero_is_bit_identical_to_plain_sac`.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


def gaussian_kl(mean, log_std, mean_ref, log_std_ref):
    """KL(N(mean, e^log_std) || N(mean_ref, e^log_std_ref)), per sample.

    For diagonal Gaussians the divergence factorizes over dimensions:

        KL = log(s_ref / s) + (s^2 + (mu - mu_ref)^2) / (2 s_ref^2) - 1/2

    and the per-dimension terms are summed, so the returned tensor has the
    batch shape of the inputs with the action axis removed.
    """
    var = th.exp(2.0 * log_std)
    var_ref = th.exp(2.0 * log_std_ref)
    kl = (
        (log_std_ref - log_std) + (var + (mean - mean_ref) ** 2) / (2.0 * var_ref) - 0.5
    )
    return kl.sum(dim=-1)


class KLAnchoredSAC(SAC):
    """SAC whose actor loss carries `beta * KL(pi_theta || pi_anchor)`.

    `beta = 0` (or an unset anchor) reduces this to plain SB3 SAC. With an
    anchor set, `beta = 0` still REPORTS `train/anchor_kl` — it just does not
    price it — so the control arm's drift is measurable.
    """

    def __init__(self, *args, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            # gSDE reparameterizes the actor: get_action_dist_params then
            # returns the exploration NOISE matrix's log_std plus a
            # latent_sde kwarg, not a diagonal action Gaussian, so
            # gaussian_kl would compute a different quantity without
            # saying so. Refuse loudly instead of reporting a wrong KL.
            raise ValueError("KLAnchoredSAC does not support use_sde=True")
        if beta < 0.0:
            # The loss is only amended when beta > 0, so a negative weight
            # would train as plain SAC while the run reported itself as an
            # anchored arm. A mis-signed ablation must fail, not mislead.
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        self.beta = beta
        self.anchor = None

    def set_anchor(self, actor) -> None:
        """Freeze a deep copy of `actor` as the reference policy.

        A copy, not a reference: the live actor keeps training and the
        anchor must stay pinned to the clone it was taken from.
        """
        self.anchor = copy.deepcopy(actor).to(self.device)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.eval()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # ---- BEGIN SB3 2.7.1 SAC.train, verbatim except the # ANCHOR lines ----
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        anchor_kls: list[float] = []  # ANCHOR

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # ANCHOR: price leaving the clone manifold. KL between the
            # ANCHOR: pre-tanh Gaussians == KL between the squashed policies.
            # ANCHOR: MEASURED whenever an anchor exists — the beta=0 control
            # ANCHOR: is the arm whose drift the diagnosis is about, so it must
            # ANCHOR: report its real drift, not a fabricated zero — but only
            # ANCHOR: PRICED into the loss when beta > 0. At beta=0 the actor
            # ANCHOR: pass runs under no_grad, so it adds no gradient, draws no
            # ANCHOR: randomness, and (no BatchNorm in the SAC actor) mutates no
            # ANCHOR: module state: the arm stays bit-identical to plain SAC.
            if self.anchor is not None:  # ANCHOR
                priced = self.beta > 0.0  # ANCHOR
                with th.set_grad_enabled(priced):  # ANCHOR
                    mean, log_std, _ = self.actor.get_action_dist_params(  # ANCHOR
                        replay_data.observations  # ANCHOR
                    )  # ANCHOR
                with th.no_grad():  # ANCHOR
                    mean_a, log_std_a, _ = self.anchor.get_action_dist_params(  # ANCHOR
                        replay_data.observations  # ANCHOR
                    )  # ANCHOR
                kl = gaussian_kl(mean, log_std, mean_a, log_std_a).mean()  # ANCHOR
                anchor_kls.append(kl.item())  # ANCHOR
                if priced:  # ANCHOR
                    actor_loss = actor_loss + self.beta * kl  # ANCHOR
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # ANCHOR: recorded unconditionally so the key exists in every arm.
        anchor_kl = float(np.mean(anchor_kls)) if anchor_kls else 0.0  # ANCHOR
        self.logger.record("train/anchor_kl", anchor_kl)  # ANCHOR
        # ---- END SB3 2.7.1 SAC.train ----
