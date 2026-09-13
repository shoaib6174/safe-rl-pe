"""pe2.anchored_ppo — PPO whose loss carries a KL anchor to a frozen clone.

Spec: docs/pathway/cp_selection_anchor_prereg_2026-09-13.md (S2). D109: S1's
control arm — PPO-GAE at lambda=1 from the PPO clone — rose from 0.470 to
0.645 at 200k and then decayed to 0.246. The autopsy found no entropy
collapse (log-std -1.0 -> -2.0) and no reversal until late; what grows
without bound is KL(clone || pi): 0.6, 0.9, 1.2, 1.7, 2.9, 3.9, 4.9, 9.0,
15.2, 17.7. Capture is high while the KL sits in ~0.6-1.7 and gone past
~5-9. The remedy with a mechanism behind it is therefore a KL anchor to the
clone — D102's mechanism, now with a reason to expect the anchored region to
contain the improvement.

Mirror of `pe2.anchored_sac.KLAnchoredSAC`: `set_anchor` freezes a deep copy
of the policy; `train()` is SB3's `PPO.train` plus `beta * KL(pi_theta ||
pi_anchor)` on the rollout observations. At `beta == 0` the KL is MEASURED
and logged but not priced, and the update is bit-identical to plain PPO
(tested), so every unanchored arm gets its drift trace for free.

DELIBERATE, BOUNDED LIBRARY OVERRIDE
------------------------------------
As in `pe2.anchored_sac`, `train` is the installed library method (SB3
2.7.1) copied from source with the additions marked `# ANCHOR`. SB3 exposes
no seam for adding a term to the PPO loss — the loss is assembled inside
`train` and never exposed — so the copy is a documented exception rather
than a violation of "RL libraries are not rewritten". It is bounded: one
method, one module, deletable in a single `rm`. To re-verify against a
future SB3, diff this method against `ppo.py` and check that only the
`# ANCHOR` lines (and the BEGIN/END markers) differ. That check is
AUTOMATED and enforced on every test run — see `TestLibraryCopyIsFaithful`
in tests/pe2/test_anchored_ppo.py, which strips the marked lines and asserts
what remains parses to the installed `PPO.train`. It has to be PPO's own:
the sibling tripwire in tests/pe2/test_anchored_sac.py only ever compares
against `SAC.train`, so a release that changed `PPO.train` alone would leave
the suite green on a stale copy. pyproject pins no SB3 upper bound, so that
test is what stops an upgrade from silently confounding the S2 arms.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from pe2.anchored_sac import gaussian_kl


class KLAnchoredPPO(PPO):
    """PPO whose loss carries `beta * KL(pi_theta || pi_anchor)`.

    `beta = 0` (or no anchor) reduces this to SB3's PPO. With an anchor set,
    `beta = 0` still REPORTS `train/anchor_kl` — it just does not price it.
    """

    def __init__(self, *args, beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            # gSDE's distribution is not the diagonal action Gaussian that
            # gaussian_kl assumes; refuse rather than report a wrong KL.
            raise ValueError("KLAnchoredPPO does not support use_sde=True")
        if beta < 0.0:
            # The loss is only amended when beta > 0, so a negative weight
            # would train as plain PPO while the run reported itself as an
            # anchored arm. A mis-signed ablation must fail, not mislead.
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        self.beta = beta
        self.anchor = None
        self.last_anchor_kl: float | None = None

    def set_anchor(self, policy) -> None:
        """Freeze a deep copy of `policy` as the reference — a copy, so the
        live policy keeps training while the anchor stays the clone.

        `policy` must not have done a grad-enabled forward pass since its last
        `no_grad` one: SB3 caches the action distribution on
        `policy.action_dist.distribution`, and `copy.deepcopy` refuses a
        `Normal` whose parameters are still grad-tracking. Anchoring a freshly
        constructed or freshly loaded policy — what `finetune_ppo_anchored`
        does — is the intended use and is never affected.
        """
        self.anchor = copy.deepcopy(policy).to(self.device)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.set_training_mode(False)

    @staticmethod
    def _gaussian_params(policy, obs):
        d = policy.get_distribution(obs).distribution  # torch Normal
        return d.mean, th.log(d.stddev)

    def anchor_kl(self, obs, grad: bool):
        """Mean KL(pi_theta(.|s) || pi_anchor(.|s)) over `obs`. With
        `grad=True` the live policy's forward keeps its graph so the term can
        be priced; the anchor's forward never does."""
        with th.no_grad():
            mean_a, log_std_a = self._gaussian_params(self.anchor, obs)
        if grad:
            mean, log_std = self._gaussian_params(self.policy, obs)
        else:
            with th.no_grad():
                mean, log_std = self._gaussian_params(self.policy, obs)
        return gaussian_kl(mean, log_std, mean_a, log_std_a).mean()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # ---- BEGIN SB3 2.7.1 PPO.train, verbatim except the # ANCHOR lines ----
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_kls: list[float] = []  # ANCHOR

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                # ANCHOR: price leaving the clone manifold. MEASURED whenever
                # ANCHOR: an anchor exists — the beta=0 control is the arm
                # ANCHOR: whose drift D109 is about, so it must report its real
                # ANCHOR: drift, not a fabricated zero — but only PRICED into
                # ANCHOR: the loss when beta > 0. At beta=0 the policy's extra
                # ANCHOR: forward runs under no_grad, so it adds no gradient and
                # ANCHOR: draws no randomness: the arm stays bit-identical to
                # ANCHOR: plain PPO.
                if self.anchor is not None:  # ANCHOR
                    priced = self.beta > 0.0  # ANCHOR
                    obs = rollout_data.observations  # ANCHOR
                    kl = self.anchor_kl(obs, grad=priced)  # ANCHOR
                    anchor_kls.append(kl.item())  # ANCHOR
                    if priced:  # ANCHOR
                        loss = loss + self.beta * kl  # ANCHOR

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # ANCHOR: recorded unconditionally so the key exists in every arm, and
        # ANCHOR: also exported on the model: Logger.dump() clears
        # ANCHOR: name_to_value on SB3's own schedule, so a reader on a probe
        # ANCHOR: schedule can find the key missing; the attribute cannot be
        # ANCHOR: cleared underneath it. None means "not measured" (no anchor),
        # ANCHOR: which is not the same as a measured 0.0.
        anchor_kl = float(np.mean(anchor_kls)) if anchor_kls else 0.0  # ANCHOR
        self.logger.record("train/anchor_kl", anchor_kl)  # ANCHOR
        self.last_anchor_kl = anchor_kl if anchor_kls else None  # ANCHOR
        # ---- END SB3 2.7.1 PPO.train ----
