# CP Selection + KL Anchor (S2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `KLAnchoredPPO` (PPO with a KL anchor to the clone), the anchored fine-tune entry point, the zero-reward wrapper, the pre-registered selection and gate rules, and the S2 runner (bars → eleven arms → summary), so the round can run.

**Architecture:** `pe2/anchored_ppo.py` subclasses SB3 `PPO` exactly as `pe2/anchored_sac.py` subclasses `SAC`: a frozen deep-copied anchor and one added loss term `beta * KL(π_θ ‖ π_anchor)`, measured-but-unpriced at `beta = 0`. `pe2/exploitability.py` gains `finetune_ppo_anchored`; `pe2/rewards.py` gains `ZeroRewardWrapper`; `pe2/derisk.py` gains the two pre-registered rules `select_checkpoint` and `gate_pass`. `scripts/derisk_s2_cp_round.py` is the runner, modelled on `scripts/derisk_s1_structured.py`.

**Tech Stack:** Python 3.12, `./venv/bin/python`, stable-baselines3 2.7 (PPO), gymnasium, numpy, pytest (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), ruff.

**Spec:** `docs/pathway/cp_selection_anchor_prereg_2026-09-13.md` (binding; read it first).

## Global Constraints

- Always `./venv/bin/python`; tests as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest <files> -q`; lint `./venv/bin/ruff format <files> && ./venv/bin/ruff check <files>` clean before each commit.
- Pre-registered constants (spec §3–§4), never tunable: `TOTAL = 1_000_000`, `EVERY = 100_000`, `EVAL_N = 200`, `GATE_N = 500`, `SEL_SEED0 = 1_040_000`, `GATE_SEED0 = 1_050_000`, `SEEDS = (7, 8, 9)`, `BETAS = {"ANC01": 0.1, "ANC001": 0.01}`, `CP_LAMBDA = 1.0`, `TOTAL_LONG = 10_000_000`, `EVERY_LONG = 1_000_000`, `S1_EVAL_SEED0 = 1_010_000`, `P_GATE = 0.05`.
- Learner for every 1M arm = S1's CP configuration: the PPO clone `results/derisk/ppo_bc_searching_pursuer.zip` loaded with `custom_objects={"gae_lambda": 1.0}`; the anchor is a frozen copy of the loaded policy taken BEFORE any training step.
- Selection rule: highest probe `win_rate`, ties → earliest `steps`. Gate rule: exact two-sided McNemar p < 0.05 with `b > c` AND Wilson lower bound of the policy > the reference's point estimate, on the paired gate stream.
- `beta = 0` must be bit-identical to plain SB3 PPO (test), and must still record `train/anchor_kl` / `last_anchor_kl`.
- Never run any arm or `--measure-start`; the controller launches those. Existing artefacts under `results/derisk/` (S1, R1) must not be touched; tests redirect every output path into `tmp_path`.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Lazy SB3/torch imports inside functions in library code where the surrounding file does so; docstrings say *why* with decision numbers (D108, D109).

---

### Task 1: `pe2/anchored_ppo.py` — `KLAnchoredPPO`

**Files:**
- Create: `pe2/anchored_ppo.py`
- Test: `tests/pe2/test_anchored_ppo.py`

**Interfaces:**
- Consumes: `pe2.anchored_sac.gaussian_kl(mean, log_std, mean_ref, log_std_ref) -> Tensor` (sums over the action axis); SB3 `PPO`, whose `policy.get_distribution(obs).distribution` is a `torch.distributions.Normal` with `.mean` and `.stddev` (PPO's log-std is state-independent); SB3 `PPO.train` (copied, see Step 3).
- Produces: `class KLAnchoredPPO(PPO)` with `__init__(*args, beta: float = 0.0, **kwargs)`, `.beta`, `.anchor` (None until set), `.set_anchor(policy) -> None`, `.anchor_kl(obs: Tensor, grad: bool) -> Tensor` (scalar mean KL(π_θ ‖ π_anchor) over `obs`), `.last_anchor_kl: float | None`, logger key `train/anchor_kl`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_anchored_ppo.py
"""pe2.anchored_ppo — PPO with a KL anchor to a frozen clone (S2, D109)."""

import copy

import numpy as np
import pytest
import torch

from pe2.anchored_ppo import KLAnchoredPPO
from pe2.anchored_sac import gaussian_kl
from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.scripted import GreedyEvader

CFG = EnvConfig()
TINY = {"n_steps": 32, "batch_size": 16, "n_epochs": 1, "verbose": 0, "device": "cpu"}


def _env(seed):
    return SingleAgentWrapper(CFG, "pursuer", GreedyEvader, seed0=seed * 1000)


def _model(beta=0.0, seed=1, cls=KLAnchoredPPO, anchor=True, **kw):
    kwargs = dict(TINY, seed=seed, **kw)
    if cls is KLAnchoredPPO:
        kwargs["beta"] = beta
    m = cls("MlpPolicy", _env(seed), **kwargs)
    if anchor and cls is KLAnchoredPPO:
        m.set_anchor(m.policy)
    return m


def _obs_batch(m, n=64, seed=20260913):
    space = m.env.envs[0].observation_space
    space.seed(seed)
    return torch.as_tensor(np.stack([space.sample() for _ in range(n)]), dtype=torch.float32)


class TestKLAnchoredPPO:
    def test_set_anchor_is_a_frozen_detached_copy(self):
        m = _model(beta=1.0)
        assert m.anchor is not m.policy
        for (ka, pa), (kp, pp) in zip(
            m.anchor.state_dict().items(), m.policy.state_dict().items(), strict=True
        ):
            assert ka == kp
            assert torch.equal(pa, pp)
            assert pa is not pp
        assert all(not p.requires_grad for p in m.anchor.parameters())

    def test_anchor_kl_is_zero_before_any_update(self):
        m = _model(beta=1.0)
        obs = _obs_batch(m)
        assert m.anchor_kl(obs, grad=False).item() == pytest.approx(0.0, abs=1e-7)

    def test_anchor_kl_matches_gaussian_kl_on_the_policies(self):
        m = _model(beta=1.0)
        m.learn(total_timesteps=64)  # moves the policy off the anchor
        obs = _obs_batch(m)
        with torch.no_grad():
            d = m.policy.get_distribution(obs).distribution
            a = m.anchor.get_distribution(obs).distribution
            expected = gaussian_kl(
                d.mean, torch.log(d.stddev), a.mean, torch.log(a.stddev)
            ).mean().item()
        assert m.anchor_kl(obs, grad=False).item() == pytest.approx(expected, rel=1e-6)

    def test_beta_zero_is_bit_identical_to_plain_ppo(self):
        from stable_baselines3 import PPO

        def trained(cls, beta):
            m = _model(beta=beta, seed=11, cls=cls)
            m.learn(total_timesteps=64)
            return {k: v.clone() for k, v in m.policy.state_dict().items()}

        plain = trained(PPO, beta=None)
        anchored = trained(KLAnchoredPPO, beta=0.0)
        assert plain.keys() == anchored.keys()
        for k in plain:
            assert torch.equal(plain[k], anchored[k]), f"beta=0 diverged from PPO at {k}"

    def test_beta_zero_still_measures_the_kl(self):
        m = _model(beta=0.0)
        assert m.last_anchor_kl is None
        m.learn(total_timesteps=64)
        assert m.last_anchor_kl is not None and m.last_anchor_kl >= 0.0
        assert "train/anchor_kl" in m.logger.name_to_value

    def test_large_beta_keeps_the_policy_near_the_anchor(self):
        def drift(beta):
            m = _model(beta=beta, seed=3)
            m.learn(total_timesteps=32 * 6)
            return m.anchor_kl(_obs_batch(m), grad=False).item()

        assert drift(beta=1e3) < drift(beta=0.0)

    def test_priced_kl_changes_the_update(self):
        # Same seed, same data: beta=0 and beta=1e3 must end at different weights.
        def weights(beta):
            m = _model(beta=beta, seed=5)
            m.learn(total_timesteps=64)
            return {k: v.clone() for k, v in m.policy.state_dict().items()}

        w0, w1 = weights(0.0), weights(1e3)
        assert any(not torch.equal(w0[k], w1[k]) for k in w0)

    def test_no_anchor_trains_as_plain_ppo_and_reports_none(self):
        m = _model(beta=1.0, anchor=False)
        m.learn(total_timesteps=64)
        assert m.last_anchor_kl is None

    def test_rejects_gsde(self):
        with pytest.raises(ValueError, match="use_sde"):
            _model(beta=1.0, use_sde=True)

    def test_rejects_negative_beta(self):
        with pytest.raises(ValueError, match="beta"):
            _model(beta=-0.1)

    def test_checkpoint_loads_as_plain_ppo_for_evaluation(self, tmp_path):
        from stable_baselines3 import PPO

        m = _model(beta=0.1)
        m.learn(total_timesteps=64)
        m.save(tmp_path / "anc.zip")
        loaded = PPO.load(tmp_path / "anc.zip", device="cpu")
        obs = _obs_batch(m, n=4)
        a0, _ = m.predict(obs.numpy(), deterministic=True)
        a1, _ = loaded.predict(obs.numpy(), deterministic=True)
        np.testing.assert_allclose(a0, a1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_anchored_ppo.py -q`
Expected: FAIL at import — `No module named 'pe2.anchored_ppo'`.

- [ ] **Step 3: Write `pe2/anchored_ppo.py`**

`train()` is SB3 2.7's `PPO.train` (`venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py`, `def train`) copied verbatim with the lines marked `# ANCHOR` added. Copy from the installed file, do not retype from memory; the block below shows the whole method as it must read after the edit.

```python
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
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        self.beta = beta
        self.anchor = None
        self.last_anchor_kl: float | None = None

    def set_anchor(self, policy) -> None:
        """Freeze a deep copy of `policy` as the reference — a copy, so the
        live policy keeps training while the anchor stays the clone."""
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
        """SB3 2.7 `PPO.train`, verbatim, plus the lines marked ANCHOR."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_kls: list[float] = []  # ANCHOR

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if self.anchor is not None:  # ANCHOR: measured whenever set
                    priced = self.beta > 0.0  # ANCHOR
                    kl = self.anchor_kl(rollout_data.observations, grad=priced)  # ANCHOR
                    anchor_kls.append(kl.item())  # ANCHOR
                    if priced:  # ANCHOR: priced only when beta > 0
                        loss = loss + self.beta * kl  # ANCHOR

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

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
        anchor_kl = float(np.mean(anchor_kls)) if anchor_kls else 0.0  # ANCHOR
        self.logger.record("train/anchor_kl", anchor_kl)  # ANCHOR
        self.last_anchor_kl = anchor_kl if anchor_kls else None  # ANCHOR
```

If the installed `PPO.train` differs from the block above in any line NOT marked ANCHOR, the installed version wins: copy it and re-insert only the ANCHOR lines at the same positions. Note the ANCHOR block sits after `loss = ...` and before the `approx_kl_div` block.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_anchored_ppo.py -q`
Expected: all PASS. If `test_beta_zero_is_bit_identical_to_plain_ppo` fails, the unpriced path is consuming RNG or grad: the `beta = 0` measurement must run entirely under `no_grad` and must not touch the optimizer.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/anchored_ppo.py tests/pe2/test_anchored_ppo.py && ./venv/bin/ruff check pe2/anchored_ppo.py tests/pe2/test_anchored_ppo.py
git add pe2/anchored_ppo.py tests/pe2/test_anchored_ppo.py
git commit -m "feat(pe2): KLAnchoredPPO — PPO with a KL anchor to the clone (S2)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `finetune_ppo_anchored`, `ZeroRewardWrapper`, and the two pre-registered rules

**Files:**
- Modify: `pe2/exploitability.py` (append after `finetune_ppo`, ~line 560)
- Modify: `pe2/rewards.py` (append after `NegatedRewardWrapper`)
- Modify: `pe2/derisk.py` (append at the end)
- Test: `tests/pe2/test_exploitability.py` (append), `tests/pe2/test_rewards.py` (append), `tests/pe2/test_derisk.py` (append)

**Interfaces:**
- Consumes: `pe2.anchored_ppo.KLAnchoredPPO` (Task 1); `pe2.eval.mcnemar_pvalue(logs_a, logs_b, role) -> (p, b, c)`, `win_count(logs, role) -> (k, n)`, `wilson_ci(k, n) -> (lo, hi)`.
- Produces:
  - `pe2.exploitability.finetune_ppo_anchored(cfg, role, opponent_factory, model_path, total_timesteps, seed, beta, device="auto", callback=None, env_wrap=None, custom_objects=None) -> KLAnchoredPPO`
  - `pe2.rewards.ZeroRewardWrapper(gym.Wrapper)` — every reward 0.0, `info["true_reward"]` keeps the real one
  - `pe2.derisk.select_checkpoint(rows: list[dict]) -> dict` — the row with the highest `win_rate`, ties → smallest `steps`; `ValueError` on empty
  - `pe2.derisk.gate_pass(logs_policy, logs_ref, role="pursuer", p_gate=0.05) -> dict` with keys `pass, p, b, c, k, n, rate, wilson, ref_rate`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pe2/test_exploitability.py`:

```python
def test_finetune_ppo_anchored_anchors_the_loaded_clone_before_training(tmp_path):
    import torch
    from stable_baselines3 import PPO

    from pe2.anchored_ppo import KLAnchoredPPO
    from pe2.exploitability import PPO_KWARGS, SingleAgentWrapper, finetune_ppo_anchored
    from pe2.scripted import GreedyEvader

    cfg = EnvConfig()
    path = tmp_path / "clone.zip"
    PPO("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=1, verbose=0, device="cpu", **PPO_KWARGS).save(path)
    clone = PPO.load(path, device="cpu")
    m = finetune_ppo_anchored(
        cfg, "pursuer", GreedyEvader, path, total_timesteps=0, seed=1, beta=0.1,
        device="cpu", custom_objects={"gae_lambda": 1.0},
    )
    assert isinstance(m, KLAnchoredPPO) and m.beta == 0.1
    assert m.gae_lambda == 1.0 and m.rollout_buffer.gae_lambda == 1.0
    for (ka, pa), (kc, pc) in zip(m.anchor.state_dict().items(), clone.policy.state_dict().items(), strict=True):
        assert ka == kc and torch.equal(pa, pc)


def test_finetune_ppo_anchored_env_wrap_wraps_training_only(tmp_path):
    from stable_baselines3 import PPO

    from pe2.exploitability import PPO_KWARGS, SingleAgentWrapper, finetune_ppo_anchored
    from pe2.rewards import ZeroRewardWrapper
    from pe2.scripted import GreedyEvader

    cfg = EnvConfig()
    path = tmp_path / "clone.zip"
    PPO("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=1, verbose=0, device="cpu", **PPO_KWARGS).save(path)
    m = finetune_ppo_anchored(
        cfg, "pursuer", GreedyEvader, path, total_timesteps=0, seed=1, beta=0.0,
        device="cpu", env_wrap=ZeroRewardWrapper,
    )
    assert isinstance(m.get_env().envs[0].env, ZeroRewardWrapper) or isinstance(
        m.get_env().envs[0], ZeroRewardWrapper
    )


def test_finetune_ppo_anchored_rejects_negative_beta(tmp_path):
    from stable_baselines3 import PPO

    from pe2.exploitability import PPO_KWARGS, SingleAgentWrapper, finetune_ppo_anchored
    from pe2.scripted import GreedyEvader

    cfg = EnvConfig()
    path = tmp_path / "clone.zip"
    PPO("MlpPolicy", SingleAgentWrapper(cfg, "pursuer", GreedyEvader), seed=1, verbose=0, device="cpu", **PPO_KWARGS).save(path)
    with pytest.raises(ValueError, match="beta"):
        finetune_ppo_anchored(cfg, "pursuer", GreedyEvader, path, total_timesteps=0, seed=1, beta=-1.0, device="cpu")
```

(`pytest` and `EnvConfig` are already imported at the top of that file; check and add if not.)

Append to `tests/pe2/test_rewards.py`:

```python
class TestZeroRewardWrapper:
    def test_every_reward_is_zero_and_the_true_reward_is_kept(self):
        from pe2.rewards import ZeroRewardWrapper

        w = ZeroRewardWrapper(SingleAgentWrapper(CFG, "pursuer", GreedyEvader))
        obs, _ = w.reset(seed=900_800)
        seen_nonzero_true = False
        while True:
            obs, r, term, trunc, info = w.step(np.array([1.0, 0.0], dtype=np.float32))
            assert r == 0.0
            assert "true_reward" in info
            seen_nonzero_true |= info["true_reward"] != 0.0
            if term or trunc:
                break
        assert seen_nonzero_true  # the terminal +/-1 was paid by the env, hidden by the wrapper
```

Append to `tests/pe2/test_derisk.py`:

```python
class TestSelectCheckpoint:
    def test_highest_win_rate_wins(self):
        from pe2.derisk import select_checkpoint

        rows = [{"steps": 100, "win_rate": 0.5}, {"steps": 200, "win_rate": 0.7}, {"steps": 300, "win_rate": 0.6}]
        assert select_checkpoint(rows)["steps"] == 200

    def test_ties_go_to_the_earliest(self):
        from pe2.derisk import select_checkpoint

        rows = [{"steps": 100, "win_rate": 0.5}, {"steps": 200, "win_rate": 0.7}, {"steps": 300, "win_rate": 0.7}]
        assert select_checkpoint(rows)["steps"] == 200

    def test_empty_is_an_error(self):
        from pe2.derisk import select_checkpoint

        with pytest.raises(ValueError):
            select_checkpoint([])


def _logs(wins, n, seed0=0):
    return [{"seed": seed0 + i, "role": "pursuer", "outcome": "capture" if i < wins else "timeout"} for i in range(n)]


class TestGatePass:
    def test_passes_when_separated_and_significant(self):
        from pe2.derisk import gate_pass

        policy = _logs(300, 500)
        ref = _logs(235, 500)  # the policy wins every episode the ref wins, plus 65 more
        g = gate_pass(policy, ref)
        assert g["pass"] is True and g["p"] < 0.05 and g["b"] == 65 and g["c"] == 0
        assert g["rate"] == 0.6 and g["ref_rate"] == 0.47 and g["wilson"][0] > 0.47

    def test_fails_on_significance_alone(self):
        from pe2.derisk import gate_pass

        policy = _logs(240, 500)
        ref = _logs(235, 500)
        g = gate_pass(policy, ref)
        assert g["pass"] is False and g["p"] > 0.05

    def test_fails_when_wilson_lower_bound_does_not_clear_the_reference(self):
        from pe2.derisk import gate_pass

        # 20 discordant wins, 0 losses: p ~ 2e-6, but lo(255/500) = 0.466 < 0.470
        policy = _logs(255, 500)
        ref = _logs(235, 500)
        g = gate_pass(policy, ref)
        assert g["p"] < 0.05 and g["pass"] is False

    def test_fails_when_the_reference_wins_more_discordant_pairs(self):
        from pe2.derisk import gate_pass

        policy = _logs(235, 500)
        ref = _logs(300, 500)
        g = gate_pass(policy, ref)
        assert g["pass"] is False and g["c"] > g["b"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py -k anchored tests/pe2/test_rewards.py -k Zero tests/pe2/test_derisk.py -k "SelectCheckpoint or GatePass" -q`
Expected: FAIL with `ImportError` / `AttributeError` for the four new names.

- [ ] **Step 3: Implement**

Append to `pe2/exploitability.py` after `finetune_ppo`:

```python
def finetune_ppo_anchored(
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    model_path,
    total_timesteps: int,
    seed: int,
    beta: float,
    device: str = "auto",
    callback=None,
    env_wrap=None,
    custom_objects: dict | None = None,
):
    """S2 (D109): `finetune_ppo` with a KL anchor to the clone. The saved PPO
    clone is loaded into a `KLAnchoredPPO` (any `custom_objects`, e.g. the
    round's `gae_lambda`, override the saved values exactly as in
    `finetune_ppo`), the anchor is snapshotted from the loaded policy BEFORE
    any training step, and the loss carries `beta * KL(pi_theta || pi_clone)`.
    `beta = 0` measures the KL and is bit-identical to `finetune_ppo`.
    `env_wrap`, if given, wraps the TRAINING env only (the NULL arm's
    `ZeroRewardWrapper`); evaluation stays unshaped."""
    from pe2.anchored_ppo import KLAnchoredPPO

    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta!r}")
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory, seed0=seed * 1000)
    env = env_wrap(wrapper) if env_wrap is not None else wrapper
    model = KLAnchoredPPO.load(
        model_path, env=env, device=device, seed=seed, custom_objects=custom_objects
    )
    model.beta = beta  # load() constructs with the default beta; set the round's
    model.set_anchor(model.policy)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model
```

Append to `pe2/rewards.py`:

```python
class ZeroRewardWrapper(gym.Wrapper):
    """TRAINING-ONLY: every reward is 0.0 — S2's negative control for the
    selection rule. A run with no signal, probed ten times and "selected",
    measures how much improvement best-of-ten manufactures by itself.
    `info["true_reward"]` keeps what the env paid. Evaluation never uses this
    wrapper (the probes build their own unshaped `SingleAgentWrapper`)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, 0.0, terminated, truncated, dict(info, true_reward=reward)
```

Append to `pe2/derisk.py`:

```python
def select_checkpoint(rows: list[dict]) -> dict:
    """S2 §3 selection rule, fixed before the run: the probe row with the
    highest `win_rate`; ties go to the EARLIEST `steps`. Applied identically
    to every arm, including the zero-reward NULL control that measures what
    this rule manufactures on its own."""
    if not rows:
        raise ValueError("select_checkpoint: no probe rows")
    return max(rows, key=lambda r: (r["win_rate"], -r["steps"]))


def gate_pass(
    logs_policy: list[dict],
    logs_ref: list[dict],
    role: str = "pursuer",
    p_gate: float = 0.05,
) -> dict:
    """S2 §3 gate rule on the PAIRED gate stream: PASS iff the exact two-sided
    McNemar test has p < p_gate with more discordant wins than losses
    (b > c) AND the policy's Wilson lower bound exceeds the reference's point
    estimate. Both conditions, so a pass is significant AND separated."""
    from pe2.eval import mcnemar_pvalue, wilson_ci, win_count

    p, b, c = mcnemar_pvalue(logs_policy, logs_ref, role)
    k, n = win_count(logs_policy, role)
    kr, nr = win_count(logs_ref, role)
    lo, hi = wilson_ci(k, n)
    return {
        "pass": bool(p < p_gate and b > c and lo > kr / nr),
        "p": p,
        "b": b,
        "c": c,
        "k": k,
        "n": n,
        "rate": k / n,
        "wilson": [lo, hi],
        "ref_rate": kr / nr,
    }
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py tests/pe2/test_rewards.py tests/pe2/test_derisk.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/exploitability.py pe2/rewards.py pe2/derisk.py tests/pe2/test_exploitability.py tests/pe2/test_rewards.py tests/pe2/test_derisk.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/exploitability.py pe2/rewards.py pe2/derisk.py tests/pe2/test_exploitability.py tests/pe2/test_rewards.py tests/pe2/test_derisk.py
git commit -m "feat: finetune_ppo_anchored, ZeroRewardWrapper, select_checkpoint, gate_pass (S2)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/derisk_s2_cp_round.py` — bars, eleven arms, summary

**Files:**
- Create: `scripts/derisk_s2_cp_round.py`
- Test: `tests/pe2/test_derisk_s2_round.py`

**Interfaces:**
- Consumes: Task 1–2 names above; `pe2.exploitability.CheckpointEvalProbe(cfg, role, opp, every, eval_n, seed0, ckpt_dir, report_path, label)` (rows have `steps, ckpt, win_rate, k, n`), `evaluate_sb3_policy`; `pe2.anchored_sac.AnchorKLProbe(rows, every)` (stamps `row["anchor_kl"]`; must be placed AFTER the checkpoint probe in the `CallbackList`); `pe2.structured` (`SearchMDP`, `StructuredEvalProbe`, `evaluate_structured`, `evaluate_scripted`, `HOLD`); `pe2.derisk.structured_verdict`.
- Produces: CLI `--measure-start | --arms <ARM...> | --summarize [--device cpu]`; module-level `ARMS`, `select_and_gate`, `hypothesis_verdicts`, `direction`, `summarize`; artefacts per spec §8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk_s2_round.py
"""scripts/derisk_s2_cp_round.py — the S2 round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s2_cp_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s2_cp_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, sel_pass, fin_pass, kind="ppo"):
    return {
        "arm": name,
        "kind": kind,
        "selected": {"gate": {"PPO_CLONE": {"pass": sel_pass}, "TEACHER": {"pass": False}}},
        "final": {"gate": {"PPO_CLONE": {"pass": fin_pass}, "TEACHER": {"pass": False}}},
        "checkpoints": [],
    }


class TestProtocol:
    def test_pre_registered_constants(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.GATE_N) == (1_000_000, 100_000, 200, 500)
        assert (R.SEL_SEED0, R.GATE_SEED0, R.S1_EVAL_SEED0) == (1_040_000, 1_050_000, 1_010_000)
        assert R.SEEDS == (7, 8, 9) and R.BETAS == {"ANC01": 0.1, "ANC001": 0.01}
        assert R.CP_LAMBDA == 1.0 and R.P_GATE == 0.05
        assert (R.TOTAL_LONG, R.EVERY_LONG) == (10_000_000, 1_000_000)

    def test_arm_table(self):
        names = set(R.ARMS)
        assert names == {"SEL7", "SEL8", "SEL9", "ANC01_7", "ANC01_8", "ANC01_9",
                         "ANC001_7", "ANC001_8", "ANC001_9", "NULL7", "SBLONG"}
        assert R.ARMS["SEL8"] == {"kind": "ppo", "beta": 0.0, "seed": 8, "zero_reward": False}
        assert R.ARMS["ANC01_9"] == {"kind": "ppo", "beta": 0.1, "seed": 9, "zero_reward": False}
        assert R.ARMS["ANC001_7"] == {"kind": "ppo", "beta": 0.01, "seed": 7, "zero_reward": False}
        assert R.ARMS["NULL7"] == {"kind": "ppo", "beta": 0.0, "seed": 7, "zero_reward": True}
        assert R.ARMS["SBLONG"] == {"kind": "sblong", "seed": 7}

    def test_arms_refuse_without_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()

    def test_measure_start_refuses_to_re_measure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s2_start.json")
        (tmp_path / "s2_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()

    def test_an_arm_refuses_to_run_twice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s2_arm_SEL7.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("SEL7")
        (tmp_path / "ckpts_ANC01_8").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("ANC01_8")
        R.require_arm_not_run("SEL9")  # clean: no raise


class TestHypotheses:
    def test_h_sel_needs_two_of_three_selected_passes(self):
        res = {"SEL7": _arm("SEL7", True, False), "SEL8": _arm("SEL8", True, False), "SEL9": _arm("SEL9", False, False), "NULL7": _arm("NULL7", False, False)}
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"] == "PASS" and h["H_SEL_passes"] == 2

    def test_h_anc_uses_final_checkpoints_per_beta(self):
        res = {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res |= {f"ANC001_{s}": _arm(f"ANC001_{s}", True, False) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", False, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_ANC"]["ANC01"] == "PASS" and h["H_ANC"]["ANC001"] == "FAIL"

    def test_null_pass_makes_h_sel_undetermined_only(self):
        res = {f"SEL{s}": _arm(f"SEL{s}", True, False) for s in (7, 8, 9)}
        res |= {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", True, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"].startswith("UNDETERMINED") and h["H_ANC"]["ANC01"] == "PASS"

    def test_missing_arms_are_incomplete(self):
        h = R.hypothesis_verdicts({"SEL7": _arm("SEL7", True, False)})
        assert h["H_SEL"] == "INCOMPLETE"


class TestDirection:
    def test_anc_pass_is_the_artefact(self):
        d = R.direction({"H_SEL": "FAIL", "H_ANC": {"ANC01": "PASS", "ANC001": "FAIL"}, "NULL": "FAIL"})
        assert "no selection" in d and "Route C withdrawn" in d

    def test_sel_only(self):
        d = R.direction({"H_SEL": "PASS", "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"}, "NULL": "FAIL"})
        assert "selection-based" in d and "needs the gate" in d

    def test_both_fail(self):
        d = R.direction({"H_SEL": "FAIL", "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"}, "NULL": "FAIL"})
        assert "seed-7-specific" in d

    def test_null_pass(self):
        d = R.direction({"H_SEL": "UNDETERMINED (selection manufactures improvement)", "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"}, "NULL": "PASS"})
        assert "manufactures" in d

    def test_incomplete(self):
        assert "incomplete" in R.direction({"H_SEL": "INCOMPLETE", "H_ANC": {"ANC01": "INCOMPLETE", "ANC001": "INCOMPLETE"}, "NULL": "INCOMPLETE"})


class TestDeterminism:
    def test_reproduced_when_curves_match(self):
        rows = [{"steps": s, "win_rate": r} for s, r in zip(range(100_000, 1_000_001, 100_000), [0.64, 0.645, 0.585, 0.605, 0.51, 0.55, 0.56, 0.38, 0.285, 0.265], strict=True)]
        assert R.determinism_note(rows, rows) == "reproduced"
        other = [dict(r) for r in rows]
        other[3]["win_rate"] = 0.6
        assert R.determinism_note(rows, other).startswith("not reproduced")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s2_round.py -q`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the runner**

```python
# scripts/derisk_s2_cp_round.py
"""S2 — the CP lead: is the mid-run improvement real, selectable and holdable?
Spec: docs/pathway/cp_selection_anchor_prereg_2026-09-13.md. Nothing here may
change a threshold, a seed, a rule or a budget after a number is seen.

Arms (1M env steps each unless noted; S1's CP config = PPO_KWARGS + lambda=1.0
from the PPO clone of SearchingPursuer):
  SEL{7,8,9}     beta=0 (KL measured only)   <- H-SEL: reproduce, select, gate
  ANC01_{7,8,9}  beta=0.1                     <- H-ANC at the stronger weight
  ANC001_{7,8,9} beta=0.01                    <- H-ANC at the weaker weight
  NULL7          beta=0, all rewards zeroed   <- negative control for SELECTION
  SBLONG         S1's SB for 10M env steps    <- side arm, no claim (amendment A1)

Usage:
  ./venv/bin/python scripts/derisk_s2_cp_round.py --measure-start   (bars on the gate stream)
  ./venv/bin/python scripts/derisk_s2_cp_round.py --arms SEL7       (one process per arm)
  ./venv/bin/python scripts/derisk_s2_cp_round.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass, select_checkpoint, structured_verdict
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader, SearchingPursuer

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s2_start.json"
PPO_CLONE = OUT / "ppo_bc_searching_pursuer.zip"
SAC_CLONE = OUT / "sac_bc_searching_pursuer.zip"
S1_INIT = OUT / "s1_ppo_init.zip"
S1_START = OUT / "s1_start.json"
S1_CP_PROBE = OUT / "s1_probe_CP.json"

# --- pre-registered (spec sections 3-5). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
GATE_N = 500
SEL_SEED0 = 1_040_000  # selection/probe stream (fresh)
GATE_SEED0 = 1_050_000  # gate stream (fresh)
S1_EVAL_SEED0 = 1_010_000  # SBLONG probes on S1's stream so S1's bars apply
SEEDS = (7, 8, 9)
BETAS = {"ANC01": 0.1, "ANC001": 0.01}
CP_LAMBDA = 1.0
P_GATE = 0.05
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
REFS = ("PPO_CLONE", "SAC_CLONE", "TEACHER")

ARMS: dict = {}
for _s in SEEDS:
    ARMS[f"SEL{_s}"] = {"kind": "ppo", "beta": 0.0, "seed": _s, "zero_reward": False}
    for _name, _b in BETAS.items():
        ARMS[f"{_name}_{_s}"] = {"kind": "ppo", "beta": _b, "seed": _s, "zero_reward": False}
ARMS["NULL7"] = {"kind": "ppo", "beta": 0.0, "seed": 7, "zero_reward": True}
ARMS["SBLONG"] = {"kind": "sblong", "seed": 7}


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(
            f"{START_PATH} already exists: the bars are already measured and the stop "
            "rule (spec section 6) forbids re-measuring them."
        )
    if any(OUT.glob("s2_arm_*.json")) or any((OUT / f"ckpts_{a}").exists() for a in ARMS):
        raise SystemExit("S2 arms have already run; the stop rule forbids re-measuring.")


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s2_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run (s2_arm_{arm}.json or ckpts_{arm} exists); "
            "the stop rule (spec section 6) forbids re-running an arm."
        )


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"gate bars missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it before any arm trains."
        )
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# bars on the gate stream (spec section 3), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO, SAC

    from pe2.exploitability import evaluate_sb3_policy
    from pe2.structured import evaluate_scripted

    require_not_measured()
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N}
    refs = {
        "PPO_CLONE": lambda: evaluate_sb3_policy(
            PPO.load(PPO_CLONE, device=device), cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0
        ),
        "SAC_CLONE": lambda: evaluate_sb3_policy(
            SAC.load(SAC_CLONE, device=device), cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0
        ),
        "TEACHER": lambda: evaluate_scripted(cfg, SearchingPursuer, GreedyEvader, GATE_N, GATE_SEED0),
    }
    for name, fn in refs.items():
        logs = fn()
        out[name] = _ci(*win_count(logs, "pursuer")) | {"logs": logs}
        lo, hi = out[name]["wilson"]
        print(f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f}, Wilson [{lo:.4f}, {hi:.4f}]", flush=True)
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def select_and_gate(rows: list[dict], final_ckpt, starts: dict, cfg, device: str) -> dict:
    """Spec section 3: the selected checkpoint AND the final one, each gated
    on the gate stream against every reference. Returns the two records."""
    from stable_baselines3 import PPO

    from pe2.exploitability import evaluate_sb3_policy

    def gate(ckpt_path) -> dict:
        model = PPO.load(ckpt_path, device=device)
        logs = evaluate_sb3_policy(model, cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0)
        return {
            "ckpt": str(ckpt_path),
            "score": _ci(*win_count(logs, "pursuer")),
            "gate": {ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE) for ref in REFS},
        }

    sel_row = select_checkpoint(rows)
    return {
        "selected": gate(sel_row["ckpt"]) | {"steps": sel_row["steps"], "probe_win_rate": sel_row["win_rate"]},
        "final": gate(final_ckpt),
    }


def run_ppo_arm(arm: str, starts: dict, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3.common.callbacks import CallbackList

    from pe2.anchored_sac import AnchorKLProbe
    from pe2.exploitability import CheckpointEvalProbe, finetune_ppo_anchored
    from pe2.rewards import ZeroRewardWrapper

    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = CheckpointEvalProbe(
        cfg, "pursuer", GreedyEvader, every=EVERY, eval_n=EVAL_N, seed0=SEL_SEED0,
        ckpt_dir=d, report_path=OUT / f"s2_probe_{arm}.json", label=arm,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    kl_probe = AnchorKLProbe(probe.rows, EVERY)  # AFTER the checkpoint probe, same `every`
    finetune_ppo_anchored(
        cfg, "pursuer", GreedyEvader, PPO_CLONE, TOTAL, seed=spec["seed"], beta=spec["beta"],
        device=device, callback=CallbackList([probe.callback, kl_probe.callback]),
        env_wrap=ZeroRewardWrapper if spec["zero_reward"] else None,
        custom_objects={"gae_lambda": CP_LAMBDA},
    )
    result = {
        "arm": arm, "kind": "ppo", "settings": spec, "timesteps": TOTAL, "eval_n": EVAL_N,
        "sel_seed0": SEL_SEED0, "gate_seed0": GATE_SEED0, "checkpoints": probe.rows,
    }
    final_ckpt = d / f"{arm}_{TOTAL}.zip"
    if not final_ckpt.exists():
        result["verdict"] = "UNDETERMINED"
        result["reason"] = f"final checkpoint missing: {final_ckpt} was never written"
        return result
    result |= select_and_gate(probe.rows, final_ckpt, starts, cfg, device)
    sel, fin = result["selected"]["gate"]["PPO_CLONE"], result["final"]["gate"]["PPO_CLONE"]
    result["verdict"] = f"selected {'PASS' if sel['pass'] else 'FAIL'} / final {'PASS' if fin['pass'] else 'FAIL'} vs PPO clone"
    result["reason"] = (
        f"selected @{result['selected']['steps']} (probe {result['selected']['probe_win_rate']:.3f}): gate "
        f"{sel['k']}/{sel['n']} = {sel['rate']:.3f}, McNemar p={sel['p']:.2e} (b={sel['b']}, c={sel['c']}) vs clone {sel['ref_rate']:.3f}; "
        f"final: {fin['k']}/{fin['n']} = {fin['rate']:.3f}, p={fin['p']:.2e} (b={fin['b']}, c={fin['c']}); "
        f"vs teacher: selected {'PASS' if result['selected']['gate']['TEACHER']['pass'] else 'FAIL'}, "
        f"final {'PASS' if result['final']['gate']['TEACHER']['pass'] else 'FAIL'}"
    )
    return result


def run_sblong(device: str, holder: dict | None = None) -> dict:
    """Side arm: S1's SB (terminal only, structured) for TOTAL_LONG env steps,
    probed on S1's stream so S1's bars in s1_start.json apply. No claim."""
    from stable_baselines3 import PPO

    from pe2.structured import HOLD, SearchMDP, StructuredEvalProbe, evaluate_structured

    if not S1_INIT.exists() or not S1_START.exists():
        raise SystemExit("SBLONG needs S1's s1_ppo_init.zip and s1_start.json")
    s1 = json.loads(S1_START.read_text())
    cfg = EnvConfig()
    arm = "SBLONG"
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg, GreedyEvader, every=EVERY_LONG, total=TOTAL_LONG, eval_n=EVAL_N, seed0=S1_EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"s2_probe_{arm}.json", label=arm, hold=HOLD,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(cfg, GreedyEvader, seed0=7 * 1000, hold=HOLD, r_acq=0.0, zero_reward=False)
    model = PPO.load(S1_INIT, env=env, device=device, seed=7)
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm, "kind": "sblong", "settings": ARMS[arm], "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps), "eval_seed0": S1_EVAL_SEED0, "checkpoints": probe.rows,
        "s1_bars": {k: s1[k]["rate"] for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S")},
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not ckpt.exists():
        result["verdict"], result["reason"] = "UNDETERMINED", f"final checkpoint missing: {ckpt}"
        return result
    logs = evaluate_structured(PPO.load(ckpt, device=device), cfg, GreedyEvader, GATE_N, S1_EVAL_SEED0, hold=HOLD)
    result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
    result["verdict"], result["reason"] = structured_verdict(
        result["final"]["k"], result["final"]["n"], s1["BAR_ORBIT"]["rate"],
        s1["FLOOR_RAND"]["wilson"][1], s1["START_S"]["wilson"][1],
    )
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()
    require_arm_not_run(arm)
    holder: dict = {}
    try:
        result = run_sblong(device, holder) if ARMS[arm]["kind"] == "sblong" else run_ppo_arm(arm, starts, device, holder)
    except Exception as e:
        (OUT / f"s2_arm_{arm}.json").write_text(json.dumps({
            "arm": arm, "kind": ARMS[arm]["kind"], "verdict": "UNDETERMINED",
            "reason": f"crashed: {type(e).__name__}: {e}", "checkpoints": holder.get("rows", []),
        }, indent=2))
        raise
    (OUT / f"s2_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# summary (spec sections 5-6), fixed before the run
# ----------------------------------------------------------------------------
def _passes(results: dict, arms: list[str], which: str) -> int | None:
    if any(a not in results or which not in results[a] for a in arms):
        return None
    return sum(bool(results[a][which]["gate"]["PPO_CLONE"]["pass"]) for a in arms)


def hypothesis_verdicts(results: dict) -> dict:
    sel_arms = [f"SEL{s}" for s in SEEDS]
    n_sel = _passes(results, sel_arms, "selected")
    null = results.get("NULL7")
    null_v = "INCOMPLETE" if null is None or "selected" not in null else (
        "PASS" if null["selected"]["gate"]["PPO_CLONE"]["pass"] else "FAIL"
    )
    if n_sel is None:
        h_sel = "INCOMPLETE"
    elif null_v == "PASS":
        h_sel = "UNDETERMINED (selection manufactures improvement)"
    else:
        h_sel = "PASS" if n_sel >= 2 else "FAIL"
    h_anc = {}
    for name in BETAS:
        n = _passes(results, [f"{name}_{s}" for s in SEEDS], "final")
        h_anc[name] = "INCOMPLETE" if n is None else ("PASS" if n >= 2 else "FAIL")
    return {"H_SEL": h_sel, "H_SEL_passes": n_sel, "H_ANC": h_anc, "NULL": null_v}


def determinism_note(rows_new: list[dict], rows_s1: list[dict]) -> str:
    a = [(r["steps"], round(r["win_rate"], 6)) for r in rows_new]
    b = [(r["steps"], round(r["win_rate"], 6)) for r in rows_s1]
    if a == b:
        return "reproduced"
    diff = [s for (s, x), (_, y) in zip(a, b, strict=False) if x != y]
    return f"not reproduced (first difference at {diff[0] if diff else 'length'})"


def direction(h: dict) -> str:
    anc_pass = [n for n, v in h["H_ANC"].items() if v == "PASS"]
    if h["H_SEL"] == "INCOMPLETE" or any(v == "INCOMPLETE" for v in h["H_ANC"].values()):
        return "incomplete: all SEL, ANC and NULL arms must run"
    if anc_pass:
        return (
            f"H-ANC PASSES at {', '.join(anc_pass)} (beta {[BETAS[n] for n in anc_pass]}): a learned pursuer that "
            "improves on its clone with no selection exists. The artefact for the paper is the anchored PPO "
            "pursuer; if it also beats the teacher that is the headline number. Route C withdrawn. Next "
            "pre-registration: the shield on this pursuer (Step 8). Prefer the beta whose final checkpoint "
            f"scores higher; report both. H-SEL: {h['H_SEL']}."
        )
    if h["H_SEL"] == "PASS":
        return (
            "H-SEL PASSES, H-ANC fails at both beta: the rise is real and reproducible but the anchor did not "
            "hold it. Report the selection-based artefact honestly (it needs the gate) and pre-register the "
            "next remedy from the ANC drift traces. Route C withdrawn on H-SEL."
        )
    if h["NULL"] == "PASS":
        return (
            "NULL PASSES: selection manufactures improvement at this probe density; H-SEL is not read; "
            "H-ANC failed at both beta, so no learned pursuer is claimed. Route C stands re-scoped."
        )
    return (
        "Both fail, NULL fails: the 0.645 was seed-7-specific or a stream artefact. Report it as such; "
        "Route C stands re-scoped (D108/D109). SBLONG is read on its own."
    )


def summarize(results: dict, starts: dict) -> dict:
    h = hypothesis_verdicts(results)
    d = direction(h)
    note = None
    if "SEL7" in results and S1_CP_PROBE.exists():
        note = determinism_note(results["SEL7"]["checkpoints"], json.loads(S1_CP_PROBE.read_text())["checkpoints"])
    sb = results.get("SBLONG")
    return {
        "spec": "docs/pathway/cp_selection_anchor_prereg_2026-09-13.md",
        "hypotheses": h,
        "bars": {k: starts[k]["rate"] for k in REFS},
        "arms": {a: {"verdict": r.get("verdict"), "reason": r.get("reason")} for a, r in results.items()},
        "determinism_sel7_vs_s1_cp": note,
        "sblong": None if sb is None else {"verdict": sb.get("verdict"), "reason": sb.get("reason")},
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }


def load_results() -> dict:
    return {p.stem.removeprefix("s2_arm_"): json.loads(p.read_text()) for p in sorted(OUT.glob("s2_arm_*.json"))}


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fam_color = {"SEL": "tab:blue", "ANC01": "tab:green", "ANC001": "tab:olive", "NULL": "tab:red"}
    ax = axes[0]
    for arm, r in results.items():
        if r.get("kind") != "ppo" or not r.get("checkpoints"):
            continue
        fam = arm.rsplit("_", 1)[0] if "_" in arm else arm[:-1]
        rows = r["checkpoints"]
        ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker=".", color=fam_color.get(fam, "grey"), alpha=0.8, label=arm)
        if "selected" in r:
            ax.plot([r["selected"]["steps"]], [r["selected"]["probe_win_rate"]], marker="*", ms=12, color=fam_color.get(fam, "grey"))
    for ref, c in zip(REFS, ("tab:red", "tab:purple", "k"), strict=True):
        ax.axhline(starts[ref]["rate"], ls=":", color=c, lw=1.2, label=f"{ref} (gate) = {starts[ref]['rate']:.3f}")
    ax.set_xlabel("env steps"); ax.set_ylabel(f"probe win rate (selection seeds {SEL_SEED0}+)"); ax.set_title("S2 — 1M arms (stars = selected)"); ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    for arm, r in results.items():
        if r.get("kind") != "ppo":
            continue
        rows = [x for x in r.get("checkpoints", []) if x.get("anchor_kl") is not None]
        if rows:
            fam = arm.rsplit("_", 1)[0] if "_" in arm else arm[:-1]
            ax.plot([x["steps"] for x in rows], [x["anchor_kl"] for x in rows], marker=".", color=fam_color.get(fam, "grey"), alpha=0.8, label=arm)
    ax.set_yscale("log"); ax.set_xlabel("env steps"); ax.set_ylabel("KL(pi || clone), training batches"); ax.set_title("anchor divergence"); ax.legend(fontsize=6, ncol=2)
    ax = axes[2]
    sb = results.get("SBLONG")
    if sb and sb.get("checkpoints"):
        rows = sb["checkpoints"]
        ax.plot([x["steps"] for x in rows], [x["win_rate"] for x in rows], marker="o", color="tab:orange", label=f"SBLONG ({sb.get('verdict')})")
        for k, c in (("BAR_ORBIT", "tab:green"), ("FLOOR_RAND", "tab:orange"), ("START_S", "tab:purple")):
            ax.axhline(sb["s1_bars"][k], ls=":", color=c, lw=1.2, label=f"{k} = {sb['s1_bars'][k]:.3f}")
    ax.set_xlabel("env steps"); ax.set_title("SBLONG vs S1 bars (side arm)"); ax.legend(fontsize=7)
    fig.suptitle("S2 — CP selection + KL anchor" + ("" if summary["complete"] else " — INCOMPLETE"))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        summary = summarize(results, starts)
        (OUT / "s2_round.json").write_text(json.dumps({"summary": summary, "arms": results}, indent=2))
        write_figure(results, starts, summary, OUT / "s2_round.png")
        print(json.dumps(summary["hypotheses"], indent=2))
        print(json.dumps(summary["arms"], indent=2))
        print(f"\ndeterminism SEL7 vs S1 CP: {summary['determinism_sel7_vs_s1_cp']}")
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
```

Notes for the implementer: `evaluate_sb3_policy` takes `n_episodes` and `seed0` as keyword-or-positional in that order (`model, cfg, role, opponent_factory, n_episodes, seed0`); `CheckpointEvalProbe` names its final checkpoint `{label}_{TOTAL}.zip` because PPO's `num_timesteps` hits every multiple of `EVERY` exactly with one env; `AnchorKLProbe` raises if it is placed before the checkpoint probe or given a different `every`. `starts[ref]["logs"]` are the paired gate logs saved by `measure_start`; `gate_pass` pairs by `seed`.

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk_s2_round.py -q`
Expected: all PASS.

- [ ] **Step 5: Smoke the wiring at a tiny budget in a scratch directory (no real artefacts)**

Write and run a throwaway script under `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/` that imports the runner via `importlib`, monkeypatches `OUT`, `START_PATH` to a scratch dir and `TOTAL=4000, EVERY=2000, EVAL_N=4, GATE_N=6`, runs `measure_start("cpu")`, then `run_arm("SEL7","cpu")`, `run_arm("ANC01_7","cpu")`, `run_arm("NULL7","cpu")`, then `summarize(load_results(), load_start())` and `write_figure(...)`. Expected: no exception; each `s2_arm_*.json` has `selected` and `final` with three gates; `anchor_kl` present on every probe row; the figure file exists. Paste the console output into your report. Delete nothing under the real `results/derisk/`.

- [ ] **Step 6: Lint, full suite, commit**

```bash
./venv/bin/ruff format scripts/derisk_s2_cp_round.py tests/pe2/test_derisk_s2_round.py && ./venv/bin/ruff check scripts/derisk_s2_cp_round.py tests/pe2/test_derisk_s2_round.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q -x
git add scripts/derisk_s2_cp_round.py tests/pe2/test_derisk_s2_round.py
git commit -m "feat(derisk): S2 runner — gate bars, SEL/ANC/NULL/SBLONG arms, selection + gate, summary" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 components → Tasks 1–2; §3 bars/streams/selection/gate → Task 3 `measure_start`, `select_and_gate`, `gate_pass`; §4 arms incl. NULL and SBLONG → `ARMS`, `run_ppo_arm`, `run_sblong`; §5 verdicts → `hypothesis_verdicts` (H-SEL ≥ 2/3 selected, H-ANC ≥ 2/3 final per β, NULL invalidates H-SEL only, determinism note, SBLONG via `structured_verdict`); §6 direction table → `direction`; stop rule → `require_not_measured`, `require_arm_not_run`; §8 artefacts and figure → runner paths, `write_figure`.
- **Placeholders:** none.
- **Type consistency:** `finetune_ppo_anchored(..., beta, device, callback, env_wrap, custom_objects)` matches its call; `gate_pass(logs_policy, logs_ref, role, p_gate)` matches; `select_checkpoint(rows)` returns a row with `ckpt`, `steps`, `win_rate` (CheckpointEvalProbe rows); `hypothesis_verdicts` reads `results[a]["selected"|"final"]["gate"]["PPO_CLONE"]["pass"]`, which `select_and_gate` produces and the tests' `_arm` fixture mirrors.
