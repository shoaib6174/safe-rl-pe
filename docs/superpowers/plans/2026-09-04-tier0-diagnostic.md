# Tier 0 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Tier 0 of the de-risk plan — a signal audit with a pre-registered kill rule, the three proven dense rewards with non-degeneracy checks, and three mechanism-matched 200k-step pilots plus a negative control — so that the learner for Tier 1 is chosen from evidence rather than named.

**Architecture:** Three small additions to `pe2/`. A `rewards` module holds a search-staleness tracker ported from the root codebase and a training-only `DenseRewardWrapper` around the existing `SingleAgentWrapper`. An `anchored_sac` module subclasses SB3's SAC with a KL-to-clone term in the actor loss. `exploitability.py` gains an `env_wrap` hook on the fine-tune path and a PPO behaviour-cloning path. Two scripts drive the audit and the pilots; every number lands in `results/derisk/`.

**Tech Stack:** Python 3.10+, numpy, gymnasium, stable-baselines3 2.7.1, torch 2.10 (CUDA available on niro-1), matplotlib, pytest, ruff.

**Spec:** `docs/pathway/derisk_plan_2026-09-04.md` §"Tier 0" (T0.1, T0.2, T0.3 and the Tier-0 direction table). Binding decisions: D80 (LayerNorm critics), D81 (fixed `ent_coef=1e-5`), D78 (critic warm-up phases), D83 (pre-registration, negative control, power), D92 (this tier exists). Reference for the ports: `envs/search_staleness.py` and `envs/rewards.py` in the root codebase (read-only).

## Global Constraints

- Always the project venv: `./venv/bin/python`, `./venv/bin/ruff`. Always `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for pytest (a system ROS install injects a broken plugin). Command shape: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2 -q --no-header`.
- Ruff must stay clean (`./venv/bin/ruff check pe2 tests/pe2 scripts && ./venv/bin/ruff format --check pe2 tests/pe2 scripts`). The tree is clean at the start; run `ruff format` on your own files only.
- **Evaluation is always on true outcomes**: `evaluate_sb3_policy` and `CheckpointEvalProbe` build their own unshaped `SingleAgentWrapper`. Dense rewards exist only inside the training env. Never route a dense reward into an eval path.
- **Every dense term is per-step bounded or potential-based**, so terminal outcomes are unchanged (D68 rule). Terminal ±1 stays the env's reward.
- **SAC is always LayerNorm-critic (`ln_sac_policy()`) with `ent_coef=1e-5`** (D80/D81). No `"auto"` anywhere.
- **Seed streams** (D66 contract; all disjoint from existing 700000–770000 ranges): pilot selection evals `780000+`; signal-audit rollouts `790000+`. Training seeds: P-A `7`, P-B `8`, P-C `9`, negative control `10` (existing arms used 1–6).
- **Pre-registered pilot pass rule** (derisk v2 T0.3): with `critic_warmup=50_000` and `total=200_000`, probe every 50k on N=200 paired seeds; **pass = win rate ≥ 0.137 (the gated clone's Wilson lower bound) at every probe from 100k onward**. The negative control (naive fine-tune, `critic_warmup=0`) must fall below 0.137 by 200k, or the harness is broken.
- **Pre-registered kill rule** (T0.1): the audit reports positive terminals per 1M steps under the clone; the P-A pilot reports the same during training with dense reward. If the P-A figure is < 50, Tier 1 does not run (Route C).
- **P-B requires overriding one SB3 method** (`SAC.train`, 101 lines copied from the installed 2.7.1 plus ~12 added). This is the smallest possible loss-level anchor and is confined to one subclass in one module. It is a deliberate, bounded exception to "RL libraries are not rewritten" and is recorded as such in Task 8's decision entry. If the human rejects it, drop Task 5 and run P-A and P-C only.
- **PPO pilot choices are fixed now, not tuned later**: `gamma=0.999`, `gae_lambda=0.95`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `ent_coef=0.0`, `policy_kwargs={"log_std_init": -1.0}` (σ ≈ 0.37 so the clone's behaviour survives rollout noise; SB3's default σ = 1 would swamp a policy whose actions live in [−1, 1]).
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Commit to `main`, no push, never `git add -A` / `git add .` / `commit -a` — the repo carries many unrelated modified files. Commit only the files each task names.
- Surgical: touch only what the task names. The deliberate duplication between `run_duel` and `run_shielded_duel` and the existing docstrings in `exploitability.py` are not yours to improve.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pe2/derisk.py` | T0.1 signal audit: roll a saved policy, compute positives/1M, capture-length histogram, visible fraction, MC-return variance by step bucket | Create (Task 1) |
| `scripts/derisk_t0_signal_audit.py` | CLI for T0.1 → `results/derisk/t0_signal_audit.{json,png}` | Create (Task 1) |
| `pe2/rewards.py` | `SearchStalenessTracker` (pe2 port using `pe2.sensing`) and `DenseRewardWrapper` (training-only dense terms + non-degeneracy checks) | Create (Tasks 2, 3) |
| `pe2/exploitability.py` | `env_wrap` hook on `_ln_finetune_model`/`finetune_br`; `train_bc_ppo`, `finetune_ppo` | Modify (Tasks 4, 6) |
| `pe2/anchored_sac.py` | `KLAnchoredSAC(SAC)`: frozen clone actor, KL term in the actor loss | Create (Task 5) |
| `scripts/derisk_t0_pilots.py` | Runs P-A / P-B / P-C / negative control with a 50k probe, applies the pass rule, writes `results/derisk/t0_pilots.{json,png}` | Create (Task 7) |
| `tests/pe2/test_derisk.py`, `tests/pe2/test_rewards.py`, `tests/pe2/test_anchored_sac.py` | Tests for the new modules | Create |
| `tests/pe2/test_exploitability.py` | Tests for `env_wrap` and the PPO path | Modify |
| `docs/worklogs/2026-09-04_S121.md` | Pre-registration record of the pilot criteria and the P-B exception | Modify (Task 8) |

---

### Task 1: T0.1 signal audit

**Files:**
- Create: `pe2/derisk.py`
- Create: `scripts/derisk_t0_signal_audit.py`
- Test: `tests/pe2/test_derisk.py`

**Interfaces:**
- Consumes: `SingleAgentWrapper(cfg, role, opponent_factory)` with `reset(seed=)`, `step(action)`, `_last_joint_obs[role][6]` (LOS-visible flag), `env.current_step`; SB3 `model.predict(obs, deterministic=)`; `results/visual_gates/step_05/sac_bc_pursuer.zip` (the gated clone).
- Produces:
  ```python
  def signal_audit(model, cfg, role, opponent_factory, min_steps, seed0, gamma=0.999,
                   deterministic=False, bucket=200) -> dict
  ```
  returning keys `total_steps`, `n_episodes`, `captures`, `positives_per_1M`, `visible_fraction`, `capture_lengths` (list[int]), `return_variance_by_bucket` (list of `{"t_lo","t_hi","n","variance","mean"}`), `return_variance_overall`, `seed0`, `deterministic`. A step's Monte-Carlo return under the terminal-only reward is `G_t = gamma**(L-t) * R` where `L` is episode length and `R` is the role's terminal reward.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_derisk.py
"""Tests for pe2.derisk — Tier 0 signal audit (T0.1)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.derisk import mc_returns, signal_audit
from pe2.exploitability import ConstantPolicy
from pe2.scripted import GreedyEvader

CFG = EnvConfig()


class TestMcReturns:
    def test_terminal_only_returns_hand_computed(self):
        # Episode of length 3 ending in capture (R=+1), gamma=0.5:
        # G_0 = 0.5**3 = 0.125, G_1 = 0.25, G_2 = 0.5.
        g = mc_returns(length=3, terminal_reward=1.0, gamma=0.5)
        assert g == pytest.approx([0.125, 0.25, 0.5])

    def test_loss_flips_sign(self):
        g = mc_returns(length=2, terminal_reward=-1.0, gamma=1.0)
        assert g == pytest.approx([-1.0, -1.0])


class _Full:
    """model.predict shim: full-forward pursuer (a heuristic from D76)."""

    def predict(self, obs, deterministic=True):
        return np.array([1.0, 0.0], dtype=np.float32), None


class TestSignalAudit:
    def test_counts_are_consistent(self):
        rep = signal_audit(
            _Full(), CFG, "pursuer", GreedyEvader, min_steps=3000, seed0=790000
        )
        assert rep["total_steps"] >= 3000
        assert rep["n_episodes"] == len(rep["episode_lengths"])
        assert rep["captures"] == len(rep["capture_lengths"])
        assert rep["positives_per_1M"] == pytest.approx(
            rep["captures"] / rep["total_steps"] * 1e6
        )
        assert 0.0 <= rep["visible_fraction"] <= 1.0
        # bucket step counts must sum to the total
        assert sum(b["n"] for b in rep["return_variance_by_bucket"]) == rep["total_steps"]

    def test_deterministic_repeat(self):
        a = signal_audit(_Full(), CFG, "pursuer", GreedyEvader, min_steps=2000, seed0=790000)
        b = signal_audit(_Full(), CFG, "pursuer", GreedyEvader, min_steps=2000, seed0=790000)
        assert a == b

    def test_variance_is_zero_when_every_episode_has_one_length_and_outcome(self):
        # stand_still never captures: every episode is a 1200-step timeout,
        # so G_t is identical across episodes for each t -> bucket variance 0.
        class _Still:
            def predict(self, obs, deterministic=True):
                return np.array([-1.0, 0.0], dtype=np.float32), None

        rep = signal_audit(_Still(), CFG, "pursuer", GreedyEvader, min_steps=2400, seed0=790000)
        assert rep["captures"] == 0
        assert all(b["variance"] == pytest.approx(0.0) for b in rep["return_variance_by_bucket"])
```

(`ConstantPolicy` is imported only to document the shim's provenance; remove the import if ruff flags it unused.)

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py -v --no-header`
Expected: ImportError `No module named 'pe2.derisk'`.

- [ ] **Step 3: Implement `pe2/derisk.py`**

```python
"""Tier 0 diagnostics (docs/pathway/derisk_plan_2026-09-04.md, T0.1).

signal_audit rolls a saved policy in the training-time regime and measures
the learning signal the terminal-only reward actually provides: positive
terminals per million steps (the exploration bottleneck), the capture-
length distribution, the fraction of steps with the opponent visible, and
the variance of the Monte-Carlo return by step index. The pre-registered
kill rule (< 50 positives per 1M even with dense reward -> Route C) is
evaluated on the P-A pilot's training episodes, not here; this audit gives
the terminal-only baseline number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper


def mc_returns(length: int, terminal_reward: float, gamma: float) -> list[float]:
    """Monte-Carlo return of every step of a terminal-only episode:
    G_t = gamma**(length - t) * R for t = 0..length-1."""
    return [gamma ** (length - t) * terminal_reward for t in range(length)]


def signal_audit(
    model,
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    min_steps: int,
    seed0: int,
    gamma: float = 0.999,
    deterministic: bool = False,
    bucket: int = 200,
) -> dict:
    """Roll `model` (anything with .predict) until >= min_steps env steps."""
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory)
    total_steps = 0
    visible_steps = 0
    episode_lengths: list[int] = []
    capture_lengths: list[int] = []
    per_step_returns: list[tuple[int, float]] = []  # (t, G_t)
    ep = 0
    while total_steps < min_steps:
        obs, _ = wrapper.reset(seed=seed0 + ep)
        ep += 1
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            visible_steps += int(wrapper._last_joint_obs[role][6] > 0.5)
            if terminated or truncated:
                break
        length = wrapper.env.current_step
        total_steps += length
        episode_lengths.append(length)
        won = (info["outcome"] == "capture") == (role == "pursuer")
        if info["outcome"] == "capture":
            capture_lengths.append(length)
        terminal = 1.0 if won else -1.0
        per_step_returns.extend(enumerate(mc_returns(length, terminal, gamma)))

    captures = len(capture_lengths)
    ts = np.array([t for t, _ in per_step_returns])
    gs = np.array([g for _, g in per_step_returns])
    buckets = []
    for lo in range(0, cfg.max_steps, bucket):
        mask = (ts >= lo) & (ts < lo + bucket)
        n = int(mask.sum())
        buckets.append(
            {
                "t_lo": lo,
                "t_hi": lo + bucket,
                "n": n,
                "variance": float(gs[mask].var()) if n else 0.0,
                "mean": float(gs[mask].mean()) if n else 0.0,
            }
        )
    return {
        "role": role,
        "seed0": seed0,
        "deterministic": deterministic,
        "gamma": gamma,
        "total_steps": total_steps,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "captures": captures,
        "capture_lengths": capture_lengths,
        "positives_per_1M": captures / total_steps * 1e6,
        "visible_fraction": visible_steps / total_steps,
        "return_variance_overall": float(gs.var()),
        "return_variance_by_bucket": buckets,
    }
```

- [ ] **Step 4: Run the tests to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py -v --no-header`
Expected: 5 PASS. (The stand-still test asserts zero variance per bucket; if a bucket shows nonzero variance, the audit is mixing episodes of different lengths — that means the stand-still shim moved, which it cannot with `v=-1 → 0 m/s`; investigate rather than loosen.)

- [ ] **Step 5: Write the CLI script**

```python
# scripts/derisk_t0_signal_audit.py
"""T0.1 signal audit (derisk plan v2). Rolls the gated SAC clone for >= 1M
env steps in BOTH the stochastic (training-time) and deterministic regimes
and writes results/derisk/t0_signal_audit.{json,png}.

Usage: ./venv/bin/python scripts/derisk_t0_signal_audit.py [--role pursuer] [--min-steps 1000000]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from pe2.config import EnvConfig  # noqa: E402
from pe2.derisk import signal_audit  # noqa: E402
from pe2.scripted import GreedyEvader, GreedyPursuer  # noqa: E402

OUT = pathlib.Path("results/derisk")
CLONE = {
    "pursuer": pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip"),
    "evader": pathlib.Path("results/visual_gates/step_05/sac_bc_evader.zip"),
}
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}
SEED0 = 790000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    p.add_argument("--min-steps", type=int, default=1_000_000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    model = SAC.load(CLONE[args.role], device="cpu")
    report = {}
    for det in (False, True):
        rep = signal_audit(
            model, cfg, args.role, OPPONENT[args.role], args.min_steps, SEED0, deterministic=det
        )
        key = "deterministic" if det else "stochastic"
        report[key] = rep
        print(
            f"[{key}] steps={rep['total_steps']} episodes={rep['n_episodes']} "
            f"captures={rep['captures']} positives/1M={rep['positives_per_1M']:.1f} "
            f"visible={rep['visible_fraction']:.3f} var(G)={rep['return_variance_overall']:.4f}",
            flush=True,
        )
    report["kill_rule"] = {
        "threshold_positives_per_1M": 50,
        "note": "evaluated on the P-A pilot's dense-reward training episodes; "
        "this file gives the terminal-only baseline",
    }
    with open(OUT / f"t0_signal_audit_{args.role}.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, style in (("stochastic", "-"), ("deterministic", "--")):
        b = report[key]["return_variance_by_bucket"]
        axes[0].plot([x["t_lo"] for x in b], [x["variance"] for x in b], style, label=key)
        axes[1].hist(
            report[key]["capture_lengths"], bins=24, range=(0, cfg.max_steps),
            histtype="step", label=f"{key} (n={report[key]['captures']})",
        )
    axes[0].set(xlabel="step index t", ylabel="var(G_t)", title="MC-return variance by step")
    axes[1].set(xlabel="capture length (steps)", ylabel="count", title="capture lengths")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"t0_signal_audit_{args.role}.png", dpi=120)
    print(f"wrote {OUT}/t0_signal_audit_{args.role}.{{json,png}}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run the script on a short budget, then lint**

Run: `./venv/bin/python scripts/derisk_t0_signal_audit.py --min-steps 20000`
Expected: two summary lines printed, JSON + PNG written under `results/derisk/`. Then `./venv/bin/ruff format pe2/derisk.py scripts/derisk_t0_signal_audit.py tests/pe2/test_derisk.py && ./venv/bin/ruff check pe2 tests/pe2 scripts`.

- [ ] **Step 7: Commit**

```bash
git add pe2/derisk.py scripts/derisk_t0_signal_audit.py tests/pe2/test_derisk.py
git commit -m "feat(pe2/derisk): T0.1 signal audit — positives/1M, capture lengths, MC-return variance

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(The full 1M-step run is Task 7's job order; do not run it here.)

---

### Task 2: `SearchStalenessTracker` for pe2

**Files:**
- Create: `pe2/rewards.py`
- Test: `tests/pe2/test_rewards.py`

**Interfaces:**
- Consumes: `pe2.sensing.is_visible(own_pose, target_xy, obstacles, sensing)`; `EnvConfig.arena_size`, `EnvConfig.sensing`.
- Produces:
  ```python
  class SearchStalenessTracker:
      def __init__(self, cfg: EnvConfig, grid_size: int = 10, t_stale: int = 50)
      def reset(self) -> None
      def observe_and_reward(self, own_pose: np.ndarray, obstacles: list[dict], current_step: int) -> float
  ```
  Port of the root tracker (`envs/search_staleness.py`) with one deliberate change: a cell counts as observed when its centre is `is_visible` under the role's FOV cone + LOS (the root used a plain radius + LOS), so the reward matches what pe2's sensing actually sees. Returns the mean staleness in `[0, 1]` of the observed cells (0 = all fresh, 1 = all fully stale) and marks them observed; `0.0` when nothing is observed. `reset()` makes every cell fully stale (`last_observed = -t_stale`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_rewards.py
"""Tests for pe2.rewards — Tier 0 dense training rewards (T0.2)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.rewards import SearchStalenessTracker

CFG = EnvConfig()


class TestSearchStalenessTracker:
    def test_first_observation_is_fully_stale(self):
        # After reset every cell has last_observed = -t_stale, so at step 0
        # steps_since = t_stale -> staleness clipped to 1.0 for every
        # observed cell -> mean 1.0.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        r = tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        assert r == pytest.approx(1.0)

    def test_immediate_reobservation_is_fresh(self):
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        # Same pose, next step: every observed cell was seen 1 step ago ->
        # staleness 1/50 = 0.02 for all -> mean 0.02.
        r = tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=1)
        assert r == pytest.approx(1 / 50)

    def test_cells_behind_the_robot_are_not_observed(self):
        # Heading +x at the origin with a 90 deg cone: the cell centred at
        # (-4.5, 0) is behind the robot and must stay stale; the cell at
        # (+4.5, 0) is ahead (within 8 m range) and must be marked observed.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        idx_behind = tr.cell_index(-4.5, 0.0)
        idx_ahead = tr.cell_index(4.5, 0.0)
        assert tr.last_observed[idx_behind] == -50
        assert tr.last_observed[idx_ahead] == 0

    def test_occluded_cell_is_not_observed(self):
        # Obstacle of radius 1.0 at (2, 0) blocks LOS from the origin to the
        # cell centred at (4.5, 0) (segment passes through the disk).
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 1.0}]
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), obstacles, current_step=0)
        assert tr.last_observed[tr.cell_index(4.5, 0.0)] == -50

    def test_nothing_visible_returns_zero(self):
        # Robot facing +x from the east wall: the cone leaves the arena, no
        # cell centre is in range -> 0.0 and no cell marked.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        r = tr.observe_and_reward(np.array([4.9, 0.0, 0.0]), [], current_step=0)
        assert r == 0.0
        assert (tr.last_observed == -50).all()

    def test_reward_varies_along_a_trajectory(self):
        # Non-degeneracy (derisk v2 T0.2 plateau check): driving straight
        # across the arena must produce a staleness signal with nonzero
        # spread, i.e. something to learn from.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        rs = [
            tr.observe_and_reward(np.array([-4.0 + 0.1 * k, 0.0, 0.0]), [], current_step=k)
            for k in range(60)
        ]
        assert np.std(rs) > 0.0
        assert 0.0 < np.mean(rs) < 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py -v --no-header`
Expected: ImportError `No module named 'pe2.rewards'`.

- [ ] **Step 3: Implement**

```python
# pe2/rewards.py
"""Tier 0 (T0.2): the three PROVEN dense training rewards, ported from the
root chassis, plus their non-degeneracy checks.

Why this module exists: every Step-5 arm (D68-D82) trained on the terminal
+/-1 alone, while lessons §3 (D20) says a per-step pursuer visibility
reward is mandatory under partial observability (0.57 vs 0.10 CR) and D24
names search-staleness as the missing search skill. Neither was ever tried
in pe2 (D92). Everything here is TRAINING-ONLY: evaluation always uses the
unshaped SingleAgentWrapper so reported win rates are true outcomes.

Terms (all per-step bounded or potential-based, so terminal outcomes are
unchanged):
- pursuer visibility: +w_vis per step the opponent is visible to the
  learning role (obs[6] == 1: FOV cone AND LOS clear, D6 combined masking).
- search staleness: +w_search * mean staleness of the grid cells the role
  can currently see (SearchStalenessTracker; D24 w=1e-4, D4: >= 3e-4
  teaches patrol-not-hunt).
- evader visibility (root "Mode B"): +w_evis if hidden, -w_evis if visible.
  KNOWN PLATEAU (lessons §5b: flat -1 in open space), so it is only enabled
  together with the PBRS obstacle-seeking potential Phi = -distance to the
  nearest obstacle surface, F = gamma*Phi(s') - Phi(s).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.sensing import is_visible


class SearchStalenessTracker:
    """Grid-cell staleness for the search reward (root envs/search_staleness.py,
    with observation defined by pe2's FOV+LOS sensing instead of a radius)."""

    def __init__(self, cfg: EnvConfig, grid_size: int = 10, t_stale: int = 50):
        self.cfg = cfg
        self.grid_size = grid_size
        self.t_stale = t_stale
        half = cfg.arena_size / 2.0
        self.cell = cfg.arena_size / grid_size
        centres = -half + (np.arange(grid_size) + 0.5) * self.cell
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        self.centres = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (G*G, 2)
        self.last_observed = np.full(grid_size * grid_size, -t_stale, dtype=np.int64)

    def cell_index(self, x: float, y: float) -> int:
        half = self.cfg.arena_size / 2.0
        ix = int(np.clip((x + half) // self.cell, 0, self.grid_size - 1))
        iy = int(np.clip((y + half) // self.cell, 0, self.grid_size - 1))
        return ix * self.grid_size + iy

    def reset(self) -> None:
        self.last_observed[:] = -self.t_stale

    def observe_and_reward(
        self, own_pose: np.ndarray, obstacles: list[dict], current_step: int
    ) -> float:
        seen = [
            i
            for i, c in enumerate(self.centres)
            if is_visible(own_pose, (float(c[0]), float(c[1])), obstacles, self.cfg.sensing)
        ]
        if not seen:
            return 0.0
        steps_since = current_step - self.last_observed[seen]
        staleness = np.clip(steps_since / self.t_stale, 0.0, 1.0)
        self.last_observed[seen] = current_step
        return float(staleness.mean())
```

(`gym` and `SingleAgentWrapper` are imported now for Task 3; if ruff flags them unused at this commit, leave them out and add them in Task 3.)

- [ ] **Step 4: Run the tests to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py -v --no-header`
Expected: 6 PASS. If `test_cells_behind_the_robot_are_not_observed` fails on the ahead cell, check the range: `(4.5, 0)` is 4.5 m from the origin, inside the 8 m cone.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/rewards.py tests/pe2/test_rewards.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/rewards.py tests/pe2/test_rewards.py
git commit -m "feat(pe2/rewards): search-staleness tracker on pe2 FOV+LOS sensing (D24 port)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `DenseRewardWrapper` with non-degeneracy checks

**Files:**
- Modify: `pe2/rewards.py` (append)
- Test: `tests/pe2/test_rewards.py` (append)

**Interfaces:**
- Consumes: `SingleAgentWrapper` (`.role`, `.env.pursuer_pose`, `.env.evader_pose`, `.env.obstacles`, `.env.current_step`, `._last_joint_obs`), `SearchStalenessTracker` (Task 2).
- Produces:
  ```python
  class DenseRewardWrapper(gym.Wrapper):
      def __init__(self, env: SingleAgentWrapper, *, w_vis: float = 0.0, w_search: float = 0.0,
                   w_evis: float = 0.0, w_pbrs: float = 0.0, gamma: float = 0.999,
                   grid_size: int = 10, t_stale: int = 50)
  ```
  `reset` resets the tracker; `step` adds `w_vis * visible + w_search * staleness + w_evis * (+1 hidden / -1 visible) + w_pbrs * (gamma*Phi(s') - Phi(s))` to the env reward, where `Phi(s) = -(distance from the learning role to the nearest obstacle surface)` (0 when there are no obstacles). Raises `ValueError` at construction if `w_evis > 0 and w_pbrs == 0` (the known plateau, lessons §5b). Also:
  ```python
  def signal_nondegeneracy(rewards: list[float]) -> dict  # {"frac_positive", "std", "mean"}
  ```
  the T0.2 check: over a rollout, the dense term must be neither always-on nor never-on (`0.05 < frac_positive < 0.95`) and have nonzero spread.

- [ ] **Step 1: Write the failing tests** (append to `tests/pe2/test_rewards.py`; extend the import line to `from pe2.rewards import DenseRewardWrapper, SearchStalenessTracker, signal_nondegeneracy` and add `from pe2.exploitability import SingleAgentWrapper` and `from pe2.scripted import GreedyEvader, GreedyPursuer`)

```python
def _place(w, pursuer, evader):
    w.env.pursuer_pose = np.array(pursuer, dtype=float)
    w.env.evader_pose = np.array(evader, dtype=float)
    w._last_joint_obs = w.env._get_obs()


class TestDenseRewardWrapper:
    def test_visibility_term_hand_computed(self):
        # Pursuer at origin heading +x, evader 2 m ahead: visible -> +0.1.
        # Evader behind: not visible -> +0.0. Terminal env reward is 0 mid-episode.
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1)
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([-1.0, 0.0]))  # v=0: nothing moves but the evader
        assert r == pytest.approx(0.1, abs=1e-9)
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (-3.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([-1.0, 0.0]))
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_search_term_uses_tracker(self):
        # First step after reset: every visible cell is fully stale -> mean
        # staleness 1.0 -> reward w_search * 1.0.
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_search=1e-4)
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (-4.0, -4.0, 0.0))  # evader behind: no vis term
        _, r, *_ = w.step(np.array([-1.0, 0.0]))
        assert r == pytest.approx(1e-4, abs=1e-12)

    def test_evader_visibility_requires_pbrs(self):
        cfg = EnvConfig()
        with pytest.raises(ValueError, match="PBRS"):
            DenseRewardWrapper(SingleAgentWrapper(cfg, "evader", GreedyPursuer), w_evis=0.1)

    def test_evader_visibility_sign_and_pbrs_hand_computed(self):
        # One obstacle r=1.0 at (3,0); evader at origin heading +x, pursuer
        # far behind it (not visible): hidden -> +w_evis. PBRS with gamma=1:
        # Phi(s) = -(3 - 1) = -2 before; the evader moves v_max*dt = 1.2*0.05
        # = 0.06 toward the obstacle -> Phi(s') = -(2.94 - 1) = -1.94;
        # F = -1.94 - (-2) = +0.06. Total = 0.1 + 0.06 = 0.16.
        cfg = EnvConfig(n_obstacles=1)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "evader", GreedyPursuer), w_evis=0.1, w_pbrs=1.0, gamma=1.0
        )
        w.reset(seed=0)
        w.env.env.obstacles = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        _place(w.env, (-4.5, -4.5, 0.0), (0.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([1.0, 0.0]))  # full speed forward
        assert r == pytest.approx(0.16, abs=1e-6)

    def test_terminal_outcome_unchanged(self):
        # Adjacent agents: capture on the first step; the env's terminal +1
        # must dominate and be present unchanged (dense terms are O(0.1)).
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1)
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (0.2, 0.0, 0.0))
        _, r, terminated, *_ = w.step(np.array([1.0, 0.0]))
        assert terminated
        assert r >= 1.0

    def test_nondegeneracy_on_a_real_rollout(self):
        # T0.2 plateau check, run for real: the FOV-limited pursuer rolled
        # against the greedy evader must see the opponent on SOME but not
        # ALL steps, so the visibility term is a live signal.
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig()
        w = DenseRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1)
        rewards = []
        for ep in range(3):
            obs, _ = w.reset(seed=790100 + ep)
            teacher = FovLimitedPursuer()
            teacher.reset()
            while True:
                v, om = teacher.act(w.env._last_joint_obs["pursuer"], w.env.env)
                a = np.array([2.0 * v / cfg.v_max - 1.0, om / cfg.omega_max], dtype=np.float32)
                obs, r, term, trunc, _ = w.step(a)
                rewards.append(r)
                if term or trunc:
                    break
        nd = signal_nondegeneracy(rewards)
        assert 0.05 < nd["frac_positive"] < 0.95
        assert nd["std"] > 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py -v --no-header`
Expected: ImportError on `DenseRewardWrapper`.

- [ ] **Step 3: Implement** (append to `pe2/rewards.py`)

```python
def _nearest_obstacle_surface(xy: np.ndarray, obstacles: list[dict]) -> float:
    if not obstacles:
        return 0.0
    return float(
        min(np.hypot(xy[0] - o["x"], xy[1] - o["y"]) - o["radius"] for o in obstacles)
    )


class DenseRewardWrapper(gym.Wrapper):
    """Training-only dense reward on top of SingleAgentWrapper (T0.2)."""

    def __init__(
        self,
        env: SingleAgentWrapper,
        *,
        w_vis: float = 0.0,
        w_search: float = 0.0,
        w_evis: float = 0.0,
        w_pbrs: float = 0.0,
        gamma: float = 0.999,
        grid_size: int = 10,
        t_stale: int = 50,
    ):
        super().__init__(env)
        if w_evis > 0 and w_pbrs == 0:
            raise ValueError(
                "evader visibility reward is a known plateau (lessons §5b); "
                "enable it only with the PBRS obstacle potential (w_pbrs > 0)"
            )
        self.w_vis, self.w_search, self.w_evis, self.w_pbrs = w_vis, w_search, w_evis, w_pbrs
        self.gamma = gamma
        self.tracker = SearchStalenessTracker(env.cfg, grid_size, t_stale) if w_search else None

    # ---- helpers on the learning role's TRUE state (training-only: the
    # policy never sees these; the reward may) ----
    def _own_pose(self) -> np.ndarray:
        e = self.env.env
        return e.pursuer_pose if self.env.role == "pursuer" else e.evader_pose

    def _visible(self) -> bool:
        return bool(self.env._last_joint_obs[self.env.role][6] > 0.5)

    def _phi(self) -> float:
        return -_nearest_obstacle_surface(self._own_pose()[:2], self.env.env.obstacles)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if self.tracker is not None:
            self.tracker.reset()
        return out

    def step(self, action):
        phi = self._phi() if self.w_pbrs else 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        dense = 0.0
        visible = self._visible()
        if self.w_vis:
            dense += self.w_vis * float(visible)
        if self.tracker is not None:
            dense += self.w_search * self.tracker.observe_and_reward(
                self._own_pose(), self.env.env.obstacles, self.env.env.current_step
            )
        if self.w_evis:
            dense += self.w_evis * (-1.0 if visible else 1.0)
        if self.w_pbrs:
            dense += self.w_pbrs * (self.gamma * self._phi() - phi)
        return obs, reward + dense, terminated, truncated, info


def signal_nondegeneracy(rewards: list[float]) -> dict:
    """T0.2 plateau check: a dense signal must vary (std > 0) and be neither
    always-on nor never-on (0.05 < frac_positive < 0.95) over a rollout."""
    r = np.asarray(rewards, dtype=float)
    return {
        "frac_positive": float((r > 0).mean()) if r.size else 0.0,
        "std": float(r.std()) if r.size else 0.0,
        "mean": float(r.mean()) if r.size else 0.0,
    }
```

- [ ] **Step 4: Run the tests to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_rewards.py -v --no-header`
Expected: 12 PASS. Two traps: in `test_visibility_term_hand_computed` the greedy evader moves during the step, but from 2 m directly ahead it cannot leave a 90° cone of 8 m range in one 0.06 m step, so visibility holds; in the PBRS test the evader's forward step is `evader_v_max*dt = 1.2*0.05 = 0.06`, not `1.0*0.05`.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/rewards.py tests/pe2/test_rewards.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/rewards.py tests/pe2/test_rewards.py
git commit -m "feat(pe2/rewards): DenseRewardWrapper — visibility, search, evader-vis+PBRS, with non-degeneracy check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `env_wrap` hook on the fine-tune path

**Files:**
- Modify: `pe2/exploitability.py` — `_ln_finetune_model` (~line 468) and `finetune_br` (~line 499)
- Test: `tests/pe2/test_exploitability.py` (append)

**Interfaces:**
- Produces: `_ln_finetune_model(..., env_wrap=None)` and `finetune_br(..., env_wrap=None)` where `env_wrap: Callable[[SingleAgentWrapper], gym.Env] | None` is applied to the training env only. Everything else unchanged, including defaults, so every existing caller behaves identically.

- [ ] **Step 1: Write the failing test** (append to `tests/pe2/test_exploitability.py`; add `from pe2.exploitability import finetune_br` to the import block and `from pe2.rewards import DenseRewardWrapper`)

```python
class TestFinetuneEnvWrap:
    def test_env_wrap_is_applied_to_training_env_only(self, tmp_path):
        # Build a tiny BC model to fine-tune from, then fine-tune for a few
        # steps with a recording wrapper; the wrapper must see every training
        # step and the returned model must still evaluate on an UNWRAPPED env.
        from pe2.exploitability import collect_demonstrations, train_bc_actor
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig(n_obstacles=0)
        demos = collect_demonstrations(cfg, "pursuer", FovLimitedPursuer, GreedyEvader, 1, 750900)
        bc, _ = train_bc_actor(cfg, "pursuer", GreedyEvader, demos, seed=0, epochs=1, device="cpu")
        path = tmp_path / "bc.zip"
        bc.save(path)

        seen = {"steps": 0}

        def wrap(env):
            w = DenseRewardWrapper(env, w_vis=0.1)
            orig = w.step

            def counting_step(a):
                seen["steps"] += 1
                return orig(a)

            w.step = counting_step
            return w

        model = finetune_br(
            cfg, "pursuer", GreedyEvader, path, total_timesteps=64, seed=0,
            device="cpu", critic_warmup=32, env_wrap=wrap,
        )
        assert seen["steps"] == 64
        logs = evaluate_sb3_policy(model, cfg, "pursuer", GreedyEvader, n_episodes=1, seed0=780900)
        assert len(logs) == 1 and "outcome" in logs[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py::TestFinetuneEnvWrap -v --no-header`
Expected: `TypeError: finetune_br() got an unexpected keyword argument 'env_wrap'`.

- [ ] **Step 3: Implement**

In `_ln_finetune_model`, add a trailing parameter `env_wrap=None` and change the model construction to use the wrapped env:

```python
def _ln_finetune_model(cfg, role, opponent_factory, model_path, seed, device, gamma, env_wrap=None):
    ...
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory, seed0=seed * 1000)
    env = env_wrap(wrapper) if env_wrap is not None else wrapper
    model = SAC(
        ln_sac_policy(),
        env,
        ...
```

In `finetune_br`, add `env_wrap=None` after `callback=None` in the signature, pass it through: `model = _ln_finetune_model(cfg, role, opponent_factory, model_path, seed, device, gamma, env_wrap=env_wrap)`, and add one sentence to the docstring: "`env_wrap`, if given, wraps the TRAINING env only (e.g. `pe2.rewards.DenseRewardWrapper`); evaluation stays unshaped."

- [ ] **Step 4: Run the new test and the whole exploitability file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py -q --no-header`
Expected: all PASS (existing + 1).

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/exploitability.py tests/pe2/test_exploitability.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/exploitability.py tests/pe2/test_exploitability.py
git commit -m "feat(pe2/exploitability): env_wrap hook so fine-tuning can use training-only dense rewards

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `KLAnchoredSAC` (pilot P-B)

**Files:**
- Create: `pe2/anchored_sac.py`
- Test: `tests/pe2/test_anchored_sac.py`

**Interfaces:**
- Consumes: SB3 2.7.1 `SAC` (`train`, `actor.get_action_dist_params(obs) -> (mean, log_std, kwargs)`, `actor.action_log_prob`, `critic`, `critic_target`, `ent_coef_tensor`, `replay_buffer`), `ln_sac_policy()`.
- Produces:
  ```python
  def gaussian_kl(mean, log_std, mean_ref, log_std_ref) -> torch.Tensor  # per-sample KL(N(mean,std) || N(ref)), summed over action dims
  class KLAnchoredSAC(SAC):
      def __init__(self, *args, beta: float = 1.0, **kwargs)
      def set_anchor(self, actor) -> None      # deep-copies `actor` as the frozen reference
      def train(self, gradient_steps, batch_size=64) -> None   # SB3's train + beta * KL in the actor loss
  ```
  KL is computed on the pre-tanh Gaussians; tanh is a fixed bijection so this equals the KL of the squashed policies exactly. The last KL value is logged as `train/anchor_kl`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pe2/test_anchored_sac.py
"""Tests for pe2.anchored_sac — pilot P-B (KL anchor to the frozen clone)."""

import copy

import numpy as np
import pytest
import torch

from pe2.anchored_sac import KLAnchoredSAC, gaussian_kl
from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper, ln_sac_policy
from pe2.scripted import GreedyEvader


class TestGaussianKl:
    def test_identical_is_zero(self):
        m = torch.zeros(4, 2)
        s = torch.zeros(4, 2)  # log_std 0 -> std 1
        assert gaussian_kl(m, s, m, s).abs().max().item() == pytest.approx(0.0, abs=1e-7)

    def test_mean_shift_hand_computed(self):
        # KL(N(mu,1) || N(0,1)) = mu^2 / 2 per dim; two dims with mu=1 -> 1.0.
        m = torch.ones(1, 2)
        z = torch.zeros(1, 2)
        assert gaussian_kl(m, z, z, z).item() == pytest.approx(1.0, abs=1e-6)

    def test_std_shift_hand_computed(self):
        # KL(N(0,s) || N(0,1)) = -log s + s^2/2 - 1/2; s = e^-1:
        # 1 + e^-2/2 - 0.5 = 0.5677 per dim, one dim.
        z = torch.zeros(1, 1)
        ls = torch.full((1, 1), -1.0)
        expected = 1.0 + np.exp(-2.0) / 2.0 - 0.5
        assert gaussian_kl(z, ls, z, z).item() == pytest.approx(expected, abs=1e-6)


def _model(beta, seed=0):
    cfg = EnvConfig(n_obstacles=0)
    env = SingleAgentWrapper(cfg, "pursuer", GreedyEvader, seed0=seed * 1000)
    m = KLAnchoredSAC(
        ln_sac_policy(), env, beta=beta, seed=seed, verbose=0, device="cpu",
        gamma=0.999, ent_coef=1e-5, learning_starts=32, train_freq=1,
    )
    m.set_anchor(m.actor)
    return m


class TestKLAnchoredSAC:
    def test_beta_zero_matches_plain_actor_update_direction(self):
        # With beta=0 the anchor term contributes nothing: a train() call must
        # leave the anchor untouched and still move the actor (SAC works).
        m = _model(beta=0.0)
        m.learn(total_timesteps=48)
        anchor_before = copy.deepcopy(m.anchor.state_dict())
        actor_before = copy.deepcopy(m.actor.state_dict())
        m.train(gradient_steps=4, batch_size=16)
        for k in anchor_before:
            assert torch.equal(anchor_before[k], m.anchor.state_dict()[k])
        assert any(
            not torch.equal(actor_before[k], m.actor.state_dict()[k]) for k in actor_before
        )

    def test_large_beta_keeps_the_actor_near_the_anchor(self):
        # Same replay data, same seed: after N updates the actor with a huge
        # anchor weight must stay strictly closer to the clone than beta=0.
        def drift(beta):
            m = _model(beta=beta, seed=3)
            m.learn(total_timesteps=64)
            m.train(gradient_steps=30, batch_size=32)
            obs = torch.as_tensor(
                np.stack([m.env.envs[0].observation_space.sample() for _ in range(64)]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                mean, log_std, _ = m.actor.get_action_dist_params(obs)
                mean_a, log_std_a, _ = m.anchor.get_action_dist_params(obs)
                return gaussian_kl(mean, log_std, mean_a, log_std_a).mean().item()

        assert drift(beta=1e3) < drift(beta=0.0)

    def test_logs_anchor_kl(self):
        m = _model(beta=1.0)
        m.learn(total_timesteps=48)
        m.train(gradient_steps=2, batch_size=16)
        assert "train/anchor_kl" in m.logger.name_to_value
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_anchored_sac.py -v --no-header`
Expected: ImportError `No module named 'pe2.anchored_sac'`.

- [ ] **Step 3: Implement**

```python
# pe2/anchored_sac.py
"""Pilot P-B (derisk plan v2, T0.3): SAC with a KL anchor to a frozen clone.

D82 root-caused the fine-tune collapse: after the critic warm-up the actor
maximizes E[Q(s, a ~ pi)] against a landscape whose action-dependence is
drift-noise-dominated, and leaves the clone manifold. This is the RLHF-style
remedy: add beta * KL(pi_theta(.|s) || pi_clone(.|s)) to the actor loss so
leaving the manifold has an explicit price. KL is computed between the
pre-tanh diagonal Gaussians; tanh is a fixed bijection, so this is EXACTLY
the KL of the squashed policies.

This module overrides ONE SB3 method (SAC.train, 2.7.1, copied verbatim
with the marked additions) — the smallest possible loss-level anchor. It is
a deliberate, bounded exception to "RL libraries are not rewritten" (D92).
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


def gaussian_kl(mean, log_std, mean_ref, log_std_ref):
    """KL(N(mean, e^log_std) || N(mean_ref, e^log_std_ref)) per sample, summed
    over action dimensions."""
    var = th.exp(2.0 * log_std)
    var_ref = th.exp(2.0 * log_std_ref)
    kl = (log_std_ref - log_std) + (var + (mean - mean_ref) ** 2) / (2.0 * var_ref) - 0.5
    return kl.sum(dim=-1)


class KLAnchoredSAC(SAC):
    def __init__(self, *args, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.anchor = None

    def set_anchor(self, actor) -> None:
        """Freeze a deep copy of `actor` as the reference policy."""
        self.anchor = copy.deepcopy(actor).to(self.device)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.eval()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # ---- SB3 2.7.1 SAC.train, verbatim, with the three ANCHOR blocks added ----
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        anchor_kls = []  # ANCHOR

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # ANCHOR: KL(pi_theta || pi_clone) on the pre-tanh Gaussians.
            if self.anchor is not None and self.beta > 0.0:
                mean, log_std, _ = self.actor.get_action_dist_params(replay_data.observations)
                with th.no_grad():
                    mean_a, log_std_a, _ = self.anchor.get_action_dist_params(replay_data.observations)
                kl = gaussian_kl(mean, log_std, mean_a, log_std_a).mean()
                actor_loss = actor_loss + self.beta * kl
                anchor_kls.append(kl.item())
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # ANCHOR: always record so the key exists even when beta == 0.
        self.logger.record("train/anchor_kl", float(np.mean(anchor_kls)) if anchor_kls else 0.0)
```

Also add to `pe2/exploitability.py`, directly after `finetune_br`, a P-B fine-tune entry point (so the pilot script has one call per arm):

```python
def finetune_br_anchored(
    cfg, role, opponent_factory, model_path, total_timesteps, seed, beta,
    device="auto", gamma=0.999, critic_warmup=200_000, callback=None,
):
    """P-B: finetune_br with a KL anchor to the clone (pe2.anchored_sac)."""
    from stable_baselines3 import SAC  # lazy

    from pe2.anchored_sac import KLAnchoredSAC

    wrapper = SingleAgentWrapper(cfg, role, opponent_factory, seed0=seed * 1000)
    model = KLAnchoredSAC(
        ln_sac_policy(), wrapper, beta=beta, seed=seed, verbose=0, device=device,
        gamma=gamma, ent_coef=1e-5,
    )
    bc = SAC.load(model_path, device=device)
    model.actor.load_state_dict(bc.actor.state_dict())
    model.set_anchor(model.actor)
    for p in model.actor.parameters():
        p.requires_grad_(False)
    model.learn(total_timesteps=critic_warmup, callback=callback)
    for p in model.actor.parameters():
        p.requires_grad_(True)
    remaining = total_timesteps - critic_warmup
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    return model
```

- [ ] **Step 4: Run the tests to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_anchored_sac.py -v --no-header`
Expected: 6 PASS. If `test_large_beta_keeps_the_actor_near_the_anchor` is flaky, raise the gradient steps to 60 rather than loosening the inequality; it compares two deterministic runs at the same seed.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/anchored_sac.py pe2/exploitability.py tests/pe2/test_anchored_sac.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/anchored_sac.py pe2/exploitability.py tests/pe2/test_anchored_sac.py
git commit -m "feat(pe2): KLAnchoredSAC — pilot P-B, KL-to-clone term in the SAC actor loss (bounded SB3 override)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: PPO behaviour-clone + fine-tune (pilot P-C)

**Files:**
- Modify: `pe2/exploitability.py` (append after `train_bc_actor`)
- Test: `tests/pe2/test_exploitability.py` (append)

**Interfaces:**
- Consumes: `collect_demonstrations` output (tuples `(obs, action, ...)`), SB3 `PPO` (`policy.get_distribution(obs).distribution.mean`, `policy.parameters()`), `SingleAgentWrapper`.
- Produces:
  ```python
  PPO_KWARGS = dict(gamma=0.999, gae_lambda=0.95, n_steps=2048, batch_size=64, n_epochs=10,
                    ent_coef=0.0, policy_kwargs={"log_std_init": -1.0})
  def train_bc_ppo(cfg, role, opponent_factory, demonstrations, seed, device="auto",
                   epochs=20, batch_size=512, lr=1e-3) -> tuple[PPO, list[float]]
  def finetune_ppo(cfg, role, opponent_factory, model_path, total_timesteps, seed,
                   device="auto", callback=None) -> PPO
  ```
  `train_bc_ppo` regresses the Gaussian mean (unsquashed; PPO clips at execution) onto the demo actions with MSE. `finetune_ppo` loads the saved PPO onto a fresh training env and calls `learn`.

- [ ] **Step 1: Write the failing tests** (append to `tests/pe2/test_exploitability.py`; extend the import block with `train_bc_ppo, finetune_ppo, PPO_KWARGS`)

```python
class TestPpoPath:
    def test_ppo_kwargs_are_the_pre_registered_ones(self):
        assert PPO_KWARGS["gamma"] == 0.999
        assert PPO_KWARGS["gae_lambda"] == 0.95
        assert PPO_KWARGS["ent_coef"] == 0.0
        assert PPO_KWARGS["policy_kwargs"]["log_std_init"] == -1.0

    def test_bc_ppo_loss_descends_and_model_predicts(self):
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig(n_obstacles=0)
        demos = collect_demonstrations(cfg, "pursuer", FovLimitedPursuer, GreedyEvader, 2, 750950)
        model, losses = train_bc_ppo(cfg, "pursuer", GreedyEvader, demos, seed=0, epochs=5, device="cpu")
        assert losses[-1] < losses[0]
        a, _ = model.predict(demos[0][0], deterministic=True)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0 + 1e-6)

    def test_finetune_ppo_runs_and_evaluates(self, tmp_path):
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig(n_obstacles=0)
        demos = collect_demonstrations(cfg, "pursuer", FovLimitedPursuer, GreedyEvader, 1, 750960)
        model, _ = train_bc_ppo(cfg, "pursuer", GreedyEvader, demos, seed=0, epochs=1, device="cpu")
        path = tmp_path / "ppo_bc.zip"
        model.save(path)
        tuned = finetune_ppo(cfg, "pursuer", GreedyEvader, path, total_timesteps=2048, seed=0, device="cpu")
        logs = evaluate_sb3_policy(tuned, cfg, "pursuer", GreedyEvader, n_episodes=1, seed0=780950)
        assert logs[0]["outcome"] in ("capture", "timeout")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py::TestPpoPath -v --no-header`
Expected: ImportError on `train_bc_ppo`.

- [ ] **Step 3: Implement** (append to `pe2/exploitability.py` after `train_bc_actor`)

```python
PPO_KWARGS = dict(
    gamma=0.999,
    gae_lambda=0.95,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.0,
    policy_kwargs={"log_std_init": -1.0},  # sigma ~ 0.37 so the clone survives rollout noise
)


def train_bc_ppo(
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    demonstrations: list[tuple],
    seed: int,
    device: str = "auto",
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
):
    """P-C first half (D84): behaviour-clone a PPO policy on demonstrations.

    PPO's DiagGaussian mean is unsquashed and clipped to the Box at
    execution, so the regression target is the demo action directly (no
    tanh). Returns (model, per-epoch losses)."""
    import torch  # lazy
    from stable_baselines3 import PPO

    wrapper = SingleAgentWrapper(cfg, role, opponent_factory, seed0=seed * 1000)
    model = PPO("MlpPolicy", wrapper, seed=seed, verbose=0, device=device, **PPO_KWARGS)
    obs = torch.as_tensor(np.stack([d[0] for d in demonstrations]), dtype=torch.float32).to(model.device)
    acts = torch.as_tensor(np.stack([d[1] for d in demonstrations]), dtype=torch.float32).to(model.device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(demonstrations)
    losses = []
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        perm = rng.permutation(n)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i : i + batch_size], device=model.device)
            mean = model.policy.get_distribution(obs[idx]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, acts[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            count += len(idx)
        losses.append(total / count)
    model.policy.set_training_mode(False)
    return model, losses


def finetune_ppo(
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    model_path,
    total_timesteps: int,
    seed: int,
    device: str = "auto",
    callback=None,
):
    """P-C second half: PPO+GAE fine-tune of the cloned PPO policy on the
    terminal-only objective (lambda-returns instead of one-step TD — the
    horizon-structure hypothesis of D84)."""
    from stable_baselines3 import PPO  # lazy

    wrapper = SingleAgentWrapper(cfg, role, opponent_factory, seed0=seed * 1000)
    model = PPO.load(model_path, env=wrapper, device=device, seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_exploitability.py -q --no-header`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./venv/bin/ruff format pe2/exploitability.py tests/pe2/test_exploitability.py && ./venv/bin/ruff check pe2 tests/pe2
git add pe2/exploitability.py tests/pe2/test_exploitability.py
git commit -m "feat(pe2/exploitability): PPO behaviour-clone + PPO+GAE fine-tune (pilot P-C, D84)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Pilot runner script + pass rule + figure

**Files:**
- Create: `scripts/derisk_t0_pilots.py`
- Test: `tests/pe2/test_derisk.py` (append: pass-rule unit test on the pure function)
- Modify: `pe2/derisk.py` (append the pure functions the script uses)

**Interfaces:**
- Consumes: `finetune_br(..., env_wrap=, critic_warmup=)`, `finetune_br_anchored(..., beta=)`, `train_bc_ppo`/`finetune_ppo`, `CheckpointEvalProbe`, `DenseRewardWrapper`, the gated clone zip, `collect_demonstrations` (for the PPO clone), SB3 `BaseCallback` (`self.locals["infos"]`, `self.locals["dones"]`).
- Produces in `pe2/derisk.py`:
  ```python
  CLONE_CI_LO = 0.137          # gated SAC clone's Wilson lower bound (S119, N=200)
  PASS_FROM_STEP = 100_000
  def pilot_passes(rows: list[dict], threshold=CLONE_CI_LO, from_step=PASS_FROM_STEP) -> bool
  class TrainingCaptureCounter  # SB3 callback: counts capture episodes and env steps during training
  ```
  and the script writes `results/derisk/t0_pilots.json` (per arm: probe rows, pass, training positives per 1M, kill-rule verdict for P-A) and `t0_pilots.png` (all arms' win-rate curves + the 0.137 line).

- [ ] **Step 1: Write the failing test** (append to `tests/pe2/test_derisk.py`; extend the import to `from pe2.derisk import CLONE_CI_LO, mc_returns, pilot_passes, signal_audit`)

```python
class TestPilotPassRule:
    def test_pass_requires_every_probe_from_100k(self):
        rows = [
            {"steps": 50_000, "win_rate": 0.05},   # before the window: ignored
            {"steps": 100_000, "win_rate": 0.150},
            {"steps": 150_000, "win_rate": 0.140},
            {"steps": 200_000, "win_rate": 0.137},  # equal to the bound passes
        ]
        assert pilot_passes(rows) is True
        rows[2]["win_rate"] = 0.136
        assert pilot_passes(rows) is False

    def test_no_probe_in_window_fails(self):
        assert pilot_passes([{"steps": 50_000, "win_rate": 0.9}]) is False

    def test_bound_is_the_gated_clone_ci_lo(self):
        assert CLONE_CI_LO == 0.137
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py::TestPilotPassRule -v --no-header`
Expected: ImportError on `pilot_passes`.

- [ ] **Step 3: Implement the pure parts** (append to `pe2/derisk.py`)

```python
CLONE_CI_LO = 0.137  # gated SAC clone 0.185 [0.137, 0.245] at N=200 (S119 D76 gate)
PASS_FROM_STEP = 100_000


def pilot_passes(
    rows: list[dict], threshold: float = CLONE_CI_LO, from_step: int = PASS_FROM_STEP
) -> bool:
    """Pre-registered T0.3 rule: win rate >= threshold at EVERY probe from
    `from_step` onward; no probe in the window -> fail."""
    window = [r for r in rows if r["steps"] >= from_step]
    return bool(window) and all(r["win_rate"] >= threshold for r in window)


class TrainingCaptureCounter:
    """SB3 callback: count training episodes ending in capture, and steps,
    so the T0.1 kill rule (positives per 1M) can be read off the P-A run."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback  # lazy

        state = {"steps": 0, "captures": 0, "episodes": 0}

        class _CB(BaseCallback):
            def _on_step(inner) -> bool:
                state["steps"] += 1
                for done, info in zip(inner.locals["dones"], inner.locals["infos"]):
                    if done:
                        state["episodes"] += 1
                        state["captures"] += int(info.get("outcome") == "capture")
                return True

        self.state = state
        self.callback = _CB()

    @property
    def positives_per_1M(self) -> float:
        return self.state["captures"] / max(self.state["steps"], 1) * 1e6
```

- [ ] **Step 4: Write the script**

```python
# scripts/derisk_t0_pilots.py
"""T0.3 pilots (derisk plan v2): three mechanism-matched 200k fine-tunes of
the gated SAC clone + the D78 naive negative control.

  P-A  change the signal   : SAC fine-tune + DenseRewardWrapper(w_vis, w_search)
  P-B  stop the drift      : KL-anchored SAC fine-tune (beta)
  P-C  change the horizon  : PPO-BC clone -> PPO+GAE fine-tune
  CTRL negative control    : naive SAC fine-tune, critic_warmup=0 (must collapse)

Pre-registered (before any run): critic_warmup=50k, total=200k, probe every
50k on N=200 paired seeds 780000+, pass = win rate >= 0.137 at every probe
from 100k. Kill rule: P-A training positives/1M < 50 -> Route C.

Usage: ./venv/bin/python scripts/derisk_t0_pilots.py --arms PA PB PC CTRL [--timesteps 200000]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402

from pe2.config import EnvConfig  # noqa: E402
from pe2.derisk import CLONE_CI_LO, TrainingCaptureCounter, pilot_passes  # noqa: E402
from pe2.exploitability import (  # noqa: E402
    CheckpointEvalProbe,
    collect_demonstrations,
    finetune_br,
    finetune_br_anchored,
    finetune_ppo,
    train_bc_ppo,
)
from pe2.rewards import DenseRewardWrapper  # noqa: E402
from pe2.scripted import FovLimitedPursuer, GreedyEvader  # noqa: E402

OUT = pathlib.Path("results/derisk")
CLONE = pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip")
BASELINES = pathlib.Path("results/visual_gates/step_05/step_05_baselines_pursuer.json")
EVAL_SEED0 = 780000
SEEDS = {"PA": 7, "PB": 8, "PC": 9, "CTRL": 10}
W_VIS, W_SEARCH, BETA = 0.1, 1e-4, 1.0
KILL_POSITIVES_PER_1M = 50


def probe(cfg, label, every, eval_n):
    d = OUT / f"ckpts_{label}"
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointEvalProbe(
        cfg, "pursuer", GreedyEvader, every=every, eval_n=eval_n, seed0=EVAL_SEED0,
        ckpt_dir=d, report_path=OUT / f"t0_pilot_{label}.json", label=label,
    )


def run_arm(arm, cfg, timesteps, warmup, every, eval_n, device):
    pr = probe(cfg, arm, every, eval_n)
    counter = TrainingCaptureCounter()
    cb = CallbackList([pr.callback, counter.callback])
    seed = SEEDS[arm]
    if arm == "PA":
        finetune_br(
            cfg, "pursuer", GreedyEvader, CLONE, timesteps, seed=seed, device=device,
            critic_warmup=warmup, callback=cb,
            env_wrap=lambda e: DenseRewardWrapper(e, w_vis=W_VIS, w_search=W_SEARCH),
        )
    elif arm == "PB":
        finetune_br_anchored(
            cfg, "pursuer", GreedyEvader, CLONE, timesteps, seed=seed, beta=BETA,
            device=device, critic_warmup=warmup, callback=cb,
        )
    elif arm == "PC":
        demos = collect_demonstrations(cfg, "pursuer", FovLimitedPursuer, GreedyEvader, 300, 750000)
        bc, losses = train_bc_ppo(cfg, "pursuer", GreedyEvader, demos, seed=seed, device=device, epochs=50)
        bc_path = OUT / "ppo_bc_pursuer.zip"
        bc.save(bc_path)
        finetune_ppo(cfg, "pursuer", GreedyEvader, bc_path, timesteps, seed=seed, device=device, callback=cb)
    elif arm == "CTRL":
        finetune_br(
            cfg, "pursuer", GreedyEvader, CLONE, timesteps, seed=seed, device=device,
            critic_warmup=0, callback=cb,
        )
    else:
        raise SystemExit(f"unknown arm {arm}")
    rows = pr.rows
    result = {
        "arm": arm, "seed": seed, "timesteps": timesteps, "critic_warmup": warmup,
        "eval_n": eval_n, "eval_seed0": EVAL_SEED0, "checkpoints": rows,
        "pass": pilot_passes(rows),
        "training": dict(counter.state, positives_per_1M=counter.positives_per_1M),
    }
    if arm == "PA":
        result["kill_rule"] = {
            "threshold": KILL_POSITIVES_PER_1M,
            "fires": counter.positives_per_1M < KILL_POSITIVES_PER_1M,
        }
    print(f"[{arm}] pass={result['pass']} positives/1M={counter.positives_per_1M:.1f}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["PA", "PB", "PC", "CTRL"])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=50_000)
    p.add_argument("--every", type=int, default=50_000)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not CLONE.exists():
        raise SystemExit(f"gated clone missing: {CLONE}")
    cfg = EnvConfig()
    results = {arm: run_arm(arm, cfg, args.timesteps, args.warmup, args.every, args.eval_n, args.device)
               for arm in args.arms}
    ctrl_ok = ("CTRL" not in results) or not results["CTRL"]["pass"]
    summary = {
        "clone_ci_lo": CLONE_CI_LO,
        "survivors": [a for a, r in results.items() if a != "CTRL" and r["pass"]],
        "negative_control_collapsed": ctrl_ok,
        "arms": results,
    }
    with open(OUT / "t0_pilots.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, r in results.items():
        ax.plot([c["steps"] for c in r["checkpoints"]], [c["win_rate"] for c in r["checkpoints"]],
                marker="o", label=f"{arm} ({'pass' if r['pass'] else 'fail'})")
    ax.axhline(CLONE_CI_LO, color="k", ls="--", label="clone CI-lo 0.137")
    ax.set(xlabel="training steps", ylabel=f"win rate (N={args.eval_n}, paired)", title="T0.3 pilots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "t0_pilots.png", dpi=120)
    print(f"survivors={summary['survivors']} negative_control_collapsed={ctrl_ok}")
    if not ctrl_ok:
        raise SystemExit("negative control did NOT collapse — pilot harness is broken, do not read the arms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the pure-function tests, then a smoke run of the script**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/pe2/test_derisk.py -v --no-header`
Expected: all PASS.

Smoke (CPU, ~minutes): `./venv/bin/python scripts/derisk_t0_pilots.py --arms PA CTRL --timesteps 400 --warmup 100 --every 200 --eval-n 2 --device cpu`
Expected: both arms print a line, `results/derisk/t0_pilots.{json,png}` written. The smoke run's pass/fail values are meaningless (tiny budgets) and must not be recorded anywhere — delete `results/derisk/t0_pilots.*` and `results/derisk/ckpts_*` after the smoke check so the real run starts clean.

- [ ] **Step 6: Lint and commit**

```bash
./venv/bin/ruff format pe2/derisk.py scripts/derisk_t0_pilots.py tests/pe2/test_derisk.py && ./venv/bin/ruff check pe2 tests/pe2 scripts
git add pe2/derisk.py scripts/derisk_t0_pilots.py tests/pe2/test_derisk.py
git commit -m "feat(pe2/derisk): T0.3 pilot runner — P-A/P-B/P-C + negative control, pre-registered pass rule

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Pre-registration record, then launch order

**Files:**
- Modify: `docs/worklogs/2026-09-04_S121.md`

No code. D83 requires the exact criteria on record before the runs.

- [ ] **Step 1: Append to "Work Done"**

```markdown
- **Tier 0 built** (plan `docs/superpowers/plans/2026-09-04-tier0-diagnostic.md`): `pe2/derisk.py` (signal audit, pass rule, capture counter), `pe2/rewards.py` (search-staleness tracker on pe2 FOV+LOS sensing; `DenseRewardWrapper` with visibility / search / evader-vis+PBRS and the non-degeneracy check), `pe2/anchored_sac.py` (KL-to-clone SAC), PPO-BC + PPO fine-tune, and the two scripts. **Pre-registered before any run**: pilots fine-tune the gated clone (0.185 [0.137, 0.245]) with `critic_warmup=50k`, `total=200k`, probe every 50k on N=200 paired seeds 780000+; **pass = win rate ≥ 0.137 at every probe from 100k**; negative control = D78 naive fine-tune (`critic_warmup=0`), must fall below 0.137 by 200k or the harness is broken. Fixed choices: P-A `w_vis=0.1`, `w_search=1e-4`; P-B `beta=1.0`; P-C `gamma=0.999`, `gae_lambda=0.95`, `n_steps=2048`, `batch=64`, `n_epochs=10`, `ent_coef=0`, `log_std_init=-1.0`. Kill rule: P-A training positives/1M < 50 → Route C. Training seeds 7/8/9/10. **Decision (D93)**: P-B overrides one SB3 method (`SAC.train`, copied verbatim + ~12 lines) — a bounded, documented exception to "RL libraries are not rewritten", confined to `pe2/anchored_sac.py`.
```

- [ ] **Step 2: Append the launch order to "Next Steps"**

```markdown
- [ ] **Launch T0.1**: `./venv/bin/python scripts/derisk_t0_signal_audit.py` (CPU, ~10 min) → record positives/1M, visible fraction, var(G_t) in this worklog
- [ ] **Launch T0.3** in tmux on niro-1 GPU: `./venv/bin/python scripts/derisk_t0_pilots.py` (all four arms; ~hours) → record the survivors, the negative-control collapse, and the P-A kill-rule verdict; sign the `t0_pilots.png` gate
- [ ] Apply the Tier-0 direction table (derisk v2) and log the outcome as a decision before any Tier-1 run
```

- [ ] **Step 3: Commit**

```bash
git add docs/worklogs/2026-09-04_S121.md
git commit -m "docs(S121): pre-register Tier 0 pilot criteria, fixed hyperparameters, and the P-B override exception (D93)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Add D93 to `docs/decisions.md` in the same commit if the executor has the file context; otherwise the controller adds it at finish.

---

## Self-review (done while writing)

- **Spec coverage**: T0.1 → Task 1; T0.2 (three terms + plateau/non-degeneracy checks) → Tasks 2–3; T0.3 P-A → Tasks 3+4+7, P-B → Task 5+7, P-C → Task 6+7, negative control → Task 7; kill rule → Tasks 1+7; pre-registration → Task 8. The Tier-0 direction table is applied after the runs (Task 8 next steps), not by code.
- **Placeholder scan**: none; every step has code or an exact command.
- **Type consistency**: `finetune_br(..., critic_warmup=, callback=, env_wrap=)` (Task 4) is what Task 7 calls; `finetune_br_anchored(..., beta=, critic_warmup=, callback=)` (Task 5) matches Task 7; `train_bc_ppo`/`finetune_ppo` (Task 6) match Task 7; `pilot_passes`/`TrainingCaptureCounter`/`CLONE_CI_LO` (Task 7 pure part) match the script; `DenseRewardWrapper(env, w_vis=, w_search=)` keyword-only args match the P-A lambda.
- **Known caveats**: (1) The PPO fine-tune has no critic warm-up analogue; its value net starts at init, so P-C may drift for a different reason than D82 — that is the experiment, not a bug. (2) `TrainingCaptureCounter` assumes a single (non-vectorized) env, which is what every `SingleAgentWrapper` path here creates. (3) The evader-side dense terms (`w_evis`, `w_pbrs`) are built and tested now but consumed only in Tier 1 T2.
