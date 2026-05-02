# BR Exploitability Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a fresh-init best-response exploitability test on s48 and s49 from the SP17b cohort (4 BR runs, 1.5 M steps each), then classify the cohort against an ε-Nash criterion to decide which paper framing the CoRL submission adopts.

**Architecture:** Two new Python scripts (`scripts/train_br_sac.py` for BR training, `scripts/analyze_exploitability.py` for verdict computation) plus operational shell glue. The training script reuses `PartialObsOpponentAdapter` and `SingleAgentPEWrapper` from the existing codebase — no orchestrator (`training/amsdrl.py`) involvement. The analyzer is a stand-alone post-hoc script that reads `history.json`-shaped outputs from the four BR runs and emits the cohort hypothesis label (H1–H5) defined in the spec.

**Tech Stack:** Python 3.12, stable-baselines3 SAC, PursuitEvasionEnv (project), `PartialObsOpponentAdapter` (project), pytest for unit tests. Two execution machines: niro-2 (BR-1, BR-2) and niro-1 (BR-3, BR-4). Code authored on niro-1 (this session's CWD) and synced to niro-2 by `rsync`/`scp`; commits happen in niro-2's git tree.

**Spec:** `docs/superpowers/specs/2026-05-02-paper-strengthening-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `scripts/train_br_sac.py` | NEW | Train a fresh SAC learner against a frozen SAC opponent loaded via `PartialObsOpponentAdapter`. CLI args drive role/seed/budget; emits `history.json` schema-compatible with the cohort. |
| `scripts/analyze_exploitability.py` | NEW | Read `history.json` from all 4 BR runs + cohort baseline files; compute `exploit_gap` per run; assign per-seed L1–L4 + cohort H1–H5; write verdict memo. |
| `scripts/snapshot_frozen_opponents.sh` | NEW | Copy `milestone_phase2550_*` from s48/s49 into read-only `results/BR_frozen/`. Idempotent. |
| `scripts/launch_br.sh` | NEW | Bash launcher that constructs the four BR commands with consistent paths/seeds. |
| `tests/test_br_sac_loading.py` | NEW | Unit test: SAC opponent loads, frozen-side params hash unchanged after one trainer step. |
| `tests/test_exploitability_analyzer.py` | NEW | Unit test: classifier emits correct L1–L4 / H1–H5 labels for synthetic histories covering each hypothesis. |
| `docs/worklogs/2026-05-02_S88.md` | NEW | Session worklog (CLAUDE.md mandate). |
| `docs/worklogs/2026-05-02_BR_verdict.md` | NEW | Written by analyzer at end. |
| `docs/decisions.md` | MODIFY | Append D34 (BR test methodology), D35 (snapshot point M2550 rationale). |
| `docs/workflow_tracker.md` | MODIFY | Append S88 entry to session index. |
| `scripts/train_pursuer_vs_evader.py` | UNCHANGED | Existing PPO BR script kept as reference; do not modify. |
| `training/amsdrl.py` | UNCHANGED | Cohort orchestrator; BR script bypasses it. |
| `envs/opponent_adapter.py` | UNCHANGED | Used as-is. |
| `envs/wrappers.py` | UNCHANGED | Used as-is. |

**File-naming note.** The cohort saves SAC weights to a file named `ppo.zip` inside each milestone dir (legacy naming hardcoded in `training/amsdrl.py`). The plan loads them with `SAC.load(...)`. Do not be tricked by the `ppo.zip` filename.

---

## Task 0: Session bootstrap

**Files:**
- Create: `docs/worklogs/2026-05-02_S88.md`
- Modify: `docs/workflow_tracker.md`

- [ ] **Step 0.1: Create the S88 worklog**

Author the worklog using the project template (CLAUDE.md mandate). Required sections: header (date, session, previous link), Context, Objectives, Work Done (filled progressively), Files Changed, Decisions Made, Issues & Blockers, Next Steps.

Write `docs/worklogs/2026-05-02_S88.md` with this initial scaffold:

```markdown
# Session 88: BR Exploitability Test — Implementation & Run

**Date:** 2026-05-02
**Session:** S88
**Previous Session:** [S87](2026-05-01_S87.md)

## Context

S87 launched the SP17b convergence cohort and drafted the CoRL abstract claiming
"Nash equilibrium convergence at capture rate 0.51 ± 0.03". S88 trajectory analysis
on 2026-05-02 revealed cohort-wide curriculum-end drift: every seed drops CR by
≈0.12 at p_full=0 (M2550). The "0.51" figure is a curriculum artifact, not the
equilibrium. This session implements and runs the BR exploitability test specified
in `docs/superpowers/specs/2026-05-02-paper-strengthening-design.md` to determine
which paper framing (A/B/C) the CoRL submission adopts.

## Objectives

1. Snapshot s48/s49 M2550 frozen opponents into `results/BR_frozen/`.
2. Implement `scripts/train_br_sac.py` (SAC BR trainer) and `scripts/analyze_exploitability.py`.
3. Smoke-test BR-1 50K steps; halt if smoke fails.
4. Run BR-1..BR-4 in parallel (niro-2 + niro-1), 1.5 M steps each.
5. Run analyzer; decide cohort hypothesis H1–H5; commit framing decision.

## Work Done

(updated per task as the plan executes)

## Files Changed

(updated per task)

## Decisions Made

(updated per task)

## Issues & Blockers

(updated per task)

## Next Steps

(updated at session close)
```

- [ ] **Step 0.2: Add S88 to the session index**

Append a one-liner to `docs/workflow_tracker.md`'s Session Index section. Read the file first to find the index, then add a row matching the existing format. Example:

```
| S88 | 2026-05-02 | BR exploitability test for SP17b cohort | docs/worklogs/2026-05-02_S88.md |
```

(Match the actual table columns in the file when editing.)

- [ ] **Step 0.3: Commit on niro-2**

Code in this working directory (`/home/niro-1/Codes/safe-rl-pe`) is not under git; the canonical git tree lives on niro-2 (`/home/niro-2/Codes/safe-rl-pe`, branch `main`). For each "commit" step in this plan, sync the new/edited files to niro-2 first, then commit there.

```bash
# From this session's CWD on niro-1:
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    docs/worklogs/2026-05-02_S88.md \
    docs/workflow_tracker.md \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/docs/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md docs/workflow_tracker.md && git commit -m "docs(S88): bootstrap session worklog for BR exploitability test"'
```

Expected: `git commit` returns "1 file changed" (worklog) plus a workflow_tracker line edit. If the commit fails because of pre-commit hooks, fix the underlying issue and create a new commit (do not amend per CLAUDE.md).

---

## Task 1: Snapshot frozen opponents (data, no code yet)

**Files:**
- Create: `scripts/snapshot_frozen_opponents.sh`
- Create on niro-2: `results/BR_frozen/s{48,49}/{pursuer,evader}/`

This task only copies files. We do it BEFORE killing s48 (per spec §5.1) so the snapshot reflects the policy at M2550 unambiguously, even if s48 has continued to drift.

**Phase choice — verified.** Both s48 and s49 have a `milestone_phase2550_*` directory under `results/SP17b_s{48,49}/checkpoints/{pursuer,evader}/` (verified on niro-2 during S88 trajectory analysis). M2550 is the last completed micro-phase before drift onset for both seeds. If a future re-run targets a different cohort where the per-seed drift onset differs, edit the `PHASE` variable in the snapshot script to a per-seed dict and use the latest phase with `p_full > 0.05` for each (per spec §5.1). For this run, a single hard-coded phase is correct.

- [ ] **Step 1.1: Write the snapshot script**

Create `scripts/snapshot_frozen_opponents.sh`:

```bash
#!/usr/bin/env bash
# Copy SP17b s48/s49 milestone_phase2550 checkpoints into a read-only
# BR_frozen/ snapshot directory. Idempotent — safe to re-run.
#
# Usage (run on niro-2): bash scripts/snapshot_frozen_opponents.sh
set -euo pipefail

BASE="/home/niro-2/Codes/safe-rl-pe/results"
DEST="${BASE}/BR_frozen"
PHASE="milestone_phase2550"

mkdir -p "${DEST}"

for seed in 48 49; do
    for role in pursuer evader; do
        src="${BASE}/SP17b_s${seed}/checkpoints/${role}/${PHASE}_${role}"
        dst_dir="${DEST}/s${seed}/${role}"

        if [[ ! -d "${src}" ]]; then
            echo "ERROR: source missing: ${src}" >&2
            exit 1
        fi

        mkdir -p "${dst_dir}"
        cp -r "${src}/." "${dst_dir}/"
        chmod -R a-w "${dst_dir}"   # read-only — protect from accidental edit
        echo "  snapshot: ${src} -> ${dst_dir}"
    done
done

echo "Snapshot complete. Contents:"
find "${DEST}" -maxdepth 3 -type f | sort
```

- [ ] **Step 1.2: Sync the script to niro-2 and run it**

```bash
# From niro-1:
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    scripts/snapshot_frozen_opponents.sh \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/scripts/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && chmod +x scripts/snapshot_frozen_opponents.sh && bash scripts/snapshot_frozen_opponents.sh'
```

Expected output ends with 8 file paths under `results/BR_frozen/`:
```
results/BR_frozen/s48/evader/meta.json
results/BR_frozen/s48/evader/ppo.zip
results/BR_frozen/s48/pursuer/meta.json
results/BR_frozen/s48/pursuer/ppo.zip
results/BR_frozen/s49/evader/meta.json
results/BR_frozen/s49/evader/ppo.zip
results/BR_frozen/s49/pursuer/meta.json
results/BR_frozen/s49/pursuer/ppo.zip
```

If any path is missing, fix and re-run.

- [ ] **Step 1.3: Sync the snapshots back to niro-1 (for BR-3, BR-4)**

niro-1 runs BR-3 and BR-4. It needs the s49 snapshots locally:

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/results/BR_frozen/ \
    /home/niro-1/Codes/safe-rl-pe/results/BR_frozen/
```

Expected: 8 files transferred. (Both seeds copied to keep symmetry; BR-3/BR-4 only need s49 but shipping both is harmless and saves a future round-trip.)

- [ ] **Step 1.4: Commit snapshot script (not the snapshots themselves — too large)**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add scripts/snapshot_frozen_opponents.sh && git commit -m "feat(S88): snapshot frozen opponents script for BR test"'
```

Note: `results/` is gitignored per project convention; the snapshots remain on disk but are not committed. The script captures the procedure.

---

## Task 2: Write SAC BR training script — Phase 1 (skeleton + opponent loader)

**Files:**
- Create: `scripts/train_br_sac.py` (lines 1–~120)
- Test: `tests/test_br_sac_loading.py`

This is built TDD-style: the unit test verifies that (a) the SAC checkpoint loads, and (b) calling the trainer for one update step does NOT mutate the frozen side's parameters. We must not let the frozen opponent silently train alongside the BR side.

- [ ] **Step 2.1: Write the failing test for SAC loading + frozen invariance**

Create `tests/test_br_sac_loading.py`:

```python
"""Verify that train_br_sac.py loads a SAC checkpoint as a frozen opponent
and that the frozen side's parameters do not change after a trainer update step.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

# Skip if no snapshot is available (CI/dev machines without niro-2 mount).
SNAP = Path("results/BR_frozen/s48/evader")
pytestmark = pytest.mark.skipif(
    not (SNAP / "ppo.zip").exists(),
    reason="snapshot results/BR_frozen/s48/evader/ppo.zip not available",
)


def _hash_params(model: SAC) -> str:
    """Return a stable hash of all policy + critic + target params."""
    h = hashlib.sha256()
    for tensor in model.policy.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_load_snapshot_returns_sac():
    """SAC.load on the snapshot returns a SAC instance with the expected obs space."""
    model = SAC.load(SNAP / "ppo.zip", device="cpu")
    assert isinstance(model, SAC)
    # The cohort uses Dict obs (obs_history + lidar + state) per partial_obs_wrapper.
    assert hasattr(model.observation_space, "spaces"), \
        "expected Dict observation space (partial-obs policy)"
    assert {"obs_history", "lidar", "state"}.issubset(model.observation_space.spaces.keys())


def test_frozen_opponent_params_unchanged_after_trainer_step(tmp_path):
    """Round-trip: build the BR trainer, run one update step, assert frozen
    opponent's state_dict hash is unchanged."""
    from scripts.train_br_sac import build_br_setup, run_one_update

    setup = build_br_setup(
        frozen_opponent_path=str(SNAP),
        frozen_role="evader",
        seed=148,
        output_dir=str(tmp_path / "br_smoke"),
        env_kwargs=None,  # use cohort defaults
    )
    h_before = _hash_params(setup.frozen_model)
    run_one_update(setup, n_train_steps=200)  # tiny update batch
    h_after = _hash_params(setup.frozen_model)
    assert h_before == h_after, "frozen opponent parameters were mutated"
```

- [ ] **Step 2.2: Run test, expect ImportError / collection error**

```bash
./venv/bin/python -m pytest tests/test_br_sac_loading.py -v
```

Expected: FAIL — `ImportError: cannot import name 'build_br_setup' from 'scripts.train_br_sac'` (script does not exist yet). If the snapshot is unavailable, the test is skipped — that's also acceptable (we'll verify post-snapshot in Task 6).

- [ ] **Step 2.3: Write the train_br_sac.py skeleton + opponent loader**

Create `scripts/train_br_sac.py`:

```python
"""Train a fresh SAC learner against a FROZEN SAC opponent for the BR
exploitability test (CoRL 2026 paper).

Spec: docs/superpowers/specs/2026-05-02-paper-strengthening-design.md

The frozen opponent is loaded from a `milestone_phase*` snapshot dir
(containing `ppo.zip` despite the legacy filename — it is SAC weights).
Wrapped with `PartialObsOpponentAdapter` so the frozen side observes the
world through its own sensors, identical to how the cohort trained it.

The trained side runs SAC with cohort-identical hyperparameters, p_full
fixed at 0.0 (full partial obs, no curriculum), no PFSP, no alternating
freeze, no forced switch — vanilla single-side training against a
fixed adversary.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper


# Cohort-identical hyperparameters (from scripts/launch_sp17b.sh).
COHORT_ENV_KWARGS = {
    "arena_width": 20.0,
    "arena_height": 20.0,
    "max_steps": 1200,
    "capture_radius": 0.5,
    "n_obstacles_min": 0,
    "n_obstacles_max": 3,
    "n_obstacle_obs": 3,
    "pursuer_v_max": 1.0,
    "evader_v_max": 1.0,
    "distance_scale": 1.0,
    "use_visibility_reward": True,
    "visibility_weight": 0.2,
    "survival_bonus": 0.0,
    "timeout_penalty": 0.0,
    "capture_bonus": 50.0,
    "partial_obs": True,
    "sensing_radius": 8.0,
    "combined_masking": True,
}

COHORT_SAC_KWARGS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "learning_starts": 10_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 0.03,           # fixed, NOT auto (D2 / D3)
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": {"net_arch": [256, 256]},
}


@dataclass
class BRSetup:
    """Aggregate of the live objects the BR trainer needs."""
    base_env: PursuitEvasionEnv
    wrapper: SingleAgentPEWrapper
    frozen_model: SAC
    learner: SAC
    output_dir: Path
    trained_role: str
    frozen_role: str


def _make_reward_computer(env_kwargs: dict) -> RewardComputer:
    diagonal = float(np.sqrt(env_kwargs["arena_width"] ** 2
                             + env_kwargs["arena_height"] ** 2))
    return RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 0.1),
        survival_bonus=env_kwargs.get("survival_bonus", 0.0),
        timeout_penalty=env_kwargs.get("timeout_penalty", 0.0),
        capture_bonus=env_kwargs.get("capture_bonus", 50.0),
    )


def _make_partial_obs_env(env_kwargs: dict) -> PursuitEvasionEnv:
    """Construct the same env shape the cohort trains on."""
    rc = _make_reward_computer(env_kwargs)
    base = PursuitEvasionEnv(
        arena_width=env_kwargs["arena_width"],
        arena_height=env_kwargs["arena_height"],
        max_steps=env_kwargs["max_steps"],
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles_max", 2),
        pursuer_v_max=env_kwargs["pursuer_v_max"],
        evader_v_max=env_kwargs["evader_v_max"],
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs",
                                       env_kwargs.get("n_obstacles_max", 2)),
        reward_computer=rc,
        partial_obs=env_kwargs["partial_obs"],
        n_obstacles_min=env_kwargs.get("n_obstacles_min"),
        n_obstacles_max=env_kwargs.get("n_obstacles_max"),
        asymmetric_obs=False,
        sensing_radius=env_kwargs.get("sensing_radius"),
        combined_masking=env_kwargs.get("combined_masking", False),
    )
    return PartialObsWrapper(base)


def build_br_setup(
    frozen_opponent_path: str,
    frozen_role: str,
    seed: int,
    output_dir: str,
    env_kwargs: dict | None = None,
) -> BRSetup:
    """Construct env + frozen opponent + fresh learner. No training yet.

    Args:
        frozen_opponent_path: Directory containing `ppo.zip` (cohort snapshot).
        frozen_role: 'pursuer' or 'evader' — the role the FROZEN side plays.
        seed: Random seed for the trainer.
        output_dir: Where eval log + checkpoints are written.
        env_kwargs: Override of COHORT_ENV_KWARGS (test/smoke-only). None ⇒ cohort.
    """
    if frozen_role not in ("pursuer", "evader"):
        raise ValueError(f"frozen_role must be 'pursuer' or 'evader', got {frozen_role!r}")
    trained_role = "evader" if frozen_role == "pursuer" else "pursuer"

    env_kwargs = dict(env_kwargs) if env_kwargs is not None else dict(COHORT_ENV_KWARGS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build base env (the SingleAgentPEWrapper will wrap it).
    partial_obs_env = _make_partial_obs_env(env_kwargs)
    base_env = partial_obs_env.unwrapped  # raw PursuitEvasionEnv for adapter

    # 2) Load frozen SAC opponent.
    zip_path = Path(frozen_opponent_path) / "ppo.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"frozen opponent zip missing: {zip_path}")
    frozen_model = SAC.load(zip_path, device="cpu")

    # 3) Wrap into PartialObsOpponentAdapter — gives frozen side its own
    #    sensor pipeline + history buffer, identical to cohort training.
    opp_adapter = PartialObsOpponentAdapter(
        model=frozen_model,
        role=frozen_role,
        base_env=base_env,
        deterministic=False,
    )

    # 4) Build SingleAgentPEWrapper for the trained side.
    #    p_full_obs=0.0 ⇒ always partial obs. NO CURRICULUM (per spec §4).
    wrapper = SingleAgentPEWrapper(
        partial_obs_env,
        role=trained_role,
        opponent_policy=opp_adapter,
        p_full_obs=0.0,
    )
    monitored = Monitor(wrapper)

    # 5) Fresh SAC learner.
    learner = SAC("MultiInputPolicy", monitored, seed=seed, verbose=0,
                  **COHORT_SAC_KWARGS)

    return BRSetup(
        base_env=base_env,
        wrapper=monitored,
        frozen_model=frozen_model,
        learner=learner,
        output_dir=out,
        trained_role=trained_role,
        frozen_role=frozen_role,
    )


def run_one_update(setup: BRSetup, n_train_steps: int = 200) -> None:
    """Run a small `learn()` call. Used by tests to verify frozen invariance."""
    setup.learner.learn(total_timesteps=n_train_steps,
                        reset_num_timesteps=False, progress_bar=False)
```

- [ ] **Step 2.4: Run the test — expect PASS**

```bash
./venv/bin/python -m pytest tests/test_br_sac_loading.py -v
```

Expected:
- If snapshot is present (after Task 1): both tests PASS.
- If snapshot is absent: both SKIPPED with "snapshot ... not available" — that is also green.

If `test_frozen_opponent_params_unchanged_after_trainer_step` FAILS — i.e. the frozen model's hash changes after a trainer step — there is a real bug: investigate. Possibilities: (a) a shared optimizer is touching the frozen params; (b) `PartialObsOpponentAdapter` accidentally calls `model.train()` — it should not.

- [ ] **Step 2.5: Commit**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    scripts/train_br_sac.py tests/test_br_sac_loading.py \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add scripts/train_br_sac.py tests/test_br_sac_loading.py && git commit -m "feat(S88): SAC BR trainer skeleton + frozen-invariance test"'
```

---

## Task 3: Add training loop + eval + history.json output to `train_br_sac.py`

**Files:**
- Modify: `scripts/train_br_sac.py` (append eval + main)

- [ ] **Step 3.1: Append the evaluation function**

Append to `scripts/train_br_sac.py`:

```python
def evaluate(setup: BRSetup, n_episodes: int = 200) -> tuple[float, float]:
    """Roll out n_episodes deterministic episodes; return (capture_rate, avg_steps).

    Build a SEPARATE eval env so the replay buffer is not polluted by eval rollouts.
    """
    eval_partial = _make_partial_obs_env(_eval_env_kwargs(setup))
    eval_base = eval_partial.unwrapped
    eval_opp = PartialObsOpponentAdapter(
        model=setup.frozen_model,
        role=setup.frozen_role,
        base_env=eval_base,
        deterministic=True,
    )
    eval_env = SingleAgentPEWrapper(
        eval_partial, role=setup.trained_role,
        opponent_policy=eval_opp, p_full_obs=0.0,
    )

    captures = 0
    steps_list: list[int] = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        terminated = truncated = False
        ep_steps = 0
        while not (terminated or truncated):
            action, _ = setup.learner.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = eval_env.step(action)
            ep_steps += 1
        if terminated:
            captures += 1
        steps_list.append(ep_steps)
    eval_env.close()
    return captures / n_episodes, float(np.mean(steps_list))


def _eval_env_kwargs(setup: BRSetup) -> dict:
    """Return the env_kwargs the setup was built with (cohort defaults)."""
    # The setup does not carry env_kwargs explicitly; reconstruct from cohort.
    return dict(COHORT_ENV_KWARGS)
```

- [ ] **Step 3.2: Append the main training loop**

Append to `scripts/train_br_sac.py`:

```python
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen_opponent_path", required=True,
                   help="Dir containing ppo.zip of the frozen SAC opponent.")
    p.add_argument("--frozen_role", required=True, choices=("pursuer", "evader"),
                   help="Role of the FROZEN side.")
    p.add_argument("--total_steps", type=int, default=1_500_000)
    p.add_argument("--eval_freq",   type=int, default=250_000)
    p.add_argument("--n_eval_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    setup = build_br_setup(
        frozen_opponent_path=args.frozen_opponent_path,
        frozen_role=args.frozen_role,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("=" * 70)
    print("BR Exploitability Test")
    print(f"  Frozen role:    {setup.frozen_role}")
    print(f"  Trained role:   {setup.trained_role}")
    print(f"  Frozen path:    {args.frozen_opponent_path}")
    print(f"  Total steps:    {args.total_steps:,}")
    print(f"  Eval cadence:   every {args.eval_freq:,} steps, "
          f"{args.n_eval_episodes} episodes")
    print(f"  Seed:           {args.seed}")
    print(f"  Output:         {setup.output_dir}")
    print("=" * 70)

    history: list[dict] = []
    best_metric = -1.0  # higher is better for the trained side
    best_step = 0
    start = time.time()
    steps_done = 0

    while steps_done < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_done)
        setup.learner.learn(total_timesteps=chunk,
                            reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk

        cr, avg_ep = evaluate(setup, n_episodes=args.n_eval_episodes)
        # For pursuer BR: high CR good. For evader BR: low CR good.
        metric = cr if setup.trained_role == "pursuer" else (1.0 - cr)
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            best_step = steps_done
            setup.learner.save(setup.output_dir / "best_model")

        elapsed = time.time() - start
        history.append({
            "phase": f"BR{steps_done}",
            "total_steps": steps_done,
            "capture_rate": cr,
            "escape_rate": 1.0 - cr,
            "avg_ep_len": avg_ep,
            "frozen": setup.frozen_role,
            "trained": setup.trained_role,
            "p_full_obs": 0.0,
            "elapsed_seconds": elapsed,
            "best_metric": best_metric,
            "best_step": best_step,
        })
        # Persist after every eval — never lose progress on crash.
        with open(setup.output_dir / "history.json", "w") as f:
            json.dump({"history": history,
                       "converged": False,
                       "elapsed_seconds": elapsed}, f, indent=2)

        flag = " *BEST*" if is_best else ""
        print(f"  step={steps_done:>10,} | CR={cr:.3f} | "
              f"avg_ep={avg_ep:>5.0f} | elapsed={elapsed/60:>5.1f}m{flag}")

    setup.learner.save(setup.output_dir / "final_model")
    print(f"\nDONE. best_metric={best_metric:.3f} at step {best_step:,}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.3: Smoke-syntax-check the script (no env load yet)**

```bash
./venv/bin/python -c "import scripts.train_br_sac; print('import ok')"
```

Expected: `import ok` followed by no errors.

If `ImportError: cannot import name 'PartialObsWrapper' from 'envs.partial_obs_wrapper'` — verify the import path matches the codebase (it does per `envs/opponent_adapter.py:21`).

- [ ] **Step 3.4: Re-run the unit test, expect PASS**

```bash
./venv/bin/python -m pytest tests/test_br_sac_loading.py -v
```

Expected: 2 PASS (if snapshot present) or 2 SKIPPED.

- [ ] **Step 3.5: Commit**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    scripts/train_br_sac.py \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add scripts/train_br_sac.py && git commit -m "feat(S88): BR SAC training loop + history.json output"'
```

---

## Task 4: Write the exploitability analyzer — Phase 1 (classifier logic)

**Files:**
- Create: `scripts/analyze_exploitability.py` (lines 1–~110)
- Test: `tests/test_exploitability_analyzer.py`

The classifier is pure logic — no I/O. We TDD it with synthetic histories that cover each of the five hypotheses.

- [ ] **Step 4.1: Write the failing classifier test**

Create `tests/test_exploitability_analyzer.py`:

```python
"""Verify the L1-L4 per-seed and H1-H5 cohort classifier in
scripts/analyze_exploitability.py."""

import pytest
from scripts.analyze_exploitability import (
    EPSILON,
    classify_per_seed,
    classify_cohort,
    exploit_gap,
)


def test_exploit_gap_pursuer_uses_max():
    # pursuer wants HIGH CR; gap = max(curve) - baseline
    g = exploit_gap(curve=[0.30, 0.40, 0.50], baseline=0.31, br_role="pursuer")
    assert g == pytest.approx(0.19)


def test_exploit_gap_evader_uses_min():
    # evader wants LOW CR; gap = baseline - min(curve)
    g = exploit_gap(curve=[0.40, 0.20, 0.35], baseline=0.31, br_role="evader")
    assert g == pytest.approx(0.11)


@pytest.mark.parametrize(
    "p_gap,e_gap,expected",
    [
        (0.02, 0.03, "L1"),     # both within ε
        (0.08, 0.02, "L2"),     # pursuer-side exceeds
        (0.02, 0.08, "L3"),     # evader-side exceeds
        (0.08, 0.08, "L4"),     # both exceed
        (0.05, 0.05, "L1"),     # boundary: ≤ ε is L1
        (0.0501, 0.05, "L2"),   # just-above-ε pursuer
    ],
)
def test_classify_per_seed(p_gap, e_gap, expected):
    assert classify_per_seed(pursuer_gap=p_gap, evader_gap=e_gap) == expected


@pytest.mark.parametrize(
    "s48,s49,expected",
    [
        ("L1", "L1", "H1"),
        ("L2", "L2", "H2"),
        ("L3", "L3", "H3"),
        ("L4", "L4", "H4"),
        ("L1", "L2", "H5"),
        ("L2", "L3", "H5"),
        ("L4", "L1", "H5"),
    ],
)
def test_classify_cohort(s48, s49, expected):
    assert classify_cohort(s48_label=s48, s49_label=s49) == expected


def test_epsilon_value():
    assert EPSILON == 0.05  # spec §3 — do not change without a new spec
```

- [ ] **Step 4.2: Run, expect collection error**

```bash
./venv/bin/python -m pytest tests/test_exploitability_analyzer.py -v
```

Expected: `ImportError: cannot import name 'classify_per_seed' from 'scripts.analyze_exploitability'`.

- [ ] **Step 4.3: Implement the classifier**

Create `scripts/analyze_exploitability.py`:

```python
"""Analyze BR exploitability test outputs — emit cohort hypothesis verdict.

Reads `history.json` from the four BR run output dirs (BR_1..BR_4) and from
the cohort baseline seeds (SP17b_s48, SP17b_s49). Computes per-run
exploit_gap, per-seed L1-L4 label, and final cohort H1-H5 hypothesis.
Writes a one-page memo to docs/worklogs/2026-05-02_BR_verdict.md.

Spec: docs/superpowers/specs/2026-05-02-paper-strengthening-design.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

EPSILON = 0.05  # spec §3
BASELINE_TAIL_LEN = 8  # last 8 micro-phase CRs for baseline (spec §3)


def exploit_gap(curve: list[float], baseline: float, br_role: str) -> float:
    """Compute exploit_gap per spec §3.

    pursuer BR wants high CR ⇒ gap = max(curve) − baseline
    evader  BR wants low  CR ⇒ gap = baseline − min(curve)
    """
    if br_role == "pursuer":
        return max(curve) - baseline
    if br_role == "evader":
        return baseline - min(curve)
    raise ValueError(f"br_role must be 'pursuer' or 'evader', got {br_role!r}")


def classify_per_seed(pursuer_gap: float, evader_gap: float,
                      eps: float = EPSILON) -> str:
    """Per-seed L1-L4 label. Boundary `gap == eps` is treated as ≤ ε (i.e. L1)."""
    p_over = pursuer_gap > eps
    e_over = evader_gap > eps
    if not p_over and not e_over:
        return "L1"
    if p_over and not e_over:
        return "L2"
    if not p_over and e_over:
        return "L3"
    return "L4"


def classify_cohort(s48_label: str, s49_label: str) -> str:
    """Cohort H1-H5 from per-seed labels per spec §3."""
    pair = (s48_label, s49_label)
    same = {("L1", "L1"): "H1",
            ("L2", "L2"): "H2",
            ("L3", "L3"): "H3",
            ("L4", "L4"): "H4"}
    if pair in same:
        return same[pair]
    return "H5"  # any disagreement


@dataclass
class BRRun:
    name: str           # "BR_1" .. "BR_4"
    frozen_seed: int    # 48 or 49
    frozen_role: str    # "pursuer" or "evader"
    br_role: str        # "evader" if frozen_role=="pursuer" else "pursuer"
    cr_curve: list[float]
    baseline_cr: float
    exploit_gap: float


def load_baseline_cr(history_path: Path) -> float:
    """Mean of the last BASELINE_TAIL_LEN micro-phase capture rates."""
    data = json.loads(history_path.read_text())
    history = data["history"]
    # Filter to micro-phase entries (skip cold-start).
    m_entries = [e for e in history if e.get("phase", "").startswith("M")]
    tail = [e["capture_rate"] for e in m_entries[-BASELINE_TAIL_LEN:]]
    if not tail:
        raise ValueError(f"no micro-phase CRs found in {history_path}")
    return statistics.fmean(tail)


def load_br_cr_curve(history_path: Path) -> list[float]:
    data = json.loads(history_path.read_text())
    return [e["capture_rate"] for e in data["history"]]
```

- [ ] **Step 4.4: Run, expect 4 PASS**

```bash
./venv/bin/python -m pytest tests/test_exploitability_analyzer.py -v
```

Expected: 4 PASSED.

- [ ] **Step 4.5: Commit**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    scripts/analyze_exploitability.py tests/test_exploitability_analyzer.py \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add scripts/analyze_exploitability.py tests/test_exploitability_analyzer.py && git commit -m "feat(S88): BR exploitability classifier + tests"'
```

---

## Task 5: Add the analyzer's I/O + memo writer + CLI

**Files:**
- Modify: `scripts/analyze_exploitability.py` (append `analyze_all` + `main` + memo writer)

- [ ] **Step 5.1: Append `analyze_all` and the memo writer**

Append to `scripts/analyze_exploitability.py`:

```python
# Hard-coded BR run schema for this experiment (spec §4).
BR_RUNS = [
    {"name": "BR_1", "seed": 48, "frozen_role": "evader",  "br_role": "pursuer"},
    {"name": "BR_2", "seed": 48, "frozen_role": "pursuer", "br_role": "evader"},
    {"name": "BR_3", "seed": 49, "frozen_role": "evader",  "br_role": "pursuer"},
    {"name": "BR_4", "seed": 49, "frozen_role": "pursuer", "br_role": "evader"},
]

FRAMING = {
    "H1": ("A defended", "Replace 0.51±0.03 with corrected value; cite BR test."),
    "H2": ("C", "Curriculum-induced pursuer-side exploitability, replicated."),
    "H3": ("C", "Curriculum-induced evader-side exploitability, replicated."),
    "H4": ("B", "Self-play fixed point is not Nash — failure analysis."),
    "H5": ("C+seed-sensitivity", "NE quality is seed-dependent across (s48, s49)."),
}


def analyze_all(results_dir: Path, baseline_dir: Path) -> dict:
    """Load all 4 BR runs + 2 baselines; return verdict dict."""
    runs: list[BRRun] = []
    for spec in BR_RUNS:
        hist_path = results_dir / spec["name"] / "history.json"
        if not hist_path.exists():
            raise FileNotFoundError(f"missing {hist_path}")
        baseline_path = (baseline_dir / f"SP17b_s{spec['seed']}" / "history.json")
        baseline = load_baseline_cr(baseline_path)
        curve = load_br_cr_curve(hist_path)
        gap = exploit_gap(curve=curve, baseline=baseline, br_role=spec["br_role"])
        runs.append(BRRun(
            name=spec["name"], frozen_seed=spec["seed"],
            frozen_role=spec["frozen_role"], br_role=spec["br_role"],
            cr_curve=curve, baseline_cr=baseline, exploit_gap=gap,
        ))

    by_seed: dict[int, dict[str, float]] = {48: {}, 49: {}}
    for r in runs:
        by_seed[r.frozen_seed][r.br_role] = r.exploit_gap

    label_s48 = classify_per_seed(
        pursuer_gap=by_seed[48]["pursuer"],
        evader_gap=by_seed[48]["evader"],
    )
    label_s49 = classify_per_seed(
        pursuer_gap=by_seed[49]["pursuer"],
        evader_gap=by_seed[49]["evader"],
    )
    cohort = classify_cohort(label_s48, label_s49)
    framing, framing_note = FRAMING[cohort]

    return {
        "runs": runs,
        "label_s48": label_s48,
        "label_s49": label_s49,
        "cohort_hypothesis": cohort,
        "framing": framing,
        "framing_note": framing_note,
        "epsilon": EPSILON,
    }


def render_table(verdict: dict) -> str:
    rows = ["| Run | Frozen seed | Frozen role | BR role | Baseline | "
            "BR best | Gap | Per-seed |"]
    rows.append("|-----|-------------|-------------|---------|----------|---------|-----|----------|")
    for r in verdict["runs"]:
        if r.br_role == "pursuer":
            best = max(r.cr_curve)
        else:
            best = min(r.cr_curve)
        per_seed = (verdict["label_s48"] if r.frozen_seed == 48
                    else verdict["label_s49"])
        rows.append(
            f"| {r.name} | s{r.frozen_seed} | {r.frozen_role} | {r.br_role} | "
            f"{r.baseline_cr:.3f} | {best:.3f} | {r.exploit_gap:+.3f} | "
            f"{per_seed} |"
        )
    return "\n".join(rows)


def write_memo(verdict: dict, out_path: Path) -> None:
    table = render_table(verdict)
    body = f"""# BR Exploitability Test — Verdict

**Date:** 2026-05-02
**Spec:** [paper-strengthening-design]({Path('../../superpowers/specs/2026-05-02-paper-strengthening-design.md').as_posix()})

## Verdict

- Cohort hypothesis: **{verdict['cohort_hypothesis']}**
- Framing decision:  **{verdict['framing']}**
- Note: {verdict['framing_note']}
- ε = {verdict['epsilon']}

## Per-seed labels

- s48: **{verdict['label_s48']}**
- s49: **{verdict['label_s49']}**

## Per-run table

{table}

## Next step

Open `paper/main.tex` and apply the abstract phrasing change for framing
**{verdict['framing']}** per spec §8.
"""
    out_path.write_text(body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="results",
                   help="Dir containing BR_1..BR_4 subdirs.")
    p.add_argument("--baseline_dir", default="results",
                   help="Dir containing SP17b_s{48,49}/history.json.")
    p.add_argument("--memo_out",
                   default="docs/worklogs/2026-05-02_BR_verdict.md")
    args = p.parse_args()

    verdict = analyze_all(Path(args.results_dir), Path(args.baseline_dir))
    print(render_table(verdict))
    print(f"\nLabel s48: {verdict['label_s48']}")
    print(f"Label s49: {verdict['label_s49']}")
    print(f"Cohort:    {verdict['cohort_hypothesis']}  → framing {verdict['framing']}")
    print(f"           {verdict['framing_note']}")

    write_memo(verdict, Path(args.memo_out))
    print(f"\nMemo written to {args.memo_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Add an end-to-end test for the memo writer**

Append to `tests/test_exploitability_analyzer.py`:

```python
def test_write_memo_creates_readable_markdown(tmp_path):
    """Smoke-test the memo writer with a synthetic verdict."""
    from scripts.analyze_exploitability import write_memo, BRRun

    fake_verdict = {
        "runs": [
            BRRun(name="BR_1", frozen_seed=48, frozen_role="evader",
                  br_role="pursuer", cr_curve=[0.30, 0.32, 0.31],
                  baseline_cr=0.31, exploit_gap=0.01),
            BRRun(name="BR_2", frozen_seed=48, frozen_role="pursuer",
                  br_role="evader", cr_curve=[0.30, 0.29, 0.28],
                  baseline_cr=0.31, exploit_gap=0.03),
            BRRun(name="BR_3", frozen_seed=49, frozen_role="evader",
                  br_role="pursuer", cr_curve=[0.37, 0.38, 0.36],
                  baseline_cr=0.37, exploit_gap=0.01),
            BRRun(name="BR_4", frozen_seed=49, frozen_role="pursuer",
                  br_role="evader", cr_curve=[0.36, 0.37, 0.35],
                  baseline_cr=0.37, exploit_gap=0.02),
        ],
        "label_s48": "L1", "label_s49": "L1",
        "cohort_hypothesis": "H1",
        "framing": "A defended",
        "framing_note": "...",
        "epsilon": 0.05,
    }
    out = tmp_path / "verdict.md"
    write_memo(fake_verdict, out)
    text = out.read_text()
    assert "Cohort hypothesis: **H1**" in text
    assert "Framing decision:  **A defended**" in text
    assert "BR_1" in text and "BR_4" in text
```

- [ ] **Step 5.3: Run all tests, expect PASS**

```bash
./venv/bin/python -m pytest tests/test_exploitability_analyzer.py tests/test_br_sac_loading.py -v
```

Expected: all PASS (or `test_br_sac_loading.py` SKIPPED if no snapshot).

- [ ] **Step 5.4: Commit**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    scripts/analyze_exploitability.py tests/test_exploitability_analyzer.py \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add scripts/analyze_exploitability.py tests/test_exploitability_analyzer.py && git commit -m "feat(S88): exploitability analyzer I/O + memo writer + tests"'
```

---

## Task 6: Smoke test BR-1 (50K steps)

**Files:** none new

This task verifies end-to-end that the BR script runs against a real frozen opponent and reaches CR within 0.10 of the s48 baseline (0.31). Per spec §5.4: a wildly different number means the frozen opponent didn't load, `p_full` is wrong, or reward scaling is off — halt and debug.

- [ ] **Step 6.1: Sync the BR script to niro-2 if not already present**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    scripts/train_br_sac.py \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/
```

- [ ] **Step 6.2: Launch the smoke run on niro-2 (foreground; ~10 min)**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && .venv/bin/python -u scripts/train_br_sac.py \
        --frozen_opponent_path results/BR_frozen/s48/evader \
        --frozen_role evader \
        --total_steps 50000 \
        --eval_freq 25000 \
        --n_eval_episodes 50 \
        --seed 148 \
        --output_dir results/BR_smoke 2>&1 | tee results/BR_smoke.log'
```

Expected stdout includes two eval lines (at 25K and 50K) printing `CR=0.XX`.

Acceptance: both eval CRs within `[0.21, 0.41]` (baseline 0.31 ± 0.10).

If acceptance fails:
- CR ≈ 0.0: frozen evader probably escapes trivially because the BR pursuer is fresh — this is normal at 50K. CR should still be > 0.10 by 50K from incidental captures. If it is exactly 0.0, suspect that the env is broken (capture impossible). Investigate with a 10-episode rollout where you visualize trajectories.
- CR ≈ 0.9: frozen evader is not loading correctly (random opponent ⇒ trivial captures). Verify `frozen_model.policy` is a learned MultiInputPolicy (not random init), check the `ppo.zip` path resolves, check that `PartialObsOpponentAdapter.predict` actually invokes `self.model.predict` (it does per `envs/opponent_adapter.py:91`).
- Crash with `KeyError: 'obs_history'`: the obs space the SAC model expects does not match what `PartialObsOpponentAdapter` builds. Compare the model's `observation_space.spaces.keys()` with the adapter's `_build_obs_dict()` keys.

- [ ] **Step 6.3: Inspect smoke history.json**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && cat results/BR_smoke/history.json'
```

Expected: 2 entries with `"phase": "BR25000"` / `"BR50000"`, `"frozen": "evader"`, `"trained": "pursuer"`, `"p_full_obs": 0.0`.

- [ ] **Step 6.4: Verify BR side actually trained (spec §11 entropy mitigation)**

The smoke test must rule out the failure mode "fresh side never trains" (spec §11). CR alone is a weak signal — verify directly that the SAC learner accumulated training updates. Run on niro-2:

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && .venv/bin/python -c "
from stable_baselines3 import SAC
m = SAC.load(\"results/BR_smoke/final_model\", device=\"cpu\")
n = m.num_timesteps
buf = m.replay_buffer.size() if m.replay_buffer is not None else 0
print(f\"num_timesteps={n}, replay_buffer_size={buf}\")
assert n >= 50000, f\"learner did not reach 50K timesteps: {n}\"
assert buf >= 40000, f\"replay buffer below expectation: {buf} < 40000\"
print(\"OK — BR side actually trained.\")
"'
```

Note: `train_br_sac.py:main` writes `final_model.zip` only after the run completes. The smoke run completes at 50K so the file exists.

Expected output:
```
num_timesteps=50000, replay_buffer_size=40000
OK — BR side actually trained.
```

If the assertion fails (e.g., `num_timesteps=0`), the BR side never executed `learn()` — root-cause that before going further.

- [ ] **Step 6.5: Decide GO / NO-GO and update worklog**

If both smoke acceptances pass (CR ∈ [0.21, 0.41] AND BR side trained): append a "Smoke test passed at 50K — CR_25K=X.XX, CR_50K=X.XX, num_timesteps=50000, replay=40000" line to the S88 worklog's Work Done section. Sync to niro-2 and commit.

If smoke fails: do NOT proceed to Task 7. Append failure details to worklog, fix root cause, repeat smoke test.

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md && git commit -m "log(S88): smoke test result"'
```

---

## Task 7: Stop drifting cohort runs (only after smoke GO)

**Files:** none

Per spec §5.2 — only stop s48 and s51 AFTER snapshot complete (Task 1) AND smoke GO (Task 6). s49, s50, ABL-1, ABL-2 keep running.

- [ ] **Step 7.1: Identify PIDs**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'ps -ef | grep "train_amsdrl.*SP17b_s4[8]\|train_amsdrl.*SP17b_s51" | grep -v grep'
```

Note the PID of the `python ... train_amsdrl.py ... --seed 48` and `--seed 51` processes.

- [ ] **Step 7.2: Send SIGTERM (graceful)**

Replace `<PID_S48>` and `<PID_S51>` below with the PIDs from Step 7.1.

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'kill -SIGTERM <PID_S48> <PID_S51>'
```

- [ ] **Step 7.3: Wait + verify the processes exited**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'sleep 30 && ps -p <PID_S48> -p <PID_S51> 2>/dev/null; echo exit=$?'
```

Expected: `exit=1` (processes gone). If still running after 30 s, retry SIGTERM. If still running after 90 s, escalate to SIGKILL — but warn in the worklog that final history.json may be truncated.

- [ ] **Step 7.4: Verify history.json final entries**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && python3 -c "import json; d=json.load(open(\"results/SP17b_s48/history.json\")); print(\"s48 last:\", d[\"history\"][-1])"'
```

Expected: a complete final entry with `phase`, `capture_rate`, `total_steps`. If the JSON is truncated/invalid, fall back to the second-to-last entry when computing baselines later.

- [ ] **Step 7.5: Log the kill in the worklog and commit**

Append to S88 worklog Work Done section. Sync + commit:

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md && git commit -m "ops(S88): killed s48 + s51 after smoke GO"'
```

---

## Task 8: Launch BR-1, BR-2 on niro-2

**Files:** none new

- [ ] **Step 8.1: Confirm niro-2 has 2 free GPU slots**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv'
```

Expected: at least 2× ≥ 8 GB free across GPUs (s48 + s51 freed ~2 slots × ~6 GB).

- [ ] **Step 8.2: Launch BR-1 in nohup background**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'cd /home/niro-2/Codes/safe-rl-pe && \
    nohup .venv/bin/python -u scripts/train_br_sac.py \
        --frozen_opponent_path results/BR_frozen/s48/evader \
        --frozen_role evader \
        --total_steps 1500000 \
        --eval_freq 250000 \
        --n_eval_episodes 200 \
        --seed 148 \
        --output_dir results/BR_1 \
        > results/BR_1.log 2>&1 &
    echo "BR_1 PID=$!"'
```

Note the printed PID.

- [ ] **Step 8.3: Launch BR-2 in nohup background**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'cd /home/niro-2/Codes/safe-rl-pe && \
    nohup .venv/bin/python -u scripts/train_br_sac.py \
        --frozen_opponent_path results/BR_frozen/s48/pursuer \
        --frozen_role pursuer \
        --total_steps 1500000 \
        --eval_freq 250000 \
        --n_eval_episodes 200 \
        --seed 248 \
        --output_dir results/BR_2 \
        > results/BR_2.log 2>&1 &
    echo "BR_2 PID=$!"'
```

- [ ] **Step 8.4: Verify both running**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 'sleep 60 && tail -3 /home/niro-2/Codes/safe-rl-pe/results/BR_1.log /home/niro-2/Codes/safe-rl-pe/results/BR_2.log'
```

Expected: both logs show the startup banner ("BR Exploitability Test", config dump). No tracebacks. If either crashes, investigate before launching BR-3/BR-4.

- [ ] **Step 8.5: Log launch in worklog and commit**

Append a line to the S88 worklog Work Done section, e.g. "BR-1, BR-2 launched on niro-2 at 2026-05-02 HH:MM (PIDs: 12345, 12346). Total budget 1.5M each, expected wall ≈4.5h." Then:

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    docs/worklogs/2026-05-02_S88.md \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md && git commit -m "ops(S88): launched BR-1 + BR-2 on niro-2"'
```

---

## Task 9: Launch BR-3, BR-4 on niro-1

**Files:** none new

niro-1 is this session's CWD. All commands here are local.

- [ ] **Step 9.1: Confirm niro-1 has 2 free GPU slots**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

If only 1 slot is free, kill ABL-2 (`pgrep -f "train_amsdrl.*ABL_no_switch_s61"` → `kill -SIGTERM <PID>`), or fall back to running BR-3 then BR-4 sequentially (≈9 h instead of ~5 h).

- [ ] **Step 9.2: Launch BR-3 in nohup background**

```bash
cd /home/niro-1/Codes/safe-rl-pe && \
    nohup ./venv/bin/python -u scripts/train_br_sac.py \
        --frozen_opponent_path results/BR_frozen/s49/evader \
        --frozen_role evader \
        --total_steps 1500000 \
        --eval_freq 250000 \
        --n_eval_episodes 200 \
        --seed 149 \
        --output_dir results/BR_3 \
        > results/BR_3.log 2>&1 &
echo "BR_3 PID=$!"
```

- [ ] **Step 9.3: Launch BR-4 in nohup background**

```bash
cd /home/niro-1/Codes/safe-rl-pe && \
    nohup ./venv/bin/python -u scripts/train_br_sac.py \
        --frozen_opponent_path results/BR_frozen/s49/pursuer \
        --frozen_role pursuer \
        --total_steps 1500000 \
        --eval_freq 250000 \
        --n_eval_episodes 200 \
        --seed 249 \
        --output_dir results/BR_4 \
        > results/BR_4.log 2>&1 &
echo "BR_4 PID=$!"
```

- [ ] **Step 9.4: Verify both running**

```bash
sleep 60 && tail -3 /home/niro-1/Codes/safe-rl-pe/results/BR_3.log /home/niro-1/Codes/safe-rl-pe/results/BR_4.log
```

Expected: clean startup, no tracebacks.

- [ ] **Step 9.5: Log launch in worklog and commit**

Append to S88 worklog Work Done: "BR-3, BR-4 launched on niro-1 at 2026-05-02 HH:MM (PIDs: nnnnn, nnnnn)." Then commit (sync via niro-2 since niro-1 is not a git repo):

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    docs/worklogs/2026-05-02_S88.md \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md && git commit -m "ops(S88): launched BR-3 + BR-4 on niro-1"'
```

---

## Task 10: Monitor + wait for completion (~5 h)

**Files:** none

- [ ] **Step 10.1: Spot-check progress hourly**

Use the Monitor tool with this watcher (check at t=1h, 2h, 3h, 4h):

```bash
{
  echo "=== niro-2 BR_1 + BR_2 ==="
  sshpass -p '123456' ssh niro-2@100.71.2.97 'tail -2 /home/niro-2/Codes/safe-rl-pe/results/BR_1.log /home/niro-2/Codes/safe-rl-pe/results/BR_2.log'
  echo "=== niro-1 BR_3 + BR_4 ==="
  tail -2 /home/niro-1/Codes/safe-rl-pe/results/BR_3.log /home/niro-1/Codes/safe-rl-pe/results/BR_4.log
}
```

Expected: each run prints a `step=...| CR=X.XX` line every 250K-step eval (≈45 min apart at ~6 step/sec). Watch for: (a) CR creeping above baseline+0.05 (hint of exploit), (b) CR collapse to 0.0 (broken run), (c) frozen-side hash drift (no — that's tested in unit; should not happen).

- [ ] **Step 10.2: Wait for all 4 to complete**

A run is complete when its log shows the final `DONE. best_metric=...` line. Total wall time ≈4.5 h after all four reach 1.5 M steps. Use a poll loop:

```bash
until [ \
  -f /home/niro-1/Codes/safe-rl-pe/results/BR_3/history.json \
] && [ \
  -f /home/niro-1/Codes/safe-rl-pe/results/BR_4/history.json \
] && grep -q '"phase": "BR1500000"' /home/niro-1/Codes/safe-rl-pe/results/BR_3/history.json \
   && grep -q '"phase": "BR1500000"' /home/niro-1/Codes/safe-rl-pe/results/BR_4/history.json \
   && sshpass -p '123456' ssh niro-2@100.71.2.97 'grep -q "\"phase\": \"BR1500000\"" /home/niro-2/Codes/safe-rl-pe/results/BR_1/history.json && grep -q "\"phase\": \"BR1500000\"" /home/niro-2/Codes/safe-rl-pe/results/BR_2/history.json'
do sleep 60; done
echo "All 4 BR runs complete."
```

- [ ] **Step 10.3: Pull niro-2 BR results back to niro-1 for analysis**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/results/BR_1/ \
    /home/niro-1/Codes/safe-rl-pe/results/BR_1/
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/results/BR_2/ \
    /home/niro-1/Codes/safe-rl-pe/results/BR_2/
# Also pull baseline histories (in case s48/s49 have grown since the prior pull)
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/results/SP17b_s48/history.json \
    /home/niro-1/Codes/safe-rl-pe/results/SP17b_s48/
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/results/SP17b_s49/history.json \
    /home/niro-1/Codes/safe-rl-pe/results/SP17b_s49/
```

- [ ] **Step 10.4: Sanity-check: 6 history.json files exist locally**

```bash
ls -la /home/niro-1/Codes/safe-rl-pe/results/BR_{1,2,3,4}/history.json /home/niro-1/Codes/safe-rl-pe/results/SP17b_s{48,49}/history.json
```

Expected: 6 lines, all non-zero size.

---

## Task 11: Run analyzer + produce verdict memo

**Files:**
- Create: `docs/worklogs/2026-05-02_BR_verdict.md`

- [ ] **Step 11.1: Run the analyzer**

```bash
cd /home/niro-1/Codes/safe-rl-pe && \
    ./venv/bin/python scripts/analyze_exploitability.py \
        --results_dir results \
        --baseline_dir results \
        --memo_out docs/worklogs/2026-05-02_BR_verdict.md
```

Expected: prints the verdict table (4 rows), per-seed labels, cohort hypothesis, framing recommendation, and "Memo written to ...".

- [ ] **Step 11.2: Read the verdict memo**

```bash
cat docs/worklogs/2026-05-02_BR_verdict.md
```

Sanity-check: the cohort hypothesis label appears, all 4 BR rows render, framing matches the spec §8 mapping.

- [ ] **Step 11.3: Handle the per-spec-§11 contingencies**

If any BR run's `cr_curve` is still rising at t=1.5 M (last-vs-prev slope > 0.02 per 250 K eval), the run is undertrained — extend that run only to 2.5 M and re-run the analyzer afterward. Implement by re-launching that run with `--total_steps 2500000` (SAC saves checkpoints, so resume by warm-loading `BR_X/best_model.zip` is not implemented in this plan — re-run from scratch with seed+1000 and the longer budget; document the extension in the worklog).

If `s48` baseline std (across last 8 m-evals) > 0.05, recompute baseline using the median rather than the mean. Edit `scripts/analyze_exploitability.py:load_baseline_cr` to switch from `statistics.fmean` to `statistics.median` and re-run. Document the switch in the worklog.

- [ ] **Step 11.4: Update S88 worklog with the verdict**

Append the cohort hypothesis, per-seed labels, and framing to the S88 worklog's Work Done section. Reference the verdict memo. Sync + commit:

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    docs/worklogs/2026-05-02_S88.md docs/worklogs/2026-05-02_BR_verdict.md \
    results/BR_1/history.json results/BR_2/history.json \
    results/BR_3/history.json results/BR_4/history.json \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md docs/worklogs/2026-05-02_BR_verdict.md && git commit -m "result(S88): BR exploitability verdict — cohort=$(grep cohort_hypothesis docs/worklogs/2026-05-02_BR_verdict.md | head -1 || echo HXX)"'
```

(`results/BR_*/history.json` is gitignored per project convention — the rsync moves it for backup, but the commit only adds the docs.)

---

## Task 12: Update decisions.md, training-runs memory, MEMORY.md

**Files:**
- Modify: `docs/decisions.md`
- Modify: `~/.claude/projects/.../memory/training-runs.md`
- Modify: `~/.claude/projects/.../memory/sp17-experiment.md`

- [ ] **Step 12.1: Append D34, D35 to decisions.md**

Add two entries to the Research Direction table in `docs/decisions.md`:

```markdown
| D34 | 2026-05-02 | S88 | **BR exploitability test as ε-NE check** | Capture rate alone cannot distinguish NE from non-equilibrium fixed point. Fresh-init BR (1.5M steps, ε=0.05) is the canonical test. SP17b cohort verdict: H{X} (see `docs/worklogs/2026-05-02_BR_verdict.md`). |
| D35 | 2026-05-02 | S88 | **Frozen snapshot point: M2550 (last full-PO phase before drift)** | Curriculum-end drift after p_full=0 confounds endpoint policies with curriculum artifacts. Snapshotting at M2550 ≈ 5.22M steps tests the cohort's *claimed NE* state. |
```

(Replace `H{X}` with the actual hypothesis from Task 11.)

- [ ] **Step 12.2: Update training-runs.md memory**

Edit `/home/niro-1/.claude/projects/-home-niro-1-Codes-safe-rl-pe/memory/training-runs.md` — add a new section "Group D: BR Exploitability Test" with a 4-row table for BR-1..BR-4 (seed, frozen role, BR role, final exploit_gap, per-seed label).

- [ ] **Step 12.3: Update sp17-experiment.md memory**

Edit `/home/niro-1/.claude/projects/-home-niro-1-Codes-safe-rl-pe/memory/sp17-experiment.md` — replace the "Current Status" block with the post-verdict state, link to the spec and verdict memo, and update "What success looks like" to reflect the chosen framing.

- [ ] **Step 12.4: Commit all doc updates**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    docs/decisions.md \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/decisions.md && git commit -m "docs(S88): D34 BR test methodology + D35 snapshot rationale"'
```

(Memory files are stored under the user's home, not the project tree; they are managed by the auto-memory system, not git.)

- [ ] **Step 12.5: Push to GitHub from niro-2**

```bash
sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git push origin main'
```

Expected: standard git push output. If push is rejected, do NOT force-push (CLAUDE.md). Pull first, resolve, push.

---

## Task 13: Close S88

**Files:**
- Modify: `docs/worklogs/2026-05-02_S88.md` (final close)
- Modify: `docs/workflow_tracker.md` (Current Blockers section)

- [ ] **Step 13.1: Fill in remaining S88 worklog sections**

Complete the worklog: Decisions Made (D34, D35), Issues & Blockers (any), Next Steps. Specifically, Next Steps must reference the *next* spec to write — "rewrite paper abstract + sections 1, 4, 5 to match Framing {chosen}" — that becomes S89.

- [ ] **Step 13.2: Update Current Blockers in workflow_tracker.md**

Per CLAUDE.md: if a blocker resolved this session, remove it; if a new one was introduced, add it. The blocker introduced/resolved in S88: "the headline CR=0.51 claim is unsupported" — RESOLVED (verdict gives a defensible framing).

- [ ] **Step 13.3: Final commit**

```bash
sshpass -p '123456' rsync -av --rsh='ssh -o StrictHostKeyChecking=no' -R \
    docs/worklogs/2026-05-02_S88.md docs/workflow_tracker.md \
    niro-2@100.71.2.97:/home/niro-2/Codes/safe-rl-pe/

sshpass -p '123456' ssh niro-2@100.71.2.97 \
    'cd /home/niro-2/Codes/safe-rl-pe && git add docs/worklogs/2026-05-02_S88.md docs/workflow_tracker.md && git commit -m "close(S88): worklog + blockers updated, BR test complete" && git push origin main'
```

---

## Quick reference: full command sequence

```bash
# Authoring & tests (niro-1)
./venv/bin/python -m pytest tests/test_br_sac_loading.py tests/test_exploitability_analyzer.py -v

# Snapshot frozen opponents (niro-2)
bash scripts/snapshot_frozen_opponents.sh

# Smoke (niro-2)
.venv/bin/python -u scripts/train_br_sac.py --frozen_opponent_path results/BR_frozen/s48/evader --frozen_role evader --total_steps 50000 --eval_freq 25000 --n_eval_episodes 50 --seed 148 --output_dir results/BR_smoke

# Production (niro-2: BR-1, BR-2; niro-1: BR-3, BR-4) — see Tasks 8 + 9 for full nohup invocations.

# Analysis (niro-1)
./venv/bin/python scripts/analyze_exploitability.py --results_dir results --baseline_dir results --memo_out docs/worklogs/2026-05-02_BR_verdict.md
```
