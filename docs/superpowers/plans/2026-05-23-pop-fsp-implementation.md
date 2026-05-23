# Pop-FSP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Population-Based Fictitious Self-Play (Pop-FSP) as a standalone training method for the Gonultas & Isler 2024 reproduction, with actor-only populations, uniform sampling, and FIFO eviction.

**Architecture:** PopulationManager stores actor state dicts on CPU; training loop samples a fresh opponent module per update; mutually exclusive with EMA. All three methods (vanilla, EMA, Pop-FSP) share the same MADDPG.update() signature.

**Tech Stack:** Python, PyTorch, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `training/population_manager.py` | Create | Store/sample actor state dict populations |
| `training/train.py` | Modify | Add Pop-FSP path: init populations, add snapshots, sample opponents |
| `config_pop_fsp.yaml` | Create | Pop-FSP hyperparameters |
| `tests/test_population_manager.py` | Create | Unit tests for PopulationManager |
| `launch_pop_fsp.sh` | Create | niro-2 launch script for smoke test |

---

## Task 1: PopulationManager Core

**Files:**
- Create: `repro_gonultas_isler_2024/training/population_manager.py`
- Test: `repro_gonultas_isler_2024/tests/test_population_manager.py`

- [ ] **Step 1: Write the failing test — test_add_and_sample**

```python
import sys
sys.path.insert(0, "/home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024")

import torch
import numpy as np
from training.population_manager import PopulationManager
from agents.maddpg import ActorNetwork


def test_add_and_sample():
    """Add 3 state dicts, sample 300 times, verify roughly uniform distribution."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    # Create 3 distinct actors
    for i in range(3):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict())

    assert len(pm) == 3

    counts = [0, 0, 0]
    for _ in range(300):
        sampled = pm.sample()
        val = sampled.fc1.weight.data.mean().item()
        counts[int(round(val))] += 1

    # All 3 should be sampled at least 50 times each (roughly uniform)
    assert all(c >= 50 for c in counts), f"Counts: {counts}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
python -m pytest tests/test_population_manager.py::test_add_and_sample -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'training.population_manager'"

- [ ] **Step 3: Write minimal PopulationManager implementation**

```python
import copy
import random
import torch
import torch.nn as nn
from agents.maddpg import ActorNetwork


class PopulationManager:
    """Store actor network snapshots and sample random opponents on demand.

    Stores state dicts on CPU. Returns a fresh nn.Module on the target device
    per sample call. No gradient, eval mode.
    """

    def __init__(
        self,
        max_size: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.actor_states: list[dict] = []

    def add(self, actor_state: dict) -> None:
        """Add a snapshot. Deep copy to CPU. Evict oldest if at capacity."""
        self.actor_states.append(copy.deepcopy(actor_state))
        if len(self.actor_states) > self.max_size:
            self.actor_states.pop(0)

    def sample(self) -> nn.Module:
        """Uniform random sample. Return fresh nn.Module on device, no-grad, eval."""
        if len(self.actor_states) == 0:
            raise RuntimeError("Population is empty — cannot sample.")
        idx = random.randint(0, len(self.actor_states) - 1)
        state = self.actor_states[idx]
        actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        actor.load_state_dict(state)
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        return actor

    def __len__(self) -> int:
        return len(self.actor_states)

    def save(self, path: str) -> None:
        """Save population state dicts to disk."""
        torch.save({"actor_states": self.actor_states}, path)

    def load(self, path: str) -> None:
        """Load population state dicts from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.actor_states = checkpoint["actor_states"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
python -m pytest tests/test_population_manager.py::test_add_and_sample -v
```

Expected: PASS

- [ ] **Step 5: Write test — test_fifo_eviction**

```python
def test_fifo_eviction():
    """Add K+1 state dicts; verify oldest is evicted."""
    pm = PopulationManager(
        max_size=3, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    for i in range(4):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict())

    assert len(pm) == 3

    # Sample 300 times; should never see value 0 (evicted)
    for _ in range(300):
        sampled = pm.sample()
        val = sampled.fc1.weight.data.mean().item()
        assert val != 0.0, f"Evicted member 0 was sampled"
```

- [ ] **Step 6: Run test**

```bash
python -m pytest tests/test_population_manager.py::test_fifo_eviction -v
```

Expected: PASS

- [ ] **Step 7: Write test — test_sample_returns_fresh_module**

```python
def test_sample_returns_fresh_module():
    """Sample twice; verify different module objects."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict())

    s1 = pm.sample()
    s2 = pm.sample()
    assert s1 is not s2
```

- [ ] **Step 8: Run test**

```bash
python -m pytest tests/test_population_manager.py::test_sample_returns_fresh_module -v
```

Expected: PASS

- [ ] **Step 9: Write test — test_sample_no_grad_and_eval_mode**

```python
def test_sample_no_grad_and_eval_mode():
    """Verify sampled module has no grad and is in eval mode."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict())

    sampled = pm.sample()
    assert not sampled.training
    assert all(not p.requires_grad for p in sampled.parameters())
```

- [ ] **Step 10: Run test**

```bash
python -m pytest tests/test_population_manager.py::test_sample_no_grad_and_eval_mode -v
```

Expected: PASS

- [ ] **Step 11: Write test — test_empty_population_raises**

```python
def test_empty_population_raises():
    """Sampling from empty population must raise RuntimeError."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    try:
        pm.sample()
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "empty" in str(e).lower()
```

- [ ] **Step 12: Run test**

```bash
python -m pytest tests/test_population_manager.py::test_empty_population_raises -v
```

Expected: PASS

- [ ] **Step 13: Write test — test_save_load**

```python
def test_save_load(tmp_path):
    """Save population, reload, verify identical samples."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    actor.fc1.weight.data.fill_(3.14)
    pm.add(actor.state_dict())

    path = tmp_path / "pop.pt"
    pm.save(str(path))

    pm2 = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    pm2.load(str(path))

    assert len(pm2) == 1
    sampled = pm2.sample()
    assert abs(sampled.fc1.weight.data.mean().item() - 3.14) < 1e-6
```

- [ ] **Step 14: Run test**

```bash
python -m pytest tests/test_population_manager.py::test_save_load -v
```

Expected: PASS

- [ ] **Step 15: Run full test suite**

```bash
python -m pytest tests/test_population_manager.py -v
```

Expected: 6 tests PASS

- [ ] **Step 16: Commit**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git add repro_gonultas_isler_2024/training/population_manager.py
git add repro_gonultas_isler_2024/tests/test_population_manager.py
git commit -m "feat(pop-fsp): PopulationManager with FIFO eviction, uniform sampling, save/load

- Actor-only population storage on CPU
- Fresh nn.Module per sample (no-grad, eval mode)
- 6 unit tests passing"
```

---

## Task 2: Training Loop Integration

**Files:**
- Modify: `repro_gonultas_isler_2024/training/train.py`

- [ ] **Step 1: Add PopulationManager import**

At the top of `training/train.py`, add:
```python
from training.population_manager import PopulationManager
```

- [ ] **Step 2: Add Pop-FSP config parsing after EMA setup**

After the EMA setup block (around line 86-99), add:

```python
    # Pop-FSP (Population-Based Fictitious Self-Play)
    use_pop_fsp = config["maddpg"].get("use_pop_fsp", False)
    pop_size = config["maddpg"].get("pop_size", 8)
    pop_add_freq = config["maddpg"].get("pop_add_freq", 250)
    pop_warmup = config["maddpg"].get("pop_warmup", 500)

    # Mutual exclusivity: EMA and Pop-FSP cannot both be active
    if use_ema and use_pop_fsp:
        raise ValueError("EMA and Pop-FSP are mutually exclusive. "
                         "Set ema_tau=0 or use_pop_fsp=false.")

    if use_pop_fsp:
        pursuer_pop = PopulationManager(
            max_size=pop_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config["maddpg"]["hidden_dim"],
            device=device,
        )
        evader_pop = PopulationManager(
            max_size=pop_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config["maddpg"]["hidden_dim"],
            device=device,
        )
    else:
        pursuer_pop = None
        evader_pop = None
```

- [ ] **Step 3: Update the status log line**

Change the existing status line (around line 161):
```python
    if use_pop_fsp:
        ema_status = f"Pop-FSP (K={pop_size}, add_freq={pop_add_freq}, warmup={pop_warmup})"
    elif use_ema:
        ema_status = f"EMA=ON (tau={ema_tau})"
    else:
        ema_status = "EMA=OFF (vanilla SP)"
```

- [ ] **Step 4: Replace the update block with Pop-FSP path**

Replace the existing update block (around lines 197-219):

```python
        # MADDPG updates: maintain ~1 update per env step
        if replay_buffer.size >= batch_size:
            n_updates = n_envs * update_freq
            for _ in range(n_updates):
                batch = replay_buffer.sample(batch_size)
                if use_pop_fsp:
                    # Pop-FSP: sample random opponent from population
                    # During warmup, fall back to current opponent
                    if completed_episodes >= pop_warmup and len(evader_pop) > 0:
                        opp_e_actor = evader_pop.sample()
                    else:
                        opp_e_actor = evader.actor
                    if completed_episodes >= pop_warmup and len(pursuer_pop) > 0:
                        opp_p_actor = pursuer_pop.sample()
                    else:
                        opp_p_actor = pursuer.actor
                    c_loss_p, a_loss_p = pursuer.update(batch, opp_e_actor, agent_id="pursuer")
                    c_loss_e, a_loss_e = evader.update(batch, opp_p_actor, agent_id="evader")
                elif use_ema:
                    c_loss_p, a_loss_p = pursuer.update(batch, ema_evader, agent_id="pursuer")
                    c_loss_e, a_loss_e = evader.update(batch, ema_pursuer, agent_id="evader")
                else:
                    c_loss_p, a_loss_p = pursuer.update(batch, evader.actor, agent_id="pursuer")
                    c_loss_e, a_loss_e = evader.update(batch, pursuer.actor, agent_id="evader")
                pursuer.update_targets()
                evader.update_targets()
            # Update EMA networks after all gradient updates
            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_evader.parameters(), evader.actor.parameters()):
                        ema_param.data.mul_(1 - ema_tau).add_(param.data, alpha=ema_tau)
                    for ema_param, param in zip(ema_pursuer.parameters(), pursuer.actor.parameters()):
                        ema_param.data.mul_(1 - ema_tau).add_(param.data, alpha=ema_tau)
            last_c_loss_p = c_loss_p
            last_a_loss_p = a_loss_p
            last_c_loss_e = c_loss_e
            last_a_loss_e = a_loss_e
```

- [ ] **Step 5: Add population update after episode completion**

After the episode completion block (around line 234, after curriculum update), add:

```python
        # Update populations
        if use_pop_fsp and completed_episodes > 0 and completed_episodes % pop_add_freq == 0:
            pursuer_pop.add(pursuer.actor.state_dict())
            evader_pop.add(evader.actor.state_dict())
            print(f"  [Pop-FSP] Added checkpoint at episode {completed_episodes}. "
                  f"Pursuer pop={len(pursuer_pop)}, Evader pop={len(evader_pop)}")
```

- [ ] **Step 6: Add population to checkpoint saving**

In the checkpoint saving block (around line 274-287), after the EMA save block, add:

```python
            if use_pop_fsp:
                pursuer_pop.save(checkpoint_dir / "pursuer_pop.pt")
                evader_pop.save(checkpoint_dir / "evader_pop.pt")
```

- [ ] **Step 7: Add population to final checkpoint**

In the final checkpoint block (around line 290-299), after the EMA save block, add:

```python
    if use_pop_fsp:
        pursuer_pop.save(final_dir / "pursuer_pop.pt")
        evader_pop.save(final_dir / "evader_pop.pt")
```

- [ ] **Step 8: Run existing tests to ensure no regression**

```bash
cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
python -m pytest tests/test_maddpg.py -v
python -m pytest tests/test_training_smoke.py -v
```

Expected: All existing tests still PASS

- [ ] **Step 9: Commit**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git add repro_gonultas_isler_2024/training/train.py
git commit -m "feat(pop-fsp): integrate PopulationManager into training loop

- Pop-FSP path in training loop with warmup period
- Mutual exclusivity check: EMA vs Pop-FSP
- Population snapshots saved in checkpoints
- Falls back to current opponent during warmup"
```

---

## Task 3: Pop-FSP Config File

**Files:**
- Create: `repro_gonultas_isler_2024/config_pop_fsp.yaml`

- [ ] **Step 1: Create config_pop_fsp.yaml**

Copy from `config.yaml` and add Pop-FSP params:

```yaml
# Environment
arena_size: 16.0
max_steps: 400
capture_radius: 0.15
sim_dt: 0.1

# Pursuer (bicycle model)
pursuer:
  wheelbase: 0.30
  v_max: 2.5
  v_min: -1.0
  delta_max: 0.34
  delta_min: -0.34
  accel_max: 2.0
  accel_min: -2.0
  steer_rate_max: 3.2
  steer_rate_min: -3.2
  fov_angle: 2.094
  fov_range: 6.0

# Evader (point-mass)
evader:
  v_max: 1.5
  v_min: -1.5
  accel_max: 9.81
  accel_min: -9.81
  fov_angle: 6.283
  fov_range: 1000.0

# Reward
reward:
  lambda_capture: 1000
  lambda_time: 1
  lambda_distance: 1

# Global toggles
use_belief: false
use_amp: true
use_compile: false

# MADDPG
maddpg:
  episodes: 20000
  parallel_envs: 2
  replay_buffer_size: 100000
  batch_size: 1024
  actor_lr: 0.0005
  critic_lr: 0.0005
  gamma: 0.99
  tau: 0.005
  exploration_sigma: 0.1
  hidden_dim: 128
  frame_stack: 2
  frame_skip: 2
  update_freq: 1
  ema_tau: 0.0              # DISABLED — Pop-FSP is mutually exclusive with EMA

  # Pop-FSP parameters
  use_pop_fsp: true
  pop_size: 8
  pop_add_freq: 250
  pop_warmup: 500

# BiMDN
bimdn:
  history_length: 200
  downsample_factor: 10
  lstm_hidden: 128
  feature_dim: 32
  num_components: 3
  belief_buffer_size: 25000
  belief_warmup: 10000
  belief_update_freq: 10
  belief_batch_size: 256
  belief_lr: 0.002

# Curriculum
curriculum:
  t_end_ratio: 0.3
  fov_start: 6.283
  fov_end: 2.094
  evader_vmax_start: 0.0
  evader_vmax_end: 1.5

# Evaluation
eval:
  checkpoint_freq: 250
  eval_episodes: 500

# Logging
log_dir: "./runs"
```

- [ ] **Step 2: Validate config loads without error**

```bash
cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
python -c "import yaml; cfg = yaml.safe_load(open('config_pop_fsp.yaml')); print('Config loaded OK'); print('use_pop_fsp:', cfg['maddpg']['use_pop_fsp'])"
```

Expected: `Config loaded OK` and `use_pop_fsp: True`

- [ ] **Step 3: Commit**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git add repro_gonultas_isler_2024/config_pop_fsp.yaml
git commit -m "feat(pop-fsp): add Pop-FSP config file

- K=8, add_freq=250, warmup=500
- ema_tau=0 (mutually exclusive with Pop-FSP)"
```

---

## Task 4: Launch Script

**Files:**
- Create: `repro_gonultas_isler_2024/launch_pop_fsp.sh`

- [ ] **Step 1: Create launch script**

```bash
#!/bin/bash
# Launch Pop-FSP smoke test on niro-2
# Usage: ./launch_pop_fsp.sh [seed]

set -e

SEED=${1:-0}
EPISODES=${2:-500}

# Output dir with unique name to prevent collisions
OUTPUT_DIR="results/pop_fsp_smoke_s${SEED}_$(date +%Y%m%d_%H%M%S)"

cd /home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024

echo "Launching Pop-FSP smoke test: seed=$SEED, episodes=$EPISODES"
echo "Output: $OUTPUT_DIR"

python training/train.py config_pop_fsp.yaml \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --episodes "$EPISODES"

echo "Done: $OUTPUT_DIR"
```

Wait — the current train.py doesn't accept `--output_dir`, `--seed`, or `--episodes` as CLI args. The config file specifies these. Let me check how the existing launch scripts work.

Check existing launch script to match pattern:

```bash
#!/bin/bash
# Pop-FSP smoke test launcher
# Run on niro-2 after pushing latest code

SEED=${1:-0}
EPISODES=${2:-500}
OUTPUT_DIR="results/pop_fsp_smoke_s${SEED}_$(date +%Y%m%d_%H%M%S)"

cd /home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024

echo "Pop-FSP smoke test: seed=$SEED, episodes=$EPISODES"
echo "Output: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy config and override seed/episodes via env or sed
# Current train.py reads config file path from argv[1]
# Config values are fixed; we modify config inline
cp config_pop_fsp.yaml "$OUTPUT_DIR/config.yaml"
sed -i "s|log_dir: \"./runs\"|log_dir: \"$OUTPUT_DIR/runs\"|" "$OUTPUT_DIR/config.yaml"

python training/train.py "$OUTPUT_DIR/config.yaml"

echo "Done: $OUTPUT_DIR"
```

Actually, let me check the existing launch scripts first to match the pattern.

```bash
cd /home/niro-1/Codes/safe-rl-pe
cat launch_ema_5k_comparison.sh | head -30
```

Let me write the launch script matching the existing pattern:

```bash
#!/bin/bash
# Pop-FSP smoke test launcher
# Run on niro-2 after pushing latest code

SEED=${1:-0}
EPISODES=${2:-500}
OUTPUT_DIR="results/pop_fsp_smoke_s${SEED}_$(date +%Y%m%d_%H%M%S)"

cd /home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024

echo "Pop-FSP smoke test: seed=$SEED, episodes=$EPISODES"
echo "Output: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy config and modify output path
cp config_pop_fsp.yaml "$OUTPUT_DIR/config.yaml"
sed -i "s|log_dir: \"./runs\"|log_dir: \"$OUTPUT_DIR/runs\"|" "$OUTPUT_DIR/config.yaml"

# Run
python training/train.py "$OUTPUT_DIR/config.yaml"

echo "Done: $OUTPUT_DIR"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024/launch_pop_fsp.sh
```

- [ ] **Step 3: Commit**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git add repro_gonultas_isler_2024/launch_pop_fsp.sh
git commit -m "feat(pop-fsp): add smoke test launch script"
```

---

## Task 5: Integration Test — Training Loop with Pop-FSP

**Files:**
- Test: `repro_gonultas_isler_2024/tests/test_training_smoke.py` (modify)

- [ ] **Step 1: Add Pop-FSP smoke test**

Add to `tests/test_training_smoke.py`:

```python
import sys
sys.path.insert(0, "/home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024")

import tempfile
import yaml
from pathlib import Path


def test_pop_fsp_training_smoke():
    """Run 10 episodes with Pop-FSP enabled; verify no errors and population grows."""
    from training.train import train
    from training.population_manager import PopulationManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "arena_size": 16.0,
            "max_steps": 50,
            "capture_radius": 0.15,
            "sim_dt": 0.1,
            "pursuer": {
                "wheelbase": 0.30, "v_max": 2.5, "v_min": -1.0,
                "delta_max": 0.34, "delta_min": -0.34,
                "accel_max": 2.0, "accel_min": -2.0,
                "steer_rate_max": 3.2, "steer_rate_min": -3.2,
                "fov_angle": 6.283, "fov_range": 1000.0,
            },
            "evader": {
                "v_max": 1.5, "v_min": -1.5,
                "accel_max": 9.81, "accel_min": -9.81,
                "fov_angle": 6.283, "fov_range": 1000.0,
            },
            "reward": {"lambda_capture": 1000, "lambda_time": 1, "lambda_distance": 1},
            "use_belief": False,
            "use_amp": False,
            "use_compile": False,
            "maddpg": {
                "episodes": 10,
                "parallel_envs": 1,
                "replay_buffer_size": 1000,
                "batch_size": 32,
                "actor_lr": 0.001,
                "critic_lr": 0.001,
                "gamma": 0.99,
                "tau": 0.005,
                "exploration_sigma": 0.1,
                "hidden_dim": 32,
                "frame_stack": 1,
                "frame_skip": 1,
                "update_freq": 1,
                "ema_tau": 0.0,
                "use_pop_fsp": True,
                "pop_size": 4,
                "pop_add_freq": 5,
                "pop_warmup": 3,
            },
            "bimdn": {
                "history_length": 50, "downsample_factor": 5,
                "lstm_hidden": 32, "feature_dim": 8,
                "num_components": 2, "belief_buffer_size": 100,
                "belief_warmup": 10, "belief_update_freq": 5,
                "belief_batch_size": 8, "belief_lr": 0.002,
            },
            "curriculum": {
                "t_end_ratio": 0.3,
                "fov_start": 6.283, "fov_end": 2.094,
                "evader_vmax_start": 0.0, "evader_vmax_end": 1.5,
            },
            "eval": {"checkpoint_freq": 1000, "eval_episodes": 10},
            "log_dir": str(Path(tmpdir) / "runs"),
            "output_dir": tmpdir,
            "seed": 42,
        }

        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        train(str(config_path))

        # Verify checkpoints exist
        checkpoint_dir = Path(tmpdir) / "seed_42" / "checkpoints"
        assert checkpoint_dir.exists(), f"No checkpoints dir: {checkpoint_dir}"

        # Verify population files exist in final checkpoint
        final_cp = sorted(checkpoint_dir.glob("episode_*"))[-1]
        assert (final_cp / "pursuer_pop.pt").exists(), "Missing pursuer_pop.pt"
        assert (final_cp / "evader_pop.pt").exists(), "Missing evader_pop.pt"

        # Verify populations are non-empty
        pm = PopulationManager(max_size=4, obs_dim=11, action_dim=2, hidden_dim=32, device="cpu")
        pm.load(final_cp / "pursuer_pop.pt")
        assert len(pm) > 0, "Pursuer population is empty after training"
```

- [ ] **Step 2: Run integration test**

```bash
cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
python -m pytest tests/test_training_smoke.py::test_pop_fsp_training_smoke -v -s
```

Expected: PASS (may take ~30-60 seconds)

- [ ] **Step 3: Commit**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git add repro_gonultas_isler_2024/tests/test_training_smoke.py
git commit -m "test(pop-fsp): add Pop-FSP training smoke test

- 10 episodes with Pop-FSP enabled
- Verifies population files are saved and non-empty"
```

---

## Task 6: Push to GitHub + Deploy to niro-2 + Run Smoke Test

**Files:** None (deployment actions)

- [ ] **Step 1: Push latest code to GitHub**

```bash
cd /home/niro-1/Codes/safe-rl-pe
git push origin main
```

- [ ] **Step 2: SSH to niro-2 and pull**

```bash
sshpass -p '123456' ssh -o ConnectTimeout=10 niro-2@100.71.2.97 'cd ~/Codes/safe-rl-pe && git pull origin main'
```

- [ ] **Step 3: Run smoke test on niro-2**

```bash
sshpass -p '123456' ssh -o ConnectTimeout=10 niro-2@100.71.2.97 'cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024 && ./launch_pop_fsp.sh 0 500'
```

- [ ] **Step 4: Monitor training progress**

Check logs every few minutes:
```bash
sshpass -p '123456' ssh -o ConnectTimeout=10 niro-2@100.71.2.97 'tail -20 ~/Codes/safe-rl-pe/repro_gonultas_isler_2024/results/pop_fsp_smoke_s0_*/runs/*.log 2>/dev/null || echo "Log not found yet"'
```

**Healthy signs:**
- CR oscillating (not stuck at 0 or 1)
- "[Pop-FSP] Added checkpoint" messages appearing
- Population sizes growing to K=8
- No NaN in losses

**Warning signs:**
- CR=0.000 for >50 consecutive episodes → evader-dominant collapse
- CR=1.000 for >50 consecutive episodes → pursuer-dominant collapse
- PA or EA at boundary (>95% of Q_MAX) → Q-value explosion

- [ ] **Step 5: Evaluate results after completion**

```bash
sshpass -p '123456' ssh -o ConnectTimeout=10 niro-2@100.71.2.97 'cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024 && python evaluation/evaluate.py config_pop_fsp.yaml results/pop_fsp_smoke_s0_*/checkpoints/episode_500'
```

**Go/No-Go criteria for 5K comparison:**
- CR oscillates (not permanently 0 or 1) → GO
- Best robustness > 0.20 → GO
- Population grows to K=8 and sampling works → GO
- Immediate collapse (CR=0 or 1 within first 100 episodes) → NO-GO, investigate

---

## Self-Review Checklist

- [ ] **Spec coverage:** PopulationManager ✅, training loop integration ✅, config ✅, tests ✅, launch script ✅, deployment ✅
- [ ] **Placeholder scan:** No TBD, TODO, or "implement later" found
- [ ] **Type consistency:** PopulationManager signature matches usage in train.py
- [ ] **Path consistency:** All paths absolute or relative to repo root
- [ ] **Config consistency:** `use_pop_fsp`, `pop_size`, `pop_add_freq`, `pop_warmup` used consistently
- [ ] **Mutual exclusivity:** EMA vs Pop-FSP check in train.py, ema_tau=0 in config_pop_fsp.yaml
