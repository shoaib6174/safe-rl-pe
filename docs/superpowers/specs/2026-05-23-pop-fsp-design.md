# Pop-FSP Design Specification

**Date:** 2026-05-23  
**Scope:** Population-Based Fictitious Self-Play for Gonultas & Isler 2024 reproduction  
**Paper framing:** EMA as partial/seed-dependent baseline; Pop-FSP as robust solution  
**Approach:** Actor-only population, standalone (not combined with EMA), uniform sampling

---

## 1. Problem Statement

EMA opponent stabilization (tau=0.001) prevents high-frequency oscillatory collapse but does not guarantee balanced convergence. Across 4 seeds:
- Seed 0: robustness=0.698 (balanced play, outlier)
- Seed 1: robustness=0.000 (evader-dominant collapse)
- Seed 2: robustness=0.148 (pursuer overfit to weak EMA opponent)
- Seed 3: robustness=0.292 (oscillated, then fell into evader-dominant trap)

**Root cause:** Training against a single opponent (even EMA-smoothed) allows overfitting. The system has multiple basins of attraction; initialization determines which basin is entered.

**Pop-FSP hypothesis:** Training against a diverse population of past opponents forces each agent to learn robust strategies that generalize across multiple opponent types, preventing overfitting and reducing seed-dependency.

---

## 2. Architecture

### 2.1 High-Level Data Flow

```
Training Loop
├── Rollout: current pi_p vs current pi_e -> replay buffer
│   (belief augmentation happens at env level; buffer stores augmented obs)
├── Every M episodes: snapshot actor state_dict -> population
└── Each update step:
    ├── Sample pi_e' from evader_population
    ├── Sample pi_p' from pursuer_population
    ├── pursuer.update(batch, opponent_actor=pi_e')
    └── evader.update(batch, opponent_actor=pi_p')
```

### 2.2 Why Actor-Only is Sufficient

The MADDPG `update()` method receives a `batch` containing pre-computed observations, actions, rewards, and next observations. The `opponent_actor` is only used to compute:
1. Target Q-value: `Q'(s', pi_p'(s'), pi_e'(s'))`
2. Actor gradient: `grad_pi_p Q(s, pi_p(s), pi_e(s))` where pi_e is frozen

Since the replay buffer observations are already belief-augmented (from rollout), the opponent actor only needs to map obs -> action. No BiMDN weights are needed during the update.

### 2.3 Three Methods for Paper Comparison

| Method | opponent_actor passed to update() | Purpose in paper |
|--------|-----------------------------------|------------------|
| Vanilla | `evader.actor` (current) | Baseline: immediate collapse |
| EMA | `ema_evader` (exponentially smoothed) | Baseline: reduces high-frequency oscillation but seed-dependent |
| Pop-FSP | `pop_e.sample()` (random past opponent) | Proposed: robust convergence via population diversity |

All three use the **same** `MADDPGAgent.update()` signature. Only the source of `opponent_actor` changes.

---

## 3. Components

### 3.1 PopulationManager

**File:** `repro_gonultas_isler_2024/training/population_manager.py`

**Responsibility:** Store actor state dict snapshots; sample random opponents on demand.

```python
class PopulationManager:
    def __init__(self, max_size: int, obs_dim: int, action_dim: int,
                 hidden_dim: int, device: str):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.actor_states: list[dict] = []  # CPU storage

    def add(self, actor_state: dict) -> None:
        """Add snapshot. Deep copy to CPU. FIFO eviction at capacity."""

    def sample(self) -> nn.Module:
        """Uniform random sample. Return fresh nn.Module on device, no-grad, eval mode."""

    def __len__(self) -> int:
        """Current population size."""
```

**Design decisions:**
- **CPU storage, GPU inference:** State dicts stay on CPU (~1MB each x K=8 = 8MB). Fresh module created and moved to GPU only when sampled.
- **Fresh module per sample:** Prevents state leakage between update steps. Each sampled opponent is independent.
- **No grad, eval mode:** Prevents accidental gradient flow through the opponent network.
- **FIFO eviction:** Simplest strategy. Can be replaced with diversity-based or strength-based eviction later.
- **Uniform sampling:** Baseline strategy. PFSP (prioritized fictitious self-play) will be added as a secondary ablation.

### 3.2 Training Loop Modifications

**File:** `repro_gonultas_isler_2024/training/train.py`

**Changes:**
1. Add `use_pop_fsp` flag from config
2. Initialize `PopulationManager` instances for pursuer and evader
3. Populate initial population from saved checkpoints (if resuming) or start empty
4. Every `pop_add_freq` episodes: add current actor snapshot to population
5. Each update step: sample opponent from population instead of using current/EMA opponent
6. Population sampling only begins after `pop_warmup` episodes (don't sample from tiny population)

**Config parameters:**
```yaml
maddpg:
  use_pop_fsp: true       # Enable Pop-FSP (mutually exclusive with EMA)
  pop_size: 8             # Max population size K
  pop_add_freq: 250       # Add checkpoint every N episodes
  pop_warmup: 500         # Min episodes before sampling starts
```

**Mutual exclusivity:** `use_pop_fsp` and `ema_tau > 0` are mutually exclusive. Assert this at startup.

**Warmup period:** During `pop_warmup` episodes, training uses vanilla self-play (current opponent) while the population accumulates. After warmup, switch to population sampling. This prevents sampling from a population of size 1-2, which would be essentially vanilla SP.

### 3.3 Checkpoint Integration

Populations are persisted across runs:
- Save `pursuer_pop.pt` and `evader_pop.pt` alongside regular checkpoints
- Each file contains a list of state dicts
- On resume, load populations and continue

**File format:**
```python
pop_checkpoint = {
    "pursuer_pop": [state_dict_0, state_dict_1, ...],  # list of dicts
    "evader_pop": [state_dict_0, state_dict_1, ...],
}
```

### 3.4 Evaluation

**File:** `repro_gonultas_isler_2024/evaluation/evaluate.py`

**Changes:**
- Add population evaluation mode: evaluate current policy against all population members
- Compute "population robustness score": min(CR vs each population member)
- This is distinct from the existing tournament evaluation

**New evaluation modes:**
| Mode | Description |
|------|-------------|
| `current_vs_pop` | Current pursuer vs each evader in population |
| `pop_vs_current` | Each pursuer in population vs current evader |
| `pop_robustness` | min(CR across all population matchups) |

---

## 4. Configuration

**New config file:** `config_pop_fsp.yaml`

Derived from `config.yaml` with these additions:
```yaml
maddpg:
  use_pop_fsp: true
  pop_size: 8
  pop_add_freq: 250
  pop_warmup: 500
  ema_tau: 0.0    # Explicitly disabled; mutually exclusive with Pop-FSP
```

**Baseline configs for paper:**
| Config file | Method | Purpose |
|-------------|--------|---------|
| `config_vanilla.yaml` | Vanilla SP | Baseline |
| `config.yaml` | EMA (tau=0.001) | Comparison baseline |
| `config_pop_fsp.yaml` | Pop-FSP | Proposed method |

---

## 5. Testing Strategy

### 5.1 Unit Tests

**File:** `repro_gonultas_isler_2024/tests/test_population_manager.py`

1. **test_add_and_sample:** Add 3 state dicts, sample 100 times, verify uniform distribution
2. **test_fifo_eviction:** Add K+1 state dicts, verify oldest is evicted
3. **test_sample_returns_fresh_module:** Sample twice, verify different module objects
4. **test_sample_no_grad:** Verify all parameters have requires_grad=False
5. **test_sample_eval_mode:** Verify module is in eval mode
6. **test_warmup_behavior:** Verify sampling returns None or falls back to current opponent when population < min_size

### 5.2 Integration Tests

1. **test_training_loop_with_pop_fsp:** Run 10 episodes with Pop-FSP enabled, verify no errors
2. **test_checkpoint_save_load:** Save population, reload, verify identical samples
3. **test_mutual_exclusion:** Verify ema_tau > 0 and use_pop_fsp=True raises AssertionError

### 5.3 Smoke Test

Run 500-episode training with Pop-FSP:
- Verify population grows to expected size
- Verify CR doesn't immediately collapse to 0 or 1
- Verify checkpoints include population state

---

## 6. Theoretical Justification

### 6.1 Why Pop-FSP Should Work

**Vanilla self-play:** Agent A trains against Agent B(t), where B is simultaneously learning. At time t+1, B has changed, so A's policy is already obsolete. This causes perpetual cycling.

**EMA self-play:** Agent A trains against a smoothed average of B's recent policies: B_EMA(t) = EMA[B(t)]. This reduces the rate of opponent change but still presents a single opponent. Agent A can still overfit to B_EMA.

**Pop-FSP:** Agent A trains against a random sample from {B_0, B_1, ..., B_K}. The expected opponent is the empirical average of the population. More importantly, the variance in opponents forces A to learn a strategy that works against multiple opponent types simultaneously. This is a form of implicit regularization.

**Connection to fictitious play:** In classical fictitious play, each agent plays a best response to the empirical distribution of the opponent's past actions. Pop-FSP approximates this by training against samples from the empirical distribution of past opponent policies.

### 6.2 Why Seed-Dependency Should Reduce

In vanilla SP, the system is a high-dimensional dynamical system with sensitive dependence on initial conditions (the initialization seed). Small differences in initial policies lead to divergent trajectories.

In Pop-FSP, the population acts as a "reservoir" of diverse strategies. Even if the current trajectory starts in an unfavorable basin, the population diversity provides alternative opponent types that can pull the agent out of local optima. This reduces the system's sensitivity to initial conditions.

### 6.3 Expected Behavior

| Metric | Vanilla | EMA | Pop-FSP (predicted) |
|--------|---------|-----|---------------------|
| Oscillation amplitude | High (immediate collapse) | Medium (damped, but seed-dependent) | Low (converges to steady-state) |
| Seed-dependency | Extreme | High | Reduced |
| Best robustness | ~0.00-0.01 | 0.00-0.70 (seed-dependent) | >0.40 consistently |
| Single-checkpoint score | ~0.25 | ~0.30-0.70 | >0.50 |

---

## 7. Experiment Plan

### 7.1 Phase 1: Implementation + Smoke Test (This session)

1. Implement PopulationManager
2. Modify training loop
3. Unit tests (6 tests)
4. Smoke test: 500 episodes, 1 seed
5. Verify population grows, CR stays non-degenerate

### 7.2 Phase 2: 5K Comparison (Next session)

Run all three methods for 5000 episodes, 3 seeds each:
- Vanilla
- EMA (tau=0.001)
- Pop-FSP (K=8)

Compare: oscillation amplitude, best robustness, seed variance.

### 7.3 Phase 3: Full 20K Runs (If Phase 2 GO)

5 seeds, 20K episodes each for Pop-FSP.
Ablations: K in {4, 8, 16}, PFSP sampling.

---

## 8. Risk Analysis

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Pop-FSP still seed-dependent | Medium | Try PFSP sampling; increase K; combined EMA+Pop |
| Population too large (GPU memory) | Low | K=8 is small; state dicts on CPU |
| Population diversity too low | Medium | Add diversity-based eviction; vary add frequency |
| Training slower (more overhead) | Low | Fresh module creation is fast; can optimize with module pool |
| Paper deadline pressure | High | Front-load implementation; have EMA-only paper as fallback |

---

## 9. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `training/population_manager.py` | Create | PopulationManager class |
| `training/train.py` | Modify | Add Pop-FSP path in training loop |
| `config_pop_fsp.yaml` | Create | Pop-FSP hyperparameters |
| `evaluation/evaluate.py` | Modify | Population evaluation modes |
| `tests/test_population_manager.py` | Create | Unit tests |
| `launch_pop_fsp.sh` | Create | niro-2 launch script |

---

*Spec status: Awaiting user review before implementation.*
