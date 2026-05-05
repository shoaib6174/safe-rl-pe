# Design: Explicit Belief-State Self-Play for Pursuit-Evasion

**Date:** 2026-05-05
**Status:** Draft (pending review)
**Target:** CoRL 2026 paper — fix curriculum-induced exploitability gap
**Approach:** Replace masking curriculum with explicit belief tracking; train under pure partial observability from start

---

## 1. Problem Statement

The masking curriculum (annealing `p_full_obs` from 1.0 → 0.0) creates a **distributional shift** that makes the learned policy pair exploitable. During training, the policy network processes a mixture of:

- **Full-state observations** (opponent position directly encoded)
- **Partial-state observations** (opponent visible only if in FOV and not occluded)

The policy learns to rely on the full-state shortcut. When `p_full` reaches 0, the shortcut vanishes and the under-trained partial-state representation collapses. This manifests as:

- SP17b s50: CR=0.46 at M2550 → drifts to 0.13 post-curriculum
- Warm-start BR: CR=0.46 → 0.115 (fine-tuning cannot rebuild representation)
- RecurrentPPO: CR=0.06 at M2550 → 0.00 (LSTM cannot compensate)

## 2. Goal

Replace the masking curriculum with an **explicit belief tracker** that provides a consistent observation representation regardless of visibility. Train under pure partial observability from the start. Achieve stable co-evolution with exploitability gap ≤ 0.05 (ε-NE).

## 3. Architecture

```
PursuitEvasionEnv → PartialObsWrapper → BeliefTracker → Dict obs → SAC (MultiInputPolicy)
                         ↑                  ↑
                    FOV/Lidar sensors   Particle filter
                                      (mean, cov, entropy, t_seen)
```

### 3.1 New Component: BeliefTracker

**File:** `envs/belief_tracker.py`

A particle filter that maintains a probability distribution over the **opponent's** position. Instantiations are role-aware: a pursuer's tracker tracks the evader; an evader's tracker tracks the pursuer.

**Interface:**
```python
class BeliefTracker:
    def __init__(self, role: str, arena_width: float, arena_height: float,
                 opponent_v_max: float, n_particles: int = 100,
                 seed: int = 0)
    def update(self, own_state: np.ndarray, opponent_state: np.ndarray,
               is_visible: bool) -> None
    def get_statistics(self) -> np.ndarray  # shape (6,)
    def reset(self) -> None
```

**Parameters:**
- `n_particles`: 100
- `arena_width`: 20.0
- `arena_height`: 20.0
- `opponent_v_max`: max speed of the opponent being tracked (pursuer tracks evader → evader_v_max; evader tracks pursuer → pursuer_v_max)

**State:**
- `particles`: [n_particles, 2] array of (x, y) positions
- `weights`: [n_particles] uniform weights (simplified — no reweighting needed)
- `last_visible_pos`: last known (x, y) when opponent was visible
- `steps_since_seen`: integer counter
- `rng`: np.random.Generator for reproducibility

**Update Rules:**

| Condition | Action |
|-----------|--------|
| Opponent **visible** | Resample all particles around observed position with Gaussian noise (σ = 0.2m). Reset `steps_since_seen = 0`. |
| Opponent **not visible** | Propagate each particle: `particle += N(0, σ_process²)` where `σ_process = opponent_v_max * dt * steps_since_seen`. `dt = 0.1s` (simulation timestep). Clamp to arena bounds. Increment `steps_since_seen`. |

**Output Statistics (belief vector, 6-D):**
```
belief = [mean_x, mean_y, std_x, std_y, entropy, time_since_seen]
```

- `mean_x, mean_y`: particle mean
- `std_x, std_y`: particle standard deviation (independent axes)
- `entropy`: log(det(covariance matrix) + ε) with ε=1e-6 — measures uncertainty. Added epsilon prevents -inf when particles collapse.
- `time_since_seen`: `steps_since_seen * dt` in seconds

### 3.2 Modified Component: PartialObsWrapper

**File:** `envs/partial_obs_wrapper.py`

Add a `BeliefTracker` instance inside `PartialObsWrapper`. Update it at each `step()` call using the true state from the base env.

**Observation Space Change:**

Current Dict obs:
```python
{
    "obs_history": Box(shape=(K, 43)),
    "lidar": Box(shape=(1, 36)),
    "state": Box(shape=(7,)),
}
```

New Dict obs:
```python
{
    "obs_history": Box(shape=(K, 43)),
    "lidar": Box(shape=(1, 36)),
    "state": Box(shape=(7,)),
    "belief": Box(shape=(6,), low=-inf, high=inf),
}
```

**Note:** The `obs_history` still contains the raw FOV detection (`d_to_opp`, `bearing`), which is redundant with the belief but kept to let the policy learn its own implicit features if needed. The belief provides explicit geometric information; the obs_history provides temporal context.

### 3.3 Modified Component: PartialObsFeaturesExtractor

**Files:** `agents/partial_obs_policy.py` (both `PartialObsPolicyNetwork` and `PartialObsFeaturesExtractor`)

**Changes to `PartialObsPolicyNetwork`:**
- Add `belief_dim: int = 6` parameter to `__init__`
- Add `belief_mlp` branch: `nn.Sequential(nn.Linear(belief_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())`
- Update `combined_dim = 256 (lidar) + 64 (state) + encoder_latent_dim + 64 (belief)`
- Update `forward()` to accept `"belief"` key from obs_dict

**Updated Architecture:**
```
obs_history → encoder (bimdn/lstm/mlp) → latent (varies by type)
lidar       → Conv1D                  → 256
state       → MLP                     → 64
belief      → MLP (6→32→64)           → 64
                                    ↓
                              concat → combined_dim → MLP → 256-D output
```

**Note:** The `encoder_type` parameter controls the `obs_history` encoder. SP17b used whatever encoder type was configured (check `configs/` or `train_amsdrl.py` defaults). The belief branch is independent and always active.

**Changes to `PartialObsFeaturesExtractor`:**
- Pass `belief_dim=6` to `PartialObsPolicyNetwork`
- The `observation_space` dict now includes `"belief"` key; `PartialObsPolicyNetwork` infers `belief_dim` from `observation_space["belief"].shape[0]`

### 3.4 Training Protocol

**Config changes from SP17b:**
- Remove `--masking_curriculum` (p_full_obs = 0.0 from start)
- Remove `--p_full_anneal_steps` (irrelevant without curriculum)
- Keep everything else identical: SAC, rewards, forced switch, vis reward, PFSP

**Launch command:**
```bash
nohup .venv/bin/python -u scripts/train_amsdrl.py \
    --algorithm sac \
    --fov_angle 90 --sensing_radius 8.0 --combined_masking \
    --arena_width 20 --arena_height 20 --max_steps 1200 \
    --n_obstacles_min 0 --n_obstacles_max 3 \
    --pursuer_v_max 1.0 --evader_v_max 1.0 \
    --lr 3e-4 --ent_coef 0.03 \
    --micro_phase_steps 2048 --eval_interval 150 \
    --alternate_freeze --freeze_switch_threshold 0.65 \
    --force_switch_steps 2000000 \
    --w_vis_pursuer 0.2 --w_search 0.0001 --t_stale 100 \
    --collapse_threshold 0.1 --collapse_streak_limit 5 \
    --pfsp \
    --max_total_steps 30000000 \
    --seed 200 \
    --output results/BELIEF_s200 \
    > results/BELIEF_s200.log 2>&1 &
```

Note: `--masking_curriculum` and `--p_full_anneal_steps` are intentionally omitted.

## 4. Data Flow

1. **Env step:** `PursuitEvasionEnv.step()` returns full state
2. **PartialObsWrapper.step()**:
   - Builds raw partial obs via FOV sensor + lidar + history buffer
   - Calls `belief_tracker.update(own_state, target_state, is_visible)`
   - Appends `belief_tracker.get_statistics()` to obs dict
3. **Episode reset:** `PartialObsWrapper.reset()` calls `belief_tracker.reset()`
4. **Policy forward:** `PartialObsFeaturesExtractor` processes all 4 keys
5. **SAC update:** Standard SAC training with Dict observations

## 5. Error Handling

| Scenario | Handling |
|----------|----------|
| Belief particles drift outside arena | Clamp to `[0, arena_width] × [0, arena_height]` |
| `steps_since_seen` exceeds 1200 (max episode length) | Cap at 1200 — entropy already captures uncertainty |
| Particle collapse (all particles identical) | Add small noise (σ=0.01) on resample to maintain diversity |
| Belief tracker called before first update | Return `mean=arena_center, std=arena_diag, entropy=max, time_since_seen=0` (safe default) |

## 6. Testing

### 6.1 Unit Test: BeliefTracker
- **Visible target:** Particles should cluster around true position (mean error < 0.1m)
- **Hidden target (10s):** Particles should spread to std ≈ v_max * 10 = 10m (clamped to arena)
- **Reappearance:** After 10s hidden, target reappears — belief should reconverge within 1 step

### 6.2 Smoke Test: Training Launch
- Launch with `--total_steps 10000` (short run)
- Verify no tracebacks, obs dict contains `"belief"` key
- Verify tensorboard logs show belief statistics

### 6.3 Validation Test: Convergence
- Train to M2550 (~5M steps, ~24h on niro-2)
- Evaluate CR under pure PO
- Run BR exploitability test if CR ≥ 0.40

## 7. Validation Criteria

| Checkpoint | Metric | Threshold | Interpretation |
|-----------|--------|-----------|----------------|
| M2550 | CR under pure PO | ≥ 0.40 | Belief enables learning without curriculum |
| M2550–M5000 | CR drift | ≤ 0.15 drop | No curriculum-end collapse |
| M5000 | BR evader gap | ≤ 0.05 | Policy is ε-NE (not exploitable) |

**Note on backward compatibility:** Old SP17b checkpoints do NOT have the `"belief"` key. BELIEF_s200 trains from scratch. BR testing of BELIEF_s200 uses belief-aware BR training (same code, frozen opponent loaded from BELIEF checkpoint).

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Belief tracker is too simple (diffusion model ignores walls/obstacles) | Medium | v1 uses simple diffusion. If needed, v2 adds obstacle-aware propagation (reject particles inside walls). |
| SAC cannot learn from belief vector (representation too abstract) | Low-Medium | Belief is only 6 numbers + raw obs still available. Policy can ignore belief if useless. |
| Pure PO from start is too hard (ABL-4 CR=0.19 suggests this) | Medium | ABL-4 had no belief. Belief should bridge the gap. If still too hard, reduce arena size (15×15) or increase sensing radius (10m) to make the game more tractable while keeping PO. **Do NOT reintroduce curriculum** — that reintroduces the distributional shift. |
| Training diverges (same freeze-sticking as before) | Low | Forced switch + vis reward + search reward are proven mechanisms. Kept unchanged. |

## 9. Definition of Done

- [ ] `envs/belief_tracker.py` implemented and unit-tested
- [ ] `PartialObsWrapper` extended with belief key and `reset()` integration
- [ ] `PartialObsPolicyNetwork` extended with belief branch
- [ ] `PartialObsFeaturesExtractor` updated to pass belief_dim
- [ ] Smoke test passes (10K steps, no errors)
- [ ] Launch BELIEF_s200 on niro-2
- [ ] CR ≥ 0.40 at M2550
- [ ] BR exploitability test shows gap ≤ 0.05
