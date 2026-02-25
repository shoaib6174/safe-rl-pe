# Research Report: Potential-Based Reward Shaping for Evader Obstacle-Seeking

**Date**: 2026-02-24
**Session**: S44
**Author**: Claude (research agent)
**Status**: Proposed fix for Level 3+ collapse (Runs H through O)

---

## 1. Problem Statement

Across 8 consecutive experimental runs (H through O), the evader agent collapses at obstacle-based curriculum levels (L3+), with SR_E ≤ 0.12. The evader cannot learn to use obstacles for cover.

### 1.1 Experimental Evidence

| Run | Key Config Delta | Best SR_E at L3+ | Outcome |
|-----|-----------------|-------------------|---------|
| H | Curriculum (4-level) | 0.14 (L3 S6) | L4 collapse → 100% pursuer |
| I | + Opponent pool | N/A (converged L2) | Never reached obstacles |
| J | + Curriculum gate | 0.03 (L4 S8) | L4 collapse → 100% pursuer |
| K | 6-level curriculum | 0.01 (L4 S5) | Collapse shifted to new L4 |
| L | + w_occlusion=0.05 | 0.01 (L4 S5) | Bonus too weak |
| M | + w_occlusion=0.2, v=1.1 | 0.04 (L3+ S5) | Bonus still drowned out |
| N | Visibility reward + survival | 0.12 (L3 S6) | Slight improvement, still fails |
| O | + Prep phase (100 steps) | 0.10 (L3 S6) | Evader wastes head start |

### 1.2 Root Cause Analysis

The evader requires three capabilities to use obstacles:

| Capability | Provider | Status |
|-----------|----------|--------|
| **Time** to reach obstacles | Prep phase (Fix 3) | Implemented, tested |
| **Incentive** to hide | Visibility reward (Fix 1) | Implemented, tested |
| **Direction** toward obstacles | ??? | **MISSING** |

The visibility reward landscape is pathological for learning:
- **Everywhere in open space**: r = -1 (visible) — a flat plateau, no gradient
- **Behind obstacles**: r = +1 (hidden) — unreachable spikes
- **No smooth transition**: the reward jumps discontinuously from -1 to +1 at the obstacle boundary

PPO follows per-step gradients. In open space, every direction yields r = -1. There is no signal pointing toward obstacles. The evader moves randomly during prep phase, gets caught after prep, and never discovers that obstacles exist as a defensive resource.

---

## 2. Literature Review

### 2.1 Potential-Based Reward Shaping (PBRS)

**Primary reference**: Ng, A.Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." *ICML 1999*.

**Core theorem**: For any MDP M = (S, A, T, γ, R), define a shaped MDP M' = (S, A, T, γ, R') where:

```
R'(s, a, s') = R(s, a, s') + F(s, a, s')
F(s, a, s') = γ · Φ(s') − Φ(s)
```

for some real-valued potential function Φ : S → ℝ. Then **M and M' have the same set of optimal policies**.

This is the ONLY form of additive reward shaping that is guaranteed to be policy-invariant. Any other form of reward shaping can change the optimal policy.

**Practical implication**: We can add a shaping reward that guides the agent toward obstacles without changing what strategy it eventually converges to. The shaping only accelerates learning — it does not introduce bias.

### 2.2 PBRS in Multi-Agent Settings

**Reference**: Devlin, S. & Kudenko, D. (2012). "Dynamic Potential-Based Reward Shaping." *AAMAS 2012*.

Extended PBRS to multi-agent settings, showing it preserves Nash equilibria under certain conditions. In our alternating self-play (AMS-DRL), each training phase freezes the opponent, making it a single-agent MDP. The standard Ng et al. 1999 guarantee therefore holds within each training phase.

### 2.3 PBRS in Our Paper Collection

**Paper N15** (RMARL-CBF-SAM, Liu et al., Information Sciences 2025) uses exactly the PBRS form for safety reward shaping:

```
r_{s,i} = γ · h_i(s') − h_i(s)
```

with a formal proof (Proposition 1) that this augmented reward does not change the optimal solution. This validates the approach in a multi-agent RL context with safety considerations — closely analogous to ours.

### 2.4 OpenAI Hide-and-Seek (Baker et al., ICLR 2020)

The most relevant prior work for obstacle-based evasion. Key design choice: **purely binary visibility-based reward** with no distance shaping. This was sufficient to produce emergent obstacle use — but required:
- 480 CPU workers + 8 V100 GPUs
- ~300 million episodes (billions of steps)
- 40% of episode length as preparation phase

We cannot afford this compute budget. PBRS provides the principled shortcut: it supplies the gradient that OpenAI's approach discovers through exhaustive exploration.

### 2.5 Why Additive Bonuses Failed (Runs L, M)

Runs L (w_occlusion=0.05) and M (w_occlusion=0.2) used additive occlusion bonuses:

```
r_evader = -r_pursuer + w_occlusion * I(hidden)
```

This is NOT potential-based reward shaping. It's a constant additive bonus when hidden. Problems:
1. **Drowned by distance shaping**: pursuer's r_dist = 10 × Δd / d_max ≈ 0.35/step dominates
2. **No gradient toward obstacles**: the bonus is 0 everywhere except behind obstacles
3. **Changes optimal policy**: unlike PBRS, this form can alter the optimal strategy

---

## 3. Proposed Solution: PBRS Obstacle-Seeking

### 3.1 Potential Function

```
Φ(s) = −d_nearest(s)
```

where:

```
d_nearest(s) = min_i (‖p_evader − p_obstacle_i‖ − r_obstacle_i)
```

clamped to ≥ 0. This is the distance from the evader to the **surface** of the nearest obstacle.

- Near obstacle surface: Φ ≈ 0 (highest potential)
- Far from all obstacles: Φ is large negative (lowest potential)

### 3.2 Shaping Reward

```
F(s, a, s') = γ · Φ(s') − Φ(s) ≈ d_prev_nearest − d_curr_nearest
```

(simplified for γ ≈ 1)

Scaled and normalized:

```
r_obs_approach = w_obs_approach × (d_prev_nearest − d_curr_nearest) / d_max
```

where d_max is the arena diagonal (normalization) and w_obs_approach is a tunable weight.

### 3.3 Complete Evader Reward at Obstacle Levels

```python
# When obstacles exist AND not a terminal step:
r_evader = (
    visibility_weight × (+1 if hidden else −1)        # strategic: hide behind obstacles
    + survival_bonus                                    # alive bonus (+1/step)
    + w_obs_approach × (d_prev_obs − d_curr_obs) / d_max   # PBRS: gradient toward obstacles
)
```

### 3.4 Properties

| Property | Value |
|----------|-------|
| **Policy-preserving** | Yes — standard PBRS guarantee (Ng et al. 1999) |
| **Naturally conditional** | Zero when no obstacles exist (L1/L2 curriculum) |
| **Smooth gradient** | Continuous, differentiable everywhere |
| **Symmetric design** | Pursuer → distance shaping toward evader; Evader → distance shaping toward obstacles |
| **Compatible with prep phase** | During prep, provides directional guidance; after prep, continues incentivizing proximity |

### 3.5 Weight Selection

Matching the pursuer's distance_scale (w=10.0) is a natural default:

```
Per step: r_obs ≈ 10 × v × dt / d_max = 10 × 1.0 × 0.05 / 14.14 ≈ 0.035
Over 100 steps of approach: ≈ 3.5 total
```

Compared to visibility reward (±1/step): the PBRS is ~3.5% of the visibility signal per step. This is intentionally small — its role is to **break the plateau** in open space, not to dominate the reward. Once the evader is near an obstacle, the visibility reward (+1 vs -1 = 2.0/step swing) takes over.

---

## 4. Interaction with Existing Fixes

### 4.1 How the three fixes work together

```
Episode timeline (600 steps, obstacle level):

Steps 0-99 (prep phase):
  - Pursuer: frozen (v=0)
  - Evader: moves toward nearest obstacle (guided by PBRS)
  - Reward: r = -1 (visibility, still visible) + 1 (survival) + 0.035 (PBRS approach)
  - Net: r ≈ +0.035/step — slight positive gradient toward obstacles

Steps 100+ (chase phase):
  - Evader behind obstacle: r = +1 (hidden) + 1 (survival) + ~0 (PBRS, already near)
  - Evader in open: r = -1 (visible) + 1 (survival) - 0.035 (PBRS, moving away)
  - Net: +2.0/step hidden vs -0.035/step exposed — strong incentive to stay hidden

Result: Evader learns approach-then-hide strategy
```

### 4.2 Why each component is necessary

| Without... | What happens |
|-----------|-------------|
| Without prep phase | Evader gets caught before reaching obstacle (pursuer 10% faster) |
| Without visibility reward | Evader has no reason to stay behind obstacles once there |
| Without PBRS | Evader has no gradient toward obstacles; wastes prep time fleeing randomly |
| Without survival bonus | Visibility -1 and +1 cancel out over episode; no net incentive to hide |

---

## 5. Sensory Feasibility Analysis: Can the Evader Sense Obstacles?

PBRS computes the shaping reward from **true state** (inside the reward function), which is standard practice. But the policy needs **sensory input** to know *which direction* to move. If the evader can't detect obstacles, the PBRS gradient is useless — the policy has no features to correlate with the reward signal.

### 5.1 What the Evader Observes About Obstacles

Under partial observability (`PartialObsWrapper`), the evader's observation is:

| Feature | Source | Details |
|---------|--------|---------|
| Lidar (36 rays, 360 deg) | `LidarSensor` | **max_range = 5.0m**, detects obstacles + walls |
| Own pose (x, y, theta) | Proprioception | Always known |
| Own velocity (v, omega) | Proprioception | Always known |
| FOV opponent (d, bearing) | `FOVSensor` | 120 deg cone, 10m range — for **opponent only**, not obstacles |

Critically, `n_obstacle_obs` is **not set** in `_make_partial_obs_env()` — it defaults to 0. The evader receives **no explicit obstacle distance/bearing features**. It only knows about obstacles through lidar returns (short echoes in 36 directions).

### 5.2 Arena Size vs. Lidar Range

| Arena | Diagonal | Avg d_nearest_obstacle | Lidar Range | Obstacle Visible? |
|-------|----------|----------------------|-------------|-------------------|
| **10×10** | 14.14m | ~2-3m (3 obstacles in 100m²) | 5.0m | **Usually yes** (~80%+ of positions) |
| **20×20** | 28.28m | ~5-8m (3 obstacles in 400m²) | 5.0m | **Often no** (~40% of positions) |

In the **10×10 arena**, the obstacle density is high enough that the 5m lidar almost always detects at least one obstacle. The PBRS gradient and the lidar readings are correlated — the policy CAN learn "move toward the short lidar return."

In the **20×20 arena**, obstacles are often beyond lidar range. The PBRS says "move toward obstacle" but the policy has no sensory input indicating which direction. This would require either:
- Increasing `lidar_max_range` to 10-15m (realistic for real lidars)
- Adding explicit `n_obstacle_obs > 0` features (obstacle dist/bearing from true state)

### 5.3 Impact on Current 10×10 Experiments

All Runs H-O used the 10×10 arena. The collapse is **not caused by sensory limitations** — the evader CAN detect obstacles via lidar in 10×10. The collapse is caused by the **reward landscape** (no gradient toward obstacles), which PBRS fixes.

### 5.4 Scaling to 20×20 (Future)

When we scale to 20×20 for final validation, we will need to address lidar range. Options:
1. Increase `lidar_max_range` from 5.0 to 10.0m (simple, realistic)
2. Increase number of obstacles from 3 to 6-8 (higher density)
3. Both

This is a **separate concern** from the PBRS fix and should be addressed when scaling.

### 5.5 Should We Run O on 20×20?

**No.** Running Run O's config on 20×20 would make things strictly worse:
- Same collapse at L3+ (reward problem persists)
- Worse sensory coverage (lidar can't reach obstacles)
- Longer episodes, more compute, no new information

The diagnosis doc already mandates 10×10 for development (line 109-113). Fix the reward problem first, then scale.

---

## 6. Alternatives Considered and Rejected

### 6.1 Spawn Evader Near Obstacles (Fix 11)

- **Approach**: Place evader within 1-2m of an obstacle at L3+ levels
- **Rejected because**: Changes the MDP (different initial state distribution). The evader wouldn't learn to REACH obstacles from arbitrary positions. Poor generalization if the evader starts far from obstacles at deployment.

### 6.2 Stronger Additive Occlusion Bonus

- **Approach**: Increase w_occlusion to 0.5 or 1.0
- **Rejected because**: Not policy-preserving. Still no gradient in open space (bonus is 0 everywhere except behind obstacles). Runs L and M already demonstrated this approach fails fundamentally.

### 6.3 Scripted Evader Pre-training (Fix 4)

- **Approach**: Pre-train evader to seek obstacles via scripted pursuer
- **Rejected because**: Introduces bias from scripted behavior. The self-play framework should discover strategies without hand-coded heuristics. PBRS achieves the same acceleration without scripting.

### 6.4 Intrinsic Motivation (RND, Fix 7)

- **Approach**: Random Network Distillation to encourage exploration of obstacle regions
- **Rejected because**: Overly complex for this specific problem. RND explores ALL novel states, not specifically obstacle regions. PBRS is more targeted and theoretically grounded.

### 6.5 Run 20×20 Arena (Run O Config at Scale)

- **Approach**: Test Run O config (visibility + prep phase) on 20×20 arena
- **Rejected because**: Makes things strictly worse. Obstacles are beyond 5m lidar range from many spawn positions. Same reward plateau problem persists. More compute for no new information. Fix the reward problem at 10×10 first, then scale.

---

## 7. Implementation Plan

### 7.1 Files to Modify

| File | Changes |
|------|---------|
| `envs/rewards.py` | Add `w_obs_approach` param to `RewardComputer.__init__()`. Add obstacle distance tracking and PBRS term in `compute()`. |
| `training/amsdrl.py` | Thread `w_obs_approach` through `AMSDRLSelfPlay.__init__()` → `env_kwargs` → `RewardComputer` |
| `scripts/train_amsdrl.py` | Add `--w_obs_approach` CLI argument (default 0.0 = disabled) |
| `tests/test_rewards.py` | Test PBRS computation, sign, magnitude, zero when no obstacles |

### 7.2 Key Design Decisions

1. **Compute d_nearest in RewardComputer**: The reward computer already receives `evader_pos` and `obstacles`. Add a helper to compute distance to nearest obstacle surface.

2. **Track d_prev_nearest**: Store previous step's nearest-obstacle distance in the env, similar to how `prev_distance` tracks agent-to-agent distance.

3. **Active only with obstacles**: When `obstacles` list is empty, PBRS term is 0. No special-casing needed.

4. **Terminal steps**: On capture/timeout, use the zero-sum fallback (no PBRS). This ensures terminal rewards are clean.

### 7.3 Recommended Run P Configuration

```bash
./venv/bin/python -u scripts/train_amsdrl.py \
    --max_phases 18 --timesteps_per_phase 300000 \
    --no_dcbf \
    --distance_scale 10.0 --pursuer_v_max 1.1 --fixed_speed \
    --arena_width 10.0 --arena_height 10.0 --max_steps 600 \
    --n_envs 16 --n_steps 2048 --batch_size 512 \
    --curriculum --opponent_pool_size 5 \
    --use_visibility_reward --survival_bonus 1.0 \
    --prep_steps 100 \
    --w_obs_approach 10.0 \
    --seed 42 --output results/stage3/run_p
```

---

## 8. Expected Outcome

### 8.1 Success Criteria

- **L3**: SR_E ≥ 0.30 (evader wins at least 30% at close-range obstacles)
- **L4+**: SR_E ≥ 0.20 (evader competitive at medium/far obstacle levels)
- **No collapse**: Health monitor does not trigger repeated rollbacks at L3+

### 8.2 Failure Modes to Watch

1. **PBRS too weak (w=10)**: If evader still doesn't approach obstacles, increase to w=50 or w=100.
2. **PBRS too strong**: If evader "orbits" obstacles instead of hiding behind them, reduce weight.
3. **Prep phase too long**: 100 steps might be excessive with PBRS providing direction. Could try 50.

---

## 9. References

1. Ng, A.Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML 1999*.
2. Devlin, S. & Kudenko, D. (2012). Dynamic Potential-Based Reward Shaping. *AAMAS 2012*.
3. Baker, B., et al. (2020). Emergent Tool Use From Multi-Agent Autocurricula. *ICLR 2020*. arXiv:1909.07528.
4. Liu, S., Liu, L., & Yu, Z. (2025). RMARL-CBF-SAM: Safe Robust Multi-Agent RL with Neural CBFs and Safety Attention Mechanism. *Information Sciences 690*, 121567. [Paper N15 in our collection]
5. Wiewiora, E. (2003). Potential-Based Shaping and Q-Value Initialization are Equivalent. *JAIR 19*.
