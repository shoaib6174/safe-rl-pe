# Spec: Asymmetric EMA (A-EMA) Probe Experiment

**Date:** 2026-06-02  
**Status:** Draft — pending review  
**Purpose:** Validate whether learning-level asymmetry (compensating the evader) stabilizes self-play in PO pursuit-evasion. If the probe succeeds, expand to full A-EMA. If not, pivot to Approach 3 (hierarchical policy).

---

## 1. Background

104 sessions of experimentation identified that self-play in symmetric competitive games with partial observability is structurally unstable. Key findings:

- **EMA opponent smoothing** (tau=0.001) produced one strong seed (robustness=0.698) but only 1/7 reproducibility.
- **Pop-FSP population training** failed in all 5 configurations, leading to the discovery of a **∀/∃ structural asymmetry**: the evader must escape ALL pursuers (∀), while the pursuer only needs to capture ONE evader (∃).
- The game environment (G&I 2024) already has speed/sensing/dynamics asymmetry, but the **learning architecture is symmetric** (same network size, same LR, same training schedule).

**Hypothesis:** Explicitly compensating the evader at the learning level (more capacity, higher LR, more training steps) will restore balance and enable reliable convergence.

---

## 2. Method: A-EMA

### 2.1 Core Idea

Combine three mechanisms:
1. **EMA smoothing** (tau=0.001) — reduces oscillation
2. **vis_reward** (0.1) — gives the pursuer gradient signal under FOV partial observability
3. **Learning-level asymmetry** — compensates the evader for its harder ∀ objective

### 2.2 Probe Configuration (Phase 1)

Only **one asymmetry** is active in the probe: asymmetric learning rate.

| Parameter | Pursuer | Evader | Notes |
|-----------|---------|--------|-------|
| `actor_lr` | 0.0005 | **0.001** | 2× for evader |
| `critic_lr` | 0.0005 | **0.001** | Match actor LR |
| `hidden_dim` | 128 | 128 | **Symmetric** (probe isolates LR) |
| `train_steps_per_phase` | 1 | 1 | **Symmetric** (probe isolates LR) |
| `ema_tau` | 0.001 | 0.001 | Symmetric |
| `vis_reward` | **0.1** | 0 | Only pursuer needs LOS reward |
| `episodes` | 5000 | 5000 | Match G&I evaluation |
| `parallel_envs` | 2 | 2 | Match G&I config |
| `batch_size` | 1024 | 1024 | Match G&I config |

### 2.3 Full A-EMA Configuration (Phase 2 — conditional)

If Phase 1 succeeds (≥2/3 seeds with robustness > 0.40), activate all three asymmetries:

| Parameter | Pursuer | Evader | Notes |
|-----------|---------|--------|-------|
| `actor_lr` | 0.0005 | **0.001** | 2× |
| `critic_lr` | 0.0005 | **0.001** | 2× |
| `hidden_dim` | 128 | **256** | 2× capacity |
| `train_steps_per_phase` | 1 | **2** | Evader trains twice per phase |
| `ema_tau` | 0.001 | 0.001 | Symmetric |
| `vis_reward` | **0.1** | 0 | Pursuer LOS reward |

**EMA update rule:** After the evader completes BOTH training steps, apply EMA once. Do not apply EMA between the two evader steps.

---

## 3. Experimental Protocol

### 3.1 Phase 1: Probe (3 seeds)

**Goal:** Determine if asymmetric LR alone produces reliable convergence.

| Run | Config | Seeds | Episodes | Purpose |
|-----|--------|-------|----------|---------|
| P1 | EMA + vis_reward + asymmetric LR | 3 | 5000 | **Probe** |
| B1 | Vanilla MADDPG (G&I baseline) | 1 | 5000 | Baseline comparison |

**Go/No-Go Criteria:**
- **GO (≥2/3 seeds with robustness > 0.40):** Proceed to Phase 2 (Full A-EMA)
- **NO-GO (<2/3 seeds):** Pivot to Approach 3 (hierarchical policy). Do not invest more compute in A-EMA.

### 3.2 Phase 2: Full A-EMA (5 seeds — conditional)

**Goal:** Validate the complete A-EMA method with all three asymmetries.

| Run | Config | Seeds | Episodes | Purpose |
|-----|--------|-------|----------|---------|
| A1 | Full A-EMA (all 3 asymmetries + vis_reward) | 5 | 5000 | **Main method** |
| B2 | Symmetric EMA + vis_reward | 2 | 5000 | Ablation: no asymmetry |

**Success Criteria:**
- **Primary:** ≥3/5 seeds with robustness > 0.40
- **Secondary:** Best seed ≥ 0.60 (matches or exceeds EMA seed 0)
- **Tertiary:** No NaN or collapse in any seed

**Comparison:** Full A-EMA (5 seeds) vs your existing EMA results (7 seeds, symmetric). If A-EMA achieves 3/5 (60%) vs EMA's 1/7 (14%), that's strong evidence.

### 3.3 Phase 3: Ablations (3 seeds — conditional on Phase 2 success)

**Goal:** Isolate which asymmetry components matter.

| Run | Config | Seeds | What It Tests |
|-----|--------|-------|---------------|
| Ab1 | A-EMA without LR asymmetry | 1 | Does LR asymmetry matter? |
| Ab2 | A-EMA without hidden_dim asymmetry | 1 | Does capacity matter? |
| Ab3 | A-EMA without train_steps asymmetry | 1 | Do extra steps matter? |

---

## 4. Implementation Details

### 4.1 Code Changes

**File:** `repro_gonultas_isler_2024/agents/maddpg.py`

```python
class AsymmetricMADDPG:
    def __init__(self, config):
        # Asymmetric learning rates
        self.pursuer_actor_lr = config.get('pursuer_actor_lr', 0.0005)
        self.evader_actor_lr = config.get('evader_actor_lr', 0.001)
        self.pursuer_critic_lr = config.get('pursuer_critic_lr', 0.0005)
        self.evader_critic_lr = config.get('evader_critic_lr', 0.001)
        
        # Asymmetric hidden dimensions
        self.pursuer_hidden_dim = config.get('pursuer_hidden_dim', 128)
        self.evader_hidden_dim = config.get('evader_hidden_dim', 128)
        
        # Asymmetric training steps
        self.evader_train_steps = config.get('evader_train_steps', 1)
    
    def update(self, agent_id):
        # Standard MADDPG update but with agent-specific LR and hidden_dim
        ...
    
    def train_phase(self):
        # Pursuer trains once
        self.update('pursuer')
        
        # Evader trains N times
        for _ in range(self.evader_train_steps):
            self.update('evader')
        
        # EMA update applied ONCE after full phase
        self.update_ema()
```

**File:** `repro_gonultas_isler_2024/config_a_ema_probe.yaml`

New config file with asymmetric parameters. See Appendix A for full config.

### 4.2 vis_reward Integration

The pursuer receives an additional reward for maintaining line-of-sight to the evader:

```python
vis_reward = w_vis * (1.0 if evader_in_fov else 0.0)
```

where `w_vis = 0.1` (tuned down from 0.2 in your 10×10 framework to account for 400-step episodes).

### 4.3 Evaluation Protocol

Same as your existing checkpoint sweep:
1. Save checkpoint every 250 episodes
2. Evaluate each checkpoint against 3 opponent types (random, greedy, self-play)
3. 100 episodes per opponent type
4. Compute robustness = min(CR_random, CR_greedy, CR_selfplay)
5. Track oscillation amplitude = std(CR) over last 20 checkpoints

---

## 5. Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Asymmetric LR overcorrects → evader-dominant collapse | Medium | Monitor CR trend. If evader-dominant, reduce evader LR to 1.5×. |
| EMA smoothing breaks with 2× evader training | Low | Apply EMA after full phase, not per-step. |
| vis_reward=0.1 too weak for 400-step episodes | Medium | If pursuer shows no improvement by ep 1000, increase to 0.15. |
| Probe succeeds but full A-EMA fails | Low | Probe tests the core mechanism. Full A-EMA adds capacity, which should only help. |
| All approaches fail → no ICRA paper | Non-zero | The ∀/∃ asymmetry finding alone is workshop-worthy. Failure informs the field. |

---

## 6. Timeline

| Phase | Duration | Compute |
|-------|----------|---------|
| Code changes + smoke test | 1 day | Local |
| Phase 1 (Probe): 3 seeds | 1.5 days | niro-2 |
| Analysis + Go/No-Go decision | 0.5 days | Local |
| Phase 2 (Full): 5 seeds | 2.5 days | niro-2 |
| Analysis + ablation planning | 0.5 days | Local |
| Phase 3 (Ablations): 3 seeds | 1.5 days | niro-2 |
| **Total** | **7.5 days** | — |

---

## 7. Fallback: Approach 3 (Hierarchical Policy)

If Phase 1 fails (<2/3 seeds succeed), pivot immediately:

**Core idea:** The pursuer fails because it cannot find the evader under partial observability. Add an explicit high-level policy that switches between SEARCH and PURSUE modes.

**High-level design:**
1. **SEARCH mode:** Coverage-based exploration (your w_search grid)
2. **PURSUE mode:** Standard EMA-based pursuit when evader is visible
3. **Mode switch:** Based on FOV visibility and time since last sighting

**Why this might work when A-EMA doesn't:**
- Separates exploration from exploitation
- SEARCH mode guarantees the pursuer eventually finds the evader
- PURSUE mode uses full capacity for capture
- Addresses the "can't find" problem directly

---

## 8. Appendices

### Appendix A: Probe Config (config_a_ema_probe.yaml)

```yaml
# Environment (same as G&I)
arena_size: 16.0
max_steps: 400
capture_radius: 0.15

# MADDPG — Asymmetric
maddpg:
  episodes: 5000
  parallel_envs: 2
  batch_size: 1024
  replay_buffer_size: 100000
  
  # Pursuer
  pursuer_actor_lr: 0.0005
  pursuer_critic_lr: 0.0005
  pursuer_hidden_dim: 128
  
  # Evader — Asymmetric!
  evader_actor_lr: 0.001
  evader_critic_lr: 0.001
  evader_hidden_dim: 128
  evader_train_steps: 1
  
  # Shared
  gamma: 0.99
  tau: 0.005
  exploration_sigma: 0.1
  ema_tau: 0.001
  
  # vis_reward
  vis_reward: 0.1

# BiMDN (same as G&I)
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

# Curriculum (same as G&I)
curriculum:
  t_end_ratio: 0.3
  fov_start: 6.283
  fov_end: 2.094
  evader_vmax_start: 0.0
  evader_vmax_end: 1.5
```

### Appendix B: Full A-EMA Config (config_a_ema_full.yaml)

Same as probe, but with:
```yaml
  evader_hidden_dim: 256
  evader_train_steps: 2
```
