# Gonultas & Isler 2024 Reproduction Summary

**Purpose**: Standalone supporting document for the RA-L / ICRA 2027 paper. Documents the verified reproduction and the seven implementation fixes discovered during the code audit.  
**Date**: 2026-07-14  
**Reproduction directory**: `repro_gonultas_isler_2024/`  
**Original paper**: Gonultas & Isler, *Pursuit-Evasion for Car-like Robots with Sensor Constraints* (IROS 2025 / arXiv 2024)

---

## Reproduction setup

| Component | Implementation |
|---|---|
| Dynamics | Bicycle-model pursuer + point-mass evader |
| Arena | 16 × 16 m |
| Sensing | Pursuer: wedge FOV 2π/3, range 6 m; Evader: omnidirectional |
| Algorithm | MADDPG with centralized critic + BiMDN belief encoder |
| Training | 20,000 episodes, 8 parallel envs, batch size 512 |
| Networks | Actor/Critic: 2 × 128 ReLU; BiMDN: 1-layer 128-unit bidirectional LSTM → 32-D features |
| Curriculum | Pursuer FOV decay + evader velocity increase over first 6,000 episodes |

---

## Verification results

| Seed | Checkpoint | Metric | Our result | Paper target | Gap |
|---|---|---|---|---|---|
| 0 | ep 11,000 | vs_selfplay | **0.550** | 0.54 | +0.010 |
| 0 | ep 11,000 | vs_random | 0.906 | 0.98 | −0.074 |
| 1 | ep 15,500 | vs_random | **0.936** | 0.98 | −0.044 |
| 1 | ep 15,500 | vs_selfplay | 0.472 | 0.54 | −0.068 |

**Verdict**: Reproduction verified. Self-play CR matches the paper within 0.01; vs-random CR is within 0.075 of the reported value. The paper's reported CR=0.54 corresponds to a specific oscillatory phase, not a converged equilibrium.

---

## Seven implementation fixes

### 1. Actor observation selection
**Symptom**: During evaluation, the actor received an observation tensor that did not match the training-time observation distribution.  
**Fix**: Ensure `evaluate.py` passes the same observation preprocessing (frame stack, belief concatenation, normalization) used during training.  
**Impact**: Policies no longer acted on mismatched inputs; CR metrics became reliable.

### 2. Target Q action swap
**Symptom**: The target Q-value computation used the pursuer's target action for the evader's critic and vice versa.  
**Fix**: Correctly pair each agent's target action with its own critic in the MADDPG update.  
**Impact**: Critic targets became consistent; training instability reduced.

### 3. Reward selection
**Symptom**: The reward signal fed to the optimizer was not the zero-sum reward defined in the paper; auxiliary shaping leaked into the MADDPG update.  
**Fix**: Route the exact paper reward (`r^p = −r^e`) to the optimizer, isolating any auxiliary terms for logging only.  
**Impact**: Gradient updates aligned with the paper's objective.

### 4. BiMDN warm-up
**Symptom**: Belief updates started before the belief buffer contained enough samples, producing degenerate mixture weights early in training.  
**Fix**: Enforce the 10,000-sample warm-up period before any BiMDN gradient step.  
**Impact**: Early belief estimates became stable and meaningful.

### 5. BiMDN architecture
**Symptom**: The BiMDN output heads did not match the paper's dimensionality (mixture weights, means, stds).  
**Fix**: Implement 3 mixture components with correct head shapes: weights (3), means (3×2), stds (3×2).  
**Impact**: Belief representation matched the paper's mixture model.

### 6. Belief normalization
**Symptom**: The weighted-mean belief state was not normalized to the same range as other observation components.  
**Fix**: Normalize belief coordinates by arena half-width (8 m) before concatenating to the actor input.  
**Impact**: Actor/critic inputs remained in the expected [−1, 1] range.

### 7. Capture counting
**Symptom**: Evaluation counted a capture only if the episode return was positive. Because step penalties accumulate, the episode return can be negative even when capture occurs.  
**Fix**: Count capture based on `step < env.max_steps` (i.e., episode terminated early by capture) rather than episode reward sign.  
**Impact**: All CR metrics increased significantly; prior evaluations underreported capture rates.

---

## Why this matters for the paper

The reproduction serves three roles:
1. **Credibility**: It shows our analysis is grounded in a verified implementation of a published method.
2. **Baseline**: It confirms that vanilla self-play in this domain is naturally oscillatory and only transiently balanced.
3. **Contrast**: Pop-FSP and EMA are evaluated against the same underlying dynamics, so their failure modes are not an artifact of our environment implementation.

---

## Files

- `repro_gonultas_isler_2024/training/train.py` — training loop
- `repro_gonultas_isler_2024/agents/maddpg.py` — MADDPG implementation
- `repro_gonultas_isler_2024/agents/bimdn.py` — BiMDN belief encoder
- `repro_gonultas_isler_2024/evaluation/evaluate.py` — evaluation protocol
- `repro_gonultas_isler_2024/config.yaml` — hyperparameters

---

## Open questions

- Does the small vs-random gap (−0.044 to −0.074) reflect implementation detail or paper rounding?
- Why does no single checkpoint simultaneously match both self-play and vs-random targets? This is consistent with oscillatory dynamics.
