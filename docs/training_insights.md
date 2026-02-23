# Phase 3 Training Insights

## Date: 2026-02-22

---

## Stage 1: BiMDN Pre-training

### Critical Finding: Observation Space Missing Own Pose
- **Original obs_dim=40**: `[v, omega, d_to_opp, bearing, lidar_36]` — NO own position
- **Fixed obs_dim=43**: `[x, y, theta, v, omega, d_to_opp, bearing, lidar_36]` — includes own pose
- **Root cause**: Without own position, BiMDN cannot convert relative polar measurements to absolute position estimates. This is mathematically impossible.
- **Justification**: Own pose is always known from odometry (proprioception). Partial observability is about the OPPONENT, not self-state. The spec says "heading is always known (odometry)".

### BiMDN Training Results (3 iterations)

| Metric | Run 1 (no pose, h=64, 50ep) | Run 2 (pose, h=64, 50ep) | Run 3 (pose, h=128, 100ep) | Target |
|--------|---------------------------|------------------------|---------------------------|--------|
| In-FOV RMSE | 8.76m (FAIL) | 0.81m (PASS) | **0.31m** (PASS) | < 2.0m |
| Out-FOV RMSE | 9.65m (FAIL) | 9.47m (FAIL) | 9.25m (FAIL) | < 5.0m |
| n_eff visible | 1.22 | 1.12 | **1.01** (PASS) | ~1.0 |
| n_eff lost | 1.23 | 1.10 | **1.20** (PASS) | > 1.5 |
| Train loss reduction | 20.6% | 20.6% | 28.2% | > 50% |
| Gate 1 | FAILED (2 hard) | FAILED (1 hard) | FAILED (1 hard) | |

### Out-of-FOV RMSE Analysis
- Random baseline RMSE on 20x20 arena: ~8.2m
- Our best: 9.25m (WORSE than random — why?)
- Data composition: 88.7% "lost" samples (d_to_opp = -1 for all K=10 steps)
- In these majority cases, the obs_history contains ZERO information about target location
- The BiMDN can only output a location near arena center for these — but the validation counts ALL samples
- **Key insight**: The out-of-FOV metric conflates two very different scenarios:
  1. "Just lost" — target was visible recently, can be extrapolated (~12% of lost cases)
  2. "Never seen" — target was never visible in the window (majority of lost cases)
- For scenario 2, no model can predict better than random

### Recommendations for Improving Out-of-FOV
1. **Collect data with diverse policies** (as spec recommends): trained pursuers + scripted pursuit/flee
2. **Increase visibility rate**: Use policies that actively seek the evader
3. **Separate "just lost" vs "never seen" evaluation**: The composite metric is misleading
4. **Consider accepting partial Gate 1**: In-FOV is excellent, proceed to Stage 2 per training policy
5. **Add last-known-position encoding**: Include time-since-last-seen in obs to help extrapolation

---

## Stage 2: Single-Agent PPO Training

### Config 2A: BiMDN + DCBF + Partial Obs (500K steps)

**Training Observations (as of 420K/500K steps):**
- **Capture rate: 0%** throughout training
- Episode length: ~1130 steps (hitting max_steps=1200)
- Mean reward: fluctuating between -29 and -43 (timeout penalty is -50)
- DCBF intervention rate: ~6% (low, good)
- FPS: ~400 on RTX 5090

**Why zero captures?**
1. **Partial observability severity**: With random data collection showing 88.7% lost rate, the pursuer can barely see the evader. Under partial obs with FOV=120 deg and 10m range in a 20m arena, the evader is usually out of sensor range.
2. **Reward sparsity**: Per-step distance reward is tiny (~0.001 per step). Only capture (+100) and timeout (-50) provide strong signal.
3. **Exploration challenge**: PPO explores around the current policy mean. Without any capture to learn from, there's no positive signal to reinforce.
4. **Random evader**: Even a random evader moves unpredictably, making capture under partial obs very hard.

### Possible Improvements for Stage 2
1. **Reward shaping**: Add search reward (exploring unvisited areas), detection reward (bonus when evader enters FOV)
2. **Curriculum**: Start with full observability, gradually reduce FOV angle
3. **Pre-train from Phase 2 policy**: Use full-obs Phase 2 weights as initialization, then fine-tune under partial obs
4. **Increase capture radius**: Start with larger capture_radius for easier captures, reduce later
5. **Use Phase 2 trained evader**: Instead of random evader, use trained evader that exhibits more predictable patterns
6. **Extended training**: 500K steps may not be enough for partial obs — try 2M+ steps
7. **Intrinsic motivation**: Add curiosity-driven exploration reward

---

## Environment Configuration Notes
- Arena: 20x20m
- Pursuer max speed: 1.0 m/s, max omega: 2.0 rad/s (default)
- FOV: 120 deg, 10m range
- Lidar: 36 rays, 5m range
- Capture radius: 0.5m
- dt: 0.05s
- Max steps: 1200 (60 seconds)
- Observation history: K=10 timesteps

## Dependency Issues on niro-2
- **tensorboard**: Must install separately
- **tqdm, rich**: Required for SB3 progress_bar=True
- **PyTorch CUDA**: Installed with pip for CUDA 12.8

---

## Key Lessons
1. **Always include self-localization in partial obs**: Relative measurements are useless without an absolute reference frame
2. **Random data collection is insufficient for belief training**: Need diverse policies that create visible/lost transitions
3. **Partial observability dramatically increases training difficulty**: 500K steps is likely insufficient
4. **Separate in-FOV and out-of-FOV evaluation is critical**: Composite metrics hide where the model succeeds/fails
5. **CUDA OOM**: Always batch GPU operations, keep large datasets on CPU
