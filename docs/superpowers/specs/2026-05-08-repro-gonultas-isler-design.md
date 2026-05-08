# Design Spec: Reproduce Gonultas & Isler (IROS 2025)

**Date:** 2026-05-08
**Status:** Draft
**Target:** CoRL 2026 — validate whether MADDPG + BiMDN closes the PO exploitability gap
**Approach:** Standalone PyTorch reproduction in isolated directory

---

## 1. Problem Statement

Our independent SAC self-play consistently drifts to an evader-dominant equilibrium under partial observability (CR ~0.10, NE gap ~0.80). Five fixes failed: curriculum, warm-start, LSTM memory, explicit belief-state, and no curriculum.

Paper [02] (Gonultas & Isler, IROS 2025) reports stable MADDPG training with BiMDN belief encoding that achieves ~0.99 capture rate against random walk and ~0.69 against their own trained evader in simulation. The critical differences from our setup:

1. **MADDPG** (centralized critic) vs independent SAC
2. **Asymmetric sensors** (evader omnidirectional, pursuer limited FOV) vs symmetric
3. **Simpler reward** (dense zero-sum with 1000:1 ratio) vs multi-component shaped reward
4. **Different dynamics** (bicycle model pursuer, point-mass evader) vs unicycle both

This reproduction tests the hypothesis: **the centralized critic in MADDPG is sufficient to close the exploitability gap**, or alternatively, that the combination of MADDPG + BiMDN + asymmetric sensing + simple reward is necessary.

## 2. Goal

Implement a standalone reproduction of Gonultas & Isler's method that:
- Matches their simulation environment exactly (16×16m, bicycle pursuer, point-mass evader)
- Implements MADDPG with their exact hyperparameters
- Implements BiMDN belief encoder with their exact architecture
- Runs 20,000 episodes with 8 parallel envs
- Evaluates capture rate against random walk, greedy, and self-play opponents
- If CR ≥ 0.60 against self-play: the centralized critic is the key missing ingredient
- If CR < 0.40: the problem is deeper than algorithm choice

## 3. Directory Structure

```
repro_gonultas_isler_2024/
├── envs/
│   ├── __init__.py
│   ├── pursuit_evasion_env.py      # Bicycle pursuer + point-mass evader
│   └── sensor.py                   # Wedge FOV sensor with binary visibility
├── agents/
│   ├── __init__.py
│   ├── maddpg.py                   # MADDPG algorithm (actor + critic)
│   └── bimdn.py                    # BiMDN belief encoder
├── training/
│   ├── __init__.py
│   ├── train.py                    # Main training loop
│   ├── curriculum.py               # FOV decay + velocity increase curriculum
│   └── replay_buffer.py            # Shared replay buffer for MADDPG
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py                 # Eval against baselines + self-play
├── utils/
│   ├── __init__.py
│   └── logger.py                   # TensorBoard logging
├── tests/
│   ├── test_env.py                 # Env physics + sensor tests
│   ├── test_bimdn.py               # BiMDN output shape + belief tests
│   └── test_maddpg.py              # MADDPG update step tests
├── config.yaml                     # All hyperparameters
├── requirements.txt
└── README.md
```

## 4. Component Design

### 4.1 Environment (`envs/pursuit_evasion_env.py`)

**State Space:**
- Pursuer: `[x, y, δ, v, Ψ]` — position, steering angle, velocity, yaw angle
- Evader: `[x, y, ẋ, ẏ]` — position and velocities (point-mass)

**Kinematics:**

Pursuer (bicycle model):
```
ẋ = v · cos(Ψ)
ẏ = v · sin(Ψ)
δ̇ = u_δ                 (steering velocity input)
ṽ = u_v                 (longitudinal acceleration input)
Ψ̇ = (v / L) · tan(δ)    (L = wheelbase = 0.15m)
```

Constraints:
- v ∈ [-1.0, 2.5] m/s
- u_v ∈ [-2.0, 2.0] m/s² (acceleration limit)
- δ ∈ [-0.34, 0.34] rad
- u_δ ∈ [-3.2, 3.2] rad/s

Evader (point-mass):
```
ẍ = u_x
ÿ = u_y
```

Constraints:
- ẋ, ẏ ∈ [-1.5, 1.5] m/s
- u_x, u_y ∈ [-9.81, 9.81] m/s²

**Observation (11-vector per agent):**
```
o^i = [x^i, y^i, v^i_x, v^i_y, opp_x, opp_y, opp_v_x, opp_v_y, I^i, t, d]
```

Where:
- `(x^i, y^i)`: own position
- `(v^i_x, v^i_y)`: own velocity (pursuer: derived from v, Ψ; evader: ẋ, ẏ)
- `(opp_x, opp_y, opp_v_x, opp_v_y)`: opponent state if visible, else zeros
- `I^i ∈ {-1, +1}`: binary visibility flag
- `t`: current timestep / T (normalized)
- `d`: distance to opponent

All values normalized to [-1, 1].

**Sensor Model:**
- Pursuer: wedge FOV, angle = 2π/3 rad, range = 6m
- Evader: full disk, angle = 2π rad, range = ∞ (omnidirectional)
- Visibility: `I = +1` iff opponent inside wedge and within range

**Reward (zero-sum):**

Pursuer:
```
r^p_t = { -λ_capture,           if t ≥ T (timeout, evader wins)
        { +λ_capture,           if d ≤ 2c and t < T (capture)
        { -(λ_t + λ_d · d),     otherwise
```

Evader: `r^e_t = -r^p_t`

Parameters: λ_capture = 1000, λ_t = 1, λ_d = 1

**Agent radius:** c ≈ 0.15m (inferred from F1TENTH vehicle dimensions; paper states capture at d ≤ 2c but does not specify c)

**Observation normalization (per-element to [-1, 1]):**
- Positions (x, y): normalized by arena half-width (8m) → `x_norm = x / 8`, `y_norm = y / 8`
- Velocities: pursuer `v_norm = (v - 0.75) / 1.75` (maps [-1.0, 2.5] → [-1, 1]), evader `ẋ_norm = ẋ / 1.5`, `ẏ_norm = ẏ / 1.5`
- Opponent state (when visible): same normalization as above; zeros when hidden
- Visibility flag `I^i`: `+1` → `1.0`, `-1` → `-1.0`
- Time `t`: `t / T` (normalized by max episode length)
- Distance `d`: `d / d_max` where `d_max = 16√2 ≈ 22.6m` (arena diagonal)

**Episode Termination:**
- Capture: d(pursuer, evader) ≤ 2c (where c ≈ 0.15m)
- Timeout: t ≥ T = 400 steps

**Reset:**
- Random uniform placement within 16×16m arena
- Random initial velocities within bounds
- Ensure minimum starting distance (paper doesn't specify; use ≥3m)

### 4.2 BiMDN Belief Encoder (`agents/bimdn.py`)

**Purpose:** Encode observation history into a belief state representing the probability distribution over opponent position.

**Architecture:**
```
Input: [o_{t-199}, ..., o_t]  (200 observations, downsampled 10× → 20 at 1Hz)
       Each observation is an 11-vector

↓ Bidirectional LSTM (1 layer, 128 hidden units)
↓ Concatenate forward + backward hidden states → 256-D
↓ Fully connected → 32-D feature vector
↓ Split into 3 heads (one per mixture component):
   - Cluster weights head: softmax → 3 weights Π_t^i
   - Mean head: linear → 3 × 2D means M_t^i (x, y coordinates)
   - Std dev head: ELU+1 → 3 × 2D stds Σ_t^i

Output: belief state b_t^i = {Π_t^i, M_t^i, Σ_t^i} for i=1,2,3
```

**Training:**
- Belief buffer: 25,000 samples
- Warm-up: 10,000 samples (no belief updates)
- Update frequency: every 10 timesteps
- Batch size: 256
- Optimizer: Adam, LR = 0.002
- Loss: Negative log-likelihood of ground-truth opponent position under the mixture model

**Inference (two variants):**
1. **Ours (weighted mean):** Compute `b_t = Σ_i Π_t^i · M_t^i` → 2D deterministic belief state representing estimated opponent (x, y). This 2D vector is concatenated with the raw 11-D observation to form the 13-D actor input.
2. **Ours-Mixed:** Sample 16 points from the mixture to represent a mixed strategy

**BiMDN → MADDPG interaction:**
- **Actor** receives `[o_i; b_t]` (13-D: 11-D raw obs + 2-D weighted mean belief)
- **Centralized critic** receives `[o_1, b_1^{full}, o_2, b_2^{full}, a_1, a_2]` where `b^{full}` is the full mixture `{Π, M, Σ}` (3 weights + 3×2 means + 3×2 stds = 15-D per agent). Alternatively, the critic may also use the 2-D weighted mean; we implement both and compare.

For the reproduction, we implement **Ours (weighted mean)** first as the primary variant.

### 4.3 MADDPG Algorithm (`agents/maddpg.py`)

**Centralized Critic Architecture:**
```
Input: [o_1, b_1, o_2, b_2, a_1, a_2]  (both agents' belief-augmented observations + both actions)
↓ Concatenate → full state-action vector
↓ MLP: hidden=[128, 128], ReLU
↓ Output: Q-value (single scalar)
```

Note: Both actor and critic use identical architecture per paper: 2 hidden layers of 128 units each with ReLU activations. The centralized critic concatenates all agents' belief-augmented observations and actions into the input vector.

**Decentralized Actor Architecture:**
```
Input: observation o_i (13-vector: 11-D raw obs + 2-D belief weighted mean)
↓ MLP: hidden=[128, 128], ReLU
↓ Output: action a_i (2D continuous), tanh → [-1, 1]
↓ Scale to action bounds
```

**Training Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Parallel envs | 8 |
| Replay buffer size | 100,000 |
| Batch size | 512 |
| Actor LR | 0.0005 |
| Critic LR | 0.0005 |
| Optimizer | Adam |
| γ (discount) | 0.99 |
| τ (soft update) | 0.005 |
| Exploration noise | Gaussian, σ = 0.1 |
| Frame stacking | 2 frames |
| Frame skipping | Every 2nd frame (action repeated) |
| MADDPG update freq | 1 update per env step (after replay buffer ≥ batch size) |
| Max episode steps | 400 |

**MADDPG Update:**
1. Sample batch from replay buffer
2. For each agent i:
   - Compute target actions: `a'_j = μ'_j(o_j)` for all agents j (using target actors)
   - Compute target Q: `Q'_i = r_i + γ · Q'_i(o_1, b_1, o_2, b_2, a'_1, a'_2)`
   - Update critic: minimize MSE between Q_i and Q'_i
   - Update actor: maximize Q_i(o_1, b_1, o_2, b_2, a_1, μ_i(o_i), a_2)
3. Soft update target networks: `θ' ← τ·θ + (1-τ)·θ'`

### 4.4 Curriculum (`training/curriculum.py`)

**Two simultaneous curricula:**

1. **Pursuer sensor decay:**
   ```
   V_α(t) = 2π · (1 - t / t_end) + (2π/3) · (t / t_end)
   ```
   Where `t_end = 0.3 × total_episodes = 6,000 episodes`
   - Starts at full omnidirectional vision (2π)
   - Linearly decays to limited FOV (2π/3)

2. **Evader velocity increase:**
   ```
   v_max(t) = 0 · (1 - t / t_end) + 1.5 · (t / t_end)
   ```
   - Starts at 0 velocity (stationary evader)
   - Linearly increases to full 1.5 m/s

**Implementation:** Curriculum parameters updated every episode, not every step.

### 4.5 Training Loop (`training/train.py`)

```python
for episode in range(20000):
    # Update curriculum parameters
    update_curriculum(episode)
    
    # Run 8 parallel environments
    for env_id in range(8):
        obs_p, obs_e = env.reset()
        for step in range(400):
            # Select actions with exploration noise
            a_p = actor_p(obs_p) + N(0, 0.1)
            a_e = actor_e(obs_e) + N(0, 0.1)
            
            # Step environment
            next_obs_p, next_obs_e, r_p, r_e, done = env.step(a_p, a_e)
            
            # Store transition
            replay_buffer.add(obs_p, obs_e, a_p, a_e, r_p, r_e, next_obs_p, next_obs_e, done)
            
            obs_p, obs_e = next_obs_p, next_obs_e
            
            # Update BiMDN every 10 steps (after warm-up)
            if step % 10 == 0 and buffer_size > 10000:
                bimdn.update()
            
            if done:
                break
    
    # Update MADDPG every episode (or every N steps)
    if replay_buffer.size >= 512:
        for _ in range(num_updates):
            maddpg.update()
    
    # Evaluate every 250 episodes
    if episode % 250 == 0:
        evaluate()
```

### 4.6 Evaluation (`evaluation/evaluate.py`)

**Evaluation protocol (matching paper):**
- Create checkpoints every 250 episodes
- Evaluate each checkpoint against:
  1. Random walk evader (500 episodes)
  2. Greedy evader (500 episodes)
  3. Trained evader (self-play, 500 episodes)
  4. Random walk pursuer (500 episodes)
  5. Greedy pursuer (500 episodes)
- Report: average capture time, standard deviation, capture rate
- Select best performing strategies by average capture rate

**Baselines:**
- Random walk: sample random actions each step
- Greedy: move directly toward/away from opponent

### 4.7 Frame Stacking and Action Skipping

**Frame skipping (action repetition):**
- Simulation timestep: 0.1s (10 Hz)
- Policy runs at 5 Hz: action selected every 2 simulation steps
- Selected action is repeated for 2 consecutive simulation steps
- Effective episode length: 400 simulation steps = 200 policy decisions

**Frame stacking:**
- The actor network receives the last 2 observations concatenated
- At policy decision step k (corresponding to simulation steps 2k and 2k+1):
  - `stacked_obs = [o_{2k}, o_{2k-1}]` (22-D vector, or 26-D with belief)
  - For k=0, use `o_0` duplicated: `[o_0, o_0]`
- This gives the network temporal context without recurrent connections

### 4.8 Action Normalization and Scaling

**Actor output:**
- Raw output: 2D vector in [-1, 1] via tanh
- Scaled to actual action bounds:
  - Pursuer: `[u_δ, u_v]` where `u_δ ∈ [-3.2, 3.2]`, `u_v ∈ [-2.0, 2.0]`
  - Evader: `[u_x, u_y]` where `u_x, u_y ∈ [-9.81, 9.81]`

**Scaling formula:** `a_actual = a_tanh · action_scale`
where `action_scale` is a per-agent vector of upper bounds.

## 5. Testing Strategy

### 5.1 Environment Tests (`tests/test_env.py`)

| Test | Purpose |
|------|---------|
| `test_bicycle_dynamics` | Verify pursuer kinematics match Eq. (1) |
| `test_point_mass_dynamics` | Verify evader kinematics match Eq. (2) |
| `test_sensor_visibility` | FOV wedge correctly flags visible/opponent |
| `test_capture_condition` | d ≤ 2c triggers capture |
| `test_reward_structure` | Verify zero-sum + λ_capture=1000 |
| `test_observation_normalization` | All values in [-1, 1] |

### 5.2 BiMDN Tests (`tests/test_bimdn.py`)

| Test | Purpose |
|------|---------|
| `test_output_shapes` | Π: (batch, 3), M: (batch, 3, 2), Σ: (batch, 3, 2) |
| `test_mixture_weights_sum_to_one` | Softmax produces valid distribution |
| `test_std_positive` | ELU+1 ensures positive std |
| `test_nll_loss_decreases` | Training reduces negative log-likelihood |

### 5.3 MADDPG Tests (`tests/test_maddpg.py`)

| Test | Purpose |
|------|---------|
| `test_critic_output_shape` | Q(s, a) → scalar |
| `test_actor_output_shape` | μ(o) → 2D action |
| `test_target_networks_update` | Soft update changes target params |
| `test_replay_buffer_sampling` | Batch has correct shapes |

## 6. Success Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Pursuer CR vs random walk | ≥ 0.90 | Basic competence |
| Pursuer CR vs greedy evader | ≥ 0.70 | Learns search behavior |
| Pursuer CR vs self-play evader | ≥ 0.50 | Stable co-evolution |
| NE gap (self-play) | ≤ 0.20 | Near Nash equilibrium |
| Matches paper's reported CR | Within ±0.10 | Reproduction validated |

If all criteria pass: MADDPG + BiMDN is the key to closing the gap.
If pursuer CR vs self-play < 0.40: The problem is structural (asymmetric sensing or dynamics), not algorithmic.

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| BiMDN implementation subtle | Start with vanilla MADDPG (no BiMDN), verify it trains; then add BiMDN |
| Bicycle model physics bugs | Compare against analytical solutions for simple cases |
| Training instability | Log Q-values and policy entropy; reduce LR if diverging |
| Slow training | 8 parallel envs on GPU; each episode is only 40s sim time |
| Paper details ambiguous | Document all assumptions (e.g., c=0.15m inferred from F1TENTH) and test sensitivity |

## 8. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Environment + sensor | 1 day | `test_env.py` passes |
| MADDPG core | 1 day | `test_maddpg.py` passes |
| BiMDN + curriculum | 1 day | `test_bimdn.py` passes |
| Training loop + eval | 1 day | First training run starts |
| Training + debugging | 3-4 days | Reach 20,000 episodes |
| Analysis + comparison | 1-2 days | Results report, comparison with our SAC results |
| **Total** | **7-9 days** | **Go/no-go for CoRL** |

---

**Dependencies:** PyTorch, NumPy, Gymnasium, TensorBoard
**Compute target:** GPU (niro-2), ~6-8 hours for 20,000 episodes with 8 parallel envs
