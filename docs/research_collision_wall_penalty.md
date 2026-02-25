# Research Report: Collision & Wall Penalty Design for Safe RL in Pursuit-Evasion

## 1. Executive Summary

Wall and obstacle collisions in our PE environment are physically enforced (position clipping / projection), but there is no reward signal discouraging agents from pressing against boundaries. This report reviews the literature on collision penalty design, its relationship to safe RL / CBF approaches, and provides calibrated recommendations for our game setup.

**Key finding**: When CBF safety filters are used (as in our project), collision penalties should be **moderate** — the CBF does the heavy lifting for safety, while the penalty helps the policy *internalize* safe behavior. The state-of-the-art dual approach (CBF filter + CBF-derived reward) outperforms either alone (99.0% vs 98.8% filter-only vs 91.9% reward-only) [N17].

---

## 2. Literature Review

### 2.1 Three Paradigms for Collision Avoidance in RL

| Paradigm | Mechanism | Pros | Cons | Papers |
|----------|-----------|------|------|--------|
| **A. Reward penalty only** | Negative reward on collision/proximity | Simple, no extra infrastructure | No formal guarantees, sensitive to weight tuning | TurtleBot3, CADRL, N20 |
| **B. Hard constraint only** (CBF/HJ filter) | QP-based action correction | Formal safety guarantees | Policy doesn't internalize safety; degrades to 38.7% without runtime filter [N17] | N18, Paper 06, Paper 16 |
| **C. Dual: CBF filter + reward penalty** | Both active during training | Best performance (99%), robust to domain randomization, policy safe even without filter | More complex | **N17 (recommended)**, Paper 05 |

### 2.2 Collision Penalty Values from Published Work

#### Navigation Tasks

| Source | Collision Penalty | Goal Reward | Ratio | Notes |
|--------|------------------|-------------|-------|-------|
| **N17** (Yang CBF-RL, 2025) | **-1.0** (obstacle), **-1.0** (wall) | +1.0 (goal) | 1:1 | + r_cbf weight=100 dominates |
| TurtleBot3 Official | -50 | +100 | 1:2 | DQN, LiDAR |
| CADRL (social nav) | -0.25 (collision), -0.1+0.05d (proximity) | +1.0 | 1:4 | Classic social navigation |
| **N20** (Jeng & Chiang, 2023) | None (episode concatenation) | +5 / -10 | N/A | Argues large penalties destabilize training |
| HMP-DRL (2024) | -1.0 to -2.5 (entity-based) | +3.0 | 1:1 to 1:3 | Social navigation |
| Arce et al. (2023) | -10*exp(-5*(d-0.335)) | +100 | Continuous | SAC mapless |

#### Pursuit-Evasion Games

| Source | Collision Penalty | Capture Reward | Ratio | Notes |
|--------|------------------|----------------|-------|-------|
| **N19** (Yang IROS 2023) | f_c(x) smooth rolloff + immobilization | 10/max(d,d_cap) | Variable | Multi-agent PE, action masking |
| **N16** (Kokolakis CDC 2022) | psi*B(x), psi=500 | Implicit (origin=capture) | Barrier → ∞ | 1v1 PE with obstacle, barrier in cost |
| **N18** (Deng 2024) | None in reward | ±0.1 tracking | N/A | CBF-only filter for pursuit |
| Chen et al. (2024) Multi-UAV PE | -10 | +2 | 5:1 | Obstacle breach penalty |
| MPE Simple Tag (PettingZoo) | Progressive: min(exp(2x-2), 10) | +10 / -10 | Up to 1:1 | Boundary penalty function |
| OpenAI Hide-and-Seek (2019) | "penalized if too far outside" | +1/-1 per step | — | Emergent tool use |

### 2.3 Relationship to Safe RL / CBF

From our downloaded papers and existing collection:

**Paper N17 (CBF-RL, Yang et al. 2025)** — Most relevant to our setup:
- Uses DTCBF safety filter during training + CBF-derived reward penalty
- Reward: `r = r_nominal + w * r_cbf` where w=100
- r_cbf has two components:
  1. **Constraint violation**: `max(∇h(q)ᵀv + αh(q), 0)` — penalizes unsafe proposed actions
  2. **Filter intervention**: `exp(-||v_policy - v_safe||²/σ²) - 1` — penalizes reliance on filter
- **Ablation result**: Dual (filter+reward) achieves 99.0% success; filter-only 98.8% but drops to 38.7% without runtime filter; reward-only 91.9%

**Paper N16 (Safe Finite-Time PE, Kokolakis 2022)** — Barrier in cost function:
- Integrates barrier function B(x) = g(x)/h(x) directly into the running cost
- Weight ψ=500 (very large) — safety dominates optimality
- B(x) → ∞ as agent approaches obstacle boundary
- Proven safe: pursuer captures evader while avoiding obstacle

**Paper N18 (CBF-Safe Pursuit, Deng 2024)** — CBF-only, no penalty:
- Three CBFs (collision, sensing range, input saturation) form QP safety filter
- Reward is purely about tracking quality (±0.1)
- No collision penalty in reward — CBF handles all safety
- Works but policy doesn't internalize safety

**Paper 05 (CBF-RL Survey)** — CBF reward shaping formula:
- `r_safe = r_task + λ * max(0, -ḣ - αh)`
- Combined filter + reward shaping outperforms either alone

**Paper 30 (CASRL)** — Dual critics:
- Separates goal-reaching critic from collision-avoidance cost critic
- Conflict-averse gradient manipulation (CAPO) finds optimal weighting
- Avoids the penalty-weight tuning problem entirely

### 2.4 Penalty Calibration Guidelines from Literature

**Risk of too-large penalties:**
- Agent freezes, spins in place, or refuses to move toward goals
- Success rates can drop from >98% to 22% with disproportionate penalties [Larsen et al. 2025]
- High variance in policy gradient updates, convergence problems
- "A sudden large penalty might cause the numerical calculation to become unstable" [N20]

**Risk of too-small penalties:**
- Agent treats collisions as acceptable cost
- Accumulated step penalties may exceed collision penalty, incentivizing collision as shortcut

**Recommended strategies:**
1. **Start with penalty ≈ task reward magnitude** (1:1 ratio), tune from there
2. **Use progressive/continuous penalties** rather than binary (smooth rolloff near boundaries)
3. **When CBFs are used, penalties can be smaller** — CBF prevents actual violations, penalty just shapes learning
4. **Adaptive Lagrangian multiplier** outperforms fixed values [ICLR 2023 blog, Yao et al. AAAI 2022]

---

## 3. Analysis for Our Game Setup

### 3.1 Current Reward Magnitudes

| Reward Component | Per-Step Magnitude | Terminal? | Notes |
|-----------------|-------------------|-----------|-------|
| Distance shaping (scale=1.0) | ~0.0035 per 0.1m closed | No | Default; too weak for PPO |
| Distance shaping (scale=10.0) | ~0.035 per 0.1m closed | No | Used in training runs |
| Capture bonus | +100.0 | Yes | Pursuer terminal reward |
| Timeout penalty | -100.0 | Yes | Pursuer terminal penalty |
| Visibility reward | ±1.0 | No | Evader (Mode B) |
| Survival bonus | +1.0 | No | Evader (Mode B) |
| PBRS obstacle-seeking | ~0.07 per 0.1m (w=10) | No | Evader only |
| **w_collision** (current) | 0.0 (disabled) | No | Obstacle collision |
| **w_wall** (current) | 0.0 (disabled) | No | Wall collision |

### 3.2 Physical Context

| Parameter | Value |
|-----------|-------|
| Arena | 20×20m (or 10×10m in some runs) |
| Robot radius | 0.15m |
| v_max | 1.0 m/s (both agents) |
| dt | 0.05s |
| Max episode | 1200 steps (60s) |
| Distance per step at v_max | 0.05m |
| Arena diagonal (d_max) | 28.28m (20×20) or 14.14m (10×10) |

### 3.3 How Often Do Wall/Obstacle Contacts Occur?

Wall contacts happen when an agent drives into the arena boundary and gets clipped. In a 20×20 arena, agents can spend many steps at the wall if heading into it. The contact is **per-step** (every step the agent is pressed against the wall counts as a contact).

Obstacle collisions are resolved by projection — the agent is pushed to the obstacle surface with tangential sliding. Again, per-step if the agent keeps driving into the obstacle.

---

## 4. Recommendations

### 4.1 Penalty Magnitude

Given our setup (capture_bonus=100, distance_scale=10, per-step distance reward ~0.035):

| Penalty | Recommended Value | Rationale |
|---------|------------------|-----------|
| **w_wall** | **0.1 – 1.0** | See analysis below |
| **w_collision** | **0.1 – 1.0** | Same reasoning |

**Rationale for 0.1–1.0 range:**

1. **Per-step distance reward** is ~0.035 (with scale=10). A wall penalty of 0.1–1.0 means hitting the wall costs 3–30× more than the distance progress gained in one step. This is significant enough to discourage wall-hugging but not so large as to cause freezing.

2. **Over an episode** of 1200 steps, if an agent spends 100 steps at a wall:
   - w_wall=0.1: cumulative penalty = -10 (10% of capture_bonus)
   - w_wall=1.0: cumulative penalty = -100 (equals capture_bonus — too harsh for sustained contact)
   - w_wall=0.5: cumulative penalty = -50 (50% of capture_bonus — meaningful deterrent)

3. **Comparison to N17** (the most relevant reference): They use r_obstacle = r_wall = -1.0 with r_goal = +1.0 (1:1 ratio). Our capture_bonus is 100, so a proportional penalty would be ~1.0. But N17's collision terminates the episode, while ours is per-step. For per-step penalties, a smaller value (0.1–0.5) is more appropriate.

4. **Comparison to MPE Simple Tag**: Uses progressive boundary penalty starting at 0, ramping to 10 (= capture reward). Our equivalent would be ~1.0–10.0 for a binary per-contact penalty.

### 4.2 Starting Point Recommendation

```
w_wall = 0.5
w_collision = 0.5
```

**Why 0.5:**
- ~14× the per-step distance reward (strong enough to matter)
- Over 100 steps of wall contact: cumulative -50 (half of capture_bonus)
- Over 10 steps of contact: cumulative -5 (5% of capture_bonus — minor)
- Same for both wall and obstacle (no reason to differentiate without evidence)
- Easy to scale up/down by 2× if needed

### 4.3 Alternative: CBF-Derived Penalty (Advanced)

Following N17's approach, instead of a fixed penalty, use a penalty proportional to how much the safety filter had to intervene:

```python
r_cbf = w_cbf * (exp(-||a_nominal - a_safe||² / σ²) - 1)
```

This is more principled but requires DCBF filter intervention data. Since our DCBF filter already exists and we track interventions, this could be implemented as a future enhancement.

### 4.4 What NOT To Do

1. **Don't set w_wall or w_collision > 5.0** — risks freezing/spinning behavior
2. **Don't use the same penalty for per-step contacts and terminal events** — per-step penalties accumulate
3. **Don't assume zero-sum preservation** — w_wall and w_collision are non-zero-sum (both agents penalized independently), which is correct for physical damage but breaks the zero-sum game structure slightly
4. **Don't skip the penalty just because DCBF is active** — N17's ablation shows dual (filter+penalty) outperforms filter-only, especially for policy robustness

---

## 5. Experimental Plan

### Phase 1: Baseline Comparison
```bash
# Run A: No penalties (current default)
--w_wall 0.0 --w_collision 0.0

# Run B: Moderate penalties
--w_wall 0.5 --w_collision 0.5

# Run C: Lighter penalties
--w_wall 0.1 --w_collision 0.1
```

### Phase 2: Tune Based on Results
- If agents still wall-hug in Run B: increase to 1.0
- If agents are overly cautious in Run B: decrease to 0.1
- Monitor: wall contact frequency, capture rate, episode length, reward curves

### Metrics to Track
- Per-episode wall contacts (pursuer + evader)
- Per-episode obstacle collisions
- Capture rate and episode length (should not degrade significantly)
- Total reward decomposition (base reward vs penalty)

---

## 6. References

### Newly Downloaded Papers (N16–N20)
- **N16**: Kokolakis & Vamvoudakis, "Safe Finite-Time RL for PE Games," CDC 2022
- **N17**: Yang, Werner, de Sa, Ames, "CBF-RL: Safety Filtering RL in Training with CBFs," arXiv 2510.14959, 2025
- **N18**: Deng, Gao, Xiao, Feroskhan, "Ensuring Safety in Target Pursuit Control: A CBF-Safe RL Approach," arXiv 2411.17552, 2024
- **N19**: Yang et al., "Large Scale PE Under Collision Avoidance Using DRL," IROS 2023
- **N20**: Jeng & Chiang, "End-to-End Autonomous Navigation with Survival Penalty Function," Sensors 23(20):8651, 2023

### From Existing Collection
- **Paper 05**: Yang et al., "CBF-RL Safety Filtering," 2025
- **Paper 03**: Guerrier et al., "CBF-RL Survey," 2024
- **Paper 13**: Kokolakis & Vamvoudakis, "Safe Finite-Time PE," CDC 2022 (= N16)
- **Paper 30**: Zhou et al., "CASRL: Conflict-Averse Safe RL," CAAI 2023
- **Paper 31**: Zhu et al., "Safe HJ MADDPG," JIRS 2024

### Web Sources
- TurtleBot3 ML Documentation (ROBOTIS)
- PettingZoo MPE Simple Tag Documentation
- OpenAI Hide-and-Seek (2019)
- ICLR 2023 Blog: "Adaptive Reward Penalty in Safe RL"
- Yao et al., "Conservative and Adaptive Penalty for Model-Based Safe RL," AAAI 2022
- Larsen et al., "RL Reward Functions for Autonomous Maritime Collision Avoidance," JMSE 2025
