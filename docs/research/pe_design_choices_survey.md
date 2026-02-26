# Pursuit-Evasion RL Design Choices: Literature Survey

**Date**: 2026-02-26 (Session 54)
**Purpose**: Benchmark our fundamental game design choices against the PE RL literature (2020-2026)

---

## 1. Arena Size and Speed Ratio

### What Other Papers Use

| Paper / Environment | Arena Size | Pursuer Speed | Evader Speed | Speed Ratio (E/P) |
|---|---|---|---|---|
| Bilgin & Karimpanal 2024 (car-like PE) | 20m x 20m | up to 2.5 m/s | up to 2 m/s per axis | ~0.8 (pursuer faster) |
| Multi-UAV PE (arXiv 2409.15866, 2024) | radius 0.9m cylinder | 1.0 m/s | 1.3 m/s | 1.3 (evader 30% faster) |
| Quadrotor PE (arXiv 2506.02849, 2025) | 12m x 12m x 6m | 8 m/s max commanded | 8 m/s max commanded | 1.0 (equal) |
| Dog-Sheep game (PMC 2022) | Circle R=200 units | 16 units/step | Variable | Range (1, pi+1) |
| Multi-agent military PE (PMC 2021) | 20km x 20km | 1.0 km/s max | 1.3 km/s max | 1.3 (evader faster) |
| MatrixWorld (arXiv 2307.14854) | 40x40 grid | Same | Same | 1.0 (equal) |
| PettingZoo Simple Tag | Unbounded (penalty >0.9) | Slower (adversaries) | Faster (good agents) | >1.0 (evader faster) |
| OpenAI Hide-and-Seek (2019) | Bounded room | Same | Same | 1.0 (equal) |
| Decentralized Multi-Agent PE (HAL 2022) | Continuous | unicycle constrained | omnidirectional | ~1.0 with maneuverability diff |
| **Ours** | **10m x 10m** | **1.0 m/s** | **1.0 m/s** | **1.0 (equal)** |

### Key Findings

1. **Arena sizes vary enormously** depending on the application domain: 0.9m (drone lab) to 20km (military). The ratio of arena size to agent speed (time to cross) matters more than absolute size.

2. **Speed ratios cluster into three design philosophies**:
   - **Evader faster** (ratio >1.0): Most common in single-pursuer setups. Makes the game non-trivial in bounded domains. The Multi-UAV paper uses 1.3x, the military PE uses 1.3x, PettingZoo Simple Tag gives the "good agent" (prey) a speed advantage.
   - **Equal speed** (ratio = 1.0): Used in grid-based and some continuous environments, particularly multi-pursuer setups where cooperation is the interesting challenge. Also in OpenAI Hide-and-Seek where environment manipulation (not speed) creates complexity.
   - **Pursuer faster** (ratio <1.0): Less common for 1v1. Used when the evader has other advantages (maneuverability, obstacles, partial observability, head start).

3. **Equal speed in bounded domains is problematic for 1v1**: Classical game theory (Besicovitch 1952) shows that in a disk, a man can evade a lion of equal speed indefinitely. However, this requires perfect play. In practice with RL, the pursuer can learn to corner the evader, making the game trivially winnable for the pursuer. This creates a tension: the evader can theoretically survive forever, but RL training converges to pursuer dominance.

4. **Arena-to-speed ratio** (time to cross the arena): Our 10m arena with 1.0 m/s speed = 10 seconds to cross. The car-like PE paper uses 20m arena with 2.5 m/s = 8 seconds to cross. The quadrotor PE uses 12m with 8 m/s = 1.5 seconds to cross. A 10-second crossing time is on the longer end.

### Implications for Our Project

Our equal-speed 1v1 setup in a bounded arena is an unusual choice. Most 1v1 PE papers give the evader a speed advantage (typically 10-30% faster) to make the game non-trivial. With equal speeds in a 10m x 10m arena, the pursuer will eventually corner and catch the evader, making the game "trivially solvable" for the pursuer. This may explain why our evader struggles to learn meaningful evasion strategies.

**Options to consider**:
- Give evader 10-30% speed advantage
- Add obstacles that the evader can exploit
- Add partial observability that favors the evader
- Give evader a goal (reach-avoid game) rather than pure survival
- Reduce arena size to make wall exploitation more viable

---

## 2. Episode Length

### What Other Papers Use

| Paper / Environment | Max Steps | dt | Total Time | Arena Cross Time |
|---|---|---|---|---|
| Bilgin & Karimpanal 2024 | 500 | 0.1s | 50s | 8-20s |
| Multi-UAV PE 2024 | 800 | varies (100Hz ctrl) | ~8s | ~1.8s |
| Quadrotor PE 2025 | 600 | 0.016s (62.5Hz) | ~10s | ~1.5s |
| Dog-Sheep 2022 | ~11 steps | varies | ~11 steps | N/A |
| Multi-agent PE 2021 | 50 | varies | varies | varies |
| MatrixWorld 2023 | 400 epochs | discrete | N/A | N/A |
| PettingZoo Simple Tag | 25 cycles | discrete | N/A | N/A |
| Aquarium 2024 | 3000 | discrete | 3000 steps | N/A |
| **Ours** | **1200** | **0.05s** | **60s** | **10s** |

### Key Findings

1. **Episode length relative to arena crossing time is the key metric**: Most papers use episodes that are 2-6x the arena crossing time. Our 60s episode is 6x our 10s crossing time, which is on the upper end but not unreasonable.

2. **Short episodes (10-50 steps) are common for simple environments**: PettingZoo uses only 25 cycles. The dog-sheep game converges in ~11 steps.

3. **Longer episodes (500-800 steps) are used for complex dynamics**: Papers with car-like or quadrotor dynamics use hundreds of steps because the dynamics are more complex.

4. **Very long episodes can hurt training**: With 1200 steps, there is a large temporal credit assignment problem. The agent has to figure out which of 1200 actions led to capture or escape. This can significantly slow learning.

### Implications for Our Project

60 seconds (1200 steps) is on the long side. Most comparable papers use 8-50 seconds of simulation time. Consider:
- Reducing to 500-600 steps (25-30 seconds) to improve credit assignment
- Using a time penalty (reward proportional to survival time) rather than relying solely on terminal reward
- The long episode may be why training is slow to converge

---

## 3. Capture Radius

### What Other Papers Use

| Paper / Environment | Capture Radius/Distance | Arena Size | Ratio (capture/arena) |
|---|---|---|---|
| Multi-UAV PE 2024 | 0.3m | 0.9m radius | 0.17 |
| Quadrotor PE 2025 | 0.5m | 12m x 12m | 0.042 |
| Bilgin & Karimpanal 2024 | 2 * agent_radius | 20m x 20m | varies |
| Dog-Sheep 2022 | 10 units (agent size) | R=200 | 0.05 |
| Multi-agent PE 2021 | 1.2 km (attack range) | 20km x 20km | 0.06 |
| PettingZoo Simple Tag | Contact/collision | ~2 unit space | ~0.1 |
| **Ours** | **0.5m** | **10m x 10m** | **0.05** |

### Key Findings

1. **Capture-to-arena ratio is typically 0.03-0.17**: Our 0.05 ratio is squarely in the middle of the range.

2. **0.5m capture radius is a reasonable choice**: The quadrotor PE paper uses exactly 0.5m in a 12m arena. Our setup is similar.

3. **Smaller capture radii make the game harder for the pursuer** and require more precise control, but also make the reward more sparse.

### Implications for Our Project

Our 0.5m capture radius in a 10m x 10m arena is well within the normal range. This is not a parameter that needs changing.

---

## 4. Reward Design

### What Other Papers Use

| Paper | Reward Type | Pursuer Reward | Evader Reward | Key Features |
|---|---|---|---|---|
| Bilgin & Karimpanal 2024 | **Zero-sum, dense** | +lambda_capture on capture, -(lambda_t + lambda_d*d) per step | Opposite | lambda_capture=1000, lambda_t=1, lambda_d=1 |
| Quadrotor PE 2025 | **Near-zero-sum, dense** | +0.5*(d_{t-1}-d_t) per step, +10 capture | +0.007 survival/step, +10 timeout | Both penalized for body-rate and out-of-bounds |
| Multi-UAV PE 2024 | **Dense team reward** | +2 capture, -0.1*d per step | N/A (scripted evader) | Collision=-10, smoothness reward |
| Dog-Sheep 2022 | **Sparse + time penalty** | N/A | +10 escape, -10 captured | Time-out penalty with log formula |
| Multi-agent PE 2021 | **Dense + terminal** | beta1*r_dist + beta2*r_final + beta3*r_safe | Opposite | 3-component weighted reward |
| MatrixWorld 2023 | **Dense zero-sum** | +10 capture, +0.1 neighbor proximity | -10 captured, -0.1 neighbor proximity | Movement cost -0.05/step |
| OpenAI Hide-and-Seek | **Pure sparse** | +1 if all hidden, -1 otherwise | Opposite | No reward shaping at all |
| PettingZoo Simple Tag | **Sparse collision** | +10 per collision | -10 per collision | No distance or time component |

### Key Findings

1. **Dense rewards dominate in PE**: Almost all successful PE RL implementations use distance-based shaping. Pure sparse rewards (OpenAI Hide-and-Seek style) require massive compute (480M games).

2. **The most common reward structure is**: `r = lambda_1 * delta_distance + lambda_2 * terminal_reward + lambda_3 * penalties`. The pursuer gets positive reward for closing distance and big terminal reward for capture. The evader gets the opposite.

3. **Time-based rewards are underused but important**: The quadrotor PE paper gives the evader +0.007 per survival step. This is a critical insight -- the evader needs per-step survival incentive, not just a terminal timeout reward. Otherwise, the evader has no gradient signal until the episode ends.

4. **Zero-sum vs independent rewards**: Pure zero-sum (r_evader = -r_pursuer) is the theoretical ideal but can cause training instability. Several papers use near-zero-sum with independent penalties (e.g., both agents penalized for boundary violations). The quadrotor PE paper is a good example -- mostly zero-sum structure but both agents independently penalized for erratic control.

5. **Capture reward magnitudes**: Range from +1 to +1000. The car-like PE paper uses lambda_capture=1000 with per-step costs of 1, giving a 1000:1 ratio. This is much higher than typical RL rewards.

6. **Movement cost / step penalty**: MatrixWorld uses -0.05 per step, encouraging efficient play. This acts as implicit time pressure.

### Implications for Our Project

Our reward design should include:
- **Dense distance-based shaping** for both agents (delta-distance per step)
- **Per-step survival bonus for evader** (not just terminal timeout reward)
- **Time pressure for pursuer** (per-step penalty encourages fast capture)
- **Large terminal rewards** (capture/escape) relative to per-step rewards (100:1 to 1000:1 ratio)
- **Independent penalties** for wall collisions, erratic control (not part of the zero-sum structure)

---

## 5. Curriculum Learning for PE

### What Other Papers Use

| Paper | Curriculum Type | What's Curricularized | Details |
|---|---|---|---|
| DualCL (arXiv 2312.12255) | **Dual curriculum** | Environment difficulty + opponent skill | Adaptive environment generator + opponent skill progression |
| Multi-UAV PE 2024 | **Adaptive environment** | Obstacle density, placement | p=0.7 local expansion, success thresholds 0.5-0.9 |
| Bilgin & Karimpanal 2024 | **Sensor curriculum** | FOV angle (2pi -> pi/2) and range (10m -> 7.5m) | Sensor capabilities decrease over training |
| USV PE (Qu et al. 2023) | **Adversarial-evolutionary** | Multiple random scenarios | Combined with curriculum learning for pursuit and escape models |
| Decentralized Multi-Agent PE (HAL 2022) | **Sweeping curriculum** | Number of pursuers | Start with few pursuers, increase |
| OpenAI Hide-and-Seek | **Autocurriculum** | Opponent skill (via self-play) | Self-play itself acts as natural curriculum |
| **Ours** | **Discrete levels** | L1-L4 (random -> self-play) | 4 fixed stages with advancement gates |

### Key Findings

1. **Curricularizing the opponent** (self-play as curriculum) is the most common and often sufficient. OpenAI proved that pure self-play with no explicit curriculum can work if the environment is rich enough.

2. **Curricularizing the environment** (obstacles, arena size, sensor range) is popular for complex setups. The sensor curriculum from Bilgin & Karimpanal is clever -- start with full observability and gradually reduce it.

3. **Curricularizing starting conditions** (starting distance) is surprisingly rare in the literature, though it seems like an obvious choice.

4. **Adaptive vs. fixed curricula**: Adaptive curricula that use success rate thresholds (like the Multi-UAV PE paper with 0.5-0.9 range) are more robust than fixed-step curricula.

5. **The most successful approach for PE specifically is adversarial co-evolution**: MatrixWorld uses 3 frameworks (specialist-vs-specialist, generalist-vs-specialist, generalist-vs-generalist) with historical opponent pools to prevent catastrophic forgetting.

### Implications for Our Project

Our 4-level discrete curriculum (L1-L4) is reasonable in structure but may be too rigid. Consider:
- Using adaptive thresholds based on NE gap rather than capture rate
- Curricularizing sensor capabilities (start with full observability)
- Curricularizing starting distance (start close, increase)
- The most robust approach is maintaining an opponent pool (which we already have) combined with adaptive advancement

---

## 6. Starting Conditions

### What Other Papers Use

| Paper | Starting Conditions |
|---|---|
| Bilgin & Karimpanal 2024 | Uniform random within 20x20m arena, simultaneous start |
| Multi-UAV PE 2024 | Varied scenarios via adaptive environment generator |
| Quadrotor PE 2025 | Not specified (likely random within 12x12x6m) |
| Multi-agent PE 2021 | Initial velocity = 0 for both agents |
| MatrixWorld 2023 | Random placement on grid |
| PettingZoo Simple Tag | Random positions |
| OpenAI Hide-and-Seek | Random room layout, random starting positions |

### Key Findings

1. **Uniform random starting positions are the standard**: Almost every paper uses random initial placement.

2. **Starting velocity = 0 is common**: Most papers start agents from rest.

3. **Minimum starting distance is rarely enforced**: Some papers just let random placement handle it, accepting that some episodes will start with immediate capture.

4. **Scenario diversity matters more than any single starting condition**: The adaptive environment generator approach (Multi-UAV PE) that randomizes obstacle placement, agent positions, and difficulty level produces the most robust policies.

### Implications for Our Project

Our approach (random starting positions) is standard. We could improve by:
- Ensuring a minimum starting distance (e.g., at least 3-5m apart)
- Randomizing initial headings
- Varying starting scenarios as part of curriculum

---

## 7. Action Space

### What Other Papers Use

| Paper | Action Space | Dimensions | Details |
|---|---|---|---|
| Bilgin & Karimpanal 2024 | **Continuous** | 2D (steering velocity, longitudinal acceleration) | Normalized to [-1, 1] |
| Multi-UAV PE 2024 | **Continuous** | 4D (thrust + body rates) | CTBR: F in [0,1], omega in [-pi, pi] |
| Quadrotor PE 2025 | **Continuous** | 4D (vx, vy, vz, omega_z) or (omega_x, omega_y, omega_z, T/m) | Two variants tested |
| Dog-Sheep 2022 | **Both tested** | Discrete (2: left/right), Continuous (DQN vs DDPG) | DDPG with continuous outperformed DQN |
| Multi-agent PE 2021 | **Continuous** | 2D via actor network (15 -> 64 -> 64 -> 2) | DDPG-based |
| MatrixWorld 2023 | **Discrete** | 5 (up/down/left/right/stay) | Grid-based movement |
| PettingZoo Simple Tag | **Discrete default** | 5 (no-op, 4 directions) | Continuous option available |
| **Ours** | **Continuous** | **2D (v, omega)** | **Linear and angular velocity** |

### Key Findings

1. **Continuous action spaces dominate in recent PE literature**: Nearly all 2023-2025 papers use continuous actions with PPO, DDPG, TD3, or SAC.

2. **Discrete actions are only used in grid worlds**: For continuous physical environments, discrete actions are essentially never used.

3. **(v, omega) is a standard choice for differential-drive robots**: Our action space is well-aligned with the literature. The alternative is (v_left, v_right) wheel velocities, but (v, omega) is more common.

4. **Action normalization to [-1, 1] is standard**: This helps training stability.

5. **DDPG outperforms DQN in PE**: The dog-sheep paper explicitly tested both and found continuous (DDPG) superior.

### Implications for Our Project

Our action space design is standard and well-justified. No changes needed.

---

## 8. The "Trivial Equilibrium" Problem

### Background

In a bounded domain with equal speeds, the pursuit-evasion game has a well-known theoretical property: the pursuer can always eventually catch the evader (in most bounded convex domains), though Besicovitch (1952) proved that in a disk the evader can delay capture indefinitely with optimal play.

In practice with RL, the game becomes "trivially solvable" for the pursuer in bounded domains with equal speeds because:
1. The evader gets trapped in corners
2. The pursuer can simply close distance systematically
3. There is no "escape zone" for the evader to exploit

### How Other Papers Handle This

| Approach | Papers | How It Works |
|---|---|---|
| **Evader speed advantage** | Multi-UAV PE (1.3x), Military PE (1.3x), PettingZoo (faster prey) | Gives evader inherent advantage to compensate for bounded domain |
| **Obstacles** | USV PE (Qu 2023), Multi-UAV PE | Obstacles create cover and escape routes for the evader |
| **Partial observability** | Bilgin 2024 (limited FOV), ViPER (CoRL 2024) | Evader can exploit blind spots |
| **Reach-avoid formulation** | Various | Evader has a target to reach, not just survival; creates richer strategy space |
| **Multi-pursuer cooperation** | MatrixWorld, many multi-agent papers | Difficulty comes from coordination, not speed |
| **Environment complexity** | OpenAI Hide-and-Seek | Rich environment with objects to manipulate |
| **Maneuverability asymmetry** | Bilgin 2024 (car-like pursuer vs point-mass evader) | Different dynamics rather than different speeds |

### Critical Insight: Why Equal Speed 1v1 in Bounded Domain is Problematic

In a 1v1 PE game with equal speeds and bounded domain (no obstacles, full observability):

1. **The game is theoretically determined**: The pursuer wins in finite time for most initial configurations in convex bounded domains. There is no "interesting" equilibrium to discover.

2. **Self-play has nothing to converge to**: The Nash equilibrium is simply "pursuer catches evader." There is no balance point where both agents play optimally and the outcome is uncertain. This is fundamentally different from games like Go or StarCraft where balanced play exists.

3. **The evader has no viable long-term strategy**: The best the evader can do is delay capture, but it cannot prevent it. This means the "optimal evader" is one that maximizes time-to-capture, which requires very precise wall-exploitation and corner-avoidance -- a hard credit assignment problem.

4. **This explains our self-play collapse**: Our L2 collapse and oscillation problems may be symptoms of trying to find a balanced equilibrium in a game where no balance exists. The evader can never "win" -- it can only lose slowly.

### What Makes PE Games Interesting (and Trainable)

The literature converges on several ways to create non-trivial PE games:

1. **Speed advantage for evader (most common)**: 10-30% faster evader creates a genuine "escape zone" where the evader can get away if it plays well. This creates a true mixed-strategy Nash equilibrium.

2. **Obstacles (second most common)**: Even with equal speeds, obstacles create escape opportunities through line-of-sight blocking, corner cutting, and creating chokepoints.

3. **Partial observability (increasingly popular)**: Limited FOV means the evader can hide, doubling back when out of sight. This creates information asymmetry that compensates for speed equality.

4. **Reach-avoid (structural change)**: Giving the evader a goal location to reach transforms the game from "delay as long as possible" (which is always a losing proposition in bounded domains) to "reach the goal before being caught" (which has genuine mixed strategies).

### Implications for Our Project

**Our current setup (equal speed, bounded, no obstacles, full observability) is the hardest possible configuration for achieving stable self-play**. We are trying to train a game where the equilibrium is trivial (pursuer always wins), which means:
- The evader has no real incentive to learn (it always loses)
- Self-play will always collapse to pursuer dominance
- Curriculum levels don't help because the fundamental game is unbalanced

**Recommended changes (in order of impact)**:
1. **Give evader 10-20% speed advantage** (e.g., evader=1.1 or 1.2, pursuer=1.0). This is the single most impactful change.
2. **Add 2-4 obstacles** in the arena. This creates tactical complexity even with equal speeds.
3. **Implement partial observability** (limited FOV for both agents). We already have sensor models.
4. **Consider reach-avoid formulation**: Give evader a target zone to reach, making it an offensive game rather than pure survival.

---

## Summary: Comparison With Our Current Setup

| Parameter | Our Setup | Literature Typical | Assessment |
|---|---|---|---|
| Arena size | 10m x 10m | 12-20m (continuous) | Slightly small, but OK |
| Speed ratio | 1.0 (equal) | 1.1-1.3 (evader faster) | **Problematic -- creates trivial game** |
| Episode length | 1200 steps / 60s | 500-800 steps / 10-50s | Long, could reduce |
| Capture radius | 0.5m | 0.3-0.5m | Fine |
| Reward design | Distance + terminal | Distance + terminal + survival | Missing survival bonus for evader |
| Action space | (v, omega) continuous | Continuous 2D | Standard, good |
| Starting conditions | Random | Random | Standard, good |
| Obstacles | None (in base) | Common | **Should add** |
| Observability | Full | Often limited | **Should limit** |
| Curriculum | Discrete 4-level | Adaptive, opponent-based | OK but could be more adaptive |

### Top 3 Changes Recommended

1. **Give evader a speed advantage (1.0 pursuer / 1.15-1.2 evader)**: This is the single most important change. It transforms the game from "pursuer always wins" to "balanced game with genuine mixed strategies."

2. **Add per-step survival reward for evader**: Instead of only terminal timeout reward, give evader a small positive reward each step it survives. This provides continuous gradient signal.

3. **Reduce episode length to 500-600 steps (25-30 seconds)**: This improves credit assignment and speeds up training without fundamentally changing the game.

---

## References

- [Bilgin & Karimpanal 2024 - Pursuit-Evasion with Car-like Robots and Sensor Constraints](https://arxiv.org/html/2405.05372v1)
- [Multi-UAV PE (2024)](https://arxiv.org/html/2409.15866v1)
- [Quadrotor PE (2025)](https://arxiv.org/html/2506.02849)
- [Dog-Sheep Game (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8980781/)
- [Multi-agent Military PE (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8625563/)
- [MatrixWorld (2023)](https://arxiv.org/html/2307.14854v2)
- [OpenAI Hide-and-Seek (2019)](https://arxiv.org/pdf/1909.07528)
- [PettingZoo Simple Tag](https://pettingzoo.farama.org/environments/mpe/simple_tag/)
- [Aquarium Framework (2024)](https://arxiv.org/html/2401.07056v1)
- [DualCL - Dual Curriculum Learning for Multi-UAV PE (2023)](https://arxiv.org/abs/2312.12255)
- [A Dynamics Perspective of PE Games (2021)](https://arxiv.org/pdf/2104.01445)
- [A Review of RL Approaches for PE Games (2025)](https://www.sciencedirect.com/science/article/pii/S1000936125005461)
- [Pursuit-Evasion Wikipedia - Besicovitch result](https://en.wikipedia.org/wiki/Pursuit%E2%80%93evasion)
- [USV PE with DRL (Qu et al. 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0029801823004006)
- [ViPER: Visibility-based PE via RL (CoRL 2024)](https://github.com/marmotlab/ViPER)
