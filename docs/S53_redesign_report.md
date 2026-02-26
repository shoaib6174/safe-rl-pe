# Self-Play Redesign Report — Session 53

## Executive Summary

After 20+ training runs (H through Z2), 52 sessions, and exhaustive analysis of both the code and the literature, the diagnosis is clear: **we have three interacting failures — a game with no meaningful equilibrium, a reward that provides no useful signal, and a self-play protocol that amplifies every instability.** Fixing any one in isolation has not worked and cannot work. This report proposes a coordinated redesign across all three.

---

## Part 1: Diagnosis — Three Interacting Failures

### Failure 1: The Game Has No Meaningful Equilibrium at Level 1

**Evidence:**
- At equal speeds (1.0/1.0) in a bounded 20×20 arena with no obstacles, classical pursuit-evasion theory says the pursuer can always corner and capture the evader (Besicovitch 1952).
- Our 60-second timeout is the evader's only hope — it cannot permanently escape, only delay.
- The Nash Equilibrium capture rate at Level 1 is estimated at 55–75%. This is barely below our advancement threshold of 0.70, meaning any slight pursuer advantage pushes past the gate.
- Run V's oscillation (0.48 → 1.00 → 0.26 → 0.03 → 0.95) confirms the game hovers near a tipping point where small policy changes swing the outcome dramatically.

**Literature comparison:**
- **No 1v1 PE paper uses equal speeds.** Every implementation surveyed gives the evader a 10–30% speed advantage (Multi-UAV PE: 1.3×, PettingZoo Simple Tag: explicitly faster prey, Dog-Sheep game: speed ratio up to π+1).
- Equal speed is used only in multi-pursuer setups (where the challenge is coordination) or environments with objects to manipulate (OpenAI Hide-and-Seek).
- Without a speed advantage or obstacles, the evader has no viable strategy beyond "run and hope for timeout."

### Failure 2: The Reward Provides No Useful Signal

**The terminal-to-shaping ratio is catastrophic:**

| Scenario | Total distance shaping | Terminal reward | Ratio |
|----------|----------------------|-----------------|-------|
| L1 capture from 3m | 0.088 | 100.0 | 1:1,136 |
| L1 capture from 5m | 0.159 | 100.0 | 1:629 |
| Timeout at equal speed | ~0.000 | 100.0 | ~1:∞ |

The distance shaping contributes less than 0.2% of total return. PPO effectively sees only a sparse binary outcome.

**The equal-speed dead zone:** When both agents play optimally (pursuer chases, evader flees at equal speed), the distance change per step is approximately zero. Both agents receive zero per-step shaping. The dense signal vanishes precisely when it should be most informative.

**The evader has no independent signal at L1:** Pure zero-sum with `survival_bonus=0` means the evader's entire learning signal is the sparse terminal reward. There is no gradient for "good fleeing that delayed capture."

**The L2→L3 reward discontinuity:** Visibility reward activates only when obstacles exist. Going from zero-sum (~0 per-step) to visibility mode (±1.0 per-step) is a 12× scale change plus a structural change. The value function trained on L1/L2 is completely miscalibrated for L3+.

### Failure 3: The Self-Play Protocol Amplifies Instability

**Our 100K–500K step frozen phases are 10–50× longer than any successful implementation:**

| System | Opponent swap frequency | Historical sampling |
|--------|------------------------|-------------------|
| Unity ML-Agents (default) | 10,000 steps | 50% current, 50% pool of 10 |
| OpenAI Five | Continuous | 80% current, 20% past selves |
| AlphaStar | Continuous | League of 900+ agents |
| **Our system** | **100,000–500,000 steps** | **0% historical** |

**Bilateral rollback is counterproductive:** In Z1 S4 (evader training), rollbacks progressively weakened the evader: step 280K → 240K → 200K. The evader was pushed to increasingly early (weaker) versions, creating a death spiral.

**The advancement criterion requires domination, not balance:** `capture_rate > 0.70` means the curriculum advances when the pursuer is winning — the opposite of balanced play. Run V advanced to Level 2 at S7 (SR_P=0.95) precisely because the pursuer dominated, then the evader immediately collapsed at L2.

---

## Part 2: The Redesign

The redesign addresses all three failures simultaneously. Each change is motivated by both our failure analysis and the literature.

### A. Game Setup Changes

#### A1. Evader Speed Advantage: v_evader = 1.15, v_pursuer = 1.0

**Rationale:** This is the single most impactful change. It transforms the game from "pursuer always eventually wins" to "genuine mixed-strategy equilibrium."

- At 1.15× speed, the evader can outrun the pursuer in open field but cannot trivially escape (the pursuer can cut corners, use walls strategically).
- The Nash Equilibrium becomes a genuine mixed outcome (~40–60% capture rate depending on level).
- Both agents have meaningful strategic depth: pursuer must predict and cut off, evader must choose between fleeing and hiding.
- Standard in the literature (Multi-UAV PE: 1.3×, PettingZoo, etc.).

**Why not 1.0/1.0 with obstacles?** We tried this (Runs V, X, Z1, Z2). At L1 (no obstacles), the game is degenerate regardless of self-play fixes. Obstacles help, but the curriculum must start somewhere, and that starting point needs a viable equilibrium.

**Why not a larger ratio (1.3×)?** Larger ratios make the pursuer's task too hard, potentially creating the reverse problem (evader always wins). 1.15× is conservative and can be tuned.

#### A2. Obstacles from Day 1: Always 2–3 obstacles present

**Rationale:** Obstacles are not a difficulty increase — they are a strategy enabler.

- With obstacles, the evader has cover (hiding strategy), the pursuer has terrain to navigate (prediction strategy).
- The visibility reward works from step 1, providing consistent reward semantics throughout training.
- No catastrophic distribution shift when "obstacles appear" because they were always there.
- Obstacle count can still be curriculated (2 → 3 → 4) if needed, but never zero.

**Implementation:** Set `min_obstacles=2` as a floor. Start with 2 medium-sized obstacles (radius 0.5–0.8m). Optionally increase to 3–4 as training progresses.

#### A3. Reduce Episode Length: 600 steps (30 seconds)

**Rationale:**
- Better credit assignment: PPO with γ=0.99 has effective horizon ~100 steps. At 1200 steps, early actions are disconnected from the terminal outcome.
- Faster iteration: 2× more episodes per wall-clock hour.
- Aligns with literature: most PE papers use 2–6× the arena crossing time (our arena crossing = 20m/1.0m/s = 20s; 30s = 1.5×).
- Forces more decisive play: agents can't "stall" as long.

### B. Reward Changes

#### B1. Rebalance Terminal vs. Shaping

**Option 1 (preferred): Reduce terminal rewards**
- `capture_bonus = 10.0` (was 100.0)
- `timeout_penalty = -10.0` (was -100.0)
- Keep `distance_scale = 1.0`
- Ratio becomes ~1:63 (L1 from 3m). Still terminal-heavy but the shaping is at least 1.6% of return.

**Option 2: Scale up shaping**
- `distance_scale = 100.0`
- Terminal stays at ±100
- Ratio becomes ~1:11. Better but may cause value function instability.

**Option 3 (most principled): Normalize rewards**
- Use reward normalization (SB3 supports this via `normalize_reward=True` in VecNormalize).
- Removes scale issues entirely. Both shaping and terminal are scaled to unit variance.

**Recommendation: Option 1 + survival bonus.** Reduce terminal to ±10, keep shaping at 1.0, add survival bonus. This gives a clean reward hierarchy: per-step shaping (small but dense) < survival bonus (monotonic) < terminal (decisive but infrequent).

#### B2. Always-On Survival Bonus: +0.02 per step

**Rationale:** Gives the evader a monotonic, independent learning signal at all levels.
- Per episode (600 steps): 0.02 × 600 = 12.0 (comparable to terminal ±10)
- The evader always has gradient: "staying alive is good, every extra step is worth +0.02"
- Combined with reduced terminal (±10), the survival signal is meaningful but not dominant.

#### B3. Visibility Reward from Day 1

**Rationale:** Since obstacles are always present (A2), visibility reward always has meaning.
- Consistent reward semantics throughout training.
- No value function recalibration needed at curriculum transitions.
- `visibility_weight = 1.0` gives ±1.0 per step. Over 600 steps, this sums to ±600, dominating everything else.

**Concern:** Visibility reward (±1.0/step) may be too large relative to terminal (±10) and survival (0.02/step). Consider reducing to `visibility_weight = 0.1` so per-step is ±0.1, summing to ±60 per episode — same order as terminal.

**Recommendation:** `visibility_weight = 0.1`, `survival_bonus = 0.02`, terminal = ±10. This gives:
- Shaping: ~0.2 total (directional)
- Survival: 12.0 total (monotonic)
- Visibility: up to ±60 total (strategic)
- Terminal: ±10 (decisive)

### C. Self-Play Changes

#### C1. Short Alternation: 4096–8192 steps per micro-phase

**Rationale:** Eliminates frozen-opponent over-exploitation (the root cause of L1 oscillation).
- 4096 steps = 1 PPO rollout. The opponent is only 1 rollout stale.
- Neither agent has time to memorize the opponent's specific patterns.
- Effectively simultaneous training within the existing SB3 framework.

**Implementation:**
- New parameter: `micro_phase_steps = 4096`
- Cache two environment sets (pursuer-training and evader-training) — don't recreate each micro-phase.
- After each micro-phase, sync the opponent weights: copy the just-trained agent's weights into the other env's opponent adapter.
- Run many micro-phases per "macro-phase" (e.g., 50 micro-phases = 200K total steps).

#### C2. Historical Opponent Pool: 50% current, 50% from pool

**Rationale:** Prevents over-fitting to any single opponent version.
- Save snapshots every 10K steps (every ~2 micro-phases).
- Pool size = 20 (covers ~200K steps of history).
- Each sub-environment samples its opponent: 50% use current weights, 50% sample uniformly from pool.
- Already have `OpponentPool` class from S43 — just needs integration with micro-phase training.

#### C3. Remove Rollback

**Rationale:** With micro-phases, oscillation amplitude is too small to trigger rollback. And rollback was counterproductive anyway (weakened the struggling agent). Remove the complexity.

#### C4. Optional: KL Regularization

**Rationale:** Prevents large policy shifts between micro-phases.
- Add KL penalty to the PPO loss: `L = L_ppo - β * KL(π_new || π_old)`
- SB3 supports `target_kl` parameter for early stopping. Set `target_kl = 0.01–0.03`.
- This is PPO's native regularization — just needs to be activated.

### D. Curriculum Changes

#### D1. Try No Curriculum First

**Rationale:** With obstacles from day 1, evader speed advantage, and fixed reward semantics, the full game may be directly trainable. Every successful self-play system (AlphaStar, OpenAI Five, Hide-and-Seek) trains at full difficulty from the start.

**Test configuration:** spawn distance 2–10m, 2 obstacles, no curriculum advancement. Just self-play until convergence.

#### D2. If Curriculum Is Needed: Distance-Only, Simple

- Start: 2–5m, 2 obstacles
- Increment: +1m to max_init_distance per advancement
- Target: 2–15m, 2–3 obstacles
- Advancement: NE-gap < 0.15 for 3 consecutive eval windows
- Never change obstacle count to zero. Optionally add 1 obstacle at max_d=10m.
- Evaluate every 50 micro-phases (200K steps), not every micro-phase.

---

## Part 3: Experiment Plan

### Phase 1: Core Redesign (2 runs)

| Run | Speed | Obstacles | Terminal | Survival | Vis Weight | Phases | Pool | Curriculum |
|-----|-------|-----------|----------|----------|------------|--------|------|------------|
| **RA1** | 1.15/1.0 | 2 always | ±10 | 0.02 | 0.1 | 4096 micro | 50/50 pool | None (2-10m) |
| **RA2** | 1.15/1.0 | 2 always | ±10 | 0.02 | 0.1 | 4096 micro | 50/50 pool | Distance (2-5m start) |

### Phase 2: Ablation (if Phase 1 succeeds)

| Run | What's different | Tests |
|-----|-----------------|-------|
| **RA3** | Equal speed (1.0/1.0) | Does speed advantage matter? |
| **RA4** | No opponent pool (current only) | Does pool matter? |
| **RA5** | Long phases (100K) with pool | Does phase length matter independently? |

### Phase 3: Scale Up

Once stable training is achieved, progressively add:
- More obstacles (3-4)
- Larger spawn distances (up to 15m)
- Partial observability (FOV sensor)
- Safety layer (DCBF)

### What to Do with Current Runs

| Run | Action | Reason |
|-----|--------|--------|
| V | **Keep running** | Furthest along (S10, L2). Useful as a baseline comparison. |
| X | **Kill** | Stuck at S4 with 5 rollbacks. T3 overhead, same broken protocol. |
| Z1 | **Kill** | Stuck at S7 with persistent rollbacks. NE-gap never triggers. |
| Z2 | **Kill** | Stuck at S4. Slowest run, same collapse pattern. |

---

## Part 4: Implementation Scope

### Code Changes Required

**Environment (`envs/`):**
- Add `evader_speed_advantage` parameter to `PursuitEvasionEnv.__init__`
- Ensure `min_obstacles` floor (never go to 0 obstacles)
- Reduce `max_steps` default to 600

**Rewards (`envs/rewards.py`):**
- Change `capture_bonus` and `timeout_penalty` defaults to ±10
- Always enable `survival_bonus` (default 0.02)
- Reduce `visibility_weight` default to 0.1
- Remove the `if obstacles:` guard on visibility reward (or handle gracefully)

**Self-play (`training/amsdrl.py`):**
- New `micro_phase_steps` parameter (default 4096)
- Refactor `_train_phase()` to cache environments across micro-phases
- Add opponent weight sync between micro-phases
- Integrate opponent pool with micro-phase sampling
- Add periodic evaluation (every N micro-phases, not every micro-phase)
- Remove rollback callbacks (optional: keep as disabled-by-default)

**Curriculum (`training/curriculum.py`):**
- Add `min_obstacles` parameter (floor for obstacle count)
- Simplify smooth curriculum to only vary distance
- Keep NE-gap advancement

**CLI (`scripts/train_amsdrl.py`):**
- Add `--evader_speed_advantage`, `--micro_phase_steps`, `--no_curriculum` flags
- Update defaults

### Estimated Effort
- Environment + reward changes: ~2 hours
- Self-play refactor (micro-phases + env caching): ~3–4 hours
- Testing + integration: ~1–2 hours
- Launch and monitor: ~1 hour
- **Total: ~7–9 hours across 1–2 sessions**

---

## Part 5: Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Evader speed advantage makes pursuer task too hard | Medium | Start at 1.15×, tune down to 1.10× if needed |
| Micro-phases cause training instability (too frequent switches) | Low | PPO's clipping naturally limits per-update change |
| No curriculum leads to slow initial learning | Medium | Fall back to distance curriculum (RA2) |
| Reward rebalancing causes value function instability | Low | Use reward normalization as fallback |
| Obstacles at spawn create collision issues | Low | Existing collision resolution handles this |

---

## References

### Self-Play Methods
1. Zhang et al. (2024) — "A Survey on Self-play Methods in RL" (arXiv:2408.01072)
2. Schafer (2020) — Competitive Gradient Descent (arXiv:1905.12103)
3. Heinrich & Silver (2016) — NFSP (arXiv:1603.01121)
4. McAleer et al. (2024) — SP-PSRO (arXiv:2207.06541)
5. Vinyals et al. (2019) — AlphaStar
6. OpenAI (2018) — OpenAI Five
7. Baker et al. (2020) — Emergent Tool Use / Hide-and-Seek (arXiv:1909.07528)
8. Unity ML-Agents — Self-Play Configuration

### PE Game Design
9. Multi-UAV PE (2024) — arXiv:2409.15866
10. Bilgin & Karimpanal (2024) — Car-like PE (arXiv:2405.05372)
11. Quadrotor PE (2025) — arXiv:2506.02849
12. Dog-Sheep Game (2022) — PMC8980781
13. RL Review for PE Games (2025) — Chinese J. Aeronautics
14. PettingZoo Simple Tag — Farama Foundation

### Convergence Theory
15. Shapley (1964) — Rate of convergence of fictitious play
16. Robinson (1951) — Iterative method for solving games
17. PG Convergence (ICLR 2025) — arXiv:2408.00751
18. KL-regularized Games (2025) — arXiv:2510.13060
