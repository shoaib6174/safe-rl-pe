# Pop-FSP Balance Analysis: Why Pursuer Dominates and How to Fix It

## 1. Root Cause Diagnosis

### The Structural Asymmetry

Pop-FSP converges to pursuer-dominant (CR=1.0) because of a fundamental asymmetry in the **training objective**:

| | Pursuer Training | Evader Training |
|---|---|---|
| **Task** | Catch ANY evader from population | Escape ALL pursuers from population |
| **Logic** | ∃ strategy that catches evader → win | ∀ strategies must be escaped → much harder |
| **Population effect** | Early evaders are weak → easy wins | Some pursuers corner effectively → impossible |

This is an **existence vs. universal quantifier** asymmetry. The pursuer only needs ONE effective capture strategy among the population; the evader must be robust to EVERY escape-threatening strategy.

### Why Uniform Sampling Makes It Worse

The current `PopulationManager.sample()` uses uniform random sampling. This means:

1. **Old/weaker opponents dilute training** — Early checkpoints (ep 250, 500) contain naive policies. The pursuer gets easy reward against these, reinforcing aggressive behavior.
2. **No curriculum** — Agents train against opponents of all difficulties equally, rather than focusing on the "edge of competence."
3. **Abrupt switch** — After warmup=500, training goes from 100% current opponent to 100% population. No gradual adaptation.

### Evidence from Data

- **Phase 1 (0-1500)**: CR oscillates, brief balance at ep ~800. This is vanilla-like behavior before population diversity dominates.
- **Phase 2 (1500+)**: CR locks to 1.0. Once the population fills (8 members by ep 2000), the evader is overwhelmed by diverse pursuer strategies.
- **Oscillation eliminated but at cost of balance** — The population prevents cycling by making the evader's task too hard.

---

## 2. What the Literature Does

### AlphaStar (DeepMind, 2019) — PFSP
AlphaStar solved this exact problem using **Priority-weighted FSP**:
- Track win rate of each population member against current agent
- Sample opponents proportionally: `P(h) ∝ exp(-win_rate(h) / σ)`
- **Key insight**: Train against opponents where win rate is INTERMEDIATE (30-70%), not 0% or 100%
- Opponents that are too easy (100% win) or too hard (0% win) provide little learning signal

### OpenAI Five (2019) — Similar PFSP
Same approach: prioritize opponents that are "just hard enough."

### Why This Would Help Us
With PFSP:
- **Evader** samples pursuers where escape rate is intermediate (not impossible, not trivial)
- **Pursuer** samples evaders where capture rate is intermediate
- Both agents train at their "edge of competence" → natural balance

---

## 3. Ranked Solutions

### Tier 1: Recency-Biased Sampling + Asymmetric Pops (Recommended)

**Recency-biased sampling** uses exponential decay weights:
```python
weights = exp(-λ * age)  # age = episodes since addition
```

**Why this works:**
- Recent opponents ≈ similar skill level → intermediate difficulty → most learning signal
- Old opponents ≈ "solved" → easy → little signal → sampled less
- Natural approximation of PFSP without expensive win-rate tracking

**Asymmetric populations** give evader more diversity:
- `pop_pursuer_size = 8` (current)
- `pop_evader_size = 16` (2x) — compensates for structural disadvantage

**Gradual mixing** replaces hard warmup:
- Ep 0-100: 100% current opponent
- Ep 100-500: linear ramp to 100% population
- Prevents early lock-in

| Aspect | Assessment |
|--------|------------|
| Implementation effort | Low (~50 lines) |
| Theoretical grounding | Medium (PFSP approximation) |
| Expected impact | High |
| Testability | Fast (single 5K run) |

---

### Tier 2: Full PFSP with Win-Rate Proxy

Track approximate win rates per population member:
- When member is added, record the capture rate at that episode
- Use capture rate as proxy for "difficulty"
- Sample via softmax over difficulty scores

| Aspect | Assessment |
|--------|------------|
| Implementation effort | Medium (~100 lines) |
| Theoretical grounding | High (direct AlphaStar adaptation) |
| Expected impact | Very High |
| Testability | Medium |

**Trade-off:** The capture rate at addition time is only a proxy for true win rate vs. current agent. Might need periodic re-evaluation.

---

### Tier 3: Environment Adjustments

Change game parameters to structurally favor evader:
- Larger arena (16 → 20 or 24)
- Faster evader (1.5 → 1.65, keeping pursuer at 2.5)
- Smaller capture radius (0.15 → 0.12)

| Aspect | Assessment |
|--------|------------|
| Implementation effort | Trivial (config only) |
| Theoretical grounding | Low (changes the game) |
| Expected impact | Uncertain |
| Testability | Fast |

**Risk:** Changes the game itself. The GI paper achieves balance with current params (CR=0.55), so the environment is not the root cause.

---

### Tier 4: Exploiter Agents (PSRO-style)

Train dedicated "exploiter" agents:
- Main pursuer + main evader (standard training)
- Exploiter pursuer: trained to beat current evader
- Exploiter evader: trained to beat current pursuer
- Add exploiters to population

| Aspect | Assessment |
|--------|------------|
| Implementation effort | High (new training loop) |
| Theoretical grounding | Very High |
| Expected impact | Very High |
| Testability | Slow |

**Risk:** Complex to implement. Overkill for current scope.

---

## 4. Specific Recommendation

**Implement Tier 1 first (recency-biased + asymmetric + gradual mixing).**

### Config changes:
```yaml
maddpg:
  use_pop_fsp: true
  pop_size: 8              # pursuer population
  pop_evader_size: 16      # NEW: larger evader population
  pop_add_freq: 250
  pop_warmup: 500          # can reduce since mixing is gradual
  
  # NEW: recency bias
  pop_recency_lambda: 0.15  # 0=uniform, higher=more recent bias
  
  # NEW: gradual mixing
  pop_mixing_start: 100     # episode to start mixing
  pop_mixing_end: 500       # episode to reach 100% population
```

### Expected behavior:
- Early training (0-100): vanilla self-play → agents bootstrap
- Mid training (100-500): gradual exposure to population diversity
- Late training (500+): full population with recency bias
- Evader has 2x population size → more escape strategies to learn from
- Recent opponents weighted higher → training at edge of competence

### Test plan:
1. Single seed, 5K episodes (~1 hour on niro-2)
2. If CR stays near 0.5-0.6 → success, run multi-seed
3. If CR still → 1.0 → increase λ or pop_evader_size
4. If CR → 0.0 → decrease λ or reduce pop_evader_size

---

## 5. Fallback Plan

If Tier 1 fails after 2-3 parameter sweeps:
1. Implement Tier 2 (full PFSP with win-rate proxy)
2. If still failing, consider Tier 3 (environment adjustments)
3. If paper deadline pressure, frame Pop-FSP as "stability method" and present EMA s0 as "balance baseline" with caveats about seed-dependency

---

*Analysis generated: 2026-05-24*
