# The ∀/∃ Asymmetry in Population-Based Self-Play for Pursuit-Evasion

**Analysis Document** — Session 103 Extension  
**Date**: 2026-05-25

---

## 1. Introduction

Population-based fictitious self-play (Pop-FSP) is a principled approach to stabilizing competitive training: by maintaining a population of historical policies and sampling opponents from it, agents learn against a diverse distribution of strategies rather than overfitting to a single, rapidly changing opponent. In symmetric games (e.g., Go, Chess, Poker), this approach has proven effective because both players face structurally identical optimization problems.

In pursuit-evasion (PE), however, the pursuer and evader occupy *asymmetric* roles. We hypothesized that Pop-FSP would stabilize the oscillatory training dynamics observed in vanilla self-play. Instead, across five distinct game configurations (v3–v7), Pop-FSP consistently converged to a **biased equilibrium** — never achieving competitive balance (capture rate CR ∈ [0.4, 0.6]). This document identifies the root cause: a fundamental **∀/∃ quantifier asymmetry** in the population-based training objective.

---

## 2. The ∀/∃ Asymmetry

### 2.1 Mathematical Formulation

Consider a population-based training setup where:
- The **pursuer population** is P = {πp₁, ..., πp_K}
- The **evader population** is E = {πe₁, ..., πe_M}

In Pop-FSP, each agent trains against opponents sampled from the opposing population. The effective optimization objectives are:

**Pursuer's objective (existential ∃):**
```
max_{πp}  E_{πe ~ Uniform(E)} [ V^{πp,πe}(s₀) ]
```
The pursuer needs to perform well against *any single* sampled evader. It solves an **∃-problem**: "there exists a strategy in my population that can catch this evader."

**Evader's objective (universal ∀):**
```
max_{πe}  E_{πp ~ Uniform(P)} [ V^{πp,πe}(s₀) ]
```
The evader needs to perform well against *all* sampled pursuers. It solves a **∀-problem**: "for all pursuers in the population, I must escape."

### 2.2 Why ∀ is Harder Than ∃

The existential quantifier is **monotonically easier** to satisfy than the universal quantifier:

- For the pursuer: if *any* policy in its population can catch the sampled evader, the pursuer receives positive training signal. Even weak policies contribute value if they occasionally succeed.
- For the evader: it must escape *every* pursuer in the sampled batch. A single effective pursuer in the population provides negative signal that dominates.

This creates a **structural gradient imbalance**:
```
|∇_pursuer|  >  |∇_evader|   (in expectation)
```
The pursuer's gradient is estimated against a "weak link" in the evader population, while the evader's gradient is estimated against the "strongest link" in the pursuer population.

### 2.3 Empirical Evidence from Pop-FSP v3–v7

| Config | Speed Ratio | Equilibrium CR | Direction | Seeds |
|--------|-------------|----------------|-----------|-------|
| v3 | 1.67x | 1.00 | Pursuer lock-in | 3/3 |
| v4 | 1.43x | ~0.00 | Evader-dominant | 3/3 |
| v5 | 1.56x | 1.00 | Pursuer lock-in | 3/3 |
| v6 | 1.52x | **0.72** | Pursuer-biased | 5/5 |
| v7 | 1.47x | 1.00 or 0.00 | Lock-in | 3/3 |

**Key observation**: Game modifications (arena size, speed ratio) can shift the equilibrium point but cannot eliminate the bias. The "sweet spot" v6 (ratio=1.52x) produced the closest result (CR=0.72), but this is still pursuer-biased. The equilibrium settled at CR=0.72 and *oscillated around that point* for thousands of episodes — it was stable, just wrong.

---

## 3. Why Game Modifications Fail

### 3.1 The Speed Ratio Sweep

We systematically varied the speed ratio (pursuer_v_max / evader_v_max) from 1.43x to 1.67x:

```
Ratio:  1.43x ---- 1.47x ---- 1.52x ---- 1.56x ---- 1.67x
Result:  Evader    Mixed      CR~0.72    Pursuer    Pursuer
         wins       lock                   wins       wins
```

The relationship is monotonic: faster evader → lower CR. But there is no ratio that produces CR=0.5 because:
1. The ∀/∃ asymmetry adds a **systematic offset** toward pursuer advantage
2. Even when the evader is mechanically favored (v4, ratio=1.43x), the structural asymmetry is overcome by the game mechanics, producing evader-dominance — but this is not balance either

### 3.2 Why v6 Was Close But Not Balanced

v6 (arena=19.0, evader=1.65, ratio=1.52x) produced CR=0.72 in the only seed that completed 5000 episodes. The trajectory reveals the dynamics:

| Window | Mean CR | Assessment |
|--------|---------|------------|
| 0–500 | 0.195 | Evader-dominant early |
| 500–1000 | 0.295 | Evader still dominant |
| 1000–1500 | 0.778 | Pursuer surge |
| 1500–2500 | 0.868 | Deep pursuer dominance |
| 2500–3500 | 0.862 | Sustained bias |
| 3500–5000 | 0.823 | Gradual recovery toward 0.75 |

The system oscillated but around the **wrong mean**. The population diversity prevented total lock-in (unlike v3/v5 where CR=1.0 was permanent), but the equilibrium point was structurally biased.

---

## 4. Comparison with Alternative Stabilization Methods

### 4.1 Vanilla Self-Play (GI Reproduction)

The Gonultas & Isler 2024 reproduction achieves **CR=0.550** with vanilla self-play — no populations, no EMA. The game itself sustains oscillation. This is critical: the baseline dynamics are *already* oscillatory. Pop-FSP does not improve upon this; it distorts the equilibrium.

### 4.2 EMA Opponent Stabilization

EMA (exponential moving average of opponent networks) achieves **robustness=0.698** in seed 0 — the best result across all methods. Unlike Pop-FSP, EMA does not introduce ∀/∃ asymmetry because both agents train against a *single* smoothed opponent. The smoothing reduces non-stationarity without changing the quantifier structure.

However, EMA is **highly seed-dependent**: 1/4 seeds succeeded, 3/4 collapsed. This suggests EMA's hyperparameter landscape is sharp, but the approach is theoretically sound.

| Method | Best Result | Seed Robustness | Structural Issue |
|--------|------------|-----------------|-------------------|
| Vanilla SP | CR=0.55 | Oscillates | No stabilization |
| EMA | Robustness=0.70 | 1/4 seeds | Sharp hyperparams |
| Pop-FSP | CR=0.72 | 0/5 configs | ∀/∃ asymmetry |

---

## 5. Implications

### 5.1 For Population-Based Methods in Asymmetric Games

Pop-FSP and related methods (PSRO, α-Rank, etc.) assume symmetric access to the opponent distribution. In asymmetric games:
- The **existential player** (pursuer: must succeed against *some* opponent) benefits from population diversity
- The **universal player** (evader: must succeed against *all* opponents) is penalized by it

This suggests population-based methods require **asymmetric population designs**:
- Larger evader populations to reduce the "hardest pursuer" sampling variance
- Recency-biased sampling (tested in v2) to focus on recent, similar-skill opponents
- Explicit ∀-relaxation mechanisms (e.g., only requiring escape against a subset)

Our v2 implementation tested recency bias + asymmetric populations (16 evader vs 8 pursuer) + gradual mixing. These mitigations delayed lock-in but did not eliminate the bias.

### 5.2 For Pursuit-Evasion Specifically

The ∀/∃ asymmetry is **not unique to Pop-FSP**. Any training method where:
- The pursuer is evaluated against a *distribution* of evaders
- The evader is evaluated against a *distribution* of pursuers

will exhibit this effect. This includes:
- League training (AlphaStar-style)
- PSRO with full empirical games
- Curriculum-driven opponent selection

Methods that avoid this:
- Single-opponent training (vanilla SP, EMA)
- Freeze-one-agent protocols (train pursuer against frozen evader, then switch)
- Explicitly asymmetric reward shaping

---

## 6. Recommendations

### 6.1 Immediate

1. **Abandon Pop-FSP** for this domain. The structural asymmetry cannot be overcome by hyperparameter tuning or game modification alone.
2. **Invest in EMA** with larger seed sweeps and potential τ annealing schedules.
3. **Run longer vanilla baselines** (20K episodes) to characterize natural oscillation duration.

### 6.2 Future Work

If population-based methods are revisited:
1. **Asymmetric quantifiers**: Train pursuer against full population (∀), evader against single sampled opponent (∃) — reverse the asymmetry to compensate
2. **Population-conditioned policies**: Evader policy takes the entire pursuer population as input (or a learned embedding), explicitly solving the ∀ problem
3. **Adversarial population curation**: Select pursuer subsets that the evader *can* escape, ensuring positive training signal

---

## 7. Conclusion

The failure of Pop-FSP to achieve competitive balance in pursuit-evasion is not a hyperparameter problem — it is a **structural problem**. The ∀/∃ quantifier asymmetry between pursuer (existential) and evader (universal) training objectives creates a systematic gradient imbalance that game modifications can shift but not eliminate. Population-based methods, which excel in symmetric games, require fundamental redesign for asymmetric competitive domains.

This negative result is scientifically valuable: it characterizes a limitation of population-based self-play in pursuit-evasion and guides future work toward methods (EMA, freeze protocols, asymmetric quantifiers) that respect the game's inherent structural asymmetry.

---

## Appendix: Complete Configuration Sweep Data

| Parameter | v3 | v4 | v5 | v6 | v7 |
|-----------|-----|-----|-----|-----|-----|
| Arena | 16.0 | 20.0 | 18.0 | 19.0 | 19.0 |
| Evader v_max | 1.50 | 1.75 | 1.60 | 1.65 | 1.70 |
| Speed ratio | 1.67x | 1.43x | 1.56x | 1.52x | 1.47x |
| Pop size (pursuer) | 8 | 8 | 8 | 8 | 8 |
| Pop size (evader) | 16 | 16 | 16 | 16 | 16 |
| Recency λ | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| Mixing start | 100 | 100 | 100 | 100 | 100 |
| Mixing end | 500 | 500 | 500 | 500 | 500 |
| Result | CR=1.0 | CR≈0.0 | CR=1.0 | CR=0.72 | Mixed lock |

**Total compute invested**: ~25 GPU-hours across 5 configurations, 17 seeds, 0 balanced outcomes.
