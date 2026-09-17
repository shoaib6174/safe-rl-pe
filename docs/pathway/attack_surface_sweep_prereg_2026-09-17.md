# S9 — Is the stall attack a property of the shield? Evaluation-only sweeps (2026-09-17, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any cell of this sweep exists.**
Governing: **D112** (the shield cuts capture; mechanism = lost free wall contact), **D115** (the structured
attacker pins a shielded pursuer), **D116** (family training removes the pin by disengagement), **D117** (paper
plan, priority 2: show the attack surface is a property of hard shields, not of one arena).

## 1. Why this round exists

Every attack-surface number so far comes from one arena (10 m, two obstacles) and one shield setting
(δ = 0, class-K gain γ = 0.2). The paper's thesis — *a hard shield moves the adversary's leverage from collisions
to stalls* — predicts that the stall attack gets **stronger as the shield gets more conservative** and **weaker
as the geometry gets emptier**. If either prediction fails, the thesis is an artefact of our configuration and
must be narrowed. No training; only frozen policies and fresh evaluation streams.

**H-GEO-δ:** for each pursuer, the deadlock-episode rate against the structured attacker is non-decreasing in the
shield margin δ (Spearman ρ over the five δ levels > 0, and the top level exceeds δ = 0 by a paired test).
**H-GEO-n:** for each pursuer, the deadlock rate against the structured attacker is lowest in the empty arena
(n = 0) and non-decreasing in the obstacle count.
**H-GEO-γ:** a smaller class-K gain (more conservative filter) does not decrease the deadlock rate.
**H-SAFE:** 0 pursuer contact steps in every cell (δ ≥ 0, any γ, any n).
These are pre-stated predictions for a descriptive sweep; each is reported as CONFIRMED / NOT CONFIRMED per pursuer.

## 2. Cells

- **Pursuers (frozen, shielded, `shield_obs` as trained):** `RC19` (huntable, S5), `RCM28` (disengaged, S7),
  `STR14` (trained unshielded, S3; `shield_obs=False`), and the **S8 artefact** if H4 passes (added by amendment
  when it exists; the sweep does not wait for it).
- **Opponents:** `ATT25` (= `CTL-D225`, structured attacker trained vs RC19), `ATK2-31` (structured attacker
  trained vs RCM28), `ObstacleHuggingEvader` (`HUG7`), `GreedyEvader`.
- **Sweep δ (margin, m):** {0, 0.05, 0.10, 0.20, 0.30}, γ = 0.2, n = 2.
- **Sweep n (obstacles):** {0, 1, 2, 3, 4}, δ = 0, γ = 0.2. Policies were trained at n = 2; the shift is stated,
  the question is the *attack's* dependence on geometry, not the policies' generalisation.
- **Sweep γ (class-K gain):** {0.1, 0.2, 0.4}, δ = 0, n = 2.
- Stream **1 240 000+**, N = **300** per cell, seed-paired across every cell of a sweep (same seeds ⇒ same initial
  poses and, for equal n, same obstacles). 3 pursuers × 4 opponents × 11 distinct settings = 132 cells.
- Readouts per cell: pursuer deadlock-episode rate, stall-step rate, intervention rate, capture rate, escape rate,
  violations, min-clearance floor; Wilson CIs.

## 3. Tests (fixed now)

- Monotonicity: Spearman ρ between the level index and the per-cell deadlock rate (5 points; 3 for γ), sign only.
- Top-vs-bottom: `paired_indicator_test(bottom_ind, top_ind)` (the top level deadlocks MORE, p < 0.05).
- H-GEO-δ(pursuer) CONFIRMED iff ρ > 0 and top-vs-bottom passes against **both** structured attackers.
- H-GEO-n(pursuer) CONFIRMED iff the n = 0 cell has the lowest rate and ρ ≥ 0 against both structured attackers.
- H-GEO-γ(pursuer) CONFIRMED iff the γ = 0.1 rate ≥ the γ = 0.4 rate against both structured attackers.
- Overall: each hypothesis CONFIRMED iff confirmed for ≥ 2 of the 3 pursuers; reported per pursuer regardless.
- Scripted opponents (HUG7, Greedy) are reported as the non-adversarial baseline of every sweep, no verdict.

## 4. What each outcome fixes

| result | direction |
|---|---|
| δ, n and γ all CONFIRMED | The attack surface scales with shield conservativeness and geometry: a property of hard filters. Figure 1 of the paper. |
| δ CONFIRMED, n NOT | The stall is about the filter, not the obstacles (walls suffice): the thesis narrows to "any bounded arena". |
| δ NOT CONFIRMED | The attack does not grow with conservativeness; the thesis is narrowed to the δ = 0 filter we studied, and the paper says so. |
| H-SAFE fails at any δ ≥ 0 or γ | Stop; the shield's guarantee is under question before any sweep claim. |

**Stop rule.** Cells are never re-run; no level is added or removed after any number is seen. The S8 artefact
is the only permitted addition (a fourth pursuer, same streams), by amendment.

## 5. Artefacts

`results/derisk/s9_cells.json` (one record per cell, written atomically per cell so a crash keeps completed
cells), `s9_round.{json,png}` (three panels: deadlock vs δ, vs n, vs γ; one line per pursuer × structured attacker,
scripted baselines dashed). Runner `scripts/derisk_s9_sweep.py` (`--cells [names] | --summarize`); cells run in
parallel worker processes (≤ 6, `OMP_NUM_THREADS=2`) alongside S8's arms.
