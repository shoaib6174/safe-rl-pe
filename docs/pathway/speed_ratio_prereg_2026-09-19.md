# S13 — Does the stall attack depend on the evader's speed advantage? (2026-09-19, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any cell of this sweep exists.**
Governing: **D112** (S4 already swept the ratio for STR14 against a SCRIPTED greedy evader and found the
shielded deadlock rate flat: 0.480 / 0.540 / 0.502 at 1.0 / 1.2 / 1.4, with the shield's capture cost flat at
≈ 0.32), **D118** and **D121** (the surface is geometric and set by how deep a filter lets the pursuer into a
pocket), **D120**/**D122** (artefact FT-B-41), **D117** (paper plan).

## 1. Why this round exists

The obvious objection to the whole result is: *the pursuer only gets pinned because it cannot simply run the
evader down.* S4's three cells say otherwise for one old policy against one scripted opponent, and the regime
that would settle it — a pursuer FASTER than its evader — has never been run. This round measures the stall
surface across the speed ratio for every current defender and, crucially, against the LEARNED attackers.

**H-SPD-1 (survival):** at **every** ratio, including ratios where the pursuer is faster, the structured
attackers produce a higher pursuer deadlock-episode rate than the strongest scripted probe at that same ratio
(paired, p < 0.05, both attackers). CONFIRMED for a pursuer iff it holds at all four ratios; overall iff it
holds for ≥ 3 of the 4 pursuers.
**H-SPD-2 (no speed dependence):** for each pursuer and each structured attacker, the paired difference in
deadlock rate between ratio **0.8** and ratio **1.2** lies within an equivalence margin of **±0.10** (the 95 %
Wald interval on the discordant pairs, `(b − c)/n`, contained in [−0.10, +0.10]). CONFIRMED overall iff it holds
for ≥ 3 of the 4 pursuers against both attackers.
**H-SPD-3 (resolution, reported, no verdict):** capture rate against `GreedyEvader` rises as the pursuer gets
relatively faster. If capture at ratio 0.8 exceeds 0.95 for every pursuer the metric has saturated and the
0.8 column is reported as a ceiling.
**H-SAFE:** 0 pursuer contact steps in every cell.

## 2. The transfer caveat, pre-registered

Every policy in this sweep — defenders and the two learned attackers — was trained at ratio **1.2**. Every cell
at another ratio is therefore out-of-distribution transfer, and a low deadlock rate at 0.8 could mean either
"the attack needs a speed advantage" or "this attacker fails to transfer". The design controls for that: the
comparison in H-SPD-1 is always against the **scripted** probes measured at the **same** ratio, and those probes
(`GreedyEvader`, `ObstacleHuggingEvader`) are speed-agnostic by construction — they recompute their commands
from the geometry and the config's speed cap at every step. A structured attacker that beats a scripted probe
at ratio 0.8 has transferred. **No cell of this round licenses a claim about what an attacker TRAINED at
another ratio could do**; that is named as future work and nothing here is reported as if it were measured.

## 3. Cells

- Pursuers (frozen, shielded with the CBF filter at δ = 0, γ = 0.2, `shield_obs` as trained): **RC19** (S5),
  **RCM28** (S7), **FT-B-41** (S10 artefact), **STR14** (S3, `shield_obs=False`, continuity with S4).
- Opponents: **ATT25** (= `CTL-D225`, trained against RC19), **ATK4-45** (trained against FT-B-41), **HUG7**,
  **GREEDY**.
- Ratios `evader_speed_ratio ∈ {0.8, 1.0, 1.2, 1.4}` (0.8 = the pursuer is 25 % faster). `n_obstacles = 2`.
- 4 × 4 × 4 = **64 cells**, stream **1 360 000+**, N = **300**, seed-paired within a ratio (identical layouts
  and initial poses across pursuers and opponents at one ratio; the RNG stream shifts with nothing else, since
  the ratio changes no draw).
- Readouts as S9/S11: deadlock, stall, intervention, capture, escape, outcome counts, violations, clearance
  floors, with Wilson intervals.

## 4. Tests (fixed now)

- H-SPD-1: per pursuer × ratio, `paired_indicator_test(attacker_ind, best_scripted_ind)` where the best scripted
  probe is the one with the higher deadlock rate in that (pursuer, ratio) pair; both attackers must pass.
- H-SPD-2: per pursuer × attacker, `d = (b − c)/n` on the paired 0.8-vs-1.2 indicators with the 95 % Wald
  interval `d ± 1.96·sqrt((b + c − (b − c)²/n))/n`; CONFIRMED iff the interval ⊂ [−0.10, +0.10].
- H-SPD-3: capture rates and the saturation check, reported.
- H-SAFE names the offending cell.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **SPD-1 and SPD-2 CONFIRMED** | The stall surface does not come from the speed asymmetry: a pursuer that can outrun its evader is pinned just as often. The reviewer objection is closed and the geometric reading is strengthened. |
| **SPD-1 CONFIRMED, SPD-2 not** | The attack survives everywhere but its strength varies with the ratio; report the curve, drop the "independent of speed" claim. |
| **SPD-1 fails at 0.8 only** | The attack needs an evader that cannot be run down. The paper's scope narrows explicitly to that regime, which is the interesting one for pursuit anyway, and the boundary is stated. |
| **SPD-1 fails at 1.2** | Harness failure (1.2 reproduces S9/S10/S11); stop and diagnose, report nothing. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Cells are never re-run; no ratio, pursuer or opponent is added or removed after any number is
seen. Consumed streams before this round: ≤ 1 338 999 (S11's V-P), plus the 1 400 000+ visual stream.

## 6. Artefacts

`results/derisk/s13_cells/*.json`, `s13_round.{json,png}`; runner `scripts/derisk_s13_speed.py`
(`--cells | --summarize`, ≤ 6 workers, `OMP_NUM_THREADS=2`), run from a frozen worktree.
