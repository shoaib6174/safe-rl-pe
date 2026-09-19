# S14 — the empirical game matrix: every defender against every attacker (2026-09-19, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any cell of this round exists.**
Governing: **D112–D123** (the whole shielded chain), **D117** (paper plan, priority 3: report exploitability
rather than a narrative), the AAMAS reframe (`docs/venue_assessment_2026-09-19.md`).

## 1. Why this round exists

The paper currently describes an arms race in prose, with pairwise numbers measured on different streams in
different rounds. That is not how a game is reported. This round evaluates **every defender against every
attacker on one stream**, giving each defender a worst-case exploitability instead of a best-guess one, and
turning the chain into a matrix a reader can check. It is evaluation only: no training, no new policies.

## 2. The matrix

**Defenders (row player, the shielded pursuer; CBF filter at δ = 0, γ = 0.2, `shield_obs` as trained):**

| id | what it is |
|---|---|
| `TEACHER` | scripted `SearchingPursuer`, shielded — the non-learned baseline |
| `STR14` | S3 learner, trained UNSHIELDED (`shield_obs=False`) |
| `RC19` | S5 shield-aware remedy |
| `RCM28` | S7 mixture-trained (the disengaged corner) |
| `FT-B-41` | S10 artefact (anchored fine-tune) |
| `SEL-A-50` | S12 selected policy (same recipe, independent batch) |

**Attackers (column player, the evader):**

| id | what it is | trained against |
|---|---|---|
| `CTL-D125` | structured learned evader | STR14-SHIELD |
| `CTL-D225` | structured learned evader | RC19-SHIELD |
| `ATK2-31` | structured learned evader | RCM28-SHIELD |
| `ATK4-45` | structured learned evader | FT-B-41-SHIELD |
| `SPLITHUG` | scripted structured evader | — |
| `HUG7` | `ObstacleHuggingEvader` | — |
| `GREEDY` | `GreedyEvader` | — |

6 × 7 = **42 cells**, stream **1 370 000+**, N = **300**, seed-paired across every cell (one arena layout and
one initial-pose set per seed, shared by all 42 cells). Two payoffs per cell: the pursuer **deadlock-episode
rate** (the exploitability axis) and the **capture rate** (the task axis), plus the usual stall, intervention,
escape, violation and clearance readouts with Wilson intervals.

**Note, stated now:** the matrix is not square in the best-response sense. Four defenders have an attacker
trained against them; `TEACHER` and `SEL-A-50` do not. Their rows are therefore transfer-only and are labelled
as such everywhere; no worst-case claim for those two is presented as a best response.

## 3. Hypotheses and derived quantities (fixed now)

**H-MAT-1 (are our best responses actually best responses?):** each of the four learned attackers achieves its
highest deadlock rate against the defender it was trained against. Per attacker, the test is
`paired_indicator_test(own_defender_ind, other_ind)` against the defender with the next-highest rate; the
attacker is `SPECIALISED` if its own defender is the argmax, `GENERAL` if another defender's rate is higher
(reported with the paired p). CONFIRMED overall iff ≥ 3 of 4 attackers are `SPECIALISED`.

**H-MAT-2 (the defender ordering, on worst case):** using each defender's worst-case deadlock rate
`w(d) = max over the 7 attackers`, the pre-registered ordering is
`RCM28 < FT-B-41 ≈ SEL-A-50 < RC19 ≤ STR14 ≤ TEACHER`. Each `<` is tested paired against the attacker that
attains the larger side's maximum, at p < 0.05; `≈` is reported, not tested. CONFIRMED iff every `<` holds.

**H-MAT-3 (no free lunch, reported):** no defender simultaneously attains the lowest worst-case deadlock rate
and the highest worst-case capture rate `c(d) = min over the 7 attackers`. The Pareto frontier over
`(w(d), −c(d))` is reported; if one defender dominates, that is stated plainly instead.

**Derived, reported without a verdict:** the zero-sum **meta-game** on the deadlock payoff, row player
minimising and column player maximising, solved exactly by linear programming (`scipy.optimize.linprog`): its
value, both maximin mixed strategies, and which policies carry positive support. This says which defenders and
which attacks survive in equilibrium of the population we built, and it is the round's headline artefact for
the paper's game-theoretic section. The same is reported for the capture payoff.

**H-SAFE:** 0 pursuer contact steps in every cell.

## 4. Visuals

One GIF per cell into `results/derisk/s14_gifs/`, named `<defender>__<attacker>.gif`, rendered on the **first
seed of the stream (1 370 000) for every cell, with no selection** — so all 42 share one arena and one set of
initial poses and can be compared directly, and no episode is cherry-picked. Each title carries the cell's
measured deadlock and capture rates, so a GIF is self-describing. Visuals carry no verdict.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **H-MAT-1 CONFIRMED** | Our best responses are genuinely defender-specific, so the chain is a real iterated best-response sequence and the exploitability numbers mean what we say. |
| **H-MAT-1 fails (attackers are GENERAL)** | The attack is a property of the shield, not of any defender: weaker as an arms-race story, STRONGER as a generality claim. Report it as the latter. |
| **H-MAT-2 CONFIRMED** | The progression the paper narrates survives a single-stream, worst-case test. |
| **H-MAT-2 fails** | A defender's advantage was stream- or opponent-specific; the narrative is corrected to what the matrix shows. |
| **meta-game support is concentrated on one defender** | That defender is the paper's recommendation, with a quantified equilibrium value. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Cells are never re-run; no defender, attacker or metric is added or removed after any number is
seen. Consumed streams before this round: ≤ 1 360 299, plus the 1 400 000+ visual stream.

## 6. Artefacts

`results/derisk/s14_cells/*.json`, `s14_round.{json,png}`, `s14_gifs/*.gif`; runner
`scripts/derisk_s14_matrix.py` (`--cells | --summarize | --gifs`, ≤ 6 workers), run from a frozen worktree.
