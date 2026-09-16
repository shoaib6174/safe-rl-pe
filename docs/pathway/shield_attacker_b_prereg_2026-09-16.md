# S6b — Step 9 shield-attacker, second round: the right comparator (2026-09-16, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D114** (S6: H-SAFE held; H2 `UNDETERMINED (harness)` because the pre-registered scripted
comparator — the structure's plain-flee positioning rule — was weaker than an unrewarded positioning
policy; the stall bonus was inert; learned evaders pinned both shielded defenders 0.61–0.69 of episodes
against the best scripted probe's 0.53 / 0.34). S6's spec, components and defenders are unchanged.

## 1. What changes and why

S6 asked whether a learned evader weaponises the shield and compared it to the wrong baseline. This round
asks the same question with the comparator the S6 diagnosis identified: **an identical structured evader
trained with every reward zeroed** — the same positioning structure, the same budget, no learning signal —
so what is measured is what *learning* adds, seed-paired. The scripted bar is the strongest scripted probe,
`ObstacleHuggingEvader` (HUG7). The explicit stall bonus is dropped (D114: inert on 6 of 6 seeds). Nothing
from S6 is re-read; every number here comes from fresh seeds and fresh streams.

**H2 (weaponisation by learning):** per defender, the learned evader raises the pursuer's deadlock-episode
rate above BOTH (i) its zero-reward twin (seed-paired) and (ii) HUG7, on ≥ 2 of 3 seeds each.
**R3-b (reported):** the learning gain (CTL − EZ) on RC19-SHIELD vs on STR14-SHIELD.
**H-SAFE** (kill criterion): 0 pursuer contact steps in every arm's training, probes and gate, and in the
comparator episodes.

## 2. Protocol

- Everything as S6 §2–§3 (`EvadeMDP`, hug law ≤ 4 m, hold 10, full state, PPO from scratch, γ 0.99, SB3
  defaults, 10 M env steps, probes every 1 M at N = 200, no selection, `OMP_NUM_THREADS=2`), with:
- Training seeds **25, 26, 27** (wrapper `seed0 = seed × 1000`); probe stream **1 140 000+**; gate stream
  **1 150 000+** (N = 500), paired within each defender across every arm and comparator (fresh; ≤ 1 130 999
  consumed).
- Defenders **D1 = STR14-SHIELD**, **D2 = RC19-SHIELD** (frozen, δ = 0), as S6.
- Comparators measured on the gate stream BEFORE any arm, per defender: **HUG7** (`ObstacleHuggingEvader`,
  the bar) and **GREEDY** (`GreedyEvader`, reported); an arm refuses to run without `s6b_start.json`.

## 3. The arms

| arm | defender | reward | seeds | purpose |
|---|---|---|---|---|
| **CTL-D1**, **CTL-D2** | D1 / D2 | terminal (+1 timeout / −1 capture), no bonus | 25, 26, 27 | the learned evader |
| **EZ-D1**, **EZ-D2** | D1 / D2 | all rewards zeroed | 25, 26, 27 | the seed-paired harness comparator: the same structure with no learning signal |

Twelve processes, ≈ 5 h concurrent.

## 4. Verdicts (fixed now)

- Deadlock test as S6 §5: `paired_indicator_test` on the per-episode pursuer-deadlock indicator, p < 0.05
  with more discordant pairs in the first evader's favour.
- **H2(D) PASSES** iff CTL-D-s beats EZ-D-s on ≥ 2 of 3 seeds (seed-paired) AND CTL-D-s beats HUG7 on
  ≥ 2 of 3 seeds. **H2 overall** iff H2(D1) or H2(D2). Both clauses are reported separately.
- **Harness:** if EZ-D-s beats HUG7 on ≥ 2 of 3 seeds, the structure alone already exceeds the scripted probe
  and H2(D)'s HUG7 clause is reported as `structure-driven`; H2(D) then rests on the EZ clause alone and is
  labelled `PASS (vs EZ only)` if that clause holds.
- **R3-b (reported):** mean over seeds of (CTL − EZ deadlock rate) on D2 minus the same on D1, with the
  per-seed values; also the absolute CTL rates on D2 vs D1 (does the remedy hold up against a learned
  attacker at all?).
- **Escape trade-off (reported):** CTL vs EZ vs HUG7 escape rates.
- Missing seed ⇒ that defender `INCOMPLETE`. H-SAFE fails ⇒ stop, census, user ruling.
- **Visual gate:** `step_09b_attack.gif` — the best CTL-D1 seed vs its EZ twin on a seed where CTL deadlocks
  the pursuer and EZ does not.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **H2 PASSES on either defender** | R2 measured with a valid control: a learned opponent weaponises a hard shield beyond the best scripted probe. Figure 1 candidate. R3-b says whether the S5 remedy resists it. |
| **H2 FAILS** | Learned positioning at this budget does not exceed unrewarded positioning and/or the scripted probe: no R2 beyond the hugging geometry. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is seen.
Seeds 25–27, the streams, the deadlock test and the ≥ 2/3 rules are fixed here.

## 6. Artefacts

`results/derisk/s6b_start.json`, `s6b_arm_{ARM}.json`, `s6b_probe_*`, `ckpts_{ARM}/` with ARM ∈
{CTL-D1{25,26,27}, CTL-D2…, EZ-D1…, EZ-D2…}, `s6b_round.{json,png}`, `step_09b_attack.gif`. Signed in the
worklog.
