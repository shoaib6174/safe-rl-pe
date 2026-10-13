# S24.3-ZS — zero-shot speed sweep of the S24.1 selectors (evader slower / same / faster)

**Status: PRE-REGISTERED. Committed BEFORE any round number exists.** Governing: `tactics_plan_2026-10-13.md`
(+ "Plan review after S24.1"), D150 (scripted game at three ratios), D151 (S24.1), D123 (S13's speed sweep),
`tactics_prereg_s24_1_2026-10-13.md` (+A1–A5), D83.

## 1. Question
The plan's speed-agnostic claim made falsifiable with the artefacts in hand: at `evader_speed_ratio` **0.8**
and **1.0**, do the 1.2-trained selectors still beat the best fixed law, and does the scripted game's capture
ordering (1.00 / 0.68 / 0.68 at 0.8 / 1.0 / 1.2, D150) hold for learned pursuers?

## 2. What is frozen
Everything from S24.1's gate at `d269737`: the v0b assembly (`RecoveryShieldV2`, dead-reckoning memory,
coverage `SWEEP`, T = 40), the three learned selectors' final checkpoints (`SEL-80/81/82_3000000`), the
`NOMEM-81` checkpoint (the strongest control), `LEAD-1.0`, `RULES-V0`. **No tactic reads the ratio** (tested in
S24.0); the only things that change are the env's `evader_speed_ratio` and the selector's `speed_ratio`
observation entry, which is **out of its training range** at 0.8 and 1.0 (stated limit).

## 3. Cells — evaluation only
Rows `{SEL-80, SEL-81, SEL-82, NOMEM-81, LEAD-1.0, RULES-V0}` × opponents `{GREEDY, HUG, TACTIC-V0}` × ratios
`{0.8, 1.0}` = **36 cells**, N = 500, gate stream **2 400 000+** (the same seeds as S24.1's 1.2 cells, so the
1.2 column is S24.1's record, not re-run; initial layouts are paired across ratios, dynamics are not).

## 4. Hypotheses — directions fixed now
- **H-ZS-1 (graceful transfer):** at each ratio, `SEL-s` > `LEAD-1.0` per opponent (`paired_indicator_test`
  p < 0.05) on ≥ 2/3 opponents, on ≥ 2/3 seeds. Reported per ratio; the round passes if it holds at both.
- **H-ZS-2 (ordering):** for every row, capture is non-increasing in the ratio: 0.8 ≥ 1.0 ≥ 1.2 per
  opponent (the D150 direction), with each adjacent drop significant or within 0.02 (ties allowed).
- **H-ZS-3 (profile shift, zero-shot; weak by construction, see §2):** the learned selectors' pick share of
  the *chase* tactics (`INTERCEPT` + `LEAD`) is higher at 0.8 than at 1.2 and the share of `SWEEP` +
  `AMBUSH` + `COUNTER-ORBIT` is lower, per seed (χ² on the 5-tactic histogram across ratios p < 0.01 AND the
  stated direction), on ≥ 2/3 seeds. A pass here is *suggestive*; a fail is not evidence against
  speed-agnosticism (the ratio input is out of range) — it is the trigger for per-ratio training.
- **H-SAFE:** 0 contact steps in all 36 cells.
- **Scripted check (reported):** `RULES-V0` reproduces D150's direction on the same seeds.

**Decision rule written now:** H-ZS-1 passes at both ratios → per-ratio training is NOT needed to claim
graceful transfer; S24.2 runs at the ratio whose scripted capture is nearest 0.5 (balanced). H-ZS-1 fails at
a ratio → per-ratio training at that ratio (3 seeds) before S24.2. H-ZS-3 fails → per-ratio training
regardless, as the profile question's proper test.

## 5. Construct gate — before any rate is read
- **CG-Z1:** the env's `evader_speed_ratio` is the only config difference between a 0.8/1.0 cell and its 1.2
  twin (asserted on built envs); the observation's ratio entry equals it.
- **CG-Z2 (watched):** `SEL-81` vs `TACTIC-V0` and vs `GREEDY`, first three gate seeds, at 0.8 and at 1.0
  (12 GIFs, no rate in captions): does the slower evader get caught by direct chase; does the pursuer
  still hop-ambush when it no longer needs to; any pin.

## 6. Stop rule, artefacts
As S24.1 (records measurable only at N = 500 / seed0 2 400 000 / the declared ratio; duplicates reject both;
UNDETERMINED on crash; no rate printed before the GIFs; gated wrapper; frozen worktree).
`results/derisk/s24_3zs_cells/{ROW}_{OPP}_r{RATIO}.json`, `s24_3zs_round.{json,png}`, `s24_3zs_gifs/`.
