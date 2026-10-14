# S24.3-PR — per-ratio training: does a selector that has SEEN a slower evader learn to run it down?

**Status: PRE-REGISTERED. Committed BEFORE any arm trains.** Triggered by D152's pre-written decision rule
(H-ZS-3 FAIL → per-ratio training at 0.8 and 1.0). Governing: `tactics_prereg_s24_3zs_2026-10-13.md` (+A1/A2),
`tactics_prereg_s24_1_2026-10-13.md` (+A1–A5), D151, D152, `tactics_plan_2026-10-13.md` review section, D83.

## 1. Question
Zero-shot, the 1.2-trained selector beats the best law at 0.8 and 1.0 but does not change *how* it plays:
it waits in `AMBUSH` for a slower orbiting evader it could simply chase (D152, CG-Z2 b). Does training at
the ratio — with the ratio now inside the observation's training range — produce a selector that (i) beats
the zero-shot one at that ratio and (ii) shifts its tactic profile toward direct chase where the evader is
slower?

## 2. What is frozen
The S24.1 learner (`pe2/selector_mdp.py`, masked PPO, OBS 55, A2.1 mask, hold 10, T = 40), the v0b game,
`RecoveryShieldV2`, dead-reckoning memory, the scripted tactical evader as training opponent, 3 M env steps,
the S17 hyper-parameters — all at the training freeze `1f9eab2` except the trainer's ratio constant, which
becomes a parameter. **Nothing else changes between a 0.8/1.0 arm and S24.1's 1.2 arms** (asserted on the
built env, CG-P1).

## 3. Arms and streams
`SELr0.8-80/81/82` at ratio 0.8; `SELr1.0-80/81/82` at ratio 1.0. Training stream `seed·1000` (as S24.1;
disjoint from all evaluation streams). Probes every 250 k on **2 450 000+** (0.8 arms) and **2 500 000+**
(1.0 arms), N = 200, vs `TACTIC-V0` at the arm's ratio. Checkpoints `ckpts_S24-3PR-{ARM}/`.

## 4. Gate — ratio-matched, joined to the zero-shot records
Each trained arm is gated at ITS ratio on gate stream **2 400 000+**, N = 500, vs `GREEDY`, `HUG`, `TACTIC-V0`
— the same seeds as S24.3-ZS, so the comparison to the zero-shot `SEL-s` at that ratio is **paired by seed**
(same dynamics, same ratio). Comparators are READ from `results/derisk/s24_3zs_cells/` (zero-shot `SEL-s`,
`LEAD-1.0`, `RULES-V0` at the ratio) and from `results/derisk/s24_1_cells/` (`SEL-s` at 1.2, for the profile
baseline); never re-run. 2 ratios × 3 seeds × 3 opponents = **18 new cells**.

## 5. Hypotheses — directions fixed now
- **H-PR-1 (seeing the ratio helps):** per ratio, `SELr-s` > zero-shot `SEL-s` (same seed, same ratio), paired
  `paired_indicator_test` p < 0.05 on ≥ 2/3 opponents, ≥ 2/3 seeds. Reported per ratio; a pass at 0.8 alone is
  the headline the question asks for.
- **H-PR-2 (the profile shifts, now properly tested):** per seed, chase share (`INTERCEPT`+`LEAD`) of
  `SELr0.8-s` at 0.8 > chase share of `SEL-s` at 1.2 (D151's record), AND χ² on the 5-tactic picks between the
  two p < 0.01; ≥ 2/3 seeds. The same for 1.0 reported, not gated.
- **H-PR-3 (still beats the law):** `SELr-s` > `LEAD-1.0` at its ratio on ≥ 2/3 opponents, ≥ 2/3 seeds.
- **H-SAFE:** 0 contact steps in training (probe rows) and in all 18 gate cells.

**Refutations.** H-PR-1 fails at 0.8 → seeing the ratio does not help; the zero-shot selector's graceful
transfer (D152) is the whole speed-agnostic story and the ratio input is inert — reported as such. H-PR-1
passes but H-PR-2 fails → the gain comes from something other than chasing more; the mechanism stays open.

## 6. Construct gate — before any capture rate is read
- **CG-P1:** on the built training env, `evader_speed_ratio` is the only config difference from an S24.1
  arm's env; the observation's ratio entry equals it (float32).
- **CG-P2 (inherited):** 0 inapplicable picks (training and gate); A2.1 at A5's tolerance; absent training
  evidence is an offender (S24.1 A3.1).
- **CG-P3 (watched):** `SELr0.8-81` vs GREEDY and vs `TACTIC-V0` on the three CG-Z2 seeds (2 400 000/1/2)
  — in particular **2 400 001**, the episode where the zero-shot selector waited in `AMBUSH` for ~700 steps:
  does the 0.8-trained selector chase it down? Plus `SELr1.0-81` on the same seeds. 12 GIFs, no rate in
  captions, watched before the table.

## 7. Stop rule, artefacts
As S24.1/S24.3-ZS. `results/derisk/s24_3pr_arm_*.json`, `s24_3pr_probe_*.json`, `s24_3pr_cells/`,
`s24_3pr_round.{json,png}`, `s24_3pr_gifs/`. Frozen worktrees for training and for the gate; gated wrappers.
Budget ≈ 0.35 M; two fix rounds max per step.

---

## Amendment A1 (2026-10-14) — CG-P3 watched; written BEFORE `--summarize`, no capture rate read

18 cells MEASURED (N = 500, seed0 2 400 000, declared ratios). 12 episodes rendered; three watched as
contact sheets. Findings:

**(a) Where it can see the evader, the 0.8-trained selector chases.** `SELr0.8-81` vs GREEDY, seed
2 400 000: `LEAD` from the first decision, brief `SWEEP`, `LEAD`/`INTERCEPT`, capture at **229** (zero-shot
`SEL-81`: 704). At 1.0, seed 2 400 001: `INTERCEPT` → `AMBUSH(0)` → `SWEEP` → **`COUNTER-ORBIT(0)`** → capture at
**212**, where zero-shot timed out — with sight, the trained selector finds the flank.

**(b) On the critical seed it still fails — but for a different and more informative reason.**
`SELr0.8-81` vs GREEDY at 0.8, seed 2 400 001 (the zero-shot 700-step `AMBUSH` episode): `INTERCEPT` reaches
the obstacle GREEDY orbits (clearance 0.023 at step 112), the evader keeps the obstacle between them, line
of sight never returns, and from step 220 to 1 200 the label is `SWEEP` — the staleness-coverage search
drags the pursuer away across the whole arena to stale cells while a 25 %-slower evader sits behind one
obstacle. **This is a library and mask gap, not a learning failure:** every tactic that goes round an
obstacle (`COUNTER-ORBIT`) is masked while the evader is unseen (correctly — the literal `applicable()` would
leak truth, S24.1 A1.3), and no tactic says "circle the obstacle where I last saw it". A slow orbiter behind
a single obstacle is uncatchable by this library whenever sight is lost. Registered: H-PR-2's chase share can
rise only on sighted spans; if H-PR-1 fails on GREEDY at 0.8 while passing on `TACTIC-V0`, this is why.
**Design note for the plan (not a change now):** a sixth pursuer tactic `ORBIT-SEARCH(k)` — circle the
obstacle nearest the dead-reckoned last-seen point — is the legal, memory-based counterpart of
`COUNTER-ORBIT` for the unseen case.

**(c) No frozen pin; interventions are brief** (clearance 0.004–0.07 m for a few steps each).

No bar, stream, seed or threshold changed. Written after watching, before reading.
