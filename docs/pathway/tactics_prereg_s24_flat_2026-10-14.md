# S24-FLAT — the baseline the publishable claim needs: flat and structured learners on the frozen tactic game

**Status: PRE-REGISTERED (draft committed before any arm trains; final check against D153 before launch).**
Governing: D151 (the selector result), the user's publishability question (2026-10-14), `tactics_plan_2026-10-13.md`,
`tactics_prereg_s24_1_2026-10-13.md` (+A1–A5), D108 (why flat RL collapsed before), D83.

## 1. Question
D151's claim — *hierarchical selection over validated tactics learns under the filter where flat learning did
not* — rests on OLD flat-RL collapses under different shields, observations and budgets. The claim is
publishable only if flat and structured learners are trained on the **exact frozen game** the selector plays
on, with the same shield, observation, opponent, budget and gate, and measured on the same seeds.

## 2. What is frozen
The S24.1 game assembly at `1f9eab2`/`994a8f3`: `RecoveryShieldV2`, dead-reckoning memory, coverage
`SWEEP` (irrelevant to FLAT — it has no tactics), `TacticOpponent(EvaderRulesV0)` as the training evader,
ratio **1.2**, 3 M env steps, S17 BASE hyper-parameters, γ = 0.99, gate stream 2 400 000+, N = 500.

## 3. Arms (3 seeds each: 80/81/82)
- **`FLAT-s`** — continuous PPO over `(v, ω)` held for 10 env steps (the same `hold`), through the recovery
  shield, at EVERY decision regardless of visibility (no intercept law, no tactics). Observation = the
  selector's 55 entries with the tactic one-hots (21) zeroed — identical information about the world and
  memory, none about tactics. Reward = terminal ±1 per span (as S24.1). This is "the same learner class
  that collapsed in D108", on today's game.
- **`STRUCT-s`** — the project's structured learner (`SearchMDP`: fixed intercept law when visible, PPO
  chooses search commands otherwise; S17 BASE config, `shield_obs`/`escape_obs`) through the same
  recovery shield against the same evader — the strongest non-hierarchical learner the project has.
Probes every 250 k on **2 550 000+** (FLAT) and **2 600 000+** (STRUCT), N = 200 vs `TACTIC-V0`.

## 4. Gate and hypotheses — directions fixed now
24 new cells: `{FLAT-80/81/82, STRUCT-80/81/82}` × `{GREEDY, HUG, TACTIC-V0}` at 1.2, N = 500 on
2 400 000+; comparators READ from `s24_1_cells/` (`SEL-s`, `NOMEM-s`, `LEAD-1.0`, `RULES-V0`), never re-run.
- **H-FLAT-1 (the claim):** `SEL-s` > `FLAT-s` (same seed, paired) on ≥ 2/3 opponents, ≥ 2/3 seeds, p < 0.05.
- **H-FLAT-2:** `SEL-s` > `STRUCT-s` likewise. A pass on H-FLAT-1 but not H-FLAT-2 is reported as "the
  selector beats flat but not the structured learner" — the honest form of the claim.
- **H-FLAT-3 (characterise, do not assume):** `FLAT-s` vs `LEAD-1.0` and `RULES-V0`, reported; and
  FLAT's immobile-step rate and pinned-run fraction (recorder) vs the selector's, reported — the D125
  regime under the recovery shield is a result either way.
- **H-SAFE:** 0 contact steps in training and all 24 gate cells.
**Refutation:** H-FLAT-1 fails → flat PPO on today's game learns as well as the selector; D151's claim is
withdrawn to "a selector matches flat learning with legible tactics" and the paper's framing changes.

## 5. Construct gate — before any capture rate is read
- **CG-F1:** FLAT's env differs from the selector's ONLY in the action space and the zeroed tactic block
  (asserted on built envs); STRUCT's only in the law/learner split (S17's configuration, asserted).
- **CG-F2 (watched):** 3 unselected gate episodes per FLAT seed vs `TACTIC-V0` and 3 vs `GREEDY`, plus the
  same for `STRUCT-81`, labelled, no rate in captions — specifically for stalling, spinning and pins.
- **CG-F3:** inherited CG-T1/A2.1 do not apply to FLAT (no mask); the A3.1 absent-evidence rule does.

## 6. Budget and models
≈ 0.5 M: FLAT env + trainer (new learning code → opus implementer; sonnet scoped re-review), STRUCT arm
(rebinding of the S17 trainer → sonnet), gate as a derivative of the S24.1 runner (sonnet), one opus first
review of the gate. Stop rule, launch discipline, records validity as S24.1/A3–A5.
