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

---

## Amendment A1 (2026-10-16, before any arm trains) — decision cadence, and a second flat family

**A1.1 — the cadence asymmetry, stated as part of the treatment.** `FLAT` decides every 10 env steps
(≈ 300 000 decisions in 3 M steps); the selector decides at tactic termination or 40 steps (measured
≈ 86 500 decisions, ≈ 35 steps each). With γ = 0.99 per decision, FLAT's credit horizon is ≈ 3.5× shorter
in env steps (0.99^120 ≈ 0.30 vs 0.99^35 ≈ 0.70 over a full episode — the D128 failure mode), while FLAT
receives ≈ 3.5× more gradient samples for the same env budget. Both follow from §3 as written and neither
is changed; both are reported beside H-FLAT-1 (`decisions`, env steps per decision, per arm).

**A1.2 — `FLAT40-s` added (3 seeds):** identical to `FLAT-s` except `hold = 40` — the selector's decision
cap — so one flat family matches the selector's cadence and the other matches its primitive. H-FLAT-1 is
evaluated against BOTH families; the claim passes only if the selector beats both. CG-F1 for `FLAT40` must
report `assembly_diff == ["hold"]` and nothing else. Probe stream for `FLAT40`: **2 650 000+**.

**A1.3 — recorded, not changed:** CG-F1's observation comparison runs at reset only (where the selector's
tactic block is zero); the structural guarantee is the shared builder and a sole `_spans` override, verified
to touch only indices 28–48. A crashed arm is skipped, never resumed (stop-rule-safe, as S24.1). Training
`STRUCT` uses one persistent recovery shield for the run (S17's convention); its probes and the gate rebuild
the shield per episode — stated. Nine arms train in parallel at `OMP_NUM_THREADS=2`.
