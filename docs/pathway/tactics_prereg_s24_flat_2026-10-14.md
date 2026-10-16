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

**A1.2 correction (2026-10-16):** CG-F1's assembly comparison reads the GAME's `hold` (10 for every arm);
`FLAT40`'s 40-step span is the env's own `hold` attribute, so its CG-F1 reads `assembly_diff == []` with
`flat_hold = 40` recorded in the arm JSON — not `["hold"]` as A1.2 first said. **Disclosure:** the first
launch at `edfe26f` died at CG-F1 (my patch passed `hold` to the `SelectorMDP` clone); nothing trained, no
probe row, no number seen; artefacts archived under `results/derisk/interrupted_2026-10-16_flat/`.

**A1.4 (2026-10-16, before any gate cell runs; arms at ≤ 1 M steps, no capture number seen):** CG-F2 watches
`FLAT40` too — `FLAT40-80/81/82` join `FLAT-80/81/82` and `STRUCT-81` in the GIF rows (3 unselected gate
episodes × {TACTIC-V0, GREEDY} each): **36 GIFs**, not 24. Reason: A1.2 made `FLAT40` a family H-FLAT-1
must clear; an unwatched family is the gap D125 came through. The gate also re-reads each of the nine arm
JSONs at summary time (verdict `MEASURED`, `resumed_from` null, CG-F1 block present and ok, `hold == 40` for
`FLAT40`); absence of any of these is an offender (A1.3's "never resumed" made checkable). The selector's
own `decisions` / env-steps-per-decision are reported in the A1.1 cadence table from the S24.1 records so the
ratio can be read off one artefact. Neither changes a hypothesis, a bar, or N.

**A1.4 correction (2026-10-16, same day, still before any gate cell):** 7 GIF rows × 2 opponents × 3 seeds =
**42 GIFs**, not 36 — an arithmetic slip in A1.4, caught by the implementer. Rows and seeds unchanged.

---

## Amendment A2 (2026-10-16, CG-F2 watched BEFORE `--summarize`; no rate seen)

All 42 GIFs watched as 8-frame contact sheets per row plus three full-resolution frames. Renderer
convention (checked in `pe2/render.py::ROLE_STYLE`): **pursuer red, evader blue**; pink = pursuer's field
of view, grey = occlusion shadows cast from the pursuer; the orange ring = the pursuer's shield intervening.
Per-episode outcomes (not rates): 3 captures in 42 (FLAT-81 vs TACTIC-V0 seed 0; FLAT40-80 vs TACTIC-V0
seed 2; STRUCT-81 vs GREEDY seed 0); the rest time out at 1200.

**What the baselines do (behavioural reading):**
- `FLAT` (hold 10): the pursuer wanders in loops and, in most episodes, ends at a wall or corner with the shield
  intervening at ≈ 0 m clearance for hundreds of steps (e.g. FLAT-82 vs GREEDY seeds 1 and 2; FLAT-80 vs
  TACTIC-V0 seed 0: pinned in the bottom-right corner from ≈ step 500 to 1200 while the evader holds 3 m away).
  The recovery shield offers forward motion but the policy keeps commanding into the wall — the D125 absorbing
  loop reappears under a recovery shield when the policy itself has learned to stall.
- `FLAT40` (hold 40): same phenotypes with longer straight segments; corner/wall pins again (FLAT40-81 vs GREEDY
  seed 1; FLAT40-80 vs GREEDY seed 2 leaves the evader to hug the left wall).
- `STRUCT`: when the evader is visible the law chases and often orbits the obstacle the evader hugs without
  closing (STRUCT-81 vs GREEDY seed 1, vs TACTIC-V0 seed 1); when hidden, the learned search spins in place for
  the whole episode (seeds 2 of both opponents, "sees opponent: NO" throughout).
- Evaders behave per their labels (GREEDY flees to a far corner; ORBIT/HOLD-HIDDEN/HOP/WALL-SLIDE switch as
  designed); obstacles, arena and shield identical across rows; 0 contact steps.

**CG-F2 reading: PASS as a construct** — the rows differ only in the pursuer, and the pursuers' failure modes are
the known collapse phenotypes (D108 flat collapse; D125 stall loop; D112 obstacle-hug), not an environment
defect. **Disclosure:** my first pass read the colours backwards (blue as pursuer); corrected from the
renderer source before this note was written. The summary is read next, with no further cut or re-run.
