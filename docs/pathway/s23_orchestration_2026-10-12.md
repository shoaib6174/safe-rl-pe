# S23 orchestration ledger — pursuit strategy (fifth upgrade)

Plan: `pursuit_strategy_plan_2026-10-12.md`. Started 2026-10-12. **Token budget: 3.0 M** (all subagents +
reviews; my own turns are not counted against it but are kept short). Hard stop at **2.7 M consumed**:
finish the current step, report, stop.

## Loop guards (binding)
1. Steps run strictly one at a time: implement → scoped review → ruling → next. No parallel implementers.
2. **At most 2 fix rounds per step.** After the second, I rule: accept with a recorded limitation, or stop
   and report to the user. No third resumption.
3. Reviews are spec-compliance only, scoped to the diff; style is out of scope unless it hides a defect.
4. The S23 runner is a thin derivative of `derisk_s19_planner.py` (parametrised laws, same gates, same
   recorder), not a rewrite. S19's runner cost ≈ 790 k across three resumptions; that is the loop to avoid.
5. Every brief carries exact names, values and the report contract; an implementer that asks a question
   gets one answer, not a redesign.
6. Budget is reconciled in the table below after every notification, from the harness's usage line.

## Steps and budget
| # | step | model (impl / review) | budget | actual | status |
|---|---|---|---|---|---|
| 1 | `LEAD-τ` + velocity estimate + CG-S1 identity test | opus / sonnet | 230 k | **228 k** (110 impl + 118 review) | DONE `8769115`; CG-S1 50/50 both τ; differs vs Greedy 42/50; review CLEAN |
| 2 | §3 isolated-obstacle diagnostic (`PLAN`, `LEAD-0.5`, `LEAD-1.0`), reading written first | opus / sonnet | 180 k | **155 k** (impl+run; self-checked, no separate review) | DONE `5e7d22e`; LEAD +0.01/+0.02 → MPC built; §3 flaw recorded (plan A1, D143) |
| — | **Decision:** build `MPC-H` only if the diagnostic says the orbit is not already broken by `LEAD` | me | — | — | **BUILD** (pre-written branch) |
| 3 | `MPC-H` vectorised sampling planner + CG-S1 for it (conditional) | opus / opus | 500 k | **479 k** (158 + 202 + 119 review) | DONE `d2ffeae`; review CLEAN; 3 prereg notes folded into Step 4 |
| 4 | CG-S2 (model error), CG-S3 seed rule, recorder `aim_point` (+ `last_feasible_count`, numeric RNG digest) | opus / sonnet | 210 k | **191 k** (impl + two runs; no separate review — ruling below) | DONE `dc04e63`; CG-S2 medians 0.82 / 2.48 / 2.43 m (HUG/ATK = full displacement, bar not crossed); CG-S3 seeds 2 090 000/5/6 |
| 5 | Pre-registration (me) | — | — | — | DONE `pursuit_strategy_prereg_2026-10-12.md` |
| 6 | S23 runner as a derivative of S19's; review + ≤ 2 fix rounds | opus / opus | 470 k | | dispatched |
| 7 | Freeze, launch via gated wrapper, CG-S3 watched, `--summarize`, D-entry (me) | — | — | | |
| | reserve | | 500 k | | |
| | **total** | | **2.09 M + 0.5 M** | | |

## Log
- 2026-10-12: ledger opened; Step 1 dispatched.
- Step 1 landed (110 k). Accepted deviation: 'exactly zero' velocity uses the module's decoding-precision tolerance (9.54e-6 m/step; measured stationary jitter 2.2e-6) — without it CG-S1 fails in the last ulps. Step index from env.current_step (a call counter would invent velocity across a learner's hold span).
- Step 1 review CLEAN (118 k). Running total **228 k / 3.0 M**. Step 2 dispatched.
- Step 2 landed (155 k): LEAD does not beat PLAN on the diagnostic → MPC-H built. Diagnostic's §3 design flaw (walls still corner the evader) recorded as plan A1 / D143; attribution moves to H-STRAT-2; no re-run. Ruling: no separate review for a once-run diagnostic whose 16 tests cover installation/seeds/min-distance and whose only decision was pre-written — self-check of seeds/provenance done. Running total **383 k / 3.0 M**. Step 3 dispatched.
- Self-check of the diagnostic artefact executed (second attempt; first hit the wrong schema key): seeds 2 090 000–2 090 099, 100 unique, identical across the three laws; captures 61/62/63 as reported; the 3 contact steps per row coincide exactly with the 3 pursuer-start-inside-disc episodes; obstacle (0, 0, r 0.8). Top-level provenance.commit is None (the agent reported f7698c0 nested) — cosmetic, noted.
- Step 3 landed (158 k). Implementer showed the pre-specified min-distance cost has no heading signal against a faster fleeing evader (only the path term decides). Plan A2: terminal-distance cost with earliest-capture bonus; direct row = PLAN's action. Fix round 1/2 dispatched. Running total **541 k / 3.0 M**.
- Step 3 fix round 1 landed (202 k): terminal-distance cost orders headings by 2.7 m; direct row = PLAN's action; 24 tests. Running total **743 k / 3.0 M**. Scoped review dispatched (opus).
- Step 3 review CLEAN (119 k). Notes for the prereg: wall-adjacent states can leave 0/64 feasible rows (law silently = PLAN; observable via last_feasible_count, logged by CG-S4); hash the numeric RNG state not its repr (folded into Step 4, additive); the round's CG-S1 record is produced on 2 090 000+ by the runner. Running total **862 k / 3.0 M**. Step 4 dispatched.
- Step 4 landed (191 k). Ruling: no separate review for Step 4 — additive recorder fields are covered by 24 tests and will be mechanically checked against S18 replays by the runner's cg_b_additive before any cell; CG-S2/S3 are once-run measurements with pre-written readings. Budget protected for the runner's review. CG-S2's honest caveat (HUG/ATK error = full displacement) written into the prereg as a stated expectation. Running total **1 053 k / 3.0 M**. Step 5 prereg committed; Step 6 dispatched.
