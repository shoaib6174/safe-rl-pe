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
| 2 | §3 isolated-obstacle diagnostic (`PLAN`, `LEAD-0.5`, `LEAD-1.0`), reading written first | opus / sonnet | 180 k | | dispatched |
| — | **Decision:** build `MPC-H` only if the diagnostic says the orbit is not already broken by `LEAD` | me | — | | |
| 3 | `MPC-H` vectorised sampling planner + CG-S1 for it (conditional) | opus / opus | 500 k | | |
| 4 | CG-S2 (model error), CG-S3 seed rule, recorder `aim_point` | opus / sonnet | 210 k | | |
| 5 | Pre-registration (me) | — | — | | |
| 6 | S23 runner as a derivative of S19's; review + ≤ 2 fix rounds | opus / opus | 470 k | | |
| 7 | Freeze, launch via gated wrapper, CG-S3 watched, `--summarize`, D-entry (me) | — | — | | |
| | reserve | | 500 k | | |
| | **total** | | **2.09 M + 0.5 M** | | |

## Log
- 2026-10-12: ledger opened; Step 1 dispatched.
- Step 1 landed (110 k). Accepted deviation: 'exactly zero' velocity uses the module's decoding-precision tolerance (9.54e-6 m/step; measured stationary jitter 2.2e-6) — without it CG-S1 fails in the last ulps. Step index from env.current_step (a call counter would invent velocity across a learner's hold span).
- Step 1 review CLEAN (118 k). Running total **228 k / 3.0 M**. Step 2 dispatched.
