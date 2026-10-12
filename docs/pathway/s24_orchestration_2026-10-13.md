# S24 orchestration ledger — the tactic game

Plan: `tactics_plan_2026-10-13.md`. Started 2026-10-13. Budget 4.0 M (hard stop 3.6 M). Loop guards as S23.

| # | step | model | budget | actual | status |
|---|---|---|---|---|---|
| 0.1 | tactic library (both roles), termination predicates, behavioural tests | opus / sonnet | 350 k | 225 k impl | `3b28ca9`; ORBIT .96 / HOP .96 / AMBUSH .83; COUNTER-ORBIT bar mis-specified (ruled); WallSlide band fix round 1/2 |
| 0.2 | memory: occlusion-aware belief (particle filter) vs dead-reckoning gate | opus / opus | 350 k | | |
| 0.3 | recogniser + accuracy gate on self-play ground truth | opus / sonnet | 250 k | | |
| 0.4 | scripted selectors v0 + labelled GIF harness; watch | opus / sonnet | 250 k | | |
| 1 | S24.1 prereg → learned pursuer selector → run → watch → read | opus / opus | 1.2 M | | |
| 2 | S24.2 | | 1.0 M | | |
| 3 | S24.3 speed sweep | | 0.6 M | | |

## Log
- 2026-10-13: plan written; Step 0.1 dispatched.
- Step 0.1 landed (225 k). Rulings in the plan. Fix round 1/2: WallSlide own band; keep scripted.py frozen. Running total **225 k / 4.0 M**.
