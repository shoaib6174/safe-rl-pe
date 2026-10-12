# S24 orchestration ledger — the tactic game

Plan: `tactics_plan_2026-10-13.md`. Started 2026-10-13. Budget 4.0 M (hard stop 3.6 M). Loop guards as S23.

| # | step | model | budget | actual | status |
|---|---|---|---|---|---|
| 0.1 | tactic library (both roles), termination predicates, behavioural tests | opus / sonnet | 350 k | **382 k** (241 + 141 review) | DONE `e5e16b4`; review CLEAN; watch item for S24.1: selectors must only return applicable tactics (adapters do not check) |
| 0.2 | memory: occlusion-aware belief (particle filter) vs dead-reckoning gate | opus / opus | 350 k | **168 k** | DONE `a76cb2e`; gates 1 AND 2 FAIL (D145/D146) → memory = LastSeenBelief; PF shelved |
| 0.3 | recogniser + accuracy gate on self-play ground truth | opus / sonnet | 250 k | 194 k impl | `18aacde`; **gate FAILS both roles** (D147); labels dropped → features; gate 2 (full-state pursuer-role) fix round 1/2 |
| 0.4 | scripted selectors v0 + labelled GIF harness; watch | opus / sonnet | 250 k | | |
| 1 | S24.1 prereg → learned pursuer selector → run → watch → read | opus / opus | 1.2 M | | |
| 2 | S24.2 | | 1.0 M | | |
| 3 | S24.3 speed sweep | | 0.6 M | | |

## Log
- 2026-10-13: plan written; Step 0.1 dispatched.
- Step 0.1 landed (225 k). Rulings in the plan. Fix round 1/2: WallSlide own band; keep scripted.py frozen. Running total **225 k / 4.0 M**.
- Fix round 1 landed (16 k). WALL-SLIDE bar asserted on the reachable-start subset (12/31), a post-measurement scope restriction — acceptable for a construct test, not for a round number; recorded. Running total **241 k / 4.0 M**. Review dispatched (sonnet).
- Step 0.1 review CLEAN (141 k). Running total **382 k / 4.0 M**. Step 0.2 dispatched.
- Step 0.2 landed (146 k): belief gate FAILS on both opponents as pre-registered (D145). Gate 2 (mode-scored) pre-registered in the plan before running; fix round 1/2 = add the statistic and run once. Running total **528 k / 4.0 M**.
- Step 0.2 closed (168 k): both belief gates fail; dead-reckoning is the memory (D146). No review needed — a failed construct gate is its own verdict and the artefact carries both tables. Running total **550 k / 4.0 M**. Step 0.3 dispatched.
- Step 0.3 landed (194 k): recogniser gate fails both roles (D147). Ruling: selectors consume motion FEATURES, never labels. Gate 2 for the pursuer-role recogniser under the evader's full-state view pre-registered; fix round 1/2. Running total **744 k / 4.0 M**.
