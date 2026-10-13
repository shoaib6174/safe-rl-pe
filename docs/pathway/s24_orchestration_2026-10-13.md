# S24 orchestration ledger — the tactic game

Plan: `tactics_plan_2026-10-13.md`. Started 2026-10-13. Budget 4.0 M (hard stop 3.6 M). Loop guards as S23.

| # | step | model | budget | actual | status |
|---|---|---|---|---|---|
| 0.1 | tactic library (both roles), termination predicates, behavioural tests | opus / sonnet | 350 k | **382 k** (241 + 141 review) | DONE `e5e16b4`; review CLEAN; watch item for S24.1: selectors must only return applicable tactics (adapters do not check) |
| 0.2 | memory: occlusion-aware belief (particle filter) vs dead-reckoning gate | opus / opus | 350 k | **168 k** | DONE `a76cb2e`; gates 1 AND 2 FAIL (D145/D146) → memory = LastSeenBelief; PF shelved |
| 0.3 | recogniser + accuracy gate on self-play ground truth | opus / sonnet | 250 k | **240 k** | DONE `73dd182`; gates 1 AND 2 FAIL (D147/D148) → features only, both roles; `features(t)` is the product |
| 0.4 | scripted selectors v0 (over features) + labelled GIF harness; watch | opus / sonnet | 250 k | **317 k** | DONE `46f986e`; v0b 1.00/0.68/0.68; pins 9/30→0/30; re-watched (D150) |
| 1 | S24.1 prereg → learned pursuer selector → run → watch → read | opus / opus | 1.2 M | **≈ 1.0 M** (1.1: 467 k; 1.2: 505 k; A5 patch mine) | DONE — D151: H-TAC-1 PASS 2/3 (held-out transfer), H-TAC-2 PASS 3/3, H-TAC-3 PASS 2/3, H-SAFE HOLDS; S24.2 licensed |
| 2 | S24.2 | | 1.0 M | | |
| 3 | S24.3 speed sweep — **reordered before S24.2** (plan review): 3-ZS zero-shot (eval only) now; per-ratio training conditional | opus / opus | 0.6 M | | 3-ZS prereg committed; runner dispatched |

## Log
- 2026-10-13: plan written; Step 0.1 dispatched.
- Step 0.1 landed (225 k). Rulings in the plan. Fix round 1/2: WallSlide own band; keep scripted.py frozen. Running total **225 k / 4.0 M**.
- Fix round 1 landed (16 k). WALL-SLIDE bar asserted on the reachable-start subset (12/31), a post-measurement scope restriction — acceptable for a construct test, not for a round number; recorded. Running total **241 k / 4.0 M**. Review dispatched (sonnet).
- Step 0.1 review CLEAN (141 k). Running total **382 k / 4.0 M**. Step 0.2 dispatched.
- Step 0.2 landed (146 k): belief gate FAILS on both opponents as pre-registered (D145). Gate 2 (mode-scored) pre-registered in the plan before running; fix round 1/2 = add the statistic and run once. Running total **528 k / 4.0 M**.
- Step 0.2 closed (168 k): both belief gates fail; dead-reckoning is the memory (D146). No review needed — a failed construct gate is its own verdict and the artefact carries both tables. Running total **550 k / 4.0 M**. Step 0.3 dispatched.
- Step 0.3 landed (194 k): recogniser gate fails both roles (D147). Ruling: selectors consume motion FEATURES, never labels. Gate 2 for the pursuer-role recogniser under the evader's full-state view pre-registered; fix round 1/2. Running total **744 k / 4.0 M**.
- Step 0.3 closed (240 k): both recogniser gates fail; separability, not observability, is the residual. Features only. Running total **790 k / 4.0 M**. Step 0.4 dispatched.
- Step 0.4 landed (272 k) and the three GIFs were watched before any reading. Two construct failures ruled (RecoveryShieldV2; coverage SWEEP; HOLD-HIDDEN guard). Fix round 1/2 dispatched; re-run once. Running total **1 062 k / 4.0 M** (S24.0 line 1.2 M).
- S24.0 closed (D150): ≈ 1 107 k of its 1.2 M line. S24.1 pre-registered (`tactics_prereg_s24_1_2026-10-13.md`) with the v0b baseline quoted. Step 1.1 dispatched. Running total **1 107 k / 4.0 M**.
- Step 1.1 landed (198 k): SelectorMDP (OBS 50) + masked-PPO trainer; 340 steps/s. Prereg A1: shield/escape features added (OBS 55), n_switches excludes zero-length spans. Fix round 1/2 dispatched. Running total **1 305 k / 4.0 M**.
- Step 1.1 fix landed (26 k; cumulative 224 k): OBS 55, n_switches rule. Review dispatched (opus) — the training env gates five hours of compute. Running total **1 331 k / 4.0 M**.
- Step 1.1 review (187 k): FIXES-REMAIN, none on the high-stakes axes. A2 rules the DONE-on-entry mask (D128's horizon failure), picks for H-TAC-2, NOMEM semantics, crash resilience. Fix round **2/2** dispatched. Running total **1 518 k / 4.0 M**.
- Step 1.1 fix round 2 landed (≈ 55 k; step total ≈ 467 k incl. review) and verified directly. Worktree frozen at `1f9eab2`; six arms launched (OMP 3). Running total **≈ 1 573 k / 4.0 M**. Step 1.2 (gate runner) dispatched to build while arms train.
- Step 1.2 gate runner landed (261 k; `b937bd6`; 93 tests; pe2/ untouched since the training freeze). Review dispatched (opus). Running total **≈ 1 834 k / 4.0 M**; S24.1 ≈ 731 k of 1.2 M.
- Step 1.2 review (199 k): APPROVE-WITH-FIXES — 2 confirmed (absent evidence scored as zero; comparator span accounting blind), 2 plausible. A3 rules them. Fix round 1/2 dispatched. Running total **≈ 2 033 k / 4.0 M**; S24.1 ≈ 930 k of 1.2 M.
- Step 1.2 fix landed (≈ 45 k; step total ≈ 505 k incl. review) and verified directly; gate worktree frozen at `e4159ba` (pe2 identical to the training freeze). Launcher pointed at the gate worktree. Running total **≈ 2 078 k / 4.0 M**; S24.1 ≈ 975 k of 1.2 M. Waiting on the six arms (3/12 checkpoints each).
- **S24.1 closed (D151).** Six arms trained, gate of 24 cells, CG-T3 watched (A4) before the table; A2.1 guard refused on one env step per cell in two cells → A5 restated the bar in the open; summary read. Running total **≈ 2.08 M / 4.0 M**; remaining ≈ 1.9 M for S24.2 (1.0 M) + S24.3 (0.6 M) + reserve.
- Plan reviewed after S24.1: reorder — S24.3 zero-shot sweep BEFORE S24.2 (evaluation only; answers the speed-agnostic question with the artefact in hand; picks the balanced ratio for the alternation). S24.2 will train against opponent mixtures and carries an 'unwinnable' kill rule. S24.3-ZS pre-registered; runner dispatched.
- S24.3-ZS runner landed (290 k; `3899fbe`; 100 tests; 1306 lines, accepted). Prereg A1 records its readings. Review dispatched (opus). Running total **≈ 2.37 M / 4.0 M**.
