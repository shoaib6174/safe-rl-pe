# Plan — ORBIT-SEARCH(k) (round S24-OS) then S24.2 (learned evader, alternation, tactic meta-game)

2026-10-17. User: "plan and do these: the ORBIT-SEARCH(k) tactic from D153's library gap, then S24.2
(learned evader and tactic meta-game, pre-registered first)." Budget state: ≈ 4.55 M of 6.0 M used,
hard stop 5.4 M → ≈ 0.85 M available. Estimates below: S24-OS ≈ 0.50 M, S24.2 ≈ 0.90 M. **S24.2 will
hit the hard stop around its stage 1 gate; I stop there and report unless the budget is raised.**

## Why these two, in this order
- D153 (CG-P3): the selector's remaining loss is an unseen evader circling one obstacle; no tactic in the
  library can go round the obstacle where it was last seen. `ORBIT-SEARCH(k)` is that tactic — memory-based
  (dead-reckoned last-seen point), no truth. It must enter the library with a behavioural test and the
  selector must be RETRAINED with it (the one-hot blocks grow; frozen S24.1 artefacts stay reproducible via
  a library flag). The retrained selector `SEL2` is then the pursuer S24.2 alternates against.
- S24.2 is the tactic-level arms race in a ≤ 6 × 5 space: a learned EVADER selector against the frozen
  pursuer, one pursuer alternation, and the empirical meta-game (S14's LP tooling) — which tactics beat
  which, whether a pure or mixed equilibrium exists, and each policy's exploitability.

## S24-OS — steps (prereg: `tactics_prereg_s24_os_2026-10-17.md`)
- **O.1 (opus, ≈ 220 k):** `pe2/tactics.py::OrbitSearch(cfg, k, belief)` — PursuerTactic; `applicable` iff
  the evader is NOT visible, the belief has a point, and obstacle k is the obstacle nearest that point with
  surface distance ≤ `ORBIT_SEARCH_RANGE = 2.0` m; `act` circles k the counter-orbit way about the estimate
  (reuse `counter_orbit_direction` / `counter_orbit_goal` with a full turn as the terminal angle, planned
  steering as `CounterOrbit`); DONE when the evader becomes visible or after
  `ORBIT_SEARCH_MAX_STEPS = 200`. Library flag: `SelectorMDP(..., library="v0"|"v1")`; `v1` adds the family
  `ORBIT-SEARCH` as the 6th action (`_candidates` resolves k = nearest obstacle to `_estimate_xy`, masked
  when visible — also DONE-on-entry when visible, A2.1), one-hot blocks grow (OBS_LEN 55 → 59); `v0` is
  bit-identical (CG-O1 replay test against an S24.1 cell log). `PursuerRulesV0` unchanged (ORBIT-SEARCH is
  available to the learned selector only — stated). Recogniser label space unchanged (labels unused, D148).
- **O.2 (sonnet, ≈ 150 k):** trainer = `derisk_s24_1_train.py` with `--library v1` (arms `SEL2-80/81/82`,
  prefix `s24_os`, same hyper-parameters, 3 M steps, probes every 250 k N = 200 vs TACTIC-V0 on
  2 750 000+); gate = derivative of the S24.1 gate: 12 cells (3 seeds × {GREEDY, HUG, TACTIC-V0} at 1.2 on
  2 400 000+ paired with `SEL-s`; 3 seeds × GREEDY at 0.8 on 2 400 000+ paired with the zero-shot `SEL-s`
  rows of `s24_3zs_cells`), joins outside binding, GIFs 18 (3 seeds × {HUG, GREEDY} × 3) watched first.
  Opus first review of the gate (≈ 110 k), ≤ 2 sonnet fix rounds.
- Launch discipline as S24-FLAT (frozen worktree, pattern-free launch line, confirm by files, never launch
  in the same response as a code change).

## S24.2 — stages (prereg: `tactics_prereg_s24_2_2026-10-17.md`)
- **E.1 (opus, ≈ 250 k):** `pe2/evader_selector_mdp.py::EvaderSelectorMDP` — the mirror of `SelectorMDP`
  for the evader seat: Discrete(5) over `EVADER_TACTICS`, masked (applicable + DONE-on-entry), decision at
  tactic DONE or 40 steps, terminal reward = −(pursuer's). The pursuer is a FROZEN selector checkpoint
  driven through `SelectorMDP`'s own `_observe`/`action_masks`/`_candidates` on the shared game (no second
  copy of the pursuer's decision code). CG-E1: with `EvaderRulesV0` as the decision source, the new env
  replays an S24.1 `SEL-81_TACTIC-V0` episode step-for-step.
- **E.2 (sonnet, ≈ 150 k + opus review 110 k):** evader trainer (arms `EVS-80/81/82`, mixture of pursuers
  per episode) + gate (15 cells) + GIFs.
- **E.3 (sonnet, ≈ 80 k):** alternation — pursuer arms `SEL3-80/81/82` via the O.2 trainer with an
  opponent-mixture flag; gate 18 cells.
- **E.4 (sonnet, ≈ 100 k):** meta-game matrix (pursuers × evaders), missing cells only, LP equilibrium and
  exploitability via `scripts/derisk_s14_matrix.py`'s tooling; tactic-usage profiles per cell. Report-only.

## Kill rules (pre-stated)
- S24-OS: if CG-O1 fails (v0 not bit-identical) nothing trains until fixed; if SEL2 is significantly worse
  than SEL on GREEDY or TACTIC-V0 on ≥ 2/3 seeds, ORBIT-SEARCH is dropped from the library and S24.2 uses SEL.
- S24.2: if the learned evader drives shielded capture below 0.05 against EVERY pursuer, the ratio is wrong
  for the alternation — stop and report (plan 2026-10-13 §review). If either learned selector puts ≥ 95 % of
  its decisions on one tactic against every opponent, the library is too narrow — the round says so.
