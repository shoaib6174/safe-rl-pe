# Pre-registration — S24-OS: ORBIT-SEARCH(k) enters the pursuer library; the selector is retrained

Written 2026-10-17 before any code of this round exists. Plan: `orbit_search_s24_2_plan_2026-10-17.md`.
Discipline as S24-FLAT: construct gates BLOCKING, GIFs watched before `--summarize`, captions carry no
rate, stop rule (nothing re-run or re-cut after a number; a crashed arm/cell is UNDETERMINED).

## 1. The tactic
`ORBIT-SEARCH(k)`: with the evader out of sight, circle obstacle k — the obstacle nearest the dead-reckoned
last-seen point (`LastSeenBelief.mean`) — the counter-orbit way about that point, until the evader is seen
again or `ORBIT_SEARCH_MAX_STEPS = 200`. Applicable iff not visible, the belief has a point, and that point's
surface distance to k is ≤ `ORBIT_SEARCH_RANGE = 2.0` m. Memory-based: reads no truth. Steering is planned
(`PlannerInterceptLaw`) exactly as `CounterOrbit`'s.

## 2. Construct gates (blocking)
- **CG-O1 (library v0 identity):** `SelectorMDP(library="v0")` reproduces an S24.1 record: the first gate
  episode (seed 2 400 000) of `SEL-81_TACTIC-V0` replayed with the frozen checkpoint gives the same outcome,
  step count, span log and violation count as the stored cell log. Also `tests/pe2/test_selector_mdp.py`
  passes unchanged.
- **CG-O2 (behavioural):** on 10 constructed states (evader hidden behind obstacle k, belief point at its
  true last-seen position, pursuer on the far side), `ORBIT-SEARCH(k)` regains line of sight within
  `ORBIT_SEARCH_MAX_STEPS` in ≥ 8/10, with the evader held still; and it is DONE on entry whenever visible.
- **CG-O3:** 18 GIFs (SEL2-80/81/82 × {HUG, GREEDY} × seeds 2 400 000–2) watched before the summary.
- Inherited: CG-T1 (mask makes an inapplicable pick impossible), A2.1 DONE-on-entry mask, H-SAFE.

## 3. Arms and cells
Arms `SEL2-80/81/82`: `derisk_s24_1_train.py` configuration unchanged except `library = "v1"` (6 actions,
OBS_LEN 59); 3 M env steps, stream seed·1000, probes every 250 000 on **2 750 000+** N = 200 vs TACTIC-V0;
checkpoints `ckpts_S24-SEL2-{s}/`. Cells (N = 500): 9 at ratio 1.2 on **2 400 000+** (GREEDY, HUG,
TACTIC-V0) paired with `SEL-s` (`s24_1_cells`); 3 at ratio 0.8 vs GREEDY on 2 400 000+ paired with the
zero-shot `SEL-s` rows of `s24_3zs_cells`. Records under `results/derisk/s24_os_cells/`.

## 4. Hypotheses
- **H-OS-1 (non-inferiority, gated):** on GREEDY and TACTIC-V0 at 1.2, `SEL2-s` is not worse than `SEL-s`:
  paired indicator test in the losing direction NOT significant (p ≥ 0.05) OR delta ≥ −0.03, on ≥ 2/3
  seeds for each of the two opponents.
- **H-OS-2 (the claim, gated):** `SEL2-s` > `SEL-s` vs HUG at 1.2 (p < 0.05, paired) on ≥ 2/3 seeds — HUG
  is the selector's weakest opponent (0.36–0.39, D151) and hides behind obstacles.
- **H-OS-3 (reported):** zero-shot at 0.8 vs GREEDY, `SEL2-s` vs `SEL-s` (D153's failure case); the
  ORBIT-SEARCH usage share per cell; decisions per episode.
- **H-SAFE:** 0 contact steps in all cells and probe rows (A3.1 absence = offender).
- Refutation: H-OS-1 fails → ORBIT-SEARCH is dropped and S24.2 uses `SEL`; H-OS-1 passes and H-OS-2 fails →
  "the library addition is harmless but does not reach the hiding evader at this ratio"; both pass → the
  library gap of D153 is closed by a memory-based tactic.

## 5. Budget
O.1 opus ≈ 220 k, O.2 sonnet ≈ 150 k + opus review ≈ 110 k, fixes ≤ 2 rounds sonnet ≈ 60 k → ≈ 0.54 M.
