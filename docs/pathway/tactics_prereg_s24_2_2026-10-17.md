# Pre-registration — S24.2: learned evader selector, one pursuer alternation, the tactic meta-game

Written 2026-10-17 before any S24.2 code exists; values fixed now, amendments dated before any number.
Plan: `orbit_search_s24_2_plan_2026-10-17.md`. Discipline as S24-FLAT.

## 1. The evader selector
`EvaderSelectorMDP`: Discrete(5) over `EVADER_TACTICS` = (FLEE, ORBIT, HOP, WALL-SLIDE, HOLD-HIDDEN);
candidates resolve as `EvaderRulesV0` does (ORBIT → nearest obstacle to the evader; HOP → the first
applicable of the obstacles ordered farthest-from-the-pursuer first); mask = `applicable()` minus
DONE-on-entry; a decision at tactic DONE or after 40 env steps; reward = −(pursuer's terminal ±1) summed
over the span; observation = the evader-seat mirror of `SelectorMDP.OBS_LAYOUT` (hidden flag, pursuer
range/bearing, own pose, obstacles, wall, the PURSUER's motion features, current/last tactics, speed
ratio; no belief, no shield/escape — the evader is unshielded). The pursuer seat is a frozen selector
checkpoint driven by `SelectorMDP`'s own observation/mask/candidate code on the shared game.
- **CG-E1 (blocking):** with `EvaderRulesV0` as the decision source, the env replays `SEL-81_TACTIC-V0`
  episode 2 400 000 step-for-step (outcome, steps, both span logs, violations).
- **CG-E2:** GIFs watched before the summary (EVS-80/81/82 × {SEL-81, RULES-V0} × 3 seeds = 18).
- **CG-E3 (kill):** if ≥ 95 % of a learned selector's decisions are one tactic against every opponent, the
  library is too narrow — reported, round continues to the meta-game but the alternation is not claimed.

## 2. Stage 1 — evader arms
`EVS-80/81/82`: MaskablePPO with the S24.1 hyper-parameters, 3 M env steps, stream seed·1000, trained vs a
per-episode uniform MIXTURE of pursuers {`SEL2-best` (or `SEL-81` if S24-OS drops ORBIT-SEARCH), `RULES-V0`,
`LEAD-1.0`}; probes every 250 000 on **2 700 000+** N = 200 vs `SEL-81`. Gate N = 500 on **2 400 000+**
vs {`SEL-80`, `SEL-81`, `SEL-82`, `RULES-V0`, `LEAD-1.0`} = 15 cells, paired to the scripted-evader rows
`*_TACTIC-V0` of `s24_1_cells` (same stream).
- **H-EV-1 (gated):** `EVS-s` lowers capture vs `SEL-s` relative to TACTIC-V0 (paired, p < 0.05, capture
  DOWN) on ≥ 2/3 seeds (same-seed pairing s ↔ s).
- **H-EV-2 (gated):** it also lowers capture vs the held-out `RULES-V0` and `LEAD-1.0` (≥ 2/3 evader seeds
  on each).
- **H-EV-3 (kill):** capture ≥ 0.05 vs at least one pursuer; otherwise the ratio is wrong — stop, report.
- H-SAFE as always (the pursuer's shield; 0 contact steps).

## 3. Stage 2 — one alternation
`SEL3-80/81/82`: the pursuer selector (library as S24-OS decided) retrained vs the per-episode mixture
{TACTIC-V0, `EVS-best`, GREEDY, HUG}; 3 M steps; gate 18 cells on 2 400 000+ vs {TACTIC-V0, EVS-80, EVS-81,
EVS-82, GREEDY, HUG} paired with `SEL-s` (or `SEL2-s`).
- **H-ALT-1 (gated):** `SEL3-s` recovers capture vs `EVS-s` (p < 0.05 UP vs the stage-0 pursuer) on ≥ 2/3
  seeds, including the two held-out evader seeds.
- **H-ALT-2 (gated):** non-inferior on TACTIC-V0/GREEDY/HUG (losing-direction p ≥ 0.05 or delta ≥ −0.03) on
  ≥ 2/3 seeds per opponent.

## 4. Stage 3 — the meta-game (reported, not gated)
Pursuers {`RULES-V0`, `LEAD-1.0`, `SEL-81`, `SEL2-best`, `SEL3-best`} × evaders {GREEDY, HUG, TACTIC-V0,
`EVS-best`}: capture matrix at 1.2, N = 500 on 2 400 000+, existing records reused, missing cells run once.
Zero-sum LP (capture as the pursuer's payoff) via `scripts/derisk_s14_matrix.py`'s machinery: equilibrium
mixtures, support, each policy's exploitability (best-response gap). Tactic-level profile per cell
(usage shares of both roles). Reading pre-stated: a pure equilibrium in the learned pair means the
alternation converged; a mixed one with cyclic support is the arms race, itself the result.

## 5. Budget and stop
E.1 opus ≈ 250 k; E.2 sonnet 150 k + opus review 110 k; E.3 sonnet 80 k; E.4 sonnet 100 k; fixes ≈ 100 k →
≈ 0.80–0.90 M. With ≈ 0.85 M available before the 5.4 M hard stop and S24-OS first (≈ 0.54 M), **the
hard stop falls inside stage 1**; I stop there and report unless the budget is raised.

## Amendment A1 (2026-10-17, before any S24.2 code)
- **A1.1 reward:** the evader's span reward is the NEGATIVE of the pursuer's reward exactly as the game emits it
  (terminal ±1 and the game's 0.2 acquisition term, `r_acq`), summed over the span — pure zero-sum, no second
  reward definition. §1's "−(terminal ±1)" is replaced by this.
- **A1.2 pursuer seats in the mixture:** `RULES-V0` = the game's default `TacticLaw` select; `LEAD-1.0` =
  `fixed(rules["pursuer"].lead)` exactly as the S24.1 gate's BASELINE row; learned = a frozen MaskablePPO
  checkpoint driven through `SelectorMDP`'s own `_observe`/`action_masks`/`_candidates` (library per checkpoint).
- **A1.3 decision mechanics:** the evader is consulted BEFORE the env step (the shield mutates state before the
  wrapper reaches the opponent, so a mid-step exception is not an option); its computed action is handed to the
  wrapper through a one-shot proxy; the pursuer's action for a step is computed once and held across an evader
  decision point. CG-E1's replay identity is the test of all of this.
