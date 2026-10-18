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

## Amendment A2 (2026-10-18, after E.1 landed, before any trainer exists)
- **A2.1 (wording):** A1.1 named an `r_acq` term; the game's step reward as `SelectorMDP` consumes it carries none, so the
  evader's reward is exactly `−(reward returned by the game's step)`, summed over the span. No second definition.
- **A2.2 (stated, not gated):** at a pursuer decision inside an evader span the frozen seat zeroes the game's span
  counters (A2.4 bookkeeping), so a recorder's `span_intervention_total` under-counts in this env; no gated metric reads it.
- **A2.3 (stated):** `HOP` resolves to the first applicable hop in `EvaderRulesV0`'s farthest-first order (its rule 2).
- E.1: `6fdbb2e`; CG-E1 IDENTICAL (outcome, 454 steps, 0 violations, 13 pursuer spans, 55 evader spans); ≈ 335 env steps/s.
- **A2.4 (§3/§4, after D155):** `SEL2` is excluded from the alternation and the meta-game (ORBIT-SEARCH was dropped);
  the pursuer library is v0 throughout. Stage 3's matrix is pursuers {`RULES-V0`, `LEAD-1.0`, `SEL-81`, `SEL3-best`} ×
  evaders {GREEDY, HUG, TACTIC-V0, `EVS-best`}; `EVS-best` / `SEL3-best` = the seed with the lowest / highest mean capture
  over its stage's gate pursuers / evaders (a pre-stated rule, applied once). All 16 cells exist from stages 1–2 and S24.1
  except none: no new cell is run for the meta-game.

## Amendment A3 (2026-10-18, after the E.2 review, before any EVS arm trains or any number exists)
- **A3.1 (labelling correction, no change to any test or pairing):** §2 called `RULES-V0` and `LEAD-1.0` "held-out"; they are
  IN every arm's training mixture (A1.2). The genuinely held-out pursuer seats are `SEL-80` and `SEL-82`, met inside H-EV-1 by
  `EVS-80` and `EVS-82` (same-seed pairing); `EVS-81`'s H-EV-1 seat is its training seat. H-EV-2 is therefore the
  in-distribution scripted check and H-EV-1 carries the held-out claim for two of three seeds. The round's sentences say so.
- **A3.2:** CG-T1 reads the three arms' probe rows (`train_inapplicable_picks`, `inapplicable_picks`; absence = offender) as
  S24.1 did, not only the 15 cells. A frozen pursuer seat's refused pick raises (UNDETERMINED), so it cannot be silently counted.
- **A3.3:** CG-E3's flag requires the SAME most-used tactic ≥ 0.95 in every cell of a seed (§1's "one tactic").
- **A3.4:** the trainer refuses to resume from an existing checkpoint (`SystemExit`): resumable-never, as A1.3's stop rule.
- **A3.5:** a probe row without `violations` is absent evidence (A3.1 of S24.1); records carry `pursuer_spec` and the round JSON
  carries `rejected_records`. Code: `8b6a042`.

## Amendment A4 (2026-10-18, CG-E2 watched BEFORE `--summarize`; no rate seen)
18 GIFs watched as contact sheets (pursuer RED, evader BLUE). Per-episode outcomes (not rates): 18/18 timeouts; the scripted
evader on the same three seeds was captured by SEL-81 3/3 (454/1003/661 steps) and by RULES-V0 2/3. Behaviour: the learned
evader keeps a tight ORBIT of its nearest obstacle for most of each episode and switches to FLEE / HOLD-HIDDEN / HOP as the
labels show (several tactics per episode, not one); the pursuer seats SWEEP, AMBUSH the gap, LEAD and COUNTER-ORBIT, lose line of
sight behind the hugged obstacle and intervene with the shield at ≈ 0 m clearance beside it — the obstacle-hug exploitation of
the shield margin (D112), now learned by the evader selector. The three evader seeds produce near-identical trajectories on the
same episode seed (EVS-80/81/82 vs RULES-V0 seed 2 400 001 are visually the same), i.e. the seeds converged to the same policy
under deterministic prediction. Evader tactics execute per label; arena, shield and pursuer seats are the S24.1 ones. **CG-E2
reading: PASS as a construct**; H-EV-3's kill rule is read next from the summary, with no cut or re-run.

**A4.1 disclosure (2026-10-18):** the first `--summarize` at `4bda889` was REFUSED by CG-T1 before reading any rate — my A3.2
addition read the probe files in the wrong shape (`rows` key), so every clean arm looked absent. Fixed in the summary code
only (`dd2503f`); the 15 cell records and the 18 GIFs are untouched and were recorded under `4bda889`. The gate worktree is
re-frozen at `dd2503f` for the summary. No number had been seen.

## Amendment A5 (2026-10-18, before stage 2 trains and before the meta-game runs)
- **A5.1 (E.4 definitions, fixed now):** capture payoff M (pursuer rows maximise, evader columns minimise); LP via
  `derisk_s14_matrix.meta_game(minimise_rows=False)`; exploitability of a pure pursuer policy i = value − min_j M[i,j],
  of a pure evader policy j = max_i M[i,j] − value; a "pure saddle" = both supports singletons; best responses are the
  arg-min / arg-max of the row / column. Script `scripts/s24_2_meta_game.py` (written by me, 3 tests), report-only.
- **A5.2:** `SEL3-best` = highest mean capture over the six stage-2 gate evaders, read from `s24_2b_round.json`.
