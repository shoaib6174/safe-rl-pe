# Pre-registration — S25: does the filter set the game's value? (margin sweep + full-observation pursuer)

Written 2026-11-01 before any S25 code or number exists. Evaluation only: every policy is frozen (no training).
Discipline as S24: construct gates BLOCKING, GIFs watched before `--summarize`, captions carry no rate, stop rule
(nothing re-run or re-cut after a number is seen; a crashed cell is UNDETERMINED, never re-run).

## 1. Question
D157's reading was that the tactic meta-game's value (0.138 at the saddle `SEL3-80` × `EVS-80`) is set by the
filter's obstacle margin, not by either learner. That is an interpretation of one matrix at one filter setting.
S25 varies the filter with the players fixed, and adds a pursuer with full observation, to test two things:
(a) removing or tightening the filter moves the value in the predicted direction; (b) the learned evader's
exploit does not depend on the pursuer's partial observability.

## 2. Cells (evaluation only, no training)
- **Pursuer rows (4):** `RULES-V0` (the game's default `TacticLaw` select), `SEL-81` (S24.1's frozen selector,
  library v0), `SEL3-80` (S24.2 stage-2 pursuer, D157's saddle row), `FULLOBS` — a fixed law that reads the TRUE
  evader pose every step and turns toward it at full speed (`pe2.scripted.GreedyPursuer`'s command, normalised as
  every law is), installed as `TacticLaw.select = fixed(tactic)`; no search, no tactics, never DONE.
- **Evader columns (4):** `GREEDY`, `HUG`, `TACTIC-V0`, `EVS-80` (D156's learned evader, via `LearnedEvaderSelect`).
- **Filter settings (5):** `NONE` (pass-through `NoOpShield`: contacts are COUNTED, not prevented — a control, not
  a safety claim); `d0.00` (the measured game: `RecoveryShieldV2`, margin 0.0, γ 0.2, recovery_after 20);
  `d0.05`, `d0.10`, `d0.20` (margin δ in metres added to every barrier radius, S9's grid minus 0.30; all else
  equal). Policies trained at δ = 0 are evaluated ZERO-SHOT at the other settings — S25 measures the value of the
  filter-induced game for FIXED players, not a re-equilibrated value (stated limit).
- 4 × 4 × 5 = **80 cells**, N = 500 paired episodes on **2 400 000+** (the S24 gate stream), ratio 1.2, hold 10.
  Records `results/derisk/s25_cells/{ROW}_{EVADER}_{SETTING}.json`; validity N == 500, seed0 == 2 400 000,
  declared row / opponent / setting; duplicates reject both; UNDETERMINED never re-run; `--cells` prints no rate.

## 3. Construct gates (blocking)
- **CG-V1 (identity at δ = 0):** the `d0.00` cells that reproduce an existing pair must match the stored record's
  `capture_ind` list exactly (same stream, same assembly): `SEL-81 × {GREEDY, HUG, TACTIC-V0}` and
  `RULES-V0 × {GREEDY, HUG, TACTIC-V0}` vs `s24_1_cells`; `SEL3-80 × {GREEDY, HUG, TACTIC-V0, EVS-80}` vs
  `s24_2b_cells`; `SEL-81 × EVS-80` and `RULES-V0 × EVS-80` vs `s24_2_cells` (`EVS-80_SEL-81`, `EVS-80_RULES-V0`,
  measured there from the evader's driver — the identity across drivers CG-A1/A2 showed for one episode is checked
  here over all 500). Any mismatch REFUSES the summary: the shield override or the seat plumbing changed the game.
- **CG-V2 (watched):** 12 GIFs before the summary — `FULLOBS × EVS-80` at `d0.00` and `NONE`, `SEL-81 × EVS-80` at
  `NONE` and `d0.20`, seeds 2 400 000/1/2 each. Captions: row, evader, setting, seed; no rate.
- **CG-V3:** the `NONE` setting is the pass-through class (`method == "none"` on every step), and every `dδ`
  record's shield block reads margin δ, γ 0.2, recovery_after 20.

## 4. Hypotheses (all on capture against `EVS-80` unless stated; paired indicator test, p < 0.05)
- **H-VAL-1 (the claim):** without the filter, capture against `EVS-80` rises for every pursuer row: `NONE` vs
  `d0.00`, paired UP, significant, **4/4 rows**. Refutation: if any row does not rise significantly, the learned
  evader's orbit beats that pursuer on geometry alone and "the filter sets the value" is overstated for it; if
  no row rises, the reading is withdrawn.
- **H-VAL-2 (monotone in the margin):** `d0.20` vs `d0.00` is paired DOWN significant on ≥ 3/4 rows and UP
  significant on none; `d0.05` and `d0.10` are reported. Refutation: an UP-significant row at any δ > 0.
- **H-VAL-3 (information vs margin), threshold fixed now:** at `d0.00`, `FULLOBS × EVS-80` < 0.25 means the exploit
  is a margin effect (perfect information does not help); ≥ 0.25 means hiding is a material part of it. Reported
  with `FULLOBS × EVS-80` at `NONE` (the pure-geometry reference) and `FULLOBS` vs the three scripted evaders.
- **H-VAL-4 (the value curve, reported):** the zero-sum LP value and saddle of the 4 × 4 capture matrix at each
  setting (`scripts/s24_2_meta_game.py::solve`, pursuers maximise); per-policy exploitability per setting.
  Pre-stated reading: value(`NONE`) > value(`d0.00`) > value(`d0.20`) with `EVS-80` in the evader support at
  every shielded setting.
- **H-SAFE:** 0 contact steps in every shielded cell (δ ≥ 0, 64 cells). `NONE` cells report their contact steps
  (expected > 0) and are excluded from any safety statement.

## 5. Procedure and budget
Runner `scripts/derisk_s25_value_sweep.py` (rebinding derivative of the S24.1 gate through the S24.2b pattern;
written and tested by me — no subagent, budget ≈ 6.8 M / 7.0 M), frozen worktree, 5 chains (one per setting,
16 cells each), `--gifs` → watch → amendment → `--summarize` → D158. Reading registered now: the shield factory is
swapped per cell by rebinding the game module's `build_shield` (the assembly calls it at construction and at every
`_begin`), so a cell's every episode is built under its own setting and nothing else in the assembly changes.
