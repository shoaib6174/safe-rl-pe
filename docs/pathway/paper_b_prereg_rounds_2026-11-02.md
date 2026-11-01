# Paper B pre-writing rounds B1–B4 — pre-registration (written 2026-11-01 night, before any runner exists)

**Status: PRE-REGISTERED. Evaluation only; no training. Nothing below may change after a number is seen** (stop rule
§8). Governing: `paper_b_l4dc_plan_2026-11-01.md` (steps 2–3), `paper_b_evidence_audit_2026-11-02.md` (§5 lists what the
audit asked these rounds to archive), D126/D127/D130 (LOG-ONLY diagnostics these rounds replace), D131 (the archived
headline these rounds extend). Round label **S26** (the next session's round number); the runners are
`scripts/paper_b_geometry.py` (B1) and `scripts/paper_b_filter_sweep.py` (B2, B3, B4 share one evaluation harness),
launched from a frozen worktree by `scripts/launch_paper_b.sh` after S25 has released the host.

## 0. Shared definitions (these are the paper's definitions; the code implements exactly these)
- Filter: `CBFShield(cfg)` at δ = 0, γ = 0.2, weights w_v = 150, w_ω = 1 (D97/D112). Second filter: `PredictiveShield(cfg, horizon=4)` (S11, D121).
- **immobile** step: v_exec ≤ 0.05·v_max. **stalled** step: immobile AND v_nom ≥ 0.5·v_max (the policy wanted to move).
  **self-stop** step: immobile and NOT stalled. **stall run**: maximal run of consecutive stalled steps.
  **deadlock episode**: an episode whose longest stall run ≥ K = 40 (2.0 s). Unchanged from `pe2.metrics` (D112 onward).
  The word *pinned* is NOT used (it names a backup-tier run in `pe2.metrics`).
- **stalled state** (B1's sampling unit): the pursuer pose, obstacles and the policy's decision observation at the
  40th step of a stall run (the deadlock onset), plus the same at the 140th step if the run is still going.
  **self-stop state**: likewise for an immobile-not-stalled run.
- **filter trap at heading**: no action in the box with v > 0.05·v_max passes the filter's exact one-step test at the
  current pose (grid 41 × 41 over [0, v_max] × [−ω_max, ω_max]). **position trap**: the same holds at ALL of 72 headings
  spaced 5° apart at the same position. A stalled state that is a filter trap at heading but not a position trap is a
  **pair trap** (escapable by rotate-then-move).
- **escape** (B2): after a recovery trigger at step t, the first step t′ > t at which the body has moved ≥ 0.10 m from
  its position at t. Escape time = t′ − t. The Prop 3 bound is N + T_turn with T_turn = ⌈2π/(ω_max·dt)⌉ = 45 steps.
- **engagement**: a maximal run of steps during which the recovery has fired at least once in the last 5 steps.
  **cycle**: an engagement that begins within K = 40 steps after the previous engagement ended. Cycle rate = cycles /
  engagements (per cell, pooled over episodes); also reported per episode.
- Paired test: `pe2.derisk.paired_indicator_test` (exact two-sided binomial on discordant pairs, p < 0.05, b > c).
- H-SAFE: 0 pursuer contact steps in every shielded cell (env contact flag, D89). Any contact → stop.

## 1. B1 — stall geometry and dwell (Prop 1 measured; archives D126 V1/V2, D127 V3, D130)
**Policies:** `BASE-60`, `BASE-61`, `BASE-62` (S16 checkpoints, 14-dim obs, shield_obs on) and `RC19`
(`ckpts_RC19/RC19_10000000.zip`, 14-dim, the D125/D126 policy). Opponent: `ATK4-45` (hold 10, d_pos as S17).
Filter: plain `CBFShield`. Stream **3 000 000+** (fresh; nothing has consumed ≥ 2 800 000). N = **100** episodes per
policy, deterministic actions; the SAME 100 seeds replayed with **stochastic** sampling (`deterministic=False`,
torch seeded with the episode seed) as the dwell arm.
**Collected per step:** pose, obstacle list id, branch (law/search), v_nom, ω_nom, u_exec, tier, immobile, stalled,
and the decision observation the policy last saw. Per stalled state and per self-stop state:
(a) tier at the state; (b) exact weighted-norm projection onto the linearised rows ∩ box (SLSQP, 4 starts, as D126)
vs the shield's own output: agreement iff ‖Δu‖_W ≤ 1e-3, and whether the projection yields v > 0.05·v_max;
(c) filter trap at heading (grid); (d) position trap (72 headings); (e) an admissible rotation with |ω| ≥ 0.5·ω_max at
v = 0 exists; (f) **leave probability** p_leave: 2 000 actions sampled from the policy's Gaussian at the decision
observation, each held for `hold` = 10 steps from the state through the filter using `pe2.env.pursuer_step` (no
opponent needed), leave iff the body moved ≥ 0.10 m; predicted stochastic dwell = 1/p_leave decisions (geometric).
**Outputs:** `results/derisk/s26_b1_cells/{policy}_{mode}.json`, `s26_b1_round.{json,png}`, 4 GIFs (one deadlock
episode per policy, deterministic) — watched before `--summarize`.
**Hypotheses (fixed now):**
- **H-B1-1 (exact tier).** ≥ 90 % of stalled states are on the exact tier. (Else Prop 1 is scoped to the exact tier
  and the relaxed/backup share is reported.)
- **H-B1-2 (pair trap).** ≥ 90 % of stalled states are filter traps at heading AND < 10 % are position traps.
  Expectation from D126/D130: ~100 % and ~0 %. If position traps are common the paper's Prop 3(b) is weakened and
  says so.
- **H-B1-3 (QP agreement).** Agreement in ≥ 99 % of stalled states, and the exact projection yields forward motion
  in ≤ 1 % (D126 V1: 0/150).
- **H-B1-4 (dwell).** Stochastic arm: deadlock-episode rate ≥ 0.5 × the deterministic arm's on ≥ 3/4 policies (the
  loop survives exploration, D127); and the median observed stall-run length (runs ≥ 40, censored at episode end
  counted at their censored length) is within a factor 2 of the median predicted dwell 10/p_leave over the sampled
  states, on ≥ 3/4 policies. A failure of the second clause means the geometric model is wrong (e.g. the evader's
  motion, not the policy's noise, ends stalls) and the Prop 1 stochastic refinement becomes a remark.
- Reported, not gated: the self-stop share of immobile steps per policy; p_leave distribution; rotation-admissible share.

## 2. B2 — recovery sensitivity, cycles, bound and the perturbation baseline (Prop 3 measured; archives D125's headline)
**Policies (5):** `BASE-60/61/62` (14-dim), `STR14` (`ckpts_STR14/STR14_10000000.zip`, 11-dim, no shield_obs — the D112
policy), `RC19` (14-dim). Opponents (2): `GREEDY` (capture), `ATK4-45` (deadlock). Stream **1 740 000+**, N = **500**,
paired across every setting (same seeds; D57 protocol). hold = 10; a fresh shield per episode (S17 requirement).
**Settings (7):** `PLAIN` (CBFShield); `RECF-N` for N ∈ {5, 10, 20, 40, 80} (`RecoveryShieldV2(recovery_after=N)`);
`PERT-20`: the **perturbation baseline** — after 20 immobile steps, draw up to 8 actions uniformly from the box
(`numpy.random.default_rng(episode_seed)`), filter each through the same CBFShield, execute the first whose filtered
speed exceeds 0.05·v_max, else the ordinary filtered result (the standard random-perturbation deadlock resolution;
same admissible set, so H-SAFE covers it identically).
**Cells:** 5 × 2 × 7 = **70**, named `{POLICY}_{OPP}_{SETTING}`; records `results/derisk/s26_b2_cells/*.json`.
**Per cell:** capture_ind, deadlock_ind, longest stall run per episode, immobile-step rate, stall-step rate, firings
(forward / rotation / policy / perturbation), engagements, cycles, escape times per trigger, violations; per-step
traces for the first 3 episodes.
**Construct gates (BLOCKING for `--summarize`):**
- **CG-B2-1 identity.** The 12 cells `BASE-6x_{ATK4-45,GREEDY}_{PLAIN,RECF-20}` must reproduce S17's stored
  `capture_ind` and `deadlock_ind` bit-for-bit (`s17_arm_BASE-6x.json`, `s17_arm_RECF-BASE-6x.json`). One flipped
  episode → REFUSE (same rule as S25's CG-V1).
- **CG-B2-2 widths.** Observation width pinned by loading each checkpoint (11 / 14) before any cell.
- **CG-B2-3 watched.** 8 GIFs: `STR14_ATK4-45_{PLAIN,RECF-20}` and `RC19_ATK4-45_{PLAIN,RECF-20}` on seed 1 740 000;
  `BASE-61_ATK4-45_{PERT-20,RECF-80}` on seeds 1 740 000 and 1 740 001. Pursuer RED, evader BLUE.
**Hypotheses (fixed now):**
- **H-B2-1 (sensitivity).** For every N ≤ 40, GREEDY capture of `RECF-N` is not significantly below `PLAIN` on any of
  the 5 policies, and is significantly above `PLAIN` on ≥ 4/5 policies for N ∈ {10, 20}. Deadlock-episode rate
  (K = 40) vs `ATK4-45` < 0.05 for every N ≤ 20 on 5/5 policies. (For N ≥ 40 the K = 40 deadlock indicator is
  insensitive by construction; the longest-stall-run distribution is reported at thresholds 40 / 80 / 160 instead.)
- **H-B2-2 (bound).** For `RECF-20`, ≥ 95 % of triggers escape within N + 45 = 65 steps on every cell; the
  escape-time distribution is plotted against the bound. For `RECF-N` generally the median escape time is < N + 45.
- **H-B2-3 (cycles).** `RECF-20` cycle rate is reported; its immobile-step rate ≤ 0.12 on every cell (D131: 0.01–0.10).
  No bar on the cycle rate itself — the paper reports it as the measured cost of the stall–recover–stall pattern.
- **H-B2-4 (baseline).** `PERT-20` clears the deadlock (< 0.05 vs ATK4-45) on ≥ 4/5 policies; `RECF-20` beats
  `PERT-20` on GREEDY capture (paired, p < 0.05) on ≥ 3/5 policies and uses fewer firings on 5/5. Pre-stated
  expectation: the perturbation baseline also clears the deadlock (any admissible motion ends it) and loses on
  capture and firings; if it matches `RECF-20`, the paper says the forward offer is not special and the recovery's
  contribution is the trigger, not the candidate order.
- **H-B2-5 (D125's headline, archived).** For `STR14` and `RC19`: `RECF-20` raises GREEDY capture significantly over
  `PLAIN` (2/2) and cuts the immobile-step rate vs ATK4-45 by ≥ 80 % relative (2/2).
- H-SAFE in all 70 cells.

## 3. B3 — the phenomenon under the second filter (no recovery)
**Policies (4):** `BASE-60/61/62`, `RC19`. Opponents: `GREEDY`, `ATK4-45`. Setting `PRED-PLAIN` = `PredictiveShield(cfg,
horizon=4)`. Stream 1 740 000+, N = 500 paired with B2's `PLAIN` cells. 8 cells `{POLICY}_{OPP}_PRED-PLAIN`.
- **H-B3-1.** Deadlock-episode rate vs ATK4-45 ≥ 0.05 on ≥ 3/4 policies (the loop is not specific to the CBF-QP
  construction; D121 found deadlock lower at longer lookahead, so the rate may be below the CBF filter's — reported).
- Reported: capture and immobile under `PRED-PLAIN` vs `PLAIN` (paired), tiers (exact / backup).

## 4. B4 — the recovery composed onto the second filter (one-day cut)
Setting `PRED-RECF-20` = `PredictiveRecoveryShield` := `class PredictiveRecoveryShield(RecoveryShieldV2, PredictiveShield)`
with no method overrides: by the method resolution order `RecoveryShieldV2.filter`'s `super().filter` IS
`PredictiveShield.filter`, so every candidate (the nominal and all five recovery candidates) is accepted or rejected
by the predictive rollout, and the executed command is one the predictive filter returned (verified mechanically
2026-11-01: MRO `[PredictiveRecoveryShield, RecoveryShieldV2, PredictiveShield, CBFShield]`). Cells: the same 4 × 2 = 8.
- **CG-B4-1 (construct, BLOCKING).** Unit test: on a firing step the executed command equals
  `PredictiveShield.filter(candidate)` for the winning candidate and `PredictiveShield._accept` holds for it; runtime
  check on the first 5 episodes of every B4 cell re-filters each executed command through a fresh `PredictiveShield`
  and asserts acceptance. Failure → B4 is dropped from the paper (plan's cut), not patched.
- **H-B4-1.** Deadlock-episode rate vs ATK4-45 < 0.05 on 4/4; GREEDY capture not significantly below `PRED-PLAIN` on
  4/4 and significantly above on ≥ 2/4. H-SAFE on 8/8.
**Cut:** if the composition does not pass CG-B4-1 by Nov 3 evening, B3 is reported alone and B4 is omitted.

## 5. What each outcome fixes in the draft
| result | consequence |
|---|---|
| H-B1-1..3 hold | Prop 1 is stated for the exact projection with the measured scope; Fig. 6 (polytope + table) as planned |
| H-B1-4 fails (dwell) | the stochastic refinement becomes a remark; the stall-ending mechanism is attributed as measured |
| H-B2-1 holds, H-B2-4 holds | Prop 3 + "the trigger matters more than the candidate order" is reported honestly; the forward offer's advantage is quantified on capture/firings |
| `PERT-20` matches `RECF-20` everywhere | the paper's remedy claim is "any filtered escape attempt after N immobile steps", with V2 as the cheapest instance |
| H-B2-2 fails | the N + one-turn bound is reported as violated with the measured tail; Prop 3(b) is weakened to "typically" with numbers |
| H-B3-1 fails | the loop is reported as filter-specific at this lookahead; B4 still runs (the composition result stands on its own) |
| H-B4-1 fails | the recovery is reported as CBF-specific; the generality claim is dropped |
| H-SAFE fails anywhere | stop; the paper's safety sentence is wrong |

## 6. Budget and schedule
B1: 4 policies × 2 modes × 100 episodes ≈ 1 M steps + geometry probes (≈ 300 states × 2 000 × 10 filter calls) — ≈ 1 h
on 2 chains. B2–B4: 86 cells × 500 episodes ≈ 45 M steps; CBF cells ≈ 7 min each (S17 timing), predictive cells ≈ 3×
— ≈ 2.5 h on 5 chains. Launch when S25 releases the host (guard pattern extended with `paper_b_`); summaries by Nov 3.

## 7. Artefacts
`results/derisk/s26_b1_cells/`, `s26_b1_round.{json,png}`, `s26_b1_gifs/`; `results/derisk/s26_b2_cells/`,
`s26_b2_round.{json,png}`, `s26_b2_gifs/` (B3/B4 cells live in the same directory under their setting names);
runners `scripts/paper_b_geometry.py`, `scripts/paper_b_filter_sweep.py`; launcher `scripts/launch_paper_b.sh`; tests
`tests/pe2/test_paper_b_geometry.py`, `tests/pe2/test_paper_b_filter_sweep.py`; frozen worktree `/home/niro-1/s26_worktree`.

## 8. Stop rule
Nothing is re-run, re-seeded, re-cut or re-thresholded after any number of a round is seen. A crashed cell is
UNDETERMINED and is not re-run. Duplicate records → reject both. Amendments are dated and precede the number they
concern. GIFs are watched before `--summarize`; the construct gates refuse the summary, they are not advisory.

## Amendment A1 (2026-11-01, before any cell of any round has run): three implementation clarifications
1. **H-B1-4 clause 2 compares like with like.** The predicted dwell 10/p_leave is the expected time BEYOND the sampled
   onset; the observed quantity is therefore the stochastic arm's stall-run length MINUS K = 40 (runs ≥ 40; censored
   runs at their censored length). States whose p_leave = 0 predict an infinite dwell and are counted as such in the
   predicted median (the `predicted_infinite_share` is reported).
2. **p_leave's hold simulation re-filters the held command at every step** (as `SearchMDP` does during a hold) and the
   body is stepped with `pe2.env.pursuer_step` on the env's inset bounds; `leave` = displacement ≥ 0.10 m from the
   sampled pose within the 10 steps.
3. **Engagements (B2):** an engagement ends when 5 consecutive steps pass without a firing (`ENGAGE_GAP = 5`); its
   escape time is measured from its first firing step. B2 settings `RECF-N` keep `RecoveryShieldV2`'s default
   `move_eps` (= the immobility threshold), as in S17.

## Amendment A2 (2026-11-01, before any cell record exists): the driving branch is recorded, and H-B1-4 clause 2 is scoped to policy stalls
**Disclosure.** The B1 runner's smoke (4 deterministic + 2 stochastic episodes of `BASE-61` on seeds 3 000 000–3 000 003,
written to a scratch directory, no record under `results/`) showed stall runs of 445 and 773 steps in the stochastic arm
whose sampled states had p_leave ≈ 0.4–0.5 — i.e. the policy's own noise would leave in ~20 steps, yet the run lasted
hundreds. The likely reason is that those steps were driven by the **intercept law** (the scripted branch `SearchMDP`
uses when the evader is visible), whose command is deterministic — the law pin of D142 — and the runner did not record
which branch drove each step. **No bar is moved.** Changes, all before any record exists:
1. Every recorded step carries `branch ∈ {law, search}` and `visible` (from `SearchMDP.trace`, appended once per filter call).
2. Each sampled state and each stall run carries the branch at its K-th step. The classification reports the
   `law_branch_share` of stalled states per policy.
3. **H-B1-4 clause 2 (dwell prediction) is evaluated on `search`-branch stalls only, on both sides** — the prediction
   is about the policy's Gaussian, and a law-driven stall has no policy noise to leave by. Clause 1 (deadlock rate
   under stochastic sampling) is unchanged and counts all stalls. Law-branch stalls are reported as their own line:
   they are the same closed-loop fixed point (Prop 1 applies to any memoryless controller) and the paper says so.
4. B1 runs as two chains (det / stoch) of 4 cells each; B1's smoke seeds are the stream's first seeds and are NOT
   excluded (the smoke wrote no record and read no gated quantity; the geometry probes it printed are not reported).

## Amendment A3 (2026-11-01, before any cell record exists): two admissibility definitions, the oracle decides the trap classification
Section 0 said "passes the filter's exact one-step test". A unit test on a constructed pose (body 5 mm from an obstacle,
heading into it — no stream episode) showed the two readings of that phrase differ: under the EXACT TIER (every DCBF
row satisfied on the true successor, the projection set of Prop 1) no rotation is admissible there because the
virtual-control-point soft row fails, yet the filter EXECUTES the rotation unchanged on its relaxed tier. D126's
oracle was "u is admissible iff filter(u) returns u unchanged". Ruling, before any number: **trap-at-heading,
position-trap and rotation-admissible use D126's oracle (what will be executed), on a 21 × 21 grid at the sampled
heading and an 11 × 11 grid per swept heading; the exact-tier versions are computed on the 41 × 41 grid and reported
beside them** (`*_exact` fields). H-B1-1 (tier) is unchanged and is what scopes Prop 1 to the exact tier. Bars unchanged.

## Amendment A4 (2026-11-02, before any cell record exists): the pre-launch review's fixes
**Disclosure.** The independent reviewer ran, in memory and writing no record, 20 gate-stream episodes each of
`BASE-61_ATK4-45_RECF-20` and `BASE-62_ATK4-45_PLAIN` (bit-identical to S17's indicators and traces — the identity path
is confirmed), and 5 episodes each of `BASE-61_ATK4-45_PRED-RECF-20` and `BASE-61_ATK4-45_PERT-20`. In the latter the
perturbation baseline deadlocked 3 of 5 episodes against 0 of 5 under `PLAIN`: it fires on self-stop runs (its trigger
is `immobile`, as V2's is) and pushes the body into a trap at its heading where no random draw moves. That is the
baseline behaving as specified; H-B2-4's bars are unchanged and that expectation may now fail. No bar below moves.
1. **H-B1-4 clause 2 (ROUND-CHANGING, fixed):** the prediction is taken from the STOCHASTIC arm's own search-branch
   stall onsets (`run_at_sample == 40`), not from the deterministic arm's states and not from 140th-step samples; a
   deterministic-arm state carries no survival selection, a stochastic onset does, so mixing arms biases the predicted
   median short inside a factor-2 bar.
2. **CG-B4-1** re-filters EVERY executed command of the first 5 episodes through a fresh `PredictiveShield._accept`,
   not only firing steps (a cell with no firings would otherwise pass on zero checks).
3. **H-B2-2 censoring:** a trigger whose escape was censored by the episode end counts as a bound violation only if
   ≥ N + 45 steps remained at the trigger; otherwise it is undecidable and excluded. Each trigger records the steps
   remaining. An engagement ends 5 steps after its last firing (A1.3 made literal in the cycle window).
4. **H-B2-4 firings clause:** firings are summed over both opponents per policy.
5. **Like with like in H-B1-4:** the observed dwell is the time from the sampled onset until the body has moved
   ≥ 0.10 m from the sampled pose (the event p_leave counts), censored at the episode end and then counted at the rest
   of the run; a stall run ending because the sampled command fell below 0.5 v_max is NOT a leave.
6. **One trap rule for both tests:** a heading admits forward motion if the exact 41 × 41 grid OR the oracle grid
   (21 × 21 at the sampled heading, 11 × 11 per swept heading) contains a moving admissible action.
7. **H-B1-3 pools exact-tier stalled states only** (the projection characterisation is for the exact tier; the
   non-exact share is H-B1-1's quantity). Rotation admissibility is tested at v = 0 exactly. Self-stop states are also
   sampled at the 140th step.
8. **B1 GIFs:** a policy without a deterministic deadlock episode renders its longest-stall episode instead, so the
   gate cannot block the summary vacuously.
9. **Crash policy made explicit:** a crashed cell writes UNDETERMINED and the chain continues to its next cell; two
   consecutive crashes abort the chain (systemic). A non-default N against the real record root is refused. Records
   whose file name differs from their `name`, or whose checkpoint hash differs from the frozen checkpoint, are
   rejected. A zero deterministic deadlock rate satisfies H-B1-4 clause 1 trivially; a `RECF-20` cell with no
   decidable trigger is vacuously within the bound and is flagged. Cycles per episode are stored; the summary reports
   the paired PRED-PLAIN vs PLAIN comparison and the predictive tiers for B3.

## Outcome (2026-11-01 11:22–12:00) — see D159 (B1) and D160 (B2–B4)
Rows of §5 that fired: H-B1-1..2 hold (Fig. 6 as planned, Prop 1 stated for the exact tier — H-B1-3 missed its bar by 1 %
and the 2 % enumeration share is reported); H-B1-4 fails (the stochastic refinement becomes a remark; the long stalls are
the intercept law's; where the head is in control the evader's motion ends the stall sooner than the geometric model);
H-B2-1 holds; H-B2-2 fails (97.8 % within the bound, tail reported, Prop 3(b) "typically"); H-B2-4 fails in the opposite
direction to the pre-stated expectation (PERT-20 clears 0/5 — it accepts only moving candidates and none exists at a
heading trap; RECF-20 beats it on capture 5/5 and fires more 5/5); H-B3-1 fails at the bar (2/4) — reported as present but
weaker, not filter-specific; H-B4-1 holds and CG-B4-1 is clean, so B4 is in. H-SAFE held in all 94 cells. No re-run.
