# Spec: Strengthening the CoRL 2026 Paper via an Exploitability Test

**Date:** 2026-05-02
**Author:** S88 brainstorming session (continuation from S87)
**Status:** Approved (sections 1–3 walkthrough, finalized)
**Target:** CoRL 2026 stretch, ICRA 2027 fallback
**Related decisions:** D28, D29, D31, D32 (decision log)

---

## 1. Problem & motivation

The SP17b convergence cohort produced two early "converged" seeds (s48, s49)
that were used to draft the abstract claiming *Nash equilibrium convergence at
capture rate 0.51 ± 0.03*. Per-seed trajectory analysis on 2026-05-02
(`/tmp/seed_analysis/analyze.py` over s48/s49/s50/s51 + ABL-1/ABL-2)
revealed three findings that put the headline claim at risk:

1. **Cohort-wide curriculum-end drift.** Every SP17b seed drops capture rate
   by ≈0.12 at the exact micro-phase where the masking curriculum reaches
   `p_full = 0` (M2550, ≈5.22 M steps). s48 keeps drifting after the drop
   (last 8-eval mean 0.315). s49 drifts more slowly (mean 0.399). s50 has not
   yet crossed the boundary at the time of writing.
2. **s51 collapse.** Pursuer success-rate dropped to 0.000 for ≥5 consecutive
   evals; rollback restored a phase-900 checkpoint with peak CR 0.13. Recovery
   without a config change is unlikely.
3. **The 0.51 number is a curriculum artifact, not the equilibrium.** It was
   measured during the late stages of curriculum decay, before `p_full = 0`.
   Under the actual deployment condition (full PO, no curriculum aid), the
   cohort sits at CR ≈ 0.31–0.40.

The central question the paper must answer is no longer
*"did self-play converge to capture rate 0.5?"* but
*"is the policy pair the cohort produced at (or near) a Nash equilibrium of
this game, regardless of the numerical CR value?"*

A capture rate of 0.34 can be a perfectly good NE — it just means the game
favours the evader under full PO. What would NOT be acceptable is a policy
pair that looks stable in self-play but is actually a non-equilibrium
local fixed point, exploitable by a simple best-response.

## 2. Goal

**Run a fresh-init best-response (BR) exploitability test on s48 and s49 to
decide which paper framing is empirically defensible.** The result of this
test gates abstract phrasing, contribution wording, and figure 1 of the paper.

This spec covers only the design and execution of the exploitability test
and the decision rule that maps its outcome to a framing choice. Writing the
paper sections is downstream and out of scope here.

## 3. Hypotheses and decision rule

Definitions (computed per BR run after it finishes):

```
br_cr_curve  = capture-rate at evals (250 K, 500 K, ... 1.5 M)
baseline_cr  = mean(last 8 micro-phase CRs of the frozen seed)
                 # ~0.31 for s48, ~0.37 for s49 — recompute at test time

# A pursuer BR wants high CR; an evader BR wants low CR.
exploit_gap  = max(br_cr_curve) − baseline_cr     if BR side is pursuer
             = baseline_cr − min(br_cr_curve)     if BR side is evader
```

Tolerance: **ε = 0.05**.

**Step 1 — classify each seed independently.** For each frozen seed (s48,
s49) consider its two BR runs and assign one of:

| Per-seed label | Definition |
|----------------|------------|
| L1: at ε-NE | both BR runs for this seed have `exploit_gap ≤ ε` |
| L2: pursuer-exploitable | the pursuer-side BR run for this seed has `exploit_gap > ε`, evader-side does not |
| L3: evader-exploitable | the evader-side BR run for this seed has `exploit_gap > ε`, pursuer-side does not |
| L4: two-sided exploit | both BR runs for this seed have `exploit_gap > ε` |

**Step 2 — combine across seeds.** Map the (s48 label, s49 label) pair to a
final hypothesis:

| (s48, s49) | Hypothesis | Framing decision |
|-----------|------------|------------------|
| (L1, L1) | **H1: cohort at ε-NE** | Adopt **Framing A** (defended). NE convergence claim survives with corrected CR ≈ 0.34 ± 0.03 and the verification step cited. |
| (L2, L2) or (L3, L3) | **H2/H3: cohort one-sided exploitable, replicated** | Adopt **Framing C**. Frozen side has a hole replicated across seeds; paper's contribution becomes "characterizing curriculum-end exploitability". |
| (L4, L4) | **H4: cohort two-sided exploitable** | Adopt **Framing B** (failure analysis). Strongest negative result; pivot to ICRA. |
| any other mismatch (e.g. (L1, L2), (L2, L3), …) | **H5: seed-dependent NE** | Treat the disagreement itself as a paper finding (NE quality is seed-sensitive). Default to Framing C with the seed-dependence as the headline observation. |

**Why two-step.** Aggregating across seeds before classifying (e.g. "BR-1 or
BR-3 > ε") would let a single rogue seed flip the entire framing without
distinguishing replication failure from genuine cohort behaviour. Per-seed
classification then explicit cross-seed combination preserves the
information needed to decide whether a finding generalizes.

## 4. Experimental matrix

| Run | Frozen opponent (loaded from snapshot) | Trained side | Frozen-seed baseline CR |
|-----|----------------------------------------|--------------|-------------------------|
| BR-1 | s48 evader  | fresh SAC pursuer | 0.31 |
| BR-2 | s48 pursuer | fresh SAC evader  | 0.31 |
| BR-3 | s49 evader  | fresh SAC pursuer | 0.37 |
| BR-4 | s49 pursuer | fresh SAC evader  | 0.37 |

Per-run config:

- Algorithm: SAC (must match cohort; PPO BR vs SAC opponent confounds
  exploitability with algorithm change).
- Hyperparameters: identical to cohort. The cohort training config is the
  source of truth (locate it under `configs/` — likely `SP17b.yaml` but the
  implementer should confirm). Critical values to preserve:
  `ent_coef = 0.03` fixed, the cohort buffer size, learning rate, batch
  size, gradient steps per env step, target entropy.
- **Curriculum disabled.** The cohort anneals `p_full` from 1.0 → 0.0 over
  ≈5 M steps. For the BR test, hold `p_full` fixed at 0.0 (full partial
  obs) for the entire 1.5 M steps. The frozen opponent already operates
  under full PO, and we want exploitability under the deployment condition,
  not under a curriculum that would aid the BR side. Cross-check the
  `p_full = 0` semantics during smoke test (in some implementations the
  flag is inverted).
- No alternating freeze, no PFSP, no forced switch — vanilla single-side
  training against a fixed adversary.
- Step budget: 1.5 M.
- Eval cadence: every 250 K steps, 200 episodes per eval.
- Random seed: `baseline_seed + 100` (148/248 for s48 runs, 149/249 for s49
  runs) — distinct from the frozen-side training seed to avoid coincidental
  shared exploration.
- Fresh weight init (no warm-start).

## 5. Pre-launch operations

### 5.1 Snapshot frozen opponents (before any kill)

Before stopping s48 or s51, copy the M2550 (≈5.22 M step) checkpoints of s48
and s49 into a read-only snapshot directory:

```
results/BR_frozen/s48/{pursuer,evader}/   ← from results/SP17b_s48/checkpoints/{pursuer,evader}/milestone_phase2550_*
results/BR_frozen/s49/{pursuer,evader}/   ← analogous
```

The choice of M2550 is deliberate: it is the **last completed micro-phase
before drift onset** — i.e. the cohort's *claimed NE* snapshot, not the
post-drift endpoint. Testing the post-drift endpoint would conflate "is the
NE claim true?" with "did the drift damage the policy?".

If the M2550 phase number differs slightly between seeds, use the latest
phase where `p_full > 0.05` for each seed individually and document the
chosen phase per seed.

### 5.2 Stop drifting / collapsed runs (only after snapshot complete)

- s48 (niro-2): `kill -SIGTERM <pid>` (graceful so final history.json flushes)
- s51 (niro-2): same

s49, s50, ABL-1, ABL-2 keep running unchanged. The BR test does **not**
depend on s50 finishing; if s50 later converges differently, run BR-5/BR-6
as a follow-up.

### 5.3 Adapt BR training script

`scripts/train_pursuer_vs_evader.py` exists but uses PPO. Create a new
`scripts/train_br_sac.py` that:

- Accepts CLI args: `--frozen_opponent_path`, `--frozen_role` ∈ {pursuer,
  evader}, `--total_steps`, `--eval_freq`, `--n_eval_episodes`, `--seed`,
  `--output_dir`.
- Loads the frozen opponent as a fixed `PartialObsOpponentAdapter` (same
  pattern used in the cohort training script). The frozen side never updates.
- Trains a fresh SAC learner for the opposite role with cohort-identical
  hyperparameters.
- Writes `history.json` in the same schema as the cohort
  (`{"history": [{"phase": "BR<step>", "capture_rate": ..., ...}]}`) so the
  existing analyzer can reuse most logic.

### 5.4 Smoke test

Before launching all four runs in parallel, smoke-run BR-1 for 50 K steps
with eval every 25 K. Acceptance criterion: the eval CR is within 0.10 of
the s48 baseline (0.31). A wildly different number (e.g. CR=0.9 in 50 K
steps) means either the frozen evader didn't load, or `p_full` is wrong, or
the reward scaling is off. Halt and debug before parallel launch.

## 6. Launch protocol

### 6.1 Compute allocation

- niro-2 (after killing s48, s51 — frees 2 GPU slots):
  - BR-1 (fresh pursuer vs frozen s48 evader)
  - BR-2 (fresh evader vs frozen s48 pursuer)
- niro-1 (after killing ABL-2 if necessary — ABL-1 keeps running):
  - BR-3 (fresh pursuer vs frozen s49 evader)
  - BR-4 (fresh evader vs frozen s49 pursuer)

If niro-1 has only one free GPU slot at launch time, run BR-3 first then
BR-4 sequentially; total wall time becomes ≈9 h instead of 5 h.

### 6.2 Launch command shape

```
./venv/bin/python -u scripts/train_br_sac.py \
    --frozen_opponent_path results/BR_frozen/s48/evader \
    --frozen_role evader \
    --total_steps 1500000 \
    --eval_freq 250000 \
    --n_eval_episodes 200 \
    --seed 148 \
    --output_dir results/BR_1
```

(Three analogous invocations for BR-2/3/4.)

Per project rules in CLAUDE.md, training is launched on niro-2 / niro-1 from
this repo's path equivalent on each remote, not from the local working dir.

## 7. Analysis pipeline

After all runs finish, `scripts/analyze_exploitability.py` (new) computes,
per BR run, the `exploit_gap` defined in §3, then applies the two-step
classification (per-seed L1–L4 → cohort H1–H5). Output:

- A verdict table with one row per BR run showing baseline CR, BR best CR
  (max for pursuer side, min for evader side), `exploit_gap`, per-seed
  label, and finally the cohort hypothesis label.
- A printed framing recommendation (one of A defended / B / C / C with
  seed-sensitivity).
- Output format mirrors the per-seed analyzer at
  `/tmp/seed_analysis/analyze.py` so the analysis can be eyeballed quickly.

The script also writes a one-page memo `docs/worklogs/2026-05-02_BR_verdict.md`
with: the four numbers, the per-seed labels, the cohort hypothesis, and the
resulting framing choice.

## 8. Mapping outcomes to paper framing

| Hypothesis (from §3) | Framing | Abstract phrasing change | Figure 1 content |
|----------------------|---------|--------------------------|------------------|
| H1 (cohort at ε-NE) | **A defended** | Replace "0.51 ± 0.03" with the corrected number; add "verified by best-response exploitability test (gap ≤ 0.05 across seeds)". | CR trajectory + horizontal NE line at the corrected value. |
| H2 / H3 (one-sided exploit, replicated across seeds) | **C** | Reframe as: "we identify a curriculum-induced exploitability hole at the `p_full = 0` transition and characterize it across seeds". | CR trajectory + per-seed BR exploitability curve (rising above baseline) for the exploitable side. |
| H4 (two-sided exploit) | **B** (failure analysis) | Reframe as: "self-play with our reward shaping reaches an empirical fixed point that is not a Nash equilibrium under full partial observability; we characterize the failure mode". | Both pursuer-side and evader-side BR curves rising above their respective baselines. |
| H5 (seed-dependent NE) | **C with seed-sensitivity headline** | Reframe as: "NE quality is seed-sensitive: across two replication seeds, only one converges to an ε-Nash policy pair under our protocol; we identify the conditions under which the curriculum produces an exploitable hole". | Side-by-side per-seed CR + BR curves, showing one seed at NE and one not. |

Decision is final once all four BR runs complete and the analyzer runs. No
re-litigation without new evidence (e.g. extended-budget BR runs per §11).

## 9. Timeline

```
t=0       snapshot opponents, write train_br_sac.py, kill s48 + s51  (≈30 min)
t=0.5h    smoke test BR-1 50K steps                                   (≈10 min)
t=0.7h    launch BR-1..BR-4 in parallel
t=5.7h    all runs complete (1.5M @ ~6 step/sec ≈ 4.5 h wall)
t=6.7h    run analyze_exploitability.py + write verdict memo          (≈1 h)
t=7h      framing decision committed
```

If smoke test fails, slip ≈2 h for debugging. If niro-1 has only one slot,
slip ≈4 h for sequential BR-3/BR-4.

## 10. Out of scope

This spec deliberately does NOT cover:

- Writing the paper sections (separate spec, downstream).
- Re-running ablations against a corrected NE baseline (downstream — the
  ablations as launched are valid regardless of which framing wins).
- Investigating *why* the curriculum-end drift happens. That is interesting
  but follow-up work; framing C accommodates it without requiring a fix.
- The s48 best-model checkpointing regression (D14 not enforced — only
  milestone snapshots, no `best_model_*/` dir). Flagged as a separate
  operational issue; does not block this test.
- Running BR for s50 once it converges. Defer until s50 has crossed the
  curriculum boundary; if it shows the same drift signature, run BR-5/BR-6
  as a robustness extension.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| BR script bug — fresh side never trains | Medium | Smoke test catches this. Acceptance: BR side's policy entropy decreases by t=50K. |
| Frozen opponent loads in *training* mode (gets updated) | Medium | Unit-test in script: hash opponent params before/after one train step; assert unchanged. |
| 1.5 M steps insufficient for BR to find an exploit | Low–Medium | If `br_cr_curve` is still rising at t=1.5 M (slope > 0.02 per 250 K), extend to 2.5 M for that run only. Document the extension. |
| s48 baseline 0.31 is itself contaminated (drift not stationary) | Medium | Recompute baseline at test time using last 8 m-evals available; if std > 0.05, use the median instead of mean. |
| Smoke test passes but parallel runs OOM | Low | niro-2 has 4 SAC slots already validated; killing s48 + s51 leaves 2 free; safe. niro-1 has been running 2 SAC concurrently. |
| Frozen-snapshot phase choice is wrong (drift starts earlier than M2550 for some seed) | Low | Per §5.1, pick the latest phase with `p_full > 0.05` per seed and document. |
| s49 / s50 finish drifting and "look fine" before BR launches, weakening the case | Low | Run BR test regardless — the question is exploitability, not visible drift. |

## 12. Definition of done

This spec is "done" when:

- All 4 BR runs have completed 1.5 M steps (or extension if §11 triggered).
- `scripts/analyze_exploitability.py` has produced a verdict.
- `docs/worklogs/2026-05-02_BR_verdict.md` records the hypothesis + framing choice.
- The S88 worklog references this spec and links to the verdict memo.

After done, the *next* spec (downstream) is "rewrite paper abstract +
sections 1, 4, 5 to match Framing {A,B,C}".
