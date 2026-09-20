# S15 — Phase 1: why does learning not escape the loop? (2026-09-20, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any measurement of this round exists.**
Governing: **D125** (the correction), **D126** (escape needs a rotate-then-move manoeuvre; not our
discretisation), **D127** (the loop survives exploration; training bought ~a tenth of the immobility away
despite an immobility penalty), **D113** (a stall penalty alone recovered nothing, 0/3).

## 1. The question

A reviewer's first question about the rebuilt paper is: *the escape is one rotation away and the agent had
five million steps — why did it not learn it?* This round answers that with measurements, not argument.
It is evaluation and instrumentation only. No policy is trained.

Four candidate explanations, each with a falsifiable prediction fixed here. **Two of them are predicted to
FAIL** and serve as controls; a round in which everything I expect to hold holds, and everything I expect to
fail fails, is a round that could have gone the other way.

## 2. Measurements and predictions

**M1 — credit assignment (predicted to HOLD).** Over episodes of FT-B-41 and RC19 against the learned
attacker ATK4-45 and against `GreedyEvader`, identify every **stall entry** (the first step of a run of ≥ 10
immobile steps). Classify each by whether the agent escaped within 40 steps. Report, for escaped vs not:
the eventual capture rate, the mean steps remaining, and the **discounted return-to-go at the entry step** at
the training γ = 0.99.
> **P1: the escape advantage is negligible — |Δ discounted return-to-go| < 0.05, and the capture rate
> following an escape is < 0.15.** If P1 holds, the gradient signal for escaping is essentially absent and
> credit assignment is the binding constraint.

**M2 — does the observation actually freeze? (predicted to HOLD, in a specific form).** During stall runs
measure per step: the L2 change in the observation vector, the L2 change in the emitted action, and whether
the evader is visible. **The naive claim "the observation is constant" may be false — the evader keeps
moving — so this measurement is stated to be able to refute it.**
> **P2: the emitted action is near-constant during stall runs — median per-step action change < 0.01 on the
> [−1, 1] action space — whether or not the observation changes.** Reported alongside: the observation change
> and the visibility fraction, so that "frozen input" and "frozen map" are distinguished rather than conflated.

**M3 — entropy collapse (predicted to FAIL; control).** Compare the policy's action σ and entropy at stalled
states against non-stalled states, same episodes.
> **P3: σ at stalled states is NOT materially lower — ratio to non-stalled σ > 0.8.** If P3 holds, entropy
> collapse is *not* the explanation, which is the useful negative. If it fails, the story changes to one about
> exploration dying exactly where it is needed.

**M4 — commitment to held actions (predicted to FAIL; control).** The structured policy commits each decision
for `hold = 10` env steps. Re-evaluate the identical policies at `hold = 1`, same seeds.
> **P4: reducing the hold does not materially reduce immobility — absolute change < 0.05.** If P4 holds,
> commitment is not the explanation. If it fails, the hold is implicated and the fix section gains an arm.

**Safety:** 0 pursuer contact steps in every cell, as always.

## 3. Protocol

- Policies: `FT-B-41` (`ckpts_FT-B-41/FT-B-41_5000000.zip`, shield_obs) and `RC19`
  (`ckpts_RC19/RC19_10000000.zip`, shield_obs). Opponents: `ATK4-45` (learned, structured) and `GreedyEvader`.
- N = **200** episodes per (policy, opponent) cell, stream **1 710 000+**, CBF shield at defaults, `hold = 10`
  except in M4 where the same cells are repeated at `hold = 1`.
- Stall entry = the first step of a run of ≥ 10 consecutive immobile steps; escape = ≥ 5 consecutive steps of
  `v_exec > 0.1 · v_max` beginning within 40 steps of the entry. Both thresholds fixed here.
- γ = 0.99 and the terminal reward (+1 capture, −1 timeout) are the training values; return-to-go is computed
  from the actual episode outcome and the steps remaining.

## 4. What each outcome fixes

| result | direction |
|---|---|
| **P1 and P2 hold, P3 and P4 fail** | The predicted picture: no gradient signal to escape, and a near-constant emitted action. The paper's mechanism section is these two measurements, and the fix must supply either the signal (dense escape reward) or the state (memory / escape features) — Phase 2's arms are then well motivated. |
| **P1 fails** (a real advantage exists) | The signal was there and learning still missed it; the explanation is optimisation, not credit assignment, and Phase 1 must continue before Phase 2 is worth running. |
| **P2 fails** (the action varies) | The policy is *trying* different things and still not escaping; the constraint is the filter, not the policy, and the paper's balance shifts back toward the filter's objective. |
| **P3 fails** (entropy collapses at stalls) | A simpler and more familiar story — exploration dies exactly where needed — which must then be reported as the primary mechanism. |
| **P4 fails** (hold matters) | Commitment is implicated; Phase 2 adds a hold-length arm and the claim narrows to temporally-committed policies. |

**Stop rule.** Nothing is re-run, re-thresholded or re-seeded after any number of this round is seen. The
thresholds (10 immobile steps for an entry, 40 for the escape window, 5 moving steps for an escape, 0.05 and
0.8 for the prediction bars) are fixed above.

## 5. Artefacts

`results/derisk/s15_cells/*.json` (per-cell records with the per-stall-entry table), `s15_round.{json,png}`;
runner `scripts/derisk_s15_mechanism.py` (`--cells | --summarize`), run from a frozen worktree.

## Amendment S15-A1 (2026-09-20, before any measurement of this round exists)

**M3 cannot be measured as written, and that is the answer to M3.** The policies are SB3 `ActorCriticPolicy`
with a `DiagGaussianDistribution` and `use_sde=False`, i.e. a **single learned `log_std` shared across all
states** (FT-B-41: `[-0.8162, -1.3118]`). The runner verifies this empirically rather than assuming it, by
recording the spread of σ over every observation seen (`sigma_spread`, expected exactly 0). P3's verdict is
therefore **`N/A (state-independent sigma)`**, and the useful statement is stronger than the failed control I
predicted: **entropy collapse at stalled states specifically is impossible for this architecture** — σ is
identical at stalled and non-stalled states by construction, so exploration cannot be dying *where it is
needed* while surviving elsewhere. The hypothesis is excluded, not merely unsupported.

**Consequence for §4.** Its first row ("P1 and P2 hold, P3 and P4 fail") can never be selected as written.
It is replaced by: **"P1 and P2 hold, P4 fails, P3 N/A"** — the same direction, with M3 read as the
architectural exclusion above. Every other row stands. If a future round uses a state-dependent σ (a learned
`log_std` head or `use_sde=True`), M3 becomes askable again and must be re-pre-registered.

## Amendment S15-A2 (2026-09-20, before any measurement of this round exists)

Three defects found in review, all in how M2 is measured. Fixed here, before any number.

**1. The observation delta was measuring the agent's own clock, not the world.** `obs_deltas_median` is taken
over the full observation, which includes the three S5 `shield_obs` features — one of which is the stall-run
counter, ticking by exactly 1/40 every immobile step. A probe showed the raw 11 world dimensions move by
median 0.00000 while the full vector moves by 0.02500, i.e. the reported number was the counter and nothing
else. **The record must carry the raw-world delta and the shield-feature delta separately**, and the
write-up must quote the raw-world one when it says whether the world stands still.

**2. P2's median bar is structurally satisfied at `hold = 10`.** A held decision emits an identical action for
ten steps, so the per-step median change is 0 by construction and P2 cannot be refuted in those cells. A probe
found 8 nonzero changes in 1056 stalled steps, with a maximum of 0.0221 — above the 0.01 bar. **P2 is
therefore evaluated on the `hold = 1` cells, where the median is not forced, and reported for `hold = 10`
alongside as the degenerate case.** The bar itself (median per-step action change < 0.01) is unchanged; only
which cells decide it is fixed here. The record must additionally carry, report-only, the **maximum** and the
**nonzero fraction** of per-step action changes, so that "the policy never varies" and "the policy varies
rarely but really" are distinguishable.

**3. `direction()` still contains the row A1 removed.** The first §4 row must read "P1 and P2 hold, P4 fails,
P3 N/A"; the code predates A1 and still requires P3 to FAIL, which A1 made impossible, so the round's own
predicted outcome currently selects no row.
