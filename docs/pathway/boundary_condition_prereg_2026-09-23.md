# S18 — the boundary condition: when does penalising filter intervention help?

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: D83 (pre-registration discipline), D125 (the correction), D129 (the penalty backfired),
D132 (the phenomenon is prior art), `docs/retrospective_2026-09-23.md` §4 (the construct gate),
`docs/pathway/ral_plan_boundary_condition_2026-09-23.md` (the plan this executes).

## 1. The disagreement this round settles

**Pizarro Bejarano et al.** (arXiv:2410.11671) report that penalising safety-filter interventions during
training **improves** the learned policy. **D129 (ours)** found the opposite across three seeds: the penalty
**backfired**, teaching the agent to stop asking to move — which satisfies the penalty perfectly and destroys
the task.

**Hypothesis B.** The two results differ because of where the task reward has gradient, not because of the
penalty. When the task reward varies across the actions available *inside the region where the filter
intervenes*, the penalty is a perturbation on an informative gradient and it helps. When the task reward is
constant there, the penalty is the only gradient on that set, so the policy optimises the penalty rather
than the task, and the penalty's minimiser is the do-nothing action.

## 2. Why this is a 2x3 and not the 2x2 the plan named

The plan wrote the independent variable as "reward density". **That is the wrong variable and I am fixing it
before any number exists.** A reward can be dense everywhere and still be flat on the intervened set: `w_vis`
pays per step for seeing the evader, and inside a stalled pocket the evader is not visible, so `w_vis` is
identically zero across every admissible action there. Density and gradient-on-`I` come apart, and only the
second is what Hypothesis B claims.

So the columns are three, and the middle one is the discriminating cell:

| column | reward | dense per step? | varies across actions on `I`? |
|---|---|---|---|
| **SPARSE** | terminal +/-1 and `r_acq` only (the S16/S17 setting) | no | no |
| **DENSE-FLAT** | `+ w_vis` (`DenseRewardWrapper`) | **yes** | **no** — the evader is invisible in the pocket |
| **DENSE-VARY** | `+ w_search` (staleness; rotation changes the visible cells) | yes | yes |

Rows: **penalty OFF** / **penalty ON**. Six cells, 3 seeds each, **18 runs**.

## 3. The penalty must be THEIR penalty

D129 penalised *stalled steps* (`SearchMDP.stall_penalty`). Pizarro Bejarano et al. penalise the *magnitude
of the filter's correction*, every step, defined everywhere. **A reviewer kills the contradiction instantly if
our penalty is not theirs**, so this round implements theirs:

`r <- r - lambda * ||a_nominal - a_safe||_2`, per env step, summed into the learner's span reward.

`stall_penalty` stays in the code untouched and is **not** an arm of this round; D129's result is quoted as
the prior observation this round explains, not as a cell.

**`lambda` is calibrated, not chosen.** Before any training: roll out the frozen BASE policy for 200 episodes
on stream 1 900 000, record the mean per-episode sum of `||a_nominal - a_safe||`, and set `lambda` so that
quantity equals **1.0**, the magnitude of the terminal reward. Written to `results/derisk/s18_lambda.json`.
**An arm that cannot read that file refuses to run.** One `lambda`, shared by all three penalty cells, so the
column comparison is not confounded by penalty strength.

## 4. Protocol

- Role: pursuer, `SearchMDP` (hold=10), shield ON, `shield_obs` ON, `escape_obs` ON, RecoveryShieldV2 OFF
  (the recovery is D131's remedy and would mask the effect under test).
- Budget **5,000,000** env steps per arm; training seeds **70, 71, 72**; probe every 500 000.
- Gate: stream **1 900 000+**, N = **500** paired episodes, every arm on the same episodes.
  Probe stream **1 950 000+**. Both fresh (1 800 000 is the highest previously consumed).
- Opponents at the gate: `GreedyEvader` (the task benchmark) and `StructuredEvaderPolicy` (robustness).
  The learned attacker family is deliberately absent — this round is about reward structure, not adversaries.
- Primary metric: **capture rate**. Mechanism metrics, pre-registered as required reading, not optional:
  deadlock rate, fraction of policy-requested actions that are zero-motion, mean intervention magnitude.

## 5. Hypotheses, with directions fixed now

- **H-CROSS (primary).** A crossover interaction: penalty **lowers** capture in SPARSE and **raises** it in
  DENSE-VARY. Pass requires the sign to hold on **>= 2/3 seeds in each direction**, with the SPARSE drop and
  the DENSE-VARY gain each significant by `paired_indicator_test` at p < 0.05 pooled.
- **H-MECH (the discriminating test).** DENSE-FLAT patterns with **SPARSE**, not with DENSE-VARY: the penalty
  lowers capture there too, on >= 2/3 seeds. **This is what separates gradient-on-`I` from density.**
- **H-ZERO (mechanism).** In every cell where the penalty lowers capture, the zero-motion request fraction
  **rises** relative to its matched no-penalty control. This is the claim that the agent wins the penalty
  while losing the task; without it, H-CROSS is a correlation.
- **H-SAFE (standing).** Zero pursuer contact steps across all 18 runs.

**Refutation, stated plainly.** No interaction (H-CROSS fails) refutes Hypothesis B and the paper does not
exist in this form; we stop and reconsider rather than re-cut the data. H-CROSS passing while **H-MECH fails**
(DENSE-FLAT behaves like DENSE-VARY) means the variable is density after all, not gradient: the result is
real but the proposition planned for S20 is wrong as drafted and gets rewritten to match.

## 6. Construct gate — BLOCKING, run before any outcome is read

The retrospective's §4 rule. Four of our five collapses would have been caught here.

- **CG1 (the column labels are true).** At 150 intervened states sampled from real episodes, measure the
  variance of the per-step task reward across the admissible action set, per reward variant. Require
  `SPARSE ~ 0`, `DENSE-FLAT ~ 0`, `DENSE-VARY > 0`. **If DENSE-FLAT is not flat on `I`, the column is
  mislabelled and the columns are renamed before any capture rate is looked at.**
- **CG2 (the metric is not satisfiable by inaction).** Confirm capture rate cannot be obtained by a policy
  that never moves, and that no mechanism metric is satisfied by the remedy's own mechanism (the D129 error).
- **CG3 (the signal exists).** `signal_nondegeneracy` on each reward variant's rollout stream.
- **CG4 (behavioural).** Render 3 unselected episodes per cell and watch them, before reading the table.

## 7. Stop rule

Nothing is re-run, extended, retuned or re-seeded after any number of this round is seen. A crashed arm is
`UNDETERMINED` with its probe history preserved. Amendments are permitted only before the number they touch
exists, and each is appended here with its timestamp.

## 8. Artefacts

`results/derisk/s18_lambda.json`, `s18_construct_gate.json`, `s18_arm_{CELL}_{SEED}.json`,
`s18_round.{json,png}`, checkpoints under `results/derisk/ckpts_S18-{CELL}-{SEED}/`, GIFs under
`results/derisk/s18_gifs/`. Launched from a frozen worktree at this commit with `PYTHONPATH` pointed at it.
Estimated wall time ~30 h for 18 arms at the usual concurrency.
