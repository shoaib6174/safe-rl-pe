# S1 — structured search: RL learns the search sub-policy under a fixed intercept law (2026-09-12, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), D82 (single-step TD at γ = 0.999 cannot
form a calibrated value estimate here), D106/D107 (the searching teacher and its clone),
D108 (R1: the collapse is independent of the prior — recorded when R1's final probe lands).
Design numbers and the R1 autopsy are in `docs/worklogs/2026-09-12_S125.md`.

## 1. Why this round exists

R1 (`requalification_prereg_2026-09-12.md`) fine-tuned the 0.458 searching clone with the
plainest SAC configuration and it collapsed exactly as the 0.178 clone did: 0.005 at 100k,
0.000 from 500k, both arms. The autopsy of RQ-NEW's checkpoints on the clone's own states:

| checkpoint | Q(clone action) | Q(drive backward) | Q(random) | policy |
|---|---|---|---|---|
| 100k | +3.44 | +3.11 | +3.28 | v ≈ 0, 54 % of steps reversing |
| 200k | +6.23 | +6.18 | +6.21 | |
| 500k | +21.52 | +21.49 | +21.50 | v = −0.93, 97 % reversing, log-std −14 |

The true value of any state lies in [−0.30, +1.0]. The critic is 3–21× outside that range,
growing monotonically, and its action-dependence is at the noise floor (0.03 on a scale of
21). The actor maximises noise, `ent_coef = 1e-5` lets it saturate at a box corner, and once
it drives backward every episode is a timeout and nothing pulls it back. **This is D82's
mechanism re-measured**: one-step TD at γ = 0.999 amplifies any per-step critic bias by
1/(1−γ) = 1000×; LayerNorm capped the divergence at ~20 instead of 10⁹ without removing it.

Every SAC fine-tune since D82 (T0.4 PA/PB, T1.0 HC, R1) re-ran a value estimator D82 had
already shown cannot work, and each collapse was recorded as new evidence about learning in
this game. They are repeated measurements of one defect in one component. What R1 legitimately
adds is narrow: the prior is not the cause. **Whether RL can improve a competent pursuer here
is still untested**, because no learner has been run from the competent prior whose update
direction does not come from that critic.

The failing component is credit assignment over ~1000 steps from a ±1 terminal. This round
changes *what has to be credited*. The teacher's chase and last-seen branches are a five-line
feedback law that provably captures (D106); they stay fixed. RL learns only the **search
sub-policy** — what to do when the scripted law has nothing to act on.

**Design numbers** (N = 200, seeds 1 000 000–1 000 199, chase + last-seen law fixed, only the
search branch swapped; this stream is now consumed):

| search branch | capture | found evader | steps to first sight (median / p90) | search stretch (median / p90) |
|---|---|---|---|---|
| orbit (`SearchingPursuer`) | **0.490** | 0.94 | 51 / 207 | 195 / 316 |
| random, held 10 steps | 0.335 | 0.70 | 50 / 852 | 419 / 1200 |
| random, every step | 0.300 | 0.62 | 328 / 1075 | 1028 / 1200 |
| stand still (`FovLimitedPursuer`) | 0.180 | 0.51 | 16 / 55 | 1092 / 1200 |
| drive straight | 0.065 | 0.20 | 1 / 23 | 1200 / 1200 |

The search branch alone spans 0.065–0.490 with the capture law untouched, and a search stretch
is ~200 env steps: with one decision held for 10 env steps that is ~20 decisions per stretch,
a horizon a policy-gradient learner can credit from realised returns.

**Hypothesis S:** with the intercept law fixed and decisions taken at the search level, PPO
from scratch learns a search policy that beats the orbit rule.
**Hypothesis C (control, option 1):** a learner whose update direction comes from realised
returns rather than a bootstrapped critic improves the competent prior on the *unstructured*
per-step problem. If C holds, the structure is not what makes learning work here.

## 2. The structured pursuer

`pe2.structured.SearchMDP` — a `gym.Env` wrapping `SingleAgentWrapper(cfg, "pursuer", opp)`.

- It owns the teacher's chase and last-seen branches **verbatim** (`SearchingPursuer.act`
  minus the fallback) and their `last_seen` state. Whenever the evader is visible, or
  `last_seen` is set and farther than `LAST_SEEN_TOL`, the wrapper acts with that law
  internally and the learner is not consulted.
- The learner is consulted only when neither branch applies. Its action (Box [−1, 1]², scaled
  exactly as `SingleAgentWrapper._scale_action`) is **held for `HOLD = 10` env steps**, ending
  early if the evader becomes visible or the episode ends. The wrapper then runs the scripted
  law until it again has nothing to act on, and returns to the learner.
- Observation: the unchanged 11-dim pursuer observation at the decision point.
- Reward per decision = the sum of env rewards over every env step since the previous decision
  (the held steps and the scripted stretch that follows) **plus `R_ACQ` × the number of
  acquisition onsets in that span**, where an onset is `obs[6]` crossing from ≤ 0.5 to > 0.5
  between consecutive env steps. `R_ACQ = 0.2` in arm SA, `0.0` in SB and SZ. No discounting
  inside a span: the learner's MDP is a semi-MDP with γ applied per decision.
- The wrapper exposes `env_steps` (cumulative underlying env steps) so budgets and probes are
  in env steps, and reports `info["outcome"]` from the underlying env unchanged.
- Episode seeds follow `SingleAgentWrapper`'s stream (`seed0 + episode_count`) unless a seed
  is passed to `reset`.

**Validity check V1 (wrapper).** Plugging the orbit rule in as the "learner" (one decision per
env step, i.e. `HOLD = 1`) must reproduce `SearchingPursuer` **episode-for-episode**: identical
outcomes and episode lengths on all 500 bar seeds. This is a unit test and a pre-flight check
in the runner; a mismatch halts the round before any arm runs.

Evaluation: deterministic PPO actions inside `SearchMDP`, frozen `GreedyEvader` at
`EnvConfig()` defaults, seeds `EVAL_SEED0 + i`, logs in `evaluate_sb3_policy`'s schema so
`win_count` / Wilson / McNemar apply unchanged. The evaluated quantity is always the
**unshaped capture rate**.

## 3. Protocol

- Role: pursuer. Opponent: frozen `GreedyEvader`, `EnvConfig()` defaults (as every prior round).
- Eval stream **1 010 000+** (fresh; ≤ 1 000 999 consumed). Training seed **7** (wrapper
  `seed0 = 7000`, as every prior arm).
- Budget **1 000 000 env steps** per arm. Probes at every **100 000 env steps** (the first
  decision boundary at or after each multiple; the recorded step is the true env-step count),
  N = 200 paired seeds; final probe at ≥ 1 000 000 env steps, N = **500**.
- Structured learner: SB3 `PPO("MlpPolicy")` from random init, **γ = 0.99** per decision,
  every other hyper-parameter at the SB3 default (`n_steps = 2048` decisions, `batch_size = 64`,
  `n_epochs = 10`, `gae_lambda = 0.95`, `ent_coef = 0.0`, `learning_rate = 3e-4`). No prior:
  the orbit rule is the bar to beat, not the thing to imitate.
- Control learner: `train_bc_ppo` clones `SearchingPursuer` from 300 demonstration episodes
  (`collect_demonstrations`, seeds 1 020 000+, the T2 recipe) onto PPO with `PPO_KWARGS`, then
  `finetune_ppo` on the unstructured per-step problem with `PPO_KWARGS` **except
  `gae_lambda = 1.0`**. The deviation from T0.4's 0.95 is deliberate and is the point of the
  control: at λ = 0.95 the advantage is still bootstrapped beyond ~20 steps, and C is a
  hypothesis about a learner that does not bootstrap.

**Bars, measured on 1 010 000+ at N = 500 BEFORE any arm trains**, written to
`results/derisk/s1_start.json` (an arm that cannot read it refuses to run):

| bar | what | role |
|---|---|---|
| `BAR_ORBIT` | `SearchMDP` + orbit rule | **the bar**: the best scripted search, = `SearchingPursuer` by V1 |
| `FLOOR_RAND` | `SearchMDP` + uniform random action held 10 steps (rng seed 0) | the floor: what the fixed law achieves with no search competence |
| `START_S` | `SearchMDP` + the untrained PPO init (seed 7) — the exact start of SA/SB/SZ | the structured arms' own start |
| `START_CP` | the PPO clone on the per-step problem | the control's own start |

**Validity check V2 (control prior).** `START_CP`'s Wilson lower bound must be ≥ **0.35** (the
SAC clone measured 0.458/0.476 on two streams). Below that the PPO clone is not the competent
prior, and CP is run but its verdict is reported as `UNDETERMINED (weak prior)`.

## 4. The arms

| arm | learner | reward | purpose |
|---|---|---|---|
| **SA** | PPO on `SearchMDP`, from scratch | terminal ±1 **+ 0.2 per acquisition** | **the test** (primary) |
| **SB** | PPO on `SearchMDP`, from scratch | terminal ±1 only | density ablation: did the event reward matter? |
| **SZ** | PPO on `SearchMDP`, from scratch | **all rewards zeroed** | **negative control** (D83): the same learner and wrapper with no signal must not learn |
| **CP** | PPO-GAE λ = 1.0 from the PPO clone, per-step | terminal ±1 | **option-1 control**: does a non-bootstrapping learner improve the competent prior without the structure? |

SA, SB and SZ share the same init (seed 7) so `START_S` is literally each one's step-0 policy.
Four arms run concurrently.

The acquisition bonus cannot be farmed: the learner acts only while the evader is invisible,
an onset ends its control, and losing the evader is the scripted law's doing, not the learner's.
Re-acquisition after a genuine loss is rewarded again, which is the intended signal.

## 5. Verdicts (fixed now)

For SA and SB, on the final N = 500 probe (`lo` = Wilson lower bound, `hi` = upper):

- **IMPROVED** iff `lo(arm) > BAR_ORBIT` point estimate.
- else **LEARNED** iff `lo(arm) > max(hi(FLOOR_RAND), hi(START_S))`.
- else **FAILED**.

Harness validity: **SZ must not be LEARNED**. If it is, the structured arms' verdicts are all
`UNDETERMINED (harness)` — something other than the reward is moving the policy.

CP: `pe2.derisk.round_verdict` unchanged (≥ 0.137 at every probe from 500k; IMPROVED iff final
`lo > START_CP` point), exactly as R1 judged its arms, so CP is comparable to every prior
fine-tune curve.

A crashed arm is `UNDETERMINED` with its probe history preserved.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **SA IMPROVED; CP collapses or preserves** | Hypothesis S confirmed, C refuted: the structure is what makes learning work. Route C withdrawn. The paper becomes *learned search under a fixed intercept law with a hard CBF shield*; next pre-registration adds the shield to the structured pursuer. SB tells whether density was needed. |
| **SA IMPROVED and CP IMPROVED** | Both hold: a non-bootstrapping learner suffices on the per-step problem. The unstructured learned pursuer is the simpler artefact; the structure is reported as an ablation, not the contribution. Route C withdrawn either way. |
| **SA LEARNED, not IMPROVED** | The search MDP is learnable but the orbit is hard to beat from an 11-dim memoryless observation. The next question is the search policy's *observation* (coverage / staleness memory, `SearchStalenessTracker` exists), not the learner. Route C re-scoped to "per-step fine-tuning"; not withdrawn. |
| **SA FAILED (SZ valid)** | PPO cannot learn even a 20-decision search MDP at this budget. The negative result is, for the first time, about the game and not about a critic — Route C stands on a stronger footing and D82/D100/D102/D103 are re-scoped as "one-step TD" results. CP's outcome is reported alongside. |
| **SB ≥ SA** | The event reward added nothing; report and prefer SB as the simpler arm. |
| **SZ LEARNED** | Harness invalid; nothing else is read. |
| **V2 fails (weak PPO clone)** | CP reported with the caveat; structured verdicts unaffected. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round
is seen. `HOLD`, `R_ACQ`, γ and the PPO defaults are fixed by this document.

## 7. What this round does NOT do

No shield, no evader family, no belief state for the learner (the wrapper's `last_seen` is the
scripted law's state, not the learner's), no second learner family, no action-hold sweep. The
hidden-state defect (D79/D105) remains unfixed for the learner and is the named next constraint
in the LEARNED-not-IMPROVED row.

## 8. Artefacts

`results/derisk/s1_start.json`, `s1_arm_{SA,SB,SZ,CP}.json` (flushed per probe),
`s1_round.{json,png}`, checkpoints under `results/derisk/ckpts_{ARM}/`, the PPO clone at
`results/derisk/ppo_bc_searching_pursuer.zip`. Figure: four curves in env steps, `BAR_ORBIT`,
`FLOOR_RAND`, `START_S` and `START_CP` as reference lines, the 0.137 floor and the 500k pass
window for CP. Signed in the worklog. Estimated wall time ≈ 4 h with the four arms concurrent.
