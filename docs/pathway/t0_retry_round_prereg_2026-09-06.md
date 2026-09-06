# The learner question: approach and pre-registration of the retry round (2026-09-06, S122)

**Status:** pre-registered BEFORE implementation and before any run. Spec for
`docs/superpowers/plans/2026-09-06-t0-retry-round.md`. Governing decisions: D83
(pre-registration discipline), D95 (A2 movement condition, A4 β = 0.1 pre-commitment),
D98 (Tier 0 rulings: collapse not sparsity; P-B preserved-undetermined; direction table's
last row; β = 0.1 does not consume the discretionary slot).

## Part A — the goal and what Tier 0 says about it

Goal (user, 2026-09-06): an RL-trained pursuer–evader **pair** with a novelty claim.

Tier 0 (S121–S122) established, on the pursuer against a frozen scripted evader:
- the reward signal exists — 184 captures per 1M steps under the clone (T0.1);
- every ONLINE learner that started from the competent clone either **died** (free SAC
  0.195 → 0.020; dense-reward SAC the same; PPO-BC 0.180 → 0.035 → 0.095 → **0.140,
  still rising at the 200k cutoff**) or **froze** (KL-anchored SAC at β = 1.0 held
  0.205–0.215 with final anchor divergence 0.00033 < 0.001: preserved, not learning);
- the harness discriminates (control collapsed to 0.010, Wilson upper 0.036).

So the open question is not "can a pursuer learn" but **"what mechanism lets a policy
improve on a competent prior without the critic dragging it off a cliff."** Answer that
and the pair follows (Part D); fail to, and no self-play machinery helps.

## Part B — the approach in three tracks

| track | what | gate |
|---|---|---|
| **1. The retry round** (this document, Part C) | one pre-registered round, three mechanism arms + one control, pursuer only, to find a learner that IMPROVES on the clone | ≥ 1 arm reaches IMPROVED |
| 2. The evader | clone `FovLimitedEvader` onto a **belief-augmented** observation (D79: a memoryless clone cannot imitate a memory-driven teacher; `pe2/belief.py` supplies the last-seen dead-reckoning belief), gate it the D76 way, apply the Track-1 mechanism | evader clone passes its battery; mechanism improves it |
| 3. The pair | both roles anchored to their clones with the Track-1 mechanism, co-adapting in self-play; **shielded vs unshielded** (shield v2, D97) as the R1 measurement | plan v3 Step 8 |

**Where the novelty sits.** Paper 1 showed unanchored self-play locks in (4/34 balanced,
structural ∀/∃ asymmetry). Paper 2 would show that co-adaptation *anchored to competent
priors*, under a hard shield that provably never lets either body touch anything, yields
a stable pair — and then measure what the shield does to that equilibrium, the claim no
one has tested (D90). Anchored co-adaptation is modest as a method; answering one's own
prior negative result is how second papers are written. Tracks 2 and 3 get their own
plans AFTER Track 1 decides; nothing in them is pre-registered here.

## Part C — pre-registration of the retry round ("T0.4")

### C1. Why a round, and why these three
The direction table (`derisk_plan_2026-09-04.md`, Tier 0, last row) permits **one more
pre-registration** before Route C. Tier 0 itself was one pre-registration with three arms
and a control, so a *round* is the honest unit. β = 0.1 was pre-committed as A4 (D95)
before any pilot ran, so its inclusion consumes nothing; the round's discretionary content
is the family-independent arm the table itself names (offline advantage-weighted
regression on the clone's rollouts). PPO at a longer budget is included because it is the
only Tier-0 arm with positive momentum at cutoff; it is a NEW arm at a NEW budget, run
fresh from the same start, not an extension of a seen result (D83).

### C2. Common protocol (binding)
- Role: pursuer. Opponent: frozen `GreedyEvader`. Eval: deterministic, paired seeds
  **780000+** (the Tier 0 probe stream, so every probe is paired with Tier 0's).
- Budget: **1,000,000 env steps per arm**. Probes at every **100,000** steps, N = 200.
- **Final probe**: at 1,000,000 steps, N = **500** (seeds 780000–780499; a superset of the
  200-seed probes). Power: Wilson half-width ≈ 0.035 at p ≈ 0.2 (plan v2 shared power).
- **Pass rule (no collapse)**: win rate ≥ **0.137** (the gated clone's Wilson lower bound,
  `CLONE_CI_LO`) at EVERY probe from **500,000** steps onward — the second half of the
  budget, the same shape as Tier 0's rule (from 100k of 200k). A missing probe in the window → `UNDETERMINED` with the reason worded (a crashed run is neither a pass nor a collapse).
- **Improvement rule**: the final N = 500 probe's Wilson **lower** bound exceeds the arm's
  own START point estimate. Starts are measured, not tuned: the gated SAC clone is
  **37/200 = 0.185** (D76; interval [0.137, 0.245]) for the SAC-family arms; the PPO-BC
  clone is **41/200 = 0.205** (Tier 0, `arms.PC.bc_start`). At N = 500 this needs
  ≥ 110/500 (0.220) against 0.185 and ≥ 121/500 (0.242) against 0.205 (computed with `pe2.eval.wilson_ci`, z = 1.96).
- **Verdict per arm** (ordered): `COLLAPSED` (fails the pass rule) < `PRESERVED` (passes,
  no improvement) < `IMPROVED` (passes AND improves). `UNDETERMINED` only when a
  pre-registered precondition is unmet (control not collapsed; start inadequate), always
  with a worded reason. **Survivor = IMPROVED.**
- Training seeds: SAC-family arm **7** (A1: the off-policy seed), PPO **9** (its Tier 0
  seed), AWR **11** and its control **11** (paired; they differ only in the advantage sign).
- Learner recipe for the SAC-family arm is D80/D81 binding (LN critics, `ent_coef = 1e-5`,
  critic warm-up 50k, D78 phases). PPO kwargs are the Tier 0 `PPO_KWARGS`, unchanged.
- No budget extension, no criterion change, no seed re-roll after any number is seen.

### C3. The arms
| arm | mechanism | start | specifics |
|---|---|---|---|
| **R-B** | KL-anchored SAC fine-tune, **β = 0.1** | gated SAC clone | `finetune_br_anchored`, identical to P-B except β. Its first 200k IS the A4 pre-commitment. Reports the A2 diagnostic (`anchor_kl_grew`, final divergence vs 1e-3) alongside the verdict; the verdict itself is decided by the pass and improvement rules above, not by the divergence. |
| **R-C** | PPO-BC → PPO+GAE, **1M** | `results/derisk/ppo_bc_pursuer.zip` (the Tier 0 PPO-BC clone, start 41/200) | `finetune_ppo`, seed 9, same kwargs. Its 50k/100k/150k/200k probes are expected to REPRODUCE Tier 0's 36/7/19/28 of 200 — a free determinism check, recorded. |
| **R-D** | **Advantage-weighted regression on Monte-Carlo returns** (Peng et al. 2019), offline on the clone's rollouts, then iterated online | gated SAC clone (37/200) | New module `pe2/awr.py`. **No bootstrapping anywhere**: V(s) is fit by regression to `G_t = γ^(L−t)·R` (the `mc_returns` already in `pe2/derisk.py`, γ = 0.999); the actor (the clone's own SAC actor, inside its SB3 model so every probe/eval path is unchanged) is updated by weighted regression of `tanh(μ(s))` onto the executed action with weights `exp((G_t − V(s)) / λ)` clipped at `w_max`. **λ = 0.5, w_max = 20** (AWR's clip; λ scaled to our ±1 return range), fixed now. Data: initial buffer = 300 stochastic clone episodes (seeds **820000+**, ≈ 300k steps, counted in the budget — this is the "offline" phase); then iterations of ≥ 50k fresh steps from the current stochastic policy, FIFO buffer of the last 300 episodes, value net (obs→256→256→1, Adam 1e-3, 20 epochs) then actor (Adam 3e-4, 10 epochs) per iteration. Chosen over IQL because IQL's expectile targets are still TD targets and the diagnosed mechanism is TD drift. |
| **R-D-neg** (control) | R-D with the advantage **sign flipped** (`exp(−(G − V)/λ)`) | same clone, same seed 11, seeds **830000+** | 200k budget, probes at 100k and 200k. **Must collapse** by `control_collapsed` (final Wilson upper < 0.137). It validates that AWR's non-SB3 training loop and its probe path measure what they claim, and that the weighting — not the regression — does the work. If it does NOT collapse, R-D's verdict is **withheld** (UNDETERMINED, reason worded), exactly as Tier 0 withheld its arms behind CTRL. R-B and R-C run inside SB3's own loop, which Tier 0's CTRL already validated on this probe machinery; they need no new control. |

Seed audit: 700000–790999 and 800000–810999 are consumed (Steps 5–6, Tier 0); tests use
900000+. **820000+ and 830000+ are fresh.**

### C4. What happens next, fixed now
- **≥ 1 IMPROVED** → the IMPROVED arm with the highest final Wilson lower bound is the
  Track-1 mechanism; Track 2's plan is written for it. Two IMPROVED arms → both run in
  Track 2 as arms, the loser becomes the ablation (Tier 0 direction-table logic).
- **0 IMPROVED, ≥ 1 PRESERVED** → the anchored/AWR *stabilisation* is reported as a
  method finding (it prevents the D82 collapse), but an RL-trained pair that IMPROVES on
  its priors is not reachable at this budget with these families → **Route C**
  (characterisation paper: shield guarantee D97, exact-game shift T3, and a learning
  failure now independent of reward shaping, algorithm family, anchoring, and
  bootstrapping). This is the table's own "if that fails: Route C".
- **R-D-neg does not collapse** → R-D withheld; the round is decided on R-B and R-C alone,
  and the AWR loop is debugged as infrastructure, not re-run as an arm.
- Stop rule: nothing is re-run or extended after a number is seen. A second retry round
  requires a new user ruling.

### C5. Artefacts
`results/derisk/t0b_arm_{RB,RC,RD,RDNEG}.json` (per arm, flushed per probe),
`results/derisk/t0b_round.{json,png}` (summary + one figure: all arms' curves on the
Tier 0 axes with the 0.137 line and each arm's improvement bar), checkpoints under
`results/derisk/ckpts_{ARM}/`. Visual gate: the figure, signed in the S122 worklog with
the per-arm verdicts. Runtime estimate from Tier 0 (≈ 1 h per 200k SAC-arm incl. probes):
R-B ≈ 5–6 h, R-C ≈ 3 h, R-D ≈ 1 h, R-D-neg ≈ 20 min — run as four concurrent processes
(24 cores, RTX 5090 idle) → ≈ 6 h wall.

## Part D — after the round (outline, NOT pre-registered)
- **Track 2 (evader).** `BeliefObsWrapper` appends the `LastSeenBelief` estimate (believed
  opponent dx, dy in body frame; staleness) to the Step-2 observation; BC `FovLimitedEvader`
  onto it (300 demos, D75 recipe); gate against the D76 battery; then the Track-1 mechanism
  with the evader's own start as the improvement bar. Pre-registered separately.
- **Track 3 (pair).** Alternating or simultaneous self-play, each role anchored to its
  clone by the Track-1 mechanism; the pursuer under shield v2 in one arm and unshielded in
  the other; R1 measured as the paired shift in the best-response landscape (plan v3
  Step 8 / 8-alt). Pre-registered separately, with paper 1's 4/34 lock-in rate as the
  comparison the stability claim is made against.
