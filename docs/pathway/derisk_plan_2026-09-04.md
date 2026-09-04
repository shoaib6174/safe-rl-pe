# De-risk Plan v2 — Tiered, Pre-Registered, Direction-Setting (2026-09-04, S121, D91→D92)

**Supersedes v1 (same day)**. v1 inherited "PPO+GAE" from D84 without analysis. The analysis (S121) found that every Step-5 arm ran on a terminal-only ±1 reward, while this project's own lessons say a per-step visibility reward is mandatory under partial observability (D20: 0.57 vs 0.10 capture) and a search-staleness reward is the missing search skill (D24). The learning-failure conclusion (D68–D82) therefore rests on runs that omitted the strongest prior evidence in the repo. v2 inserts a diagnostic tier that decides *what* to try before anything is trained at scale.

**Rules for every tier** (D83): hypothesis, pass/fail numbers, N with a power rationale, a negative control that must fail, budget, all written **before** the run; seeded paired frozen-opponent eval (D55/D57), Wilson CIs (D56), best-checkpoint (D14), paired McNemar primary; results in `results/derisk/`; a visual gate per tier signed in the worklog. **No budget extension or criterion change after a result is seen.**

**Power (shared)**: N = 500 paired episodes → Wilson half-width ≈ 0.035 at p ≈ 0.2, so a 5-point delta resolves on the CI alone and paired McNemar has more power; N = 1000 within 0.03 of any threshold.

---

## Tier 0 — Diagnose the learning signal before choosing a learner (CPU + small GPU, ≈ 3 days)

### T0.1 Signal audit (CPU, ½ day)
Roll out the gated SAC clone (0.185) for 1M env steps against the frozen scripted evader and record: positive terminals per 1M steps; episode-length distribution of captures; the fraction of steps with the evader visible; and the empirical variance of the Monte-Carlo return per step. Do the same for the D79 evader clone (0.015) in its role.
- **Output**: `results/derisk/t0_signal_audit.json` + `t0_signal_audit.png` (return-variance vs step index).
- **Pre-registered kill rule for the whole Tier 1**: if positive terminals < 50 per 1M steps for the pursuer **even after T0.2 adds dense reward**, the bottleneck is exploration, no policy-gradient variant fixes that at this budget, and the direction goes straight to **Route C** (§ Decision tree) without spending Tier 1's GPU.

### T0.2 Reward audit against the lessons (CPU, 1 day)
Implement, as options on `SingleAgentWrapper`, the three proven dense terms from the root chassis, each checked for plateaus on the relevant state region the way lessons §5b requires (a dense reward with a flat region is rejected before use):
- **Pursuer visibility reward**: `+w_vis` per step the evader is inside the FOV cone and unoccluded, `w_vis ∈ {0.1, 0.2}` (D20; root `w_vis_pursuer`). Plateau check: reward-vs-bearing curve must be monotone into the cone.
- **Search-staleness reward**: `w_search × mean staleness of observed cells`, `w_search = 1e-4` (D24; root `SearchStalenessTracker`; D4: ≥ 3e-4 teaches patrol-not-hunt). Plateau check: staleness must be non-constant along a random trajectory.
- **Evader visibility reward** (Mode B): `w × (+1 hidden, −1 visible)` — **known plateau** (§5b: flat −1 in open space), so it ships only with the PBRS obstacle-seeking potential that fixed it. Plateau check: PBRS potential must have a nonzero gradient at ≥ 90 % of sampled open-space states.
- Every term must be **potential-based or per-step bounded** so terminal outcomes are unchanged; eval is always on true outcomes (D68).

### T0.3 Three mechanism-matched pilots (GPU, 200k steps each, 1 seed, ≈ 1 day)
Each pilot starts from the gated SAC clone and uses LN critics + fixed `ent_coef = 1e-5` (D80/D81, binding). Pass criterion for a pilot: **no collapse below the clone's 0.185 CI-lo (0.137) at 100k and 200k**, evaluated on 200 paired seeds — i.e. the pilot has to beat D82's signature, not beat the teacher.
- **P-A "change the signal"**: SAC fine-tune with T0.2's visibility + search rewards. Targets D68–D74 (flat from scratch) and the sparse-signal half of D82.
- **P-B "stop the drift"**: SAC fine-tune, terminal-only reward, plus a KL penalty to the frozen clone (β swept over {0.1, 1.0}) — the RLHF-style anchor. Targets D82's mechanism directly (actor leaves the clone manifold under a drift-dominated Q).
- **P-C "change the horizon structure"**: PPO-BC clone → PPO fine-tune with GAE λ = 0.95, γ = 0.999 (the D84 retry). Targets the TD-bootstrap residual. Requires the PPO-BC path (~1 day build).
- **Negative control**: the naive SAC fine-tune from D78 re-run on the same 200 seeds — must collapse (< 0.137 by 200k). If it does not, the pilot harness is broken.

**Tier 0 output**: a ranked list of survivors (0–3) and the T0.1 numbers. **Visual gate**: `t0_pilots.png` — capture-rate curves of P-A/B/C + control on one axis, clone CI-lo as a horizontal line.

### How Tier 0 changes direction
| T0 result | Direction |
|---|---|
| T0.1 kill rule fires (< 50 positives / 1M even with dense reward) | Skip Tier 1. **Route C** (characterization paper). Learning failure reported as exploration-bound, algorithm-independent |
| P-A survives, P-B/C do not | Tier 1 runs on the dense-reward chassis. The paper's Methods inherit the root chassis (D20/D24) explicitly; D68–D82 are reframed as "terminal-only reward is insufficient", not "learning is impossible" |
| P-B survives (with or without P-A) | Tier 1 uses the KL-anchored fine-tune; the paper gains a small methodological point (imitation prior + anchored RL under sparse adversarial reward) |
| Only P-C survives | Tier 1 = the D84 plan as originally written; the result is algorithm-family-specific and must be reported so |
| Two or three survive | Tier 1 runs the **best two** at full budget as arms; the loser becomes the ablation |
| None survive but T0.1 shows > 50 positives / 1M | Signal exists, no learner uses it → one more pre-registration allowed (advantage-weighted regression / IQL, offline on the clone's rollouts). If that fails: **Route C** |

---

## Tier 1 — Establish learned agents (GPU, 2–3 weeks)

### T1 Pursuer (blocking)
**Hypothesis**: the Tier-0 survivor, run at full budget, produces a pursuer that *learns* — not merely preserves the clone.
- 3 seeds × 2M steps; checkpoints every 100k; D14 selection; opponent = frozen scripted evader (FLe) as in Step 5.
- **Negative control**: the same learner from a **random** init, 1 seed, same budget — must fail the D76 gate.
- **Pass** (all three, ≥ 2/3 seeds): passes the D76 battery; beats the SAC clone (0.185) by paired McNemar p < 0.05 on 500 paired seeds; **beats its own BC start** by paired McNemar p < 0.05.
- **Reported**: D70 scenario suite — any capture in `occluded_start`/`shadow_adjacent` (clone: 0/30) is the first learned search.

### T2 Evader (blocking)
**Hypothesis**: an evader conditioned on the Step-4 `LastSeenBelief` (last-seen position, velocity, time-since-seen ⊕ 11-D obs) is Markov enough to be cloned from FLe and improved by the Tier-0 survivor, with the evader-side dense reward from T0.2 (visibility + PBRS) if P-A survived.
- (a) belief-conditioned clone; (b) fine-tune, 3 seeds × 2M vs the frozen T1 pursuer (or the SAC clone if T1 is not done); (c) RecurrentPPO clone+fine-tune, 1 seed, **only if (a) fails**; (d) **negative control**: the D79 memoryless clone on the same seeds — must stay ≈ 0.015.
- **Pass**: (a) escape ≥ 0.15 CI-lo on 500 paired seeds (≈ ⅔ of the teacher's 0.223); (b) beats (a) by paired McNemar p < 0.05 in ≥ 2/3 seeds.
- **Fail**: (a) < 0.10 CI-hi **and** (c) < 0.10 CI-hi.

### How Tier 1 changes direction
| T1 pursuer | T2 evader | Route |
|---|---|---|
| pass | pass | **Route A — strong paper**, plan v3 as written: shielded self-play (R1), learned shield-attacker (R2), shield-aware training (R3), two robots (R4) |
| pass | fail | **Route B-P — asymmetric**: learned pursuer vs an evader **family** (FLe, belief-clone, speed ratios 1.0–1.4, hugging probe). R1 measured on the best-response landscape, not self-play; R2 = learned pursuer exploiting a *shielded evader* family; R3 on the pursuer only; R4 unchanged |
| fail | pass | **Route B-E — mirror**: learned evader vs a pursuer family. R2 = learned evader as shield-attacker (the more natural direction: the evader has the incentive to bait the pursuer's shield). R1 on the BR landscape; R4 unchanged |
| fail | fail | **Route C — characterization paper**: exact-game shift (T3) + shield gates (T4) + hardware (R4) on cloned/scripted policies; the algorithm-family-independent learning failure (terminal-only *and* dense reward, SAC *and* PPO, anchored *and* free) is a reported finding with paper-1 rigor |

---

## Tier 2 — Establish the two direction-independent contributions (CPU, run alongside Tier 0–1)

### T3 Exact-game shield-shift twin (2–3 days)
Extend `scripts/experiment_1_tabular_forall_exists.py`: hazard cells; a **projection shield** = "a move into a hazard is replaced by the nearest non-hazard move", applied to both players and, separately, to the pursuer only (the asymmetry confound of plan v3 Step 8). Compute exact Nash value and exact BR-dynamics fixed points for unshielded / both-shielded / pursuer-only, on three layouts (sparse, medium, corridor).
- **Pass**: |V_shielded − V_unshielded| > 0.02 on ≥ 2/3 layouts; the **sign is recorded as the R1 direction prediction** before any deep run.
- **Null is publishable**: "projection shields are benign in the exact game" *contra* Jia et al.'s assertion (D90).
- **Negative control**: hazards placed where no policy ever moves → shielded value equals unshielded to numerical precision.
- **Visual gate**: `t3_exact_shift.png`.

### T4 Step-6 gates (2–3 days)
`scripts/pe2_step06_gate.py` + `ObstacleHuggingEvader` + robust-margin arm (D87). G1 0 violations/1000 (seeds 760000+); G2 shielded CR ≥ 0.5× unshielded, McNemar reported; G3 0 violations vs the hugging probe (770000+); G4 unshielded probe ≥ 5 % episodes-with-violation (else fix the probe); G5 robust-margin arm passes G1/G3. Rules from the S120 review: shielded and control sweeps in separate `summarize_shield` calls; control-arm stall/deadlock/pinned are structurally zero; G1's Wilson bound from `eval.wilson_ci`. **Visual gate**: `step_06_shield.gif`.

### How Tier 2 changes direction
| T3 | T4 | Effect |
|---|---|---|
| shift > 0.02 | pass | R1 has a theory prediction with a sign; the system half is validated; **every route above keeps a theory + system contribution regardless of Tier 1** |
| null | pass | R1 becomes "shields are benign in the exact game; is the same true under deep learning and PO?" — still a first test of D90's claim; Jia et al. become the counter-position |
| any | **fail** | Stop. A shield that violates G1/G3 is a Step-6 bug, not a result; nothing downstream runs until fixed and the gate re-run on fresh seeds |

---

## Tier 3 — Direction commit and Steps 8–12

Only after Tier 0–2 results are in the worklog with their visual gates signed:
- Apply the Tier-1 route table **as written**; record the route as a decision (D9x) with the numbers that selected it.
- Route A → plan v3 Steps 8–12 unchanged.
- Route B-P / B-E → plan v3 Step 8-alt (opponent family) replaces Step 8; Steps 9–12 proceed with the learned role; the paper's R1 sentence changes from "equilibrium shift" to "best-response-landscape shift", stated plainly.
- Route C → Steps 8–10 are dropped; Step 11 becomes the cohort of T3/T4/hardware; the learning failure is written up as §IV of the paper with the full refutation chain (D68–D82 + Tier 0–1).
- In every route, R4's wording carries all four D90 qualifiers (deep-RL, onboard-sensing/PO, model-free policy, seeded campaign).

---

## Calendar and parallelism

```
Week 1:   T0.1 audit (CPU ½ d) → T0.2 rewards (CPU 1 d) → T0.3 pilots (GPU 1 d) ; T3 + T4 start on CPU
Week 2–3: T1 + T2 full runs on GPU (survivor learner) ; T3 + T4 finish ; checkpoint evals as they land
Week 4:   decision meeting — Tier 1 table applied, route logged (D9x), Steps 8–12 re-planned to the route
```

## What moves the odds
Today: strong paper 40–50 %. P-A or P-B surviving Tier 0 → 55 %. T1 pass → 60 %. T1 + T2 pass → 70 %. T3 nonzero shift → +5 % on every route (theory contribution). T0.1 kill rule firing → Route C at ≈ 75 % publishable, ≈ 0 % strong.
