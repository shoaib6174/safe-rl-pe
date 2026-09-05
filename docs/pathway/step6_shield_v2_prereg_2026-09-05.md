# Step 6 shield v2 — design and pre-registration (2026-09-05, S122)

**Status:** pre-registered BEFORE implementation and before any v2 run. This document is
the spec for `docs/superpowers/plans/2026-09-05-step6-shield-v2.md`. The criteria below
are copied verbatim from the S119 pre-registration; nothing in them changed. What changed
is the shield's design (Part B) and the seed streams (fresh, because the v1 streams were
consumed by the failed run). Anything not written here is not allowed after the run.

## Part A — what the v1 run showed (evidence, S122)

The v1 gate run (`results/visual_gates/step_06/step_06_gate.json`, fingerprint
`b72fe438…`, commits `94c9690…2db8555`) failed its kill criteria: G1 519 violation
steps / 16 episodes, G3 72 / 17, G5 56 + 1283. The T4 census attributed the failures to
(A) nanometre wall grazing below the solver tolerance and (B) first-order linearization
error feeding a relaxed tier that drops obstacle rows. A reset audit run today shows the
census was incomplete. The root causes, in order of weight:

| id | root cause | evidence |
|---|---|---|
| RC1 | **Reset states lie outside the barrier set.** The DCBF only guarantees forward invariance of {h ≥ 0}; `PursuitEvasionEnv._sample_free_pose` only keeps the *body* free, and the VCP is 0.1 m ahead with a uniform heading. | 2000 resets (seeds 800000+): **29 (1.45 %) have a VCP barrier < 0 at step 0 (δ = 0)**; 52 (2.6 %) have the body inside the δ = 0.05 set. G1 had 16 offending episodes in 1000 — the same rate. Census: `shielded_probe` first offence at **step 3** with h = −0.0088 on the *exact* tier; `robust_main` first offence at **step 1** with h = −0.203; `robust_probe` at step 2. |
| RC2 | **VCP–body gap.** h_vcp ≥ 0 does not imply body clearance. At the surface (‖b−p‖ = r_eff) the VCP barrier is non-negative for any heading with e·(b−p) ≥ −d/2, i.e. up to arccos(−d/(2 r_eff)) from the outward normal (≈ 96° for r_eff = 0.45). Moving forward from such a heading penetrates the body ~3 mm per step while h' ≥ (1−γ)h still holds. | Analytic (worked example in Part B). `shielded_main` has 32 obstacle contacts on the exact tier with p50 penetration 1.2 cm. |
| RC3 | **Relaxed tier drops obstacle rows** (boundary-only), so once RC1/RC2 make the exact program infeasible there is no obstacle protection at all. | `robust_probe`: 1255 of 1283 contact steps on the relaxed tier, parked at penetration ≈ δ. |
| RC4 | **Feasibility tolerance drift.** `FEAS_TOL = 1e-9` accepts rows violated by < 1e-9 each step, so h can sit slightly negative and the body grazes a wall at 1e-10 m. | 479 of G1's 519 steps (Mechanism A). |
| RC5 | **First-order linearization** of h(x_{k+1}); neglected term d·(ω dt)²/2 ≈ 1 mm/step at ω_max. Real, but only bites once RC1/RC2 have already put the state at the surface. | T4 census (Mechanism B), reproduced. |

The v1 verdicts stand as recorded (kill criteria failed, no criterion changed). This
document is the redesign the user ruled for on 2026-09-05 ("plan according your
recommendation").

## Part B — shield v2 design

**B1. Two rows per surface.** For each of the 4 walls and each obstacle the shield carries
- a **hard body row**: `h = x_lim − x` (walls, metres) and `h = hypot(b − p) − r_eff`
  (obstacles, metres). These are *exactly* the env's own contact predicates
  (`_advance_agent` clips on `x > x_max`; `_project_out_of_obstacles` fires on
  `hypot < r_obs + r_agent`), so `h ≥ 0 ⟺ no contact`, bitwise, at δ = 0.
- a **soft VCP row**: the existing virtual-control-point barrier (walls: `x_lim − q_x`;
  obstacles: `‖q − p‖² − r_eff²`), which supplies steering (relative degree 1 in ω).

**B2. Exact one-step check, zero tolerance.** A candidate action u satisfies row i iff
`h_i(euler_step(x, u)) ≥ (1 − γ)·h_i(x)` evaluated on the true Euler successor, using a
single `pe2.env.euler_step` function shared with the environment (so the prediction is
the env's next state, bit for bit). No linear row is ever the arbiter; `FEAS_TOL` is
removed. Linearized rows are used only to *enumerate candidates* (D85 enumeration:
nominal, projections, pairwise intersections, line∩box-edge, box corners), plus the two
candidates `u = 0` and the clipped nominal, always.

**B3. Three tiers, nothing dropped.**
| tier | condition | selection |
|---|---|---|
| `exact` | some candidate satisfies every row | minimise the weighted distance to the nominal (unchanged objective, w_v = 150, w_ω = 1) |
| `relaxed` | no candidate satisfies every row, some satisfies every hard row | among hard-feasible candidates maximise the worst *normalised soft-row progress*; ties → nearest to nominal |
| `backup` | no candidate satisfies every hard row | maximise the worst normalised hard-row progress, then the worst soft-row progress (so a body stuck heading at a surface still turns its VCP out); ties → nearest to nominal |

Progress of row i under u is `(h_i(x⁺) − (1−γ) h_i(x)) / σ_i` with `σ_i = 1` for every
metre-valued row and `σ_i = 2 r_eff` for the obstacle VCP row (m² → m near the surface).
The brake-and-turn `backup()` heuristic, `BACKUP_V_FRAC` and `HEAD_ON_TIE_TOL` are
deleted. Tier names are unchanged so `ShieldMetrics`, `summarize_shield` and the gate
JSON keep their shape; `pinned` (≥ 40 consecutive backup steps) now means "persistent
hard-row infeasibility", which is exactly the diagnostic the robust arm needs.

**B4. Guarantee (δ = 0).** (i) Every reset satisfies every hard row, because the env only
samples body-free poses and clips to the arena. (ii) `u = 0` leaves every hard row
bit-identical (`x + 0.0·c·dt = x`; hard rows do not depend on θ). (iii) Hence when every
hard row is ≥ 0 the exact or relaxed tier finds a hard-feasible candidate and chooses one;
the chosen action gives `h_i(x⁺) ≥ (1−γ) h_i(x) ≥ 0` for every hard row, computed by the
same code the env will run. (iv) By induction every hard row stays ≥ 0 at every step,
and hard-row ≥ 0 is the env's own no-contact predicate. So the shielded role produces
**zero env contact steps, in floating point, for any nominal policy and any opponent**.
The backup tier is unreachable at δ = 0. For δ > 0 the guarantee holds from any reset
that satisfies the inflated hard rows (97.4 % of resets); the rest enter the backup tier
and are *measured*, not assumed.

**B5. What this costs.** More candidates (≈ 130 per step with 12 rows) evaluated
vectorised; sliding along obstacles becomes tangent-only (a body row with a_ω = 0 binds
speed, the VCP row supplies the turn), so interventions and stall/deadlock rates may rise
and G2 may tighten. Those are the reported quantities; G2 is the gate that decides.

## Part C — pre-registration v2 (criteria verbatim from S119; seeds fresh)

- **G1 (kill criterion):** shielded `GreedyPursuer` vs `GreedyEvader`, N = 1000 seeded
  episodes, seeds **800000+** → 0 violation steps. Any violation → shield math wrong; stop.
- **G2 (capture intact):** shielded capture rate ≥ 0.5 × unshielded capture rate on the
  same paired seeds; paired McNemar reported as the primary comparison with effect size
  (descriptive, no p-threshold).
- **G3 (adversarial probe):** `ObstacleHuggingEvader` vs shielded `GreedyPursuer`,
  N = 1000, seeds **810000+** → 0 violations; timeout rate reported.
- **G4 (negative control):** the identical probe WITHOUT the shield → episodes with ≥ 1
  violation must be ≥ 5 %. If fewer, the PROBE is too weak and gets fixed, never the gate.
- **G5 (robust-margin arm, D87):** the same shield with every constraint radius and
  boundary limit inflated by δ = 0.05 m, gated on the G1 and G3 criteria, plus its
  intervention rate reported against the unmargined arm's.
- **Violation definition — UNCHANGED:** an env contact step (wall or obstacle flag) of the
  shielded role, magnitude-blind. The contact-depth census already in the gate runner
  (`n_above_1e-6_m`, `n_above_1e-3_m`, …) is **reported as a secondary description and
  is never a gate**. This is stricter than the tolerance the S122 recommendation floated;
  B4 says the strict definition is attainable, so it is the one we test.
- **Reported, never gated (D89):** infeasibility rate, tier histogram, `min_clearance`
  floor (metres), stall / immobile / deadlock / pinned rates with run-length
  distributions, intervention rate; **new:** per-episode `start_in_barrier_set`
  (all rows ≥ 0 at reset) and `start_hard_feasible` (all hard rows ≥ 0 at reset), so RC1
  can be separated in the report without re-running.
- **Seeds:** 760000–760999 and 770000–770999 are consumed by v1 and are not reused.
  800000+/810000+ are unused by every pe2 script (audited 2026-09-05: 700000–790999 are
  all taken).
- **Stop rule:** any G1 or G3 violation → the run stops, the census is written, and no
  third design iteration starts without a new user ruling. A G5 failure is reported and
  blocks the hardware-margin claim (R4) only.
- **Visual gate:** `step_06_shield.gif` re-rendered from episodes of the NEW streams,
  with C1 re-derived from the exact one-step check (`CBFShield.progress`), C2–C4 as
  before; episode choice is documented in the JSON and every rendered number is
  cross-checked against the gate row (unchanged mechanism).
- **Before the run:** the v1 artifacts move to
  `results/visual_gates/step_06_v1_failed_2026-09-05/` so both runs stay on disk.

Nothing else may change between this commit and the run.
