# Step 6 — pre-registered shield gates G1–G5: build, run, verdicts

**Date**: 2026-09-05 · **Config fingerprint**: `b72fe43819dd960d` (locked `EnvConfig` defaults)
**Artifacts**: `results/visual_gates/step_06/step_06_gate.json` (4.0 MB, all 6000 episodes),
`step_06_shield.gif`, `step_06_visual_gate.json`
**Commits** (on `main`, not pushed): `94c9690`, `a25b0f8`, `051727b`, `f154c54`, `bff74d5`, `2db8555`

---

## Headline

**G1 and G3 — the two kill criteria — FAIL.** The shielded pursuer makes env contact in
16/1000 episodes on the main stream and 17/1000 against the adversarial probe. Per the
pre-registration this stops Step 6: *"any shielded violation → shield math wrong; stop."*
Nothing was reinterpreted, no threshold moved. G2 and G4 pass. G5 fails on the same
criteria and fails worse.

The failure is **not** an artifact of the new probe: G1 runs the pre-existing
`GreedyPursuer` vs `GreedyEvader` matchup and fails on its own.

Two distinct mechanisms, separated by the violation census (below), demand different
fixes and are reported separately rather than pooled into one number.

---

## 1. What was built

### 1.1 `ObstacleHuggingEvader` (`pe2/scripted.py`) — the G3/G4 probe

Full-obs scripted evader (L1.5: baselines see truth). While the pursuer is within
`HUG_THREAT_RANGE = 7.0 m` it drives onto a hug circle `HUG_CLEARANCE = 0.05 m` outside
the nearest in-range obstacle's swept radius (`radius + agent_radius`) and orbits toward
the point diametrically opposite the pursuer, keeping the obstacle between the two. With
no obstacle within `HUG_RANGE = 4.0 m` of its surface it presses to `WALL_HUG_BAND = 0.35 m`
from the nearest arena wall and slides along it in whichever tangential direction opens
distance from the pursuer. Both modes are `tangent + RADIAL_GAIN · clip(error, ±RADIAL_CLIP) · normal`
(gain 2.0, clip 1.0 m), so the approach is a spiral that arrives already orbiting rather
than a head-on dive. Beyond the threat range it is a plain flee-away agent.

The probe's purpose is what it does to the *other* robot: a pursuer that beelines at an
evader glued to an obstacle or a wall drives itself into that obstacle or wall.

**Probe iteration required: none.** Measured before the gate ran, on 200 unshielded
episodes from the pre-registered probe stream (seeds 770000–770199): **74.5 %** of episodes
contained a pursuer contact step, against G4's 5 % floor — a 15× margin. The full N=1000
gate run then measured **74.4 %**. The first design passed; no behaviour was changed and
the 5 % threshold was never touched.

### 1.2 Robust-margin arm (`CBFShield(cfg, margin=δ)`) — G5

`margin` inflates every obstacle's effective radius and shrinks every boundary limit by δ,
so each barrier reaches zero exactly δ metres of clearance earlier. The backup tier honours
the same inflated limits, so the two tiers cannot disagree about where a wall is.
`margin` defaults to `0.0`, and a test asserts the default arm is bit-identical to an
explicit zero margin — the arm gated by G1–G4 is unchanged.

**δ = 0.05 m, provenance stated not tuned**: `docs/phases/phase4_sim_to_real_transfer.md`
requires localization error **< 5 cm at the 95th percentile** (`test_localization_accuracy`,
line 1606), and its **§8.7** prescribes *"increase safety margins to account for localization
uncertainty (add position_uncertainty to obstacle radii in CBF)"*. That is the
measurement-robust CBF construction D87 names (Cosner/Ames arXiv 2104.14030; Dean/Taylor
CoRL 2020). δ is the hardware bound, transcribed.

### 1.3 `scripts/pe2_step06_gate.py`

Six arms on the pre-registered streams (main 760000+, probe 770000+, N=1000 each; 6000
episodes, ~34 min single-core). Writes one JSON in which every number traces to a seed and
the config fingerprint: per-arm shield spec (class, γ, VCP d, weights, margin), per-episode
rows for all 6000 episodes, and the violation census.

The three S120 review notes are honoured structurally:

| Note | How |
|---|---|
| Keep shielded and control sweeps in separate `summarize_shield` calls | One call per arm; the aggregator is never handed a pooled sweep |
| Control arm's stall/deadlock/pinned are structural zeros | Both controls use `pe2.shield.NoOpShield` (never a hand-rolled pass-through); their stall / immobility / deadlock / pinning / backup / infeasibility fields are labelled `structurally_zero_fields` in the JSON with the reason in words |
| G1's Wilson bound comes from `pe2.eval.wilson_ci` | `violation_upper_bound()` calls it at z = 1.645 (one-sided 95 %); `summarize_shield` is never asked for a CI |

### 1.4 `ShieldOverlay` (`pe2/render.py`) + `scripts/pe2_step06_visual_gate.py`

Overlay added to the shared renderer, as Steps 2 and 4 did: the shield's own constraint
surfaces (margin included), the nominal action as a dashed grey unicycle arc against the
executed action as a solid coloured one, an intervention flash ring + caption, and a
readout of tier and reportable clearance in metres. It draws from a `live` dict the caller
fills with the **logged** filter output and never runs the shield itself — which is what
makes the flash rate and the logged intervention rate the same number by construction.

---

## 2. Gate verdicts

All arms: shielded role = pursuer; violation = an env contact step (wall clip or obstacle
projection) of the shielded role, per the pre-registered semantics.

### G1 — **FAIL** (kill criterion)
Shielded `GreedyPursuer` vs `GreedyEvader`, seeds **760000–760999**, N=1000.
- **519 violation steps** in 769 803 steps; **16/1000 episodes** with ≥ 1 violation.
- Criterion is 0 violation steps. One-sided 95 % Wilson upper on the per-episode violation
  rate = **0.0240** (`pe2.eval.wilson_ci`, z = 1.645). Had it been 0/1000 the bound would
  be 0.0027, matching the pre-registered "rule of three ≈ 0.003".
- Offending seeds: 760006, 760175, 760216, 760372, 760411, 760441, 760451, 760564, 760649,
  760672, 760782, 760872, 760878, 760881, 760949, 760979.

### G2 — **PASS**
Paired on the same seeds, control = `NoOpShield`.
- Shielded CR **0.428** ≥ 0.5 × unshielded CR 0.415 = **0.2075**.
- Paired McNemar (primary, descriptive, no threshold): **p = 0.0192**, b = 20 episodes the
  shielded pursuer captures and the unshielded does not, c = 7 the other way; ΔCR **+0.013**.
- The shield *helps* capture slightly: an unshielded greedy pursuer wastes time pinned
  against walls (552 577 control contact steps, 70.6 % of all its steps).

### G3 — **FAIL** (kill criterion)
`ObstacleHuggingEvader` vs shielded `GreedyPursuer`, seeds **770000–770999**, N=1000.
- **72 violation steps**; **17/1000 episodes**. Criterion is 0.
- Timeout rate (reported) **0.690**. One-sided 95 % upper on the violation-episode rate 0.0251.
- Offending seeds: 770052, 770081, 770089, 770140, 770317, 770327, 770340, 770386, 770479,
  770482, 770520, 770595, 770604, 770656, 770901, 770925, 770955.

### G4 — **PASS** (negative control)
Identical probe, `NoOpShield`, same seeds.
- **744/1000 = 0.744** episodes with ≥ 1 violation ≥ 0.05 required (481 708 violation steps).
- The probe is 15× stronger than the floor, so the G3 failure cannot be dismissed as a weak probe.

### G5 — **FAIL** (robust-margin arm, δ = 0.05 m)
G1's criterion on the main stream and G3's on the probe stream.
- main: **56 violation steps**, 5/1000 episodes, CR 0.422.
- probe: **1283 violation steps**, 10/1000 episodes, timeout 0.691.
- Intervention rate vs the unmargined arm (the additionally-required report):
  main **0.7881 vs 0.7064** (+0.0817); probe **0.4928 vs 0.3334** (+0.1594).
- Clearance floor: main −0.1500 m vs −0.0927 m unmargined; probe −0.1500 vs −0.0997.
  The −0.15 = −(VCP d + δ) is the expected signature.

---

## 3. The violation census — two mechanisms, not one

The gate replays every offending episode and measures how far past a surface the executed
step would have gone (replicating the env's clip-then-project arithmetic exactly, with the
obstacle overlap taken after the wall clip). This is diagnosis, not mitigation: the verdicts
above are decided by the violation count alone, upstream of any of it.

| arm | eps | steps | wall / obstacle | tier at contact | wall overshoot p50 / max (m) | obstacle penetration p50 / max (m) |
|---|---|---|---|---|---|---|
| shielded_main | 16 | 519 | 487 / 32 | exact 503, relaxed 16 | 5.8e-12 / 9.0e-3 | 1.2e-2 / 4.7e-2 |
| shielded_probe | 17 | 72 | 34 / 38 | exact 50, relaxed 16, backup 6 | 3.4e-3 / 9.8e-3 | 9.0e-3 / 5.0e-2 |
| robust_main | 5 | 56 | 5 / 51 | relaxed 45, exact 8, backup 3 | 1.6e-3 / 2.2e-3 | 3.0e-2 / 5.0e-2 |
| robust_probe | 10 | 1283 | 18 / 1265 | relaxed 1255, exact 14, backup 14 | 2.2e-3 / 3.9e-3 | 5.0e-2 / 5.0e-2 |

### Mechanism A — float-scale wall grazing (479 of G1's 519 steps)
On `shielded_main`, 479 of the 487 wall contacts have overshoot **below 1e-9 m** (p50
5.8e-12 m). Worked example, the first offending step (seed 760006, step 248, tier exact,
in the JSON in full): the pursuer sits at x = 4.849 999 999 562, i.e. 4.4e-10 m inside the
x_lim of 4.85, heading θ = 1.570 796 315, a hair below π/2. The VCP leads by d = 0.1 m, so
qx = x + 0.1·cos θ overshoots x_lim by 1.13e-9 and h = **−6.9e-10 m**. The DCBF row is then
violated by 5.4e-10 — **below the QP's `FEAS_TOL = 1e-9`** — so the nominal passes through
unchanged, the Euler step moves x by 1.3e-10 past the clip, and the env flags `wall_contact`.

This is the shield holding the body on the wall surface to within a nanometre while the
pre-registered violation definition ("the clip engages") counts a 1e-10 m correction as a
violation. Structurally the body cannot *cross* x_lim (crossing needs cos θ > 0, where the
VCP barrier is strictly tighter than the body limit), so this is a tolerance-scale artifact
— but it is a genuine failure **as pre-registered**, and both candidate fixes (the 1e-9
solver tolerance, or the magnitude-blind violation definition) touch a pre-registered
constant. Not mine to change.

### Mechanism B — real centimetre-scale penetration on the relaxed fallback tier
This one is substantive and is not a tolerance question.

- G1 also has **32 obstacle contacts** with penetration p50 **1.2 cm**, max **4.7 cm**, and
  8 wall overshoots above 1 mm (max 9.0 mm).
- G3's contacts are **all** above 3.3e-4 m: wall p50 3.4 mm, obstacle p50 9.0 mm, max 5.0 cm.
- `min_clearance` at contact reaches **−0.0997 m ≈ −d** on G3 and **−0.15 m ≈ −(d + δ)** on
  G5 — the VCP is a full lead-length inside the surface while the body is at it.

Chain: the DCBF row is a **first-order linearization** of h(x_{k+1}). At ω_max = 2.84 rad/s,
dt = 0.05 s and d = 0.1 m the neglected second-order term in the VCP position is of order
d·(ω dt)²/2 ≈ **1 mm per step** — exactly the scale of the millimetre overshoots. A barrier
driven slightly negative makes the exact QP infeasible (at v = ω = 0 every row reduces to
γh ≥ 0, so infeasibility ⟺ some h < 0 already). The **relaxed tier then drops the obstacle
constraints entirely** and keeps only the boundary, so the shield offers *no* obstacle
protection precisely when an obstacle barrier has gone negative, and the pursuer drives in
until the env projects it out.

G5 makes this dramatically worse rather than better: inflating the radii shrinks the
feasible set, so infeasibility rises **25×** on the probe (0.001525 vs 0.000055) and the
relaxed tier is entered 1261 times vs 18. Once there, the arm parks inside an obstacle —
1265 obstacle-contact steps over only 10 episodes (~127 consecutive steps per episode) at a
p50 penetration of 5.0 cm, which is δ. **The robust margin buys earlier warning at the cost
of far more frequent recourse to a fallback tier that has no obstacle protection at all.**

---

## 4. Reported, not gated (plan v3 + D89)

| arm | infeas rate | clearance floor (m) | tier histogram | stall rate | immobile rate | deadlock eps | pinned eps |
|---|---|---|---|---|---|---|---|
| shielded_main | 0.000030 | −0.0927 | exact 769780, relaxed 17, backup 6 | 0.00462 | 0.00462 | 0.0030 | 0.0000 |
| control_main | 0† | −0.1000 | none 783143 | 0† | 0† | 0† | 0† |
| shielded_probe | 0.000055 | −0.0997 | exact 887866, relaxed 18, backup 31 | 0.00002 | 0.00002 | 0.0000 | 0.0000 |
| control_probe | 0† | −0.1000 | none 887594 | 0† | 0† | 0† | 0† |
| robust_main | 0.000113 | −0.1500 | exact 775715, relaxed 47, backup 41 | 0.00154 | 0.00154 | 0.0010 | 0.0000 |
| robust_probe | 0.001525 | −0.1500 | exact 887802, relaxed 1261, backup 95 | 0.00002 | 0.00002 | 0.0000 | 0.0000 |

† **structurally zero, not measured** — `NoOpShield` returns the nominal unchanged and
solves no program, and the scripted nominal is always full speed, so these flags are
unsatisfiable by construction. Flagged as such in the JSON and readable only beside the
`none` tier.

`min_h_floor` (the mixed-unit QP margin, never a distance) is carried separately:
−0.1358 / −0.2198 / −0.1820 / −0.2198 / −0.2346 / −0.3360.

**Run-length distributions** (the D89 auditability requirement — the 40-step deadlock cut
is reported *on* the distribution, not as a hidden constant):

- `shielded_main` longest stall run per episode: 991 episodes at 0, 6 in 1–4, **3 at ≥ 320
  (max 1192)**; 3 episodes still stalled at the end → deadlock rate 0.0030. Those three are
  the Grover force-equilibrium signature: a shielded pursuer immobile for an entire
  1200-step episode. Nothing lands in the ambiguous 20–39 band, so the 40-step cut does not
  arbitrate any episode here.
- `robust_main`: 992 at 0, 7 in 1–4, **1 at ≥ 320 (max 1185)**, 1 stalled at end.
- Probe arms: max stall run 8 (shielded) and 4 (robust) — no deadlocks.
- Backup runs never reach 40 anywhere (max 9), so `pinned` is 0 on every arm.

---

## 5. Visual gate — SIGNED (on its own terms)

`results/visual_gates/step_06/step_06_shield.gif` — 544 frames in the file (589 submitted;
the encoder coalesces the HOLD_FRAMES tails), 20 fps, STRIDE = 1, three episodes drawn from
the pre-registered streams, each cross-checked line by line against its row in
`step_06_gate.json` (outcome, steps, violations, interventions — all matched):

| label | arm | seed | outcome | steps | violations | interventions |
|---|---|---|---|---|---|---|
| probe_slide | shielded_probe | 770083 | capture | 201 | 0 | 38 |
| probe_capture | shielded_probe | 770094 | capture | 189 | 0 | 38 |
| main_capture | shielded_main | 760045 | capture | 151 | 0 | 38 |

**GIF md5, two independent renders: `1fe0e9de0d9e4cf066d0432655ea671b` and
`1fe0e9de0d9e4cf066d0432655ea671b`** (second via `--md5-only`, which renders to a temp
path and deletes it). Deterministic.

### Checklist, with my judgment

**C1 — nominal vs executed arrows diverge exactly at constraint approach: PASS.**
Asserted, not just shown: on all 541 exact-tier steps (0 steps fell to another tier), every
flagged intervention had at least one violated DCBF row, re-derived from
`shield._dcbf_rows` — the same rows the QP solved — and 0 sub-threshold corrections were
found in the other direction. The largest clearance at which any intervention occurred is
**0.258 m**, inside the analytic bound of **0.410 m** (the largest clearance at which any
row can bind given the action box; derived in the script's docstring and hand-checked in
tests). *Visually*: at main_capture step 41 the dashed grey nominal arrow points straight
into the obstacle's dotted constraint circle while the solid red executed arc peels up and
over it, with the flash ring lit. Unmistakable. I sign this.

**C2 — slides along an obstacle boundary without contact: PASS.**
Longest contact-free run inside 0.30 m of an obstacle surface: **60 steps** (3.0 s,
main_capture), with 59 and 41 in the two probe episodes; all three episodes have zero
violations. *Visually*: at main_capture step 70 the pursuer rides the dotted constraint
circle at a reported clearance of **+0.006 m**, trail curving around the obstacle, tier
exact. This is the behaviour the gate is meant to show. I sign it — with the caveat that
these are episodes where the shield works, and §3 shows episodes where it does not.

**C3 — the pursuer still makes contact-capture: PASS.**
All three episodes end in capture at final distances 0.289, 0.264, 0.291 m, all ≤ the
0.30 m capture distance, i.e. bodies in contact. This item initially passed its assertion
while the GIF never *showed* a capture — action frames are drawn before their step, so each
episode ended on a frame titled "running". Fixed (`2db8555`): each episode now carries one
overlay-free terminal frame, and the final frame reads "step 151 / 1200 — capture" with the
two disks touching. I sign it.

**C4 — intervention flash rate matches the logged rate: PASS.**
STRIDE = 1 and the overlay flashes off the logged `info["intervention"]` alone, so the two
are the same number by construction and are asserted equal per episode:
**114 flash frames / 541 rendered frames = 0.21072088724584104**, identical to the logged
**114 interventions / 541 steps**. The terminal frames carry no overlay and add no flash.
I sign it.

**Signed with one boundary condition**: the visual gate certifies that the GIF demonstrates
the four pre-registered items. It is **not** evidence that G1/G3 pass — the three episodes
were selected for illustrative value and have zero violations by construction of the
selection. Step 6 as a whole is **not** signed off.

---

## 6. Covering tests

**416 pe2 tests pass, 0 fail** (56 of them new here: 360 before this session);
`ruff format` + `ruff check` clean on `pe2`,
`tests/pe2`, and the two new scripts (lint scoped, never the whole `scripts/` tree). Every
numeric expectation is hand-computed in a comment.

| file | new tests | covers |
|---|---|---|
| `tests/pe2/test_scripted.py::TestObstacleHuggingEvader` | 9 | truth access; the antipodal orbit direction on both sides; the inward spiral off the hug circle; wall slide inside and outside the band; the range and threat gates; and the G4 property in miniature (unshielded pursuer must contact, 20 seeds) |
| `tests/pe2/test_shield.py::TestRobustMargin` | 8 | default margin 0 and bit-identity with an explicit zero; inflated obstacle r_eff (h 0.75 → 0.6975, clearance 0.50 → 0.45); shrunk boundary limits (4.75 → 4.70); every clearance exactly δ smaller across both kinds; a state where the margined shield intervenes and the plain one does not (h 0.26 vs 0.21 against the 0.25 binding threshold); the backup tier's shared limits; the control keeps δ = 0 |
| `tests/pe2/test_step06_gate.py` | 25 | the pre-registered constants themselves (a criterion change becomes a failing test); config fingerprint stability/sensitivity; the Wilson bound at 0/1000 = 0.00269873; G2's verdict at, below and above the boundary; run-length distribution incl. exhaustive histogram buckets and the DEADLOCK_STEPS cut; the structural-zero labelling; `min_h` never published as a distance; `contact_geometry` incl. that obstacle overlap is measured from the **wall-clipped** point (0.10 m, not the 0.14 m the unclipped point gives); `distribution` |
| `tests/pe2/test_step06_visual_gate.py` | 14 | the analytic clearance bound (boundary-only 0.321 m; obstacle 0.4237576322 m; margin-shifted 0.4151725339 m; largest-over-constraints); `nominal_violates_a_row` against the filter with the tolerance asymmetry made explicit and a non-vacuity check; `longest_run`; the overlay's no-op-before-first-step contract and its arc integrator |

---

## 7. Anything not as pre-registered

**Nothing in the criteria.** No threshold, seed range, episode count or definition was
altered. G1, G3 and G5 fail as written and are reported as failures.

Three process notes:

1. **The gate was run twice.** The first full run tripped G1 and exited from inside the
   kill path *before* writing `step_06_gate.json`, so a 33-minute six-arm sweep left only
   the console tail. The script was restructured (`bff74d5`) so all arms run, the JSON is
   written, every gate's verdict is computed and printed, and only then does it exit
   non-zero with the kill report — and the violation census was added, because a single
   offending step cannot distinguish a 5e-12 m graze from a 5 cm penetration, and the two
   demand different fixes. The verdicts are identical across both runs (identical seeds and
   deterministic policies); only the evidence retained differs.
2. **The kill path reports and stops; it does not diagnose beyond measuring.** I did not
   change `FEAS_TOL`, the DCBF discretization, the relaxed tier's constraint set, δ, or the
   violation semantics. Each is a candidate root cause named in §3 and each is a
   pre-registered or root-proven constant.
3. **The visual gate is signed; Step 6 is not.** Exit requires G1–G4 to pass. They do not.

### What a fix would have to address (for the next pre-registration, not decided here)

- **Mechanism A** is a collision between `FEAS_TOL = 1e-9` and a violation definition that
  is blind to magnitude. Either the solver must enforce the DCBF row strictly at the
  surface, or the pre-registration must say what magnitude of clip counts as contact. Both
  are amendments.
- **Mechanism B** is the substantive one and is where I would look first: the relaxed tier
  drops exactly the constraints that have failed. A tier that keeps obstacle constraints
  with a slack variable (soft-min penalty) instead of deleting them, and/or a
  higher-order/exact discrete-time barrier condition rather than the first-order
  linearization whose neglected term is ~1 mm per step at ω_max, would attack the cause
  rather than the symptom. Xiao et al.'s learned feasibility constraint (ECC 2023, already
  in the plan for Step 10) is the pre-registered remedy for the infeasibility itself.
- **G5's result is informative in its own right** and is worth carrying into the paper: the
  measurement-robust margin, applied on top of a fallback tier that abandons obstacle
  constraints, makes hardware safety *worse*, not better. That is a finding about
  margin-plus-fallback composition, not an argument against δ.
