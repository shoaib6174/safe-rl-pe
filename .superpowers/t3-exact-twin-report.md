# T3 — exact-game shield-shift twin

**Spec**: `docs/pathway/derisk_plan_2026-09-04.md`, Tier 2 / T3.
**Date**: 2026-09-05. **Status**: complete, criterion met.

- Script: `scripts/derisk_t3_exact_shield_shift.py`
- Tests: `tests/pe2/test_derisk_t3_exact_shield.py` (29 tests, all passing)
- Results: `results/derisk/t3_exact_shield_shift.json`
- Figure: `results/derisk/t3_exact_shift.png`

`scripts/experiment_1_tabular_forall_exists.py` was **not modified**; its
`solve_zero_sum_nash` is imported and reused.

---

## 1. What was built

A 7×7 gridworld simultaneous-move zero-sum pursuit-evasion game, extending the
paper-1 tabular script. Both players observe the full state and choose from
{stay, up, down, left, right}; capture is Chebyshev distance ≤ 1 after the joint
move and is absorbing, paying the pursuer 1. Horizon 7, **discount 1.0**, so the
value of the game *is* the capture probability — the unit the pre-registered
0.02 threshold is written in. (The parent script uses γ = 0.95 on a 3×3 grid; a
γ = 0.95 shadow value survives only as a tie-break inside the best-response
solvers so that "capture sooner" beats "capture later" and best responses are
not lazy. It never enters a reported number.)

Two quantities per (layout × shield configuration):

1. **Exact Nash value**, by backward induction with a matrix-game minimax solve
   at every state. This is the exact equilibrium value in behavioural
   strategies. No double oracle, no policy population, no approximation.
2. **Best-response-dynamics fixed point**: alternating exact best response from
   (greedy_catch, greedy_escape) with cycle detection. Every configuration
   entered a 5-cycle; the reported `br_value` is the mean over the cycle.

### Grid and horizon were calibrated on the unshielded hazard-free game only

A value pinned at 0.0 or 1.0 cannot move by 0.02 whatever the shield does, so
the game had to be sensitive before any arm was run. On a 5×5 grid the pursuer
captures everything by horizon 6 (V = 1.000) and only a 4-step horizon is
interior; capture radius 0 gives V = 0.000 at every horizon (the evader always
has an escape cell). A 7×7 grid at horizon 7 gives V = 0.562745 — interior and
deep enough for the BR dynamics to develop. **No shielded information entered
this choice**; the full calibration table is in the JSON
(`horizon_calibration`).

### The shield

`nearest_cell` (**primary**): if an action's target cell is a hazard, the move is
replaced by the reachable non-hazard cell **nearest in L1 to the intended
target**, ties broken by lowest action index in the order (stay, up, down, left,
right). "Reachable" = the target of one of the five actions from the current
cell. The current cell is never a hazard under shielded play, so "stay" is
always a candidate and the shield never has to fail open on an occupiable cell
(asserted, and 0 in every layout).

**Stated because it is a property of the discrete action set, not a modelling
choice**: on a 4-neighbourhood grid the L1-nearest safe cell to a blocked target
is always the cell you are standing on, so `nearest_cell` coincides exactly with
"hazards act as walls for the shielded player". That is the strongest possible
projection here, which is what makes the effect sizes below large — and would
have made a null correspondingly strong.

`motion_preserving` (**secondary, robustness only**): the candidate set excludes
staying put whenever any other safe move exists, so the player is deflected
sideways instead of stopped. It never decides the criterion.

**Result: the two projections give bit-identical values in all six shielded
configurations.** This is not a coincidence and it matters for how the headline
is read. Both rules have the same *image* — the set of safe cells one action
reaches — so they induce the same game up to a relabelling of actions. In an
exactly-optimised game only the induced reachable set can matter; the projection
rule can only affect a *learner*, whose gradient sees which action was taken.
So the shift reported below is **not** an artefact of how "nearest" was defined.
This is covered by a test, together with a test that the two tables really do
differ (otherwise the first would be vacuous).

---

## 2. Layouts and why

All on the 7×7 grid, 49 cells.

| Layout | Hazards | % of grid | Why |
|---|---|---|---|
| `sparse` | (1,5), (3,3), (5,1) | 6 % | Smallest perturbation that still lets the shield fire in more than one region. On the anti-diagonal and spread rather than clustered, so it is invariant under 180° rotation and privileges no region. |
| `dense` | all (r,c) with r,c ∈ {1,3,5} | 18 % | A regular lattice of pillars: 3× the density of `sparse` with the same symmetries, so density is the *only* thing that differs between them. Every remaining cell has an even row or an even column, so the safe set is provably still connected. |
| `corridor` | row 3 except (3,3) | 12 % | A hazard wall with a single gap. A shielded player must funnel through the chokepoint while an unshielded one crosses anywhere — the layout where a projection shield restricts the reachable set most. Mirror- and rotation-symmetric; safe set connected through the gap. |

They were chosen to span sparse / denser / chokepoint as the spec asks, before
any value was computed, and were not revisited afterwards.

### How the hazard confound was handled

Within a layout, **both arms carry the identical hazard set**, and the hazards
define the **identical initial distribution**: uniform over ordered pairs of
distinct non-hazard, non-wall cells at Chebyshev distance > 1 (not already
captured). The only difference between arms is whether the projection is applied
to a move. The unshielded arm walks straight through hazards — that is what
being unshielded means, and it is the unsafe behaviour the shield exists to
prevent (verified by a test: the unshielded arm does enter hazard cells at
equilibrium; the shielded arm never does).

Because hazards remove start states, the *unshielded* value differs slightly
between layouts (0.5748 / 0.5603 / 0.5032 against the hazard-free 0.5627). That
offset is common to both arms of a layout and cancels in every delta. The
hazard-free game is reported as a reference point only and never enters a delta.

---

## 3. Exact values

Exact Nash value = capture probability. `Δ` = shielded − unshielded.

| Layout | Configuration | Nash value | Δ Nash | BR fixed point | Δ BR |
|---|---|---|---|---|---|
| — | hazard-free reference | 0.562745 | — | 0.140686 | — |
| sparse | unshielded | 0.574751 | — | 0.136988 | — |
| sparse | both shielded | **0.136213** | **−0.438538** | 0.078627 | −0.058361 |
| sparse | pursuer only | **0.127353** | **−0.447398** | 0.069657 | −0.067331 |
| dense | unshielded | 0.560345 | — | 0.144397 | — |
| dense | both shielded | **0.057471** | **−0.502874** | 0.049138 | −0.095259 |
| dense | pursuer only | **0.057471** | **−0.502874** | 0.045259 | −0.099138 |
| corridor | unshielded | 0.503185 | — | 0.136815 | — |
| corridor | both shielded | **0.612739** | **+0.109554** | 0.363694 | +0.226879 |
| corridor | pursuer only | **0.189809** | **−0.313376** | 0.084586 | −0.052229 |

The secondary `motion_preserving` projection reproduces every shielded number
above exactly (Nash *and* BR).

### Criterion and sign

Pre-registered: effect present iff |ΔV| > 0.02 on ≥ 2 of 3 layouts.

- **Both-shielded: criterion MET, 3/3 layouts.** Sign is **mixed** — strongly
  negative on `sparse` (−0.44) and `dense` (−0.50), positive on `corridor`
  (+0.11). The magnitudes are 5–25× the threshold.
- **Pursuer-only: criterion MET, 3/3 layouts, sign uniformly NEGATIVE**
  (−0.45, −0.50, −0.31).

The BR-dynamics deltas agree in sign with the Nash deltas on every single
layout × configuration, which is a useful independent concordance: two different
solution concepts, same direction each time.

### The signed prediction for the deep R1 experiment

1. **A projection shield is not benign.** Jia et al.'s asserted mechanism is
   real and, in an exact game, very large. The claim is now measured rather than
   asserted. R1 gets a theory prediction with a sign.
2. **Symmetric shielding has a layout-dependent sign, and the geometry predicts
   it.** Scattered obstacle-like hazards give the evader cover and *lower*
   capture; a chokepoint confines the evader to one side of the arena and
   *raises* it. Predicted deep-run direction: negative for scattered obstacle
   fields, positive for corridor/chokepoint arenas. A deep experiment that
   averages over both geometries can easily read as a null and should not.
3. **Asymmetric shielding is unambiguous: shielding only the pursuer lowers
   capture probability on every layout, including the one where symmetric
   shielding raises it.** This is the cleanest signed prediction here, and it
   is exactly the confound the plan flags for Step 8: an asymmetrically shielded
   deep run will show a capture drop that has nothing to do with learning
   dynamics.

---

## 4. Negative controls

**NC-A (placement, the one the spec asks for).** A uniform initial distribution
on a fully connected grid puts positive probability on *every* cell at t = 0, so
a "cell no optimal policy ever moves into" cannot exist without changing the
geometry. Rather than work around that, the control layout adds two **wall**
cells at (0,5) and (1,6) which seal cell (0,6) into an unreachable pocket, and
places the single hazard there. Walls are environment geometry and are present
identically in both arms, so the arms still differ only by the projection.

| NC-A arm | Nash value | BR fixed point |
|---|---|---|
| unshielded | 0.6732283464566929 | 0.18188976377952754 |
| both shielded | 0.6732283464566929 | 0.18188976377952754 |
| pursuer only | 0.6732283464566929 | 0.18188976377952754 |

**Δ = 0.0 exactly**, for both the Nash value and the BR fixed point, in both
shielded configurations. Supporting diagnostics: the hazard cell is occupied
with probability 0 in the unshielded arm; projected action entries on occupiable
cells = 0; equilibrium shield activations = 0. The control passes.

A side observation that validates the fail-open counter: the raw all-cells
counter reports 5 fail-open entries, all five actions from (0,6) itself — from
inside the sealed pocket every action leads back into the hazard, so there is no
safe candidate. The cell is never occupied, so it has no effect; the
occupiable-cells counter is 0 everywhere, and that is the one that matters.

**NC-B (code path).** Shield enabled with an empty hazard set: 0.5627450980392157
shielded vs 0.5627450980392157 unshielded, Δ = 0.0 exactly.

---

## 5. Comparability and reachability findings

The spec's two failure modes were both checked, per arm, and are in the JSON.

- **The safe set stays strongly connected in every arm of every layout**,
  including `corridor` (through the single gap) and `dense` (every free cell has
  an even row or column). No shielded arm fragments the arena.
- **Capture remains possible in every arm** — the smallest shielded value is
  0.0575, well above zero. No arm was rendered degenerate.
- **Reachable joint states** (from the initial set, over all action pairs;
  unshielded is 2397 in all three layouts, i.e. the full space):
  sparse 2112 both / 2250 pursuer-only; dense 1596 / 1956; corridor 1841 / 2103.
  A 6–33 % reduction — the shield genuinely restricts the reachable set, which
  is precisely the mechanism Jia et al. name, without disconnecting anything.
- **Occupiable cells** shrink only because hazards are excluded from the start
  set in both arms (46 / 40 / 43 of 49), identically per layout.
- **Shield activations per episode at equilibrium are near zero even where the
  shield binds hard** (0.000–0.148). This is expected and is *not* evidence the
  shield is inert: an exact equilibrium already knows the projection and simply
  never selects an action that would be projected. The structural measures are
  the projected-entry count (12 / 36 / 14 on occupiable cells) and the
  reachable-state reduction above. The JSON carries this caveat inline.

### One result worth flagging as a finding rather than a bug

On `dense`, the both-shielded and pursuer-only arms give **identical values to
full float precision**. This was checked and is real: the two value functions
agree at *every one* of the 2040 non-captured states, not merely on the average.
Shielding the evader in the pillar field is a genuine no-op — the evader is
already escaping from ~94 % of starts, and in the states where it is doomed
within the horizon the nine extra cells do not save it. The evader's action set
is genuinely restricted there (36 projected entries, 360 fewer reachable joint
states), so this is not the shield failing to apply.

---

## 6. Correctness

The strongest available check is run for **every** configuration and stored in
the JSON as `equilibrium_saddle_check`: each player's exact best response is
computed against the opponent's stored behavioural strategy, and both must equal
the Nash value (no profitable unilateral deviation). **Every configuration
returned max_gap = 0.00e+00 exactly.** This is exact arithmetic, so any nonzero
gap would have been a bug.

### Tests — `tests/pe2/test_derisk_t3_exact_shield.py`, 29 passing, no skips

Purely tabular; no learning library is imported, so nothing is skipped.

- *Shield transition* (7): unshielded moves really do enter hazards (otherwise
  both arms would be shielded); shielded moves never land in one; no fail-open
  on any occupiable cell; safe moves are left untouched; the projection is
  verified to be the actual L1 minimiser cell by cell; pursuer-only leaves the
  evader's table equal to the raw table; walls block movement identically in
  both arms.
- *Initial distribution* (2): all three configurations of a layout share one
  initial pair set — the property that makes the delta a shield effect and not a
  hazard effect — and start states exclude hazards, walls and captures.
- *Exact solvers* (5): the saddle check; Nash bounded by a best-response value;
  a brute-force single-state deviation sweep against a computed best response;
  values are probabilities in [0,1]; BR dynamics terminate in a cycle.
- *Negative control* (3): the sealed pocket is unreachable from every occupiable
  cell; the shield is a no-op on that layout; an empty hazard set is a no-op.
- *Projection independence* (2): `motion_preserving` matches `nearest_cell`, and
  the two tables genuinely differ.
- *Structural comparability* (2): the shielded safe set stays strongly
  connected; shielding never *adds* reachable joint states.
- *Criterion arithmetic* (5): a delta of exactly 0.02 does not pass; two layouts
  suffice; opposite signs report "mixed" rather than being averaged away; a null
  reports as a null; the pursuer-only arm is scored separately.
- *Pre-registered constants* (3): threshold 0.02, ≥ 2 layouts, exactly the three
  named layouts, each carrying a stated rationale, densities ordered.

---

## 7. Deviations from the prescription, and open items

1. **Grid 7×7, not the parent's 3×3, and γ = 1.0 rather than 0.95.** The parent's
   3×3 arena has no room for a corridor or a hazard field, and its γ = 0.95 makes
   the value a discounted return rather than a capture probability, which is the
   unit the 0.02 threshold is stated in. γ = 0.95 survives as a tie-break only.
2. **The negative control needed wall cells.** The spec's "hazards where no
   optimal policy would ever move" is not constructible on a fully connected
   grid under a uniform initial distribution, because every cell is a start
   state. Two walls sealing a pocket make it constructible and provable. NC-B
   (empty hazard set) is an extra control that was not asked for.
3. **The horizon was calibrated**, on the unshielded hazard-free game alone,
   before any arm was run. This is a sensitivity choice, not a criterion change;
   the table is in the JSON so it can be audited.
4. **The BR-dynamics "fixed point" is a 5-cycle in every configuration**, not a
   single fixed point. Alternating best response between deterministic policies
   in a zero-sum Markov game is not expected to converge to a point; the
   reported value is the mean over the detected cycle, and the full trajectory
   and cycle values are in the JSON.
5. **`results/` is gitignored in this repo** (`.gitignore:37`), matching the
   existing `t0_*` outputs, so the JSON and PNG are produced but not committed.
   Only the script and the test file are committed.
6. **Not implemented**: nothing from the T3 spec was dropped. Runtime is ~90 s
   end to end on CPU.
