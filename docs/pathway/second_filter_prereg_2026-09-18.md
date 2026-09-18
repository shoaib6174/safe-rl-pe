# S11 — Is the stall attack a property of hard shields in general? A second filter formulation (2026-09-18, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any cell of this sweep exists.**
Governing: **D118** (S9: under our CBF-QP shield the stall attack grows with margin and needs obstacles; a more
permissive gain makes it worse), **D120** (FT-B-41 is the artefact), **D117** (paper plan, priority 2 follow-up).

## 1. Why this round exists

Every shielded number so far comes from one filter: a discrete-time control-barrier-function QP. A reviewer will
ask whether the stall surface is a property of that construction. This round evaluates the same frozen
policies under a **different hard filter** with the same guarantee (no contact, ever) built a different way,
and asks whether the same adversary still finds a stall surface and whether it scales with the new filter's own
conservativeness knob.

**The second filter — `PredictiveShield`.** A sampling/lookahead safety filter: for the nominal command and the
same finite candidate set the CBF shield enumerates, simulate the environment's **exact** pursuer dynamics
(Euler unicycle step, wall clipping, obstacle projection — `pe2/env.py`) with the candidate held for `H` steps
followed by a full stop, and accept a candidate iff no simulated step touches a wall or an obstacle and every
simulated body clearance is ≥ δ. Execute the accepted candidate nearest to the nominal; if none is acceptable,
execute the candidate whose worst simulated clearance is largest (`backup`, always includes `u = 0`, which is
safe by construction because the pursuer is never in contact at the start of a step). It reports the same
per-step info keys as `CBFShield` (built from the same constraint geometry, so `shield_obs` features keep
their meaning) and is used through the identical `filter(action, state, obstacles, v_max)` interface.
**Its guarantee is exact for this deterministic environment**: a candidate is executed only if its simulated
successor is contact-free, and the simulation is the environment's own step. H-SAFE therefore tests the
implementation, not a modelling assumption.

**H-FILT-1 (existence):** under the predictive filter, at every horizon, the structured attackers produce a
HIGHER deadlock-episode rate than the strongest scripted probe for every pursuer (paired test, p < 0.05).
**H-FILT-2 (scaling):** the deadlock rate under the structured attackers is non-decreasing in the horizon `H`
(Spearman ρ > 0 over the four levels, and `H = 8` vs `H = 1` paired, for ≥ 2 of 3 pursuers against both
attackers) — a longer lookahead brakes earlier and makes the pocket bigger, the analogue of S9's margin.
**H-FILT-3 (transfer of the ordering):** the ordering of defenders by deadlock under attack seen with the CBF
shield (RCM28 < FT-B-41 < RC19 ≈ STR14) is preserved under the predictive filter at `H = 4`.
**H-SAFE:** 0 pursuer contact steps in every cell.

## 2. Cells

- Pursuers (frozen, `shield_obs` as trained): `RC19`, `RCM28`, `STR14` (H-FILT-2 scoring set), plus `FT-B-41`
  (reported; part of H-FILT-3). Opponents: `ATT25`, `ATK2-31`, `HUG7`, `GREEDY` (as S9).
- Predictive filter, δ = 0, n = 2, `H ∈ {1, 2, 4, 8}` steps (0.05–0.4 s of lookahead): 4 × 4 × 4 = 64 cells.
- CBF base cells for `FT-B-41` (δ 0, n 2, γ 0.2) × 4 opponents: 4 cells, completing S9's base row set.
- Stream **1 330 000+**, N = **300**, seed-paired across every cell (same n).
- Readouts as S9 (deadlock, stall, intervention, capture, escape, violations, clearance floors) plus the
  predictive filter's own `backup` rate.

## 3. Tests (fixed now)

- H-FILT-1: per pursuer × H, `paired_indicator_test(attacker_ind, best_scripted_ind)` where the best scripted
  probe is the one with the higher deadlock rate in that cell pair; CONFIRMED for a pursuer iff it passes for
  both attackers at all four H; overall iff ≥ 2 of 3 scoring pursuers.
- H-FILT-2: as S9's δ rule with levels H (ρ sign, top-vs-bottom paired); overall iff ≥ 2 of 3.
- H-FILT-3: at H = 4, against each attacker, the paired ordering RCM28 < FT-B-41 < RC19 holds with p < 0.05 for
  both inequalities (STR14 reported); CONFIRMED iff it holds for both attackers.
- Validity check **V-P** (preflight, before any cell): (i) with `H = 0` the predictive filter must return the
  nominal unchanged on 10 000 random states (it degenerates to a clip); (ii) on 200 random episodes of RC19 vs
  `ATT25` under `PredictiveShield(H = 4)` the env's contact flag never fires (the guarantee); (iii) a
  brute-force check on 1 000 random (state, candidate) pairs that the filter's accept decision equals an
  independent re-simulation with the env class itself.

## 4. What each outcome fixes

| result | direction |
|---|---|
| FILT-1, -2, -3 CONFIRMED | The stall surface is a property of hard shields, not of the CBF construction; the defender ordering transfers. The paper's generality claim is earned. |
| FILT-1 CONFIRMED, -2 NOT | The surface exists under the second filter but does not scale with its horizon; report the surface, drop the scaling generalisation. |
| FILT-1 NOT CONFIRMED | The attack is specific to the CBF-QP geometry; the paper narrows to that filter and says so. |
| H-SAFE fails | An implementation defect in the predictive filter (its guarantee is exact); fix and re-run is NOT permitted inside this round — the round is reported as failed and a new one pre-registered. |

**Stop rule.** Cells are never re-run; no level is added or removed after any number is seen.

## 5. Artefacts

`pe2/shield.py::PredictiveShield`, `results/derisk/s11_preflight.json`, `s11_cells/*.json`, `s11_round.{json,png}`;
runner `scripts/derisk_s11_filter.py` (`--preflight | --cells | --summarize`, ≤ 6 workers).

## Amendment S11-A1 (2026-09-18, before any cell exists): horizon levels, and the backup tier at δ = 0

- The implementer's timing note (Task 2 report) shows one env step moves ≤ 0.05 m, so `H ∈ {1, 2, 4, 8}` spans
  only 0.05–0.4 m of lookahead and may not separate. **Levels are changed to `H ∈ {1, 4, 16, 32}`** (0.05–1.6 s,
  up to 1.6 m of lookahead), keeping four levels and all rules (`H = 32` vs `H = 1` for the top-vs-bottom test;
  `H = 4` stays the level for H-FILT-3, unchanged). 64 predictive cells remain.
- At δ = 0 the `backup` tier of the predictive filter is unreachable from a contact-free state (u = 0 is always
  accepted), so its backup rate is identically 0 by construction; it is reported as such, not as a finding.

## Amendment S11-A2 (2026-09-18, before `--preflight`): V-P readings

- V-P (i) compares the `H = 0` output with the clipped nominal taken through the same float32 action-grid
  round-trip the evaluation path applies (the filter snaps every candidate to that grid so that the executed
  command is bit-identical to the simulated one); the runner transcribes the round-trip independently of the
  filter's own helper, and additionally bounds the deviation from the raw float64 clip by 1e-6.
- V-P (iii) draws half of its 1 000 (state, candidate) pairs aimed at an obstacle so that both accept and reject
  decisions are exercised (uniform draws are ~98 % accepts).
- V-P (i)'s reset seeds overlap the cell stream numerically; no episode is played and nothing is read out, so
  the cell stream is untouched. FT-B-41's four CBF base cells run on S11's stream and are therefore NOT paired
  with S9's base row; they are reported, never paired-tested against it.
