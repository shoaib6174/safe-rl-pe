> # SUPERSEDED BY D125 (2026-09-19) — THE CENTRAL ARGUMENT OF THIS NOTE IS WRONG
>
> This note argues the stall is a geometric pocket in which the filter's admissible set collapses. It is not.
> At a real stalled state one turn direction is blocked and **the opposite turn is returned by the filter
> untouched, with no intervention**; the policy asks for the blocked direction for 1085 consecutive steps
> (command σ = 0.002) because its observation is frozen and nothing in it says which way is free. A 20-step
> rotation recovery drops the immobile-step rate from 0.392 to 0.024 with zero violations and equal or better
> capture. See D125.
>
> **§1 of this note contains its own refutation and I did not notice**: it states that in a pocket the
> admissible set leaves "only commands with `v ≈ 0`: the pursuer may turn in place but not move", and then
> calls that a stall without ever asking whether turning in place would let it escape. It would.
>
> **What remains usable.** The three-knob table in §2 is valid *data* — the measured frequencies do scale with
> the filter's margin, with a more permissive gain, and with obstacle count. What changes is the
> interpretation: those knobs govern how often the policy is put into a position it cannot reason its way out
> of, not how often the geometry genuinely traps it. §3 and §4's readings rest on the refuted claim.

---

# Why a hard shield creates a stall attack surface — mechanism note (draft, 2026-09-17)

Priority 4 of the paper plan (D117). Evidence: D112 (S4), D115 (S6b), D116 (S7), D118 (S9), D119 (S8).

## 1. The filter

The pursuer's command `u = (v, ω)` passes through a control-barrier-function QP: minimise `‖u − u_nom‖²`
subject to `ḣ_i(x, u) ≥ −γ h_i(x)` for every wall and obstacle constraint `h_i` (signed clearance minus the
margin δ), plus the actuator box. The admissible set `U(x)` is a polytope in `(v, ω)` whose faces are the active
constraints. Far from any surface `U(x)` is the whole box. Near one surface it loses the half-plane of commands
that drive into it. Near **two** surfaces that face each other at an angle — a wall meeting an obstacle, two
obstacles, a wall corner — the two lost half-planes can leave only commands with `v ≈ 0`: the pursuer may turn
in place but not move. That is the **stall**: the shield is doing exactly what it guarantees, and the pursuer
is safe, stationary, and useless.

## 2. Why the stall is a pocket, not a tax

Three predictions distinguish "the stall is a geometric trap" from "the stall is the price of conservativeness".
S9 (D118) tested them with 132 evaluation cells on frozen policies, zero violations throughout.

| knob | trap prediction | tax prediction | measured (RC19 vs its best response) |
|---|---|---|---|
| margin δ ↑ (constraints inflated) | pockets grow and close earlier → more stalls | more stalls | 0.59 → 0.78 → 0.80 → 0.86 → 0.89 — **both agree** |
| obstacles n ↓ to 0 (walls only) | no concave pockets → stalls vanish | conservativeness unchanged → stalls persist | 0.59 → **0.13**, capture 0.87 — **trap** |
| class-K gain γ ↑ (filter brakes later, more permissive) | pursuer enters pockets at speed → **more** stalls | less conservative → fewer stalls | 0.36 → 0.59 → **0.87** — **trap** |

Two of the three knobs separate the hypotheses and both side with the trap. A wall alone is a half-space and
cannot pin a unicycle that may rotate freely; the pocket needs a second surface. And a filter that brakes late
does not reduce stalls, it increases them, because the pursuer arrives deeper inside the pocket before the
admissible set collapses. Conservativeness per se is not the cost; the geometry the filter is asked to respect
is.

## 3. Why an adversary finds it

The evader is 20 % faster and only needs to survive. Its structured best response (S6b, D115) has two parts:
keep a solid surface between itself and the pursuer whenever the pursuer is within 4 m, and otherwise position
freely. The first part is enough on its own — an unrewarded evader with that law already pins STR14-SHIELD in
0.65–0.68 of episodes; learning adds nothing against a pursuer that has not learned to avoid pockets, and adds
+0.04…+0.23 against one that has (RC19). The attacker is not exploiting the learner; it is exploiting the
geometry the learner was trained under.

## 4. Why the defender's remedies trade capture for stalls

A pursuer that never enters a pocket cannot be pinned and also cannot corner a faster evader, because cornering
is done in pockets. S7's mixture-trained pursuer (D116) sits at that end: deadlock 0.09–0.22, capture 0.13.
S5's shield-aware pursuer (D113) sits at the other: capture 0.56, deadlock 0.6 under attack. S8 (D119) shows
the frontier between them is not empty — one anchored fine-tune reaches deadlock 0.43–0.47 with capture 0.67 —
but it is narrow: two of three seeds at the same setting stayed at RC19's corner, and the anchor at β = 0.1
never left it. S10 measures how reliably that frontier point is reached.

## 5. What this says about shields in adversarial settings

A hard shield converts the adversary's objective from "cause a collision" to "cause a stall". The size of that
surface is set by the shield's own knobs (margin, gain) and by the arena's concavity, not by how well the
policy was trained. Any evaluation of a shielded policy against non-adversarial opponents (Greedy, Hug7) will
understate the surface: the scripted probes reach 0.30–0.34 where the learned attacker reaches 0.60–0.63
against the same defender (S8 references). The honest deployment statement is a pair — capture kept, stall
rate under best response — never a single number.

*Open for the paper:* a one-paragraph formal statement of the pocket condition (two active constraints whose
`ḣ` rows are linearly dependent in the direction of `v`) and a figure of `U(x)` shrinking as the pursuer
approaches a wall–obstacle pocket, drawn from the shield's own QP (`pe2/shield.py`), would make §1 exact.
