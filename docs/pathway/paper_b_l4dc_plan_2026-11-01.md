# Paper B → L4DC 2027 — plan, formal section, pre-writing rounds (2026-11-01)

**Decision (user, 2026-11-01):** Paper B (the absorbing-loop paper) goes to **L4DC 2027** (9th Learning for Dynamics &
Control, Stockholm, June 16–18 2027; OpenReview group L4DC.org/2027; **deadline Nov 13 2026, FIRM**; decisions Jan 25;
camera-ready Apr 19). Paper A (tactic game, D151–D157 + S25) stays on the RA-L plan (late Feb 2027, IROS option). B only
at L4DC: time, venue match, independence of outcomes. Playable-site project PAUSED until Nov 13. **Budget for this
phase: 5.0 M tokens (user, 2026-11-01)**; the S24/S25 research ledger closed at ≈ 6.8 M / 7.0 M.
Rule in force: publish only when strong → **go/no-go on Nov 10** after the internal review panel; fallback RA-L spring.

## What Paper B is
Safety filters trap memoryless learners in an absorbing observation loop: the forward-invariance guarantee holds,
violations are zero, the task fails silently (D125, D126/D127 Phase 0); mechanism = frozen input + a discount horizon
≈ 6× too short (D128); the standard remedy (stall penalty) backfires and the deadlock metric dissociates from capture
three times (D129, D138, D142); a forward-offering recovery (one counter of shield state, substituted action from the
SAME admissible set) clears the deadlock on 3/3 policy families, capture up 9/9 seeds, 0 violations (D131); the plain
CBF reproduced the loop inside the tactic game and the recovery fixed it there too (D149). Artefacts: S15–S17 round
JSON/PNG under `results/derisk/`. The pre-D125 draft fragments in `docs/paper_*.md` are SUPERSEDED — rewrite, do not reuse;
D112–D124 must not be cited as written (D125 banner).

## Venue facts (verified 2026-11-01)
Exemplar: Kim, Corsi, Rodríguez et al., "Realizable Continuous-Space Shields for Safe RL", L4DC 2025, PMLR 283:932–945
(arXiv 2410.02038): formal object (realizability of stateful shields) + modest empirics (2 envs, PPO, 5 seeds × 100,
success/collision only, no code); states "the shield alone leads to overly conservative behavior" (oscillations,
timeouts) and fixes it on the SPECIFICATION side (loop-avoidance requirement). Same volume: "Neuro-Symbolic Deadlock
Resolution in Multi-Robot Systems" (Wang, He, Pajic), "A Dynamic Safety Shield…" (Dawood et al.), "Data-Driven and
Stealthy Deactivation of Safety Filters" (Arnström, Teixeira), "Safe Exploration… Backup CBFs" (Rabiee, Safari), STLGame.
TO VERIFY from the OpenReview template: page limit (earlier editions ≈ 10 pp + refs), double-blind? (if yes: no link to
the public repo; fix the CLAUDE.md password first regardless), supplementary video policy.

## Formal section — critique of the first draft and the revised skeleton (binding for the draft)
Critique: Prop 1 as first written ("filtered action zero ⇒ fixed point") is a tautology AND its real content is a KNOWN
result — the KKT/normal-cone deadlock characterisation of CBF-QP controllers (Grover, Liu, Sycara: "Why does symmetry
cause deadlocks?" and the later IJRR deadlock analysis — VERIFY exact citations before use). Prop 2 (advantage ≤ γ^T R)
is an upper bound on a signal, not a proof that learning fails → remark. Prop 3 was under-specified.
Implementation facts checked 2026-11-01: recovery candidates pass through the SAME `CBFShield.filter` call (Prop 3(a)
holds); the shield has three tiers (exact / relaxed / backup — the projection characterisation is for the exact tier;
D126 V1 licenses it); the shield-obs stall-run feature saturates at 2 × DEADLOCK_STEPS = 80 steps (observation is only
EVENTUALLY constant); QP weights w_v = 150, w_ω = 1 (weighted norm).
Revised skeleton (≈ 1 page, Section 3):
- Definitions: projection (minimal-deviation) filter onto convex A(s) in the weighted norm; **filter trap** (no heading
  admits forward motion at position p) vs **pair trap** (some heading does, the policy never requests it there);
  **pinned** (position within r for K steps — the metric the rounds gate on, NOT immobility, which counts rotation);
  **liveness** (reach capture / leave a pinned ball within K) vs forward invariance (safety).
- Prop 1 (characterisation, cited to Grover et al., applied to a learned memoryless policy): output zero iff request in
  the weighted normal cone of A(s) at 0; refinements that are ours: position traps vs point traps (rotation in place is
  not a state fixed point); eventual constancy of the observation (stall-run feature saturates → eventual fixed point;
  explains D113's partial recovery); stochastic variant: per-step leave probability = policy mass outside the cone,
  dwell time geometric; with the collapsed σ ≈ 0.002 (D125) it is ≈ 0 → D127; entropy collapse makes the trap absorbing.
  Corollary: violations stay 0 along the loop — safety metrics are blind to it. D130 (326/326 straight commands move)
  = the D125 states are pair traps, not filter traps.
- Remark (D128): frozen input (credit must attach to one shared input vector) + discounted escape value ≤ γ^T R with
  T ≈ 629 ≈ 6 effective horizons at γ = 0.99; terminal-only reward assumed (true for S17 BASE); D127 carries the claim.
- Prop 3 (recovery): (a) substituted action ∈ the same admissible set ⇒ guarantee no weaker than the base filter's;
  (b) if p is NOT a filter trap, the position is left within N + one full turn at ω_max (rotate-then-move, D126 V2);
  explicit caveat: a stall–recover–stall cycle (duty N/(N+1)) is not excluded — measured by B2 and bounded by the
  observed immobile rate 0.024 (D131); the anti-spin test H3 is reported as uninformative.
- Positioning: the characterisation covers any minimal-deviation filter incl. Kim et al.'s "closest safe action" shield
  → predicts their oscillations/timeouts; theirs fixes the spec (loop-avoidance, realizability), ours is the minimal
  stateful shield (one counter), spec and guarantee untouched, evaluated on learned policies in a game.

## Pre-writing rounds (evaluation only, ≈ 0.3–0.4 M; launch when S25 clears the host; pre-register first)
- **B1 stall geometry (Prop 1 measured):** stall states of the frozen BASE policy under the plain CBF shield; per state:
  exact weighted-QP projection zero? admissible set contains motion at this heading / at some heading (filter vs pair
  trap)? Gaussian-policy leave probability vs observed dwell. Figure: admissible polytope in (v, ω) with request and
  projection; table: trap classification; dwell prediction vs observation.
- **B2 recovery sensitivity + cycles (Prop 3 measured):** frozen S17 families under `recovery_after ∈ {5,10,20,40,80}` +
  plain-shield control, N = 500 paired: pinned-run fraction, capture, recoveries, stall–recover–stall cycle rate within K.
- NOT before the deadline: porting the recovery to a second filter class; any training; Paper-A work beyond closing S25.

## Twelve-day schedule
Nov 1–2 venue facts + skeleton (outline, formal section, figure list from S15–S17 records, related-work map) → user
approval before prose · Nov 3–6 draft (ARS pipeline, entered at Stage 2 with B's materials; it was interrupted before
dispatch on 2026-11-01) · Nov 7–8 integrity + 5-reviewer panel · Nov 9–10 revise · **Nov 10 go/no-go** · Nov 11 L4DC
template + 2-min supplementary video from recorded GIFs · **Nov 12 submit**.
User-only items: author list/affiliation; rotate the niro-2 password (public repo); blind-review implications.
