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

## Next steps (written 2026-11-01 evening; budget 5.0 M; deadline Nov 13 FIRM)
| # | When | Step | Output | Gate / owner | Est. tokens |
|---|------|------|--------|--------------|-------------|
| 0 | tonight | S25 closes (Paper A): 80 records → `--gifs` 12 → watch → amendment → `--summarize` → D158 | D158 | me (own steps) | 0.1 M |
| 1 | Nov 2 am | **Evidence audit** of B: for each claim (D112, D125–D131, D149, D138/D142) quote the exact numbers, N, seeds, conditions from the round JSON/decisions; FIRST check the S16 MEM and GAM arms against the mechanism story; list every figure's data source | `docs/pathway/paper_b_evidence_audit_2026-11-02.md` | me; blocks the skeleton | 0.15 M |
| 2 | Nov 2 | **Pre-register B1–B4** (one document): B1 stall geometry + dwell prediction; B2 recovery_after sweep {5,10,20,40,80} + plain control + **perturbation-recovery baseline arm** + cycle rate + escape-time-vs-bound; B3 phenomenon under PredictiveShield (no recovery, pinned metric); B4 recovery composed onto PredictiveShield (construct test: every candidate passes the predictive filter) with a one-day cut | `docs/pathway/paper_b_prereg_rounds_2026-11-02.md` | me; before any number | 0.1 M |
| 3 | Nov 2–3 | Implement B1–B4 runners (own code; tests; smoke; frozen worktree; launch from a pattern-free command) — B4 first so its cut decision is early | runners + launch | me; opus review ONLY for the one script producing gated numbers (B2) | 0.5 M |
| 4 | Nov 2–3 | **Skeleton for approval**: outline (L4DC length), formal section per the revised skeleton above, figure list with data sources, related-work map (Grover et al., Kim et al., Wang/He/Pajic, Dawood, Arnström, S2C) — citations VERIFIED (DOI/arXiv) before entry | skeleton doc | **user approval before prose** | 0.2 M |
| 5 | Nov 3–4 | Venue facts from the OpenReview template: page limit, blind policy, supplementary; author block; anonymised code/record release prepared (frozen commits, bit-identical replays) | checklist | me + user (authors) | 0.05 M |
| 6 | Nov 4–7 | **Draft** via the ARS pipeline entered at Stage 2 with the audit + skeleton + B1–B4 results as material (B results land Nov 3–5 into Figs. 3–5) | full draft v1 | pipeline Stage 2 → 2.5 integrity (mandatory) | 1.5 M |
| 7 | Nov 7–8 | Stage 3: 5-reviewer panel incl. Devil's Advocate; Stage 4 revise | v2 + response | pipeline (mandatory checkpoints) | 0.9 M |
| 8 | Nov 9–10 | Stage 3' re-review → Stage 4' if needed → **Nov 10 go/no-go** (user; rule: strong or not at all) | decision | **user** | 0.5 M |
| 9 | Nov 11 | Stage 4.5 final integrity → Stage 5 L4DC template; 2-min supplementary video from recorded GIFs; anonymisation check | submission package | me | 0.3 M |
| 10 | Nov 12 | **Submit** (one day inside the firm deadline); D159 "Paper B submitted"; worklog/tracker/memory | — | user submits on OpenReview | 0.05 M |
| — | after Nov 13 | Resume: Paper A write-up for RA-L (S25/D158 in hand); playable-site brainstorm (pending question: who plays first) | — | — | reserve ≈ 0.6 M |
Hard cuts: B4 dropped if not composing by Nov 3 evening; any B round not summarised by Nov 6 is reported as "in progress" in the draft and excluded from claims. User-only items, needed by Nov 3: author list/affiliation; niro-2 password rotation; confirmation of blind policy handling.

## Venue facts — VERIFIED 2026-11-01 from the 2027 CFP (site l4dc2027.control.ee.ethz.ch, OpenReview L4DC.org/2027)
- Deadline **Nov 13 2026, 23:59 AoE** (OpenReview `duedate` = 2026-11-14 11:59 UTC); site: "WILL NOT BE EXTENDED". No separate abstract deadline.
- **10 pages** in PMLR format including figures/tables/conclusions; references unlimited; the flyer says appendices unlimited, the author page says reviews are based on the 10 pages only and supplementary goes in the same portal → everything that must be reviewed stays inside 10 pages; proofs/extra results after the references. Template `L4DC2027_template.zip` (l4dc2027.cls = jmlr/pmlr; options `anon`, `final`). PDF only, max 50 (MB).
- **Single-blind.** The public repo and frozen commits may be linked (rotate the CLAUDE.md password first).
- Supplementary: allowed (long proofs, extra results) in the portal; no size or video policy stated; no separate OpenReview field → single PDF. Generative-AI use allowed with transparency.
- Decisions Jan 25 2027; camera-ready Apr 19; conference June 16–18 2027, KTH Stockholm; proceedings PMLR; all papers posters, selected orals.
- Topics 2027: "Reinforcement learning and adaptive control", "Agentic AI and algorithmic robustness", data-driven control, applications in robotics. **"Safe RL" is NOT in the 2027 list** (it was in 2025/2026) → frame as RL + robustness of the learning–control interface, robotics as the application.
- Dual-submission/arXiv policy: not stated in any year; 2025 explicitly allowed arXiv-hosted supplements.


## Status 2026-11-02 04:00 (after the night session)
Steps 1 (audit), 2 (prereg B1–B4 + A1–A3), 3 (runners, tests, launcher — opus review pending; launch waits for S25 to
release the host), 4 (skeleton `docs/paper_b/skeleton_2026-11-02.md` — **awaiting user approval**) and the archived
figures F1–F5 are DONE. Step 0 DONE 2026-11-01 08:25 (S25 closed, D158: the filter's role QUALIFIED; EVS-80 minimax without the filter). S26 LAUNCHED 08:22 from `b535af6` (7 chains; GIFs → watch → summarize next). Step 5 venue facts DONE (above); the author
block and the password rotation remain the user's.

**Status 2026-11-01 12:00:** S26 CLOSED (D159, D160; 94/94 cells, H-SAFE everywhere, all construct gates PASS). Figures F6–F8
rendered. Outcomes that change the draft: Prop 1 scoped to the exact tier with a 2 % enumeration share (H-B1-3); the
stochastic dwell law is dropped, R3 becomes an exploration remark (H-B1-4); Prop 3(b) is "typically" with 97.8 % and the tail
(H-B2-2); the perturbation baseline cannot leave a heading trap — the recovery's contribution is the rotation candidate
(H-B2-4); the loop is present but weaker under the predictive filter (H-B3-1) and the composition clears it 4/4 (H-B4-1, B4 IN).
Next: the user's remaining skeleton decisions, then drafting.

**Status 2026-11-01 (late, D161):** draft, figures and supplement now use paper-facing names (Experiments 1–9; \textsc{Base/Obs/Hor/ObsHor/Free/Aware}; G/H/A); the archive keeps the internal identifiers and `draft/supplement.tex` carries the crosswalk. The paper's `[repository URL]` will point at a SEPARATE release repository (`safe-and-stuck`, assembled at `/home/niro-1/Codes/safe-and-stuck`, private until the Nov 10 go/no-go; pushed to https://github.com/shoaib6174/safe-and-stuck, commit 5485369). Password rotation of THIS repository's history remains the Nov 10 item. Next: review (in-pair evaluator or `/ars-reviewer` panel).

**Status 2026-11-02 (D162):** PANEL REVIEW DONE → MAJOR REVISION (`docs/paper_b/review/editorial_decision.md`). Required: correct §3 (the intercept law keeps a last-seen point and acts out of view — verified in code), rescope C2/Remark 2 (inequality direction wrong; 629 = remaining horizon), Lemma + fixed-point Proposition, Prop 3(b) conditional, missing registered verdicts (P1, H-CROSS), abstract phrasings, figures at printed size, observation/reward paragraph, disclosures (Free/Aware selection, Aware's 1e-3 stall penalty). User decides two evaluation-only rounds (N1 branch attribution ~3 h; N2 rotation-accepting baseline < 1 h). Then `/ars-revision` → re-review Nov 7–8 → go/no-go Nov 10.
