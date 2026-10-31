# Paper B — evidence audit (written 2026-11-01 night, dated for the Nov 2 step of the plan)

Purpose: before a skeleton exists, every claim Paper B will make is traced to its record, with N, seeds, conditions
and the exact numbers, and its evidential status is classified. Plan step 1 of `paper_b_l4dc_plan_2026-11-01.md`.
Nothing here is re-run or re-cut; the only computations are reads of the stored JSON (`results/derisk/`) and the
decision log. Rule for the draft: **quote only ARCHIVED numbers in results sections; LOG-ONLY diagnostics may be
narrated as the discovery path with their N stated, never gated on; DECISION-ONLY numbers are not quoted.**

Status classes:
- **ARCHIVED** — pre-registered spec, frozen worktree commit, record JSON under `results/derisk/`, figure PNG.
- **LOG-ONLY** — diagnostic run from a scratchpad script (now gone); the stdout log survives under
  `/home/niro-1/s14_logs/` or `/home/niro-1/s16_logs/`; nothing under `results/`; not pre-registered.
- **DECISION-ONLY** — numbers exist only in `docs/decisions.md` text.

## 1. Claim-by-claim

| # | Claim in the paper | Round / decision | Record | N, seeds, conditions | Numbers to quote | Status |
|---|---|---|---|---|---|---|
| C1 | A plain CBF filter on a learned pursuer costs capture; the cost is large | S4 / D112 | `s4_round.json`, spec `shielded_learned_pursuer_prereg_2026-09-14.md` | 21 paired cells, N = 500, stream 1 090 000+, STR14 (10 M PPO), δ = 0 and 0.05 | Greedy 1.2: 0.728 → 0.396 (Δ −0.33 [−0.38, −0.29], p = 3e-44); 5/6 opponents; teacher 0.558 → 0.426; stall-step 0.615, deadlock-episode 0.540 | ARCHIVED |
| C2 | A stall penalty alone does not recover it; filter-state observation does, partially | S5 / D113 | `s5_round.json` | 10 arms × 10 M shielded steps; gate N = 500 paired | RB (penalty) 0.426/0.464/0.436 vs shielded teacher 0.426 (0/3); RA (features) 0.522/0.512/0.544 (3/3, p ≤ 8e-4); deadlock eps STRSH 0.37–0.46 → RA 0.11–0.30 | ARCHIVED |
| C3 | The stall is a missing escape behaviour; the filter admits the opposite turn while the policy asks for the blocked one | D125 | `/home/niro-1/s14_logs/escape_diag*.log` | probe at one stalled state; 1085 consecutive steps, command σ = 0.002 | narrative only | LOG-ONLY |
| C4 | A 20-step rotation recovery removes the stall with zero violations and raises capture | D125 | same logs | N = 100 per cell, stream 1 500 000+, 3 policies × 4 opponents | RC19 immobile 0.392 → 0.024; STR14 vs GREEDY 0.39 → 0.71; RC19 0.55 → 0.66; 0 violations | LOG-ONLY — **superseded by C10 (archived, N = 500, 9 seeds)**; the STR14/RC19 rows are re-measured in B2 |
| C5 | Not our discretisation: an exact QP returns no forward motion either; translation is inadmissible but a rotation is always admissible | D126 (V1, V2) | `/home/niro-1/s14_logs/v1_qp.log` | 150 stalled states from RC19/FT-B-41/STR14 vs two learned attackers; SLSQP from 4 starts; 19 × 21 action grid | 0/150 forward (ours and QP); 0/150 any admissible forward action; 150/150 admissible rotation | LOG-ONLY — **re-measured by B1** (trap classification) |
| C6 | The loop survives the policy's own exploration noise | D127 (V3) | `/home/niro-1/s14_logs/v3_stoch.log` | N = 60 per cell vs ATK4-45, stream 1 700 000+ | immobile det → stoch: FT-B-41 0.276 → 0.275, RC19 0.293 → 0.333; untrained σ = 1: 0.484 → 0.312, deadlock 0.300 | LOG-ONLY — **B1 adds a stochastic-sampling arm (dwell prediction) so the paper's version is archived** |
| C7 | Mechanism: frozen input + consequence too delayed for the training discount | S15 / D128 | `s15_round.json`, `s15_cells/*.json`, spec `loop_mechanism_prereg_2026-09-20.md`, worktree 7fcf9ba | 8 cells × 200 eps, stream 1 710 000+, FT-B-41 and RC19 vs ATK4-45/GREEDY, hold 1 and 10; 772 stall entries | M1 (hold-10 cells, 383 entries, 97 escaped): capture after escape 0.299 vs 0.042 (7.1×), mean delay 629 steps, Δ discounted return 0.0092; γ needed for a 0.05 advantage ≈ 0.9974. M2 (hold-1): action Δ median 0.000, max 0.0030; world-obs Δ 0.000; evader visible 2.4 % of stalled steps. M3: σ state-independent by architecture (0.356 / 0.426). M4: hold 10 → 1 changes immobility < 0.05. All 8 cells pooled (my read tonight): 772 entries, 208 escaped, 0.298 vs 0.044, mean delay 677 | ARCHIVED (quote M1 as pre-registered on the hold-10 cells; the 8-cell pool is consistent and may be shown as a distribution) |
| C8 | Learned remedies: penalty backfires, memory confounded, longer horizon and admissible-set features help partially; scripted rotation recovery removes deadlock but spins | S16 / D129 | `s16_round.json`, `s16_arm_*.json`, spec `loop_fixes_prereg_2026-09-20.md` (+A1–A4), worktree 32caa57 | 5 remedies × 3 seeds × 5 M steps from scratch (BASE recipe: PPO, γ 0.99, stall_penalty 1e-3, shield_obs on); gate N = 500, stream 1 720 000+, vs ATK4-45/HUG7/GREEDY | see §2 for the MEM/GAM check. PEN: stall-step 0.143–0.249 → 0.093–0.158 while immobile 0.436–0.448 → 0.633–0.833 and GREEDY capture 0.384–0.478 → 0.204–0.348. REC: deadlock 0.000 ×9, firings 47 205 / 414 928 / 133 616 per 1500 eps | ARCHIVED |
| C9 | The long spins of the rotation-only recovery are not a trap: a straight command moves 100 % of the time | D130 | `/home/niro-1/s16_logs/rec_anatomy.log` | 25 + 12 + 10 eps BASE-61 vs ATK4-45, stream 1 800 000+; 326 sampled poses | engagements 1.0/episode, median 2 steps, mean 314, max 1111; (0.5, 0) and (1.0, 0) move in 100 % | LOG-ONLY — the design rationale for V2; the archived evidence that forward escapes dominate is C10's firing counts |
| C10 | **Headline:** a forward-offering recovery (one counter, same admissible set) clears the deadlock on 3/3 policy families, capture up 9/9 seeds, 0 violations; bare learned remedies raise capture but none clears the deadlock; layering reduces firings by 2–3 orders | S17 / D131 | `s17_round.json`, `s17_arm_*.json` (per-step traces for 5 eps per cell), spec `loop_fixes_b_prereg_2026-09-21.md` (+A1–A3), worktree d1ed3aa | 7 families × 3 seeds, gate N = 500 paired, stream 1 740 000+, vs ATK4-45 (deadlock), HUG7, GREEDY (capture); `RecoveryShieldV2(recovery_after=20)` fresh per episode; ~31 M shielded steps, 0 contacts | Deadlock vs ATK4-45: BASE 0.298/0.186/0.252; RECF-BASE 0.000 ×3; RECF-OBS 0.000 ×3; RECF-OBSGAM 0.006/0.002/0.002; OBS 0.338/0.256/0.414; GAM 0.290/0.384/0.612; OBSGAM 0.380/0.332/0.262. GREEDY capture: BASE 0.466/0.366/0.402 → RECF-BASE 0.640/0.718/0.442 (p 2.9e-25, 2.1e-53, 3.6e-5); RECF-OBS 0.556/0.614/0.706; RECF-OBSGAM 0.624/0.668/0.686; OBS 0.532/0.520/0.638; GAM 0.546/0.432/0.414; OBSGAM 0.560/0.634/0.636. Firings: RECF-BASE 13 035 / 152 675 / 27 974; RECF-OBS 1 153 / 7 268 / 28 649; RECF-OBSGAM 35 755 / 363 / 513. Kinds RECF-BASE-61: 144 362 forward, 8 313 rotation. Spin fraction ≤ 0.025 everywhere (H3 uninformative — report as such) | ARCHIVED |
| C11 | Dissociation 2: an intervention-magnitude penalty improves the deadlock metric in 9/9 cells while capture falls | S18 / D138 | `s18_round.json`, 18 arm records, spec `boundary_condition_prereg_2026-09-23.md` | 18 arms × 5 M, gate N = 500, stream 1 900 000+ | DVARY capture 0.442/0.480/0.528 → 0.338/0.300/0.212 (pooled p 3.6e-51); zero-motion requests 0.07/0.16/0.00 → 0.37/0.78/0.80; DVARY-72 deadlock 0.224 → 0.106, immobile 0.23 → 0.81. Carries Case 5 confound (penalty charged for the law's interventions too) | ARCHIVED |
| C12 | Dissociation 3: removing the law pins (−94…−99 %) leaves capture flat or worse | S19 / D142 | `s19_round.json`, `s19_cells/`, `s19_gifs/`, frozen a252bc8 | 12 cells, N = 500 paired, stream 2 000 000+ | law-pin fraction GREEDY 0.170 → 0.002, HUG 0.132 → 0.008, ATK 0.368 → 0.004; shielded capture PLAN vs LINE +0.026 / −0.018 / −0.022 (p 0.015 / 0.035 / 0.007) | ARCHIVED |
| C13 | The loop recurs in a different controller (scripted tactic game) under the plain filter and the recovery fixes it there | S24.0 / D149 | GIFs at commit 231a362 (`results/derisk/s24_0*`?) — **location to confirm**; the re-run baseline with `RecoveryShieldV2` is S24.1's game | 3 × 100 paired eps, seeds 2 290 000+ | pins of 760 and 787 steps; immobile in 11/30 replayed episodes | ARCHIVED as GIFs/decision; **supporting anecdote only** (no paired recovery-vs-plain numbers at N = 500 for that controller) |
| C14 | H-SAFE: zero pursuer contact steps in every shielded cell of every round above | all | every round JSON `hsafe` | — | 0 everywhere (S4 ~6 M + 30 M steps; S17 ~31 M) | ARCHIVED |

## 2. The S16 memory and discount arms against the mechanism story (checked first, as the plan required)

**GAM (γ = 0.999, the discount half of C7).** S16 gate (stream 1 720 000+): immobile vs ATK4-45 0.333/0.381/0.424 vs
BASE 0.448/0.436/0.448 — below BASE by ≥ 0.05 on 2/3 (H-FIX-3 passed). S17 re-gate (stream 1 740 000+): GREEDY capture
0.546/0.432/0.414 vs BASE 0.466/0.366/0.402 — significantly up on 2/3; **deadlock-episode rate vs ATK4-45
0.290/0.384/0.612 vs BASE 0.298/0.186/0.252 — not lower, and GAM-62 is the worst deadlock number in the round.**
Reading: lengthening the horizon raises the task metric and does not clear the deadlock. That is CONSISTENT with the
mechanism as a two-part story (frozen input AND short horizon: fixing the horizon alone leaves the frozen input) but
it is NOT evidence that the discount is the binding cause. **Consequence for the draft:** the discount argument stays
a *remark* (an upper bound on the escape signal), GAM is reported as "helps the task, does not remove the failure",
and the paper must not say "the horizon explains the loop".

**MEM (RecurrentPPO).** Confounded before any number by amendment S16-A4 (`n_steps` 128 vs 2048, `batch_size` 128 vs
64). Gate: immobile 0.787/0.763/0.641, GREEDY capture 0.268/0.262/0.286 — the worst arm. **Licenses no conclusion
about memory.** Consequence: the paper's object is the *memoryless* policy (Prop 1's hypothesis), and the limitations
paragraph must state that a clean memory arm was not run; the claim "memory would fix it" is NOT made. If a reviewer
asks, the honest answer is "untested"; a matched-hyperparameter MEM arm is future work (one training round, not
feasible before Nov 13 — training is excluded by the plan).

**OBS (admissible-turn features, the information half).** S16 immobile 0.274/0.359/0.241 (better than BASE 3/3); S17
capture up 3/3 (0.532/0.520/0.638), deadlock 0.338/0.256/0.414 — not cleared. Same reading as GAM: information helps
the task, does not remove the absorbing state. Together with D113 (features recover capture 3/3, deadlock stays
0.11–0.30) this is a consistent, archived pattern across two policy generations: **learning-side remedies trade the
task metric up and leave the failure mode in place; only the escape mechanism removes it.** That is C10's sentence.

## 3. Findings from tonight's reads that change the draft

1. **Two kinds of immobility, and the paper must separate them.** In S17 `BASE` is immobile 0.46–0.62 of steps but
   *stalled* (immobile while commanding ≥ 0.5 v_max) only 0.08–0.25; in the five archived per-step traces of
   `BASE-61` vs ATK4-45 the long immobile runs (807–1140 steps to episode end) have the policy's OWN v_nom = 0.000 or
   0.132 — it has stopped asking. Among the 45 bare-family traces, 28 contain a *stalled* run ≥ 40 steps; the longest
   is 1106 steps (`OBS-61`, episode 3) with v_nom = 1.000 ± 0.000 and ω_nom std 0.002 — the D125 picture, archived.
   Both are fixed points of the same closed loop (frozen observation → constant command → no motion); one is
   filter-blocked, one is self-stop (the PEN/D138 mechanism). Prop 1 should be stated for "executed speed zero"
   and the two cases named; the recovery fires on `immobile`, i.e. on both, which is why RECF clears both.
   **Figure source for the frozen-input panel: `s17_arm_OBS-61.json` → gate HUG7 or GREEDY → traces[3], steps 94–1200.**
2. **The headline zero-shot numbers (STR14 0.39 → 0.71, RC19 0.55 → 0.66, immobile 0.392 → 0.024) are LOG-ONLY,
   N = 100.** The archived equivalent is C10 (5 M-from-scratch policies). B2 adds the two D112/D125 policies
   (`STR14`, `RC19`) as rows at N = 500 paired so the paper can quote the headline on the policies the story began with.
3. **Metric names.** `pe2.metrics` already uses `pinned` for "≥ 40 consecutive backup-tier steps". The paper's formal
   object must not reuse the word: use *stall run* (≥ K consecutive steps with v_exec ≤ 0.05 v_max and v_nom ≥ 0.5
   v_max), *deadlock episode* (an episode with a stall run ≥ K = 40, 2 s), *immobile-step rate* (no condition on the
   nominal), and *position trap* for the formal notion (no heading admits forward motion at the position). The plan
   doc's "pinned (position within r for K steps)" is renamed accordingly.
4. **C7's M1 is pre-registered on the hold-10 cells (383 entries).** The 8-cell pool (772 entries) gives the same
   reading (0.298 vs 0.044; delay 677) and may be shown as the distribution figure, labelled as the pool.
5. **Spin test H3 (S17) is uninformative** and must be presented as a design error, not a pass (D131 text).
6. **S16's REC arm (rotation-only) is the one archived instance of a remedy that satisfies a deadlock metric by
   spinning** (deadlock 0.000 ×9, firings up to 277/episode, immobile 0.118–0.257). It is a dissociation instance in
   its own right and the motivation for offering forward candidates; cite it as such.
7. **Anonymity is not an issue** (L4DC is single-blind; author-information page, read 2026-11-01), so the public
   repo and the frozen commits can be linked. The password in `CLAUDE.md` must still be rotated before the link goes in.

## 4. Figures that can be made from archived data now (sources)

| Fig | Content | Source |
|---|---|---|
| F1 | Frozen input: v_nom, v_exec, ω_nom over a 1106-step stall; inset: recovery firing on the RECF twin | `s17_arm_OBS-61.json` traces (bare); `s17_arm_RECF-OBS-61.json` traces (same seeds) |
| F2 | Mechanism: capture-after-escape 0.299 vs 0.042; delay distribution; discounted advantage vs γ with 0.99 / 0.999 marked | `s15_round.json` m1 + `s15_cells/*.json` entries |
| F3 | Remedies: deadlock-episode rate (ATK4-45) vs GREEDY capture, 21 arms, families coloured; bar at 0.05 | `s17_round.json` arms |
| F4 | Firings per family (log scale) with forward/rotation split — the layering argument | `s17_arm_*.json` final.recovery_kinds |
| F5 | Three dissociations: (a) S16 PEN stall-step ↓ vs immobile ↑ vs capture ↓; (b) S18 DVARY deadlock ↓ vs capture ↓; (c) S19 law-pin −94…−99 % vs capture flat | `s16_arm_PEN/BASE-*.json`, `s18_round.json` effects/arms, `s19_round.json` verdicts |
| F6 (B1) | Admissible set in (v, ω) at a real stalled state: exact one-step admissible region, linearised rows, nominal, projection; trap classification table; predicted vs observed dwell | B1 runner |
| F7 (B2) | Deadlock, capture, firings and cycle rate vs `recovery_after` ∈ {5, 10, 20, 40, 80}; perturbation baseline; escape time vs the N + one-turn bound | B2 runner |
| F8 (B3/B4) | The phenomenon and the fix under the predictive filter | B3/B4 runner |

## 5. What the audit changes in the pre-registration (carried into `paper_b_prereg_rounds_2026-11-02.md`)
- B1 collects *stalled* states (filter-blocked) and *self-stop* states separately and classifies both.
- B1 includes the stochastic-sampling dwell measurement (C6's archived replacement) and the exact-QP agreement check (C5's).
- B2 includes rows for `STR14` and `RC19` (plain vs recovery_after = 20) at N = 500 paired (C4's archived replacement),
  and reproduces S17's `BASE`/`RECF-BASE` cells bit-for-bit as a construct gate (same stream, same seeds).
- C13's record location is confirmed before the draft cites it; otherwise it is an anecdote with a GIF.
