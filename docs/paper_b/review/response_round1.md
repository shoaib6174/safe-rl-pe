# Paper B — response to the round-1 panel (revision 1, 2026-11-03)

Revised files: `docs/paper_b/draft/main.tex` (10 pages + references) and `supplement.tex`. The first draft is kept at `draft/v1_round1/`. Facts used in the revision, with sources, are in `review/revision_facts_2026-11-03.md`. The new figures are `figures/R0_game`, `R1_loop.tex`, `R2_stall`, `R4_recovery` and `R5_dissoc`. They are produced by `scripts/paper_b_game_figure.py` and `scripts/paper_b_figures_rev.py`, and their data is in `figure_data_rev.json`.

**Experiment renumbering.** Experiments are now numbered in reading order. Supplement Table S5 maps the numbers. In the form new = old: 1 = 1, 2 = 7, 3 = N1 (new), 4 = 2, 5 = 3, 6 = 4, 7 = 8, 8 = 9, 9 = 5, 10 = 6. Below, "Exp." uses the **new** numbers.

**Title.** "Learned Policies" is now "Learning-Enabled Controllers". The authors chose this after N1, as the roadmap's question 1 requires.

## Two changes beyond the roadmap (found while doing R3 and R1)

1. **The discount explanation is withdrawn (D165).**
   - The learner discounts once per decision. A decision is one PPO transition: up to 10 held head steps plus the law's run, about 15.9 env steps in training.
   - The S15 analysis, and Remark 2 of draft 1, discounted per env step.
   - On the learner's clock the escape advantage is 0.29 at γ = 0.99 (S15's return-to-go form), and A₀·E[γ^T] = 0.18. That is not "worth 0.009".
   - The registered P1 verdict was FAIL. Its registered direction was "the signal was there and learning still missed it; optimisation, not credit assignment". The paper now says exactly that.
   - Remark 2 is replaced by Remark 4 ("the size of the signal"), labelled as a post-hoc recomputation. The mechanism figure is removed.
2. **Draft 1's opening episode was a scripted-law stall.**
   - Fig. 1 of draft 1 (Obs seed 2 vs the wall-hugging evader, seed 1 740 003) is a stall of the intercept law chasing a VISIBLE evader. N1 attributes all of its 1 106 stalled steps to "chase".
   - The new Fig. 1 replays Free vs greedy, seed 1 740 044. It is a learned-head stall from step 8 to the end. Under the recovery the same episode is captured at step 507.
   - The replay reproduces the archived capture indicators and N1's first 40-step run, and the script refuses to draw otherwise.

## P0 items

| # | Done | Where |
|---|---|---|
| R1 | The pursuer is described correctly: the law chases while the evader is visible, drives to a remembered point while it is not, and the head commands only when the law has nothing to do. The law's one-point memory is stated, and is frozen while the evader is out of view. A decision is defined. New Fig. 2 shows the closed loop. | §3 "The pursuer's controller", Fig. 2 |
| R2 | The learning claim is rescoped with N1's numbers (rule D-2). Law stalls happen inside one decision, so no learned parameter is in their loop. Head stalls carry a signal of ≥ 0.18 that is not used, and the cause is open, with aliasing as the candidate and memory untested. For Free, the stall is a deployment without the filter seen in training. The title, abstract, C2, §5.3 and the conclusion follow. Exp. 4's registered direction is stated and reconciled. | Abstract, §1, §5.3 |
| R3 | Remark 2 is corrected and goes further than asked (see change 1). 629 was the mean steps from stall entry to episode end, and 346 the mean to capture on the captured escapes; both are now stated. Draft 1's Fig. 2 is removed. "Remark 2 is a bound" is gone. | Remark 4 |
| R4 | Lemma 1 is the KKT/normal-cone condition with W = diag(150, 1) and outward normals, credited to Grover et al. and stated non-circularly. The implemented filter is described as a candidate search with an exact test that agrees with the projection on 98.1 %. Proposition 2 is the fixed point under explicit hypotheses (i)–(iii), with the 80-step counter saturation. C1 is reworded to the pair-trap versus position-trap measurement. | §4 |
| R5 | Proposition 3(b) is conditional on H1–H3, including the same-direction rotation. T_turn is one full turn (45 steps at ω_max, 89 at ½ω_max), plus 40 displacement steps. The 0.1 rad/s threshold is stated. N + 45 is the stronger registered bound, held on 97.8 %. (c) claims no liveness. The tail split into creep versus interrupted turn is not available without a replay, as the facts file §2 records. | §4, §5.4 |
| R6 | Registered verdicts are stated in the text and in supplement Table S1. They are: Exp. 4 P1 FAIL with its direction; Exp. 5 primary (immobility) FAIL on every arm; Exp. 9 H-CROSS FAIL, with the SPARSE and DRARE columns in Table S3; Exp. 10 H-PLAN-1 FAIL 1/3 and H-PLAN-3 PASS. The section is retitled "Stall metrics can improve while the task fails", and "failed in the registered direction" is removed. | §5, §6, supplement |
| R7 | The abstract and §1 phrasings follow the body. Deadlock is "≤ 0.022 for N ≤ 20 on all eleven policies tested", where Hor never had the recovery. N ≥ 40 uses length-matched runs. The penalties lower capture. The 262 states say "a turn would free" rather than "98 %". | Abstract, §1, §5.4 |
| R8 | All three asides are removed: "predicts the oscillations", "separate game" and "any filter … Kim et al.". | §2, §5.2 |
| R9 | Every main-text figure is redrawn at its printed width (433.6 pt) with 7–8 pt type. Draft 1's Fig. 1 is replaced by a game view. New Fig. 3(b) shows the N1 attribution. Draft 1's Fig. 4 (firings), Fig. 5 (dissociations), Fig. 6(c) and Fig. 8 move to the supplement. The Exp. 7 deadlock panel is cut at N ≤ 20, with the reason in the caption. | Figs 1–4, supplement Figs S1–S5 |
| R10 | Exp. 10 is reconciled and compressed. The pursuer is fully scripted, with orbit search and no learned part. The planner turned 16 of the 85 law-stalled greedy episodes into captures; the editorial's 13 was the net change, 190 → 203. The verdicts are stated. It is two sentences in §6 plus supplement Table S4. | §6, Table S4 |
| R11 | Free is the best of 5 seeds, and Aware the best of 3 RC seeds. Aware's recipe (features plus a 10⁻³ penalty) is disclosed, and so is the fact that **every trained family, Base included, uses the same recipe**. The RC arm values 0.564 / 0.550 / 0.444 are reported. The two Free values are different experiments; §5.1 uses only the Exp. 1 value. | §3, Table 1, §5.1 |
| R12 | The multiplicity policy sentence, 95 % Newcombe paired intervals on the headline deltas, an MDD of 0.05–0.08 at N = 500, and the fragility of small effects are all added. | §5 Protocol, §5.1, §5.4 |
| R13 | An observation and reward paragraph is added: 11 base dimensions, obstacle centres observed and radii not, walls only through own pose, the filter features with the counter saturating at 80, Obs's allowed-turn range, the reward and penalty, and discounting per decision. The supplement gives the full detail. | §3, supplement §2 |
| R14 | Protocol vocabulary (cell, arm, gate, hold-10, pins, tiers) is removed from the main text rather than defined. Base/Obs/Hor/ObsHor/Free/Aware are defined in Table 1 before first use. The attacker is "L", not "A". | throughout |
| R15 | Median escape is now 4–10 steps. The 10 466 vs 10 475 gap is explained in the Fig. S4 caption. The 1 085-step episode is distinct from the Fig. 1 episode. The visible-fraction pool is stated as 3.4 % on the hold-10 pool. | §5.3, §5.4, §7, Fig. S4 |
| R16 | The predictive filter is less conservative (no barrier-rate contraction), so "weaker" is not attributed to the filter's construction alone. Static geometry is a stated boundary condition. Scope moves into the Discussion. | §5.4, §7 |
| R17 | The email and repository URL remain placeholders, as the user decides them on Nov 10. PPO settings are in the supplement. The licence and the README's commit-hash note are still pending in the release repository. | — |

## P1 rounds

- **N1 was run.** It is pre-registered, 40/40 cells reproduce the archive, all three predictions pass, and the result is rule D-2 (D164). It is Exp. 3 of the revision.
- **N2 was not run.** The paper says "a baseline that also accepts turns was not run".

## Answers to the panel's six questions

1. **Law versus head share.** Over the four trained families against the attacker, 1 747 of 2 002 deadlock episodes are the law driving to its remembered point, 12 are the law chasing, and 243 are the head. Per family the head's share is Base 0.005, Obs 0.03, ObsHor 0.06 and Hor 0.30. Free's head share is 0.76 (0.85 against the greedy evader) and Aware's is 0.35 (0.58). The title now says "Learning-Enabled Controllers".
2. **The discounted advantage.** On the env-step clock it is A₀·E[γ^T] = 0.0136 at γ = 0.99 and 0.183 at 0.999. On the learner's per-decision clock it is 0.18 to 0.21 at 0.99. The paper carries the registered 0.0092 (per env step), the per-decision 0.29, and A₀·E[γ^T] = 0.18.
3. **Exp. 2/4's registered direction.** It is adopted: the signal was there and optimisation missed it. The discount mechanism is withdrawn.
4. **The observation.** Its contents are given in §3 and the supplement. The obstacle the pursuer is pressed against has its centre observed and its radius not observed. Walls are not observed except through the pursuer's own pose.
5. **Exact-tier feasibility.** It was lost on exactly one step per 500-episode Exp. 6 evaluation. Every sampled stalled state was on the exact tier, and the backup tier was never used. The paper says so.
6. **Selection and Aware's reward.** Free is the best of 5 seeds by gate capture (0.738). Aware is the best of the 3 seeds of the RC remedy, trained with 3 filter features, a 10⁻³ stall penalty and the CBF in the loop for 10 M steps.

## Not done

- **Release repository.** The crosswalk and the new figure scripts need to be mirrored to `safe-and-stuck`, and the licence is still open.
- **Escape-tail split.** This needs a replay.
- **N2.** Not run.
- **Recurrent arm.** Not run.
