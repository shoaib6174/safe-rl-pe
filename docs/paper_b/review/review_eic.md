# Phase 2 — EIC review (paper-visible)

Reviewer: Editor-in-Chief (Card 1: L4DC area chair, safety-filter background). Contract `reviewer/reviewer_full/v1`.
Materials read: `draft/main.tex`, `review/main_text_for_review.txt` (page references below follow this layout: main text pp. 1–10, references pp. 11–12), `draft/supplement.tex`, `draft/references.bib`, the eight figure PNGs, and `safe-and-stuck/README.md`. Phase 1 plan: `review/phase1_eic.md`, applied as committed; no dissent.
Numbers in the manuscript are archived registered results I cannot re-run; I judged whether the text states them faithfully and whether the inferences hold.

## Dimension Scores

| Dim | Score | Evidence (one sentence) |
|---|---|---|
| D1 methodology_rigor | **pass** | Pre-registration, frozen commits, paired seeds, exact binomial tests on discordant pairs, failed hypotheses reported as failures (p. 5 "Every verdict below is the registered one, including the failures"; p. 6 "The registered dwell test failed"), and a release repository whose README maps every paper name to a record path; no Phase 1 block or warn trigger is met (three seeds per family is stated as a limitation on p. 10), although confidence intervals are absent and multiplicity is not discussed (left to Reviewer 1). |
| D2 domain_accuracy | **pass** | The closest prior work (Grover et al. 2020/2021/2023; Reis et al. 2021; Tan and Dimarogonas 2024; Reis and Aguiar 2026) is cited and characterised accurately, Proposition 1 is explicitly labelled "known" (p. 4), and terminology is used correctly; two courtesy references are missing (below) but neither qualifies the positioning. |
| D3 argumentative_coherence | **warn** | Remark 2 (p. 4) states the discounted escape advantage "is at most γ^T A_0 … 0.0005 at γ = 0.99" and in the same sentence reports the measured discounted advantage as 0.0092, eighteen times the stated upper bound, so the formal centrepiece of contribution C2 is contradicted by the paper's own measurement as written; together with abstract phrasings stronger than the body (below) this meets my Phase 1 warn trigger "thesis stated with different strength in different sections / a central inference needing repair", not the block trigger, since the conclusion drawn (the signal is small) survives the repair. |
| D4 cross_disciplinary_relevance | **pass** | Definitions are gathered in §3 (p. 3) before use, the naming Base/Obs/Hor/ObsHor is self-explanatory and Free/Aware, G/H/A are defined where introduced; the boundary-crossing claim "learning does not remove it" is backed by four trained families and scoped to memoryless policies (p. 10); residual protocol jargon ("cell", "arm", "gate episodes", "hold-10", "the policies of the discovery") is a suggestion, not a trigger. |
| D5 writing_and_structure | **warn** | At the widths set in `main.tex` (Fig. 1 at 0.46, Fig. 4 right panel at 0.39, Fig. 6 at 0.46, Fig. 7 at 0.60 of the line width against native widths of 453, 422, 494 and 532 pt) the tick and legend type prints at an estimated 4–5 pt (assuming the PMLR 6-inch text block), which meets my Phase 1 warn trigger "several figures below comfortable legibility at printed size"; two placeholders remain (author email "[to confirm]", p. 1; "[repository URL]", p. 10). |

## Failure Condition Checks

| Condition | Expression | Fired? |
|---|---|---|
| F1 (sev 90) | any mandatory dimension scores block | **No** — D1 pass, D2 pass, D3 warn. |
| F2 (sev 70) | two or more mandatory dimensions warn or worse | **No** — exactly one mandatory warn (D3). |
| F3 (sev 60) | any high-priority dimension scores block | **No** — D4 pass. |
| F0 (sev 10) | every mandatory dimension scores pass | **No** — D3 is warn. |

No failure condition fires. The contract therefore neither forces acceptance (F0) nor a major revision (F2); the decision falls between them, and the issues that produced the D3 and D5 warns are each correctable without new experiments.

## Review Body

### Reviewer identity and focus
L4DC area chair, safety-filter background (unified-view school). Focus: fit to the learning-and-control readership; originality against the CBF-QP deadlock and undesired-equilibria literature; significance of the one-counter recovery; whether "the guarantee holds and the task fails" is earned by the evidence. Statistical detail and figure data are Reviewer 1's; the control-theoretic depth of Propositions 1 and 3 is Reviewer 2's.

### Overall recommendation
Minor Revision. Confidence 4/5.

### Summary assessment
The paper shows that a minimal-intervention CBF filter in front of a memoryless PPO pursuer produces an absorbing closed-loop stall: the filter zeroes the command, the observation freezes, the policy re-emits the blocked command for the rest of the episode, and the violation count stays at zero. It states the stall as the known normal-cone condition applied to a learned nominal, measures its geometry on 262 archived stalled states (98 % traps at the current heading, none a trap of the position), measures why discounting hides the escape, and reports nine pre-registered experiments in which observation features and a longer horizon raise capture without clearing the deadlock, a stall penalty makes the task worse, and a one-counter recovery that offers filtered forward then rotation candidates clears it (0.19–0.30 → ≤ 0.006) with capture up on nine of nine seeds and zero contacts, across an N sweep, a perturbation baseline and a second filter. Three dissociations show liveness metrics improving while the task worsens or stands still.

This is a well-executed, unusually honest empirical paper whose central framing is earned: the Free policy's capture falls from 0.728 to 0.396 under the filter with 0.540 deadlock episodes (p. 5), and fifteen passing specifications missed it (p. 10). The theory is thin by design and says so. The weaknesses are a bound in Remark 2 that the paper's own measurement violates as stated, several abstract phrasings a notch stronger than the body, and figures set too small to read in print. All are fixable in a revision cycle without re-running anything; none touches the archived evidence. I recommend minor revision.

### Strengths
**S1 — The framing is earned, and the failure is shown to be silent in the authors' own hands.** Figure 1 (p. 5) and the first paragraph of §1 give a concrete archived episode (1 106 identical commands, σ = 0.000, timeout; capture at step 262 under the recovery, zero violations in both). Page 10 reports that fifteen pre-registered specifications passed while measuring this artefact and that a person watching animations caught it. That admission is the most persuasive sentence in the paper and the reason the title is not rhetoric.

**S2 — The remedy comparison is designed so that the negative half is informative.** Experiment 4 (p. 7, Fig. 4) puts four learned families and the recovery on the same paired seeds against the same opponents and shows the learned remedies move capture (Obs 0.52–0.64, ObsHor 0.56–0.64) while deadlock stays 0.26–0.61, and the recovery moves both. The perturbation baseline (p. 8) is the right control for the "add a timeout" objection: random moving commands fail on five of five policies because the admissible set at a heading trap is the zero-speed segment (Fig. 3a), so the contribution is isolated to the rotation candidate, not the trigger.

**S3 — Failures are reported as failures, in the main text, with their registered bars.** The dwell test (p. 6), the 99 % projection-agreement bar missed at 98.1 % (p. 6), the Proposition 3(b) bound missed on two cells (94.5 %, 92.3 % against 95 %, p. 7), the predictive-filter bar met on two of four (p. 8), the recurrent arm confounded and withdrawn (p. 7), and the uninformative anti-spin test (p. 7). For a 10-page venue paper this is rare and should be rewarded.

**S4 — Positioning against the deadlock literature is accurate and modest.** §2 (p. 2) distinguishes multi-robot force equilibria (Grover et al.) from single-agent undesired equilibria of a fixed nominal (Reis et al.; Tan and Dimarogonas) and places the paper as the learned-nominal case with the filter left untouched. Page 4 says plainly "The characterisation is known".

**S5 — Reproducibility affordances are concrete.** The supplement crosswalk, the README's per-experiment table of runners, pre-registration files, frozen commits, seed streams and records, and construct gates that refuse to summarise until shared cells reproduce bit for bit.

### Weaknesses
**W1 — Remark 2's bound is contradicted by its own measurement (p. 4, Remark 2; abstract p. 1).**
*Problem.* The remark asserts the discounted advantage "is at most γ^T A_0" with T "the delay", instantiates T = 629 (the mean, Fig. 2b) to get 0.0005 at γ = 0.99, and in the same sentence reports the measured discounted advantage as 0.0092. With T a random delay, E[γ^T] ≥ γ^{E[T]} (Jensen), so γ^{mean T} A_0 is a lower bound on the expected discounted advantage, not an upper bound; the measured 0.0092 > 0.0005 is exactly what one expects from the early-capture mass visible in Fig. 2b. The abstract uses the measured 0.009, the remark headlines 0.0005, Fig. 2(c) draws the γ^629 curve as "the bound" with an unexplained "0.05 bar", and p. 10 says "Remark 2 is a bound".
*Why it matters.* C2 is one of four stated contributions; a control reader will check the inequality and find it fails on the paper's own numbers.
*Fix.* Either state the bound correctly (A_0 · E[γ^T] over the measured delay distribution, which the authors can compute from the 208 escapes already archived, and which will come out near the measured 0.0092), or drop "at most" and present Remark 2 as the measured discounted advantage with γ^{T̄} A_0 as the mean-delay heuristic; use one number consistently in abstract, remark and figure; define the 0.05 line in Fig. 2(c) or remove it. No new experiment is needed.
*Severity.* Major (for correctness of a stated result), small to repair.

**W2 — Abstract and §1 are a notch stronger than the body in three places.**
(i) Abstract: "clears the deadlock on three of three policy families (0.19–0.30 to 0.000)". The 0.19–0.30 range is the Base family only; Obs and ObsHor bare deadlock is 0.26–0.61 (p. 7); and ObsHor with the recovery is 0.002–0.006, not 0.000. Write "(0.19–0.61 to ≤ 0.006, bar 0.05)".
(ii) Abstract: "is insensitive to N over 5–80 steps". Figure 5(a) shows the 40-step deadlock indicator back at plain levels for N ≥ 40, which the text (p. 7) explains is by construction; the abstract's sentence is true of capture and of the length-matched run indicators, and should say so ("capture is insensitive to N over 5–80 and the matched long-run indicators stay below 0.02").
(iii) Abstract: "every learning-side remedy improves the task and none removes the failure, a stall penalty makes it worse" — the first clause includes the penalty that the second clause says worsens it; write "every learning-side remedy that improves the task leaves the failure in place; a stall penalty makes it worse".
*Fix.* Three sentence-level edits; the body already contains the accurate statements.
*Severity.* Major in aggregate (claims must sit inside the evidence at the point where they are made), trivial to repair.

**W3 — Figures 1, 3, 4(right), 6 and 7 will not be legible in print (figure sizes in `main.tex` ll. 135–207).**
Native widths are 453, 558, 422, 494 and 532 pt; set at 0.46, 0.72, 0.39, 0.46 and 0.60 of a ~433 pt line, the scale factors are 0.44, 0.56, 0.40, 0.40 and 0.49, so 10-pt matplotlib type prints at roughly 4–5.6 pt. Specific problems at that size: Fig. 3(b) two-line category labels and the four-bar legend; Fig. 4(right) rotated x labels and the 10^1–10^6 log axis; Fig. 6 and Fig. 1 axis ticks; Fig. 7(a) and 7(c) have no y-axis label at all (panel (a) mixes two rates and a capture rate on one unlabeled axis). Figure 5 (six panels at 0.68) is borderline but readable.
*Fix.* Re-render each figure at its printed width with 8–9 pt type (matplotlib `figsize` equal to the printed size, not scaled), or set Figs. 1, 6 and 7 at full line width and move Fig. 4(right) to the supplement, where the firing counts are quoted in full in the text anyway. Label the y-axes of Fig. 7(a), (c). Add to the Fig. 5 caption that the 40-step indicator is insensitive for N ≥ 40 by construction, or plot the length-matched indicator, so the panel does not visually contradict the text.
*Severity.* Major for a venue paper (results must be checkable from the printed figures), mechanical to repair.

**W4 — Two unsupported asides should be cited, evidenced or cut.**
(i) p. 2, §2: "Proposition 1 predicts the oscillations and timeouts these papers report" — no evidence is offered that the oscillations in Kim et al., Dawood et al. or Wang et al. are instances of the normal-cone condition; soften to "is consistent with".
(ii) p. 6: "the same stall was seen in a rule-based pursuer in a separate game" — unreferenced and unarchived; either point to a record in the release or remove.
(iii) p. 5: "Any filter that executes the nearest admissible command, including the closest-safe-action shield of Kim et al. (2025), satisfies the hypothesis of Proposition 1" — the proposition's proof uses convexity of A(s) (p. 3, "hence convex"); Kim et al.'s admissible set need not be convex. Restrict the sentence or state the convexity assumption.
*Severity.* Minor each; together they are the places where the paper reaches past its evidence.

**W5 — Contribution C1 and the "known" label sit awkwardly (p. 2, C1; p. 4).**
C1 reads "A characterisation of filter-induced stalls … (Proposition 1, Experiment 7)", and p. 4 says "The characterisation is known". The substance of C1 is R1 (position trap versus pair trap) and its measurement; make that the formal object. One option: keep Proposition 1 as "Fact 1 (Grover et al. 2023; Reis et al. 2021)" and promote R1 to a one-line definition plus the measured statement "98 % pair traps, 0 % position traps" as the paper's own result; reword C1 accordingly. This also answers the question the authors asked the panel (below).
*Severity.* Minor; it is a presentation decision, but it is the first thing a control-theory reviewer will press on.

### Detailed comments from the editorial angle

**Journal fit.** Squarely in L4DC's scope: a safety filter (control), a learned nominal (learning), a liveness failure at their interface, with four L4DC 2025 papers cited and engaged (Kim, Dawood, Wang, Arnström). The style — short, declarative, numbers in the text — suits the venue. Length: main text ends on p. 10 with references on pp. 11–12; this complies if the 2027 call keeps the 10-pages-excluding-references rule (authors to confirm against the call). The generative-AI statement is present (p. 10).

**Originality against Grover et al. / Reis et al. / Tan et al.** The normal-cone/KKT stall condition is Grover's; single-agent undesired equilibria of a CBF-QP with a fixed nominal are Reis et al. 2021 and Tan and Dimarogonas 2024; those papers modify the program. What is new here is (a) the nominal is learned through the filter and sees only a frozen observation, so the equilibrium is absorbing in observation space (R2) and invisible to the learning signal (Remark 2, once repaired); (b) the pair-trap/position-trap distinction with a measured 98/0 split; (c) the pre-registered demonstration that the obvious learning-side remedies do not remove it and a penalty inverts it; (d) the dissociations. (a)–(d) are not in the cited works and I know of no paper that does them. The novelty is empirical and diagnostic rather than theoretical, and the paper says so; that is an acceptable form of contribution at this venue provided C1 is reworded (W5).

**Significance.** For the control reader: a reminder, with numbers, that forward invariance plus zero violations is compatible with total task failure, and that the failure hides from metric-based evaluation. For the learning reader: observation features and horizon move the fixed point without removing it; a stall penalty teaches the agent to stop asking (R4, §6) — this is the result most likely to change practice. The recovery itself is a small mechanism, closer to classical deadlock resolution than to a new method; its significance is that it is the minimal stateful element that breaks R2 while leaving A(s) untouched, and that the paper shows why the random-perturbation alternative cannot work. That is enough for a contribution when it is framed, as it is, as the control arm of a diagnosis rather than as a method.

**Structural coherence.** Title → abstract → §1 → §8 tell one story. The discussion (p. 10) is candid about scope (memoryless; one game; one dynamics; two filters; three seeds). The over-promising is confined to the phrasings in W2.

**Title and abstract.** The title is good and accurate. The abstract is dense but complete; apply W2 and W1 (one number for the discounted advantage).

**Conclusion.** Matches the evidence except for the sentence "measured why the learner cannot see the way out", which should read "cannot see it at the training discount" given p. 10's own hedge that Remark 2 is a bound and memory is untested.

### Answers to the questions the authors asked the panel

**(a) Is Proposition 1 a result or a restatement?** A restatement, correctly labelled as such on p. 4, with one trivially new sentence (constant observation ⇒ fixed point ⇒ absorbing). The genuine addition is R1 and its measurement. Recommendation in W5: demote the known part to a Fact with citations, promote the pair-trap definition and the 98 %/0 % measurement to the paper's own numbered statement. Do not present the normal-cone condition under a "Proposition" heading in a contribution list that a CBF reviewer will read as a novelty claim.

**(b) Does the nine-experiment protocol section earn its space?** Yes, because pre-registration is part of the thesis (silent failure; fifteen passing specifications) and because Table 1 is what lets a reader place each number. It can lose about a third: the "Policies and experiments" paragraph of §3 (p. 3) and the Table 1 "Policies" column repeat each other, and the "N / cell" column could move to the caption as a sentence ("500 paired unless stated; 200 in Exp. 2, 100 in Exp. 7"). Keep the protocol paragraph on p. 5 as is; it is the methodological spine.

**(c) Are Figures 1, 3, 6, 7 legible at printed size?** No, at the widths currently set (W3, with scale factors). Fig. 1 at 0.46 and Fig. 6 at 0.46 are the worst; Fig. 3 at 0.72 is borderline with illegible category labels in panel (b); Fig. 7 at 0.60 is too small and lacks y-labels on (a) and (c). Fig. 4's right panel at 0.39 is the smallest in the paper. Re-render at printed size.

**(d) Does the three-dissociations section belong?** Yes for (a) and (b): the stall penalty and the correction-magnitude penalty are the two interventions a safe-RL reader would try first, and both improve a liveness metric while the task falls; that is the title's claim at the metric level and the paper's most transferable warning. (c) — replacing the intercept law with a planner — is about the authors' own pursuer architecture and is the weakest fit; compress it to two sentences ("removing the pins removed 94–99 % of law-held stalls and moved capture by at most 0.026") or move it to the supplement, and use the recovered space for the figures. Keep the "Consequence" paragraph (p. 10, top): it is the actionable recommendation.

**(e) Is the naming clear to a first-time reader?** Experiments 1–9 and Base/Obs/Hor/ObsHor: yes, the names say what they are. Free/Aware: adequate once defined (p. 3), but "Free" invites "free of what?" on first sight; "Unfiltered"/"Aware" or a parenthetical at first use in §1 would help. G/H/A appear only in Table 1 and in its caption, so they are harmless. The residual obstacle is protocol vocabulary, not policy names: "cell", "arm", "gate episodes", "hold-10", "the policies of the discovery" (p. 7), "decidable triggers" (p. 7) are used without definition; one sentence in the Protocol paragraph defining cell, arm and gate would clear it.

**(f) Do the claims stay within the evidence?** In the body, almost everywhere, and conspicuously so where hypotheses failed. The exceptions are listed in W1 (Remark 2's inequality), W2 (three abstract phrasings), and W4 (three asides). None requires new data.

### Questions for the authors
1. Please recompute the discounted escape advantage as A_0 · E[γ^T] over the archived delay distribution (208 escapes) and compare it with the measured 0.0092; which number will Remark 2, the abstract and Fig. 2(c) carry?
2. Proposition 1 uses convexity of A(s). Does the closest-safe-action shield of Kim et al. (2025) have a convex admissible set, and if not, what remains of the sentence on p. 5 that extends the proposition to it?
3. The recovery's firing count spans 363 to 152 675 per 1 500 episodes for equal capture (p. 7). What does a firing cost on a physical platform, and should the paper recommend the learned half (Obs/ObsHor) plus the recovery rather than the recovery alone?
4. What is the "0.05 bar" in Fig. 2(c)?

### Missing references (suggested, not required for positioning)
- K. P. Wabersich and M. N. Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems," *Automatica*, vol. 129, art. 109597, 2021, doi:10.1016/j.automatica.2021.109597. Confident. The paper names "a predictive filter" in §1 and builds one in §5.3 without citing the originating work; Hsu et al. 2024 covers it as a survey, but the direct citation belongs beside the first mention.
- M. Jankovic, M. Santillo and Y. Wang, "Multiagent systems with CBF-based controllers: collision avoidance and liveness from instability," *IEEE Transactions on Control Systems Technology*, 2024 (arXiv:2207.04915). To verify (volume, pages, year of the journal version). Relevant to the §2 sentence "These works fix the filter": it treats liveness under CBF controllers via instability of the deadlock equilibrium, a third route besides modifying the QP and adding a stateful recovery.
- Optional for the safe-RL reader: L. Brunke, M. Greeff, A. W. Hall, Z. Yuan, S. Zhou, J. Panerati and A. P. Schoellig, "Safe learning in robotics: from learning-based control to safe reinforcement learning," *Annual Review of Control, Robotics, and Autonomous Systems*, vol. 5, pp. 411–444, 2022, doi:10.1146/annurev-control-042920-020211. Confident.

### Minor issues
- p. 1 author block: "[to confirm]" email; p. 10: "[repository URL]"; README: licence "to be added". Resolve before submission.
- p. 1 §1 "1 106 steps (standard deviation 0.000)" vs p. 10 "1 085 consecutive blocked requests, command standard deviation 0.002": different episodes (Fig. 1 vs the discovery diagnostic); say so in one word on p. 10 to pre-empt the apparent inconsistency.
- p. 2 §1: "This is not the multi-robot deadlock of the CBF literature, a force equilibrium between agents whose constraint normals cancel the nominal" — fine, but Grover's condition is equally a single-agent statement when the constraints are static; the sentence slightly understates how much of Proposition 1 is already theirs (consistent with W5).
- p. 4 Proposition 3(b): "T_turn = 45 steps at ω_max" — 180° at 2.84 rad/s and Δt = 0.05 s is 22 steps; 45 presumably covers the worst-case heading plus the forward candidate's test. State what T_turn counts.
- Numbering: Proposition 1, Remark 2, Proposition 3 share a counter; the metadata says "two propositions". Either is acceptable; keep it consistent with any mention in the abstract or cover letter.
- Fig. 2(a) title "383 stall entries" and Fig. 2(b) "208 escapes, 8-cell pool" refer to different pools (hold-10 cells vs all cells, p. 6); the caption should say which pool feeds which panel.
- Fig. 4 left panel: the axis label "deadlock-episode rate vs learned attacker" is correct; the connecting lines between bare and recovery markers imply pairing by seed, which should be stated in the caption.
- §6 (c), p. 10: "Removing the pins removes a symptom" — the p-values 0.015, 0.035, 0.007 attach to capture changes of +0.026, −0.018, −0.022; make explicit that these are significant but small in both directions, which is the point.
- Supplement Table B1 column "QP" is explained in the caption only as "the exact weighted projection agrees with the filter"; carry the main text's "agrees in 98.1 %, yields forward motion in the remaining 1.9 %" into the caption so the supplement stands alone.
- README: "working-repository commit b99ee53" dates the snapshot; the paper's frozen commits refer to a different (private) repository, which is fine, but say in the README that those hashes are not resolvable in the release.

### Recommendation to peer reviewers
Reviewer 1 (methodology): check the numbers in §5.1–5.3 against `figures/figure_data*.json` and the records; weigh the absence of intervals and of a multiplicity statement against the pre-registered bars; confirm that the "three of three families" claim is paired by seed. Reviewer 2 (control): the Remark 2 inequality (W1); whether Proposition 1's convexity assumption is stated precisely enough for the discrete-time exact one-step test; whether R1 deserves to be the formal object (W5); Proposition 3(b)'s T_turn. Reviewer 3 (safe RL): whether "learning does not remove it" is over-claimed given the withdrawn recurrent arm; whether the firing-count variance across policies matters in practice; naming and figure legibility. Devil's advocate: the "add a timeout" objection is answered by the perturbation baseline (p. 8) — test whether that answer is complete, since a perturbation that accepted rotations was not run (p. 10 admits this).

## Editorial Decision

**Minor Revision.**

Reason. No contract failure condition fires: all three mandatory dimensions are pass or warn with only D3 at warn, so neither F1/F2 (major revision or reject) nor F0 (accept) applies, and the decision falls to editorial judgement within the contract's bounds. The paper fits L4DC, is original in its diagnostic claim relative to Grover, Reis and Tan, reports failures candidly, and earns its title. The defects are (1) Remark 2's inequality, contradicted by the paper's own measured 0.0092 and repairable from archived data; (2) three abstract phrasings stronger than the body; (3) figures set too small to read at their printed widths, plus two placeholders. Each is a text or rendering change with no new experiment, so re-review by the full panel is unnecessary; the EIC can check the revision. The C1/"known" wording (W5) and the two citations are recommended but not conditions.
