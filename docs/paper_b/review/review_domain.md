# Phase 2 — Peer Reviewer 2 (Domain), paper-visible review

Contract `reviewer/reviewer_full/v1`. Phase 1 commitment: `phase1_domain.md` (applied unchanged; no dissent). Page numbers refer to the compiled PDF (`main_text_for_review.txt`); line references are to `main.tex`. Material examined: `main.tex`, `supplement.tex`, `references.bib`, the eight figure PNGs at native and printed scale (pages 5–9 rendered from `main.pdf`), and `safe-and-stuck/pe2/shield.py` (read-only) to check the filter, recovery and predictive-filter descriptions.

## Dimension Scores

| Dimension | Score | Evidence (one sentence) |
|---|---|---|
| D1 methodology_rigor | **pass** | The design is adequate to test the control-theoretic claims — frozen policies, paired cells, an exact weighted-QP cross-check of the implemented filter on every stalled state (Experiment 7, p. 6), archived records and a name crosswalk — and the one gap between the analysed object (exact projection onto a convex set) and the simulated object (candidate enumeration judged by the exact one-step test) is acknowledged numerically (98.1 % / 1.9 %, p. 6); the absence of any sensitivity to the filter's weights $w_v/w_\omega$ and rate $\gamma$ is advisory, not a trigger. |
| D2 domain_accuracy | **warn** | Proposition 1's displayed condition drops the weight matrix and the executed command and attributes convexity of $\mathcal A(s)$ to the exact nonlinear one-step test rather than to the linearised rows (pp. 3–4); Remark 2's "at most $\gamma^T A_0$" is the wrong inequality direction and is exceeded eighteen-fold by the paper's own measurement (p. 4); Proposition 3(b) is a "typically" statement with unstated hypotheses (p. 4); the primary discrete-time-CBF and predictive-safety-filter sources are missing — every intended reading is recoverable, so warn, not block. |
| D3 argumentative_coherence | **warn** | The thesis is consistent and carried by the evidence, but the abstract makes claims the body qualifies away ("insensitive to $N$ over 5–80 steps"; "every learning-side remedy … none removes the failure" with memory untested, p. 10), p. 3 asserts that "Proposition 1 predicts the oscillations and timeouts these papers report" without argument, and p. 10 calls Remark 2 "a bound" after p. 4 reports a measurement above it. |
| D4 cross_disciplinary_relevance | **pass** | Control objects (barrier test, projection, invariance) and learning objects (policy, discount, observation) are defined where used (Sections 3–4) and the bridging mechanism (frozen input + discounted escape reward) is isolated empirically rather than asserted (Experiment 2; the Obs and Hor families); the one lapse is that the observation vector is never specified for a paper whose thesis is "the trap is in observation space". |
| D5 writing_and_structure | **warn** | Computed from the 200-dpi PNGs and the 6.0 in text block, Figures 1, 6, 7 and the right panel of Figure 4 print at roughly 4–5 pt type (0.39–0.46 linewidth of 6.3–6.9 in native images), and the rendered pages 5–9 confirm their labels and legends cannot be read without magnification while the pages are 40 % empty above the text. |

## Failure Condition Checks

| Condition | Expression | My scores | Fired? |
|---|---|---|---|
| F1 (sev 90) | any mandatory dimension scores block | D1 pass, D2 warn, D3 warn | **not fired** |
| F2 (sev 70) | two or more mandatory dimensions score warn or worse | D2 warn, D3 warn | **fired** (from this reviewer; the contract's cross-reviewer quantifier "majority" is applied by the synthesizer) |
| F3 (sev 60) | any high-priority dimension scores block | D4 pass | not fired |
| F0 (sev 10) | every mandatory dimension scores pass | D2, D3 warn | not fired |

Highest fired severity: F2 → `editorial_decision = major_revision`.

## Review Body

# Peer Review Report

## Manuscript Information
- **Title**: The Guarantee Holds and the Task Fails: Absorbing Stalls of Learned Policies under CBF Filters
- **Manuscript ID**: (pre-submission draft, L4DC 2027)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Control theorist working on control barrier functions and the closed-loop analysis of CBF-QP filters: deadlock and undesired equilibria, KKT and normal-cone characterisations, liveness of safety filters (Reviewer Configuration Card 3).

### Review Focus
Correctness, precision and novelty of Proposition 1 and its proof sketch against the KKT / normal-cone deadlock characterisations (Grover et al.) and the undesired-equilibria results (Reis et al.; Tan and Dimarogonas); the refinements R1–R4; whether Remark 2 is a bound; Proposition 3's invariance inheritance and $T_{\mathrm{turn}}$ argument; the predictive-filter composition; and coverage of the deadlock / liveness literature.

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision** — derived from the contract (F2 fired). The required changes are textual and mathematical, not experimental; a careful revision could be re-reviewed quickly.
- [ ] Reject

### Confidence Score
4 — the CBF-QP closed-loop analysis and the filter construction are squarely within my expertise; the RL training practice and the statistical reporting are Reviewer 1's and 3's.

### Summary Assessment
The paper reports that a discrete-time CBF projection filter in front of a memoryless learned unicycle pursuer produces a closed-loop fixed point at which the barrier guarantee holds, violations are zero, and the pursuer stands still for the rest of the episode; it states the stall as the normal-cone condition of CBF-QP deadlock, measures that 98 % of stalled states are traps of the policy–filter pair at the current heading and none a trap of the position, argues that the discounted reward for escaping is too small to train against, and shows across nine pre-registered experiments that learning-side remedies improve capture without clearing the stall while a one-counter recovery that offers filtered forward and rotation candidates clears it with the filter's guarantee intact. From the control-theory side the phenomenon is real, correctly positioned against the Grover and Reis/Tan lines, and the pair-trap distinction (R1) with the observation-space fixed point (R2) is a genuine addition. The formal statements, however, are below the precision the venue expects: Proposition 1 omits the weight matrix and the executed command, assigns convexity to the wrong set, and its "absorbing" clause assumes what it should derive; Remark 2's inequality points the wrong way; Proposition 3(b) is an empirical rate dressed as a proposition. The abstract also overstates two results that the body qualifies. These are fixable without new experiments, and the recovery's measured properties and the three dissociations stand. I recommend major revision on the strength of the fixes required to the formal section and the abstract.

## Strengths

### S1: The right object, correctly placed in the literature
Section 2 (p. 3) correctly distinguishes the multi-agent force equilibrium of Grover et al. (2020, 2021, 2023) from the single-agent undesired equilibria of Reis et al. (2021) and Tan and Dimarogonas (2024), and Section 4 (p. 4) says plainly "The characterisation is known". The novelty is honestly located in what the learned nominal adds, not in the KKT condition itself.

### S2: The pair-trap / position-trap split is measured, not asserted
Refinement R1 (p. 4) is operationalised with checkable definitions (p. 3) and measured with an independent exact weighted-QP solve on all 262 stalled states (Experiment 7, p. 6; Table B1 in the supplement). Figure 3(a) shows the admissible set at a stalled state collapsed to the zero-speed segment with a rotation available — exactly the geometry the argument needs. That the authors cross-checked their enumeration filter against the exact projection and report the 1.9 % disagreement is good control-engineering practice.

### S3: The recovery is argued from the same admissible set and is honest about its cost
Proposition 3(a) (p. 4) is the correct and minimal safety argument: every executed command is an output of the base filter, so nothing about the guarantee changes. I checked `RecoveryShield.filter` in `pe2/shield.py` (lines 726–770): every candidate, including the rotation fallback, is passed through `CBFShield.filter` on the current state, as the paper says. The cost (cycles 8–56 % of engagements, a 2.2 % tail on the escape bound, firing counts spanning three orders of magnitude) is reported in the Discussion (p. 10) rather than hidden.

### S4: The perturbation baseline is a genuine control
Section 5.2 (p. 8) registers the standard random-perturbation deadlock resolution and shows, with Figure 3(a), why it cannot work at a heading trap: a baseline that executes only moving commands cannot rotate out. "The recovery's contribution is the rotation candidate, not the trigger" is the correct reading and a useful one for practitioners.

### S5: The two penalty dissociations close the loop with R4
Experiments 3 and 5 (Section 6, pp. 9–10) show that a stall penalty and a correction-magnitude penalty move the nominal to zero — the "second kind of zero" of R4 — so the same fixed point is reached through the policy instead of the filter. This is a control-relevant warning about intervention penalties that the Pizarro Bejarano et al. (2025) line does not report because, as the paper notes, standing still is never a local optimum in their tracking task.

## Weaknesses

### W1: Proposition 1 is imprecise in three places
**Problem** (Section 4, p. 4, lines 103–111 of `main.tex`).
(i) *The displayed condition.* For the weighted projection $u^\star = P^W_{\mathcal A(s)}(u_\pi)$ the variational inequality $\langle u_\pi - u^\star, u - u^\star\rangle_W \le 0$ gives $W(u_\pi - u^\star) \in N_{\mathcal A(s)}(u^\star)$, i.e. $W(u_\pi - u^\star) = \sum_j \mu_j n_j + \nu$ with $n_j$ the outward normals of the active barrier rows and $\nu$ in the box's normal cone at $u^\star$. The paper writes $u_\pi = \sum_j \mu_j a_j$: it drops $W = \mathrm{diag}(150, 1)$, which is far from the identity and shears the cone in command space; it drops $u^\star$, which on the zero-speed face is $(0, \omega^\star)$ with $\omega^\star = \mathrm{clip}(\omega_\pi)$ and need not be zero; and it does not fix the sign of $a_j$ (the rows are $\Delta t\, a_j\!\cdot\!u + \gamma h_j \ge 0$, so the outward normal is $-a_j$).
(ii) *The set.* Page 3 defines admissibility by "the exact one-step barrier condition … on the true Euler successor", and page 4 then calls $\mathcal A(s)$ "the command box intersected with the half-spaces of the linearised barrier rows, hence convex". These are different sets. For the obstacle row, $h(q^+)=\lVert q^+ - c\rVert^2 - r^2$ with $q^+ = p + \Delta t\, v(\cos\theta,\sin\theta) + d(\cos(\theta+\Delta t\,\omega), \sin(\theta+\Delta t\,\omega))$ is quadratic in $v$ and trigonometric in $\omega$; the exact admissible set is not a polytope and need not be convex. The implemented filter (`shield.py`, module docstring and `filter`, lines 365–428) enumerates candidates from the *linearised* QP geometry and accepts them by the *exact* test, so the executed command is the nearest accepted candidate, not $P_{\mathcal A}(u_\pi)$ for either set. This is why Experiment 7 can find a 1.9 % disagreement, and why the sentence "Proposition 1 is stated with this share" (p. 6) exists — but a proposition cannot be stated with a share.
(iii) *The absorbing clause.* "If moreover $o(s)$ is constant along the stall, the closed loop is at a fixed point and the stall is absorbing" assumes the conclusion: constancy of $o$ along the stall presupposes that the stall continues. The content that is actually true and useful is: if the executed command is exactly $(0,0)$ — which on the zero-speed segment means the policy's requested turn rate projects to zero — and the evader is out of view with the filter features saturated, then $s_{t+1}=s_t$, $o_{t+1}=o_t$, and by induction the loop stays at $s$ until the evader enters the field of view. Note that a heading trap alone does not freeze the state: a nonzero admissible turn request would leave it within one $T_{\mathrm{turn}}$. The fixed point is the conjunction "heading trap" ∧ "zero-turn request" ∧ "frozen observation", and the paper's own Figure 3(a) (star at $\omega \approx 0$) shows exactly this.
**Why it matters**: these are the paper's only formal statements and the venue's control readership will check them first; as written, (i) is false for $\omega^\star \ne 0$ or $W \ne I$, (ii) misattributes convexity, and (iii) is a tautology.
**Suggestion**: define $\mathcal A_{\mathrm{lin}}(s)$ (box ∩ linearised half-spaces, a convex polytope) and state the stall condition for the exact weighted projection onto it as a *Lemma* ("stall condition, after Grover et al. 2023"): executed speed is zero iff $W(u_\pi - u^\star) \in N_{\mathcal A_{\mathrm{lin}}(s)}(u^\star)$ with $u^\star$ on the zero-speed face. Promote the fixed-point statement, with its three hypotheses, to Proposition 1 — that is the new object. Then report, as a measured fact, that the implemented enumeration agrees with the exact projection on 98.1 % of stalled states. The result is the same evidence, stated correctly, in fewer words.
**Severity**: Major.

### W2: Remark 2 is not a bound, and the inequality points the wrong way
**Problem** (Section 4, p. 4, lines 115–117; Discussion p. 10 "Remark 2 is a bound"; Figure 2(c) caption "The bound $\gamma^T A_0$"). If the escape's capture arrives after a random delay $T$, the discounted advantage from capture is $A_0\,\mathbb E[\gamma^T]$ (up to the no-escape branch's own discounting). Since $t \mapsto \gamma^t$ is convex, Jensen gives $\mathbb E[\gamma^T] \ge \gamma^{\mathbb E T}$: the quantity $\gamma^{629}\cdot 0.257 = 0.0005$ is a *lower* estimate of the discounted advantage, not an upper bound. The paper's own measurement confirms the direction: the measured discounted advantage is $0.0092$, eighteen times the claimed "at most" value, consistent with the wide delay distribution in Figure 2(b) (substantial mass at 100–400 steps). The reward function is also never stated in the main text (Experiment 5 mentions a "dense-reward column", so capture is not the only reward), which a reader needs to interpret $A_0$.
**Why it matters**: the mechanism contribution (C2) rests on this remark; a control reader who notices that the measured value exceeds the stated bound will discount the whole argument, although the conclusion — the signal is small at $\gamma = 0.99$ — survives on the measured 0.0092 and on the Hor family test.
**Suggestion**: restate as an estimate: "the discounted advantage is $A_0\,\mathbb E[\gamma^T]$, measured 0.0092 at $\gamma=0.99$; evaluating at the mean delay gives 0.0005 and understates it (Jensen)". Recompute the $\gamma = 0.999$ figure with $\mathbb E[\gamma^T]$ from the same 208 escapes rather than at the mean delay, relabel Figure 2(c) ("point estimate at the mean delay"), and replace "Remark 2 is a bound" on p. 10. State the reward function in Section 3 (one sentence) and say what the "0.05 bar" in Figure 2(c) is — it is not explained anywhere in the text.
**Severity**: Major.

### W3: Proposition 3(b) is an empirical rate dressed as a proposition
**Problem** (Section 4, p. 4, lines 121–125). "the admissible rotation reaches it within $T_{\mathrm{turn}} = 45$ steps at $\omega_{\max}$ … so the body typically leaves the position within $N + T_{\mathrm{turn}}$ steps." $T_{\mathrm{turn}} = 45$ is one full turn at $\omega_{\max}$ ($2\pi/(2.84 \times 0.05) = 44.2$). The hypotheses needed are not stated: (H1) some heading at $p$ admits forward motion (given); (H2) a rotation at $\omega_{\max}$ is admissible at *every* intermediate heading — not guaranteed, because the virtual control point sits 0.1 m ahead and sweeps a circle during a turn in place, so the soft rows change with heading; Table B1 certifies only $|\omega| \ge 0.5\,\omega_{\max}$ at the stalled heading, and at $0.5\,\omega_{\max}$ a full turn is 89 steps, not 45; (H3) once a heading admits motion, the projection of $(v_{\max}, 0)$ has filtered speed above $0.05\,v_{\max}$ — true because $w_v \gg w_\omega$ makes the projection prefer speed, but it should be said; (H4) the counter keeps the recovery firing during the turn — true because rotation steps are immobile, but it should be said. Separately, the bound is checked against a 0.10 m *displacement* while the forward candidate only needs filtered speed above 0.05 m/s; at that speed 0.10 m takes 40 steps, so even under H1–H4 the escape time can reach $N + T_{\mathrm{turn}} + 40 = 105$ steps. Figure 5(f) shows a tail to about 200 steps. The paper says the tail is "reported, not explained" (p. 5); part of it may be explained by slow creep and part by H2 failing mid-turn. The code also applies a threshold the text omits: if the larger admissible rotation is below 0.1 rad/s nothing fires (`RECOVERY_OMEGA_MIN`, `shield.py` line 75 and 763).
**Why it matters**: a proposition with "typically" in it is not a proposition; stated conditionally it becomes a correct, checkable bound, and the tail becomes diagnosable instead of unexplained.
**Suggestion**: state (b) as "under H1–H4 the recovery leaves the heading trap within $N + T_{\mathrm{turn}}$ steps and the position within $N + T_{\mathrm{turn}} + 0.10\,\mathrm{m}/(0.05\,v_{\max})$ steps", with $T_{\mathrm{turn}}$ computed at the rotation rate actually certified; report 97.8 % as the rate at which the stronger 65-step statement held; split the 2.2 % tail into slow-creep and rotation-interrupted triggers from the archived records if that is cheap. Mention the 0.1 rad/s threshold. Name the stall–recover–stall cycle of (c) as livelock and say explicitly that no liveness guarantee is claimed.
**Severity**: Major.

### W4: Three generality claims exceed the argument
**Problem**. (i) p. 3: "Proposition 1 predicts the oscillations and timeouts these papers report" — Proposition 1 characterises a zero-speed fixed point; oscillation (chattering against the filter, Oh et al. 2026) is a different closed-loop phenomenon and nothing in Section 4 derives it. (ii) p. 5: "Any filter that executes the nearest admissible command, including the closest-safe-action shield of Kim et al. (2025), satisfies the hypothesis of Proposition 1" — the normal-cone characterisation needs a convex admissible set and an inner-product metric; a generic shield's safe-action set has neither, and the paper's own second filter (a rollout acceptance test) is a case in point. (iii) p. 6: "the same stall was seen in a rule-based pursuer in a separate game" — no data, no citation.
**Why it matters**: each sentence invites a counter-example from a reader who knows the cited work, and none is needed for the paper's claims.
**Suggestion**: delete (i) or weaken to "the stall of Proposition 1 is one mechanism behind the timeouts these papers report"; restrict (ii) to filters whose admissible set is convex and whose selection is a projection, and say the predictive filter is handled by the composition result, not by Proposition 1; delete (iii) or cite the record.
**Severity**: Minor (each), Major in aggregate for D3.

### W5: The observation and the reward are never defined
**Problem**. Section 3 defines the game, filter and metrics, but not the observation vector $o(s)$ (which sensors, which filter features, the saturating stall counter of R2) or the reward. The thesis is "the trap is in observation space", and the discount argument depends on the reward.
**Why it matters**: a control reader cannot verify R2 ("the input is constant after at most 80 stalled steps") without knowing what is in $o$; a learning reader cannot interpret $A_0$ without the reward.
**Suggestion**: one paragraph in Section 3 (or a supplement table) listing the observation components and the reward terms; name the hold-10 search action for what it is — a 10-step sample-and-hold — since "both are memoryless" (p. 3) is not strictly true of a policy with a hold counter, even if Experiment 2 shows it does not matter.
**Severity**: Major (one paragraph fixes it).

## Detailed Comments

### Title & Abstract
The title is accurate and will travel. Two abstract claims need qualifying: "is insensitive to $N$ over 5–80 steps" — the body (p. 7) says the 40-step deadlock indicator is insensitive by construction for $N \ge 40$ and Figure 5(e) shows the immobile-step rate rising to 0.1–0.2 at $N = 80$; the robust statement is "capture is insensitive to $N$ and runs of at least $2N$ steps are cleared at every $N$". "every learning-side remedy … none removes the failure" should read "every tested learning-side remedy", since memory is untested (p. 10). "(0.19–0.30 to 0.000)" is the Base family's range, while "three of three policy families" includes ObsHor at 0.002–0.006; say so. "the task collapses" overstates capture changes of −0.1 to −0.3, and Experiment 6 is a dissociation of a different kind (metric improves, task flat).

### Introduction
The framing paragraph on p. 2 ("This is not the multi-robot deadlock … the nearest prior objects are the undesired equilibria") is correct. One precision point: in Reis et al. (2021) the undesired equilibria are already asymptotically stable, so "absorbing" is not what the learned nominal adds; what it adds is that the nominal is a trained function of a frozen observation and that training moves the fixed point without removing it (R2). Say that rather than "makes the equilibrium absorbing in observation space" (p. 3).

### Literature Review / Theoretical Framework
- **Coverage**: the deadlock / undesired-equilibria line (Grover ×3, Reis ×3, Tan) and the shielding line are covered. Missing are the primary sources for two constructions the paper uses: the discrete-time CBF condition $h(x_{k+1}) - h(x_k) \ge -\gamma h(x_k)$ (Agrawal and Sreenath 2017) and the predictive safety filter (Wabersich and Zeilinger 2021), which is what the first sentence of the paper and Section 5.3 mean by "predictive filter". See **Missing Key References**.
- **Integration quality**: the related-work section is a synthesis, not a list; each paragraph ends with the paper's delta. Good.
- **Research gap argument**: convincing for the fixed-nominal versus learned-nominal distinction; the claim "We found no prior work that separates filter-adjacent liveness metrics from task success under pre-registration" (p. 3) is safe because of the pre-registration qualifier but is weak as a gap statement — the dissociations' value is the control content (R4), not the registration.

### Methodology / Research Design (domain aspects only)
- **The filter description (p. 3) should match the implementation.** Say: "the filter enumerates the candidate optima of the linearised weighted QP (vertices, projections onto active rows, the zero command and the clipped nominal), accepts a candidate if the exact one-step condition holds on the true Euler successor, and executes the nearest accepted candidate in the weighted norm". That is what `shield.py` does and it is what makes Experiment 7's 98.1 / 1.9 split meaningful.
- **Forward invariance.** The paper never states the safe set or the one-line invariance argument (with $h(x_{k+1}) \ge (1-\gamma)h(x_k)$ and $h(x_0) \ge 0$, $h(x_k) \ge 0$ for all $k$ while the exact tier is feasible). It should, and it should say that exact-tier feasibility is not proven — the relaxed and backup tiers exist because it can fail — so that the safety claim in the paper is the empirical zero-contact record and Proposition 3(a) inherits exactly that. [Severity here is advisory; this is a precision point, not a field-norm claim.]
- **Second filter (Section 5.3).** From `PredictiveShield` (lines 880–1132): the candidate set is the *same* enumeration as the CBF filter; a candidate is accepted iff holding it for 4 steps and then stopping is contact-free with body clearance $\ge$ margin (0). Two consequences for the text: (a) "sampling lookahead" is a misnomer — nothing is sampled; call it a rollout-based filter with a stop backup; (b) at margin 0 and with no $\gamma$-contraction this filter is strictly less conservative than the CBF filter, so "the loop is present but weaker under the predictive filter at this lookahead" (p. 8) conflates filter class with conservativeness. Either say that the comparison is not at matched conservativeness or add one cell at a matched margin. The composition claim itself (every candidate passes the predictive test; 39 162 commands re-filtered without rejection) is sound, and Proposition 3(a) carries over to it.
- **Filter parameters.** $w_v = 150$, $w_\omega = 1$, $\gamma = 0.2$ and the 0.1 m control-point offset are fixed without comment. The projection metric decides whether the filter slows or steers on approach, and the hard body rows have $a_\omega = 0$ (`shield.py` lines 809–830: "translating is the only way to change a body barrier"), which is the structural reason a heading trap is a zero-speed segment. One sentence connecting these would make R1 mechanistic rather than observational; a sensitivity row would be better but I do not require it. [FIELD-NORM UNVERIFIED as a requirement; advisory.]

### Results / Findings
- **Experiment 7 (p. 6).** Solid. State that the 262 stalled states come from one opponent (the learned attacker) and say whether the shares hold against the greedy evader; the abstract's "98 %" is quoted without that scope.
- **Experiment 8 (p. 7).** The $T_{\mathrm{turn}}$ bound discussion belongs with W3. "The cost is cycles, not immobility" is the right summary; Grover et al.'s "after" analysis and the Lee et al. (2025) deadlock-prevention line are the places to connect the cycle rate to.
- **Experiment 9 (p. 8).** See the conservativeness point above.
- **Figures**: see Minor Issues; legibility is a real problem at printed size (question (c) below).

### Discussion
The limitations paragraph (p. 10) is honest. Add the three the review raises: the implemented filter is an enumeration that agrees with the exact projection on 98 % of stalled states; the predictive filter is not matched in conservativeness; the recovery can livelock and no liveness guarantee is claimed.

### Conclusion
"showed that its geometry is the admissible set's" is supported (98 %). "measured why the learner cannot see the way out" is supported by Experiment 2 together with the Obs/Hor tests, once Remark 2 is restated as an estimate. The last sentence is good.

### References
Complete and verified by the authors' own protocol; see Missing Key References for four additions and three to verify. Nakamura et al. (2025) and Lee et al. (2025) are in the `.bib` but not cited in the text — either cite or drop.

## Answers to the authors' questions to the panel

**(a) Is Proposition 1 a result or a restatement?** As written, a restatement of the KKT / normal-cone condition (Grover et al. 2020, 2021, 2023; and, for the single-agent static case, Reis et al. 2021 and Tan and Dimarogonas 2024 obtain the same stationarity condition for their equilibria), followed by a clause that assumes what it should derive. The authors already say "The characterisation is known". The new content is in R1 and R2: that the trap is of the pair (policy, filter) at a heading and not of the position, and that the closed loop (policy ∘ observation, filter) has a fixed point whenever a heading trap meets a zero-turn request with a frozen observation — a statement about the *composition* that the fixed-nominal literature does not make, because for a fixed nominal the question "does training move the fixed point" does not arise. My recommendation (W1) is to demote the known condition to a lemma with correct hypotheses and promote the fixed-point statement, with its three hypotheses, to Proposition 1. That turns a restatement into a modest but genuine result and removes the "stated with this share" sentence.

**(b) Does the nine-experiment protocol section earn its space?** The *Protocol* paragraph (p. 5, nine lines) does: the paper's meta-claim in the Discussion and its credibility with a control reader both rest on it. Table 1 does not, in the main text: it duplicates the "Policies and experiments" paragraph of Section 3 and the supplement's crosswalk, and its half page is better spent on the observation and reward definitions (W5) and on a correctly stated formal section (W1–W3). Experiment 1 (p. 5) and the failed dwell test (p. 6) can each lose half their length.

**(c) Are Figures 1, 3, 6, 7 legible at printed size?** No for 1, 6 and 7; marginal for 3. Computed from the PNGs (200 dpi) and the 6.0 in text block: Figure 1 is 6.3 in native printed at 2.76 in (scale 0.44, about 4 pt type); Figure 6 is 6.9 in native printed at 2.76 in (scale 0.40, about 4 pt); Figure 7 is 7.4 in at 3.6 in (scale 0.49, about 5 pt); Figure 3 is 7.8 in at 4.3 in (scale 0.56, about 5.5 pt), and its legend floats 1.5 cm below panel (a); the right panel of Figure 4 is 5.9 in at 2.3 in (scale 0.40, about 4 pt); Figure 5 is about 6 pt. The rendered pages 5–9 confirm this and show the figures as narrow bands with 40 % of the page empty above the text. Regenerate each figure at its printed width with 7–8 pt type (for example `figsize=(6, h)` at full text width, font 8), and let Figures 1, 5, 6 and 7 take the full text width — their aspect ratios make the height cost small.

**(d) Does the three-dissociations section belong?** Dissociations (a) and (b) do: they are the second kind of zero of R4, reached through the nominal, and (b) is a control-relevant warning about correction-magnitude penalties that generalises beyond this game (any task in which standing still is admissible). Dissociation (c) is about the authors' own intercept law — a symptom of this pursuer design — and belongs in the supplement or in one sentence. The "Consequence" paragraph is a recommendation, not a result, and belongs in the Discussion. Half the current length.

**(e) Is the naming clear to a first-time reader?** Base / Obs / Hor / ObsHor: yes. Free / Aware: no — "Aware" carries filter-state features and so does "Obs", so a first-time reader will conflate them; the difference is provenance (10 M steps, the discovery experiments), which the names do not say. Consider "Free" → "Unfiltered" and "Aware" → "Obs-10M" or a footnote in Table 1. G / H / A: fine, though "H" (hugging) and "Hor" (horizon) sit uncomfortably close on the same page. Experiments 1–9: fine with Table 1, but the narrative visits them out of order (7 before 3 and 4); one sentence saying the numbering is chronological would help. Four labels are used for two concepts — "filter trap at heading", "trap at heading", "pair trap", and "Trap" (Table B1) — unify.

**(f) Do the claims stay within the evidence?** The central claims do (fixed point exists; 98 % heading traps, 0 % position traps; recovery clears the deadlock indicator on all tested policies with zero contacts; perturbation baseline fails for a stated geometric reason; penalties move the zero to the nominal). The exceptions are listed in W2, W4 and the abstract comments: the $N$-insensitivity claim, "every learning-side remedy", "Proposition 1 predicts the oscillations", "Remark 2 is a bound", the hypothesis claim about arbitrary nearest-admissible shields, the unsupported "separate game" sentence, and "collapses". All are repairable by wording.

## Questions for Authors
1. At the 262 stalled states, what is the executed turn rate $\omega^\star$? If it is zero in (nearly) all of them, the fixed-point statement in W1(iii) is exactly what you measured and should be stated as such; if some have $\omega^\star \ne 0$, those states are not fixed points and the stall-run definition is catching slow self-resolving rotations.
2. Do the heading-trap and pair-trap shares change with the filter's $\gamma$ (barrier rate) and with $w_v/w_\omega$? Your earlier work on the filter margin suggests they do; even one sentence of sensitivity would turn R1 from an observation into a mechanism.
3. For the 2.2 % of triggers outside the $N + T_{\mathrm{turn}}$ bound: how many are slow creep (forward candidate accepted at speed just above 0.05 m/s) and how many are turns interrupted by the soft rows becoming active at intermediate headings (H2 in W3)?
4. Is exact-tier feasibility ever lost during evaluation (relaxed or backup tier executed on any step, not only stalled ones)? If never, say so; if sometimes, the "guarantee" in the title is the empirical zero-contact record and should be described as such.

## Missing Key References
Confident (authors to verify under their own protocol before citing):
- Ayush Agrawal and Koushil Sreenath, "Discrete Control Barrier Functions for Safety-Critical Control of Discrete Systems with Application to Bipedal Robot Navigation", *Robotics: Science and Systems XIII*, 2017, DOI 10.15607/RSS.2017.XIII.073 — the discrete-time condition $h(x_{k+1}) - h(x_k) \ge -\gamma h(x_k)$ that the base filter implements; cite at "discrete-time CBF projection" (p. 3).
- Kim P. Wabersich and Melanie N. Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems", *Automatica* 129:109597, 2021, DOI 10.1016/j.automatica.2021.109597 — the predictive safety filter; cite at the first sentence of the Introduction and in Section 5.3.
- Li Wang, Aaron D. Ames and Magnus Egerstedt, "Safety Barrier Certificates for Collisions-Free Multirobot Systems", *IEEE Transactions on Robotics* 33(3):661–674, 2017, DOI 10.1109/TRO.2017.2659727 — the CBF-QP multi-robot filter in whose closed loop the deadlock line of Grover et al. arose; cite with the Grover references (p. 3).
- Peter Trautman and Andreas Krause, "Unfreezing the robot: Navigation in dense, interacting crowds", *IEEE/RSJ IROS*, 2010, DOI 10.1109/IROS.2010.5654369 — the freezing-robot problem, the classic "safe but stuck" lineage outside the CBF literature; one clause in the Introduction.

To verify (I am confident of the works, less of the bibliographic fields):
- Thomas Gurriet, Andrew Singletary, Jacob Reher, Laurent Ciarletta, Eric Feron and Aaron D. Ames, "Towards a Framework for Realizable Safety Critical Control through Active Set Invariance", *ACM/IEEE ICCPS*, 2018 (DOI 10.1109/ICCPS.2018.00018, to verify) — backup-controller / rollout-based safety filtering, which is the construction of the second filter (hold then stop).
- Pol Mestres and Jorge Cortés, "Optimization-Based Safe Stabilizing Feedback with Guaranteed Region of Attraction", *IEEE Transactions on Automatic Control*, 2023 (volume/pages/DOI to verify) — undesired equilibria of CBF-CLF QPs and a fix, in the same line as Reis et al. and Tan and Dimarogonas.
- Jun Zeng, Bike Zhang and Koushil Sreenath, "Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function", *American Control Conference*, 2021 (DOI 10.23919/ACC50511.2021.9483029, to verify) — the DCBF decay rate as a design parameter, relevant to the fixed $\gamma = 0.2$.

## Minor Issues

### Language / Terminology
- p. 3, "both are memoryless": the search head's 10-step hold is a counter; say "memoryless apart from a 10-step sample-and-hold, which Experiment 2 shows is immaterial".
- p. 3, "sampling lookahead": nothing is sampled; "rollout-based filter (hold 4 steps, then stop)".
- p. 4, R2 "bounded filter features whose stall counter saturates at 80 steps": the feature is never defined in the main text (W5).
- p. 4, Proposition 3(b) "the admissible rotation": there are two candidate rotations and a 0.1 rad/s floor (`RECOVERY_OMEGA_MIN`); say which is executed when both are admissible (the code takes the larger executed $|\omega|$, ties to positive).
- "deadlock" is used both for the metric (deadlock episode) and the phenomenon; the metric uses are clear from context, but the first use on p. 3 should say "we use *deadlock episode* as a metric name; the phenomenon we characterise is a stall".

### Citation Format
- Nakamura et al. (2025) and Lee et al. (2025) are in `references.bib` but never cited; `kim2025realizable` is cited twice in one sentence on p. 3 (fine) — no format errors found.
- Georgescu et al. (2026) and Oh et al. (2026): the bib carries both a venue and an arXiv note; confirm the venue versions exist before submission (they post-date my confidence).

### Figures and Tables
- Figure 2(c): the "0.05 bar" is unexplained; relabel "bound" as "point estimate at the mean delay" (W2).
- Figure 3: the legend for panel (a) floats 1.5 cm below the panel row with empty space above it; place it inside panel (a) or to its right.
- Figure 4: right panel is a quarter of the width of the left with 4 pt type; make it full width below the left panel or move the firing counts into the text (they are already there).
- Figure 5(f): label the dashed line with its derivation ($N = 20$ plus 45 steps).
- Figure 7(a): the y-axis mixes three quantities (stall-step rate, immobile-step rate, capture) with no label; give it "rate" and separate the panels or use a dumbbell per quantity.
- Table 1: "Measures" column mixes quantities and counts ("772 stall entries", "18 arms"); either make it "What it tests" or split.

### Layout
- Pages 5, 6, 7, 9: the figures occupy a narrow band and leave the upper 35–40 % of the page empty; widening the figures to the text width fixes both problems at once.
- The main text ends at the top of page 10 with the AI-use statement; the page budget is not the constraint the figure widths suggest.

## Dimension Scores (template rubric, 0–100; the contract scores above are the binding ones)

| Dimension | Score | Descriptor | Notes |
|---|---|---|---|
| Originality (20 %) | 62 | Adequate–Strong | Known stall condition; new pair-trap / observation-space framing, recovery and penalty dissociations |
| Methodological Rigor (25 %) | 70 | Strong | From the domain angle; the exact-QP cross-check is good practice; filter parameters fixed without sensitivity |
| Evidence Sufficiency (25 %) | 68 | Strong | Registered and archived; one game, one dynamics model, two filters at unmatched conservativeness |
| Argument Coherence (15 %) | 60 | Adequate | Thesis consistent; abstract and three sentences overreach; Remark 2 direction |
| Writing Quality (15 %) | 50 | Adequate text, Weak figures | Prose is tight; figures unreadable at printed size |
| Literature Integration (R2) | 62 | Adequate | Deadlock line covered and synthesised; primary DCBF and predictive-filter sources missing |
| **Weighted Average** | **63.4** | — | Decision taken from the contract's failure conditions, not from this number |

## Editorial Decision

**Major Revision.** Failure condition F2 fired (D2 domain_accuracy = warn; D3 argumentative_coherence = warn); F1 and F3 did not fire. The reason, in one paragraph: the phenomenon, the measurements and the recovery are sound and well positioned against the CBF-QP deadlock and undesired-equilibria literature, but the paper's two propositions and its remark are stated below the precision a control audience will hold them to — Proposition 1 drops the weight matrix and the executed command, attributes convexity to the wrong set, and its absorbing clause is circular; Remark 2's inequality is in the wrong direction and is contradicted by the paper's own measurement; Proposition 3(b) is an empirical rate with unstated hypotheses — and the abstract and three sentences in the body claim more than the evidence (N-insensitivity, "every" remedy, "predicts the oscillations", "Remark 2 is a bound"). All of this is repairable by rewriting the formal section, defining the observation and reward, restating the second-filter comparison, adding four primary references, and regenerating Figures 1, 3, 4, 6, 7 at printed width; no new experiments are required for the domain concerns. I would expect to recommend acceptance of a revision that does these things.
