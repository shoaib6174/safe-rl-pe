# Editorial Decision — Paper B, L4DC 2027 submission draft (panel review, round 1)

## Manuscript Information
- **Title**: The Guarantee Holds and the Task Fails: Absorbing Stalls of Learned Policies under CBF Filters
- **Draft reviewed**: `docs/paper_b/draft/main.tex` at commit b99ee53 (main text pp. 1–10, references pp. 11–12), `supplement.tex`, `references.bib`, figures F1–F8, release repository `safe-and-stuck` (5485369)
- **Review mode**: ARS academic-paper-reviewer, full (contract `reviewer/reviewer_full/v1`, panel 5, two-call discipline: each reviewer wrote a paper-blind pre-commitment `phase1_<role>.md` before opening the manuscript, then `review_<role>.md`)
- **Decision date**: 2026-11-02 · **Review round**: 1
- **Synthesizer disclosure**: the synthesis below was performed by the orchestrating session, which also wrote the draft. The decision itself is the contract's arithmetic (Section 2) and does not depend on that judgement. The arbitration of the Devil's Advocate's critical finding (Section 4) rests on checks of the code and the archive that are listed there so that the authors can repeat them; the authors should weigh the arbitration with that conflict in mind.

---

## 1. Decision

### MAJOR REVISION

Contract failure condition F1 fired (one reviewer scored a mandatory dimension `block`); its action is `reject_or_major_revision`, and Major Revision is chosen within that disjunction because every reviewer, the Devil's Advocate included, finds the central empirical results (Experiments 4, 7, 8, 9) reproducible from the archive and the defects repairable in one cycle, most of them from text and archived records alone. A Devil's Advocate CRITICAL finding exists (Section 4), so Accept was not available under the panel rules in any case.

---

## 2. Scoring matrix and failure-condition evaluation (mechanical)

| Dimension (priority) | EIC | R1 Methodology | R2 Domain | R3 Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| D1 methodology_rigor (mandatory) | pass | warn | pass | pass | warn |
| D2 domain_accuracy (mandatory) | pass | warn | warn | pass | warn |
| D3 argumentative_coherence (mandatory) | warn | warn | warn | warn | **block** |
| D4 cross_disciplinary_relevance (high) | pass | pass | pass | warn | warn |
| D5 writing_and_structure (normal) | warn | warn | warn | warn | warn |

| Condition | Severity | Quantifier | Expression | Evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 90 | any | any mandatory dimension scores 'block' | DA: D3 block | **yes → reject_or_major_revision** |
| F2 | 70 | majority (≥ 4 of 5) | two or more mandatory dimensions 'warn' or worse | holds for R1, R2, DA (3 of 5) | no |
| F3 | 60 | any | any high-priority dimension scores 'block' | D4 never block | no |
| F0 | 10 | all | every mandatory dimension 'pass' | fails for every reviewer (D3 ≥ warn ×5) | no |

Highest-severity fired condition: F1 → **Major Revision** (chosen within the disjunction; see Section 1). Note that D3 is at least `warn` for all five reviewers: the coherence of the learning claim with the evidence is the panel's unanimous concern, differing only in severity.

Individual recommendations: EIC Minor Revision (confidence 4/5); R1 Major (4/5); R2 Major (4/5); R3 Minor (4/5); DA Major (1 CRITICAL, 5 MAJOR, 10 MINOR).

---

## 3. Consensus analysis (four non-DA reviewers; DA corroboration noted separately)

Sub-claims decomposed from the four reports; `raised`/`corroborated` counted over the four, silence is not agreement.

| ID | Sub-claim | EIC | R1 | R2 | R3 | DA | Disposition |
|---|---|---|---|---|---|---|---|
| SC-1 | Remark 2's inequality is mis-derived: γ^{E[T]}·A₀ = 0.0005 is a lower estimate of A₀·E[γ^T] (Jensen); the paper's own measured 0.0092 exceeds it 18×; Fig. 2(c) "bound" and p. 10 "Remark 2 is a bound" follow | raised (W1) | corroborated (W1) | raised (W2) | — | M1 | **CONSENSUS-3** (R3 silent) |
| SC-1b | "T ≈ 629 = delay from escape to capture" is the mean remaining horizon after an escape over 97 escapes of which 68 timed out; delay to capture on the 29 captured entries is 346 (median 314); Fig. 2(b) caption/axis/pool disagree | — | raised (W1, spot-check 5) | — | — | — | single-reviewer, archive-verified |
| SC-2 | "Why learning does not remove it" is over-stated as a general mechanism: the discount argument is not sufficient (Hor at γ = 0.999 keeps deadlock 0.29–0.61), memory is untested, and Exp. 2's registered direction ("optimisation, not credit assignment") is unreconciled | raised (conclusion sentence) | raised (W2(i), D3) | corroborated ("memory untested") | raised (W1) | C1 | **CONSENSUS-4** |
| SC-3 | Abstract/§1 phrasings exceed the body: "(0.19–0.30 to 0.000)" is Base-only (ObsHor ≤ 0.006); "insensitive to N over 5–80" holds for capture and the length-matched run indicators, not the K = 40 indicator; "every learning-side remedy improves the task" includes the penalty (and recurrent) arms that worsened it | raised (W2) | raised (W6, (f)) | raised (W4-adjacent, abstract) | raised ((f)) | m1, m2, M3 | **CONSENSUS-4** |
| SC-4 | Figures composed at full width and printed at 0.39–0.72 of it: type at 3.4–5.6 pt; Fig. 7(a) no y-label, 7(c) legend collides with the axis; Fig. 4-right tick labels unreadable | raised (W3) | raised (D5: Figs 6, 7) | raised (Figs 1, 6, 7, 4-right) | raised (W4) | m9 | **CONSENSUS-4** (which figures are worst differs; see Disagreement 1) |
| SC-5 | Proposition 1 as written is a restatement of the known normal-cone/KKT condition; demote it to a Lemma with citation and promote the paper's own object (pair-trap vs position-trap measurement, or the fixed point with its hypotheses) | raised (W5) | raised ((a)) | raised (W1) | raised ((a)) | (a) | **CONSENSUS-4** |
| SC-5b | The displayed condition drops W = diag(150, 1) and u* and does not fix the sign of the normals; convexity is attributed to the exact nonlinear one-step set instead of the linearised polytope; the implemented filter is an enumeration accepted by the exact test; the "absorbing" clause assumes its conclusion | — | — | raised (W1 i–iii) | — | — | single-reviewer, within R2's expertise (confidence 4), checkable |
| SC-6 | Three asides exceed the evidence: "Proposition 1 predicts the oscillations and timeouts these papers report" (p. 2); "the same stall was seen in a rule-based pursuer in a separate game" (p. 6); "Any filter … including Kim et al.'s shield satisfies the hypothesis" needs convexity (p. 5) | raised (W4) | — | raised (W4) | raised ((f), minor) | M5, m4 | **CONSENSUS-3** (R1 silent) |
| SC-7 | Protocol vocabulary undefined (cell, arm, gate, hold-10, pins, dense-reward column, decidable triggers, exact tier); Table 1's "six" opponents and "its anchored fine-tune" dangle; Base/Obs/Hor/ObsHor used before they are defined | raised ((e)) | corroborated ((e): "Aware" under-describes, "A" collides with A₀/𝒜, experiments out of order) | — | raised (W5) | (e) | **CONSENSUS-3** (R2 silent) |
| SC-8 | Two registered verdicts are missing (Exp. 2 P1 FAILS with its direction; Exp. 5 H-CROSS FAILS, SPARSE/DRARE columns unreported) and the "pre-registered dissociations" are post-hoc readings of measurements whose registered hypotheses failed; Exp. 3's registered primary metric and verdict are not named | — | raised (W2, W3) | — | — | M4, m6 | single-reviewer among the four, archive-verified, DA-corroborated |
| SC-9 | No multiplicity policy over ~60 paired tests, no confidence intervals, no power/MDD statement; the fragile p-values are exactly the Exp. 6 "flat" deltas | corroborated (D1 note) | raised (W4) | — | — | D1 | corroborated finding |
| SC-10 | The observation vector and the reward are never defined, although the thesis is "the trap is in observation space" and A₀ depends on the reward | — | — | raised (W5) | raised (W3) | — | corroborated finding |
| SC-11 | Proposition 3(b)'s hypotheses are unstated (rotation admissible at every intermediate heading; Table B1 certifies only ½ ω_max, a full turn then takes 89 steps, not 45; the 0.10 m displacement at 0.05 v_max adds 40 steps; the 0.1 rad/s code threshold is unmentioned); T_turn = 45 should say "one full turn" | raised (minor) | raised (minor) | raised (W3, Major) | — | — | **CONSENSUS-3** at different severities (R2's version is a superset of the others'; adopt it) |
| SC-12 | Exp. 6 (planner law) is the weakest third of §6: compress or move to the supplement (EIC, R3), relabel with its registered verdict (R1), and reconcile with Exps 4/8 since removing 94–99 % of law-held stalls left capture flat (DA) | raised ((d)) | raised (W3) | — | raised ((d)) | M4 | **CONSENSUS-3**; actions compatible (see Disagreement 2) |
| SC-13 | Free and Aware are best-of-seed selected checkpoints and Aware was trained with filter-state features AND a 10⁻³ stall penalty; §3 says neither; Free is quoted at 0.396 (Exp. 1) and 0.426 (Exp. 8) without comment | — | raised (W5) | — | — | m3 | single-reviewer, archive-verified |
| SC-14 | Missing references: Wabersich & Zeilinger 2021 (EIC, R2); Agrawal & Sreenath 2017, Wang–Ames–Egerstedt 2017, Trautman & Krause 2010 (R2); Koren & Borenstein 1991, a navigation-stack recovery reference, Singh–Jaakkola–Jordan 1994 (R3); optional Jankovic et al., Brunke et al. 2022, Gurriet et al. 2018, Mestres & Cortés, Zeng et al. 2021, Sutton–Precup–Singh 1999, Hausknecht & Stone 2015, Jansen et al. 2020 | raised | — | raised | raised | — | CONSENSUS-3 that the related-work map is incomplete; each item must pass the authors' verification protocol before citation |
| SC-15 | Fig. 2's three denominators (383 / 208 / 772; 97 escapes) need one sentence | raised (minor) | raised (minor) | — | raised (minor) | — | CONSENSUS-3 (minor) |
| SC-16 | The second filter is less conservative (margin 0, no contraction), so "the loop is weaker under the predictive filter" conflates filter class with conservativeness | — | — | raised | — | alt. 5 (horizon sweep) | single-reviewer |
| SC-17 | Static geometry is a boundary condition of the "guarantee holds" dissociation (moving agents, lanes) | — | — | — | raised (Q4) | stakeholder note | single-reviewer, scope sentence |

### Points of disagreement and resolutions

**Disagreement 1 — which figures are illegible.** R1 finds Fig. 1 legible and Figs. 6, 7 not; the DA finds Fig. 6 legible and Fig. 5 the worst; EIC, R2, R3 list Figs. 1, 3(b), 4-right, 6, 7. *Type*: severity. *Resolution*: moot once every figure is regenerated at its printed width with 8–9 pt type (the fix every reviewer proposes); treat Figs. 1, 3(b), 4-right, 5, 6, 7 as the set to re-render and check the result at page size.

**Disagreement 2 — what to do with Experiment 6.** EIC and R3: compress to two sentences or move to the supplement. R1: keep but relabel as a post-hoc reading with the registered verdict named. DA: it is in tension with the thesis until reconciled with Exps 4/8. *Type*: action. *Resolution*: the actions are compatible and ordered — first reconcile from the archive (the S19 cells carry per-episode outcomes: against the greedy evader, the planner converted 13 of the 85 law-held episodes into captures, 190 → 203 of 500; the record carries no whole-episode deadlock or immobility metric, as the DA says), then state the registered verdict and the pursuer used, then compress to two sentences in §6 or move to the supplement if the reconciliation needs more than that.

**Disagreement 3 — severity of the learning-mechanism problem.** DA: block (C1). EIC, R1, R2, R3: warn, each on a different facet (SC-2). *Type*: severity. *Resolution*: the contract resolves it (F1 fires on any block). The editor's assessment of C1's substance is in Section 4; it supports the DA on the facts and prescribes repair rather than rejection.

**Disagreement 4 — overall recommendation.** Minor (EIC, R3) versus Major (R1, R2, DA). *Resolution*: contract arithmetic (Section 2). The EIC's own report concedes that the Remark 2 inequality is contradicted by the paper's data, which is the coherence defect the other three score more heavily.

**Disagreement 5 — whether new runs are needed.** R3: a clean recurrent arm is welcome but text suffices. DA: either rescoping or the arm; plus a rotation-accepting perturbation baseline (M2) because the one comparator cannot rotate by construction. EIC: the perturbation baseline answers the "add a timeout" objection, but asks the DA to test whether that answer is complete. R1, R2: no new experiments needed for their concerns. *Resolution*: no new run is a condition of the revision; two evaluation-only rounds are recommended as the cheapest way to turn C1 and M2 from holes into results (roadmap N1, N2); the recurrent arm is optional and not advised before the deadline.

---

## 4. Devil's Advocate CRITICAL finding and the editor's assessment

**DA C1.** The stalled nominal in this architecture is overwhelmingly the scripted intercept law's go-to-last-seen branch, which carries memory (a remembered point) and acts while the evader is out of view; §3 describes the law as acting "when it is visible" and the pursuer as "memoryless", and C1, C2, §4 and Remark 2 are built on a learned, memoryless nominal. Evidence cited: supplement Table B1 Law = 1.00/0.97/1.00 (Base) and 0.57 (Aware); `s26_b1_round.json` `law_branch_share`; per-cell `visible` = false at every law-branch stalled state; Exp. 2's "evader visible in 2.4 % of stalled steps"; `pe2/structured.py`. Consequence claimed: "learning cannot change a command it does not emit" explains "learning does not remove it" more parsimoniously than the discount mechanism and better fits the Hor result.

**Corroboration by the four reviewers.** None of the four identified the law-branch memory themselves. All four independently found the learning-mechanism claim over-stated (SC-2): R3 by the Hor contradiction and the aliasing alternative; R1 by the unstated registered direction of Exp. 2's P1 ("optimisation, not credit assignment"); R2 by the mis-derived bound and the undefined observation; EIC by the conclusion sentence. The DA's finding supplies the mechanism behind what the four observed.

**Editor's checks (repeatable).**
1. `pe2/structured.py`, class `InterceptLaw`, lines 52–83 of the release: `act()` chases when `obs[6] > 0.5` and updates `last_seen`; otherwise, while `last_seen` is set and the pursuer is farther than `LAST_SEEN_TOL` from it, it commands full speed toward that point; it returns `None` (hands off to the head) only when the point is reached. The law therefore carries one remembered point and acts out of view. The §3 sentence is inaccurate on both counts.
2. Exp. 2 (`s15_cells/*.json`): the stall entries carry no branch field, so the 772 / 383 entries, the 629-step figure and the 0.0092 advantage are not split by branch; the registered finding that reducing the head's hold from 10 to 1 changes immobility by less than 0.05 is consistent with the law branch producing most stalls.
3. Exp. 4 gate records (`s17_arm_*.json`): per-cell summaries carry deadlock, stall and immobile rates but no law/search split, and the stored traces carry no branch flag; the headline deadlock numbers cannot be attributed by branch from the archive. Only Exp. 7 (`s26_b1_cells`) carries per-state branch and visibility.
4. Exp. 6 (`s19_cells/*.json`): no whole-episode deadlock or immobility metric; outcomes show 13 of 85 law-held greedy-evader episodes became captures under the planner.

**Assessment.** VALID on the facts, and critical for the paper's framing: the sentence in §3 is wrong, and the contribution C2 attributes to a learned memoryless nominal a mechanism that, for most archived stalled states, operates on a scripted branch in which no learned parameter is in the loop. It does NOT invalidate Proposition 1's geometry (nominal-agnostic), Proposition 3, or Experiments 4, 7, 8, 9, and the DA says so. Two points qualify the DA's reading without rescuing the current text: (i) the law's remembered point is itself frozen during a stall (it updates only when the evader is visible), so the frozen-input fixed point generalises to "a fixed nominal with frozen internal state", scripted or learned; (ii) Aware's 43 % search-branch stalled states show the learned head reaches the same fixed point, so a learning-side account is still owed for that share, with the aliasing argument (R3) and the discount estimate (correctly stated) as candidates rather than as a measured mechanism. **Required author response**: correct §3; split every headline deadlock number by branch where the archive allows and, where it does not (Exp. 4), either run the branch-attribution round N1 or state the gap; rescope C2, the §4 title, Remark 2's title, the abstract and the conclusion to what is measured; reconcile Exp. 2's registered direction; decide the title's "Learned Policies" wording on the outcome of N1.

---

## 5. Decision rationale

The panel is unanimous that the paper's empirical core is sound and unusually well documented: five reviewers checked numbers against the archive (the methodology reviewer 23 of them, recomputing the Exp. 8 and Exp. 9 p-values from per-episode indicators), found no cherry-picked seed or cell, and found every failed bar printed. The recovery's deadlock, capture, violation and composition claims are exactly what the records show, and the two penalty dissociations are, in three reviewers' words, the paper's most transferable finding. The venue fit is right and the positioning against Grover, Reis and Tan is accurate.

The decision is Major Revision because the paper's learning claim is not supported in the form it is made. Four independent facets converge: the formal centrepiece of C2 states an upper bound that the paper's own measurement exceeds eighteenfold (the inequality points the wrong way); its headline delay is a different quantity from the one named; the registered direction of the experiment that measured the mechanism contradicts the mechanism offered and is not stated; and the architecture's stalled command is, for most archived stalled states, emitted by a scripted branch with memory that §3 describes incorrectly. Around this sit a set of consensus defects that are individually small: abstract phrasings stronger than the body, a proposition that restates a known condition under a novelty heading and drops the weight matrix, a "typically" proposition without hypotheses, two registered verdicts omitted in a paper that promises all of them, figures that cannot be read at printed size, and undefined protocol vocabulary.

None of this requires new training. Almost all of it is text and archived data; two short evaluation-only rounds would convert the two largest holes into results. The expected outcome of a careful revision, on every reviewer's own statement, is acceptance.

---

## 6. Revision roadmap (prioritised; input for `academic-paper` revision mode)

Legend: **P0** required for the revision; **P1** strongly recommended, evaluation-only runs, authors' decision; **P2** optional. Source = reviewers raising it. Evidence = text / archive / new run.

### P0 — required (text and archive)

| # | Item | Source | Location | Action | Evidence |
|---|---|---|---|---|---|
| R1 | Describe the pursuer correctly | DA C1; R2 W5; R3 W3 | §3 Policies | Law chases while visible, drives to a remembered last-seen point while not, hands off to the PPO head only when it has nothing to act on; head = 10-step sample-and-hold; "memoryless" restricted to the head, with the law's one-point memory stated and noted as frozen during a stall | text + code |
| R2 | Rescope the learning claim | SC-2 (CONSENSUS-4); DA C1 | Title (decide after N1), abstract, C2, §4 title, Remark 2 title, §7, §8 | "Why a memoryless learner does not remove it" → what is measured: (a) law-branch stalls: no learned parameter in the loop; (b) head-branch stalls: frozen input, with aliasing of a memoryless policy stated as the candidate explanation (cite the POMDP literature after verification) and memory named as the first follow-up; (c) the discount channel as an estimate, not a bound. State Exp. 2's registered P1 verdict and direction and reconcile | text + archive |
| R3 | Repair Remark 2 and Figure 2 | SC-1, SC-1b | p. 4 Remark 2; Fig. 2(b),(c); abstract; p. 10 | Replace "at most γ^T A₀" by the correct estimate A₀·E[γ^T], computed from the archived escape delays at γ = 0.99 and 0.999 (expect ≈ the measured 0.0092 at 0.99); carry the measured value as the construct; rename 629 as the mean remaining horizon and report the 346-step delay to capture on captured entries; make Fig. 2(b)'s caption, axis and pool agree with the text; define or remove the "0.05 bar" in Fig. 2(c); replace "Remark 2 is a bound" (p. 10) | archive (s15 cells) + text |
| R4 | Restate the formal section | SC-5, SC-5b | §4 | Lemma (after Grover et al.): executed speed is zero iff W(u_π − u*) lies in the normal cone of the linearised polytope A_lin(s) at u* on the zero-speed face, with W = diag(150, 1) and the sign of the normals fixed; say the implemented filter is an enumeration accepted by the exact test and agrees with the projection on 98.1 % of stalled states. Proposition 1 becomes the fixed point with explicit hypotheses (heading trap ∧ zero-turn request ∧ frozen observation ⇒ s_{t+1} = s_t until the evader enters view). C1 reworded to the pair-trap / position-trap measurement (98 % / 0 %). Report ω* at the 262 states (R2 Q1) if cheap | text + archive (s26_b1 cells carry u_shield) |
| R5 | Proposition 3(b) conditional | SC-11 | p. 4 | State H1–H4; T_turn as one full turn at the certified rotation rate (45 at ω_max, 89 at ½ ω_max); add the displacement term 0.10 m / (0.05 v_max) = 40 steps; mention the 0.1 rad/s rotation threshold; report 97.8 % as the rate at which the stronger 65-step statement held; name (c) livelock and say no liveness guarantee is claimed | text (+ archive for the tail split, R2 Q3, if the escape records allow) |
| R6 | Registered verdicts and labels | SC-8; DA M4, m6 | §5.1 (Exp. 2), §5.2 (Exp. 3), §6 (Exps 5, 6), Table 1 | Name P1 FAILS with its direction; name H-CROSS FAILS and add the SPARSE/DRARE null columns to the supplement; name Exp. 3's registered primary metric and verdict (rotation-only recovery won the deadlock indicator by spinning, 414 928 firings on one seed); name Exp. 6's registered verdict; retitle §6 "Three dissociations observed in pre-registered measurements" and fix "failed in the registered direction"; consider a registered-verdict column in Table 1 | archive + text |
| R7 | Abstract and §1 phrasings | SC-3 (CONSENSUS-4) | Abstract, C3, p. 7 | "(Base 0.19–0.30 and Obs to 0.000; ObsHor to ≤ 0.006; bar 0.05)"; "capture is insensitive to N over 5–80 and the length-matched run indicators stay ≤ 0.02"; "the three observation/horizon families raise capture without removing the failure; the penalty arms lower it"; conclusion: "cannot see the way out at the training discount" | text |
| R8 | Remove or restrict three asides | SC-6 | p. 2, p. 5, p. 6 | "predicts the oscillations" → "is one mechanism behind the timeouts" or delete; "separate game" → delete; "Any filter … Kim et al." → convex-projection filters only; the predictive filter is covered by the composition result, not by the Lemma | text |
| R9 | Figures at printed size | SC-4 (CONSENSUS-4) | Figs 1, 3(b), 4-right, 5, 6, 7 | Regenerate with figsize equal to the printed width and 8–9 pt type; Fig. 7(a) y-label and (c) legend; Fig. 4-right three group labels or move to the supplement; Fig. 3(b) as a one-line summary or the supplement table; Fig. 1 annotation into the caption; Fig. 5 either full width or (a),(b),(f) in the main text; add that the K = 40 indicator is insensitive for N ≥ 40 by construction to the Fig. 5 caption; no figure carries an interval (R1) | figure scripts (label/size only; data unchanged) |
| R10 | Reconcile and compress Experiment 6 | SC-12; DA M4 | §6(c), Fig. 7(c) | State the pursuer used; compute from the S19 cells what became of the law-held episodes under the planner (greedy: 13 of 85 captured); name the registered verdict; two sentences in §6 or move to the supplement | archive + text |
| R11 | Disclose the artefacts | SC-13 | §3, §5.1 | Free and Aware are single checkpoints selected as the best seed on an earlier gate (state the rule); Aware was trained with filter-state features and a 10⁻³ stall penalty; report the RC arms (0.564, 0.550, 0.444) with RA and RB; state once that Free's 0.396 (Exp. 1 stream, greedy evader at speed ratio 1.2) and 0.426 (Exp. 8 stream) are different cells | archive + text |
| R12 | Statistical reporting | SC-9 | §5 Protocol | One sentence on the multiplicity policy (pre-specified primary per experiment, no correction within the registered family) naming which secondary p-values are fragile (Exp. 6's deltas); 95 % paired-difference intervals for the headline deltas (Exp. 1 has one archived: [−0.375, −0.289]); one sentence on the minimum detectable difference at N = 500 paired | archive + text |
| R13 | Observation and reward paragraph | SC-10 | §3 | Components and dimension of o(s), which components Obs and Aware receive from the filter, the 80-step saturating counter, the reward terms; whether the pressed-against obstacle is observed (R3 Q1) | code (`pe2/env.py`, `obs.py`, `rewards.py`) + text |
| R14 | Vocabulary and naming | SC-7 | throughout; Table 1 | Define cell, arm, gate, hold-10, exact/relaxed/backup tiers, "decidable triggers", "dense-reward column"; replace "pins" by "stalls on the intercept-law branch"; resolve Table 1's "six" and "anchored fine-tune"; define Base/Obs/Hor/ObsHor before Table 1; gloss "Free" at first use; avoid "A" for the attacker next to A₀ and 𝒜(s) | text |
| R15 | Minor numerics and consistency | R1 W6, minors; EIC minors; R3 minors | various | Median escape "4 to 10"; visible-fraction pool stated; "p ≤ 9 × 10⁻⁴"; "five of 262"; explain `figure_data_s26.json` n_decided 10 466 vs the record's 10 475; say the 1 106-step and 1 085-step episodes differ; Fig. 2 denominators sentence; supplement Table B1 rotation qualifier aligned with p. 3; drop the uncited "and others" bib entries; T_turn wording; Exp. 2 / Exp. 7 N not paired (footnote) | text + archive |
| R16 | Scope sentences | SC-16, SC-17; DA M5 | §5.3, §7, title | Note the predictive filter's margin 0 / no contraction when calling the loop "weaker"; state static geometry as a boundary condition of the dissociation; move the one-game / one-dynamics / two-filters scope sentence earlier; decide the title's "Learned Policies" on N1 | text |
| R17 | Placeholders and artefact | EIC, R1, R3 | p. 1, p. 10, README | Email; repository URL (Nov 10); licence; README note that the frozen-commit hashes are not resolvable in the release; PPO hyperparameters and network in the reproducibility paragraph or supplement | text |

### P1 — recommended evaluation-only rounds (pre-register first; no training; authors' decision)

| # | Round | Answers | Design | Cost | Why |
|---|---|---|---|---|---|
| N1 | Branch attribution of deadlock episodes | DA C1; SC-2 | Re-evaluate the Exp. 4 headline cells (7 families × 3 seeds vs the learned attacker; optionally vs the greedy evader) with the per-step branch and visibility flags recorded; report the law/search split of stalled steps and of deadlock episodes beside each headline number; also classify Free's stalled states as Exp. 7 did | 21–42 cells × 500 episodes; Exp. 8's 86 cells ran in about three hours on 24 cores | Turns the CRITICAL from a hole into a result either way: if head-branch stalls are a minority, the paper's learning story is rescoped to that share with numbers; if they are substantial, the learning claim is kept with evidence |
| N2 | Rotation-accepting perturbation baseline | DA M2; EIC's question to the DA | Same trigger as PERT-20, candidates drawn uniformly from the box and accepted if they change the pose (rotations included); five policies × two opponents × 500 paired | 10 cells; under an hour | The current comparator cannot rotate by construction (foreseen in Amendment A4); without N2 the claim "the recovery's contribution is the rotation candidate" rests on geometry alone, and "a structured candidate list beats random resolution" is untested |
| N3 | Archive-only diagnostics | R2 Q1, Q3 | ω* at the 262 stalled states; split of the 2.2 % escape tail into slow creep vs interrupted turns, if the escape records carry the fields | hours of analysis, no run | Supports R4 and R5 |

### P2 — optional

| # | Item | Source | Note |
|---|---|---|---|
| O1 | A clean recurrent arm (one family, three seeds, 5 M steps) | R3 W1(iii); DA alt. 3 | Would settle whether the claim is about memoryless policies or this controller; R3 states that R2's text rescoping satisfies them without it. Not advised before Nov 13 |
| O2 | RECF-Hor evaluation arm | DA m7 | Hor is the only family not composed with the recovery; evaluation-only, 3 cells |
| O3 | Predictive-filter horizon sweep | DA alt. 5; R2 | Beyond this submission |
| O4 | Comparison with a filter modified to remove undesired equilibria (Tan and Dimarogonas 2024) | DA alt. 4 | Beyond this submission; mention in §7 as the untested alternative |

### Suggested order and timeline (deadline Nov 13 23:59 AoE; go/no-go Nov 10)
1. Nov 2–3: decide N1/N2; if yes, pre-register and launch (both evaluation-only). In parallel: R3, R4, R5, R13 (the formal and definitional repairs), R6, R11 (archive look-ups).
2. Nov 4–5: R1, R2, R7, R8, R10, R16 (framing, using N1's result if run); R9 figures; R12; R14, R15.
3. Nov 6: re-compile to 10 pages; verification pass on every new number and every new citation (R16's list through the citation protocol).
4. Nov 7–8: re-review (reviewer skill `re-review` mode) against this roadmap.
5. Nov 10: go/no-go; URL and password decision; Nov 12 submit.

### Questions the authors must answer in the response
1. Which share of the headline deadlock episodes arises on the law branch versus the learned head (N1), and what does the title say after that?
2. What is the correct discounted escape advantage computed from the archived delay distribution at γ = 0.99 and 0.999, and which single number do the abstract, Remark 2 and Fig. 2(c) carry?
3. How is Exp. 2's registered direction ("optimisation, not credit assignment") reconciled with the mechanism the paper offers?
4. What is in the observation vector, and is the obstacle the pursuer is pressed against represented in it?
5. Is exact-tier feasibility ever lost during evaluation (R2 Q4)? If never, say so; if sometimes, describe the "guarantee" as the empirical zero-contact record.
6. Free's and Aware's selection rule and Aware's training reward.

---

## 7. Panel artefacts
`00_reviewer_configuration.md` · `phase1_{eic,methodology,domain,perspective,devils_advocate}.md` (paper-blind pre-commitments, all with `[CONTRACT-ACKNOWLEDGED]` and five scoring-plan subsections) · `review_{eic,methodology,domain,perspective,devils_advocate}.md` · `main_text_for_review.txt`, `supplement_text_for_review.txt` (page-layout extracts of the PDFs). No reviewer edited the manuscript, the archive or the release repository.
