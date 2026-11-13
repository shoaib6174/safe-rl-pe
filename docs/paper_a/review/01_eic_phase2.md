# EIC Review — Phase 2 (paper-visible)

Contract `reviewer/reviewer_full/v1`, mode `reviewer_full`, baseline v3.6.2. Reviewer: Editor-in-Chief persona (area chair, multi-agent learning track; MARL and game-theoretic evaluation). Manuscript: "Learned Tactic Selection for Both Players of a Safety-Filtered Pursuit–Evasion Game", `docs/paper_a/draft/main.tex` (8 pp. IEEE conference, 4 figures, 4 tables, 23 references; body 4,844 words incl. abstract). Target: IC-FMAS 2027, Track 2. Scores below apply the Phase 1 triggers as committed; no dissent is filed.

## Dimension Scores

### D1: methodology_rigor
score: warn
Design-level signals (i)–(iii) of my plan are met. Section III-G fixes 500 paired episodes on a declared seed stream (2 400 000+), three training seeds, and defines "held-out" as absent from the row's training mixture; Section III-B defines the filter and the contact metric before any result; each side is tested against opposition built to exploit it (pursuer vs the learned evader, Table 3; evader vs held-out selector seeds and the retrained pursuer Sel-R, Section IV-C). Two reporting gaps match my warn trigger ("a checkpoint-selection rule stated post hoc rather than pre-specified; no code or configuration availability"): (a) the representative seeds that populate the evader's training mixture (Sel-2, Section III-E), the meta-game pool (Sel-2, Sel-R-1, Evs-1; Section IV-C "Meta-game"), the margin sweep (Section IV-D) and Table 2 ("seed 2") are introduced with no stated rule. The pre-registrations do contain rules (S24.2 A2.4/A5.2: Evs-best = lowest mean capture, Sel-R-best = highest mean capture over its gate), and Table 2's seed 2 is the strongest seed at rho = 1.2 on two of three evaders (Table 1) — none of this is in the manuscript. (b) Everything the protocol rests on is "to be added at camera-ready" (footnote 1, Section III-G), so no record is inspectable at review. My block trigger ("best-of-N presented as representative with no disclosed selection protocol") does not fire: the transfer claim is stated over all seeds in the Section IV-B text, Tables 1 and 3 give every seed, and I verified that the rules were pre-stated — the defect is non-disclosure, not selection.

### D2: domain_accuracy
score: warn
No load-bearing error found. The CBF material is careful: Section III-B states the discrete-time condition with the right attribution [12], the exact-successor check, and Proposition 1 with an explicit scope; my glance at `pe2/shield.py` (ranges supplied) confirms the three facts the proof uses — u = 0 is always a candidate, hard rows are judged on the exact Euler successor, and the recovery re-filters every candidate (the domain reviewer owns the full check). Exploitability is defined and attributed [5]; EGTA [21] and PSRO [22] are cited for the matrix reading. Two items match my warn trigger ("equilibrium used loosely where an empirical meta-game solution is meant; an obviously adjacent line uncited"): (a) Section VII "converges in one evader step to an obstacle-orbit equilibrium" and Section V "so the equilibrium can be read off the episodes" drop the pool qualifier that Section VI states ("'minimax' is relative to that pool, not to the full game"); (b) the one prior statement in the field that bears on Q3 — the Control Engineering Practice 2026 paper on CBF-filtered RL in pursuit–evasion whose abstract claims convergence to "conservative safe equilibria" rather than the true saddle point, flagged in the authors' own scoop check as must-obtain — is absent from Section II, and the lineage of learned selection over hand-written behaviours (macro-actions and skill selection in MARL and robot games) is represented only by [11]. Section V's "a known asymmetry of pursuit–evasion among obstacles" carries no citation.

### D3: argumentative_coherence
score: warn
Title, abstract, C1–C4, experiments and conclusion trace to real experiments, and every number in the abstract matches the archived records (checked against `numbers_extract_2026-11-02.txt` and D151–D158). The warn trigger ("scope qualifiers present in the body but dropped from the abstract or conclusion; claims stated more broadly than tested") fires repeatedly: (a) abstract: "the pursuer's selector beats the best fixed intercept law on held-out evaders" — C1 and Section IV-A say two of three seeds, and Table 1 shows seed 3 below Lead on Greedy (0.528 vs 0.558) and marginal on Hug (p = 0.056); Section VII repeats "it beats fixed laws"; (b) C3 "the filter accounts for 0.046 of the value at the default setting" and Section VII "the filter's share of the value is small" omit the frozen-player qualifier Section VI states ("the unfiltered value in particular may understate what an unfiltered pursuer could learn"); (c) Section V "the meta-game confirms that the pursuer's library, not its learning, is binding" — a 4-pursuer pool and one retraining arm are consistent with that reading, not a confirmation; Section VII "an equilibrium that the pursuer cannot escape within its library" turns H-ALT-1's one-round failure (3 seeds, one budget, one mixture) into an impossibility; (d) Section V "speed ratio, obstacle geometry and the libraries set most of the value" is attribution by elimination — Section VI concedes the geometry was not varied; (e) magnitude slips: Section V "reduces every pursuer by 0.6 of capture" (Lead falls 0.41–0.44 and Rules 0.54–0.56 in Table 3; 0.6 is the selector's drop), Section IV-C "losing 0.05–0.08 against the tactical evader on two seeds" (both drops are 0.082). No block trigger fires: both players are evidenced, no abstract figure disagrees with the body, no contribution lacks an experiment, and the safety claim is a scoped proposition plus counts rather than a guarantee inferred from counts.

### D4: cross_disciplinary_relevance
score: pass
Section III-B explains the filter in terms a MARL reader can follow (what it does to the command, the admissibility test (1), the three tiers, the recovery), and Sections II and IV-C give a robotics reader the game-theoretic constructs (exploitability defined in text; saddle point and value shown on the matrix of Figure 3). The bridging claim "the filter sets the value" is tested rather than asserted (Section IV-D), and the MAS take-home is explicit (Section V: report the filter's margin as a parameter of the game with the unfiltered value beside it). Residual, below the warn trigger: "support" (Figure 4(b), Section V) and "duality gap" (Section IV-C) are used without a one-line definition.

### D5: writing_and_structure
score: warn
Venue envelope met: 8 pages (4–12), full-word "Figure"/"Table" captions ending in periods, six keywords, no subtitle, every figure and table cited; figure text in Figures 1, 3 and 4 prints at 7.7–9.3 pt (measured on the figure PDFs, whose native widths equal their printed widths). The warn trigger ("naming inconsistencies resolvable with effort; small-font figures") fires: (a) the figures name policies differently from the text — Figure 1 "Selector (ours) / Selector, no memory / Hand-written rules / Best fixed law (LEAD) / Structured search + law / Flat PPO, 10-step hold / 40-step hold" against Sel / Sel-NM / Rules / Lead / Struct / Flat-10 / Flat-40; Figure 3 "Lead law / Selector / Selector, retrained / Hugger / Scripted tactical / Learned selector" against Lead / Sel-2 / Sel-R-1 / Hug / Tactic / Evs-1 — with thirteen named policies the reader must build the crosswalk; (b) Figure 2 (episode.png, 1650 x 1500 px at column width) prints its title, tick labels and footer at about 6 pt, the footer "tier exact · clearance +0.023 m" is drawn over a second faded string, and two crosses and two shaded regions appear while the caption explains one cross and one cone. "Email: [to confirm]" remains in the author block. The block trigger does not fire: the mapping is recoverable and no venue hard constraint is breached.

## Failure Condition Checks

### F1
fired: false — no mandatory dimension scores block.

### F2
fired: true — D1, D2 and D3 all score warn (three of three mandatory dimensions at warn or worse).

### F3
fired: false — D4 (high priority) scores pass.

### F0
fired: false — not every mandatory dimension scores pass.

## Review Body

# Peer Review Report

## Manuscript Information
- **Title**: Learned Tactic Selection for Both Players of a Safety-Filtered Pursuit–Evasion Game
- **Manuscript ID**: Paper A draft, `docs/paper_a/draft/main.tex` (2026-11-02)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
EIC

### Reviewer Identity
Area chair, multi-agent learning track of an IEEE multi-agent-systems conference; multi-agent reinforcement learning and game-theoretic evaluation. Judges venue fit, originality against the nearest prior work, significance for a MAS readership, and whether abstract and conclusion claim exactly what the tables show. Interval and test details are left to the methodology reviewer; the CBF proof to the domain reviewer.

### Review Focus
Does the paper belong in IC-FMAS Track 2, what is new relative to Wu et al. [5], HMARL-CBF [16], Acharya et al. [17] and the work it does not cite, and do the title, abstract, contribution list and conclusion say what Tables 1–4 and Figures 1–4 support — no more, no less.

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] Major Revision (by the contract's F2; the required changes are textual and use data already in hand, so the re-review should be short)
- [ ] Reject

### Confidence Score
4 — MARL and empirical game-theoretic evaluation are my area; the CBF implementation details are not, and I defer there.

### Summary Assessment
The paper builds both players of a one-on-one obstacle pursuit–evasion game as maskable-PPO selectors over five hand-written closed-loop tactics, with every pursuer command passing through a discrete-time CBF filter with a forward-offering recovery. Three pre-registered rounds are reported: the pursuer's selector beats the best fixed law on two of three seeds and beats flat PPO at two cadences and a structured learner in 9/9 cells at zero contacts (Table 1, Figure 1); the evader's selector finds an obstacle orbit that drops every pursuer to 0.09–0.22 and is the pure minimax of a 4 x 4 pool at value 0.138 (Table 3, Figure 3); a margin sweep with an unfiltered control finds that removing the filter moves that value only to 0.184 for frozen players and that a 0.20 m margin switches the minimax evader to a scripted hugger (Table 4, Figure 4). Fit for Track 2 is exact: MARL on both sides, evaluated as a game. The architecture (options over scripted skills) is standard; the originality is the two-sided learned-selector game under a model-based filter and the direct test of whether the filter sets the pool's value, which [5], [16] and [17] do not do — a real but incremental delta that the paper states fairly. For a MAS readership the most reusable parts are the protocol (paired seeds, bit-identical replays, refutation sentences applied as written) and the corrective negative result on "the filter sets the value". The main weaknesses are that the abstract and conclusion speak more broadly than the tables, and that the representative-seed choices behind the mixtures, the meta-game and Table 2 are undisclosed. Both are fixable in text; because the headline sentences change, re-review is warranted.

## Strengths

### S1: Refutation-first protocol with the failures kept in the paper
Section I commits to pre-registration with refutation sentences, and the paper honours it: the "filter sets the value" reading is registered as the hypothesis of the next round (Section IV-C, last sentence) and then refuted in the form stated (Section IV-D, "Both hypotheses failed"); the per-ratio training and the sixth tactic are reported as failures (Section IV-B). This is rare at conference length and is the paper's strongest credential.

### S2: A genuinely two-sided learned game with held-out opposition on both sides
The pursuer's selector is tested on evaders absent from its training (Table 1) and then on a learned evader (Table 3); the evader's selector is tested on selector seeds absent from its mixture (Sel-1, Sel-3 for Evs-1, Evs-3) and on a pursuer retrained against it (Section IV-C). The 4 x 4 matrix with per-policy exploitability (pursuers 0–0.052, evaders 0–0.644) gives a MAS reader a precise, checkable statement.

### S3: The margin sweep with an unfiltered control
Table 4 and Figure 4 answer the question the field keeps asserting in words. The delta = 0 cells reproduce the stored records bit for bit (Section IV-D), the unfiltered control reports its contact steps rather than hiding them, and the observation that a 0.20 m margin changes which evader is minimax rather than how strong the learned one is (Figure 4(b)) is new.

### S4: A scoped safety claim
Proposition 1 replaces "certified" with a statement whose assumptions are listed (true pose, known static map, exact one-step map; Section III-B), and 101 000 filtered evaluation episodes carry zero contact steps. The abstract's "we prove" is proportionate to what is proved.

### S5: Ablations that locate the gain
Flat-40 matches the selector's cadence, Sel-NM removes the memory block, and Rules is a hand-written selector over the same library; Section V uses all three to say what the gain is not (cadence, memory alone, library alone). The Ambush finding (14–19 % of decisions against the tactical evader, never in the rule table's top three) is the kind of legible evidence a hierarchical design should produce.

## Weaknesses

### W1: The abstract and conclusion drop the qualifiers the body carries
**Problem**: Abstract: "beats the best fixed intercept law on held-out evaders"; Section VII: "it beats fixed laws". C1 and Section IV-A say two of three seeds; Table 1 has seed 3 at 0.528 against Lead's 0.558 on Greedy. C3 and Section VII state the filter's share of "the value" without the frozen-player caveat of Section VI. Section VII: "an obstacle-orbit equilibrium that the pursuer cannot escape within its library"; Section V: "the meta-game confirms that the pursuer's library, not its learning, is binding".
**Why it matters**: These are the sentences that will be quoted. Each is a shade stronger than the evidence: 2/3 seeds, a frozen-player pool value, one failed retraining round, a 4-pursuer pool.
**Suggestion**: Abstract: "beats the best fixed pursuit law on held-out evaders on two of three seeds". C3/Section VII: "of the frozen-player pool value". Section VII: "that one round of pursuer retraining did not escape". Section V: "is consistent with the reading that the library, not the learning, is binding".
**Severity**: Critical

### W2: Representative-seed choices are undisclosed
**Problem**: Sel-2 is the evader's training seat and the meta-game and sweep row; Evs-1 and Sel-R-1 are the meta-game and sweep columns/rows; Table 2 shows "seed 2" only. No rule is stated anywhere in the manuscript. The pre-registrations contain the rules (Evs-best = lowest mean capture, Sel-R-best = highest mean capture), and seed 2 is the strongest pursuer seed at rho = 1.2 on two of three evaders.
**Why it matters**: A reader cannot tell whether 0.138 is a best-of-seed value on both sides (it is, symmetrically) or a typical one, and Table 2 reads as representative when it shows the best seed. The claims survive — Tables 1 and 3 give every seed — but the paper must say so.
**Suggestion**: One sentence in Section III-E or III-G stating the three rules and that they were pre-registered; Table 2 with all three seeds (or a range column); one sentence in Section IV-C giving the saddle value's spread when the learned entries are replaced by the other seeds (the records for Sel-1..3 x Evs-1..3 and Sel-R-1..3 x Evs-1..3 already exist in Table 3 and the stage-2 round, so no new run is needed).
**Severity**: Critical

### W3: The nearest prior statement on Q3 is uncited, and the hierarchy lineage is thin
**Problem**: Section II positions against [5], [16], [17], [18] well, but the Control Engineering Practice 2026 paper on CBF-filtered RL in pursuit–evasion — whose abstract claims convergence to "conservative safe equilibria" rather than the true saddle point — is absent although it is the one published statement bearing on Q3; the paper's own scoop check names it as must-obtain. Learned selection over hand-written behaviours is grounded only in [11]; Section V's "known asymmetry of pursuit–evasion among obstacles" has no reference.
**Why it matters**: For a Track 2 reader the originality case is "two-sided learned game plus measured filter cost"; it should be made against the closest statement on each half.
**Suggestion**: Obtain the CEP paper, quote its claim, and state the delta (their claim is about training-time convergence; the sweep here finds a small frozen-player share). Add two or three representative macro-action / skill-selection MARL references with one sentence of delta. Cite the obstacle-evasion asymmetry result the paper has in mind.
**Severity**: Major

### W4: Artefacts and AI-use statement are not yet at the venue's bar
**Problem**: Footnote 1 defers the repository to camera-ready, so the pre-registrations, per-cell records and rendered episodes that the protocol section leans on cannot be inspected. The Acknowledgment names "Claude Opus and Fable 5.x models" for "drafting and editing of the manuscript"; the venue's AI page asks for tool name, version, function and specific use, and states that GenAI may not write the whole paper.
**Why it matters**: Reviewers are asked to credit a protocol they cannot see; and an under-specified AI statement is a desk-level risk at this venue.
**Suggestion**: Provide the repository (or an anonymised snapshot) at submission. Give exact model versions and dates, and add one sentence on the authors' own writing contribution (which sections were authored or rewritten by the authors).
**Severity**: Major

### W5: Figure/text naming mismatch and Figure 2 legibility
**Problem**: Figures 1 and 3 use descriptive labels; the text uses short names (Sel, Sel-NM, Sel-R-1, Evs-1, Rules, Lead, Tactic, Hug, Struct, Flat-10/40, Full). Figure 2 prints at about 6 pt with an overlapped footer and unexplained marks (two crosses, a grey shadow, dotted rings and an inner dotted rectangle).
**Why it matters**: Thirteen policies and two naming schemes make the reader do the crosswalk; Figure 2 is the only episode-level evidence and must be readable at column width.
**Suggestion**: Use the short names in the figures (or add a one-line crosswalk in Section III-F). Re-render Figure 2 at 8 pt or larger, fix the footer overlap, and explain every mark in the caption or a small legend.
**Severity**: Major (naming) / Minor (Figure 2)

## Detailed Comments

### Title & Abstract
The title is accurate: both players are learned selectors, the game is filtered on one side (the abstract says so immediately). The abstract is complete and every number in it matches the records. Three edits: add "on two of three seeds" to the fixed-law sentence; "best fixed intercept law" should be "best fixed pursuit law" (the baseline is lead pursuit plus coverage search, and Intercept is a different tactic in the library); add "for frozen players" to the sentence on removing the filter.

### Introduction
Q1–Q3 are stated crisply and each is answered. The sentence "in our own rounds, a flat PPO pursuer under the filter ends most episodes pinned in a corner" is supported by Section IV-A. Contributions C1–C4 are all evidenced; C1 already carries the 2/3 qualifier that the abstract and conclusion should also carry.

### Literature Review / Theoretical Framework
Positioning against [5] (learned filter, legged game, exploitability definition adopted), [16] (hierarchy over CBF skills, cooperative) and [17] (LLM-chosen roles over a DCBF filter) is fair and the deltas are correctly stated. Missing: the CEP 2026 CBF-PE-RL paper (W3); representative macro-action / skill-selection MARL work; a citation for the "known asymmetry". "Equilibria of CBF-QPs" is correctly scoped out.

### Methodology / Research Design
From the editorial seat the design fits the claims: paired evaluation, held-out opponents defined, both sides exposed to adaptive opposition, filter and contact metric fixed before results. The gaps are reporting: selection rules for representative seeds (W2) and artefact availability (W4). One question for the domain reviewer is recorded below (barrier form).

### Results / Findings
Tables 1–4 and Figures 1, 3, 4 are complete and legible at print size. Table 2 shows one seed (W2). Section IV-B's "Every seed beats the fixed law at 0.8 on all three evaders and at 1.0 on at least two of three" is the correct all-seed statement and should be the one the conclusion echoes. Section IV-C's "0.05–0.08 against the tactical evader on two seeds" should read "0.08" (both drops are 0.082). Section IV-D is the best-written results section and already carries the frozen-player caveat in the Limitations; move one clause of it up into the headline sentence.

### Discussion
"What the hierarchy buys" is well argued from the ablations. "The evader side wins, and quickly" over-reaches in two places ("reduces every pursuer by 0.6"; "confirms that the pursuer's library, not its learning, is binding"). "The filter's share is small" is the paper's most useful paragraph for MAS readers; its last sentence (report the margin as a game parameter with the unfiltered value beside it) is the take-home — but "speed ratio, obstacle geometry and the libraries set most of the value" should be marked as inference, since the geometry was not varied (Section VI says so).

### Conclusion
Over-infers in three clauses (W1). The future-work list is right (players retrained per margin; a geometry sweep; a tactic that goes round an unseen obstacle).

### References
23 entries, all with a year and a link; IEEE style. [6] and [15] link to the PMLR volume index rather than the paper. The venue checklist asks for superscript "[n]" in-text citations; IEEEtran produces inline brackets — confirm against the venue template. Authors span more than three countries, as required.

## Questions for Authors
1. Were Sel-2, Evs-1 and Sel-R-1 chosen by the pre-registered best-of-seed rules (lowest/highest mean gate capture), and what is the range of the 4 x 4 saddle value when the learned entries are replaced by the other two seeds on each side (all needed cells already exist)?
2. How does the Control Engineering Practice 2026 claim that CBF-filtered RL converges to "conservative safe equilibria" relate to your Q3 finding that the filter's share of the frozen-player value is 0.046? Is the apparent disagreement one of training-time vs frozen-player evaluation?
3. Section III-B writes the obstacle hard row as h = ||p - c|| - r_eff; the implementation's own notes describe obstacle barriers in m^2. Which form does condition (1) use, and does Proposition 1 or the admissible set depend on the choice? (For the domain reviewer; the answer belongs in Section III-B.)
4. Which parts of the manuscript text were written or rewritten by the authors, for the venue's AI-use statement?

## Minor Issues

### Language / Grammar
- Section V: "reduces every pursuer by 0.6 of capture" → give per-pursuer drops (selector 0.59–0.63, rule table 0.54–0.56, fixed law 0.41–0.44).
- Section IV-C: "losing 0.05–0.08" → "losing 0.08".
- Section IV-C heading "The pursuer cannot answer within its library" → "did not answer in one retraining round".
- Section IV-D: "stops every pursuer from approaching an obstacle-hugging evader at all (capture 0.042–0.058 …)" — the range is for the three library pursuers; Full is 0.034.

### Citation Format
- [6], [15]: paper-level URL or DOI instead of the volume index.
- Confirm in-text citation style (superscript vs inline brackets) against the venue template.

### Figures and Tables
- Figure 2: text about 6 pt at column width; footer overlaps a faded string; two crosses and the grey occlusion shadow are not explained; dotted rings and inner rectangle unexplained.
- Figures 1 and 3: labels differ from the text's policy names.
- Table 2: one seed shown; add the other two or a range.

### Layout
- Author block: "Email: [to confirm]".
- Footnote 1 and the Acknowledgment both promise the repository; one statement with a live link suffices.

## Actionable Items (ranked by severity)

1. **CRITICAL — Abstract and conclusion claim more than Tables 1–4 show.** Where: abstract sentence 3 ("beats the best fixed intercept law on held-out evaders"); Section VII sentences 1–3; C3; Section V paragraph 2 ("confirms") and paragraph 3 ("set most of the value"). Fix: abstract → "beats the best fixed pursuit law on held-out evaders on two of three seeds"; C3 and Section VII → "the filter accounts for 0.046 of the frozen-player pool value"; Section VII → "an obstacle-orbit saddle of the 4 x 4 pool that one round of pursuer retraining did not escape"; Section V → "is consistent with the library, not the learning, being binding" and "the remainder is attributable, by elimination, to speed ratio, geometry and the libraries, which this study did not vary".
2. **CRITICAL — Disclose the representative-seed rules and show the spread.** Where: Section III-E (Sel-2, Evs-1), Section IV-C "Meta-game" (Sel-2, Sel-R-1, Evs-1), Section IV-D, Table 2 caption ("seed 2"). Fix: state the pre-registered rules in one sentence (Evs-best = lowest mean gate capture; Sel-R-best = highest; Sel-2 named before S24.2 as the strongest seed at rho = 1.2); give Table 2 for all three seeds; add the saddle value's range over seed substitutions, computed from the existing Table 3 and stage-2 records.
3. **MAJOR — Frozen-player qualifier on every statement of the sweep's value.** Where: abstract ("Removing the filter raises the value only from 0.138 to 0.184"), C3, Section IV-D first sentence after Table 4, Section VII. Fix: "for frozen players" or "of the frozen-player pool" in each; keep the Section VI sentence.
4. **MAJOR — Related work: the nearest statement on Q3 and the hierarchy lineage.** Where: Section II "Hierarchy above a filter" and "Empirical game analysis"; Section V "known asymmetry". Fix: obtain and cite the Control Engineering Practice 2026 CBF-PE-RL paper with its exact claim and your delta; add two or three macro-action / skill-selection MARL references with a one-line delta; cite the obstacle-evasion asymmetry result.
5. **MAJOR — Make the artefacts available at review.** Where: footnote 1, Acknowledgment "Data and code". Fix: a repository or anonymised snapshot with the pre-registrations, per-cell JSON records and rendered episodes, linked in the submitted version.
6. **MAJOR — AI-use statement to the venue's specification.** Where: Acknowledgment. Fix: exact model names, versions and dates; the specific functions; one sentence on the authors' own authorship of the text; keep it in the Acknowledgment or move to Methodology as the venue allows.
7. **MAJOR — One naming scheme across text, tables and figures.** Where: Figure 1 and Figure 3 labels vs Section III-F names. Fix: use the short names in the figures, or add a crosswalk line to Section III-F and keep the descriptive labels.
8. **MINOR — Figure 2.** Fix: re-render at 8 pt or larger at column width; remove the overlapped footer string; explain both crosses, the grey occlusion shadow, the dotted hug circles and the inner dotted rectangle in the caption or a legend.
9. **MINOR — Magnitudes.** Section V "by 0.6" → per-pursuer values; Section IV-C "0.05–0.08" → "0.08"; Section IV-D "every pursuer … 0.042–0.058" → state that the range covers the three library pursuers (Full 0.034).
10. **MINOR — Terminology.** Abstract "best fixed intercept law" → "best fixed pursuit law (lead pursuit with coverage search)"; define "support" and "duality gap" once (Section IV-C); Section IV-C heading → "did not answer in one retraining round".
11. **MINOR — References and front matter.** Paper-level links for [6] and [15]; confirm the venue's in-text citation style; fill the corresponding author's email.

## Dimension Scores (rubric, 0–100, advisory — ordinal only; the contract's failure conditions govern the decision)

| Dimension | Score | Descriptor | Notes |
|-----------|-------|------------|-------|
| Originality (20%) | 70 | Adequate–Strong | Standard options-over-skills architecture; new two-sided learned-selector game under a model-based filter and a direct filter-margin test of the pool value |
| Methodological Rigor (25%) | 75 | Strong | Paired, pre-registered, replayed; reporting gaps on seed selection and artefacts |
| Evidence Sufficiency (25%) | 65 | Adequate | One simulator, one speed ratio for the two-sided rounds, three seeds, 23 references |
| Argument Coherence (15%) | 68 | Adequate | Thesis clear; abstract/conclusion over-reach by a shade in several places |
| Writing Quality (15%) | 72 | Adequate–Strong | Dense but precise; naming split between figures and text |
| Significance & Impact (optional) | 74 | Good | Reusable protocol and a corrective negative result for safety-filtered competitive systems |
| **Weighted Average** | **70.0** | Rubric band: Minor Revision | Contract arithmetic (F2) yields major_revision; the contract governs |

## Editorial Decision

editorial_decision=major_revision

Rationale: F2 fired (D1, D2 and D3 each at warn) and nothing of higher severity fired; the required revisions are textual and reporting changes that can be made from records the authors already hold, so the re-review should be short.
