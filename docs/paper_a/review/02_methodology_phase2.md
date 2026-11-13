# Peer Reviewer 1 (Methodology) — Phase 2 review

Contract `reviewer/reviewer_full/v1` (v3.6.2). Scored against my Phase 1 scoring plan; no dissent is filed.

## Dimension Scores

### D1: methodology_rigor
score: warn

Warn triggers matched. (i) "3–4 seeds per arm behind a comparative claim": Sec. III-E (main.tex l. 133) "three seeds (1–3)"; Sec. VI "Each learned arm has three seeds"; the headline verdicts are per-seed votes (C1 "on two of three seeds"). (ii) "ten or more uncorrected comparisons with the count stated but no family-wise reading": roughly 150 paired tests across Secs. IV-A–IV-D at a per-test alpha of 0.05 (Sec. III-G, l. 139) with no multiplicity policy; marginal cells at IV-A (p = 0.043, p = 0.056), IV-A (smallest p = 0.024 vs Struct), IV-C (p = 0.02) and Table 4 (p = 0.046). I checked that no headline verdict rests on a marginal cell (H-TAC-1 seed 1 passes without its Greedy cell; H-FLAT-2 seed 3 without its Hug cell), so this stays at warn. (iii) "code 'on acceptance'": footnote 1 (l. 139) and the Acknowledgment (l. 288) defer the records, pre-registrations and code to camera-ready. (iv) "frozen-population results correctly scoped in the methods but the scope qualifier missing from abstract or conclusion": Sec. IV-D's first sentence (l. 240) and Sec. VI (l. 281) scope the sweep to frozen players; the abstract (l. 42, "Removing the filter raises the value only from 0.138 to 0.184"), C3 (l. 66, "the filter accounts for 0.046 of the value") and the Conclusion (l. 284, "the filter's share of the value is small") do not. (v) A reproducibility gap not in my trigger list but in the same class: the rule that chose Sel-R-1 and Evs-1 for the meta-game and the sweep (pre-registration S24.2 A2.4: highest / lowest mean capture over the stage's gate opponents, on the same 500-episode stream the matrix then reports) appears nowhere in the manuscript (Sec. IV-C "Meta-game", l. 229; Fig. 3).

Block clauses considered and not applied. "A checkpoint chosen on the reported evaluation set ... where the headline number is that checkpoint's": the saddle value 0.138 is indeed the best-of-three pursuer seed against the best-of-three evader seed on the episodes that constitute the matrix. I do not apply the block because the rule was pre-registered, it picks the strongest policy on both sides rather than favouring the proposed method, all three seeds' cells are in Table 3 and the IV-C text, and I verified from the records that the qualitative result holds for every one of the nine seed pairings (pure saddle, learned evader in the support, value 0.136–0.192). "An equilibrium, game-value or unexploitability claim drawn from a sweep over frozen players alone": the value claimed is explicitly that of the 4×4 empirical game, which is defined over fixed policies, and Sec. VI states the retraining caveat; the defect is the missing qualifier in the abstract and conclusion, which is the warn pattern.

Pass elements verified against the records: n = 500 and seed0 = 2 400 000 in every cell of all seven round files; the final checkpoint is the evaluated one (training scripts, `ckpt_path` returns the 3 000 000 mark); every arm has the same 3 M-step budget and SB3 defaults, disclosed as untuned (Sec. VI); every p-value and delta quoted in Secs. IV-A, IV-C and Table 4 reproduces from the per-episode `capture_ind` lists with the exact two-sided binomial on discordant pairs (`pe2/derisk.py::paired_indicator_test`); CG-V1 shows the 12 delta = 0 sweep cells bit-identical to the stored records over all 500 episodes.

### D2: domain_accuracy
score: warn

Trigger matched: "a solution concept named correctly in the methods but loosely in the discussion". Sec. IV-C (l. 229) defines exploitability with the citation and says "the minimax strategy of this pool"; Sec. VI (l. 281) says "'minimax' is relative to that pool, not to the full game". The abstract (l. 42, "the learned orbit remains the minimax evader with no filter at all"), C3 (l. 66, "the learned orbit is minimax without any filter"), the Discussion heading (l. 275, "its margin changes who is minimax") and the Conclusion (l. 284, "which evader is minimax") drop the pool qualifier, and "the value" is used throughout for the 4×4 game's value as if it were the game's. No block trigger: Proposition 1 (l. 99–106) states its scope (true pose, static map, hard rows only, nothing about progress or the evader) and is matched by zero contact steps in every shielded record; learned and scripted components are labelled consistently (Sec. III-F); no prior-work number is used as a comparator. Representation of the literature is Reviewer 2's lane.

### D3: argumentative_coherence
score: warn

Triggers matched. "A single over-reaching sentence in abstract ... that the body qualifies": abstract (l. 42) "the pursuer's selector beats the best fixed intercept law on held-out evaders" against C1 and the Fig. 1 caption "on two of three seeds" and Sec. IV-A (seed 3: −0.030 on Greedy). "A failed secondary hypothesis reported in results but absent from the discussion", applied to the registered record as a whole: the Discussion (l. 277) states "Three of the pre-registered hypotheses of this study failed"; the round records show nine gated failures (H-ZS-2, H-ZS-3, H-PR-1, H-OS-1, H-OS-2, H-ALT-1, H-ALT-2, H-VAL-1, H-VAL-2) and one registered reported comparison that failed (H-TAC-1b, selector vs rule table, 1/3 seeds) which the manuscript never states; H-ZS-2 and H-ZS-3 (Sec. IV-B) and H-ALT-2 (Sec. IV-C) are not named as failed. "Restricted-game or frozen-population qualifier present in methods and results but dropped in the discussion": as under D1. A further coherence gap: the Discussion's "it is the selector's choices rather than its library alone (it departs from the rule table ...)" (l. 271) rests on a tactic-profile chi-square, while the registered capture comparison against the rule table over the same library failed on the held-out evaders (recomputed from the records: Greedy +0.024 p = 0.26, +0.038 p = 0.048, −0.054 p = 0.014; Hug +0.004, +0.016, −0.018, all n.s.; the gain over the rule table is confined to the training evader, +0.086 to +0.122, p < 0.001). No block trigger: no failed hypothesis is reported as supported; Tables 1–4 agree with the text and with the records; both players are evidenced at the same standard (3 seeds, N = 500, pairing); mechanism statements are tied to rendered episodes and hedged.

### D4: cross_disciplinary_relevance
score: pass

The filter, the tactic libraries, the selector decision process and the evaluation protocol are each defined in one place (Secs. III-B, III-C, III-D, III-G); exploitability is defined with its source (Secs. II and IV-C); the solution concept is the pure saddle of a normal-form matrix, stated plainly; cross-field claims are scoped (Prop. 1's assumptions; pool-relative minimax in Sec. VI). Minor: "exploitability" appears in the abstract before any definition.

### D5: writing_and_structure
score: pass

Tables carry N and seeds in their captions; Figs. 1 and 4 carry Wilson 95% bars with the definition in the caption; the rendered pages print at 7–8 pt type and are legible (checked pages 5–7 of the compiled PDF); captions use "Figure"/"Table" with terminal periods and Arabic numbers; 8 pages within the 4–12 limit. Minors (listed below): the Table 1 caption's half-width range is 0.04–0.05 but the records give 0.036–0.044; Tables 2 and 3 carry no interval; Table 4 prints "n.s." for two cells whose p-values (0.58, 0.32) exist; the author e-mail is a placeholder.

## Failure Condition Checks

### F1
fired: false — no mandatory dimension scores block.

### F2
fired: true — D1, D2 and D3 all score warn (two or more mandatory dimensions at warn or worse).

### F3
fired: false — D4 (high) scores pass.

### F0
fired: false — not every mandatory dimension scores pass.

## Review Body

# Peer Review Report

## Manuscript Information
- **Title**: Learned Tactic Selection for Both Players of a Safety-Filtered Pursuit–Evasion Game
- **Manuscript ID**: (none; IC-FMAS 2027 Track 2 submission draft of 2026-11-02)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Experimental reinforcement-learning researcher with a statistics background, reviewing for robotics and ML venues.

### Review Focus
Whether the study design licenses each inference: seeds and unit of analysis, pairing and N, the exact paired test and Wilson intervals as reported, multiplicity, baseline fairness, faithfulness to the pre-registration (including failures and deviations), the checkpoint rule, reproducibility, and in particular what a sweep over frozen players and a 4×4 meta-game can and cannot support.

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] Major Revision
- [ ] Reject

### Confidence Score
4 — the evaluation methodology is squarely my area; the CBF content and the literature positioning belong to other reviewers.

### Summary Assessment
The paper trains maskable-PPO selectors over five hand-written tactics for both players of a filtered pursuit–evasion game and reports, under a pre-registered protocol (3 seeds, 500 paired episodes per cell, exact paired test, Wilson intervals), that the pursuer's selector beats the best fixed law on two of three seeds and all learning baselines in 9/9 cells, that one round of evader learning drops capture from 0.53–0.78 to 0.09–0.22 against every pursuer, that the 4×4 meta-game has a pure saddle at 0.138, and that a margin sweep over frozen players attributes only 0.046 of that value to the filter. The design is unusually careful for a conference paper: fixed evaluation stream, exact paired test matched to the binary outcome, final-checkpoint rule, same-budget baselines with a cadence-matched flat family, construct gates watched before any rate was read, and failures reported. Every number I recomputed from the per-episode records matches. The weaknesses are of reporting and inference rather than of design: the registered record is under-reported (the paper says three hypotheses failed; nine gated ones did, and a registered comparison against the rule table that undercuts the "choices, not library" reading is omitted), the seed-selection rule behind the meta-game cell and the sweep is undisclosed and the value carries no uncertainty (my paired bootstrap gives 95% [0.118, 0.174] with the saddle's pursuer unresolved between Sel-2 and Sel-R-1), the frozen-player and pool-relative qualifiers are dropped from the abstract and conclusion, and there is no multiplicity policy. All of these are fixable from the archived records without new training; the contract's F2 condition fires, so the decision is major revision.

## Strengths

### S1: Pre-registration backed by machine-checked provenance
Each round file records the pre-registration's SHA-256 and the frozen commit; the sweep's CG-V1 gate re-derived twelve delta = 0 cells bit-identically over all 500 episodes against the stored records; the training scripts implement "the final checkpoint is always the one evaluated" (Sec. III-E). This is the strongest provenance chain I have seen in a submission of this size.

### S2: Evaluation design matched to the outcome, and the numbers hold
Fixed stream (seeds 2 400 000 onwards), N = 500 paired per cell, exact two-sided binomial on discordant pairs, Wilson intervals (Sec. III-G). I recomputed from the `capture_ind` lists: the nine Sel-vs-Lead cells (e.g. +0.048, p = 0.0433; +0.054, p = 0.00046; +0.032, p = 0.0559), the three memory-ablation cells (p = 0.76, 0.0152, 0.00129), the H-EV-1 drops (−0.616/−0.590/−0.628, p ≤ 2.4e−74), the two significant Sel-R losses (p = 0.00037, 0.021), all eight Table 4 tests, the maximum p against flat PPO (1.9e−6) and the minimum against Struct (0.0236). All agree with the manuscript.

### S3: Same-budget baselines, a cadence control and a no-memory ablation
Every learned arm has 3 M environment steps and SB3 defaults (Sec. III-E, III-F); Flat-40 removes the decision-rate confound; Sel-NM separates the memory inputs from the architecture; the untuned-baseline caveat is stated (Sec. VI).

### S4: Failures that changed the design are reported
Per-ratio training (Sec. IV-B), the sixth tactic (IV-B), the pursuer's retraining (IV-C) and the filter-sets-the-value reading (IV-D) are all reported as failures with their numbers, and the Limitations name the frozen-sweep, pool-relative and three-seed limits.

### S5: The safety claim is scoped and matched by the records
Proposition 1 replaces "certified" with an explicit statement (true pose, static map, hard rows), and the empirical counterpart (0 contact steps in 202 shielded cells × 500 episodes) is consistent with `violation_steps = 0` in every shielded record I read.

## Weaknesses

### W1: The registered record is under-reported and the count of failures is wrong
**Problem**: Sec. V (l. 277) says "Three of the pre-registered hypotheses of this study failed". The round records show nine gated failures (H-ZS-2, H-ZS-3, H-PR-1, H-OS-1, H-OS-2, H-ALT-1, H-ALT-2, H-VAL-1, H-VAL-2). H-ZS-2 (ordering), H-ZS-3 (profile shift; the registered trigger for the per-ratio round) and H-ALT-2 (non-inferiority) are never named as verdicts, and the registered reported comparison H-TAC-1b (selector vs the rule table over the same library) is absent although it failed on 2/3 seeds.
**Why it matters**: pre-registration is a stated contribution of the paper; a reader checking the registered record against the text will find the count false. H-TAC-1b is the comparison that isolates learning from library, and it bears directly on the Discussion's "it is the selector's choices rather than its library alone".
**Suggestion**: add a registered-verdict table (hypothesis, bar, verdict, where reported) in the paper or supplement; correct the sentence; state H-TAC-1b with its numbers and revise the Discussion sentence to "it chooses differently from the rule table; on the held-out evaders it does not capture more than the rule table (1/3 seeds), and its gain over the rule table is on the training evader".
**Severity**: Critical (text-only fix, but required).

### W2: The meta-game cell is a selected cell with no uncertainty and an undisclosed selection rule
**Problem**: Sel-R-1 and Evs-1 were chosen by the pre-registered rule "highest / lowest mean capture over the stage's gate opponents" on the same 500-episode stream that the matrix reports (S24.2 A2.4); the manuscript does not say so (Sec. IV-C, Fig. 3). The value 0.138 is reported without an interval; Sel-2 and Sel-R-1 differ by 0.002 against Evs-1.
**Why it matters**: a paired bootstrap over episodes (2 000 resamples, all 16 cells jointly) gives value 95% [0.118, 0.174], with the maximin pursuer Sel-R-1 in 52% and Sel-2 in 46% of resamples; across the nine seed pairings the value is 0.136–0.192 and the saddle row alternates between Sel-2 and Sel-R. The qualitative claim (pure saddle, learned evader support) is robust; the specific cell and value are not resolved at this N.
**Suggestion**: state the selection rule; report the bootstrap interval on the value and on each setting of Fig. 4(b) (my values: no filter [0.158, 0.218], delta = 0 [0.120, 0.170], 0.05 [0.116, 0.170], 0.10 [0.136, 0.196], 0.20 [0.042, 0.078]); say that the saddle's pursuer identity is tied; mark the training relationships (Evs-1 trained against Sel-2; Sel-R-1 trained against Evs-1) in the Fig. 3 caption; if budget allows, confirm the four headline cells on a fresh evaluation stream, since selection and every later test share stream 2 400 000+.
**Severity**: Major.

### W3: Frozen-player and pool-relative qualifiers are dropped where the claims are strongest
**Problem**: abstract, C3 and Conclusion state "the filter's share of the value", "removing the filter raises the value", "the learned orbit remains the minimax evader with no filter at all" without "of the 4×4 game of policies trained at delta = 0 and not retrained"; Sec. IV-D and Sec. VI carry the qualifier.
**Why it matters**: an unfiltered pursuer trained without the filter could learn a different policy; the paper says so in Sec. VI but the abstract reader will not see it.
**Suggestion**: add the qualifier in the three places; keep "Future work ... players retrained at each margin".
**Severity**: Major.

### W4: No multiplicity policy, and non-significance read as absence of effect
**Problem**: roughly 150 paired tests at alpha = 0.05 with per-seed voting as the only control; several reported cells are marginal (p = 0.043, 0.046, 0.056, 0.024, 0.02). In Sec. IV-D "Removing the filter raises capture ... significantly for the two learned pursuers (+0.034 and +0.046) and not for the rule table": the rule table's change is +0.022 with b = 27, c = 16, p = 0.13, and a 95% interval on the paired difference of [−0.004, +0.048], which is compatible with the learned pursuers' rises.
**Why it matters**: the verdicts survive (I checked the marginal cells are not load-bearing), but the paper should show that rather than leave it to the reader; and "not for the rule table" is a power statement at N = 500, not a null finding.
**Suggestion**: state a policy (e.g. Holm within each registered family) and report that no verdict changes; report paired-difference intervals for the headline comparisons (b and c are in the records); rephrase the rule-table sentence and attach the interval [0.018, 0.074] to C3's 0.046.
**Severity**: Major.

### W5: "The pursuer cannot answer within its library" overreaches one retraining round
**Problem**: Sec. IV-C heading, Sec. V ("the pursuer's library, not its learning, is binding") and the Conclusion ("an obstacle-orbit equilibrium that the pursuer cannot escape within its library"). The evidence is one retraining round at 3 M steps with SB3 defaults, the learned evader one quarter of training episodes, final checkpoint; H-ALT-1 failed 0/3.
**Why it matters**: the result shows this round did not recover capture; it does not show that no selector over this library can.
**Suggestion**: "did not answer in one retraining round at this budget"; list the untested alternatives (longer budget, learned-evader-only opponent, curriculum) in Sec. VI.
**Severity**: Major.

## Detailed Comments

### Title & Abstract
Accurate title. The abstract drops two qualifiers the body carries: "on two of three seeds" for the fixed-law claim, and "of the 4×4 game of frozen players" for the value and minimax statements. "Exploitability" is used before any definition.

### Introduction
Q1–Q3 are answerable by the design. The claim "no cell was re-run after a number had been seen" is consistent with the stop rules and amendment logs in the pre-registrations I read; the S24.1 A5 amendment (a construct-gate tolerance restated after the guard refused, disclosed) and the S24.2 A3.1 held-out relabelling (before training) should be acknowledged in the released protocol, as the text promises.

### Methodology / Research Design
Sound: fixed stream, paired cells, exact paired test, final checkpoint, equal budgets, pre-registered bars. Missing: the seed-selection rule for the meta-game and sweep (W2); a multiplicity policy (W4); the number of decisions per arm (the prereg records ≈86 500 for the selector against ≈300 000 for Flat-10, so Flat receives about 3.5× more PPO updates and a shorter credit horizon per environment step — state both sides of the cadence confound beside Flat-40). The evaluation N of 500 gives Wilson half-widths of 0.036–0.044 at these rates and resolves paired differences of about 0.04; the sweep's hypotheses asked for differences of 0.02–0.05 and were therefore under-powered for the rule-table and Full rows.

### Results / Findings
Tables 1–4 agree with the records. Table 3 and Table 2 would benefit from a caption-level interval note (Table 3's intervals span [0.064, 0.114] to [0.184, 0.256]). Table 4: print the two "n.s." p-values (0.58, 0.32) and mark that Full's no-filter cell falls. Fig. 4(b) needs uncertainty bands (W2). The support switch at delta = 0.20 is robust: Hug is below Evs-1 for the three library pursuers (p = 0.00046, 0.0094, 1.1e−5) and in 100% of bootstrap resamples.

### Discussion
"Three of the pre-registered hypotheses failed" is wrong (W1). "It is the selector's choices rather than its library alone" is not supported by the capture comparison against the rule table on held-out evaders (W1). The frozen-sweep reading is correctly hedged in the last paragraph of Sec. V but not in the heading sentence.

### Conclusion
"Cannot escape within its library" (W5) and "the filter's share of the value is small" (W3) need their qualifiers.

### References
Not my lane; I note only that the venue checklist asks for superscript citation numerals, which the IEEE template does not produce — confirm with the organisers.

## Ranked Actionable Items

1. **CRITICAL** — Registered verdicts. Where: Sec. V l. 277; Secs. IV-A, IV-B, IV-C. Wrong: "three failed" (nine gated failures in the records); H-ZS-2, H-ZS-3, H-ALT-2 unnamed; H-TAC-1b omitted (vs Rules on held-out evaders: Greedy +0.024 p = 0.26, +0.038 p = 0.048, −0.054 p = 0.014; Hug +0.004, +0.016, −0.018 n.s.; training evader +0.086/+0.122/+0.118 p < 0.001). Fix: registered-verdict table; correct the count; state H-TAC-1b and revise the "choices rather than library" sentence.
2. **MAJOR** — Meta-game selection and uncertainty. Where: Sec. IV-C l. 229, Fig. 3, abstract. Wrong: selection rule for Sel-R-1/Evs-1 undisclosed; value without interval; saddle row tied (0.138 vs 0.136). Found: paired bootstrap 95% [0.118, 0.174]; Sel-R-1 52% / Sel-2 46% of resamples; nine seed pairings give 0.136–0.192, always a pure saddle with the learned evader. Fix: state the rule, report the interval and the tie, add the nine-pairing range, mark training relationships in the caption; optionally confirm the four headline cells on a fresh stream.
3. **MAJOR** — Frozen/pool qualifiers. Where: abstract l. 42, C3 l. 66, Sec. V l. 275 heading, Conclusion l. 284. Fix: "of the 4×4 game of policies trained at delta = 0 and not retrained" and "minimax within this pool"; add bootstrap bands to Fig. 4(b).
4. **MAJOR** — Multiplicity and paired-difference intervals. Where: Sec. III-G l. 139; marginal cells at l. 146, 169, 227, Table 4. Fix: state a family-wise policy and that verdicts are unchanged; report b, c and a 95% interval for the headline paired differences.
5. **MAJOR** — Non-significance as null. Where: Sec. IV-D l. 266 "and not for the rule table"; Sec. V "the filter's cost ... is 0.046". Found: Rules +0.022, 95% [−0.004, +0.048]; Sel-R-1's +0.046, 95% [0.018, 0.074]. Fix: "not resolved at N = 500"; carry the interval on 0.046.
6. **MAJOR** — Pre-registration not inspectable at review. Where: footnote 1 l. 139; Acknowledgment l. 288. Fix: supplementary archive at submission with the seven pre-registration documents, their commit hashes and the `spec_sha256` values the round files already record, the per-cell JSON records and the gate scripts.
7. **MAJOR** — "Cannot answer within its library". Where: Sec. IV-C l. 227 heading, Sec. V l. 273, Conclusion l. 284. Fix: "did not answer in one retraining round at this budget"; name untested alternatives in Sec. VI.
8. **MINOR** — Abstract l. 42: add "on two of three seeds".
9. **MINOR** — Table 1 caption: half-widths are 0.036–0.044 by the records, not 0.04–0.05; add interval notes to Tables 2 and 3.
10. **MINOR** — Table 4: replace "n.s." with p = 0.58 and p = 0.32; mark the direction of Full's no-filter change.
11. **MINOR** — Range slips against the records: l. 146 "switches tactic 9–11 times" is 8.0–11.5; l. 179 "0.318–0.346 against 0.382–0.394" is 0.316–0.346 against 0.360–0.394 (seed 3 omitted); l. 227 "losing 0.05–0.08 against the tactical evader on two seeds" is −0.082 on both (0.700 vs 0.782; 0.696 vs 0.778).
12. **MINOR** — Sec. III-F / VI: state the decisions-per-arm asymmetry (≈86.5 k selector vs ≈300 k Flat-10 vs ≈90 k Flat-40 in 3 M steps) beside the cadence argument.
13. **MINOR** — Author e-mail and repository placeholders; confirm the venue's superscript-citation rule against the IEEE template.

## Questions for Authors
1. In the no-filter control, what does the simulator do on contact: clamp the body at the surface, allow penetration (the text reports −0.10 m clearance for Full), or neither? The unfiltered pursuer's reachable set decides whether 0.184 is the value of "the same game without the filter".
2. Confirm that Sel-R-1 and Evs-1 were selected by the A2.4 rule on the same 2 400 000+ cells that the meta-game and the delta = 0 sweep report, and whether any headline cell has been evaluated on a different stream.
3. How many paired tests were run across the seven rounds, and does any registered verdict change under Holm within its family?
4. For Sel-R, was the learned evader 25% of training episodes, and was a run against the learned evader alone (or at a larger budget) considered before concluding that the library is binding?

## Minor Issues

### Language / Grammar
- l. 42: "beats the best fixed intercept law on held-out evaders" — add the seed qualifier.

### Citation Format
- Venue checklist asks for superscript "[n]"; the template prints inline brackets.

### Figures and Tables
- Table 1 caption half-width range; Tables 2–3 without intervals; Table 4 "n.s."; Fig. 4(b) without bands; Fig. 3 caption should name the training relationships of the four policies.

### Layout
- Author e-mail "[to confirm]"; repository URL placeholder.

## Dimension Scores (rubric, 0–100; advisory — the contract's failure conditions govern the decision)

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 72 | Adequate–Strong | Two-sided learned tactic selection under a filter with a margin sweep; weigh with Reviewer 2 |
| Methodological Rigor (25%) | 68 | Adequate | Sound paired design and provenance; 3 seeds, no multiplicity policy, selection rule undisclosed, no interval on the value |
| Evidence Sufficiency (25%) | 68 | Adequate | 23 references; the empirical evidence is strong and reproducible from records |
| Argument Coherence (15%) | 64 | Adequate | Abstract/conclusion overreach; registered failures under-counted |
| Writing Quality (15%) | 78 | Strong | Dense but precise |
| Literature Integration (optional) | — | — | R2 focus |
| Significance & Impact (optional) | — | — | R3 focus |
| **Weighted Average** | **70** | **Minor-revision band by the rubric; Major Revision by the contract (F2)** | |

## Editorial Decision
editorial_decision=major_revision

Rationale: F2 fires (D1, D2 and D3 at warn) and no condition of higher severity fires; the defects are of reporting and inference and are correctable from the archived records without new training.
