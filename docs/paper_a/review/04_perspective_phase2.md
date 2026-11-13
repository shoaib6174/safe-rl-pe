## Dimension Scores

### D1: methodology_rigor
score: warn
The empirical-game procedure is sound where it counts (500 paired episodes per cell on one seed stream, so rows are compared under common random numbers; Wilson intervals on every cell in the records and in the Table 1 caption; the full 4×4 matrix printed in Figure 3 and the five sweep matrices recoverable from Figure 4 and Table 4; both pools pre-registered before any cell was read). Three of my Phase 1 warn triggers nevertheless fire. (i) "Uncertainty reported but no robustness check of the equilibrium or value against entry noise": the equilibrium objects are reported as exact — §IV-C p.5 "a pure saddle point at (Sel-R-1, Evs-1) with value 0.138 and zero duality gap", Figure 3 caption p.6 "The pure saddle point (orange frame) is the retrained selector against the learned evader", the value to three decimals in the Abstract, C2, C3, §IV-D and Figure 4(b) — while the archived record (results/derisk/s24_2_meta_game.json) shows the row support rests on 0.138 against 0.136, i.e. 69 against 68 captures in 500, one episode, with Wilson intervals [0.111, 0.171] and [0.109, 0.169]; no interval is given for the value, no perturbation or bootstrap of the matrix is reported, and the same holds for the row-support switch to Sel-2 at δ = 0.10 (0.162 vs 0.152). The column support and the evader exploitabilities (0.256–0.644) are far outside noise, so no central claim collapses, which is why this is warn and not block. (ii) "The procedure for choosing the pool members unstated": the manuscript never says that the saddle row and column are best-of-three seeds chosen by pre-stated rules (pre-registration A5.2: Sel-R-1 = highest mean capture over six evaders; A2.4: Evs-1 = lowest mean capture), nor why Sel-2, the strongest Sel seed in Table 1, is the Sel row; §IV-C p.5 simply lists the pools. (iii) "Code or data promised rather than available": footnote in §III-G and the Acknowledgment, release at camera-ready.

### D2: domain_accuracy
score: warn
What I verified as correct: the zero-sum premise holds (§III-A "+1 on capture and −1 on timeout"; pe2/env.py lines 282–292: every episode ends in capture or timeout, contact never terminates, the evader receives the negation; capture rate = (1 + E[reward])/2 is an affine image, so the capture-rate matrix is constant-sum and value, saddle and supports carry over unchanged); the pure saddle is true on the printed matrix (row minima 0.110/0.086/0.136/0.138 give maximin 0.138, column maxima 0.620/0.394/0.782/0.138 give minimax 0.138); the exploitability definition in §IV-C ("the value minus a pure policy's worst-case payoff") is the standard zero-sum one and the eight numbers recompute exactly; [wellman2006methods] and [lanctot2017unified] are characterised correctly in §II. The Phase 1 warn trigger "terms used correctly but unscoped — 'the value of the game' where only the k×m restricted game is meant" fires in every front- and back-matter location: Abstract p.1 "the learned orbit remains the minimax evader with no filter at all" and "lowers the value to 0.058 while switching the minimax evader"; §I Q3 "Is the resulting value of the game a property of the filter"; C3 "the learned orbit is minimax without any filter"; §V p.6 "the learned orbit is minimax with no filter at all"; §VII "its margin changes which evader is minimax". The scoping ("the minimax strategy of this pool", §IV-C; "relative to that pool, not to the full game", §VI) exists only in Results and Limitations. Three further loose uses of the same kind: §I p.1 "It is zero-sum, it has a well-defined value" refers to the Isaacs game, and the paper then uses "value" for the LP value of a 4×4 empirical game without ever distinguishing the two objects (and the game actually played has asymmetric information, §III-A, for which the textbook existence statement is not the one cited); §IV-D p.6 "the pursuer side stays non-exploitable within its library" is one notch too strong for exploitability ≤ 0.054 against four evaders and names the wrong restriction — a pursuer's measured exploitability is bounded by the evaders tried (one evader oracle, trained against Rules/Lead/Sel-2 and never against Sel-R-1), not by the pursuer's library; §IV-C "zero duality gap" is offered as a property of this game although it holds for every finite zero-sum matrix game and is only a solver check. Also §IV-D/Figure 4(b): "The value of the 4×4 game is 0.184 with no filter, 0.138 at δ = 0" is the value of a different 4×4 game from Figure 3's (Full replaces Lead); the δ = 0 values coincide only because the saddle pair sits in both pools, which is not said. None of this reaches my block trigger: the payoff is zero-sum, the saddle is not false, and the exploitability numbers are backed by one evader best-response oracle for three rows and by the retraining step on the pursuer side.

### D3: argumentative_coherence
score: warn
Both players' claims rest on printed tables (Tables 1–3, Figure 3), the held-out asymmetry (scripted families for the pursuer, selector seeds for the evader) is disclosed in §IV-C and in the Abstract's "held-out selector seeds", and the Results do not inflate the saddle into full-game optimality, so the block triggers do not fire. Two of my warn triggers do. (i) "Conclusion verbs one notch stronger than the evidence": §VII p.7 "Applied to both players it converges in one evader step to an obstacle-orbit equilibrium" — "converges" is restricted-game/PSRO vocabulary whose test is that neither player's best-response oracle improves on the current profile; the paper ran the pursuer half (Sel-R, §IV-C, 0.11–0.17 against the learned evaders) and never the evader half (no evader trained against Sel-R-1); the pre-registration's own reading "a pure equilibrium in the learned pair means the alternation converged" is a reading of a point estimate, not a convergence test. (ii) "The obvious rival explanation left unaddressed where it is the natural alternative reading": §V p.6 "the meta-game confirms that the pursuer's library, not its learning, is binding" and "Its limits are the library's limits ... adding a plausible one made the pursuer worse" (sixth tactic, §IV-B p.4, 0.318–0.346 against 0.382–0.394). For an exact best responder, enlarging a strategy set cannot lower its maximin value; a learner that gets worse with a sixth tactic is therefore direct evidence that the learner (3 × 10^6 steps, library-default hyper-parameters, one round) is not a reliable best-response oracle — and that same oracle's failure to improve as Sel-R is the sole evidence that the library binds. The paper reads both results as "library" and never as "oracle". (iii) The frozen-player caveat on the sweep's "value" lives only in §VI, while the Abstract ("Removing the filter raises the value only from 0.138 to 0.184"), C3 ("the filter accounts for 0.046 of the value") and §VII ("the filter's share of the value is small") are unqualified — my trigger "caveat confined to the limitations section while abstract and introduction speak unqualified". Minor but of the same kind: §IV-C "The two selector seeds that were not in the evader's training mixture (Sel-1 and Sel-3 for Evs-1 and Evs-3) fare no better than the one that was" silently omits Evs-2, for which the held-out Sel-3 captures 0.218 against the in-mixture 0.192 (Table 3).

### D4: cross_disciplinary_relevance
score: pass
A multi-agent-systems reader can reconstruct the game without robotics background: §III-A gives players, dynamics, arena, payoff and the two terminal outcomes, and the information structure (90° field of view with occlusion against a true-state evader); §III-B says exactly where the filter sits (every pursuer command from any tactic; the evader unfiltered; no inter-agent constraint, so capture is never prevented); §III-C/D give the strategy sets as tactics with applicability and done predicates under a semi-MDP selector; §IV-C gives the meta-game payoff, both pools and a one-line exploitability definition; §II places the reading in EGTA and PSRO. The transferable lesson is articulated and argued from the sweep rather than gestured: §V "the filter's margin should be reported as a parameter of the game, with the unfiltered value beside it, before any statement about what the filter costs is made", with the explicit complement to [wu2026competence] (perfect filter, filter not varied); the observation that the margin changes which evader is minimax rather than how strong the learned one is transfers to any safety-filtered competitive system. §VI surfaces the pool restriction, the frozen-player caveat, the information asymmetry, the single ratio and the un-isolated obstacle role. Gaps that do not reach my warn trigger because the lesson and several assumptions are present: the strategy-exploration framing is missing (the libraries are the strategy spaces and were authored by the experimenters; which seed enters the pool is part of that design); the evader-side exploitability formula is not written; §V "a known asymmetry of pursuit–evasion among obstacles" is asserted without a citation; and the one foundational assumption surfaced nowhere is that the learner is a reliable best-response oracle (see D3).

### D5: writing_and_structure
score: warn
Eight pages inside the 4–12 window; captions use the full words with terminal periods and Arabic numbers as the venue checklist asks; all three figure PDFs are natively at the printed width (252 pt and 515.5 pt), so nothing is downscaled and Figure 3's cell values, row labels "pursuer" and column labels "evader" with the "pursuer capture rate" colour bar are legible at 100 %; the saddle is framed. My warn trigger "inconsistent policy naming between text and tables" fires: the text and tables use Sel-2, Sel-R-1, Evs-1, while Figures 3–4 use "Selector", "Selector, retrained", "Learned selector"/"learned" with no seed identifier, so a reader cannot see from the figures that the saddle row and column are specific best-of-three seeds; Table 3 (rows = evaders, columns = pursuers) is transposed relative to Figure 3 (rows = pursuers); Table 4's p column mixes numeric values with "n.s." in two rows. Outside my lane but noted: the venue checklist asks for superscript "[n]" in-text citations, and IEEEtran sets them inline.

## Failure Condition Checks

### F1
fired: false — no mandatory dimension scores block (D1 warn, D2 warn, D3 warn).

### F2
fired: true — three mandatory dimensions (D1, D2, D3) score warn; the condition needs two.

### F3
fired: false — D4, the only high-priority dimension, scores pass.

### F0
fired: false — not every mandatory dimension scores pass.

## Review Body

# Peer Review Report

## Manuscript Information
- **Title**: Learned Tactic Selection for Both Players of a Safety-Filtered Pursuit–Evasion Game
- **Manuscript ID**: not assigned (pre-submission panel)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1
- **Target venue**: IC-FMAS 2027, Track 2 (Game Theory Based Multi-Agent Optimization / MARL Systems)

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Multi-agent-systems researcher with an empirical- and algorithmic-game-theory background (empirical game-theoretic analysis, policy-space response oracles, exploitability, equilibrium selection); no robotics hardware experience. I read the paper as a Track 2 reader would: for whether the game-theoretic objects are the ones named, with their limits stated, and for what transfers beyond this testbed. I defer the control-barrier-function implementation and the learning-side statistics to Reviewers 1 and 2 and the fallacy audit to the Devil's Advocate.

### Review Focus
Whether value, minimax, pure saddle point and exploitability are used with their home-field meaning and scoped to the restricted 4×4 game wherever they appear; whether the zero-sum premise holds under the stated payoff; whether an adjacent-field reader can reconstruct the game; what transfers and whether it is argued; which foundational assumption went unquestioned; and the matrix presentation.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] Major Revision
- [ ] Reject

### Confidence Score
4 — the game-theoretic content is squarely in my field and I recomputed it from the archived records; the filter's implementation details are not, and I have not checked Proposition 1 beyond reading it.

### Summary Assessment
The paper builds both players of a safety-filtered pursuit–evasion game as learned selectors over five hand-written tactics, shows the pursuer's selector beats fixed laws and flat and structured learners under the filter, shows the evader's selector finds an obstacle orbit that drives every pursuer to 0.09–0.22 capture, reads a 4×4 pool of policies as a zero-sum game (pure saddle, value 0.138), and then falsifies its own registered reading that the filter sets that value by sweeping the filter's margin with an unfiltered control. From my seat the game-theoretic machinery is correct: the payoff is genuinely zero-sum, the LP and the saddle check out on the printed matrix, and the exploitability definition is the standard one. The paper's best quality is that it keeps its claims narrower than it set out to make them and says so. Its central weaknesses are of scope and inference rather than of fact: "value", "minimax" and "non-exploitable" are left unqualified in the abstract, contributions, discussion and conclusion while the restriction to a 4×4 pool of frozen players is stated only in the results and limitations; the equilibrium objects are reported without uncertainty although the saddle's row is a one-episode tie; and the inference that "the library, not the learning, binds" never considers the rival reading its own sixth-tactic experiment supports — that the learner is not a reliable best-response oracle. These are revisable in one cycle, mostly by rewording and by analyses of records that already exist, so I recommend major revision under the contract's arithmetic rather than rejection.

---

## Strengths

### S1: The empirical-game machinery is the real thing, and pre-registered
The capture-rate matrix is read as a zero-sum game with an exact LP (scripts/s24_2_meta_game.py, pre-registration A5.1 fixed before any cell was read), the pure saddle at (Sel-R-1, Evs-1) is true on the printed matrix (§IV-C, Figure 3), and the exploitability definition (§IV-C, "the value minus a pure policy's worst-case payoff") is the standard zero-sum one — the same quantity the field uses, not a proxy. The pools were fixed in advance and no cell was run for the matrix. This is better EGTA practice than most robotics papers that borrow the vocabulary.

### S2: The zero-sum premise actually holds, and the paper says how
§III-A states the payoff (+1 capture, −1 timeout, two terminal outcomes only, contact never terminates), and the code matches; capture rate is an affine image of that payoff, so the matrix game's value and supports are those of the stated game. The evader's reward is the exact negation. A game-theory reader does not have to take the zero-sum reading on trust.

### S3: The margin sweep falsifies the authors' own registered reading
C3 and §IV-D report that "the filter sets the value" failed in the form registered: removing the filter moves the value 0.138 → 0.184 and the learned orbit stays the pool's minimax evader with no filter at all. The further observation that a 0.20 m margin changes which evader is minimax rather than how strong the learned one is (§IV-D, Figure 4b) is the kind of result that transfers: in a safety-filtered competitive system, the margin re-ranks the threats, not only the score.

### S4: The game is reconstructible by an adjacent-field reader
§III gives players, dynamics, information structure, payoff, the exact placement of the filter (pursuer only, every tactic, no inter-agent constraint) and the strategy sets as tactics with applicability and done predicates. Proposition 1 replaces "certified" by a statement with an explicit scope, which is the honest way to carry that word into a game-theory venue.

### S5: Paired evaluation and transparent failures
Every cell shares one seed stream (§III-G), so row comparisons are under common random numbers — the right design for an empirical game. Three failed hypotheses and two construct errors are reported as such (§V, "Construct discipline").

---

## Weaknesses

### W1: Game-theoretic vocabulary is scoped in the results and unscoped everywhere a reader skims
**Problem**: "value", "minimax" and "non-exploitable" appear without "of the 4×4 pool" or "for frozen players" in the Abstract ("the learned orbit remains the minimax evader with no filter at all"; "Removing the filter raises the value only from 0.138 to 0.184"), §I Q3, C3, §V and §VII; §IV-D says "the pursuer side stays non-exploitable within its library"; §I says the game "has a well-defined value" and then uses the same word for the LP value of a 4×4 empirical game; §IV-C reports "zero duality gap" as if it were a property of this game.
**Why it matters**: at a game-theory track the value of a restricted game and the value of the underlying game are different objects, exploitability measured against four evaders is a lower bound on exploitability, and "non-exploitable" means zero against every strategy. Readers of the abstract and conclusion will take the stronger meaning.
**Suggestion**: qualify every occurrence ("value of the 4×4 pool", "minimax within the pool", "for players trained at δ = 0 and frozen"); replace "non-exploitable within its library" with "no evader in the pool exploits any library pursuer by more than 0.054"; name the two objects once in §I ("the differential game's value" vs "the empirical game's value"); drop "zero duality gap" or call it the LP check it is.
**Severity**: Major

### W2: "The library, not the learning, binds" has an unexamined rival: the learner is not a best-response oracle
**Problem**: §V infers from Sel-R's failure to improve (§IV-C) that the pursuer's library is binding, and reads the sixth-tactic result (§IV-B: the enlarged library made the selector worse, 0.318–0.346 vs 0.382–0.394) as "the library's limits". For an exact best responder, adding a tactic cannot lower the maximin value; getting worse is evidence about the optimiser, not the strategy set. §VII then says the two-sided game "converges in one evader step", although only the pursuer half of a convergence check was run.
**Why it matters**: the paper's second contribution rests on which of these two readings is right. If the oracle is weak, the pool's saddle says little about what a pursuer over this library can do, and "converges" is not earned.
**Suggestion**: state the rival reading in §V and §VI; soften §VII to "one pursuer best-response step fails to improve"; if budget allows, run the missing half (an evader selector trained with Sel-R-1 in its mixture, evaluated in a 5×5 pool) or a longer/tuned pursuer run over the six-tactic library — either is a sharper test of "library vs learning" than the geometry sweep proposed as future work.
**Severity**: Major

### W3: Equilibrium objects carry no uncertainty, and the pool's membership is a best-of-seed choice the paper does not disclose
**Problem**: the value is printed to three decimals with no interval; the saddle row is 0.138 vs 0.136 — one episode in 500 — and Figure 3 frames it as "the" saddle; the row support switches at δ = 0.10 on a similar margin. Sel-R-1 and Evs-1 are best-of-three seeds by pre-stated rules (A5.2, A2.4) and Sel-2 is the strongest Sel seed, none of which §IV-C says.
**Why it matters**: an empirical game's equilibrium is a statistic of estimated cells; a game-theory reader expects to know how far it moves under the stated cell error, and which seed enters the pool is part of the strategy-set design that determines the equilibrium.
**Suggestion**: report the value with its interval (0.138 [0.111, 0.171]); resample the 16 cells from their (k, n) (or the paired indicators) and report how often Evs-1 is the column support, the value's distribution and each row's support frequency, for the main game and the five sweep settings — a few lines over existing records, no new episodes; say that the δ = 0.20 switch to the hugger is robust (0.058 vs 0.128 do not overlap) and the row identity is not; disclose the seed-selection rules and give the value's seed range (the Sel3 × Evs grid in §IV-C already spans 0.114–0.170).
**Severity**: Major

### W4: The sweep's 4×4 game is a different game from Figure 3's, and its "value" is for frozen players
**Problem**: §IV-D's pool replaces Lead by Full, so "the value of the 4×4 game is 0.184 with no filter, 0.138 at δ = 0" compares a different row set with Figure 3; the δ = 0 values coincide only because the saddle pair is in both pools. The frozen-player caveat (policies trained at δ = 0 evaluated zero-shot) appears only in §VI.
**Why it matters**: the headline "filter share 0.046 of value" is a difference between two restricted games of frozen players; a MAS reader will otherwise read it as a property of the game at each margin.
**Suggestion**: one sentence in §IV-D and the Figure 4(b) caption naming the pool and the frozen-player scope; qualify the Abstract and C3 accordingly.
**Severity**: Major

### W5: Presentation of the matrices
**Problem**: text and tables use Sel-2/Sel-R-1/Evs-1; Figures 3–4 use descriptive labels with no seed identifiers; Table 3 is transposed relative to Figure 3; Table 4 mixes numeric p-values with "n.s.".
**Why it matters**: a reader cannot map the saddle's framed cell to a specific seed, or read Table 3 and Figure 3 against each other without rotating one.
**Suggestion**: add seed identifiers to the figure labels or a crosswalk in the captions; orient Table 3 as Figure 3 (rows = pursuers) or say "transposed"; print all p-values; write the evader-side exploitability formula once.
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
The title is accurate and has no subtitle. The abstract reports the matrix results with the right numbers but without the pool or frozen-player scope (W1, W4); "held-out selector seeds" is the correct, modest description of the evader-side hold-out and should stay.

### Introduction
Q1–Q3 are a clear organising device. Q3's "value of the game" and the sentence "It is zero-sum, it has a well-defined value" conflate the underlying differential game with the empirical 4×4 game; name both objects once. The game as played has asymmetric information (§III-A), so the existence statement is not the textbook one in any case.

### Related Work
From my field the EGTA/PSRO paragraph is correct but thin for a Track 2 audience: it cites the method but not the two things a reader will ask about — strategy exploration (the equilibrium of an empirical game is a function of the strategy set the analyst chose; Jordan, Schvartzman & Wellman 2010) and equilibrium estimation error from sampled payoffs (Tuyls et al. 2020). One sentence each, with the paper's own numbers (five tactics per side; ±0.03 per cell), would place the 4×4 reading honestly. The adoption of Wu et al.'s exploitability definition is appropriate; note that theirs is measured against a checkpoint pool and yours against four pure policies, so both are pool-exploitabilities.

### Methodology / Research Design
Section III is the best part of the paper for an outsider: the game is fully specified. Two things a MAS reader needs that are missing: that the matrix's learned members are selected seeds (W3), and that the selector optimises a per-decision discounted objective (γ = 0.99 per decision) while the meta-game payoff is the undiscounted capture rate — harmless at this horizon, but say so in one clause.

### Results
§IV-C is correct and well scoped ("the minimax strategy of this pool"). §IV-D's "non-exploitable within its library" is the one phrase I would call a misuse rather than an omission (W1). The held-out sentence in §IV-C should cover all three evader seeds (Table 3 shows Sel-3 at 0.218 against Evs-2's in-mixture 0.192). Figure 3 is readable at print size and the frame is a good device; mark row minima and column maxima lightly so the saddle can be verified by eye, and add seed identifiers.

### Discussion
The transferable reading ("report the margin as a parameter of the game, with the unfiltered value beside it") is the paper's most useful sentence for a practitioner and is earned by the sweep. "The pursuer's library, not its learning, is binding" needs the rival reading (W2). "A known asymmetry of pursuit–evasion among obstacles" needs a citation or the word "known" dropped.

### Conclusion
"Converges in one evader step to an obstacle-orbit equilibrium" over-infers by one notch (W2); "the pursuer cannot escape within its library" is well scoped and could carry the sentence alone. The future-work list is the right one; I would add the second evader oracle before the geometry sweep.

### References
Twenty-three entries, correct for the game-theoretic citations I can judge; Wellman 2006 and Lanctot 2017 carry URLs but no DOIs, and Isaacs 1965 carries neither — the venue checklist asks for a DOI or link on every entry.

---

## Assumption Audit (R3)
- **Explicit assumptions**: zero-sum payoff (holds); a 4×4 pool stands in for the game (stated in §VI); frozen players across margins (stated in §VI). All withstand scrutiny once scoped.
- **Implicit assumptions**: that the PPO selector at 3 × 10^6 steps is a best-response oracle, so that its failure to improve licenses "no better pursuer exists in this library" — not stated anywhere, and the sixth-tactic result argues against it; that one selected seed represents its family in the pool; that the empirical game's equilibrium approximates the underlying game's.
- **Paradigmatic assumptions**: that a normal-form reading of a continuous-time, partial-information differential game captures its strategic structure, and that "equilibrium" is the right explanatory unit for what is, after one iteration, a restricted-game snapshot. The paper is candid about the first in §VI; the second is what makes "converges" the wrong verb.

## Cross-Disciplinary Connections and Reading Recommendations
- Jordan, Schvartzman & Wellman (2010), "Strategy exploration in empirical games", AAMAS — the equilibrium of an empirical game is a function of which strategies the analyst admits; your libraries are that choice.
- Tuyls, Pérolat, Lanctot, Leibo & Graepel (2020), "Bounds and dynamics for empirical game-theoretic analysis", JAAMAS — how far an equilibrium of a sampled matrix can sit from the true game's as a function of per-cell error; directly applicable to 500-episode cells.
- McMahan, Gordon & Blum (2003), "Planning in the presence of cost functions controlled by an adversary", ICML (double oracle) and Lanctot et al. (2017), already cited — your alternation is one double-oracle iteration; the stopping test is that neither oracle improves.
- Balduzzi et al. (2019), "Open-ended learning in symmetric zero-sum games", ICML — gamescapes and non-transitivity; why a single best-response step is not evidence of convergence and how to report population-level strength.
- Johanson, Waugh, Bowling & Zinkevich (2011), "Accelerating best response calculation in large extensive games", IJCAI — the home-field exploitability (value minus worst case against a full best response), useful to state the gap between pool-exploitability and the real thing.

## Practical Impact and Broader Implications
- **Real-world application**: the recommendation to report the filter margin as a game parameter beside the unfiltered value is cheap to adopt and would improve any paper that claims a "cost of safety" in a competitive setting.
- **Implementation feasibility**: the fixes above are text changes plus analyses of existing records; the one optional experiment (a second evader oracle) is one training round of the kind the paper already ran.
- **Stakeholders / ethics**: the paper studies an opponent that learns to exploit a safety filter's margin; this is standard adversarial-robotics material and the paper's framing (margin as a reportable parameter) is the responsible one. Certification stakeholders will appreciate Proposition 1's explicit scope over the word "certified".

---

## Questions for Authors
1. If an evader selector were trained with Sel-R-1 in its mixture and the pool enlarged to 5×5, do you expect the value to fall below 0.138? If this was not run, why does "converges" describe a one-sided test?
2. Since enlarging the pursuer's tactic set cannot lower an exact best responder's maximin value, how do you separate "the library binds" from "the PPO oracle at this budget is not a best responder"? Would a longer or tuned run over the six-tactic library be a cheaper test than the geometry sweep?
3. What is the range of the LP value when each of the nine Sel3 × Evs seed pairs is placed in the saddle cell, and how often is Evs-1 the column support under resampling of the 16 cells?
4. Would you accept rewriting C3 and the abstract as "for players trained under the filter at δ = 0 and frozen", given §VI's own statement that the sweep does not measure a re-equilibrated value?

---

## Minor Issues
- §IV-C: "zero duality gap" — trivially true for a finite zero-sum matrix game; drop or rename.
- §IV-C: held-out sentence should name Evs-2 as well (Table 3: Sel-3 0.218 vs Sel-2 0.192).
- §IV-D / Figure 4(b): say the sweep's pool differs from Figure 3's (Full for Lead).
- Table 4: print numeric p-values for all rows instead of "n.s.".
- Table 3 vs Figure 3: orientation differs; align or label.
- Figures 3–4: add seed identifiers (Sel-2, Sel-R-1, Evs-1) to labels or captions.
- §I: "a well-defined value" — distinguish the differential game's value from the empirical game's.
- §V: "a known asymmetry" — cite or drop "known".
- References: add DOIs or stable links for Wellman 2006, Lanctot 2017 and Isaacs 1965 (venue checklist).
- Venue: confirm whether IEEEtran's inline "[n]" satisfies the checklist's "superscript Arabic numerals".

---

## Actionable Items (ranked by severity)
1. **MAJOR — Scope the vocabulary.** Where: Abstract (two sentences), §I Q3 and the "well-defined value" sentence, C2–C3, §IV-D "non-exploitable within its library", §V paragraph 3, §VII. Fix: add "of the 4×4 pool" / "within the pool" to every value and minimax statement, "for players trained at δ = 0 and frozen" to every sweep value, replace "non-exploitable within its library" with "no evader in the pool exploits any library pursuer by more than 0.054", name the two value objects once in §I, drop "zero duality gap".
2. **MAJOR — Address the oracle-reliability reading and soften "converges".** Where: §V "the pursuer's library, not its learning, is binding" and "Its limits are the library's limits"; §VII "converges in one evader step". Fix: state that enlarging a strategy set cannot lower an exact best responder's maximin value, so the sixth-tactic degradation indicts the learner, not the library; rewrite §VII as "one pursuer best-response step fails to improve within its library"; add the second evader oracle (5×5 pool) as a run or as the explicitly open half of the convergence test.
3. **MAJOR — Put uncertainty on the equilibrium and disclose pool membership.** Where: §IV-C, Figure 3 caption, §IV-D, Figure 4(b). Fix: value with Wilson interval; resample the cells from (k, n) and report support frequencies and the value's distribution for the main game and each sweep setting; state that the row support is a one-episode tie and the δ = 0.20 switch to the hugger is not; disclose that Sel-R-1 and Evs-1 are best-of-three by pre-stated rules and why Sel-2 is the Sel row; give the value's seed range.
4. **MAJOR — Name the sweep's game.** Where: §IV-D second paragraph, Figure 4(b) caption, Abstract and C3. Fix: "the 4×4 game over {Rules, Sel-2, Sel-R-1, Full}" and "for frozen players"; note the δ = 0 value coincides with Figure 3's because the saddle pair is in both pools.
5. **MINOR — Related work for a Track 2 reader.** Where: §II "Empirical game analysis". Fix: one sentence each on strategy exploration (Jordan, Schvartzman & Wellman 2010) and equilibrium estimation error (Tuyls et al. 2020), and name the alternation as one double-oracle iteration.
6. **MINOR — Held-out sentence.** Where: §IV-C. Fix: report all three evader seeds, including Evs-2's 0.218 (Sel-3) vs 0.192 (Sel-2).
7. **MINOR — Matrix presentation.** Where: Figures 3–4, Tables 3–4. Fix: seed identifiers in labels or captions; align Table 3's orientation with Figure 3; numeric p-values throughout; write the evader-side exploitability formula; optionally mark row minima and column maxima in Figure 3.
8. **MINOR — Citations and venue format.** Where: references.bib, in-text citation style. Fix: DOIs/links for Wellman 2006, Lanctot 2017, Isaacs 1965; confirm the citation style with the venue; cite or drop "known asymmetry" in §V.

---

## Dimension Scores (rubric, 0–100; reported for the synthesis narrative — the editorial decision below follows the contract's failure conditions, not this average)

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 78 | Strong | Two-sided learned selection under a certified filter with a falsifying margin sweep; the "margin changes who is minimax" finding is new to me. |
| Methodological Rigor (25%) | 74 | Adequate/Strong | Pre-registered, paired, intervals on cells; no uncertainty on equilibrium objects; pool selection undisclosed. |
| Evidence Sufficiency (25%) | 70 | Adequate | Key claims supported by tables; the "library binds" claim rests on one oracle step. |
| Argument Coherence (15%) | 70 | Adequate | Clear spine; conclusion one notch over; rival reading unaddressed. |
| Writing Quality (15%) | 80 | Strong | Dense, precise prose; naming inconsistencies between text and figures. |
| Significance & Impact (optional, R3) | 76 | Good | A cheap, adoptable reporting practice for safety-filtered competitive systems; moderate influence potential. |
| **Weighted Average** | **74.1** | **Minor Revision band by the rubric table** | Overridden by the contract: F2 fired (three mandatory warns) → major revision. |

## Editorial Decision
editorial_decision=major_revision
F2 is the highest-severity condition that fired on my scores (D1, D2 and D3 all warn; no block anywhere), and under the contract that outcome is major revision — the required changes are scoping, disclosure and one unaddressed rival inference, all achievable in one revision cycle.
