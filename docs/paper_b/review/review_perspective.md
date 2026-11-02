# Phase 2 — Peer Reviewer 3 (Perspective): paper-visible review

Manuscript: "The Guarantee Holds and the Task Fails: Absorbing Stalls of Learned Policies under CBF Filters" (Shoaib, Siddique), L4DC 2027 draft `docs/paper_b/draft/main.tex`, supplement, bibliography, eight figures, release repository `safe-and-stuck`. Page numbers refer to the compiled PDF (`main_text_for_review.txt`). Contract `reviewer/reviewer_full/v1`. Phase 1 commitment: `phase1_perspective.md` (written before any manuscript file was opened; applied unchanged below, no dissent).

## Dimension Scores

| Dimension | Score | Evidence (one sentence) |
|---|---|---|
| D1 methodology_rigor | pass | The design states what was registered and when, names the test, pairs seeds, reports every failed bar (Exp. 3 recurrent arm, Exp. 7 dwell test, Exp. 8 escape bound, Exp. 9 bar met on two of four), and ships a release repository whose README maps every paper name to a record, commit and seed stream; the natural safe-RL baselines for a memoryless policy (observation features, longer horizon, stall penalty, correction penalty, training through the filter) are all present, and the one absent arm (memory) is outside the stated policy class and its omission is disclosed (p. 6, p. 10). |
| D2 domain_accuracy | pass | The cited shielding and filter-training work is represented accurately where it matters (Pizarro Bejarano et al.'s squared correction penalty and the quadrotor caveat, p. 9; Kim et al.'s realizability requirement, p. 2; Tan and Dimarogonas's program modification, p. 2), and the only looseness I found is the intro's single-sentence collapse of discrete shields into "replaces it with the nearest admissible command" (p. 1), which the paper itself corrects on p. 5 and which changes no result. |
| D3 argumentative_coherence | warn | The abstract ("We measure why learning does not remove it"), the intro ("that is why learning does not escape it", p. 1), Remark 2's title and the conclusion ("measured why the learner cannot see the way out", p. 10) state a general mechanism, while the paper's own evidence shows the discount bound is not sufficient (at γ = 0.999 the bound is 0.137, above the 0.05 bar, yet Hor deadlocks 0.41–0.61) and memory is untested; the Discussion softens this correctly (p. 10), which is exactly my Phase 1 warn pattern, not a block. |
| D4 cross_disciplinary_relevance | warn | Both readerships can follow the framing and the practitioner takeaway is explicit ("keep the filter, add the counter, log the firings, watch the episodes", p. 10), but two adjacent connections that change how a practitioner reads the result are absent (the stall as perceptual aliasing of a memoryless policy in a POMDP; the rotation candidate as the wall-following escape from a potential-field local minimum, already standard as a navigation "recovery behaviour"), and the policy's observation vector, on which the "trap in observation space" claim rests, is never described. |
| D5 writing_and_structure | warn | The figures were composed at roughly full text width and then printed at 0.39–0.72 of it, so the 8-pt base font lands at 3.4–5.4 pt on the page (Figures 1, 4-right and 6 at about 3.4 pt are not readable in print), and internal record jargon leaks into the text ("hold-10 cells", "dense-reward column", "the pins", "gate episodes", "decidable triggers", "exact tier"); every main number is restated in the text, so this is a warn, not a block. |

## Failure Condition Checks

| Condition | Expression | Fired on my scores? |
|---|---|---|
| F1 (severity 90) | any mandatory dimension scores block | No (D1 pass, D2 pass, D3 warn). |
| F2 (severity 70) | two or more mandatory dimensions score warn or worse | No (one mandatory warn: D3). |
| F3 (severity 60) | any high-priority dimension scores block | No (D4 warn). |
| F0 (severity 10) | every mandatory dimension scores pass | No (D3 warn). |

No failure condition fires on this reviewer's scores. The contract therefore does not force Reject, Major Revision or Accept; the decision below is the one consistent with one mandatory warn and two non-mandatory warns, all addressable in a single revision cycle without new experiments.

## Review Body

# Peer Review Report

## Manuscript Information
- **Title**: The Guarantee Holds and the Task Fails: Absorbing Stalls of Learned Policies under CBF Filters
- **Manuscript ID**: n/a (pre-submission, L4DC 2027)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Safe reinforcement learning and shielding researcher, practice-oriented: continuous-space shields, training through the filter, reward shaping for filter interventions. Not a CBF theorist and not the statistics reviewer; I read the propositions for what they mean to someone who trains and deploys filtered policies, not for their formal rigour.

### Review Focus
(1) Does the paper speak to the safe-RL reader (memoryless policies, discounting, observation design, penalty shaping, training through the filter)? (2) Is "learning does not remove it" over-claimed given untested memory and a bounded discount argument? (3) Do the three dissociations belong here and generalise? (4) The recovery versus learned alternatives in practice; naming and figure legibility for a first-time reader; cross-disciplinary connections and adoption.

## Overall Assessment

### Recommendation
- [ ] Accept
- [x] **Minor Revision**
- [ ] Major Revision
- [ ] Reject

### Confidence Score
4. The shielding and safe-RL content is squarely my area; the normal-cone characterisation and the exact-tier filter construction are adjacent and I defer to Reviewer 2 on them.

### Summary Assessment
The paper shows that a minimal-intervention CBF filter in front of a memoryless PPO pursuer produces an absorbing stall (filter zeroes the command, observation freezes, policy re-emits the same command for the rest of the episode) with zero violations, characterises the stall as a normal-cone condition plus eventual observation constancy, measures the geometry on 262 stalled states, and shows across nine pre-registered experiments that observation features, a longer discount horizon, and two penalty schemes improve the task without removing the stall, while a one-counter recovery that offers filtered forward and rotation candidates removes it on every family and composes onto a second filter. From the safe-RL side this is a paper I would want colleagues to read: the failure is one that anyone who has trained behind a shield has probably produced without noticing, the remedies tested are the ones a practitioner would try first, and the two penalty dissociations are a direct warning about the most common fix. The protocol (pre-registration, frozen commits, failed bars reported, release repository with a name crosswalk) is well above the field's norm. The main weaknesses are a mechanism claim ("why learning does not remove it") stated more generally than the evidence allows, two missing adjacent connections (perceptual aliasing; potential-field local minima and the navigation-stack rotate recovery) that would both sharpen the contribution and place the recovery honestly, under-described observation design, and figures that are not readable at their printed widths. All are fixable in one revision without new runs, hence Minor Revision.

## Strengths

### S1: The failure mode is the one safe-RL practitioners actually produce, and it is shown rather than asserted
Figure 1 and the intro paragraph (p. 1: "emits the same full-speed command at each of the next 1 106 steps (standard deviation 0.000) while the executed speed stays at zero") give the reader the phenomenon in one episode before any statistic. The Discussion's admission that "Fifteen specifications passed their bars while measuring the artefact this paper is about" (p. 10) is the strongest argument in the paper for a safe-RL audience: our standard evaluation (violation count plus return) does not see this.

### S2: The remedy set is the practitioner's own list, each run under pre-registration, and the negative results carry equal weight
Observation features (Obs), longer horizon (Hor), both (ObsHor), stall penalty, correction-magnitude penalty in the form of Pizarro Bejarano et al. (2025), a random perturbation baseline, and training through the filter for all families (Section 3, p. 3; Sections 5.2 and 6). The text reports "Every remedy that learns improves the task; only the escape removes the failure" (p. 7) with the per-seed numbers beside it. That the stall penalty halves the stalled-step rate while capture falls from 0.38–0.48 to 0.20–0.35 (p. 9) is the most useful single number in the paper for someone designing a reward.

### S3: Safety and liveness are kept separate in the construction, not only in the claim
Proposition 3(a) (p. 4) makes the recovery execute only outputs of the base filter, so the guarantee is untouched by construction, and the composition onto the predictive filter re-filtered 39 162 commands without a rejection (p. 8). This is how a shielding reader wants a liveness add-on to be built: outside the filter, with no change to the admissible set.

### S4: Reproducibility is unusually concrete
The release README maps every paper name to a repository identifier, frozen commit, seed stream and record path; the figure scripts read only archived records and write the JSON of every number drawn; construct gates refuse to summarise until shared cells reproduce bit for bit. The supplement's crosswalk (Tables 2–3) does the same for the reader of the PDF.

## Weaknesses

### W1: "Why learning does not remove it" is stated as a general mechanism, but the paper's own test shows the stated mechanism is not sufficient
**Problem**: Abstract: "We measure why learning does not remove it: the reward for escaping arrives 629 steps later and is worth 0.009 at γ = 0.99." Intro (p. 1): "the trap is in observation space, and that is why learning does not escape it." Remark 2 is titled "Why learning does not remove it"; Conclusion (p. 10): "measured why the learner cannot see the way out." But Remark 2 itself says the bound at γ = 0.999 is 0.137, well above the 0.05 bar, and the Hor family trained at that γ still deadlocks 0.41–0.61 (Fig. 4, p. 7). So the discount channel, as bounded, predicts that Hor should have had enough signal, and it did not learn the escape. Memory, the other obvious channel, is untested (p. 6, p. 10). The Discussion says this correctly ("Remark 2 is a bound: the longer-horizon family shows the horizon is implicated and lengthening it insufficient", p. 10), but the abstract, intro, Remark title and conclusion say more.
**Why it matters**: A safe-RL reader will take the headline as "learning cannot fix this" and then notice that the paper's own γ = 0.999 arm contradicts the offered explanation. The real reason is almost certainly the one the paper half-states in Remark 2 ("the advantage attaches to one shared input vector"): the stalled observation is aliased with many non-stalled observations, so a deterministic memoryless policy cannot assign a different action to the stall without changing its action everywhere that observation occurs. That is perceptual aliasing, a known property of memoryless policies in POMDPs (Whitehead and Ballard 1991; Singh, Jaakkola and Jordan 1994), and it is a better and more general explanation than the discount. It also predicts exactly what the paper found: Hor cannot fix it (the signal is there but the policy class cannot use it), Obs moves the fixed point without removing it (the features partially disambiguate), and memory is the natural learned remedy, which is why its absence is felt.
**Suggestion**: (i) Rescope the four sentences above to "why a memoryless learner does not remove it" and retitle Remark 2 "A bound on the learning signal". (ii) Add two or three sentences to Section 4 naming the aliasing argument explicitly and citing the POMDP memoryless-policy literature; state as a conjecture that memory is the learned remedy. (iii) Either run a clean recurrent arm (one pre-registered family, three seeds, the protocol already exists) or state plainly in the Discussion that the paper's claims do not extend to recurrent policies and that this is the first thing to test. (i) and (ii) alone would satisfy me.
**Severity**: Major (addressable by text).

### W2: Two adjacent literatures that a roboticist or a navigation engineer would reach for are absent, and one of them places the recovery
**Problem**: The paper frames the stall against the CBF-QP deadlock and undesired-equilibria literature only (Section 2). From the navigation side, a filtered policy stopping against static geometry with an admissible rotation available is the local minimum of an artificial potential field (Koren and Borenstein 1991), and the two classical escapes are a random walk (the paper's perturbation baseline) and wall-following (a rotation until forward motion is admissible, the paper's rotation candidate). In deployed stacks this is a "stuck detector plus rotate recovery" (the ROS navigation stack's recovery behaviours, described in Marder-Eppstein et al. 2010). The paper's p. 8 finding that "The recovery's contribution is the rotation candidate, not the trigger" is the potential-field result re-derived for a learned nominal behind a CBF filter.
**Why it matters**: Without this framing the recovery reads as an ad hoc counter; with it, it reads as the paper making a standard engineering practice precise for learned policies behind certified filters, proving it keeps the guarantee (Prop. 3a), and measuring its cost (cycles, firings). That is a stronger and more honest position, and it pre-empts the reviewer who says "this is just rotate_recovery". It also explains to a safe-RL reader why the random perturbation fails here while it works in the potential-field literature: the baseline accepts only moving commands and the admissible set at a trap is the zero-speed segment (Fig. 3a).
**Suggestion**: One paragraph in Related work ("Local minima and recovery behaviours in navigation") citing Koren and Borenstein (1991) and one navigation-stack reference; one sentence in Section 5.2 after "The recovery's contribution is the rotation candidate" connecting it to wall-following; one sentence in the Discussion acknowledging that practitioners already deploy such recoveries and that the contribution is the guarantee-preserving construction plus the measured cost. References listed below with verification marks.
**Severity**: Major (addressable by text).

### W3: The observation vector is never described, yet the central claim is about observation space
**Problem**: Section 3 says what the filter is in detail (p. 3) but describes the policy's input only as "the filter's admissible-turn features" (Obs), "the filter's state" (Aware), and in R2 (p. 4) "the pose, the opponent's visibility, and bounded filter features whose stall counter saturates at 80 steps". The reader cannot tell whether the obstacle the pursuer is pressed against is in the observation at all, whether the Base policy sees any filter signal, how large the vector is, or what "admissible-turn features" are.
**Why it matters**: For a safe-RL reader observation design is the first lever (the paper's own Obs family shows it matters), and for a control reader "the trap is in observation space" (p. 1) is unverifiable without the space. The aliasing argument in W1 also depends on it: if the obstacle is not observed, a stalled state and a free-moving state at the same relative evader bearing are literally the same input.
**Suggestion**: A five-line "Observation" paragraph in Section 3 (components, dimension, which components the filter supplies to Obs and Aware, what saturates at 80 steps), or a supplement table referenced from Section 3.
**Severity**: Major (addressable by text; the information exists in the repository's `pe2/env.py`).

### W4: Figures are composed for full text width and printed at half of it
**Problem**: Native widths are 453–558 pt (close to the text width) with an 8-pt base font and 5.5–7-pt titles, legends and tick labels (figure scripts, `font.size: 8`). Printed scales: Fig. 1 0.46 (effective base font about 3.7 pt, legend 3.2 pt); Fig. 2 0.54 (4.3 pt); Fig. 3 0.58 (4.7 pt base, panel (b) legend and tick labels about 3.2–3.4 pt); Fig. 4-left 0.67 (5.4 pt, acceptable); Fig. 4-right 0.42 (3.4 pt; the rotated tick labels are unreadable, which is why the layout extraction renders them as "se Ba", "Ob sH or"); Fig. 5 0.61 (4.9 pt base, annotations 3.7 pt, six panels); Fig. 6 0.42 (3.4 pt); Fig. 7 0.51 (4.1 pt; panel (c)'s legend overlaps the y-axis "0.6" tick, visible in the extraction as "0.6law", and panel (a) has no y-axis label).
**Why it matters**: L4DC reviewers read the PDF at page size; text under 5 pt is not read, and the paper's argument that one must "watch the episodes" is undercut by figures that cannot be seen.
**Suggestion**: Regenerate each figure with `figsize` equal to its printed width in inches (0.46 × 6.3 in ≈ 2.9 in for Fig. 1 and Fig. 6; 0.39 × 6.3 ≈ 2.5 in for Fig. 4-right) so the 8-pt font prints at 8 pt, dropping panels where necessary: Fig. 1 can be one panel with both runs overlaid; Fig. 3(b) is nearly information-free (every bar at 1.00 or 0.00) and already duplicated by supplement Table 1, so keep (a) and (c) and print at 0.72 with two panels; Fig. 6 is a 4 × 4 grid of numbers and could be a small table; Fig. 7(c) needs its legend moved and (a) a y-axis label.
**Severity**: Major for D5, Minor for the science.

### W5: Internal record vocabulary leaks into the paper
**Problem**: "hold-10 cells" (p. 6), "8-cell pool" (Fig. 2b), "dense-reward column" (p. 9), "Removing the pins removes a symptom" (p. 10; "pins" is never defined), "gate episodes" (Fig. 4 caption), "decidable triggers" (p. 7), "exact, relaxed and backup tiers" (p. 3; named, not described), "18 arms" and "three penalty schedules" (Table 1, unnamed), "six" opponents in Exp. 1 (Table 1, never listed), "its anchored fine-tune" (Table 1, Exp. 2; never mentioned again), "the same stall was seen in a rule-based pursuer in a separate game" (p. 6; unsupported aside). Fig. 2's panel counts (383 entries in (a), 208 escapes from an 8-cell pool in (b), 772 entries and 97 escapes in the text) are three different denominators with no sentence tying them together.
**Why it matters**: A first-time reader has no access to the records; each of these costs a re-read or a trip to the supplement, and the Exp. 1 and Exp. 2 dangling references look like cut text.
**Suggestion**: Define or replace each term at first use ("hold-10" → "the pre-registered cells with the 10-step search hold"; "pins" → "stalls on the intercept-law branch"; "dense-reward column" → name the schedule); list the six Exp. 1 opponents in a footnote or drop "six"; either use the anchored fine-tune or remove it from Table 1; drop the "separate game" aside or cite it; add one sentence in Section 5.1 explaining the 772 / 383 / 208 denominators.
**Severity**: Minor.

## Detailed Comments

### Title & Abstract
The title is accurate and will be remembered. The abstract is dense but complete; its one over-reach is the "why learning does not remove it" sentence (W1). "Three dissociations show liveness metrics improving while the task collapses" is earned by Section 6.

### Introduction
Motivation is persuasive for the safe-RL reader; the one-episode exhibit is the right opening. "that is why learning does not escape it" (p. 1) should be "why a memoryless learner does not escape it" (W1). "Proposition 1 predicts the oscillations and timeouts these papers report" (p. 2) is a claim about other papers' results that is not demonstrated; "is consistent with" is what the evidence supports.

### Literature Review / Theoretical Framework
Coverage of the CBF-QP deadlock line and the recent L4DC shielding papers is appropriate (Reviewer 2's domain). From my side, two adjacent literatures are missing (W2), and the POMDP memoryless-policy literature would make the central mechanism precise (W1). Within safe RL the paper cites the filter-training line (Oh et al., Wu et al., Pizarro Bejarano et al.) correctly; a one-sentence pointer to recurrent/memory-based policies for partial observability (for example Hausknecht and Stone 2015) would make the "memory untested" limitation legible as a specific next experiment rather than a gap.

### Methodology / Research Design
The authors' questions, answered from my angle:

**(a) Is Proposition 1 a result or a restatement?** The first sentence is a restatement, and the paper says so ("The characterisation is known", p. 4). The second sentence (constant observation → fixed point → absorbing) is close to a tautology for a deterministic memoryless policy. The genuine content is in R1 (pair trap versus position trap) and R2 (eventual constancy after at most 80 stalled steps because the filter features saturate and the opponent is out of view), and in the measurement (Exp. 7). I would restate: label the known characterisation a Fact or Lemma with citation, and promote R1 + R2 to the proposition ("Absorbing pair trap": under a deterministic memoryless policy whose observation is a function of pose, opponent visibility and bounded filter features that saturate within 80 steps, any stall with the opponent out of view is absorbing until the opponent re-enters view, and the stalled position need not be a position trap). That statement has falsifiable content, is what Experiment 7 actually measures (98.1% traps at heading, 0 position traps), and an area chair will not read it as a restatement.

**(b) Does the nine-experiment protocol section earn its space?** Yes. The protocol paragraph (p. 5) is part of the contribution: the "fifteen specifications passed their bars" point in the Discussion (p. 10) only lands because the reader knows what a specification was, and a safe-RL audience is not used to seeing the sentence-to-write-on-failure rule. Table 1 is the only key to the experiment numbers and must stay. It could be tightened: the "Measures" column carries internal counts ("772 stall entries", "18 arms") that do not help, and the "six" and "anchored fine-tune" entries dangle (W5). I would not cut the section; I would spend the saved words from Exp. 6 (see (d)) on figure legibility.

**Training through the filter and the recovery.** All families were trained behind the plain filter and the recovery was added at evaluation. A safe-RL reader will ask whether training through the filter-plus-recovery (so the policy observes the recovery firing and can learn to pre-empt it) changes the firing counts; Fig. 4-right (ObsHor seed 2: 363 firings versus Base seed 2: 152 675) suggests the learned half already does most of this. One sentence in the Discussion naming this as the untested combination would be useful.

### Results / Findings
Results are complete and each number carries its N and p (Reviewer 1's domain). Figure quality is the issue (W4). Alignment with the questions is good; the perturbation-baseline paragraph (p. 8) is the clearest piece of reasoning in the paper and should be kept whole.

**(c) Are Figures 1, 3, 6, 7 legible at their printed size?** No for Figures 1 and 6 (effective font about 3.4–3.7 pt), partly for Figure 3 (the main labels at about 4.7 pt are marginal; panel (b)'s legend and tick labels at about 3.2–3.4 pt are not), and marginally for Figure 7 (about 4.1 pt, with a legend-over-axis collision in panel (c) and no y-axis label in panel (a)). Figure 4-right (3.4 pt) and Figure 2 (4.3 pt) have the same problem. Numbers and fixes in W4.

### Discussion
**(d) Do the three dissociations belong in this paper?** The first two, yes, and for the safe-RL reader they are among the paper's most valuable parts: the stall penalty (Exp. 3) and the correction-magnitude penalty (Exp. 5) are the first two things a practitioner tries, and the paper shows each improves the liveness metric it was aimed at while capture falls and zero-motion requests rise (p. 9). They generalise exactly as far as the paper's own caveat about Pizarro Bejarano et al.'s quadrotor task says: wherever a zero command is admissible and cheap, the penalty's local optimum is to stop asking. I would state that condition as the general boundary ("the dissociation arises whenever the admissible set contains a zero command that the penalty does not charge") rather than leave it inside the comparison with one paper. The third (Exp. 6, planner replacing the intercept law) is an internal control showing the intercept law is not the cause; it is worth one or two sentences in Section 5.1 or the supplement, and its removal from Section 6 would free a third of a page and Fig. 7(c). The "Consequence" paragraph (report task, immobility and stalled steps separately; firings per episode for any recovery) is the right practical recommendation and should stay.

**Practical adoption.** "keep the filter, add the counter, log the firings, watch the episodes" (p. 10) is actionable. Two conditions should be stated for a deployer: (i) the clean separation of safety from liveness assumes that standing still is safe for all time, which holds for static obstacles and a non-aggressive opponent but not for a filtered vehicle stopped in a lane or a robot in a shared corridor; the "guarantee holds" framing is only as good as that assumption; (ii) cycles (8–56% of engagements at N = 20) are oscillation against an obstacle, which in a physical system is a wear, predictability and human-comfort issue; the honest reporting is here (p. 10), but one sentence on when to prefer a larger N or a different candidate order would help. On the recurrent alternative: a stateful learned policy would be a fair competitor to the counter on firings and cycles and is the natural follow-up; the paper should say so rather than leave memory as a bare limitation.

**Options and temporal abstraction.** The recovery is an option in the sense of Sutton, Precup and Singh (1999): initiation set = N consecutive immobile steps, policy = the candidate list, termination = a mobile step. Saying so costs one sentence, places the one-counter construction in a vocabulary the learning audience owns, and makes "learn when to invoke it" (an option-selection policy over {nominal, recover}) an obvious and cheap future experiment that would compete with the counter on firings.

**Exploration.** R3 (p. 4) argues that stochastic sampling does not shorten the stall. A learning reader will connect this to the known result that stochastic memoryless policies can outperform deterministic ones in POMDPs (Singh et al. 1994), and to the "detachment" failure in exploration (Ecoffet et al. 2021, to verify) where a frozen input gives the explorer nothing to return to. One clause citing the first would strengthen R3.

### Conclusion
**(f) Do the claims stay within the evidence?** Mostly yes, and the Discussion is a model of candour. The exceptions are: the "why learning does not remove it" family of sentences (abstract, p. 1, Remark 2 title, p. 10) (W1); "Proposition 1 predicts the oscillations and timeouts these papers report" (p. 2); "the same stall was seen in a rule-based pursuer in a separate game" (p. 6, unsupported); and the abstract's "insensitive to N over 5–80 steps", which is true for capture but, as the text says (p. 7), the deadlock indicator is insensitive by construction for N ≥ 40 and is replaced by longer-run metrics there, so "capture insensitive to N over 5–80, deadlock removed for N ≤ 20 and long runs removed at N ≥ 40" is the accurate form.

### References
Formatting is consistent. Recency is good (five 2025–2026 L4DC and arXiv items). From my side the list is missing the POMDP memoryless-policy line, the potential-field local-minimum line, and the options framework (below).

### Naming
**(e) Is the naming clear to a first-time reader?** Experiments 1–9 with Table 1 as the key: workable, provided the reader keeps Table 1 in view; adding the measured quantity to the first mention of each experiment in the text (which the text mostly does) is enough. Base / Obs / Hor / ObsHor: good, self-explaining. Free / Aware: acceptable once defined (p. 3), but Aware and Obs both observe filter features and differ in training length and feature set; a reader will conflate them, so one clause distinguishing them at the definition ("Aware: the filter's full state, 10 M steps; Obs: its admissible-turn features only, 5 M steps") is needed. G / H / A: used only in Table 1 and spelled out in the text; fine. The real naming problem is the leaked record vocabulary (W5), not the public names.

## Questions for Authors
1. What exactly is in the Base policy's observation vector, and is the obstacle the pursuer is pressed against represented in it? If not, a stalled state and a free state at the same evader bearing are the same input, which is the aliasing argument in W1 and should be the stated mechanism.
2. Given that the γ = 0.999 bound (0.137) exceeds your 0.05 bar yet Hor does not clear the deadlock, what do you now believe prevents Hor from learning the escape, and will you state that in Section 4?
3. Would you run a clean recurrent arm under your protocol (one family, three seeds), or state that the paper's claims stop at memoryless policies and that memory is the first follow-up? Either is acceptable; silence is not.
4. In a deployment where standing still is not safe for all time (moving agents, a lane), does the "guarantee holds" framing survive, and should the paper state static geometry as a boundary condition of the dissociation?

## Minor Issues

### Language / Grammar
- p. 2, Related work: "Proposition 1 predicts the oscillations and timeouts these papers report" → "is consistent with".
- p. 6: "the same stall was seen in a rule-based pursuer in a separate game" → remove or cite.
- p. 10, Exp. 6 heading: "Removing the pins removes a symptom" → define "pins" or rename ("Replacing the intercept law removes its stalls, not the loss").

### Citation Format
- `nakamura2025latent` and `lee2025merrygoround` use "and others" in the author field; L4DC/PMLR style prints full author lists. (Neither appears to be cited in the main text; if unused, drop them from the .bib.)

### Figures and Tables
- Fig. 2: three different denominators across (a), (b) and the text (383 / 208 / 772, 97 escapes); one sentence tying them together.
- Fig. 3(b): near-constant bars; consider replacing with the supplement table or a one-line summary.
- Fig. 4-right: rotated tick labels unreadable at 0.39 width; use three group labels.
- Fig. 5: six panels at 0.68 width; consider (a), (b), (f) in the main text and (c)–(e) in the supplement.
- Fig. 7(a): no y-axis label; (c): legend collides with the y-axis.
- Table 1: "six" opponents and "its anchored fine-tune" are not resolved anywhere in the text.

### Layout
- Fig. 1 and Fig. 2 share p. 5 with only five lines of text; regenerating Fig. 1 at its printed width would recover space.
- Placeholders "[to confirm]" (email) and "[repository URL]" remain.

## Cross-Disciplinary Reading Recommendations
Verification marks: "confident" = I am confident of authors, title, venue and year; "to verify" = check the identifier before citing.
- S. P. Singh, T. Jaakkola, M. I. Jordan, "Learning Without State-Estimation in Partially Observable Markovian Decision Processes", Proc. 11th Int. Conf. on Machine Learning (ICML), 1994, pp. 284–292 (confident; no DOI). Memoryless policies in POMDPs, aliasing, and why stochastic memoryless policies can beat deterministic ones; the vocabulary for W1 and R3.
- S. D. Whitehead, D. H. Ballard, "Learning to Perceive and Act by Trial and Error", Machine Learning 7, 1991, pp. 45–83, DOI 10.1007/BF00058926 (to verify DOI). The original perceptual-aliasing paper.
- Y. Koren, J. Borenstein, "Potential Field Methods and Their Inherent Limitations for Mobile Robot Navigation", Proc. IEEE ICRA 1991, pp. 1398–1404, DOI 10.1109/ROBOT.1991.131810 (to verify DOI). Local minima of potential-field controllers against static geometry and the classical escapes.
- E. Marder-Eppstein, E. Berger, T. Foote, B. Gerkey, K. Konolige, "The Office Marathon: Robust Navigation in an Indoor Office Environment", Proc. IEEE ICRA 2010, DOI 10.1109/ROBOT.2010.5509725 (to verify DOI). The navigation stack whose recovery behaviours (clear, rotate) are the deployed form of the paper's counter.
- R. S. Sutton, D. Precup, S. Singh, "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning", Artificial Intelligence 112(1–2), 1999, pp. 181–211, DOI 10.1016/S0004-3702(99)00052-1 (confident). The option framing of the recovery and the "learn when to invoke it" follow-up.
- M. Hausknecht, P. Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs", AAAI Fall Symposium 2015, arXiv:1507.06527 (confident). The standard pointer for the memory arm the paper did not get to run.
- N. Jansen, B. Könighofer, S. Junges, A. Serban, R. Bloem, "Safe Reinforcement Learning Using Probabilistic Shields", CONCUR 2020, LIPIcs 171, 3:1–3:16, DOI 10.4230/LIPIcs.CONCUR.2020.3 (to verify). Formal shielding where the shield is required to leave the agent a way to make progress; complements the Georgescu et al. GR(1) citation on liveness.

## Editorial Decision

**Minor Revision.** No contract failure condition fires on this reviewer's scores (D1 pass, D2 pass, D3 warn, D4 warn, D5 warn). The one mandatory warn (D3) is a scoping problem in four sentences and one Remark title, not a defect in the evidence; the D4 and D5 warns are a missing paragraph of adjacent literature, an undescribed observation vector, and figures regenerated at the wrong width. All can be done in one cycle without new runs. If the authors prefer to answer W1 with a clean recurrent arm rather than with wording, that is welcome but not required for acceptance from my side.
