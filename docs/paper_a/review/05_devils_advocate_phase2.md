# Devil's Advocate Reviewer, Phase 2 (paper-visible)

Contract `reviewer/reviewer_full/v1`; Phase 1 pre-commitment adopted as my own; manuscript `docs/paper_a/draft/main.tex` / `main.pdf` (8 pp). Evidence read: `pe2/{obs,sensing,tactics,evader_selector_mdp,selector_mdp,shield,env,scripted}.py`, `results/derisk/{s24_1,s24_3zs,s25}_round.json`, `docs/paper_a/numbers_extract_2026-11-02.txt`, the S24.1, S24-FLAT, S24.2 and S25 pre-registrations, and D151–D158. Nothing under `docs/paper_a/review/` was read.

## Dimension Scores

### D1: methodology_rigor
score: warn

This matches two Phase 1 warn triggers: "payoff matrix reported with episode counts but without resampling or sensitivity to pool composition", and "reproducibility partial (code promised rather than available …; environment details recoverable only by inference)".

- **No uncertainty on game values.** Sec. IV-C ("Meta-game") and Sec. IV-D give game values to three decimals (0.184, 0.138, 0.136, 0.162, 0.058) with no interval.
- **The saddle's pursuer is not identified.** One episode in 500 decides it (Sel-2 0.136 vs Sel-R-1 0.138, Fig. 3). It would change if the already-measured third retrained seed, which captures 0.140 against Evs-1, were in the pool.
- **Seed selection is not stated.** The manuscript does not say that Sel-R-1 and Evs-1 are best-of-three seeds picked on the gate stream.
- **Reproducibility is partial.** Code and records are promised "at camera-ready" (footnote 1; Acknowledgment). The contact model, which defines what the "no filter" control means, appears nowhere (position projection; `pe2/env.py` 8–10, 98–122).

Why this is not a block:
- Every learned arm has three seeds, with per-seed values and Wilson intervals (Table 1, Fig. 1).
- The final checkpoint is always the one evaluated (Sec. III-E).
- The baselines share observation, filter and budget, and the paper documents that no arm was tuned ("Hyper-parameters were not tuned for any arm").
- The meta-game includes an exploitability computation and one added best response (Sel-R).

### D2: domain_accuracy
score: warn

This matches the Phase 1 warn trigger "the correct caveat exists in the method section but the abstract or conclusion drops it".

- **The pool caveat is dropped in the Conclusion.** Sec. VI says "'minimax' is relative to that pool, not to the full game". Sec. VII says the game "converges in one evader step to an obstacle-orbit equilibrium that the pursuer cannot escape within its library". No evader best response to Sel-R was ever trained, so nothing tests "converges".
- **The abstract over-states Proposition 1.** "From any contact-free start" holds only at δ = 0. At δ > 0 the proposition needs a start in H at margin δ, but spawns only guarantee body clearance (`_sample_free_pose`).
- **Classical positioning is missing.** "A known asymmetry of pursuit–evasion among obstacles" (Sec. V) has no citation. The orbit outcome is never set against the classical faster-evader analysis for the stated kinematics. This is weak positioning rather than a false characterisation.

Why this is not a block:
- Proposition 1 is correct as stated and matches the implementation.
- The empirical-game caveat does exist in the manuscript.
- I found no cited work that is demonstrably mischaracterised.

### D3: argumentative_coherence
score: block

This matches the Phase 1 block trigger "a data–conclusion mismatch where a table or figure contradicts the sentence citing it on the central claim".

**The sentence.** Sec. V, "What the hierarchy buys": "The ablations locate the gain: … it is the selector's choices rather than its library alone (it departs from the rule table over the same library on every opponent and finds Ambush)".

**The contradiction.** The ablation that separates choice from library is the Rules row of Table 1. On the two held-out evaders it contradicts the sentence:

| | Greedy | Hug |
|---|---|---|
| Rules | 0.582 | 0.378 |
| Sel (seeds 1/2/3) | 0.606 / 0.620 / 0.528 | 0.382 / 0.394 / 0.360 |

- The mean held-out margin over Rules is +0.002.
- One cell is significantly better and one is significantly worse.

**The comparison was registered and not reported.**
- The authors pre-registered this comparison as H-TAC-1b ("reported, not gated"). It failed on 2 of 3 seeds, and the manuscript does not report it.
- The Introduction says failures "are reported as such".
- The Discussion counts "three" failed hypotheses, where the round records show at least eight.

**Which trigger applies.** Rival (b), a hand-written component credited to learning, was tested by the authors, lost, and went unreported. That is the mismatch trigger, not the no-ablation trigger.

**The other two rivals are warn-level on their own.** Rival (a), information asymmetry, and rival (c), game conclusions from a small pool, are acknowledged in Sec. VI. However, the abstract and the conclusion do not carry those caveats.

### D4: cross_disciplinary_relevance
score: warn

This matches the Phase 1 warn trigger "the necessary objects are present but scattered or implicit".

- **The contact model is never stated.** A control reader cannot learn from the manuscript that contact is resolved by position projection: the body is pushed back to the surface and slides, with no penalty and no termination. This is why the "no filter" control in Sec. IV-D has almost the same reachable set as δ = 0. The only hint in the text is "it presses into the obstacle body".
- **The evader's observation is described incompletely.** Sec. III-D says the evader observes "true pursuer range and bearing, no belief, no filter block". It does not say that the evader also observes:
  - the pursuer's own visibility flag;
  - the pursuer's motion features, computed from the true state (`evader_selector_mdp.py`, reading 5 and lines 442–461).

  C2 rests on this information structure.

Why this is not a block:
- The filter, dynamics, payoff, solution concept and RL set-up can all be reconstructed from the text.
- Both seats are learned, through alternating best responses. The two-player framing is therefore not one learner presented against scripts.

### D5: writing_and_structure
score: warn

This matches the Phase 1 warn trigger "abstract headline numbers that are a favourable subset (best seed, best opponent)".

- **The abstract drops the seed qualifier.** It says the selector "beats the best fixed intercept law on held-out evaders". C1 and Sec. IV-A say "two of three seeds".
- **Table 2 shows the best seed and omits Rules.** It shows only seed 2, the best seed, and omits the Rules row, although Rules was evaluated at every ratio.
- **The Figure 1 caption names only the wins.** It lists only the comparisons the selector wins. The "Hand-written rules" row it plots sits level with the selector on both held-out panels.
- **Figure 3 has no intervals.**

Why this is not a block:
- The paper is eight pages, inside the 4–12-page bound, in IEEE two-column format.
- Every headline comparison has a table or figure.
- The labels map to the arms.

## Failure Condition Checks

### F1
fired: true. D3 (mandatory) scores block.

### F2
fired: true. Three mandatory dimensions score warn or worse: D1 warn, D2 warn, D3 block.

### F3
fired: false. The only high-priority dimension, D4, scores warn.

### F0
fired: false. D1, D2 and D3 do not all pass.

## Review Body

### Strongest Counter-Argument

**What the paper does well.** It uses paired 500-episode cells on a fixed seed stream, reports per-seed values, and watches construct gates before reading any rate. It states a correct and honestly scoped invariance proposition. It also publishes a refutation of the authors' own favoured reading of the margin sweep.

**The counter-argument.** The positive results belong to the hand-written library, the kinematics and the information structure, not to learned selection. On the two held-out evaders the authors registered a comparison of the learned selector with a hand-written rule table over the same five tactics (H-TAC-1b). It failed on two of three seeds: the mean held-out margin is +0.002, and seed 3 is significantly worse (−0.054, p = 0.014). The manuscript never reports it, and the Discussion asserts the opposite: "the selector's choices rather than its library alone". Every headline win is over comparators the rule table also clears: Lead off the training opponent, flat PPO, Struct, and transfer across speed ratios (Table 2 omits Rules, which matches the selector at ρ = 0.8). The learned gain that survives, +0.09–0.12, is on the training opponent. There the selector's signature move is Ambush, the tactic whose facing step reads the true evader pose.

The evader wins because it is 20 % faster, unfiltered, sees the pursuer's true pose and whether the pursuer sees it, and runs a hand-written antipodal orbit. With the paper's parameters, its angular speed on the hug circle, 1.2/(r + 0.20), exceeds the pursuer's best about the obstacle, 1/(r + 0.15), for every radius above 0.1 m. No single-pursuer library can therefore close the orbit. Against the scripted hugger, captures do not move from ρ = 1.0 to 1.2, which points at occlusion. "The library is binding" names neither cause. The margin sweep refutes a hypothesis that was weak a priori: contact is resolved by projection, so at δ = 0 the filter's hard rows sit on the physical boundary. The remaining value is then credited to speed ratio, geometry and libraries, none of which was varied. Information, the asymmetry the authors flag, is left off the list.

### Thesis, Inferential Links and Rivals

**Thesis (one sentence).** Each player learns a selector over hand-written closed-loop tactics, and this is a workable way to put RL behind a CBF filter in obstacle pursuit–evasion: the pursuer's selector beats fixed laws and flat or structured learners and transfers across ratios; the evader's selector finds an obstacle orbit that pins the game value at about 0.14; and the filter accounts for only a small share of that value.

| # | Link (claim ⇐ evidence) | Strongest rival consistent with all reported data | Tested / acknowledged / neither |
|---|---|---|---|
| L1 | The pursuer's held-out gain comes from learned selection (Table 1, vs Lead) | The hand-written library carries the gain: Rules, over the same tactics, matches Sel on held-out evaders (mean margin +0.002) | Tested by the authors (H-TAC-1b FAIL), not reported, and contradicted in Sec. V. Block-level. |
| L2 | "The hierarchy learns where flat RL collapses" (Fig. 1, 9/9 cells) | Hand-written closed-loop controllers beat sparse-reward PPO at default settings; both scripted pursuers also beat the flat arms on 17/18 cells | Consistent with the paper's own "mostly the architecture"; untuned baselines acknowledged. Warn if credited to learning. |
| L3 | Zero-shot transfer across ρ (Table 2) | The tactics never read ρ and the selector's ratio input is inert (Sec. IV-B); Rules transfers identically (0.926 / 0.842 / 0.998 at ρ = 0.8) | Tested (Rules was evaluated) but omitted from Table 2. Part of K1. |
| L4 | Learned evader selection works (Table 3) | Speed advantage + full information + the hand-written antipodal orbit | The fixed-selection comparators (Hug = always-orbit; Tactic = rule table) were tested, and the learning gain is real. The information asymmetry is acknowledged only in Sec. VI. Warn. |
| L5 | The library is binding; the game "converges" (Sel-R, Fig. 3) | A kinematic bound at ρ = 1.2 plus occlusion bind any library; column dominance makes the saddle automatic; only one best-response iteration was run | Pool caveat acknowledged. Kinematic rival and convergence: neither. Major. |
| L6 | The filter's share of the value is small (Table 4, Fig. 4) | Projection contact makes δ = 0 ≈ no filter; the players are frozen at their δ = 0 training | Frozen players acknowledged. Contact model: neither. Major. |
| L7 | The filter is safe (Prop. 1, zero contacts) | None for the stated scope; it is sound at δ = 0 for the kinematic model | Scope over-stated only in the abstract. Minor. |

### Linchpin Test

The title, "Learned … for Both Players", rests on two linchpins.

- **Pursuer side (Table 1): fails as a test of learning.** Delete the in-distribution column and the learned selector's advantage over hand-written selection on the same library disappears. Of the six held-out cells, one is better, four are not significant and one is worse; the mean is +0.002. The linchpin survives as evidence for the library, not for learning.
- **Evader side (Table 3 with the Hug and Tactic comparators): holds.** Against the same pursuers, capture rates are 0.086–0.218 for the learned evader, 0.328–0.394 for the always-orbit Hug and 0.528–0.782 for the rule-table Tactic. This holds on a construction that is asymmetric in information, speed and constraint.
- **C3 (Table 4): sound as a measurement only.** It is a valid frozen-policy measurement. Reading it as "the filter's cost" depends on the contact model, which the paper does not state.

"Learned for both players" is therefore currently supported for one player.

### Cross-Section Consistency

The numbers agree with the archived records everywhere I checked: Tables 1–4, the abstract, and the meta-game values and exploitabilities. The scope does not agree across sections:

| Earlier claim | Later or other section |
|---|---|
| Abstract: beats the law "on held-out evaders" | Sec. IV-A / C1: on two of three seeds |
| Abstract: "the same construction on the evader side" | Sec. III-D: a different observation and no filter; Sec. VI: the asymmetry favours the evader |
| Sec. IV-D: "on geometry alone"; Sec. V: "speed ratio, obstacle geometry and the libraries set most of the value" | Sec. VI: one speed ratio; geometry not varied; information asymmetry |
| Sec. VII: "converges … equilibrium" | Sec. VI: "minimax" is relative to the pool |
| Sec. VII: "does not survive a direct test" | Sec. VI: frozen players, "not the value of the game with players retrained" |
| Sec. I: "the failures are reported as such"; Sec. V: "Three … failed" | Round records: at least eight failed registered verdicts; H-TAC-1b absent |

### "So What?" Test

Suppose every claim holds. What does each audience learn?

- **RL readers:** options over hand-written controllers beat sparse-reward flat PPO under a filter (the motivation of Sutton et al. 1999, which the paper cites). Learned selection adds an in-distribution gain over hand-written selection that did not transfer.
- **Game-theory readers:** a faster, omniscient evader orbits an obstacle.
- **Control readers:** a zero-margin filter in a projection-contact simulator costs little, and a large margin can flip the minimax evader. The paper sensibly recommends reporting the margin and the unfiltered value together.

The most transferable contribution is methodological: pre-registered, watched evaluation of a learned game. The H-TAC-1b omission undercuts exactly that. Re-scoped as "what the library, the filter, speed and information each contribute to a learned pursuit–evasion game", the paper passes the test. As "learned tactic selection for both players", it does not yet.

### Issue List

#### CRITICAL

**K1. The held-out gain belongs to the library, the registered test showing this is missing, and the Discussion asserts the reverse.**

- **Dimension:** D3 (primary); D1, D5.
- **Location:** Abstract, sentence 3; Sec. I, Q1 and C1; Sec. IV-A ¶1 and Table 1; Fig. 1 caption; Sec. IV-B and Table 2; Sec. V, "What the hierarchy buys" and "Construct discipline"; Sec. VII.

**What is wrong.**

1. **No learned advantage on held-out evaders.** Sel vs Rules on Greedy and Hug, both held out:

   | Seed | Greedy Δ (p) | Hug Δ (p) |
   |---|---|---|
   | 1 | +0.024 (0.26) | +0.004 (0.90) |
   | 2 | +0.038 (0.048) | +0.016 (0.37) |
   | 3 | −0.054 (0.014) | −0.018 (0.34) |

   The mean is +0.002. This comparison is the pre-registered H-TAC-1b (`tactics_prereg_s24_1_2026-10-13.md` §4, "reported, not gated"). The round record (`results/derisk/s24_1_round.json` → `verdicts.h_tac_1b`) gives FAIL, 1/3 seeds. The manuscript never reports it.

2. **The Discussion offers a behavioural difference as an outcome attribution.** It credits the gain to "the selector's choices rather than its library alone" on the strength of a χ² difference in tactic profiles and the Ambush share. That is not an outcome test. Off the training opponent, the outcome test contradicts it.

3. **The other headline wins do not separate learning from the library either.** Rules also does all of the following:
   - clears Lead off the training opponent (+0.024 and +0.050);
   - clears the flat and Struct arms (H-FLAT-3: the flat arms fall below both scripted pursuers on 17/18 cells, and Struct falls below Lead on Greedy and Tactic);
   - transfers across ρ (at ρ = 0.8, Rules scores 0.926 / 0.842 / 0.998 against the selector's 0.890–0.938 / 0.842–0.872 / 0.992–1.000).

   Yet Table 2 omits Rules and shows only the best seed.

4. **The count of failed hypotheses is wrong.** Sec. V says "three of the pre-registered hypotheses … failed". The records show failures of H-TAC-1b, H-ZS-2, H-ZS-3, H-PR-1, H-ALT-1, H-ALT-2, H-VAL-1 and H-VAL-2, plus the S24-OS refutation.
   - H-ZS-2 and H-ALT-1/2 appear in substance but are not labelled as failures.
   - H-ZS-3 (the chase share moved opposite to the prediction) and H-TAC-1b do not appear at all.

**Fix.**
- Report H-TAC-1b in Sec. IV-A with the per-seed deltas.
- Add Rules, for all seeds, to Table 2.
- In the abstract, C1 and Sec. VII, state that the learned selector matches the hand-written rule table over the same library on held-out evaders, and exceeds it (+0.09–0.12) only on its training opponent.
- Rewrite the attribution in Sec. V.
- List every failed registered verdict.

**Experiments that would settle it.**
- (a) Train Sel on {Greedy, Hug}, gate it on Tactic (and the reverse), and compare against Rules. This shows whether learned selection generalises anywhere.
- (b) Add a uniform-random masked selector over the same library. This bounds what any non-degenerate selection can achieve.

**Field-norm boundary.**
- The logical contradiction needs no external norm: it lies between Table 1 and Sec. V.
- On the omission, the paper adopts pre-registration as its own standard (Sec. I: "the failures are reported as such"). The accepted boundary for pre-registered work is that every registered analysis is reported and every deviation flagged (Nosek et al., *PNAS* 115:2600, 2018). Omitting pre-specified outcomes is selective outcome reporting (Chan et al., *JAMA* 291:2457, 2004).

**Evidence-crossing rationale.** H-TAC-1b was registered as "reported". The record gives FAIL. The manuscript is silent and its Discussion states the reverse attribution. This is not a generic demand for more baselines; it is the omission of the one registered comparison that bears on the word "Learned" in the title.

#### MAJOR

**M1. The information asymmetry is acknowledged, but the headline presents the evader's construction as a mirror, and the attribution of the value leaves information out.**

- **Dimension:** D3, D4.
- **Location:** Abstract ("The same construction on the evader side"); C2 ("The mirror construction"); Sec. III-D; Sec. IV-D ("beats the unfiltered pursuers on geometry alone"); Sec. V ("speed ratio, obstacle geometry and the libraries set most of the value"); Sec. VI.

**What is wrong.**
- **The two seats differ on three axes.**
  - The evader is 20 % faster and unfiltered.
  - It observes the pursuer's true pose, the pursuer's own visibility flag (`hidden = obs_p[6] <= 0.5`) and the pursuer's motion features computed from the true state. Every evader tactic also reads the true state.
  - The pursuer has a 90° / 8 m sensor with occlusion.
- **The S25 test of information failed as a construct.** The S25 pre-registration made "the exploit does not depend on the pursuer's partial observability" its question (b). Its instrument (FULLOBS) failed as a construct, as Sec. IV-D itself says. Information therefore remains an untested rival.
- **The headline text ignores this.** The Results speak of "geometry alone". The Discussion assigns the value to three factors, none of which was varied (Sec. VI: one ratio; "the sweep varied the filter, not the geometry"). It leaves out the one asymmetry the paper itself flags.
- **Table 2 hints at it.** Against Hug, every pursuer's capture is flat from ρ = 1.0 to 1.2 (0.35–0.38 → 0.33–0.39). Speed is not what binds there.

**Fix.**
- Describe the evader's observation fully.
- Replace "same/mirror construction" in the abstract and C2 with an explicit sentence on the asymmetry.
- Drop "on geometry alone".
- Add information to the Discussion's list as an untested factor.

**Experiments that would settle it.**
- Re-run S24.2 stage 1 with an evader selector whose observation passes through the same FOV/occlusion model plus dead reckoning (3 seeds).
- Run a pursuer selector over the same library that sees the true evader pose, with Counter-Orbit unmasked.

**Field-norm boundary:** none needed. This is attribution by incomplete elimination, an internal logic defect.

**M2. The orbit's success comes from a kinematic bound, so "cannot answer within its library" and "the library is binding" misattribute it.**

- **Dimension:** D3, D2.
- **Location:** Sec. IV-C (heading "The pursuer cannot answer within its library"); Sec. V ("the meta-game confirms that the pursuer's library, not its learning, is binding"; "a known asymmetry"); Sec. VII.

**What is wrong.**
- **The evader out-rotates the pursuer on every obstacle.** Using the paper's own parameters:
  - The evader, on its hug circle of radius r + 0.20 at 1.2 m/s, has angular speed 1.2/(r + 0.20) about the obstacle.
  - The pursuer's best, hugging at r + 0.15, is 1/(r + 0.15). This holds at δ = 0, and also unfiltered, because contact is projected to the same surface.
  - The evader is faster in angle for every r > 0.10 m, and all obstacles have r ∈ [0.3, 1.0].
  - Its turning radius (1.2/2.84 = 0.42 m) fits every hug circle (≥ 0.50 m).
- **No single-pursuer library can catch a steady orbit.** An evader that holds the antipode with full information is uncatchable while it orbits, and that is exactly what the hand-written Orbit/Hug law does. The 0.09–0.22 captures therefore measure the evader's departures from the orbit.
- **"Every pursuer within 0.05 of the value" cannot single out the library.** It is what such a bound predicts for any library.
- **The "cannot" rests on one retraining protocol.** Evs-1 was a quarter of the training mixture, with 3 M steps and default hyper-parameters.

**Fix.**
- State the kinematic argument, or refute it using the unicycle constraints.
- Cite the classical result behind "a known asymmetry".
- Report capture conditioned on whether the evader was in an Orbit span.
- Reword "cannot" to "did not, under this retraining protocol".

**Experiments that would settle it.**
- A two-sided round at ρ = 1.0, where the pursuer out-rotates the evader on every obstacle. The bound needs ρ > (r + 0.20)/(r + 0.15), which is about 1.04–1.11 over the obstacle range.
- A dedicated best-response pursuer trained on Evs-1 alone with a larger budget.

**Field-norm boundary:** none needed. The argument is derived from the manuscript's own parameters.

**M3. The margin sweep's "direct test" is a frozen-policy sensitivity analysis in a simulator whose contact model is never stated, and at δ = 0 the refuted hypothesis was weak a priori.**

- **Dimension:** D3, D4, D1.
- **Location:** Abstract ("raises the value only from 0.138 to 0.184"); C3; Sec. IV-D; Sec. V ("The filter's share is small"); Sec. VII ("does not survive a direct test").

**What is wrong.**
- **"No filter" and δ = 0 share a boundary.** Contact is resolved by projection (`pe2/env.py` 8–10, 98–122). An unfiltered pursuer that touches an obstacle is pushed back to the surface at r + 0.15 and slides along it, which is exactly the δ = 0 hard-row boundary. The two conditions differ only in the filter's approach-rate conservativeness (γ = 0.2) and its soft rows. A small difference is the expected outcome, and it says little about filters whose margin or contact consequences matter.
- **The players are frozen at their δ = 0 training.** Sec. VI concedes that the unfiltered value "may understate what an unfiltered pursuer could learn", yet Sec. VII calls this "a direct test".
- **The δ = 0.20 result may be a frozen-policy artefact.** At δ = 0.20 the pursuers receive filter-feature values they never saw in training. The switch of the minimax evader to Hug may therefore be a failure of frozen pursuers rather than a property of the margin.

**Fix.**
- State the contact model in Sec. III-A.
- Call S25 a frozen-policy sensitivity analysis.
- Scope C3 to "for fixed players, in a simulator where contact is resolved by projection".
- Drop "only" and "direct".

**Experiments that would settle it.**
- Retrain both seats at NONE and at δ = 0.20 (already named as future work).
- Add an unfiltered control in which contact ends the episode or is penalised.

**Field-norm boundary:** none needed.

**M4. The meta-game's game-theoretic vocabulary over-reads a dominant column.**

- **Dimension:** D2, D1.
- **Location:** Abstract ("pure saddle point at value 0.138 with pursuer exploitability at most 0.052"); C2; Sec. IV-C, "Meta-game"; Fig. 3; Sec. IV-D ("Every setting has a pure saddle point"); Sec. VII ("converges in one evader step to an obstacle-orbit equilibrium").

**What is wrong.**
- **The saddle is automatic.** Evs-1's column lies entrywise below every other column of Fig. 3. A pure saddle is therefore guaranteed, and the "value" is simply that column's maximum.
- **The exploitabilities are just the spread of that column.** The pursuer exploitabilities (≤ 0.052) are about one Wilson half-width.
- **The saddle's pursuer is not identified:**
  - One episode in 500 decides it (Sel-2 0.136 vs Sel-R-1 0.138).
  - It flips to Sel-2 at δ = 0.10 (`s25_round.json`).
  - It would flip to Sel-R-3 (0.140 against Evs-1, already measured) if the other retrained seeds were in the pool.
- **The value is biased upward and has no interval.** Taking the maximum over noisy cells inflates it.
- **"Converges" has no test behind it.** It rests on the S24.2 pre-registered reading "a pure equilibrium in the learned pair means the alternation converged" (§4). No evader best response to Sel-R was trained.

**Fix.**
- Report the full all-seed matrix (3 Sel, 3 Sel-R and 3 Evs, plus the scripted policies).
- Bootstrap the value over episodes and seeds and report it as an interval.
- Describe the result as "the learned evader dominates the pool".
- Drop "converges" unless an evader trained against Sel-R fails to improve on Evs-1.

**Field-norm boundary.**
- Empirical game-theoretic analysis treats payoff entries as estimates whose sampling error must qualify any equilibrium claim (Wellman, AAAI 2006; Tuyls et al., *Auton. Agents Multi-Agent Syst.* 34:7, 2020).
- The PSRO view the paper cites (Lanctot et al. 2017) measures convergence by the absence of an improving best response.

**Evidence-crossing rationale.**
- The values are given to three decimals with no interval.
- The identity of the saddle's pursuer changes with a one-episode difference and with an already-measured added policy.
- Convergence is asserted with zero further best-response iterations.

**M5. The selector's showcase of "choice" runs through a tactic that reads the true evader pose.**

- **Dimension:** D3.
- **Location:** Sec. IV-A ("spends 14–19% of its decisions in Ambush, a tactic the rule table never ranks among its top three"); Sec. V ("finds Ambush"); Sec. VI.

**What is wrong.**
- **Two pursuer tactics read the true evader pose.** Ambush's facing step turns the pursuer towards the true evader position while the evader is hidden (`pe2/tactics.py` 702–705). Counter-Orbit's termination check also reads the true pose (615–638).
- **The only robust learned gain is where Ambush is used most.** The gain over the rule table (+0.09–0.12, H-TAC-1b) appears only on the training opponent, where Ambush use is highest.
- **A rival explains that gain as well as "choosing" does.** A selector that learned to route decisions through the privileged channel would produce it.
- **The authors' own notes point to specialisation.** Their watched-episode note records the learned selector over-applying the training opponent's answer against Greedy (S24.1 pre-registration, A4(b)).

**Fix.**
- Connect the Sec. VI limitation to the Ambush finding.
- Re-gate Sel vs Rules on Tactic with a leak-free Ambush that faces the dead-reckoned estimate instead.
- Report capture conditioned on Ambush spans.

**Field-norm boundary:** none needed.

#### MINOR

| # | Dimension | Issue (and fix) | Location |
|---|---|---|---|
| m1 | D5, D3 | The abstract drops "on two of three seeds"; seed 3 is −0.030 (not significant) against Lead on Greedy. Add the qualifier. | Abstract, sentence 3 |
| m2 | D3 | "With zero contacts" / "makes zero contacts" is credited to the selector. Proposition 1 makes it a property of every filtered arm, flat PPO included. Move it to C4. | Abstract; C1; Sec. VII |
| m3 | D1, D2 | "101 000 filtered episodes" counts the 12 δ = 0 cells of S25, which are bit-for-bit replays of earlier records (6 000 episodes). All 202 cells also share the same 500 initial conditions. Report distinct episodes and distinct initial conditions. | Abstract; C4; Sec. III-B |
| m4 | D2 | "From any contact-free start" holds only at δ = 0. At δ > 0 the proposition needs a start in H_δ, and spawns guarantee only body clearance. Say "from any start outside the margin". | Abstract; Prop. 1 |
| m5 | D2, D4 | The proof's key step, f(x, 0) = x, is the instant stop of a zero-inertia kinematic model. State that the argument does not carry over to second-order dynamics. | Sec. III-B, proof |
| m6 | D3 | χ² "p ≈ 0" over about 9 500 decisions per cell is overpowered and says nothing about magnitude or relevance to outcomes. Report an effect size, such as the total-variation distance between tactic profiles. | Sec. IV-A |
| m7 | D3 | Two standards are applied. A +0.046 change in value (+33 % relative) is called "only" or "small", while +0.048–0.066 over Lead is a headline win. Use one standard. | Abstract; C3; Sec. V |
| m8 | D2 | "Zero duality gap" is a property of an exactly solved LP, not a finding. | Sec. IV-C |
| m9 | D1 | Sel-R-1 and Evs-1 are best-of-three seeds chosen by pre-stated rules on the gate stream (S24.2 pre-registration, A2.4 and A5.2). Say so. | Sec. IV-C; Fig. 3 |
| m10 | D5 | "The paired same-seed drop is −0.62, −0.59 and −0.63" follows a list of Lead, Rules and Sel, so it reads as mapping onto them. The figures are actually for the three Sel seeds. | Sec. IV-C ¶1 |
| m11 | D3 | Flat-40 holds an open-loop command for 2 s, so it confounds decision cadence with closed-loop actuation. The conclusion "so the selector's advantage is not its decision rate" should read "… is the closed-loop tactics", that is, the library. | Sec. IV-A ¶2; Sec. V |
| m12 | D2 | The "value" is the ex-ante expected capture over a uniform initial-state distribution for a policy pool. It is not the state-dependent value V(x₀) that the Introduction invokes ("it has a well-defined value"). Name the object being measured. | Sec. I ¶2; Sec. IV-C/D |
| m13 | D5 | Placeholders remain ("Email: [to confirm]"; repository "at camera-ready"). The venue checklist asks for superscript bracketed citations and a Word + PDF submission; the EIC should verify these. | Title block; footnote 1; references |

### Ignored Alternative Explanations

1. **The library carries generalisation (K1).** The hand-written tactics, under any reasonable selection, produce the held-out captures. The authors' own registered test shows this.
2. **Opponent specialisation plus a privileged tactic (M5).** The in-distribution learned gain is the selector specialising to Tactic's hop rhythm through Ambush, whose facing step reads the true pose.
3. **The information structure (M1).** An evader that knows the pursuer's true pose, and whether it is seen, explains several observations at once:
   - flat captures against Hug from ρ = 1.0 to 1.2;
   - the ρ = 0.8 "library gap" (an evader orbiting out of view);
   - the usefulness of Hold-Hidden;
   - the retrained pursuer's shift to Sweep ("it looks for the evader more").
4. **A kinematic bound (M2).** At ρ = 1.2 the antipodal orbit out-rotates any single pursuer, whatever its library.
5. **Projection contact (M3).** At δ = 0 the filter and the physics share one boundary, so a small "filter share" is the expected result.
6. **Distribution shift in frozen pursuers.** Two features of the margin sweep could be out-of-distribution behaviour of policies trained at δ = 0, rather than properties of the margin:
   - the non-significant bump in value at δ = 0.10;
   - the switch of the minimax evader to Hug at δ = 0.20.

   At those settings the pursuers see new filter-feature inputs and different frequencies of the filter's tiers.
7. **Over-powered behaviour tests (m6).** With roughly 10⁴ decisions per cell, "the profile differs" is close to guaranteed by sample size alone.

### Missing Stakeholder Perspectives

- Field-robotics practitioners who deploy CBF filters on platforms with inertia, latency and estimation error.
- Classical differential-game and visibility-based pursuit–evasion theorists.
- Safety-assurance and certification practice: what "certified" may mean for a simulation-only kinematic study.
- Multi-pursuer and team-capture researchers (single-pursuer bounds are the classical reason teams are used).
- People subject to robotic pursuit: the surveillance, security and military dual-use of capture policies.

### Unexamined Premise

The paper assumes that a pre-registered reading is a validated one. Pre-registration fixes the analysis in advance; it does not make an inference fixed in advance valid. Three of the paper's over-readings were registered:

- **"A pure equilibrium in the learned pair means the alternation converged"** (S24.2, §4). This becomes "converges in one evader step".
- **"If any row does not rise significantly, the learned evader's orbit beats that pursuer on geometry alone"** (S25, refutation sentence of H-VAL-1). This becomes "on geometry alone" and silently rules out information.
- **"Its gain comes from choosing"**, operationalised as a behavioural χ² test (S24.1, Hypothesis T and H-TAC-2). This becomes "the selector's choices rather than its library alone".

The manuscript cites registration ("the refutation sentence stated that…", "Every round was pre-registered") as the warrant for readings that each need an independent argument.

### Observations (Non-Defects)

- **The evader-side learning gain is cleanly isolated, and it is the strongest learning result in the paper.**
  - Hug is a fixed always-orbit policy using the same antipodal hug law as the Orbit tactic; Tactic is a rule table over the same five evader tactics.
  - Against the same pursuers, capture is 0.086–0.218 for the learned Evs, 0.328–0.394 for Hug and 0.528–0.782 for Tactic.
  - The paper sells this short relative to C1.
- **Proposition 1 is correct and matches the code.**
  - u = 0 is always a candidate (`_qp_candidates`).
  - The hard row on the Euler successor equals the environment's contact predicate at δ = 0.
  - The recovery passes every candidate through the same filter.
- **The S25 reporting is honest, against the authors' interest.** It reports H-VAL-1/2 as failed and declines the pre-stated "margin-only" reading of H-VAL-3 because FULLOBS was confounded.
- **The comparison design is right.** Paired evaluation on a fixed seed stream with exact discordant-pair tests suits these comparisons. Per-seed values are shown everywhere except Table 2, and every number I spot-checked matches the archived records.
- **Sec. VI is candid.** It names most of the confounds raised here: the frozen sweep, the 4×4 pool, the information asymmetry, the truth-reading tactics and the lack of a geometry sweep. My objection is that the abstract, Discussion and Conclusion do not carry those caveats.
- **The cadence-matched Flat-40 arm was a thoughtful pre-training addition** (S24-FLAT A1.2), even though its interpretation needs narrowing (m11).

## Editorial Decision

editorial_decision=reject_or_major_revision

F1 (severity 90) fires on D3, because the generalisation leg of the title's "Learned" is contradicted by the paper's own Table 1 and by an unreported pre-registered comparison. The data are sound and the defect is one of claim scope and reporting, so I would place this at the major-revision end of the F1 action.
