# Paper B Study Guide

2026-11-03 · Mohammad Shoaib. Updated to **revision 1** of the draft (commit fd3508b, D165, 3 Nov). The first version of this guide (2 Nov, session S135) followed draft 1 and the panel review. The Claude Doc at https://claude.ai/artifact/AezeLtmxguMQHUEAxJsq9G still holds that version and was **not** updated. This markdown file is the current guide. Sources are `docs/paper_b/draft/main.tex` and `supplement.tex`, `docs/paper_b/review/response_round1.md`, `review/revision_facts_2026-11-03.md`, `review/editorial_decision.md` and `docs/decisions.md` (D125 to D165).

**Numbering warning.** Revision 1 renumbered the experiments in reading order and added one. Every "Experiment k" in this guide uses the **new** numbers. The old-to-new map is: old 1, 2, 3, 4, 5, 6, 7, 8, 9 became new 1, 4, 5, 6, 9, 10, 2, 7, 8. The new Experiment 3 is the branch-attribution round N1. Supplement Table S5 is the official crosswalk.

## One-page summary

**Title.** The Guarantee Holds and the Task Fails: Absorbing Stalls of Learning-Enabled Controllers under CBF Filters.

**Venue and dates.** L4DC 2027, 10 PMLR pages plus references, single-blind. Go/no-go is Nov 10. The deadline is Nov 13, 23:59 AoE.

**Status on 3 Nov.** Draft 1 was reviewed by a five-reviewer panel (D163), which returned Major Revision. N1 was run (D164). Revision 1 was written and committed (D165) with a response letter. The next step is a re-review of revision 1.

**The claim in one sentence.** A minimal-intervention CBF filter in front of a learning-enabled pursuit controller creates a fixed point where the guarantee holds, collisions are zero and the task fails, and a one-counter recovery that offers the filter a forward and a turning command removes it without touching the guarantee.

**The controller.** A scripted intercept law chases the evader while it is visible and drives to a remembered last-seen point while it is not. A learned PPO search head commands only when the law has nothing to do, and each head command is held for 10 steps. One head command plus the law steps that follow it is a **decision**, about 16 env steps in training. The law carries one remembered point; the head has no memory.

**The three contributions.**
- **C1, geometry.** On 262 sampled stalled states, 98.1 % are traps of the pair (controller, filter) at the current heading. None is a trap of the position, and a turn is always allowed (Lemma 1, Proposition 2, Experiment 2).
- **C2, ownership and learning.** In the policies trained behind the filter, the scripted law owns 87 % of deadlock episodes, and it stalls inside a single learner decision. In the policy trained without the filter, the learned head owns 76 %. Where the head is stuck, escaping is worth at least 0.18 in discounted return, a signal learning does not use (Experiments 3 and 4).
- **C3, remedies.** Learned remedies raise capture but leave the stall in place. Two penalties improve a stall metric while capture falls. The recovery removes the stall on every policy it was applied to, over its parameter sweep, against a random baseline and on a second filter (Proposition 3, Experiments 5 to 9).

**Headline numbers.**

| What | Number |
| --- | --- |
| The filter's cost on Free | capture 0.728 → 0.396, interval [−0.375, −0.289] |
| Deadlock under the plain filter, Base seeds, against the attacker | 0.298, 0.186, 0.252 |
| Deadlock with the recovery, N ≤ 20, all eleven policies tested | ≤ 0.022 |
| Capture of Free with the recovery | 0.426 → 0.728 |
| Share of deadlock episodes owned by the law, trained families | 1 747 of 2 002 (87 %) |
| Escape signal on the learner's clock | 0.29 (return-to-go), 0.18 (A₀·E[γ^T]) |
| Collisions in every filtered evaluation | 0 |

**What revision 1 changed, in order of importance.**
1. **The discount explanation is withdrawn.** Draft 1 said the escape reward arrives too late to matter at γ = 0.99 (worth 0.009). The learner discounts once per decision, not once per env step, so the signal is about 0.18 to 0.29. The paper now adopts the pre-registered reading: the signal was there and optimisation missed it.
2. **The pursuer is described correctly.** The intercept law has memory and acts out of view. N1 measured who owns each stall, and the paper's learning claim is scoped by that measurement.
3. **The title** says "Learning-Enabled Controllers" instead of "Learned Policies".
4. **The opening figure is a different episode.** Draft 1's Fig. 1 turned out to be a law stall. The new one is a learned-head stall of Free against the greedy evader.
5. **Every registered verdict is stated,** failures included, in the text and in supplement Table S1.
6. **A disclosure.** Every trained family, Base included, uses Aware's recipe: three filter features plus a 10⁻³ stall penalty.
7. **Figures redrawn at printed size,** four main-text figures plus Table 1; the rest moved to the supplement.

## The story in order

The revised paper tells the story in this order.

1. **The failure in one episode (Fig. 1).** Free was trained without a filter and is deployed behind one. It starts facing the top wall and asks to drive forward from step 8. The filter allows only standing still or turning, so it stops the pursuer. The observation does not change, so the head asks again, for the remaining 1 192 steps. With the recovery at N = 20 the same episode fires seven times, the pursuer turns away from the wall and captures at step 507. There are zero collisions in both runs.
2. **Name it.** The filter's output and the controller's frozen input form a fixed point of the closed loop, an **absorbing stall**. It is related to multi-robot CBF deadlock (Grover et al.) and to single-robot undesired equilibria (Reis et al., Tan and Dimarogonas). The new element is what a controller with a learned part contributes.
3. **The closed loop (Fig. 2).** The law commands whenever the evader is visible or the remembered point is not yet reached. The head commands otherwise. Every command passes the CBF filter. The recovery is a counter beside the filter that offers it a short candidate list.
4. **Why the stall is a fixed point (§4).** Lemma 1 is the KKT condition for zero speed in the weighted projection, after Grover et al. Proposition 2 says that if the filter executes zero, the evader stays out of view and the commanding part's internal state is unchanged, then nothing changes except a stall counter that saturates at 80 steps.
5. **The cost (Experiment 1).** The filter takes Free's capture from 0.728 to 0.396. Filter features recover part of it, and a stall penalty alone recovers none.
6. **What is stuck (Experiments 2 and 3).** The geometry says a turn would free the pursuer at almost every stalled state. The ownership round says which part of the controller keeps asking: mostly the law in trained families, mostly the head in Free.
7. **Why learning does not remove it (§5.3).** Law stalls happen inside one decision, so no learned parameter is in their loop. Head stalls carry a signal of at least 0.18 that the learner does not use (Experiment 4). Why the head maps the stalled observation to the blocked command is open. Aliasing is the candidate, and memory is untested.
8. **Learned remedies fail the registered metric (Experiment 5).** A rotation-only recovery clears deadlock by spinning, which is why the paper's recovery offers forward first.
9. **The recovery works (Experiments 6 to 8).** It does so across four families, over a sweep of N, against a random-perturbation baseline and on a second filter.
10. **Stall metrics can improve while the task fails (§6).** This is shown by a tenfold stall penalty (Experiment 5), a correction-magnitude penalty (Experiment 9) and a planner replacing the law (Experiment 10).
11. **Discussion.** Scope, the recovery's costs, how the failure was found (a person watching episodes, after fifteen specifications had passed) and the practical advice: keep the filter, add the counter, log its firings, watch the episodes.

## The setting and the names

One game, a CBF filter plus a second filter, six named policies and three evaders. Every name maps to an internal identifier that the records keep unchanged. Supplement Tables S5 and S6 and the release README carry the crosswalk.

**Game (§3).**

| Element | Value |
| --- | --- |
| Pursuer | unicycle; speed up to 1 m/s, turn rate up to 2.84 rad/s, time step 0.05 s |
| Arena | 10 × 10 m with two disc obstacles, radius 0.3 to 1.0 m, position random per episode |
| Sensing | 90° field of view, 8 m range, line of sight not blocked by an obstacle |
| Capture | centres within 0.30 m |
| Episode | 1 200 steps (60 s); a timeout is a non-capture |
| Evader speed | 1.2 times the pursuer's, stated in §3 since revision 1 |
| Evaders | G greedy flee (GREEDY); H wall-hugging (HUG7); L a learned attacker trained to stall filtered pursuers (ATK4-45). Draft 1 called the attacker A; revision 1 uses L to avoid a clash with A₀ and the admissible set |
| Reporting convention | deadlock is reported against L, the evader that stalls pursuers most; capture is reported against G |

**The pursuer's controller (§3, Fig. 2).** The intercept law drives at full speed toward the evader while it is visible and remembers where it last saw it. While the evader is out of view, the law drives to that remembered point until it is within 0.5 m. Only then does the learned search head choose a command, held for 10 steps or until the evader comes into view. The remembered point updates only on sight of the evader, so it is frozen during a stall. One head command plus the law steps that follow it is a **decision**, one PPO transition, about 16 env steps in training (14.0 to 16.6 per policy). Draft 1 said "a fixed intercept law when it is visible; both are memoryless", which was wrong on both counts (the panel's critical finding).

**Observation and reward (§3, supplement §2).**

| Item | Content |
| --- | --- |
| Base observation | 11 numbers: own position and heading; the evader's relative position and heading, zero when not visible, plus a visibility flag; the two obstacle centres in the pursuer's frame |
| Not observed | obstacle radii; walls, except through the pursuer's own position |
| Filter features | for every policy trained behind the filter: whether the filter changed the last command, the smallest clearance, consecutive stalled steps divided by 40, saturating at 80 steps |
| Obs extra | two more numbers: the range of turn rates the filter would allow at zero speed |
| Reward | +1 capture, −1 timeout; policies trained behind the filter pay 10⁻³ per stalled step |
| Discount | applied once per decision, not once per env step |

**Policies (Table 1).** All share the same intercept law and differ in the learned head.

| Paper name | Identifier | Trained | Filter in training | Extra observation | Discount |
| --- | --- | --- | --- | --- | --- |
| Base, 3 seeds | BASE-60/61/62 | 5 M steps | CBF, stall penalty 10⁻³ | 3 filter features | 0.99 |
| Obs, 3 seeds | OBS-60/61/62 | 5 M | same | + allowed-turn range | 0.99 |
| Hor, 3 seeds | GAM-60/61/62 | 5 M | same | 3 filter features | 0.999 |
| ObsHor, 3 seeds | OBSGAM-60/61/62 | 5 M | same | + allowed-turn range | 0.999 |
| Free | STR14 | 10 M | none | none | 0.99 |
| Aware | RC19 | 10 M | CBF, stall penalty 10⁻³ | 3 filter features | 0.99 |

- **Free** is the best of 5 seeds by gate capture (0.738). It never saw a filter in training.
- **Aware** is the best of the 3 seeds of Experiment 1's features-plus-penalty remedy (0.564, 0.550, 0.444). The paper calls it Base's recipe trained twice as long.
- **The disclosure** added by D165 is that Base is not a plain policy. Every trained family uses Aware's recipe at 5 M steps.

**Other arms and baselines.**

| Name | Identifier | What it is | Experiment |
| --- | --- | --- | --- |
| tenfold-penalty family | PEN-60/61/62 | Base with a stall penalty ten times larger | 5 |
| recurrent family | MEM-60/61/62 | RecurrentPPO; confounded by its rollout length before its number existed (amendment A4); licenses no conclusion | 5 |
| anchored fine-tune of Aware | FT-B-41 | the second policy of the escape-signal analysis | 4 |
| Experiment 1 remedies | RA, RB, RC, seeds 19 to 21 | filter features, stall penalty, both (RC seed 19 is Aware) | 1 |
| rotation-only recovery | REC | after 20 stalled steps asks for the allowed rotation; never offers forward | 5 |
| the paper's recovery | RECF-N (RecoveryShieldV2, recovery_after = N) | N = 20 unless stated | 6, 7, 8 |
| perturbation baseline | PERT-20 | after 20 immobile steps draws up to eight commands from the box, filters each, executes the first that moves | 7 |
| predictive filter | PRED-PLAIN, PRED-RECF-20 | PredictiveShield; composition with the recovery by method resolution order | 8 |
| penalty reward variants | DVARY, SPARSE, DRARE, seeds 70 to 72 | the three reward variants of the correction penalty | 9 |
| line law, planner law | LINE, PLAN | the intercept law and the obstacle-avoiding planner, on a fully scripted pursuer with orbit search | 10 |

**Filters.**

| Element | CBF filter (base) | Predictive filter (Experiment 8) |
| --- | --- | --- |
| Test | exact one-step barrier condition on the next state, barrier rate 0.2, control point 0.1 m ahead of the axle | a four-step rollout in the simulator followed by a stop step makes no contact |
| Executed command | the allowed candidate nearest the request in 150 (Δv)² + (Δω)² | same interface |
| Implementation | a candidate search accepted by the exact test; agrees with the exact projection on 98.1 % of stalled states | same candidate handling |
| Tiers | exact, relaxed, backup; the exact test failed on one step per 500-episode evaluation of Experiment 6; every stalled state of Experiment 2 was on the exact tier; backup never used | backup never reached |
| Conservativeness | contracts the barrier at rate 0.2 | no barrier-rate contraction, so it is the less conservative filter |

Why the filter stops instead of turning: at a trap at heading, the allowed set at the current pose is the zero-speed segment. No turn buys any speed, so the nearest allowed command to "full speed ahead" is a stop with the turn rate left at zero. The minimal-intervention objective never proposes the rotation that would free the heading.

**Stall definitions (§3).**

| Term | Definition |
| --- | --- |
| immobile step | executed speed at most 0.05 v_max |
| stalled step | immobile while the requested speed is at least 0.5 v_max |
| deadlock episode | contains at least 40 consecutive stalled steps (2 s) |
| trap at heading | no allowed command at the current pose moves the pursuer |
| position trap | the trap holds at every heading at that position |
| escape | a displacement of 0.10 m from the stalled pose |
| engagement, cycle | a cluster of recovery firings fewer than five steps apart; an engagement that ends in another stall within 40 steps |

**Experiments and their records.**

| Exp. | Draft 1 | Internal round | Seed stream | Records under results/derisk/ |
| --- | --- | --- | --- | --- |
| 1 | 1 | S4, S5 | 1 090 000+ | s4_cell_*.json, s5_arm_*.json |
| 2 | 7 | S26 B1 | 3 000 000+ | s26_b1_round.json, s26_b1_cells/ |
| 3 | new | S27 (N1) | 1 740 000+ | s27_n1_round.json, s27_n1_cells/ |
| 4 | 2 | S15 | 1 710 000+ | s15_round.json, s15_cells/ |
| 5 | 3 | S16 | 1 720 000+ | s16_round.json, s16_arm_*.json |
| 6 | 4 | S17 | 1 740 000+ | s17_round.json, s17_arm_*.json |
| 7 | 8 | S26 B2 | 1 740 000+ | s26_b2_round.json, s26_b2_cells/ |
| 8 | 9 | S26 B3, B4 | 1 740 000+ | s26_b2_cells/ (PRED-*) |
| 9 | 5 | S18 | 1 900 000+ | s18_round.json, s18_arm_*.json |
| 10 | 6 | S19 | 2 000 000+ | s19_round.json, s19_cells/ |

**Formal statements share one counter.** They are Lemma 1 (zero speed), Proposition 2 (absorbing stall), Proposition 3 (recovery) and Remark 4 (the size of the signal). Draft 1's Proposition 1 and Remark 2 no longer exist.

## Section-by-section walkthrough

| Section | What it does | What to know |
| --- | --- | --- |
| Abstract | failure, ownership, signal, remedies, recovery | "at least 0.18" is Remark 4's per-decision number; "eleven policies" means Base, Obs and ObsHor ×3 plus Free and Aware (Hor never had the recovery); "silent failure" is still there |
| §1 Introduction | Fig. 1 episode, the name, related deadlock work, C1 to C3 | the claim of novelty is what a learning-enabled controller contributes, not the stall itself |
| §2 Related work | three paragraphs: CBF deadlock and undesired equilibria; shields, liveness and recovery; learning through the filter | draft 1's three asides ("predicts the oscillations", "separate game", "any filter … Kim et al.") are gone; BIG-CBF and DMPS are still absent |
| §3 Setting | game, controller, observation and reward, policies (Table 1), filter, stall definitions | the controller paragraph is the panel's critical fix; read it twice |
| §4 Why the stall is a fixed point | Lemma 1, Proposition 2 with proof sketch, the recovery, Proposition 3 | Lemma 1 is credited to Grover et al.; Prop. 2's hypotheses (i)–(iii) include the law's remembered point; Prop. 3(b) holds under H1–H3, and (c) claims no liveness |
| §5 Protocol | registration, paired seeds, exact binomial test, Newcombe intervals, no multiplicity correction, MDD 0.05–0.08 | Table S1 lists every verdict |
| §5.1 The cost of the filter | Experiment 1 | the 0.426 bar is the filtered scripted pursuer |
| §5.2 What is stuck | Experiments 2 and 3, Fig. 3 | Experiment 2 missed its 99 % agreement bar (98.1 %) and its dwell prediction failed |
| §5.3 Why learning does not remove it | law stalls outside learning; head stalls and Experiment 4; Remark 4; open cause; Experiment 5 | this section replaced draft 1's discount argument |
| §5.4 The recovery | Experiments 6, 7, 8; Fig. 4 | the bound is "typically" (97.8 %), with cycles as the cost |
| §6 Stall metrics can improve while the task fails | Experiments 5, 9, 10 | retitled from draft 1; the recommendation is to report the task, immobility and stalled steps separately, plus firings |
| §7 Discussion and limitations | scope, the recovery's costs, how the failure was found | memory is untested; the head's share ranged from 0.005 to 0.85 |
| §8 Conclusion, reproducibility | five sentences; release placeholder | the repository URL is still "[repository URL]" |
| Supplement | Table S1 verdicts, training details, Table S2 and Fig. S1 geometry, Fig. S2 dissociations, Tables S3–S4, Figs S3–S5 recovery, Tables S5–S6 crosswalk and names | draft 1's figures 4, 5, 6(c) and 8 live here now |

## Experiments 1 to 10 in detail

Each entry gives the question, the design, the registered verdict, the numbers and where it sits. All numbers are from the revised main text or supplement unless marked.

### Experiment 1: the cost of the filter (S4, S5; draft 1 Exp. 1)
- **Question.** What does the filter cost a policy trained without it, and can retraining recover it?
- **Design.** Free against the greedy evader, 500 paired episodes, with and without the CBF filter. Three retrained remedies, three seeds each, with a bar at the filtered capture of the scripted pursuer Free was first compared with (0.426).
- **Numbers.** Free's capture is 0.728 without the filter and 0.396 with it: difference −0.33, interval [−0.375, −0.289], p = 3 × 10⁻⁴⁴. Of the filtered episodes, 0.540 are deadlock episodes. The stall penalty alone gives 0.426, 0.464, 0.436 (fails). Filter features give 0.522, 0.512, 0.544 (p ≤ 8 × 10⁻⁴, passes) and still leave 0.11 to 0.30 deadlock. Both together give 0.564, 0.550, 0.444, and the 0.564 seed is Aware.
- **Verdict.** Filter cost confirmed; features recover part of it; the penalty alone recovers none.
- **Where.** §5.1.

### Experiment 2: geometry of the stalled states (S26 B1; draft 1 Exp. 7)
- **Question.** Is the stall a property of the position or of the pair (command, filter)?
- **Design.** Stalled states of Base ×3 and Aware against the learned attacker, 100 episodes each. Every command on a grid was tested at each state, and the filter's output was compared with the exact weighted projection.
- **Numbers.** There are 262 stalled states, all on the exact tier. In 98.1 % no allowed command moves the pursuer at its current heading. None is a position trap: at every one, another heading allows motion and a rotation of at least ½ ω_max is allowed. Deadlock with sampled actions is 0.42, 0.24, 0.29, 0.57, against 0.38, 0.20, 0.21, 0.55 deterministic.
- **Verdict.** Projection agreement 98.1 % missed its 99 % bar. The dwell prediction (sampling noise ends the stall after a geometric waiting time) failed. The stall survives exploration.
- **Where.** §5.2, Fig. 3(a), supplement Table S2 and Fig. S1. It also checks H1 of Proposition 3.

### Experiment 3: who commands the stall (S27, N1; new in revision 1)
- **Question.** Which part of the controller commanded each deadlock episode?
- **Design.** 40 archived evaluations were replayed: the four families against all three evaders, plus Free and Aware against the attacker and the greedy evader. The rule was pre-registered with predictions P-N1-1 to P-N1-3 and decision rules D-1 to D-3. Each episode is attributed by the part that commanded the 40th stalled step ("first-K"), one of law chasing, law driving to the remembered point, or head.
- **Reproduction gate.** All 20 000 replayed episodes reproduced the archive exactly. Amendment A1 re-ran 13 cells whose checkpoint hash was missing, from a new frozen commit.
- **Numbers, against the attacker.** Of the four families' 2 002 deadlock episodes, the law driving to a remembered point owns 1 747, the head 243, and the law chasing a visible evader 12. The head's share is 0.005 for Base, 0.03 for Obs, 0.06 for ObsHor and 0.30 for Hor. For Free the head owns 0.76 (0.85 against the greedy evader). For Aware it owns 0.35 (0.58).
- **Watched episodes.** The law stalls against an obstacle between the pursuer and its remembered point. The head stalls against a wall or in a corner.
- **Verdict.** All three predictions held; decision rule D-2 applies (D164).
- **Where.** §5.2, Fig. 3(b). The abstract's 87 % is 1 747 of 2 002, and its 76 % is Free's head share.

### Experiment 4: is there a signal to escape (S15; draft 1 Exp. 2)
- **Question.** Does escaping a stall pay, and is the input frozen?
- **Design.** 383 stall entries of Aware and its anchored fine-tune FT-B-41.
- **Numbers.** Of the 97 escaped stalls, 0.299 were followed by a capture, against 0.042 of the 286 that were not; the capture advantage A₀ is 0.257. During a stall the world part of the observation does not change (median change 0.000). The evader is visible in 3.4 % of stalled steps. Shortening the head's hold from 10 steps to 1 changed immobility by less than 0.05.
- **Registered prediction P1.** It said the escape advantage is negligible: |Δ return-to-go| below 0.05 **and** capture after escape below 0.15. It **failed**, because capture after escape is 0.299. Its registered direction for that outcome was: "the signal was there and learning still missed it; the explanation is optimisation, not credit assignment." Revision 1 adopts that sentence.
- **Remark 4, the size of the signal.** The registered analysis discounted per env step and found Δ return-to-go = 0.0092. The learner discounts per decision, and a head stall advances 10 steps per decision. On that clock the difference is 0.29 at γ = 0.99. Also A₀·E[γ^T] = 0.18, where T counts decisions from the stall to the capture on the 29 captured escapes (346 steps on average). The remark is labelled a post-hoc recomputation from the archived entries.
- **What it withdrew.** D128 read the failed P1 as "the reward is discounted away" (629 steps later, worth 0.009). That reading is superseded. The 629 was never a delay to capture; it is the mean number of steps from stall entry to episode end.
- **Where.** §5.3, Remark 4. Draft 1's mechanism figure (F2) is removed.

### Experiment 5: five learned remedies and a rotation-only recovery (S16; draft 1 Exp. 3)
- **Question.** Can learning-side changes remove the stall?
- **Design.** Five arms of three seeds each: Base, Obs, Hor, the tenfold penalty and the recurrent arm. A scripted rotation-only recovery (REC) runs alongside.
- **Verdict.** The registered primary metric, the immobile-step rate, failed on every arm. It was a mis-chosen metric, because immobility counts rotation, which is the remedy's own mechanism.
- **Numbers.** REC reached zero deadlock by spinning in place, with 414 928 firings over 1 500 episodes on one seed. A straight command would have moved the pursuer at every sampled stalled pose (326 of 326 in D130), so the paper's recovery offers forward first. The tenfold penalty is a dissociation: stalled-step rate 0.143–0.249 → 0.093–0.158, immobile-step rate 0.436–0.448 → 0.633–0.833, capture 0.384–0.478 → 0.204–0.348. The policy learns to stop asking to move.
- **The recurrent arm.** It was confounded before its number existed, so memory is untested.
- **Where.** §5.3 and §6, supplement Fig. S2(a).

### Experiment 6: the recovery across four families (S17; draft 1 Exp. 4)
- **Question.** Does the forward-offering recovery at N = 20 clear the stall and raise capture?
- **Numbers.** Base's deadlock against the attacker is 0.298, 0.186, 0.252. With the recovery it is 0.000 on the three Base and three Obs seeds, and 0.006, 0.002, 0.002 on the three ObsHor seeds; the bar was 0.05. Capture of the greedy evader rises on 9 of 9 seeds. For Base it goes 0.466, 0.366, 0.402 → 0.640, 0.718, 0.442, with differences +0.17 [+0.14, +0.21], +0.35 [+0.31, +0.39], +0.04 [+0.02, +0.06].
- **Without the recovery.** Obs captures 0.532, 0.520, 0.638, Hor 0.546, 0.432, 0.414 and ObsHor 0.560, 0.634, 0.636. They keep deadlock between 0.26 and 0.61. §5.3 quotes "0.19 to 0.61" for every family without the recovery, Base included.
- **Firings over 1 500 episodes.** Base seeds fire 13 035, 152 675 and 27 974 times; ObsHor seeds 35 755, 363 and 513, for similar capture. Learning changes how often the recovery is needed.
- **Verdict.** The recovery passes on 3 of 3 families. Hor had no recovery arm.
- **Where.** §5.4, Fig. 4(a), supplement Fig. S3.

### Experiment 7: the trigger N and a random baseline (S26 B2; draft 1 Exp. 8)
- **Question.** Is the recovery sensitive to N, does the escape bound hold, what does it cost, and does a random perturbation do as well?
- **Design.** N ∈ {5, 10, 20, 40, 80} on Base ×3, Free and Aware, 500 paired episodes, plus PERT-20.
- **Capture and deadlock.** Capture of the greedy evader is significantly above the plain filter at every N on all five policies. For N ≤ 20, deadlock against the attacker is at most 0.022 on every policy, against 0.186 to 0.710 plain. For N = 40 and 80, stalls of at least 2N steps occur in at most 0.020 of episodes, against 0.176 to 0.682 plain. Free goes 0.426 → 0.728 (+0.30 [+0.26, +0.34]) and its immobile-step rate 0.594 → 0.071. Aware goes 0.560 → 0.654 (+0.09 [+0.07, +0.12]).
- **The bound.** The median time from a trigger to a 0.10 m displacement is 4 to 10 steps. Of 10 475 triggers at N = 20, 97.8 % fall within the registered bound N + 45. The per-evaluation 95 % bar failed on Base seed 3 (94.5 % and 92.3 %), so it was missed on 2 of 10 evaluations.
- **The cost.** At N = 20, between 8 and 56 % of engagements end in another stall within 40 steps. The immobile-step rate stays between 0.023 and 0.101.
- **Baseline.** PERT-20 cleared the deadlock on 0 of 5 policies (0.304, 0.226, 0.280, 0.486, 0.560 against plain 0.298, 0.186, 0.252, 0.710, 0.586). It was known before launch to be unable to rotate (amendment A4). A rotation-accepting baseline (N2) was not run, and the paper says so.
- **Where.** §5.4, Fig. 4(b, c), supplement Fig. S4. Fig. 1's episode comes from this experiment.

### Experiment 8: a second filter (S26 B3, B4; draft 1 Exp. 9)
- **Question.** Does the stall exist under a different filter, and does the recovery compose with it?
- **Numbers.** Under the predictive filter, deadlock against the attacker is 0.104, 0.036, 0.046 for the three Base seeds and 0.468 for Aware, against 0.298, 0.186, 0.252, 0.586 under the CBF filter. With the recovery composed onto it, deadlock falls to 0.000, 0.000, 0.002, 0.014. Capture rises on 4 of 4: 0.468 → 0.628, 0.454 → 0.702, 0.396 → 0.432, 0.554 → 0.668, all p < 0.05. The gate rejected 0 of 39 162 commands.
- **Verdict.** The bar, deadlock of at least 0.05 on 3 of 4 policies, was met on 2. The stall exists under the second filter but is weaker at this horizon. That filter is also less conservative, so the difference is not attributed to its construction alone. The composition passes 4 of 4.
- **Where.** §5.4, supplement Fig. S5.

### Experiment 9: a correction-magnitude penalty (S18; draft 1 Exp. 5)
- **Question.** Does Pizarro Bejarano et al.'s penalty on the filter's correction help here?
- **Design.** The penalty in unsquared normalised form, three reward variants (DVARY dense, DRARE the other dense, SPARSE), three seeds each.
- **Numbers.** In the dense variant, capture falls on 3 of 3 seeds: 0.442, 0.480, 0.528 → 0.338, 0.300, 0.212, pooled p = 3.6 × 10⁻⁵¹. Zero-speed requests rise from 0.07, 0.16, 0.00 to 0.37, 0.78, 0.80. Deadlock improves in 9 of 9 evaluations.
- **Verdict.** The registered cross-over prediction (H-CROSS: the penalty helps under the dense reward and hurts under the sparse one) failed. The penalty never raised pooled capture: sparse 0.535 with and without, the other dense variant 0.555 → 0.535 (p = 0.13).
- **Confound.** The penalty was also charged for the law's corrections, a registered confound (the law carries a median 0.784 of the correction magnitude, D139/D141).
- **Where.** §6, supplement Table S3 and Fig. S2(b).

### Experiment 10: the law replaced by a planner (S19; draft 1 Exp. 6)
- **Question.** If the law's stalls are removed at the source, does capture rise?
- **Design.** A fully scripted pursuer with orbit search and no learned part. An obstacle-avoiding planner replaces the intercept law.
- **Numbers.** Law stalls fall from 0.170, 0.132, 0.368 of episodes to 0.002, 0.008, 0.004. Capture moves by +0.026, −0.018, −0.022. The planner turned 16 of the 85 law-stalled greedy episodes into captures; the net change is 190 → 203.
- **Verdict.** H-PLAN-1 failed (1 of 3) and H-PLAN-3 passed: law stalls removed 3 of 3, capture gain 1 of 3.
- **Where.** Two sentences in §6, supplement Table S4. It is the third dissociation.

## Figures and tables

Revision 1 redrew every main-text figure at its printed width (433.6 pt) with 7 to 8 pt type. File names now start with R; the remaining F files appear only in the supplement.

**Main text.**

| Float | File | Shows | Experiment |
| --- | --- | --- | --- |
| Fig. 1 | figures/R0_game.png | (a) Free vs greedy, seed 1 740 044, with the CBF filter: stall at the top wall from step 8 for 1 192 steps, inset of the request; (b) the same episode with the recovery at N = 20: 7 firings, capture at step 507; (c) commanded and executed speed | 7 |
| Fig. 2 | figures/R1_loop.tex (TikZ) | the closed loop: law, head, filter, the recovery counter beside the filter | — |
| Table 1 | in main.tex | the six policies (trained, filter in training, extra observation, discount) | — |
| Fig. 3 | figures/R2_stall.png | (a) the allowed commands at one stalled state of Base seed 2: only standstill and turning; (b) the N1 attribution against L (dark) and G (light), with the head's share and the episode counts | 2, 3 |
| Fig. 4 | figures/R4_recovery.png | (a) deadlock against L versus capture of G per seed, without (filled) and with (open) the recovery, arrows joining each seed; (b, c) deadlock and capture against N, open markers for PERT-20; deadlock drawn for N ≤ 20 only | 6, 7 |

**Supplement.**

| Float | File | Shows |
| --- | --- | --- |
| Table S1 | — | the ten experiments with their registered verdicts |
| Table S2 | — | Experiment 2's per-policy stall classification |
| Fig. S1 | F6_geometry | Experiment 2: allowed set with the request, the output and the exact projection; classification; observed vs predicted dwell |
| Fig. S2 | R5_dissoc | (a) Experiment 5's stall penalty; (b) Experiment 9's correction penalty |
| Table S3 | — | Experiment 9 capture without and with the penalty, all reward variants |
| Table S4 | — | Experiment 10, what became of the law-stalled episodes |
| Fig. S3 | F4_firings | Experiment 6 firings per family, forward versus turning candidates |
| Fig. S4 | F7_recovery_sweep | Experiment 7 against N: deadlock, capture, firings per episode, plus the 10 466 vs 10 475 trigger-count note |
| Fig. S5 | F8_predictive | Experiment 8 under both filters, without and with the recovery |
| Table S5 | — | experiment crosswalk: new number, draft 1 number, round, seed stream, records |
| Table S6 | — | policies, remedies and evaders: paper name to identifier |

**Removed in revision 1.**
- **Draft 1's Fig. 1 (F1).** It showed Obs seed 2 against the wall-hugging evader, seed 1 740 003, 1 106 stalled steps. N1 attributes all of those steps to the law chasing a visible evader, so it was not the learned failure the text described.
- **Draft 1's Fig. 2 (F2, the discount mechanism).** Withdrawn with the discount explanation.
- **F3 and F5.** Superseded by R4_recovery and R5_dissoc.

**Data and scripts.** The data is in `figures/figure_data_rev.json`, produced by `scripts/paper_b_game_figure.py` (Fig. 1) and `scripts/paper_b_figures_rev.py`. The Fig. 1 script reproduces the archived capture indicators and N1's first 40-step run before it draws, and refuses to draw otherwise. The supplement's F figures come from the older `paper_b_figures.py` and `paper_b_figures_s26.py` with `figure_data.json` and `figure_data_s26.json`.

## Key numbers to memorise

| Claim | Number | Where |
| --- | --- | --- |
| The failure in one episode | Free vs G seed 1 740 044: stall from step 8 for 1 192 steps; with the recovery, 7 firings, capture at step 507; zero collisions | Fig. 1, Exp. 7 |
| The filter's cost | Free 0.728 → 0.396, −0.33 [−0.375, −0.289], p = 3 × 10⁻⁴⁴; 0.540 deadlock episodes | Exp. 1 |
| Remedies of Exp. 1 | penalty 0.426, 0.464, 0.436; features 0.522, 0.512, 0.544; both 0.564, 0.550, 0.444 (bar 0.426) | Exp. 1 |
| The stall is the pair's, not the position's | 262 states; 100 % exact tier; 98.1 % trap at heading; 0 position traps; turn always allowed; projection agrees 98.1 % (bar 99 %) | Exp. 2 |
| Exploration does not end it | sampled 0.42, 0.24, 0.29, 0.57 vs deterministic 0.38, 0.20, 0.21, 0.55 | Exp. 2 |
| Who owns deadlock, trained families vs L | law to remembered point 1 747, head 243, law chasing 12, of 2 002 (87 % law-to-point) | Exp. 3 |
| Head share per policy | Base 0.005, Obs 0.03, ObsHor 0.06, Hor 0.30; Free 0.76 (0.85 vs G); Aware 0.35 (0.58 vs G) | Exp. 3 |
| Reproduction | 40 cells, 20 000 episodes, all exact | Exp. 3 |
| Escaping pays | capture after escape 0.299 of 97 vs 0.042 of 286; A₀ = 0.257 | Exp. 4 |
| The input is frozen | world-observation change median 0.000; evader visible 3.4 % of stalled steps; hold 10 → 1 changes immobility < 0.05 | Exp. 4 |
| The signal | registered 0.0092 per env step; per decision 0.29 at γ = 0.99; A₀·E[γ^T] = 0.18; 346 steps mean to capture on 29 captured escapes | Remark 4 |
| A decision | ≤ 10 held head steps plus the law's run; 14.0–16.6 env steps in training (Base-60: 15.9) | §3, D165 |
| Rotation-only recovery spins | 414 928 firings over 1 500 episodes on one seed | Exp. 5 |
| Tenfold penalty dissociation | stalled 0.143–0.249 → 0.093–0.158; immobile 0.436–0.448 → 0.633–0.833; capture 0.384–0.478 → 0.204–0.348 | Exp. 5 |
| The recovery clears deadlock | Base 0.298, 0.186, 0.252 → 0.000; Obs → 0.000; ObsHor → 0.006, 0.002, 0.002; bar 0.05 | Exp. 6 |
| And raises capture | Base 0.466, 0.366, 0.402 → 0.640, 0.718, 0.442; 9 of 9 seeds | Exp. 6 |
| Firings vary for equal capture | Base 13 035, 152 675, 27 974 vs ObsHor 35 755, 363, 513 per 1 500 episodes | Exp. 6 |
| Insensitive to N | capture up at every N on 5 of 5; deadlock ≤ 0.022 for N ≤ 20 vs 0.186–0.710 plain; ≥ 2N-step stalls ≤ 0.020 for N ≥ 40 | Exp. 7 |
| Free and Aware with the recovery | Free 0.426 → 0.728 (+0.30), immobile 0.594 → 0.071; Aware 0.560 → 0.654 (+0.09) | Exp. 7 |
| The bound | 97.8 % of 10 475 triggers within N + 45; median 4–10 steps; Base seed 3 at 94.5 and 92.3 % | Exp. 7, Prop. 3 |
| The cost | cycles 8–56 % of engagements; immobile 0.023–0.101 | Exp. 7 |
| Random baseline | PERT-20 clears 0 of 5 | Exp. 7 |
| Second filter | predictive 0.104, 0.036, 0.046, 0.468 vs CBF 0.298, 0.186, 0.252, 0.586; composed 0.000, 0.000, 0.002, 0.014; capture up 4 of 4 | Exp. 8 |
| Correction penalty | capture 0.442, 0.480, 0.528 → 0.338, 0.300, 0.212, pooled p = 3.6 × 10⁻⁵¹; zero-speed requests up to 0.37, 0.78, 0.80 | Exp. 9 |
| Planner | law stalls 0.170, 0.132, 0.368 → 0.002, 0.008, 0.004; capture +0.026, −0.018, −0.022; 16 of 85 converted; 190 → 203 net | Exp. 10 |
| Prop. 3(b) constants | T_turn 45 steps at ω_max, 89 at ½ω_max; 40 steps for 0.10 m at the threshold; rotation threshold 0.1 rad/s | §4 |
| Protocol | 500 paired episodes; exact binomial on discordant pairs, p < 0.05; Newcombe 95 % intervals; MDD 0.05–0.08; no multiplicity correction | §5 |
| Process | fifteen specifications passed before a person watching episodes found it; 1 085 consecutive blocked requests in one episode | §7 |

## How the result was found: the decision history

The paper exists because a submission was withdrawn. Newest first; D-numbers are entries in docs/decisions.md, S-numbers are rounds. Entries D143 to D158 belong to Paper A (the tactic game) and are omitted except where they touched this paper.

**Added 2–3 Nov (D163 to D165).** D163: a five-reviewer panel returned MAJOR REVISION, with one CRITICAL finding (the intercept law has memory and acts out of view), four consensus defects and a 17-item roadmap. D164: the branch-attribution round N1 (now Experiment 3) found the law owns 87 % of the trained families' deadlock episodes and the head 76 % of Free's. D165: revision 1 withdrew the discount explanation, replaced Fig. 1, renumbered the experiments and changed the title. Experiment numbers in the table below are the ones used at the time (draft 1).

| Date | Decision | What it changed for the paper |
| --- | --- | --- |
| 2026-11-03 | D165: revision 1. PPO discounts per decision (about 16 env steps), so the escape signal is 0.18 to 0.29, not 0.009; the discount explanation is withdrawn and P1's registered direction adopted. Draft 1's Fig. 1 was a law-chase stall and is replaced. Every trained family uses Aware's recipe (disclosed). Title "Learning-Enabled Controllers" (user) | The current draft, response letter and facts file |
| 2026-11-03 | D164: N1 result. 40/40 cells reproduce the archive; first-K attribution; P-N1-1 to 3 pass; decision rule D-2 | Experiment 3; C2 rescoped |
| 2026-11-02 | D163: five-reviewer panel, MAJOR REVISION (first logged as D162; renumbered because Paper A's entry also took D162) | Roadmap R1 to R17, N1 to N3, O1 to O4 |
| 2026-11-01 | D161: paper-facing names (Experiments 1 to 9, Base/Obs/Hor/ObsHor/Free/Aware, G/H/A); a separate release repository safe-and-stuck instead of a branch | The draft, figures and supplement carry no internal labels; the archive keeps them; the public branch of this repository (670 commits, a password in its history) is not what the paper links |
| 2026-11-01 | D160: S26 B2 to B4 results | Prop. 3 confirmed and sharpened: insensitive to N, bound "typically" (97.8 %), cost is cycles, the perturbation baseline cannot leave the trap, the discovery's headline archived at N = 500, loop present but weaker under the predictive filter, composition 4 of 4 (B4 in) |
| 2026-11-01 | D159: S26 B1 result | Prop. 1's geometry confirmed on the exact tier, the QP characterisation scoped by the 2 % enumeration share, the geometric dwell law dropped and R3 demoted to a remark |
| 2026-11-01 | User decisions on the skeleton: title; Fig. 1 archived trace opens the paper and the discovery story moves to the process paragraph; the scripted-controller section dropped to one sentence; Pizarro Bejarano named as the penalty's source with the squared/unsquared caveat; password rotation deferred to Nov 10; template-minimal AI statement; author block | The draft's shape |
| 2026-11-01 | Paper B goes to L4DC 2027 (user); budget 5.0 M tokens; go/no-go Nov 10; Paper A stays on the RA-L plan | Venue, page limit, single-blind, deadline |
| 2026-11-01 | Evidence audit (claims C1 to C14 traced to records; LOG-ONLY diagnostics may be narrated, never gated on); verified citations; Yeke et al. dropped (abstract does not mention deadlock); the word pinned banned | The rules the draft follows |
| 2026-10-12 | User: stop the paper; the next direction is the user's (after D144 closed the agent-intelligence line) | Paper B paused for three weeks while Paper A's tactic game ran (D145 to D157) |
| 2026-10-13 | D149 (inside Paper A): the plain CBF filter reproduced the absorbing loop in the scripted tactic game (pins of 760 and 787 steps) and RecoveryShieldV2 fixed it | The one-sentence anecdote in Section 5.1; no archived paired numbers, so no claim rests on it |
| 2026-10-06 | D142: S19 result (Experiment 6): law pins fall 94 to 99 %, capture flat or worse | Dissociation 3; the construct-validity thesis has three instances |
| 2026-10-06 | D139, D141: Case 5 quantified by exact replay of all 36 S18 cells: the scripted law carries a median 0.784 of the intervention magnitude and 0.863 of the stalled steps | The registered confound stated in Section 6; the branch tag was added to the recorder |
| 2026-10-06 | D138: S18 result (Experiment 5): the correction-magnitude penalty lowers capture in every column and improves the deadlock metric 9 of 9 | Dissociation 2; the boundary-condition paper withdrawn |
| 2026-09-23 | D133 to D137: the boundary-condition plan (penalty helps when the task reward has gradient inside the intervened region), the penalty form fixed as Pizarro Bejarano's, weights calibrated, construct gates CG1 and CG2 failed before any number and were handled in the open | The design of Experiment 5; the "published contradiction" framing later withdrawn, the paper names the source and the contrast instead |
| 2026-09-22 | D132: novelty check. BIG-CBF (arXiv:2609.14343) states "safe but stalled behavior" first, with a classical nominal and no rates; Kim et al. see looping actions; Georgescu et al. repair specifications; DMPS searches recovery actions under a guarantee; Pizarro Bejarano report the penalty helping. Verdict: the phenomenon is prior art, the diagnosis, measurement and remedy comparison are ours; stop experimenting and write | The paper's claimed delta; see the related-work section for what did and did not reach the bib |
| 2026-09-22 | D131: S17 result (Experiment 4): the forward-offering recovery solves all three registered hypotheses on 3 of 3 families; bare learned families fail; firings 363 against 152 675; H3 anti-spin uninformative | The headline |
| 2026-09-21 | D130: anatomy of the rotation-only recovery's spins: one engagement per episode, 9 of 24 ran to the end; a straight command moves in 100 % of 326 sampled poses | Forward candidates first; per-step logging of nominal and executed commands |
| 2026-09-21 | D129: S16 result (Experiment 3): primary metric mis-chosen (immobility counts rotation); REC clears deadlock by spinning; Obs best learned; Hor supports the discount; the stall penalty backfires; MEM confounded | Dissociation 1; the next round's primary metric became the deadlock-episode rate |
| 2026-09-20 | D128: S15 result (Experiment 2): P1 failed on a mis-written conjunct; escaping multiplies capture 7.1 times but 629 steps later; input frozen; exploration and commitment excluded | The mechanism and Remark 2; the discount reading was withdrawn by D165 |
| 2026-09-19 | D127: the loop survives the policy's own exploration noise (V3) | Phase 0 complete; later archived as Experiment 7's stochastic arm |
| 2026-09-19 | D126: not our discretisation: an exact QP returns no forward motion either (0 of 150); translation blocked, rotation admissible 150 of 150; escape needs rotate then move and nothing proposes it | The corrected mechanism; later archived as Experiment 7's classification |
| 2026-09-19 | D125: the stall is a missing escape behaviour. The S14 clips showed the pursuer stalled in most episodes; at a stalled state the filter admits the opposite turn; the policy asked for the blocked one for 1 085 steps at σ = 0.002; a 20-step rotation recovery removed 94 % of immobility with zero violations and better capture. D112 to D124 re-scoped; the AAMAS 2027 submission withdrawn; fifteen specifications had passed | The paper's thesis and its process paragraph; rebuild plan docs/rebuild_plan_2026-09-19.md |
| 2026-09-14 to 15 | D112, D113: S4 and S5, the shield costs 0.728 to 0.396 and filter-state features recover part of it | Experiment 1, cited only through the re-scoped reading |

**What was dropped, and why.**

- The whole shielded-games chain D112 to D124 as written: it measured a policy frozen in the loop, not an attack surface on hard filters.
- The stochastic dwell law (R3): B1 could not test it on two policies and found it 3 to 12 times too long on the other two; deadlock under exploration is reported instead.
- The section on the loop in a scripted controller: no paired numbers at N = 500 exist for that controller, so it is one sentence with no numbers.
- The "published contradiction" framing of D133: Pizarro Bejarano et al. are the penalty's source and the contrast case, not an opponent.
- Yeke et al. (USENIX Security 2025): the abstract names position deviation, herding and collisions, not deadlock.
- The word pinned: it names a different metric in the code.
- The 5 310-word budget: the class holds about 500 words a page and the figures take about 2.3 pages, so the draft is about 3 400 words; the 16-page and 14-page long formats are kept under draft/long/.

## Related work and positioning

Twenty-one of twenty-three verified references are cited. The delta is diagnostic and empirical, and the paper says so; the normal-cone condition is presented as known.

| Line | Works in the bib | What they show | The paper's delta |
| --- | --- | --- | --- |
| Deadlock of CBF-QP controllers | Grover, Liu and Sycara: IFAC 2020 (why symmetry causes deadlocks), WAFR 2020 (analysis and resolution), IJRR 2023 (before, during and after) | Multi-robot deadlock under CBF-QPs characterised by KKT: the projected command is zero exactly when the nominal is a conic combination of active constraint normals; the deadlock set has measure zero | One robot against static geometry; the nominal is a learning-enabled controller; the equilibrium becomes absorbing because the controller's input and internal state freeze; the filter stays untouched. Revision 1 states Lemma 1 as theirs |
| Undesired equilibria of CBF-QPs | Reis, Aguiar and Tabuada (L-CSS 2021); Tan and Dimarogonas (Automatica 2024); Reis and Aguiar (Automatica 2026); Reis, Carvalho and Aguiar (arXiv 2025) | CLF-CBF programs create spurious stable equilibria for a fixed nominal; all closed-loop equilibria can be enumerated and removed by modifying the program; liveness is not guaranteed | Those works fix the filter; we keep it and its guarantee and remove the stall with one counter outside the filter |
| Safety filters and shields (framing) | Hsu, Hu and Fisac (2024 review); Ames et al. (ECC 2019); Alshiekh et al. (AAAI 2018); Cheng et al. (AAAI 2019) | The filter paradigm; forward invariance | The guarantee says nothing about progress |
| L4DC 2025 neighbours | Kim et al. (realizable shields; the shield alone is overly conservative; loop avoidance as a stateful requirement); Dawood et al. (adaptive MPC shield); Wang, He and Pajic (neuro-symbolic multi-robot deadlock resolution); Arnström and Teixeira (stealthy deactivation of filters) | Conservativeness and deadlock handled on the specification side, by adaptive weights, or after the fact | One counter, guarantee untouched, learned policies in an adversarial game; the "predicts their oscillations" aside was removed in revision 1 |
| Learning through the filter | Oh, Nguyen, Hu and Fisac (2026): a filter-unaware policy chatters; training through a permissive enough filter is optimal. Wu et al. (2026, S2C): minimally exploitable policies via filtered RL. Georgescu et al. (NeSy 2026): GR(1) specification repair for liveness. Pizarro Bejarano, Brunke and Schoellig (RA-L 2025): penalise the filter's correction during training | Training through the filter helps when the filter is permissive; penalties help in tracking | At the stall the filter is not permissive; training through it with its state observed leaves the stall in place, and the escape signal is there but unused; the penalty inverts in a task where standing still is locally free |
| In the bib but not cited | Nakamura et al. (2025, switching filters hurt task completion); Lee et al. (IROS 2025, Merry-Go-Round, deadlock prevention) | | Dropped from the related-work paragraph for space |

**Named in the novelty check (D132) but in neither the verified list nor the bib; still absent after revision 1.** Decide before the go/no-go.

- BIG-CBF, Li, Wang, Cao, Yang and Wu, arXiv:2609.14343 (submitted 2026-09-13). Its abstract says a minimum-intervention CBF filter "may fail to select a productive avoidance direction ... leading to safe but stalled behavior". Classical nominal, no learning, a two-rate architecture as the contribution, no stall rates. D132 called it the decisive near-scoop. The delta survives (learned nominal, measured rates, pre-registered remedy comparison) but a reviewer who knows it will ask why it is absent.
- DMPS, Banerjee, Rahmani, Biswas and Dillig, NeurIPS 2024, arXiv:2405.13863: a static backup controller halts the agent; they search candidate recovery actions by tree search while preserving the guarantee. The nearest neighbour of the recovery; ours is a fixed candidate list after N immobile steps, measured.
- "Silent Failures in Physical AI", arXiv:2606.00090, uses "silent failure" for a filter wrongly authorising an unsafe action, the opposite event; the abstract's "fails silently" may want one disambiguating clause.

**Suggested by the editor-in-chief review, not required.** Wabersich and Zeilinger (Automatica 2021, the predictive safety filter) beside the first mention of a predictive filter; Jankovic, Santillo and Wang (IEEE TCST 2024, liveness from instability of the deadlock equilibrium) as a third route beside modifying the QP and adding a recovery; Brunke et al. (Annual Review 2022) for the safe-RL reader. The full panel (D163, item SC-14) adds Agrawal and Sreenath 2017 and Wang, Ames and Egerstedt 2017 for discrete-time CBFs, Trautman and Krause 2010, Koren and Borenstein 1991 for potential-field local minima, a navigation-stack recovery reference, Singh, Jaakkola and Jordan 1994 for aliasing of memoryless policies in POMDPs, and optionally Gurriet et al. 2018, Mestres and Cortés, Zeng et al. 2021, Sutton, Precup and Singh 1999, Hausknecht and Stone 2015, Jansen et al. 2020; each must pass the verification protocol before it is cited.

**Topic fit.** "Safe RL" is not in the L4DC 2027 topic list; the paper is framed under reinforcement learning and the robustness of the learning-control interface, with robotics as the application. The title avoids "safe RL" for that reason.

## Limitations, caveats and weak points

Revision 1 answered the panel's 17 required items (see the checklist). What follows is what a re-review can still press on, strongest first.

| Weak point | Why it matters | The honest reply |
| --- | --- | --- |
| The head-stall cause is open | C2 now says the signal is there and unused, but not why | Aliasing is the stated candidate: the observation has neither walls nor obstacle radii. Memory is untested because the recurrent arm was confounded before its number existed. Say "open", not "explained" |
| Remark 4 is post-hoc and overturns the registered analysis's clock | A reviewer can ask why the pre-registered computation discounted per env step | It is labelled post-hoc. It moves the paper toward the registered direction ("optimisation, not credit assignment"), not away from it. The registered 0.0092 is still reported |
| The learning claim is narrow for the trained families | 87 % of their deadlock is the law, inside one decision; for Base the head's share is 0.005 | The title says "learning-enabled controllers", and the paper says law stalls are outside learning. The learned-head story rests mainly on Free (0.76), Hor (0.30) and Aware (0.35) |
| "Base" is not plain | Every trained family uses Aware's recipe: filter features plus a 10⁻³ stall penalty | Disclosed in §3 and Table 1 since D165. There is no family trained behind the filter without the features |
| The rotation-accepting baseline (N2) was not run | "The recovery's contribution is the turn candidate" rests on geometry plus a baseline known to be unable to rotate | The paper says it was not run. It would take under an hour, pre-registered first |
| Hor never had the recovery | "eleven policies" excludes the family with the longest horizon | Stated. An optional evaluation of 3 cells (O2) |
| Prop. 3(b) hypotheses H2, H3 are unchecked along the turn | The bound is "typically" (97.8 %) and fails on one seed's evaluations | The paper says so. The tail split into creep versus interrupted turn needs a replay and was not done |
| Experiment 2 missed its 99 % agreement bar | 98.1 % agreement between the implemented filter and the exact projection | Stated; in the 1.9 % the projection would have moved the pursuer |
| Experiment 10's tension | Removing law stalls at the source left capture flat | Reported as a dissociation: report the task, not the stall metric |
| Experiment 9's confound | The correction penalty was also charged for the law's corrections | Registered and stated |
| Selection | Free is the best of 5 seeds, Aware the best of 3 | Disclosed. The headline recovery results use the Base seeds trained from scratch |
| Statistics | No multiplicity correction; MDD 0.05–0.08 at 500 pairs; three seeds per family | Policy sentence, intervals and MDD are in §5 |
| Scope | One game, one robot model, static obstacles, two filters | Moving obstacles would break Proposition 2's frozen input. Stated in §7 |
| Recovery costs | Cycles of 8–56 %, firings varying by three orders of magnitude, no liveness guarantee | §7 lists them. On hardware a firing is a manoeuvre |
| BIG-CBF and DMPS still absent | Both were near-scoops in the September novelty check (D132) | Not in the bib after revision 1. Decide before Nov 10 |
| "Silent failure" still in the abstract | arXiv:2606.00090 uses the phrase for the opposite event | One disambiguating clause, or keep it |
| Release repository out of date | The crosswalk and the new figure scripts are not mirrored to safe-and-stuck; licence open; README commit-hash note pending | Listed under "Not done" in the response |
| Placeholders | Corresponding email and repository URL | User decides on Nov 10 |

## Anticipated reviewer questions

The answers revision 1 supports, in the order a re-review would ask.

1. **Is Lemma 1 new?** No. It is the KKT condition of the weighted projection, the single-robot case of Grover et al.'s force-equilibrium condition, and the paper credits it. The paper's own parts are Proposition 2 (the fixed point with the controller's internal state) and the 98.1 % / 0 % measurement.
2. **The law has memory. Is this a learning failure at all?** Partly, and the paper now says exactly how much. In trained families the law owns 87 % of deadlock episodes, inside a single decision, so no learned parameter is in their loop. In Free the head owns 76 %. The title says "learning-enabled controllers" for this reason.
3. **Why does learning not remove the head's stalls?** The signal is there: capture after escape is 0.299 against 0.042, worth about 0.18 to 0.29 on the learner's clock. The learner does not use it. The registered reading is optimisation, not credit assignment. Aliasing is the candidate cause, and memory is untested.
4. **Draft 1 said the reward is discounted away. What changed?** The analysis discounted per env step, but PPO discounts per decision, about 16 env steps in training and 10 in a head stall. On the right clock the signal is not small. The explanation is withdrawn and the registered direction adopted (D165).
5. **Why not a recurrent policy?** The recurrent arm was confounded before its number existed (a 128-step rollout against 2 048 elsewhere). The paper says memory is untested.
6. **Isn't the recovery just a timeout?** The random-perturbation baseline is that timeout and clears deadlock on 0 of 5 policies, because at a trap at heading no moving command exists at the current heading. That baseline could not rotate. A rotation-accepting baseline was not run, and the paper says so.
7. **Does the recovery change the guarantee?** No. Every executed command is an output of the base filter at the current state (Proposition 3(a)). The exact test failed on one step per 500-episode evaluation of Experiment 6, the backup tier was never used, and collisions are zero in every filtered evaluation.
8. **Is there a liveness guarantee?** No. Proposition 3(c) claims none. Cycles are measured: 8 to 56 % of engagements at N = 20.
9. **Why is the bound N + T_turn + 40 but the test N + 45?** The proposition's bound is the provable one under H1 to H3 (one full turn plus the displacement steps). N + 45 is the stronger pre-registered bound, which held on 97.8 % of triggers.
10. **Does it depend on your filter implementation?** The candidate search agrees with the exact projection on 98.1 % of stalled states (bar 99 %, missed). The stall also appears under a predictive filter, weaker at horizon 4, and that filter is less conservative. The recovery composes onto it 4 of 4.
11. **What is in the observation?** 11 base numbers: own pose, evader relative pose with a visibility flag, the obstacle centres. Radii and walls are not observed. Policies trained behind the filter get three filter features, the stall counter saturating at 80 steps. Obs adds the allowed-turn range.
12. **Was Base trained without the filter's help?** No. Every trained family has the filter features and the 10⁻³ stall penalty. That is disclosed.
13. **Were Free and Aware cherry-picked?** They are best-of-seed (5 and 3), disclosed with the RC siblings' values. The headline recovery results use the three Base seeds.
14. **Multiplicity?** Primary hypotheses per experiment, no correction, stated. Headline effects come with Newcombe intervals, and the MDD is 0.05 to 0.08, so smaller effects are called fragile.
15. **Why does the stall penalty backfire?** Standing still costs nothing per step, so the cheapest way to avoid a penalty on stalled steps is to stop asking to move. The stalled-step rate falls, immobility rises and capture falls.
16. **Pizarro Bejarano et al. found their penalty helps.** In their tracking task, standing still is never locally optimal. Here it is, and the penalty lowers capture on 3 of 3 seeds. The registered cross-over prediction failed, and the confound with the law's corrections is stated.
17. **Experiment 10 removed law stalls and capture did not rise. Doesn't that contradict the thesis?** It is reported as a dissociation. 16 of 85 law-stalled greedy episodes became captures, but other episodes went the other way (190 → 203 net). Removing stalls is necessary for the task, not sufficient.
18. **What about BIG-CBF?** It names "safe but stalled behavior" for a classical nominal, with no learning and no rates. It is still not cited after revision 1. Better to cite it before being asked.
19. **How was this found, and why trust the protocol?** Fifteen specifications passed before a person watching episodes found it. Watching designated episodes is now a gate. Since then, the registration recorded failures honestly, including P1 and H-CROSS, and the panel found every spot-checked number reproducible.
20. **Why L4DC?** The interface between a certified filter and a learning-enabled controller is L4DC's subject, and four L4DC 2025 papers are engaged directly.

## Glossary

| Term | Meaning in this paper |
| --- | --- |
| CBF, control barrier function | a function h(s) ≥ 0 on the safe set whose one-step decrease is bounded by a rate (0.2 here); a command is admissible if the barrier condition holds on the successor state |
| forward invariance | the guarantee: a trajectory that starts in the safe set stays in it; it says nothing about progress |
| safety filter | takes the policy's command, tests it, and executes the nearest admissible command when the test fails |
| admissible set A(s) | commands in the box [0, v_max] × [−ω_max, ω_max] that pass the exact one-step barrier test at s; the box intersected with half-spaces, hence convex |
| exact, relaxed, backup tiers | the filter's three fallback levels; every stalled state in the paper is on the exact tier |
| projection filter, weighted norm | the executed command minimises 150 (Δv)² + (Δω)² over A(s) |
| normal cone, KKT condition | the projection of u onto a convex set is zero exactly when u lies in the normal cone at zero, equivalently u is a non-negative combination of the active constraint normals |
| nominal u_π | the policy's command before filtering |
| memoryless policy | a policy whose command depends on the current observation only (no recurrence); the head is memoryless, the law is not |
| decision | one head command plus the law steps that follow it; one PPO transition, about 16 env steps in training, 10 in a head stall |
| first-K attribution | N1's rule: a deadlock episode belongs to the part of the controller that commanded its 40th stalled step |
| observation o(s) | pose features, the opponent's visibility and bounded filter features; the filter's stall counter saturates at 80 steps |
| structured search policy | a scripted intercept law (chase while visible, drive to the remembered last-seen point while not) with a PPO head whose search action is held for 10 steps when the law has nothing to act on |
| intercept law, search branch, law branch | the scripted pursuit rule: chases when the evader is visible, otherwise drives to its remembered last-seen point; it carries one remembered point, frozen during a stall; the S26 records tag each step by branch, and N1 (Experiment 3) recovered episode ownership for the older records by replay |
| hold-10, decision hold | the number of environment steps a head decision is held |
| immobile, stalled, self-stop | executed speed ≤ 0.05 v_max; immobile with nominal ≥ 0.5 v_max; immobile and not stalled |
| stall run, deadlock episode, K | maximal run of stalled steps; an episode with a run of at least K = 40 steps (2 s) |
| filter trap at heading, position trap, pair trap | no admissible motion at this pose; at every heading of this position; the former without the latter |
| escape, engagement, cycle | 0.10 m displacement from the stalled pose; a cluster of firings fewer than 5 steps apart; an engagement ending in a stall within 40 steps of the previous one |
| firing | one step on which the recovery replaced the nominal |
| N, recovery_after | consecutive immobile steps before the recovery fires; 20 unless stated |
| T_turn | 45 steps, one full turn at maximum turn rate (89 at half rate, the rate Experiment 2 certifies, Table S2) |
| candidates | (v_max, 0), (v_max / 2, 0), the nominal, (0, ±ω_max), filtered in that order |
| perturbation baseline | PERT-20: up to eight uniform commands from the box, filtered, first mover executed |
| predictive filter, lookahead horizon | PredictiveShield: a command is admissible if a sampled rollout through the simulator survives 4 steps |
| composition by MRO | PredictiveRecoveryShield inherits from RecoveryShieldV2 and PredictiveShield so every candidate passes the predictive test |
| violation, contact flag | the simulator's pursuer-obstacle contact indicator; the safety hypothesis requires zero contact steps |
| discount γ | 0.99, or 0.999 for Hor and ObsHor, applied once per decision; A₀ the undiscounted capture advantage of escaping, T the decisions from stall to capture (Remark 4) |
| paired seeds, seed stream | the same episode seeds across arms, drawn from a fresh block (for example 1 740 000+) never used in training |
| exact two-sided binomial test on discordant pairs | the paired test: episodes where exactly one arm captured, tested against one half |
| cell, arm, gate | one (policy, opponent, setting) evaluation of N episodes; one trained or configured policy; the evaluation stage of a round |
| pre-registration, bar, verdict rule, failure sentence | the document fixed before any number: hypotheses, thresholds, how to decide, what to write on failure |
| construct gate (CG) | a check that refuses to summarise a round until, for example, shared cells reproduce bit for bit and the designated episodes have been watched |
| stop rule | no cell is re-run or re-cut after its number exists |
| frozen commit, worktree | the code state at which a round ran, hashed into each record |
| H-SAFE, safety hypothesis | zero contact steps in every filtered cell |
| Case 5 confound | in Experiment 5 the penalty was also charged for the intercept law's interventions |
| dwell, p_leave | steps spent in a stall; the per-decision probability that a sampled action moves the body (the dropped geometric law predicted dwell 10 / p_leave) |
| zero-motion request | a nominal with speed below the threshold: the policy asked not to move |
| Base, Obs, Hor, ObsHor, Free, Aware | the six policy families (see the setting section) |
| G, H, L | greedy evader, wall-hugging evader, learned attacker (draft 1 used A) |
| Experiment k, S-round, B1 to B4, D-number | the paper's numbering (revision 1: 1 to 10; crosswalk Table S5); the internal round; the four pre-registered rounds of S26; an entry in the decision log |
| pinned | a different metric in the code (backup-tier steps); never used in the paper |

## Submission checklist and open items

**Done in revision 1 (D165, per `review/response_round1.md`).**
- [x] R1 the controller described correctly, with "decision" defined and Fig. 2 added.
- [x] R2 the learning claim rescoped with N1's numbers; title, abstract, C2, §5.3 and the conclusion follow.
- [x] R3 Remark 2 replaced by Remark 4; 629 and 346 both stated correctly; the mechanism figure removed.
- [x] R4 Lemma 1 with W = diag(150, 1), credited; Proposition 2 with hypotheses (i)–(iii); C1 reworded.
- [x] R5 Proposition 3(b) under H1–H3 with T_turn and the displacement term; (c) claims no liveness.
- [x] R6 every registered verdict stated; Table S1; §6 retitled.
- [x] R7 abstract and §1 phrasings match the body.
- [x] R8 the three asides removed.
- [x] R9 figures redrawn at printed width; new Fig. 1; four figures moved to the supplement.
- [x] R10 Experiment 10 reconciled and compressed.
- [x] R11 selection, recipe and the "every family uses Aware's recipe" disclosure.
- [x] R12 multiplicity sentence, Newcombe intervals, MDD.
- [x] R13 observation and reward paragraph.
- [x] R14 protocol vocabulary removed; the attacker renamed L.
- [x] R15 the minor numerics fixed.
- [x] R16 the predictive filter's conservativeness and the scope statement.
- [x] N1 branch attribution run (D164), now Experiment 3.

**Still open.**
- [ ] Re-review of revision 1 (the editorial decision planned it for Nov 7–8).
- [ ] R17: corresponding email and repository URL (user, Nov 10); licence; README commit-hash note.
- [ ] Mirror the crosswalk and the new figure scripts to the release repository safe-and-stuck.
- [ ] Rotate the niro-2 password before any public link (user, Nov 10).
- [ ] Decide on citing BIG-CBF (arXiv:2609.14343) and DMPS (arXiv:2405.13863).
- [ ] Decide on the phrase "silent failure" in the abstract.
- [ ] Optional rounds, all not run: N2 (rotation-accepting baseline, under an hour), O2 (Hor with the recovery, 3 cells), the escape-tail split (needs a replay), a recurrent arm (not advised before the deadline).
- [ ] Commit the untracked files: `main.bbl`, `main.blg`, this guide, the worklogs.
- [ ] Final checks: 10 pages plus references, supplement compiled, AI statement, go/no-go Nov 10, submit by Nov 13 23:59 AoE.

## Where everything lives

Paths are relative to /home/niro-1/Codes/safe-rl-pe unless stated.

| What | Where |
| --- | --- |
| Draft, revision 1 (LaTeX, PDF, bib) | docs/paper_b/draft/main.tex, main.pdf, references.bib |
| Draft 1 as reviewed | docs/paper_b/draft/v1_round1/main_v1.tex, supplement_v1.tex |
| Response and revision facts | docs/paper_b/review/response_round1.md, revision_facts_2026-11-03.md |
| Supplement | docs/paper_b/draft/supplement.tex, supplement.pdf |
| Long formats (old labels, not updated) | docs/paper_b/draft/long/v1_full_length.pdf (16 pp), v2_fourteen_pages.pdf (14 pp) |
| Writer's pre-commitment and self-review | docs/paper_b/draft/ars/phase4a_precommitment.md, phase4b_self_review.md |
| Figures and the numbers drawn | docs/paper_b/figures/R0_game, R1_loop.tex, R2_stall, R4_recovery, R5_dissoc with figure_data_rev.json (revision 1); F4, F6, F7, F8 in the supplement with figure_data.json, figure_data_s26.json |
| Figure scripts | scripts/paper_b_game_figure.py (Fig. 1), scripts/paper_b_figures_rev.py (revision 1); scripts/paper_b_figures.py, paper_b_figures_s26.py (supplement F figures) |
| Skeleton (binding outline, user decisions) | docs/paper_b/skeleton_2026-11-02.md |
| Plan, venue facts, status log | docs/pathway/paper_b_l4dc_plan_2026-11-01.md |
| Evidence audit (claims to records, status classes) | docs/pathway/paper_b_evidence_audit_2026-11-02.md |
| Pre-registration of Experiments 7 to 9 | docs/pathway/paper_b_prereg_rounds_2026-11-02.md |
| Pre-registrations of Experiments 1 to 6 | docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md, shield_remedy_prereg_2026-09-15.md, loop_mechanism_prereg_2026-09-20.md, loop_fixes_prereg_2026-09-20.md, loop_fixes_b_prereg_2026-09-21.md, boundary_condition_prereg_2026-09-23.md, planner_intercept_prereg_2026-10-06.md |
| Verified citations | docs/references/paper_b_citations_verified_2026-11-01.md |
| Novelty check notes | docs/literature-review/novelty_A_liveness, _B_absorbing, _C_remedies (2026-09-22) |
| Rebuild plan after the withdrawal | docs/rebuild_plan_2026-09-19.md |
| Review panel | docs/paper_b/review/: 00_reviewer_configuration.md, phase1_*.md and review_*.md (five each), editorial_decision.md (synthesis, scoring matrix, roadmap R1 to R17, N1 to N3, O1 to O4, the six questions to answer) |
| L4DC template | docs/paper_b/template/ |
| Decision log | docs/decisions.md (D125 to D165 for this paper) |
| Worklogs | docs/worklogs/2026-11-01_S132.md (drafting; D163's panel review is logged under S132), 2026-11-02_S135.md (this guide) |
| Archived records | results/derisk/: s4_*, s5_*, s15_*, s16_*, s17_*, s18_*, s19_*, s26_b1_*, s26_b2_*, s27_n1_* (round JSON, per-cell JSON, PNG, GIF directories) |
| Checkpoints | results/derisk/ckpts_<identifier>/ (final checkpoint of every evaluated arm; attacker ckpts_ATK4-45/ATK4-45_10000000.zip) |
| Discovery logs (log-only, not in results) | /home/niro-1/s14_logs/escape_diag*.log, v1_qp.log, v3_stoch.log; /home/niro-1/s16_logs/rec_anatomy.log |
| Runners | scripts/derisk_s4_shield.py, derisk_s5_remedy.py, derisk_s15_mechanism.py, derisk_s16_fixes.py, derisk_s17_fixes_b.py, derisk_s18_boundary.py, derisk_s19_planner.py, paper_b_geometry.py, paper_b_filter_sweep.py, launch_paper_b.sh |
| Code | pe2/env.py, pe2/shield.py (CBFShield, RecoveryShieldV2, PredictiveShield, PerturbationShield), pe2/structured.py, pe2/scripted.py, pe2/evade.py, pe2/metrics.py, pe2/recorder.py, pe2/config.py |
| Tests | tests/pe2/ (1 117 pass in the release subset) |
| Release repository | /home/niro-1/Codes/safe-and-stuck (341 MB, one commit 5485369); github.com/shoaib6174/safe-and-stuck, private |
| This repository | github.com/shoaib6174/safe-rl-pe, branch searching-teacher (public; password in history) |
| Training host | niro-2 (100.71.2.97); nothing left to train for this paper |
| This guide | this file (current, revision 1); the Claude Doc https://claude.ai/artifact/AezeLtmxguMQHUEAxJsq9G holds the 2 Nov version |

## Self-test

Answer without looking, then check.

1. **What does the paper's title say about the controller, and why the change?** "Learning-Enabled Controllers", because the law, not the head, owns most stalls in trained families (N1).
2. **What are the two parts of the pursuer's controller?** A scripted intercept law with one remembered point, and a memoryless PPO search head whose commands are held 10 steps.
3. **What is a decision?** One head command plus the law steps that follow it: one PPO transition, about 16 env steps in training.
4. **Fig. 1's episode?** Free vs the greedy evader, seed 1 740 044. Stall at the top wall from step 8 for 1 192 steps; with the recovery, 7 firings and capture at step 507.
5. **Why was draft 1's Fig. 1 replaced?** N1 attributed all 1 106 of its stalled steps to the law chasing a visible evader.
6. **The filter's cost on Free?** 0.728 → 0.396, with 0.540 deadlock episodes.
7. **How many stalled states were sampled, and what share are traps at heading? Position traps?** 262; 98.1 %; 0 %.
8. **Lemma 1 in one sentence?** The weighted projection has zero speed iff the weighted difference to a zero-speed allowed command is a non-negative combination of the active constraint normals.
9. **Proposition 2's three hypotheses?** The filter executes zero; the evader stays out of view or fixed in view; the commanding part's internal state is unchanged.
10. **Why do law stalls lie outside learning?** They happen inside one decision, so the learner takes no decision while they last.
11. **The law's share of the trained families' deadlock episodes against L?** 1 747 of 2 002, 87 %.
12. **Free's and Aware's head shares against L?** 0.76 and 0.35.
13. **Capture after escape vs no escape?** 0.299 vs 0.042.
14. **P1's registered verdict and direction?** FAIL; "the signal was there and learning still missed it; optimisation, not credit assignment."
15. **What does Remark 4 say?** On the learner's per-decision clock, escaping is worth 0.29 (return-to-go) and A₀·E[γ^T] = 0.18, not 0.0092. It is a post-hoc recomputation.
16. **What was 629?** The mean steps from stall entry to episode end after an escape, not the delay to capture (346).
17. **The recovery's candidate order?** (v_max, 0), (½v_max, 0), the controller's command, (0, ω_max), (0, −ω_max); then the larger filtered rotation if above 0.1 rad/s.
18. **Deadlock with the recovery for N ≤ 20?** At most 0.022 on every policy tested.
19. **Why does the recovery offer forward first?** The rotation-only recovery cleared deadlock by spinning (414 928 firings on one seed), while a straight command would have moved at every sampled pose.
20. **What did PERT-20 achieve, and what is its limit?** It cleared 0 of 5. It could not rotate, and a rotation-accepting version was not run.
21. **The three dissociations?** The tenfold stall penalty (Exp. 5), the correction penalty (Exp. 9), the planner (Exp. 10).
22. **What is the disclosure about Base?** Every trained family uses Aware's recipe: three filter features plus a 10⁻³ stall penalty.

