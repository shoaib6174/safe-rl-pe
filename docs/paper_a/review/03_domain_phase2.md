## Dimension Scores

### D1: methodology_rigor
score: warn

This matches three Phase-1 warn triggers: code not available, few seeds with uncertainty pooled over episodes rather than seeds, and a filter described only partly.

- **Code not available.** The code and per-cell records cannot be found at review time (§III-G footnote: "Repository URL to be added at camera-ready").
- **Single seeds behind the headline claim.** The 4×4 meta-game (§IV-C, Figure 3) and the whole margin sweep (§IV-D, Table 4, Figure 4) each use one selected seed per learned player:
  - Sel-2;
  - Sel-R-1, the best of three by the stage-2 selection rule;
  - Evs-1, the lowest-capture of three.

  Their uncertainty is over episodes only, while the headline is the "filter share" of 0.046 (norm: Henderson et al. 2018; Agarwal et al. 2021). The seed spread matters here. Putting Evs-2 or Evs-3 in place of Evs-1, with the same four pursuer rows (Table 3 plus the stored Sel-R-1 rows), gives a restricted value of 0.192 or 0.156 instead of 0.138. That spread of 0.054 is larger than the effect being reported.
- **Filter partly described.** Several parts of the filter and simulator are not given:
  - the form of the soft obstacle row;
  - the relaxed-tier normalisation;
  - the backup rule;
  - the recovery's fallback;
  - the projection-based collision model;
  - the float32 command path (items 3 and 9 below).

  The game's behaviour therefore cannot be re-implemented from the text. The safety rows can be, so the block trigger ("safety result cannot be re-implemented") does not fire.

### D2: domain_accuracy
score: block

This matches the Phase-1 block trigger "forward invariance asserted for a discrete-time or tiered filter without the feasibility/fallback-invariance hypotheses and then relied on for a 'guaranteed safe' claim".

- **The claim.** The abstract says "We prove that the filter keeps the pursuer contact-free from any contact-free start". C4 claims "a short invariance proposition for the filter as implemented". §III-B presents the 101 000-episode count as the Proposition's "empirical counterpart".
- **What the proof covers.** Proposition 1 proves invariance of H_δ only from starts in H_δ, and its proof states "the backup tier is unreachable from H".
- **Gap (i): starts outside H_δ.** The reset samples starts against the margin-free contact predicate (pe2/env.py:330–343). On the evaluation stream:
  - 23, 38 and 72 of the 500 starts lie outside H_δ at δ = 0.05, 0.10 and 0.20;
  - on 10, 20 and 48 of them the first filtered step runs in the backup tier.

  About 2 100 of the 24 000 δ > 0 sweep episodes are therefore outside the proof, and the fallback they exercise is never analysed.
- **Gap (ii): the applied successor.** The premise that the hard rows are "evaluated on the same successor the simulator applies" is false in the implemented loop. The filter's float64 command is re-encoded as float32 before env.step (pe2/structured.py:257–259). In a stress test the displayed inequality failed by up to 3.2×10⁻¹⁰ m.
- **Secondary findings (warn level).**
  - "Value", "equilibrium" and "non-exploitable" are used beyond the 4×4 restricted game (§I, §IV-D, §V, §VII).
  - §V invokes a "known asymmetry of pursuit–evasion among obstacles" without a citation.
  - The §II novelty sentence is contradicted by §IV-D.

Both blocking defects can be fixed, and item 1 supplies a corrected statement and proof. I do not judge the empirical safety record to be wrong.

### D3: argumentative_coherence
score: warn

This matches the Phase-1 warn triggers "alternative explanations left undiscussed", "limitations stated but contradicted by framing elsewhere" and "abstract … generalising".

- **(a) Novelty sentence against the results.** §II says the evader "learns to exploit a model-based filter's obstacle margin, which is what our evader does". §IV-D finds instead that the orbit "beats the unfiltered pursuers on geometry alone" and stays minimax with no filter.
- **(b) Untested attributions.** §IV-C and §V conclude that "the pursuer's library, not its learning, is binding". §V also states that "speed ratio, obstacle geometry and the libraries set most of the value". Neither the speed ratio nor the geometry was varied in the two-sided game (§VI says so), and the classical explanation is not discussed (item 5).
- **(c) The δ = 0.20 reading.** "Changes which evader is minimax" is read without noting that the frozen pursuers' planner and hug-circle geometry ignore δ (item 4).
- **(d) Abstract wider than C1.** The abstract's "beats the best fixed intercept law on held-out evaders" drops C1's qualifier "on two of three seeds".

No block trigger fires: an unfiltered control exists, and both players' selectors are trained and evaluated against several opponents.

### D4: cross_disciplinary_relevance
score: warn

This matches the Phase-1 warn triggers "control-theoretic and RL notation mixed without a bridge" and "definitions present but delayed or scattered".

- γ means the DCBF gain (0.2, §III-B) and also the PPO discount (0.99, §III-E).
- "Value", "minimax" and "saddle point" are used from the abstract onwards but are defined only in §IV-C, and there only by citation.
- §I asserts that the partially observed pursuit–evasion game "has a well-defined value", which is stronger than a game-theory reader would accept.
- A multi-agent-systems reader is never told that "no filter" keeps obstacles impenetrable through the simulator's projection.

No block trigger fires. The zero-contact count is presented as the counterpart of a proof, not as a proof, and §VI limits "minimax" to the pool.

### D5: writing_and_structure
score: warn

This matches the Phase-1 warn triggers "notation drift … a policy or opponent renamed between sections" and "figures without uncertainty".

- **Name drift.** Figures 1, 3 and 4 label players "Selector, retrained", "Lead rule", "Hugger", "Learned selector" and "Full-state greedy". The text and tables call the same players Sel-R-1, Lead, Hug, Evs-1 and Full.
- **Notation.** γ is used for two different quantities.
- **Reporting.** Table 4 mixes p-values with "n.s.". Figure 4b (value curve) and Figure 3 (cells) show no uncertainty.
- **Placeholders.** "Email: [to confirm]" and the repository footnote remain.

The manuscript is otherwise reviewable: 8 pages, IEEE two-column, and every cited section and table exists.

## Failure Condition Checks

### F1
fired: true. D2, a mandatory dimension, scores block.

### F2
fired: true. D1 warn, D2 block and D3 warn: all three mandatory dimensions are at warn or worse.

### F3
fired: false. D4, the only high-priority dimension, scores warn.

### F0
fired: false. No mandatory dimension scores pass.

## Review Body

### Manuscript information
- **Title:** Learned Tactic Selection for Both Players of a Safety-Filtered Pursuit–Evasion Game
- **Manuscript:** Paper A draft (`docs/paper_a/draft/main.tex`, compiled `main.pdf`, 8 pp.), target IC-FMAS 2027
- **Review date:** 2026-11-03
- **Review round:** Round 1, sprint-contract Phase 2

### Reviewer information
- **Role:** Peer Reviewer 2 (Domain).
- **Identity:** a control theorist working on control barrier functions, discrete-time safety filters and pursuit–evasion games, who follows safe-RL work.
- **Focus:** whether the formal safety statement and the filter description match the code; whether game-theoretic terms are used accurately; whether prior work is represented fairly; and whether the domain contribution is real or incremental.

### Overall assessment
- **Recommendation:** Major Revision. The contract maps F1 to reject-or-major-revision, and the blocking defect can be fixed.
- **Confidence:** 4. CBFs, discrete-time filters and pursuit–evasion are within my expertise; the RL training details only partly.

**Summary assessment.**

*What the paper does.* Both players of a 1v1 unicycle pursuit–evasion game among two disc obstacles are maskable-PPO selectors over five hand-written tactics. Every pursuer command passes through a three-tier discrete-time CBF filter with a forward-offering recovery. The results are:
- The pursuer's selector beats fixed, flat and structured baselines.
- The evader's selector learns an obstacle orbit that holds every pursuer to a capture rate of 0.09–0.22.
- In the 4×4 empirical game the learned evader is minimax.
- A pre-registered margin sweep with an unfiltered control refutes the authors' own reading that the filter sets the value.

*What is strong.* From the control side, the filter is built with unusual care. Its hard rows are the simulator's own contact predicates, the one-step check is exact, and u = 0 is always a candidate. I verified the core of Proposition 1 line by line, and the restricted-game numbers reproduce exactly from the stored matrices.

*The main defect.* The formal safety claim is wider than its proof:
- At δ > 0, about 9 % of the sweep's episodes start outside the Proposition's set, and the backup tier the proof calls "unreachable" is exercised. The abstract's "from any contact-free start" is therefore not covered.
- The "as implemented" premise fails because the filter output passes through a float32 round trip before it reaches the simulator.

Both can be fixed, and I give a corrected statement below.

*Secondary issues.*
- **Interpretation of C3.** Three things weaken it: the unfiltered control keeps obstacles impenetrable, the frozen pursuers' geometry ignores δ, and classical theory predicts the orbit result in advance.
- **Terminology and positioning.** Game-theoretic terms drift beyond the restricted game, and the positioning needs work.

*Contribution.* Real but incremental, and empirical rather than theoretical.

### Strengths

**S1. A filter built so that a proof is possible and checkable.**
- The hard rows are the simulator's contact predicates evaluated on a transcription of its Euler step (`pe2/shield.py:296–338`; `pe2/env.py:40–55`).
- Condition (1) is checked exactly, not linearised: `sat = H1 >= (1-γ)·H0` (`shield.py:404`).
- u = 0 is placed in every candidate set (`shield.py:199`).

This is the bridge most safe-RL papers lack: they quote a continuous-time CBF condition and implement a linearised QP. The key step of the proof also holds in floating point, since 0.8·h ≤ h for h ≥ 0.

**S2. A pre-registered falsification of the authors' own hypothesis, with an unfiltered control.** §IV-D and Table 4 report that H-VAL-1 and H-VAL-2 failed and change the claim accordingly. Replacing "certified" with a scoped proposition (C4) is the right instinct.

**S3. Restricted-game bookkeeping that reproduces.**
- *4×4 matrix* (`results/derisk/s24_2_meta_game.json`): pure maxmin = pure minmax = 0.138, and an independent LP also gives 0.138. Exploitabilities are 0.028/0.052/0.002/0 for the pursuers and 0.482/0.256/0.644/0 for the evaders, all matching §IV-C.
- *Margin sweep* (`results/derisk/s25_round.json`, per-setting games): values 0.184/0.138/0.136/0.162/0.058, with Hug in the evader support at δ = 0.20, all matching §IV-D.

**S4. The zero-sum construction is right for the evaluated payoff.**
- The environment reward is antisymmetric, +1 for capture and −1 for timeout (`env.py:291–292`).
- The selectors' rewards are the environment's terminal reward and its negation.
- There is no shaping and no contact penalty (`selector_mdp.py` reading 6; `evader_selector_mdp.py` reading 1).

**S5. Plain limitations.** §VI states the frozen-player sweep, the 4×4 pool, the information asymmetry and the single speed ratio without hedging.

### Weaknesses
- **W1 (Critical).** Proposition 1 does not cover the experiments it is used to certify: starts outside H_δ at δ > 0, a backup tier that is reached, and a float32 command path. See items 1–2.
- **W2 (Major).** The C3 refutation is partly predictable and partly confounded. The unfiltered control keeps non-penetration through projection; the frozen pursuers' geometry ignores δ; and a faster evader orbiting an obstacle is uncatchable in classical analysis. See items 3–5.
- **W3 (Major).** Game-theoretic terms drift from the restricted game to "the game". The pursuer side of the saddle is within sampling noise, exploitability is reported more precisely than the data allow, and the filter share is measured on one evader seed. See items 6–7.
- **W4 (Major).** Positioning: the novelty sentence is contradicted by the paper's own results, two relevant clusters of prior work are missing, and two neighbours identified by the authors' own check are not cited. See item 8.
- **W5 (Major).** The filter as described differs from the code in ways that matter for re-implementation. See item 9.

### Code check of Proposition 1 (what I verified, file:line)

**Confirmed:**
- u = 0 is always a candidate: `cands = [clip_box(v_nom, w_nom), [0.0, 0.0]]` (`pe2/shield.py:199`).
- The successor is a vectorised transcription of the unclipped Euler step (`shield.py:323–338` against `env.py:52–55`). At u = 0 the position is unchanged bit for bit, so every hard row is unchanged. The heading is re-wrapped (`shield.py:337`), which the hard rows do not depend on.
- Condition (1) is applied row-wise with an exact comparison, `sat = H1 >= (1-γ)·H0` (`shield.py:404`); γ = 0.2, the direction is correct, and γ lies in (0, 1].
- Tier selection (`shield.py:408–418`) works as stated: exact selects from candidates satisfying every row, relaxed from candidates satisfying every hard row. From H_δ the hard-feasible pool is non-empty, so the backup tier is indeed unreachable from H_δ.
- The recovery executes only `CBFShield.filter` outputs at the current state (`shield.py:746–769`).
- The margin tightens the barrier, consistent with the text: radii are inflated (`shield.py:279, 317`) and wall limits shrunk (`shield.py:265–266`).
- At δ = 0 the hard rows equal the environment's contact predicates (`env.py:88–95, 104–109`). When they hold on the Euler successor, the clip and projection are inactive, so the applied successor equals the predicted one.

**Gaps:**
- **(A) Scope.** Starts are sampled against the margin-free predicate (`env.py:330–343`, line 338). I ran `env.reset` plus one `filter` call on seeds 2 400 000–2 400 499, with no training:
  - starts outside H_δ: 23, 38 and 72 of 500 at δ = 0.05, 0.10 and 0.20;
  - first filtered step in the backup tier: 10, 20 and 48 of those.
- **(B) Applied successor.** The filter's float64 output is re-encoded as float32 (`pe2/structured.py:257–259`). It is decoded in `SingleAgentWrapper._scale_action` (`pe2/exploitability.py:145–148`). The tactic game's step path reaches this code through `pe2/recorder.py:219–230` and `pe2/structured.py:238`. In a stress test (item 2), the displayed inequality failed on 3 of 7 488 steps; the same loop without the round trip had 0 failures in 7 516 steps.
- **(C) Description mismatches.**
  - The soft obstacle row is a squared-distance barrier in m² (`shield.py:106, 320`).
  - The backup rule (`shield.py:419–426`) is not described.
  - The recovery counter is keyed to the filtered *policy* command (`shield.py:734–740`).

  See item 9.

### Actionable items

#### CRITICAL

**1. Proposition 1 does not cover starts outside H_δ, and the backup tier is reached.**

*What is wrong.* The Proposition's set is H_δ, and the proof needs x₀ ∈ H_δ ("the backup tier is unreachable from H"). Yet the abstract claims contact-freedom "from any contact-free start", and §III-B counts all 48 δ > 0 sweep cells in the Proposition's "empirical counterpart". The reset uses `o["radius"] + c.agent_radius` and an arena inset by the agent radius only (`env.py:210–213, 330–343`). Initial poses depend only on the seed, so about 2 128 of the 24 000 δ > 0 episodes start outside H_δ, and the backup tier (`shield.py:419–426`) is exercised there without being described or analysed.

*Where.* Abstract; C4; §III-B (Proposition 1, its proof and the paragraph after it); §IV-D.

*Fix.* State the stronger result that the code satisfies; it makes the abstract true.

> **Proposition 1′.** Fix δ ≥ 0 and let h⁰ᵢ denote the margin-0 hard rows, i.e. the simulator's contact predicates. If h⁰ᵢ(x_k) ≥ 0 for every hard row, then the command executed by the margin-δ filter, in any tier and with or without the recovery, satisfies h⁰ᵢ(x_{k+1}) ≥ (1−γ)·h⁰ᵢ(x_k) for every hard row.

*Proof.*
1. Every hard row satisfies hᵟᵢ = h⁰ᵢ − δ, so hᵟᵢ(x) ≥ −δ.
2. *Exact and relaxed tiers.* These give hᵟᵢ(x⁺) ≥ (1−γ)hᵟᵢ(x). Hence h⁰ᵢ(x⁺) ≥ (1−γ)h⁰ᵢ(x) + γδ.
3. *Backup tier.*
   - The backup maximises m(u) = minᵢ[hᵟᵢ(f(x,u)) − (1−γ)hᵟᵢ(x)] over a candidate set that contains u = 0; the hard rows have scale 1.
   - At u = 0, m(0) = γ·minᵢ hᵟᵢ(x) ≥ −γδ, so the selected command also has m ≥ −γδ.
   - Therefore hᵟᵢ(x⁺) ≥ (1−γ)hᵟᵢ(x) − γδ, which is h⁰ᵢ(x⁺) ≥ (1−γ)h⁰ᵢ(x).
4. *Recovery.* It executes only filter outputs, so the same bounds apply.

Keep the present H_δ invariance as a corollary for x₀ ∈ H_δ. In §IV-D, report the fraction of starts outside H_δ at each setting and a tier histogram per setting. `ShieldMetrics` already counts backup steps (`pe2/metrics.py`, `n_backup`), but the stored cell records do not carry them.

**2. "The same successor the simulator applies" is false in the implemented loop.**

*What is wrong.* The command chain is:
- `SearchMDP._step_inner` re-encodes the filter's float64 output as a float32 normalised action (`structured.py:257–259`);
- `_scale_action` decodes it before `env.step` (`exploitability.py:145–148`);
- the simulator therefore applies f(x, ũ) with |ṽ − v| ≤ 2⁻²⁶·v_max, not f(x, u).

I ran a stress test that reproduces those two lines exactly. The setup was RecoveryShieldV2 at δ = 0 over 40 seeded episodes, with a full-speed pursuer steering at the true pose of an evader parked behind an obstacle. Results:
- the applied command differed from the filter's output on 7 166 of 7 488 steps;
- the displayed inequality h_i(x_{k+1}) ≥ (1−γ)h_i(x_k) failed on 3 steps, by at most 3.2×10⁻¹⁰ m;
- without the round trip, the identical loop had 0 failures in 7 516 steps;
- there was no contact, and the minimum body clearance was 6.6 µm.

The empirical record is not threatened, but C4's "for the filter as implemented" is false as written.

*Where.* C4; the proof of Proposition 1 ("evaluated on the same successor the simulator applies"); §III-B ("evaluated exactly").

*Fix.* Either of two routes:
- Pass the float64 command straight to `env.step`. Note that the reported rounds used the quantised path; reproducing them bit for bit would require re-running.
- Or state the Proposition for the exact loop and add a remark. Quantisation moves the successor position by at most ε = 2⁻²⁶·v_max·dt ≈ 7.5×10⁻¹⁰ m, so the implemented loop guarantees h⁰ᵢ(x_{k+1}) ≥ (1−γ)h⁰ᵢ(x_k) − ε. Report the minimum body clearance over all 101 000 filtered episodes (`ShieldMetrics.min_clearance_hard_floor`) to show it stayed far above ε.

#### MAJOR

**3. What "no filter" removes in C3.**

*What is wrong.* The simulator resolves contact by clipping at walls and projecting radially out of obstacles, for both players (`env.py:58–122, 269–274`). Under NoOpShield the obstacles therefore stay impenetrable, and the pursuer slides along them without friction; the unfiltered evader always slides. §III-A never describes this collision model.

At δ = 0 the hard rows coincide with that non-penetration constraint. The filter therefore adds only two things:
- the (1−γ) limit on the rate of approach;
- the soft rows evaluated at the virtual control point.

A small "filter share" at δ = 0 (0.046) is therefore expected by construction, and calling it the cost of "the filter" invites readers to assume the obstacle constraint itself was removed.

*Where.* §III-A, C3, §IV-D, §V, abstract.

*Fix.*
- Describe the collision model in §III-A: projection, unpenalised contact for both players, and an evader that slides.
- Restate the δ = 0 contrast as "the cost of the DCBF's conservativeness relative to sliding contact".
- Present the δ sweep as the actual manipulation.

**4. The frozen pursuers' geometry ignores δ, which confounds the δ ≥ 0.10 readings.**

*What is wrong.*
- The A* planner hard-codes `MARGIN = 0.0` "so the planner's notion of 'clear' is exactly the filter's notion of 'no contact'" (`pe2/planner.py:37`). That identity fails at δ > 0.
- Counter-Orbit waypoints lie on the hug circle r + 0.15 + 0.05 = r + 0.20 m (`pe2/tactics.py:99–102, 197–226`; `pe2/scripted.py:28`). That circle is exactly the filter boundary at δ = 0.05, and lies 0.05 m and 0.15 m inside the forbidden ring at δ = 0.10 and 0.20.
- The selector does not observe δ.

Part of the δ = 0.20 result may therefore come from frozen tactics commanding into the forbidden ring rather than from the margin itself. That result is "a margin of 0.20 m stops every pursuer from approaching an obstacle-hugging evader at all", with the minimax evader switching to Hug.

*Where.* §IV-D, §V, C3, conclusion.

*Fix.*
- Say this in §IV-D and §VI.
- Report, per setting, the share of pursuer steps whose tactic waypoint lies inside the ring, and the relaxed- and backup-tier rates.
- Run an evaluation-only control at δ = 0.20 with δ-aware geometry: planner `MARGIN = δ` and hug radius r + 0.20 + δ. Use the three library pursuers against Hug and Evs-1.

**5. Classical theory predicts the orbit result, so "the library is binding" is not established.**

*What is wrong.* §V calls the result "the two-sided version of a known asymmetry of pursuit–evasion among obstacles" but cites nothing. A back-of-envelope check shows how strong that asymmetry is here.
- *Angular rate.* On the evader's orbit circle (radius r + 0.20 m) at ρ = 1.2, the evader's angular rate is 1.2/(r + 0.20). Any pursuer's is at most 1/(r + 0.15 + δ). The evader's rate is the larger whenever 0.2r + 1.2δ > 0.02. Every obstacle has r ≥ 0.3, so this holds at every δ ≥ 0, with or without the filter.
- *Turn rate.* Holding that circle needs at most 2.4 rad/s, below ω_max = 2.84 rad/s.
- *Separation.* Antipodal separation is at least 2r + 0.35 ≥ 0.95 m, well above the 0.30 m capture distance.

So a full-information evader that reaches the orbit and keeps the obstacle between the players cannot be captured by any single pursuer, filtered or not. This is the classic pillar argument (cf. Bhadauria et al. 2012, where a single equal-speed pursuer no longer suffices once obstacles are present). It has two consequences:
- It predicts C3's refutation in advance.
- It undercuts "the pursuer's library, not its learning, is binding" (§IV-C, §V). No library beats a well-played orbit, so the residual capture rate of 0.09–0.22 measures the learned evader's departures from the orbit (Orbit is only 43–77 % of its decisions, and it commits for up to 40 steps) plus the initial conditions.

*Where.* §IV-C, §V, §VII.

*Fix.*
- Cite the classical results and include the calculation; it also gives the §V attribution ("speed ratio, obstacle geometry … set most of the value") the analytic support it currently lacks.
- Reword the binding-constraint claim.
- Report how captures split between the learned evader's Orbit spans and its Flee, Hold-Hidden or transit spans.

**6. Game-theoretic terms are used beyond the restricted game, and are reported more precisely than the data allow.**

*What is wrong.*
- **(a) Existence of a value.** §I says pursuit–evasion among obstacles "is zero-sum, it has a well-defined value". For this partially observed game (90° field of view, occlusion, full-state evader), existence of a value is not established; Bhattacharya & Hutchinson 2010 study exactly this question under visibility constraints. What the paper measures is the value of a 4×4 matrix game of frozen policies, which always exists (von Neumann).
- **(b) Unqualified "value" and "equilibrium".** Q3, the §IV-D title, §V ("0.046 of value") and §VII ("converges in one evader step to an obstacle-orbit equilibrium"; "the filter's share of the value") speak of "the value" or "equilibrium" without qualification. Only §VI limits "minimax" to the pool.
- **(c) Exploitability wording.**
  - §IV-D says "non-exploitable within its library" when exploitability reaches 0.054.
  - §IV-C and §V say "within 0.05" when Lead's exploitability is 0.052.
  - Exploitability is defined only for the maximiser.
- **(d) Precision.** At p ≈ 0.14 and n = 500 the Wilson half-width is about 0.03.
  - The pursuer side of the "pure saddle" is not resolved: Sel-R-1 has 0.138 and Sel-2 0.136, one episode apart, and the saddle row switches to Sel-2 at δ = 0.10.
  - Exploitabilities such as 0.002 are below the resolution of the data.

  The robust fact is strict dominance: at δ ≤ 0.10, the learned evader's column is the minimum of every row by at least 0.19.

*Fix.*
- Define value, saddle point and exploitability for the restricted game at first use, including the evader's mirror definition.
- Replace "pure saddle point at (Sel-R-1, Evs-1)" with: "the learned evader strictly dominates the scripted evaders in this pool; the restricted value is 0.136–0.138 [CI]".
- Bootstrap the per-episode outcomes to give confidence intervals on the value and on exploitability (norm: Wellman 2006, already cited; Tuyls et al. 2020).
- Replace "converges … equilibrium" with "a fixed point of one best-response step within the libraries".
- Correct "within 0.05" and "non-exploitable", and scope or drop the §I sentence.

**7. The filter share is measured on one evader seed, and the seed spread is larger than the share.**

*What is wrong.* The meta-game and the sweep use only Evs-1, the lowest-capture seed. With Evs-2 or Evs-3 in its place, the same four pursuer rows give a restricted value of 0.192 or 0.156. Those figures come from Table 3 for Rules, Lead and Sel-2, and from the stored stage-2 records for Sel-R-1 against Evs-2/Evs-3 (0.160 and 0.156). The spread across evader seeds (0.054) exceeds the reported filter share (0.046).

*Where.* Abstract, C3, §IV-D, §V.

*Fix.* As an evaluation-only run, repeat the no-filter versus δ = 0 contrast (ideally δ = 0.20 as well) for Evs-2 and Evs-3, and report the filter share per seed with its spread (norm: Henderson et al. 2018; Agarwal et al. 2021).

**8. Positioning and representation of prior work.**

*What is wrong.*
- **(a) Novelty sentence contradicted.** §II claims novelty because the evader "learns to exploit a model-based filter's obstacle margin, which is what our evader does". §IV-D shows the orbit is minimax with no filter and "beats the unfiltered pursuers on geometry alone". The sentence is left over from the framing used before the sweep.
- **(b) Missing clusters.**
  - Classical pursuit–evasion with obstacles and visibility: Bhadauria et al. 2012; Bhattacharya & Hutchinson 2010; Guibas et al. 1999. Severity rests on the uncited "known asymmetry" claim in §V.
  - Margins in the CBF literature, i.e. measurement-robust and input-to-state-safe CBFs: Kolathaya & Ames 2019; Dean et al. 2020; Cosner et al. 2021. The paper never says why a practitioner would set δ > 0, even though the authors' own code motivates δ by localisation error and cites Cosner et al. (`shield.py:236–246`).
  - Hamilton–Jacobi reach–avoid formulations of pursuit–evasion: Mitchell, Bayen & Tomlin 2005; Margellos & Lygeros 2011. This one is advisory [FIELD-NORM UNVERIFIED].
- **(c) Neighbours the authors already identified.** Their related-work check (`docs/literature-review/tactic_selection_shield_scoop_check_2026-11-01.md`) names two papers that are not cited:
  - A *Control Engineering Practice* 2026 paper whose abstract, according to the authors' notes, claims that CBF-filtered RL in pursuit–evasion converges to conservative safe equilibria rather than the true saddle point. That is the closest prior statement to C3. I could not obtain or verify it.
  - Yeke et al. (USENIX Security 2025), on deadlock as an adversarial objective against collision avoidance.
- **(d) Mis-grouped citation.** §I cites Arnström & Teixeira 2025 for "an opponent can aim at the filter's conservativeness". That paper deactivates CBF filters through false-data injection, i.e. it attacks safety, not conservativeness. §II describes it correctly.

*Fix.* Rewrite the §II novelty sentence to agree with §IV-D. Add the missing clusters. Obtain the CEP 2026 paper and engage with it. Correct the §I grouping.

**9. The filter as described differs from the code.**

*What is wrong.*
- **(a) Soft obstacle row.** §III-B calls the soft row "the same barrier evaluated at a virtual control point". For obstacles the soft row is ‖q − c‖² − r_eff², in m² (`shield.py:106, 320`), not ‖q − c‖ − r_eff. Condition (1) is not invariant to squaring the barrier, so a re-implementation from the text would produce a different exact tier and different trajectories.
- **(b) Relaxed tier.** The worst soft-row progress it maximises is normalised by 2·r_eff for obstacle rows (`shield.py:120, 405`). This is not stated.
- **(c) Backup tier.** The backup rule is not described. It selects lexicographically: maximum worst normalised hard-row progress, then worst soft-row progress, then distance to the nominal (`shield.py:419–426`). Item 1 shows this tier is exercised.
- **(d) Which one-step map.** "f the simulator's own one-step map" is in fact the unclipped Euler map (`shield.py:323–338`). The simulator applies Euler plus a wall clip plus projection (`env.py:58–122`). The two coincide whenever the hard rows hold on the Euler successor, and the text should say so.
- **(e) Recovery.**
  - *Counter.* The counter counts steps at which the filter's output for the *policy's* command is immobile (`shield.py:734–740`), not "the executed speed". Once it reaches 20 it fires on every further such step.
  - *Fallback.* If no candidate moves, the recovery falls back to a rotation with |ω| > 0.1 rad/s (`shield.py:761–769`). The text omits this tail.
- **(f) Projections.** "Projections onto each active row": the code projects onto every row (`shield.py:200–205`). This is harmless.

*Fix.* Add one algorithm box or paragraph in §III-B covering (a)–(e), with units.

#### MINOR

10. §III-B says "capture requires contact, and the filter never prevents it". §IV-D shows the margin does prevent captures (Hug: 0.38 → 0.05). Write "the filter has no row involving the evader".
11. §III-A says "pairwise gap at least r_i + r_j + 0.5 m". The code bounds the centre distance (`env.py:319–322`), so the surface gap is at least 0.5 m. Obstacle surfaces are also kept at least 0.5 m from the walls (`env.py:314–318`). State both.
12. In the proof, "f(x,0) = x" is not exact because the heading is re-wrapped (`shield.py:337`). Write "leaves the position, the only argument of the hard rows, unchanged".
13. H-VAL-3: the registered reading was "margin-only", since Full vs Evs-1 at δ = 0 is 0.064 < 0.25 (`s25_round.json`, `verdicts.h_val_3`). §IV-D instead says Full "says nothing about information". State the registered reading, then declare the deviation and its reason: Full also fails without the filter (0.042).
14. §II says the recovery "empties [undesired equilibria] in practice". Support this with immobility or deadlock rates for every row in this paper, not only the selector's 0.03–0.06, or cite the companion study.
15. Options framing: the discount is γ = 0.99 per decision, not γ^τ per option duration as in Sutton et al. 1999. Because the two selectors decide at different times, their discounted training objectives are not exact negatives of each other, even though the evaluated payoff is zero-sum. Add one sentence.
16. Rename one of the two γ's, e.g. α for the DCBF gain.
17. Use the same player names in the figures and the text. Report p-values instead of "n.s." in Table 4. In Figure 4b, draw "no filter" as a separate marker rather than joining it to the δ axis with a line.
18. §I describes safety filters as replacing a command by "the nearest command that keeps the system inside a safe set". That does not describe discrete shields (Alshiekh et al. 2018) or the least-restrictive switching filters covered by Hsu et al. 2024. This paper's filter executes the nearest admissible candidate from a finite set; say so.
19. Abstract: add "on two of three seeds" to "beats the best fixed intercept law on held-out evaders", as C1 already does.

### Contribution to the field
The contribution is real but incremental.

- **C1** is an instance of option selection above a CBF-guarded primitive (cf. Ahmad et al. 2025, HMARL-CBF).
- **C2** is a learned rediscovery of the classical obstacle-circling escape. The evader's library already contains Orbit, and the scripted Hug evader already orbits, so what is learned is when to use it.
- **C3** is the most novel element: a pre-registered refutation with an unfiltered control. Items 3–5 and 7 weaken its interpretation.
- **C4** is an elementary stoppability argument: u = 0 is always admissible for rows that depend only on position. It is valuable once its scope is correct (items 1–2).

The paper is a careful empirical study, not an advance in CBF theory or in pursuit–evasion theory, and its framing should say so.

The manuscript's descriptions of the 2026 neighbours (Wu et al.; Acharya et al.; Wang et al.) match the authors' related-work notes. I could not verify them independently.

### Missing key references
- Bhadauria, Klein, Isler, Suri (2012). Capturing an evader in polygonal environments with obstacles: the full visibility case. *IJRR* 31(10).
- Bhattacharya, Hutchinson (2010). On the existence of Nash equilibrium for a two-player pursuit–evasion game with visibility constraints. *IJRR* 29(7).
- Guibas, Latombe, LaValle, Lin, Motwani (1999). A visibility-based pursuit–evasion problem. *IJCGA* 9(4–5).
- Mitchell, Bayen, Tomlin (2005). A time-dependent Hamilton–Jacobi formulation of reachable sets for continuous dynamic games. *IEEE TAC* 50(7).
- Margellos, Lygeros (2011). Hamilton–Jacobi formulation for reach–avoid differential games. *IEEE TAC* 56(8).
- Kolathaya, Ames (2019). Input-to-state safety with control barrier functions. *IEEE L-CSS* 3(1).
- Dean, Taylor, Cosner, Recht, Ames (2020). Guaranteeing safety of learned perception modules via measurement-robust CBFs. *CoRL*.
- Cosner et al. (2021). Measurement-robust control barrier functions: certainty in safety with uncertainty in state. *IROS* (arXiv 2104.14030, already cited in the authors' code).
- Ames, Xu, Grizzle, Tabuada (2017). Control barrier function based quadratic programs for safety critical systems. *IEEE TAC* 62(8). This is the CBF-QP source; ECC 2019 is a survey.
- Tuyls et al. (2020). Bounds and dynamics for empirical game theoretic analysis. *AAMAS journal* 34.
- Gleave et al. (2020). Adversarial policies: attacking deep reinforcement learning. *ICLR*.
- Yeke et al. (USENIX Security 2025), and the *Control Engineering Practice* 2026 paper "Safe exploration for pursuit–evasion games based on CBFs and RL". Both were identified by the authors' own related-work check and should be verified before citing.

### Questions for authors
1. In the δ > 0 sweep cells, how many episodes started outside H_δ, how many steps ran in the backup tier, and what was the minimum body clearance? The stored cell records carry none of these.
2. Does the restricted-game value with no filter and at δ = 0 change materially when Evs-2 or Evs-3 replaces Evs-1?
3. With δ-aware pursuer geometry (planner `MARGIN = δ`, hug radius r + 0.20 + δ), does the minimax evader at δ = 0.20 still switch to Hug?
4. What fraction of captures against the learned evader happen while it executes Orbit, as opposed to Flee, Hold-Hidden or transit to the orbit?

## Editorial Decision
editorial_decision=reject_or_major_revision

Rationale: F1 (severity 90) fires because D2 is block. The abstract-level safety claim is wider than Proposition 1's proof at δ > 0, and the "as implemented" premise fails because the command passes through a float32 round trip. Both can be fixed with the corrected statement in item 1, so from the domain side I lean towards major revision rather than reject.
