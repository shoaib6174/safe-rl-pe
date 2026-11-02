# Paper B — round N1 (internal S27): which component commands the absorbing stall?

**Pre-registration, written 2026-11-03 before any N1 replay ran and before any N1 number existed.**
Origin: panel review D162/D163 (`docs/paper_b/review/editorial_decision.md`), Devil's Advocate CRITICAL C1 and roadmap item N1; user decision 2026-11-03 ("continue with path 1"). Evaluation only: no training, no new policy, no new opponent.

## 0. What is known before this round
- The structured pursuer is a composite (`pe2/structured.py`): `InterceptLaw.act` chases while the evader is visible (it updates a remembered `last_seen` point), drives toward `last_seen` while the evader is not visible and the point is farther than `LAST_SEEN_TOL`, and returns None otherwise; only then does the PPO head (branch `"search"`, decision held 10 steps) command. The paper's §3 ("a fixed intercept law when it is visible; both are memoryless") is wrong on both counts and is corrected regardless of this round's outcome.
- Experiment 7 (S26 B1, stream 3 000 000+, N = 100 per arm, against the learned attacker) classified stalled STATES: law-branch share 1.00 / 0.97 / 1.00 for BASE-60/61/62 and 0.57 for RC19 (deterministic arm). The Devil's Advocate reported from `s26_b1_cells` that every law-branch stalled state had the evader out of view. Not re-derived here; used only as the basis of the predictions in §6.
- The records behind the paper's headline deadlock numbers carry no branch: S15 stall entries have no branch field; S17 gate summaries have no law/head split and their stored traces no branch flag; S26 B2 cells likewise.

## 1. Question
For the deadlock episodes behind the paper's headline numbers, which component commanded the step at which the stall became a deadlock: the intercept law chasing a visible evader, the intercept law driving to a remembered point, or the learned head?

## 2. Cells (literal; 40 cells × 500 episodes)
**Set A — Experiment 4 (S17) bare arms**, archive `results/derisk/s17_arm_{ARM}.json` → `final.gate[{OPP}]`:
ARM ∈ {BASE-60, BASE-61, BASE-62, OBS-60, OBS-61, OBS-62, GAM-60, GAM-61, GAM-62, OBSGAM-60, OBSGAM-61, OBSGAM-62} × OPP ∈ {ATK4-45, HUG7, GREEDY} → 36 cells, named `{ARM}_{OPP}`.
**Set B — Experiment 8 (S26 B2) plain cells of the two discovery policies**, archive `results/derisk/s26_b2_cells/{P}_{OPP}_PLAIN.json`:
P ∈ {STR14, RC19} × OPP ∈ {ATK4-45, GREEDY} → 4 cells, named `{P}_{OPP}`.
No recovery arm, no PERT, no predictive filter: deadlock there is at most 0.022 and N1 is about where the deadlock lives.

## 3. Procedure
Each cell is replayed through `pe2.recorder.BranchTaggedSearchMDP` (bit-identical to `SearchMDP`, `tests/pe2/test_recorder.py`) with the archived gate's configuration:
- Set A: `EnvConfig()`, `hold = pe2.structured.HOLD`, `shield_obs = True`, `escape_obs` = the family's (`OBS`, `OBSGAM` True; `BASE`, `GAM` False), a FRESH plain `CBFShield(cfg)` per episode (S17 wrapped it in an observation-free logger), opponent from S17's `gate_opponent_factory`, model = S17's `source_ckpt(arm)` loaded on CPU, `play_episode(model.predict, 1 740 000 + i)` for i = 0…499.
- Set B: the same with S26 B2's `POLICIES[p]` (`shield_obs` STR14 False, RC19 True; `escape_obs` False), `shield_factory("PLAIN", cfg)(seed)`, `opponent_factory`, seeds 1 740 000 + i.
- `play_episode` acts with `deterministic=True`. Device CPU. Code frozen in a git worktree at the commit that contains this document and the runner; `results/` linked to the one record root; the frozen commit, checkpoint sha256 and this document's sha256 go into every record's provenance.

## 4. Definitions
- **Component of a step**: `head` if the recorder's `branch == "search"`; `chase` if `branch == "law"` and `visible` (the evader was visible in the observation the command was computed from); `last_seen` if `branch == "law"` and not `visible`.
- **Stall run**: a maximal run of consecutive steps whose recorder `stalled` flag is True (the flag `ShieldMetrics` counts). **K-run**: a stall run of length ≥ K = 40 (`DEADLOCK_STEPS`). **Censored**: the run is still open at the episode's last step.
- **First-K attribution of a deadlock episode**: the component of the step at index `start + K − 1` of the episode's FIRST K-run (the step at which the episode became a deadlock episode; the convention of S26 amendment A2).
- **Crossing K-run**: a K-run whose steps are not all one component.

## 5. Measures (per cell; families pooled over their three seeds)
- **M1 (primary)**: counts and shares of the cell's deadlock episodes by first-K component {chase, last_seen, head}.
- M2: stalled steps by component (counts, shares).
- M3: all K-runs by component at K; number of crossing K-runs.
- M4: deadlock episodes with ANY K-run reaching K on the head.
- M5: steps and immobile steps by component (context for M2).
For the decision rule: **h_F** = head share of first-K attributions pooled over family F's three seeds against ATK4-45 (deadlock episodes of passing cells only); h_Free, h_Aware likewise for STR14 and RC19.

## 6. Predictions (basis: §0)
- **P-N1-1**: BASE against ATK4-45 — last_seen share of first-K attributions ≥ 0.80 on each of the three seeds.
- **P-N1-2**: every Set A cell against ATK4-45 — chase share of first-K attributions ≤ 0.05.
- **P-N1-3**: RC19 against ATK4-45 — head share of first-K attributions ≥ 0.20.
No prediction for OBS, GAM, OBSGAM, STR14, or the HUG7 and GREEDY cells: measured.

## 7. Construct gates (blocking)
- **CG-N1-1 reproduction (per cell)**: the replayed per-episode capture indicators and deadlock indicators equal the archived `capture_ind` and `deadlock_ind` element-wise (500/500), and the replayed `deadlock_episode_rate`, `stall_step_rate`, `immobile_step_rate` (from `pe2.metrics.summarize_shield`) and capture rate equal the archived values exactly. A failing cell is UNDETERMINED: the runner writes the failure and no attribution field, and `--summarize` excludes it with the count stated. A harness error may be fixed only by an amendment written before any attribution of that cell has been produced.
- **CG-N1-2 consistency (per episode)**: the deadlock flag recomputed from the recorder's per-step `stalled` flags (longest run ≥ K) equals the environment's own `log["shield"]["deadlock"]`; any mismatch fails the cell.
- **CG-N1-3 watched**: before `--summarize`, GIFs of the designated episodes are rendered with the component and the remembered point in the frame title, and watched; a watch note is appended to this document as "CG-N1-3 WATCHED" with what was seen. Designated (literal rule): in each of BASE-61_ATK4-45, OBSGAM-61_ATK4-45, STR14_ATK4-45 and RC19_ATK4-45, the lowest-seed deadlock episode whose first-K component is last_seen, and the lowest-seed one whose first-K component is head (if any): at most 8 GIFs. `--summarize` refuses until the watch note exists.
- **CG-N1-4 provenance**: every Set A model's sha256 equals the archived `source_sha256`; every Set B model's sha256 equals the archived `ckpt_sha256`; device CPU.

## 8. Decision rules: what the paper says (registered now)
Let h_max = max over the four Set A families of h_F (families with fewer than two passing ATK4-45 cells are excluded and named).
- **D-1, h_max < 0.10**: in the trained families the absorbing stall is, in at least 90 % of deadlock episodes, the scripted law driving to a remembered point. The paper replaces "learning does not remove it" by "the learned remedies change where the law is handed the approach, not the stall"; C2 is rescoped to the head's share with the Free and Aware numbers beside it; a title without "Learned Policies" is proposed to the user. Registered sentence: "In {x} of {n} deadlock episodes of the four trained families the stalled command came from the intercept law driving to a remembered point; the learned head commanded it in {y}."
- **D-2, 0.10 ≤ h_max < 0.50**: both mechanisms are reported with their shares; C2 is restricted to the head-branch stalls and states their share; the title is put to the user.
- **D-3, h_max ≥ 0.50**: for the families with h_F ≥ 0.50 the deadlock is a learned-head phenomenon and the learning framing stands for them with the number; the law's share is reported for the others.
- **In every case**: §3 is corrected; the component split is printed beside each headline deadlock number; Remark 2 and C2 are rescoped per roadmap R2/R3; P-N1-1…3 are reported with their verdicts, failures included.

## 9. Stop rule and scope
No cell is re-run or re-cut after its record exists (the runner refuses). Amendments are dated and appended before the number they could affect. This round does not train, does not change any policy, filter, opponent or seed, and does not add cells beyond §2.

## 10. Outputs
`scripts/paper_b_branch_attribution.py` (runner, `--cells`, `--gifs`, `--summarize`), `tests/pe2/test_paper_b_branch_attribution.py`; records `results/derisk/s27_n1_cells/{CELL}.json`, GIFs `results/derisk/s27_n1_gifs/`, round `results/derisk/s27_n1_round.json`; decision entry in `docs/decisions.md`.

## Amendment A1 (2026-11-03 03:25, before any OBSGAM or Set B attribution existed)
The first launch (frozen 27d9c71) wrote 27 records, all PASS on CG-N1-1, CG-N1-2 and CG-N1-4 (BASE, OBS, GAM × 3 seeds × 3 opponents). Every OBSGAM cell then raised `KeyError: 'source_sha256'` in `archived()` after its replay and BEFORE its record was written: the OBSGAM arms trained in S17 and their records archive no hash of their own checkpoint. Each chain stopped there, so the four Set B cells queued after them never ran. No attribution of these 13 cells was written, printed or read. Fix (harness only, CG-N1-4's reference, nothing else): for an S17 arm without `source_sha256`, the archived hash is the `source_sha256` of `s17_arm_RECF-{arm}.json`, the recovery arm that gated the same checkpoint file. The 13 missing cells are run once from a new frozen commit; the 27 existing records stand (stop rule).

## CG-N1-3 WATCHED (2026-11-03, before --summarize)
All 40 cells PASS CG-N1-1, CG-N1-2 and CG-N1-4 (27 from 27d9c71, 13 from 268a1ee). The eight designated episodes were rendered (`results/derisk/s27_n1_gifs/`; every replay ended with the recorded outcome) and watched through four-frame contact sheets (30 steps before the stall, its start, its 40th step, 200 steps later). Seen:
- last_seen (BASE-61 ep 1740005, OBSGAM-61 ep 1740002, STR14 ep 1740001, RC19 ep 1740001): the law chases the attacker round an obstacle, loses sight, and drives at the remembered point on the far side; the pursuer stops against that obstacle at clearance 0.000 with the filter intervening, the remembered point unchanged in the title, for the rest of the episode while the attacker circles. The label says last_seen from the K-th step on in all four.
- head (BASE-61 ep 1740124, OBSGAM-61 ep 1740080, STR14 ep 1740002, RC19 ep 1740000): the learned head holds the pursuer against the ARENA BOUNDARY (a wall or a corner) in three, and against a small obstacle after the law handed off at its remembered point in RC19's; the label says head throughout the run. In STR14's the stall lasts 853 steps, in the others 40–84 steps.
- No frame contradicts its label. Observation, not a registered measure: head stalls in these episodes are at walls and corners, law stalls at obstacles between the pursuer and a remembered point.
