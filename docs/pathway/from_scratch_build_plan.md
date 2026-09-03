# Build Plan v3 — Shielded Games: Safe Deep-RL Pursuit-Evasion on Real Ground Robots

**Status**: Plan v3 (2026-09-03, S120, D86). Supersedes v2 (archived: `archive/from_scratch_build_plan_v2_2026-08-23.md`) and v1 (`pathway_B_safe_exploitability_bounded_PE.md`).
**Execution state inherited from S119**: Steps 0–5 COMPLETE and gate-signed (D65–D84); Step 6 shield IN PROGRESS (`pe2/shield.py`, gates G1–G4 pre-registered in the S119 worklog, D85; one shield test currently failing). **v3 does not re-do any of it.** Step numbering 0–6 is kept as executed; new steps are added after.
**Direction (user decision 2026-09-03)**: shielded-game *study* as the scientific core + safety-first *system* as the apparatus and deployable claim + **two real TurtleBot3s** as validation.
**Reading**: `docs/reference/reading_list_shielded_games.md` (D88) — **Tier 0 blocks Step 3b and Step 8**.
**Novelty basis**: `docs/reference/novelty_audit_2026-09-03.md` — CEP May-2026 predicts (analytically, full-obs) that projection shields shift adversarial learning to "conservative safe equilibria"; CBF-RL (Oct 2025) studies shield-in-training single-agent only. Open: the shift measured in deep self-play, strategic shield exploitation, a remedy under co-adaptation, hard-safe deep-RL PE on real robots.
**Method rule (unchanged)**: every step ships with an automated verify **and** a visual gate signed off in the worklog; gates follow D83 (paired McNemar primary, mandatory negative control, power-based N, criteria pre-registered before running).

---

## 1. Paper claims

**Core question**: *What does a hard safety filter do to an adversarial learning game, and can two real robots play that game safely?*

| # | Claim | Type | Produced by |
|---|---|---|---|
| R1 | A projection shield measurably shifts the learned outcome of a PE game (balance, NashConv, regime taxonomy), with the direction predicted exactly in a small game | Study | Steps 3b, 8 (or 8-alt) |
| R2 | A learning opponent can (or provably cannot) weaponize the shield — drive it into infeasibility / high intervention — and by how much | Study | Step 9 |
| R3 | Shield-aware training (shield-in-training + learned feasibility constraint + margin shaping) recovers performance vs shield-at-deploy while keeping the guarantee | Constructive | Step 10 |
| R4 | The full system runs on two TurtleBot3s with zero collisions, including against the shield-attacker, with sim-predicted rates within CI. **Wording is load-bearing (D90)**: first *deep-RL, onboard-sensing / partially-observable, model-free-policy* hard-safe PE on real ground robots, as a *seeded statistical campaign*. A bare "first hard-safe RL PE on real ground robots" is refuted by Jia et al. 2026, who have two UGVs — but with motion-capture full state, off-board computation, a critic-ADP policy, and one demonstration run | System / hardware | Step 12 — **in scope, hardware confirmed (D87)** |

**Non-claims**: no Nash convergence; no population opponent sampling (D37); no differentiable QP in the policy (D31); no CR≈0.5 as equilibrium (D35). Safety guarantee is stated *conditionally on QP feasibility*; infeasibility is measured, never hidden. Exploitability claims use an opponent **family**, never one scripted opponent (D83 flag).

## 2. Pre-registered hypotheses (fixed before the step that tests them runs)

- **H1 (shift)**: shielded vs unshielded training differs in ≥1 of {balanced-seed rate, NashConv, regime distribution} with CI-separated effect at N≥9 seeds; Step 3b states the predicted direction in advance.
- **H2 (exploitation)**: a best-response evader rewarded for pursuer shield events raises the pursuer's infeasibility rate vs the scripted `ObstacleHuggingEvader` probe (paired McNemar, N by power).
- **H3 (remedy)**: shield-aware training lowers deploy-time intervention and infeasibility vs shield-at-deploy, capture within 10pp (≥4 seeds each).
- **H4 (transfer)**: real-robot violation count = 0 over the hardware campaign; real capture rate within the sim 95% CI on the same paired init set.
- **H5 (baselines)**: reward-penalty and Lagrangian learners reach lower violation only at lower capture, or violate; the shield dominates the violation axis at equal capture.

## 3. Metrics (all seeded, paired, frozen-opponent)

Capture rate + Wilson CI · NashConv (sum of both roles' approx-BR gains; BR budget stated; ≥3 BR seeds) · balanced-seed rate · regime class · violation steps (env contact flags, S119 semantics) · **infeasibility rate** · intervention rate · min barrier value per episode · fallback-tier histogram · LOS-break rate · scenario-stratified CR (D70 suite) · (hardware) collisions, e-stops, sim-vs-real gap, control-loop latency.

## 4. Assumptions (what changed vs v2)

1. `pe2/` as built; SB3 for RL; old code read-only. **Learner findings from S119 are binding**: LayerNorm critics (D80), fixed ent_coef=1e-5 (D81), imitation-first (D75), RL fine-tune of the clone refuted at 1M with single-step SAC (D82); the RL retry is PPO+GAE (D84). The "proven chassis" parameters from the root codebase (ent 0.03 etc.) do **not** transfer to pe2 — D80/D81 supersede them.
2. **Sim scale stays 10×10 / v 1.0 / ω 2.84 / dt 0.05** (gated in S119). Hardware uses a **similarity transform**, not a re-train: unicycle kinematics are invariant under length scale λ and time scale τ with v' = vλ/τ, ω' = ω/τ. With λ = 0.44, τ = 2: arena 4.4 m, v_max 0.22 m/s (TB3 Burger), ω_max 1.42 rad/s (< TB3's 2.84), sensing range 8 m → 3.52 m (≈ LDS-01 3.5 m), control period 0.1 s (10 Hz). Verified by an invariance test in Step 12a (same policy, rescaled env, bit-equal normalized trajectories).
3. **Sensing on hardware = the same software-masked 90° cone** applied to the 360° LiDAR, range-capped at 3.5 m, LOS by the real occluders. Sim and real see the same information.
4. Hardware: 2× TurtleBot3 Burger (user-owned, phase4 spec §2.10), RPi4, LDS-01, ROS 2; arena with 2 fixed padded obstacles; AMCL on a mapped arena; overhead camera or mocap for **ground-truth logging only**; e-stop.
5. Training on niro-1. Eval bitwise reproducible; GPU training seed-cohort reproducible (D25).

## 5. Steps

Format: build → **verify** → visual gate → exit / kill. Steps 0–5: COMPLETE (S119). Gate rules: D83.

### Step 6 — Shield (IN PROGRESS from S119)
As pre-registered in the S119 worklog: VCP-DCBF γ=0.2, obstacle + boundary only, never inter-agent on pursuer, exact 2-D QP (D85), 3-tier fallback; gates G1 (0 violations, N=1000), G2 (capture intact, paired McNemar), G3 (`ObstacleHuggingEvader` probe, 0 violations), G4 (negative control ≥5% unshielded contact episodes); `step_06_shield.gif`.
**v3 additions (log before running)**: record infeasibility rate, min-h, fallback-tier histogram in the gate JSON (needed by R2); add a **robust-margin arm** (constraint radii + δ for estimated-state use on hardware) as a config flag, gated identically — **REQUIRED, not optional (D87)**, with δ derived from measurement-robust CBF theory (Cosner/Ames arXiv 2104.14030; Dean/Taylor CoRL 2020) rather than hand-tuned. **First action**: fix the failing `test_backup_turns_away_from_nearest_danger`, then run G1–G4.

### Step 3b — Exact-game twin (new, CPU, 2–3 days; any time before Step 8)
Gridworld PE from Experiment 1 + hazard cells + a shield (projection of moves away from hazards). Exact BR dynamics / Nash for shielded vs unshielded; measure value and balance shift; one proposition on when the shielded game's value differs from the original.
**Verify**: exact BR converges (as Exp 1); shift reproducible. **Visual gate** `step_03b_exact_shift.png`. **Output**: the H1 direction prediction, written into §2 before Step 8.

### Step 7 — Learned policies for both roles (GPU, 2–4 weeks) — *prerequisite, not contribution*
Per D84/D79: BC a PPO policy onto FLp demos → PPO+GAE fine-tune at γ=0.999 (pursuer); belief-conditioned or recurrent evader (D79: the evader teacher is non-Markov, so the evader is learned here with belief input; target = the D70 occluded/shadow scenarios where the clone scores 0). External baselines on the same learner: unconstrained, reward-penalty, Lagrangian (thin subclass), shield-at-deploy, shield-in-training (CBF-RL replication). 4 seeds each; power-based N per D83.
**Verify / H5**: violation-vs-capture frontier; shield-in-training reproduces CBF-RL's decreasing-intervention pattern; BarrierNet pattern (>40% intervention, stalled learning) = alarm.
**Visual gate** `step_07_frontier.png` + `step_07_agents.gif`.
**Exit**: a gated learned pursuer AND a gated learned evader (each beats the D76 battery, paired McNemar). **Fallback if the evader cannot be learned**: proceed to **Step 8-alt** with an opponent family.

### Step 8 — Shielded vs unshielded self-play (GPU, 3–5 weeks) — *R1 / H1*
Alternating self-play on the learned policies (freeze→unfreeze curriculum, forced role switch, bilateral rollback per lessons §3/§5b; visibility/search rewards only if pre-registered). Arms: no shield; shield both roles; shield pursuer only (asymmetry confound). Pilot 4 seeds/arm; NashConv probe every K episodes; regime annotation.
**Verify**: H1 at pilot scale; direction vs Step 3b prediction. **Visual gate** `step_08_shift.png` + `step_08_regimes.gif`.
**Step 8-alt (fallback)**: no self-play; R1 becomes "the shield shifts the best-response landscape" — for each opponent in a family (scripted greedy/FOV-limited/obstacle-hugging, speed ratios 1.0–1.4, BC clones), compare shielded vs unshielded learned BR on capture and NashConv-proxy. Weaker but still first-of-kind; exact-game twin carries the equilibrium statement.

### Step 9 — Shield-attacker (GPU, 1–2 weeks) — *R2 / H2*
Best-response evader vs the frozen shielded pursuer with reward += bonus per pursuer intervention / infeasibility step (and a control BR without the bonus). Compare pursuer infeasibility, min-h, fallback usage vs the G3 scripted probe. Both shield arms (true-state, robust-margin).
**Verify**: H2 paired test; attack strategies classified from trajectories. **Visual gate** `step_09_attack.gif` — the exploit must be *intelligible* (which corner, which bait). **Figure 1 candidate.**

### Step 10 — Shield-aware training remedy (GPU, 2–3 weeks) — *R3 / H3*
Shield-in-training + learned feasibility constraint (Xiao et al. ECC 2023: sample, classify, add H(z)≥0 to the QP, resample ×3) + CBF-margin shaping; in self-play if Step 8 ran, else on the Step 8-alt family. Re-run the Step 9 attacker against the remedied agent.
**Verify**: H3 at ≥4 seeds; attacker gain reduced. **Visual gate** `step_10_remedy.png` + `step_10_remedy_vs_attack.gif`.

### Step 11 — Cohort, ablations, figures (GPU, 3–4 weeks)
Main cohort N≥9 per key arm (unshielded / shielded / remedied); ablations ≥4 seeds: no-occlusion, pursuer-only shield, no-feasibility-constraint, no-margin-shaping. Every number traces to seeded JSON + config hash; figures regenerate in CI. **Visual gate** figure slate + `step_11_gallery.gif`.

### Step 12 — Hardware (6–10 weeks, overlaps Step 11) — *R4 / H4*
- **12a Pipeline**: similarity-transform invariance test (assumption 2); ONNX export (parity rtol 1e-5); ROS 2 nodes (LiDAR cluster → opponent detection with static-map subtraction, software cone mask, belief, policy, shield); C++/OSQP or ported exact-2-D DCBF at 10 Hz; AMCL; GP disturbance residual + cold-start protocol (phase4 §2.6–2.7); e-stop. **Verify**: hardware-in-the-loop on recorded bags; latency histogram; C++ shield bit-matches Python on a test vector set. **Visual gate** `step_12a_hil.mp4`.
- **12b Gazebo**: two TB3s, mapped arena, domain randomization (phase4 §2.2). **Gate**: 100 Gazebo duels, 0 collisions, CR within sim CI.
- **12c Real duels**: (i) shield validation — scripted aggressive pursuer + shield, 30 runs, 0 collisions; (ii) learned vs learned, paired inits (marked floor positions), N≥30 per matchup; (iii) **shield-attacker on hardware**, N≥20 — headline video; (iv) sim-vs-real table. **Verify**: H4. **Visual gate** `step_12c_real.mp4` with overlaid barrier values. **0 collisions or the claim is downgraded to "N collisions, all in infeasible states", reported honestly.**

## 6. Dependency graph

```
[0–5 done] → 6 → 7 → 8 (or 8-alt) → 9 → 10 → 11
                 3b ──────────↗ (prediction before 8)
                 6 → 12a → 12b → 12c (12c after 10)
```

## 7. Budget

| Resource | Estimate | Priority |
|---|---|---|
| GPU (niro-1) | Step 7: ~24 runs; Steps 8–10: ~30 long runs; Step 11: ~40 long runs | 7 → 8 → 9 → 10 → 11; ablations are the first cut |
| Hardware | 2× TB3 Burger + spare batteries, 2 padded obstacles, ~5×5 m mapped floor, overhead camera, e-stop | Confirm arena + camera before 12a |
| Wall time | Sim core (6–11): ~4–5 months; hardware (12): 2–3 months overlapping | No deadline stated |

## 8. Risk register

| Risk | Mitigation |
|---|---|
| ~~CEP 2026 already has deep-RL / PO experiments~~ | **CLOSED 2026-09-03 (D90)**: read in full. Critic-only ADP, control-affine, full state, single run, off-board motion capture. R1 survives and strengthens — their equilibrium-shift claim is asserted as motivation and never measured, so ours is the first test of it, and their formulation is our comparator. R4 reworded (see §1). R2/R3 unaffected |
| R3 remedy does not remove the exploit | Expected, not fatal: adversarial training did NOT remove the KataGo vulnerability (Wang ICML 2023; Tseng arXiv 2406.12843). Report reduction with CIs, never "fixed" |
| Shield creates standoffs / deadlock | Predicted by CBF deadlock theory (Grover et al.); detect via force-equilibrium diagnostics from Step 6; report timeout rate per arm |
| Learned evader unattainable (D79 headroom ≈ 22%) | Step 8-alt opponent family; R1 stated on the BR landscape; exact twin carries the equilibrium statement |
| PPO-GAE also fails (D84) | Algorithm-family-independent negative result on long-horizon credit assignment; BC clones remain the policies; Step 8-alt |
| Shield-in-training stalls (BarrierNet pattern) | Step 7 alarm; R3 falls back to feasibility constraint + margin shaping |
| H1 null | Projection shields are benign for this game class, contra the CEP prediction; exact twin explains why |
| H2 null | Positive robustness result; attacks tried are reported |
| Localization error breaks the shield on hardware | Robust-margin arm + GP residual; ground truth logged for audit |
| LiDAR clustering confuses obstacle vs opponent | Static-map subtraction; validated in 12b |
| Real collisions | Low speed, padding, e-stop; report honestly, conditioned on feasibility |
| Compute overrun | §7 priority order |

## 9. Hard rules

1. Seed everything (L1.1, D66); eval bitwise, training seed-cohort reproducible.
2. Frozen + seeded paired eval only; training-log CR is a health metric (D55).
3. NashConv with stated BR budget and ≥3 BR seeds; opponent family, never one opponent (D83).
4. Gates per D83: paired McNemar primary, mandatory negative control, power-based N, pre-registered criteria; ≥4 seeds per claim, N≥9 for rate claims.
5. Baselines see truth; learned policies are sensor-limited (L1.5).
6. Best-checkpoint, Q-monitoring, NaN guards; LN critics + fixed ent_coef for any SAC (D80/D81).
7. Inter-agent CBF never constrains pursuer→evader.
8. No populations as mechanism (L2.2); no differentiable QP in the gradient path (L5.1); no blind tuning (L2.3); RL libraries not rewritten.
9. Safety claims conditional on feasibility; infeasibility always reported.
10. Hardware ground truth is for logging only, never for the policies or the shield arm under test.
11. No step closes without its visual gate signed in the worklog; never kill runs for disk (D16).
