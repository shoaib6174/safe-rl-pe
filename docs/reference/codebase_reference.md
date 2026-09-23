# Codebase Reference — What Exists, What Works, What to Reuse

**Purpose**: Module-level inventory of this repository for the Pathway B system paper, with explicit reuse verdicts. Companion: `docs/reference/lessons_learned.md` (the *why*), `docs/pathway/` (the *plan*).

**Verification basis**: (a) structured code inventory (2026-08-23, S118); (b) full root test suite run: **750 passed / 68 failed / 3 skipped** (38 min, venv Python, plugins isolated); (c) repro test suite: **48/48 unit tests pass**, 3 smoke tests fail on a stale checkpoint-path assertion (training itself completes — cosmetic test bug, not code rot).

**The 68 root failures share ONE root cause**: `qpth` is not installed and absent from `requirements.txt`. They are confined to 4 files, all in the **abandoned BarrierNet path** (`test_barriernet_actor.py`, `test_barriernet_training.py`, `test_differentiable_qp.py`, BarrierNet branches of `test_comparison_framework.py`). **Everything else passes.**

---

## 1. The Big Picture: Two Codebases

| Codebase | Role | Algorithms | Dynamics | Status |
|---|---|---|---|---|
| **Root** (`envs/`, `safety/`, `training/`, `agents/`, …) | Pathway A system | PPO / SAC / RecurrentPPO (SB3) | **Unicycle (nonholonomic)** | Active, tested, reusable |
| **`repro_gonultas_isler_2024/`** | GI 2025 reproduction + negative-result experiments | MADDPG (+EMA, Pop-FSP, PBT, shared-policy) | Bicycle pursuer / point-mass evader; symmetric point-mass mode | Complete, archived results; code is the evidence base for paper 1 |

**Key realization for Pathway B**: the root codebase already has nonholonomic dynamics, FOV/LOS sensing, belief tracking, a CBF safety stack, and a proven self-play chassis — the "six-element" system is an *integration + convergence-mechanism* project on top of working parts, not a from-zero build.

---

## 2. Root Codebase — Environment (`envs/`)

| File | What it provides | Reuse verdict |
|---|---|---|
| `dynamics.py:24` | Unicycle (differential-drive) Euler integrator + wall clipping + obstacle-collision projection | **Reuse** — element #3 (nonholonomic) already satisfied |
| `sensors.py` | `FOVSensor` (directional cone, :16), `LidarSensor` (36-ray 360°, :73), `WallSegment` line obstacles (:264) | **Reuse** — FOV + LOS + occlusion (element #4) |
| `observations.py:44` | Normalized 14-D base vector + 2·K obstacle features + `los_visible` flag | **Reuse** |
| `pursuit_evasion_env.py:30` | Core 2-agent Gymnasium env: obstacles, prep phase, partial/FOV/radius/combined masking, randomized obstacle counts | **Reuse** — but add seedable-reset hard requirement (lesson L1.1) |
| `partial_obs_wrapper.py:28` | Dict obs (`obs_history`, `lidar`, `state`, `belief`); exposes true state for safety filters | **Reuse** |
| `dcbf_action_wrapper.py:18` | **Post-hoc DCBF filter on actions using TRUE state while policy sees partial obs** | **Reuse — this is the shield architecture** (CBF well-posed under PO: filter sees map/true state, policy sees sensors) |
| `belief_tracker.py:13` | Particle-filter opponent belief, 6-D statistics | **Evaluate vs BiMDN** — two belief implementations exist |
| `rewards.py:177` | `RewardComputer` (distance shaping, capture/timeout, visibility evader reward, PBRS obstacle seeking) + `SafetyRewardComputer` (CBF margin rewards) | **Reuse with care** — reward pitfalls documented (lessons §4) |
| `search_staleness.py:16` | Grid-cell staleness reward (pursuer search skill, D24) | **Reuse** |
| `rnd.py:82` | RND intrinsic exploration | Optional |
| `layouts.py:17` | Corridor / L-shaped / warehouse wall layouts | **Reuse for generalization evals** |
| `wrappers.py` | `SingleAgentPEWrapper` (:16), `FixedSpeedWrapper` (:147, omega-only action) | Reuse as needed |
| `discrete_action_wrapper.py`, `rendering.py` | DQN actions; pygame rendering with FOV/CBF/lidar/belief overlays | Utilities |

## 3. Root Codebase — Safety (`safety/`)

| File | What it provides | Reuse verdict |
|---|---|---|
| `vcp_cbf.py` | The real CBF stack: `vcp_cbf_obstacle` (:77), `vcp_cbf_boundary` (:125), `vcp_cbf_collision` (**inter-robot**, :190), `vcp_cbf_wall` (:246); `VCPCBFFilter` SLSQP QP (:594); discrete-time `dcbf_filter_action` (:713) | **Reuse — tests pass.** Design note: inter-robot collision constraint exists but must NOT be enabled pursuer→evader (capture requires contact) |
| `infeasibility.py:349` | `SafeActionResolver`: 3-tier fallback (exact QP → hierarchical relaxation → backup controller) + `FeasibilityClassifier` SVM | **Reuse** |
| `beta_distribution.py` / `beta_policy.py` / `safe_beta_policy.py` | SB3 Beta policy with per-step CBF-derived action bounds | Available; Beta path was Phase 2 exploration — decide vs post-hoc filter |
| `differentiable_qp.py:33` | `DifferentiableVCPCBFQP` (qpth) for BarrierNet | **Abandoned (D31); untestable** (qpth not installed). Do not revive without reinstalling qpth + re-audit |
| `cbf.py`, `safety_filter.py` | Stubs | Ignore |

## 4. Root Codebase — Training (`training/`)

| File | What it provides | Reuse verdict |
|---|---|---|
| `amsdrl.py` (2962 lines) | `AMSDRLSelfPlay` (:770) — **the proven self-play chassis**: PPO/SAC/RecurrentPPO, partial/full obs, BiMDN, DCBF, curriculum, opponent pools, EWC, RND, health callbacks, rollback. Integration points: PO env stack (:57), BiMDN extractor (:246), DCBF on pursuer (:931), pool sampling (:1996), EWC hooks (:2059) | **Reuse as the chassis** (lessons §3). It is large and knob-heavy — Pathway B should configure, not extend, it |
| `self_play.py:49` | Vanilla alternating self-play (no safety/PO) | Reference only |
| `safe_self_play.py` | `SafeSelfPlayConfig` (:42), SB3-independent `safe_self_play_rollout` (:162) with `SafeActionResolver` | Reuse pattern for safety metrics |
| `opponent_pool.py:23` | Reservoir/FIFO pool + PFSP-lite sampling | **Reuse — but see lesson L2.2**: population sampling is ruled out as the *convergence mechanism*; pool remains useful as an evaluation/robustness tool |
| `curriculum.py` | `CurriculumManager` (6 levels, :35), `SmoothCurriculumManager` (:258) | **Reuse** |
| `ewc.py:25` | Fisher-diagonal EWC | Drop unless role-switching (D22) |
| `health_monitor.py:15`, `selfplay_callbacks.py`, `checkpoint_manager.py:30` | Collapse detection + rollback, entropy floor, Elo-vs-baselines, best/rolling checkpoints | **Reuse — infrastructure lessons L1.6/L2.4** |
| `ne_tools.py:16` | `compute_ne_gap`, `analyze_convergence` | **Reuse for exploitability claims** |
| `safety_metrics.py:22` | Violation / intervention / backup / CBF-margin tracker | **Reuse — shield-intervention rate is a headline metric** |
| `safe_ppo.py` | `CBFBoundsComputer` (:48); `CBFSafetyCallback._on_step` is **empty** (:164) | SB3-native safe PPO never wired — don't assume it works |
| `baselines.py:11` | Random / greedy pursuer / greedy evader (full-obs — lesson L1.5) | **Reuse** |

## 5. Root Codebase — Agents, Evaluation, Config, Scripts

- `agents/`: `bimdn.py:28` (BiMDN belief encoder — proven in both codebases), `encoders.py:22` (encoder factory: BiMDN/LSTM/MLP), `partial_obs_policy.py:31` (SB3 multi-branch extractor). `barriernet_*.py` — abandoned (D31), untestable (qpth).
- `evaluation/comparison_framework.py` — BarrierNet-era comparison harness; SB3 branch usable.
- `conf/` — Hydra: env (20×20, dt=0.05, v_max=1.0, omega_max=2.84), PPO, `safety/cbf.yaml` (VCP-CBF params, `enabled: false` default), self_play (10 phases × 200K, conv threshold 0.10), ablations A–E.
- `scripts/` entry points: `train.py`, `train_self_play.py`, **`analyze_exploitability.py:109`** (BR exploitability verdicts — headline measurement), `evaluate_tournament.py` (wraps repro tournament), `run_ablation.py`.
- Tests: 30 files / ~720 tests; all non-BarrierNet pass (verified 2026-08-23).

---

## 6. Repro Codebase (`repro_gonultas_isler_2024/`)

*(Filled from inventory + S115–S117 session knowledge.)*

| Component | What it provides | Reuse verdict |
|---|---|---|
| `training/train.py:24` | MADDPG self-play loop: pretrain-vs-greedy, EMA, Pop-FSP, curriculum, AMP; `episode/capture_rate` logging (:423) | Paper-1 evidence; **not** the Pathway B chassis (MADDPG was the repro target's algorithm) |
| `training/vectorized_env.py:8` | N parallel envs, frame stacking, frame skip (:171), BiMDN sample collection | Reference for vectorization patterns |
| `agents/maddpg.py:13` | Twin centralized critics, clipped Q, AMP; exploration noise `select_actions_batch` (:110) | Paper-1 evidence |
| `agents/shared_maddpg.py`, `training/train_shared.py` | Role-conditioned shared actor (D45 failure) | Archived negative result |
| `training/train_pbt.py:18` | PBT with tournament + balance-fitness mutation (D46) | Archived negative result (0/3) |
| `training/population_manager.py:9` | Pop-FSP snapshot store, recency-biased sampling | Archived negative result (0/17) |
| `envs/pursuit_evasion_env.py:6` | Symmetric point-mass + bicycle/point-mass modes; **unseeded resets at :74** (lesson L1.1) | Do not extend; the flaw that motivated seeded protocol |
| RNG state (whole codebase) | **`train.py` never seeds numpy/torch/random from `config["seed"]`** (only `train_pbt.py:287-292` does); `vectorized_env.py:84` resets unseeded; exploration noise (`maddpg.py:111,135`), replay sampling (`replay_buffer.py:75`), population sampling (`population_manager.py:48,70`) all use global RNG | **Training runs are not reproducible** — only evaluations made deterministic post-hoc (D57). Pathway B must seed everything at every entry point |
| `evaluation/evaluate.py:33` | Frozen-opponent tournament eval (`run_evaluation`, `run_full_tournament`) | **Reuse protocol** (D55); wrapped by root `scripts/evaluate_tournament.py` |
| `reeval_seeded_cohort.py` | **Seeded paired N=500 eval (`env.reset(seed=700000+i)`) — the reproducibility standard (D57)** | **Reuse pattern in Pathway B eval harness** |
| `diagnose_eval_discrepancy.py` | Single-variable mechanism diagnosis (noise/frame-skip/opponent-drift conditions) | Reference methodology |
| Results dirs | `results_symmetric/` (canonical: `eval_seeds_9_33_final.json`), `results_pbt/`, `results_pop_fsp*/`, `results_ema*/`, `results_gi_vanilla_20k/` | Paper-1 data; frozen |
| Tests | 48/48 pass; 3 smoke tests stale (checkpoint-path assertion — cosmetic) | — |

---

## 7. Reuse Map for the Six Elements

| Paper element | Status in repo | Gap to close |
|---|---|---|
| 1. Safe DRL (CBF, learned policies) | VCP-CBF + DCBF wrapper + SafeActionResolver + safety metrics **exist and pass tests** | Shield-during-training integration + intervention-rate study; never enable inter-agent constraint on pursuer |
| 2. Adversarial 1v1 zero-sum PE | Core env + self-play chassis exist | None — proven |
| 3. Nonholonomic ground robots | Unicycle dynamics **already implemented** | None in sim; sim-to-real (Phase 4 spec: TurtleBot3, Isaac Lab, ONNX) untouched |
| 4. Partial observability (FOV+LOS+occlusion) | FOV/lidar/LOS masking + Dict PO wrapper + BiMDN/particle-filter beliefs exist | Choose belief stack; own-pose lesson enforced |
| 5. Self-play convergence | **The open research problem.** Chassis stabilizes (SP17b CR≈0.51) but BR-exploitable (D35); populations ruled out (D37–D39) | NFSP-style average-policy + measured exploitability — the actual research contribution |
| 6. Model-free | PPO/SAC are model-free policies; CBF filter is model-based | Frame as "model-free policy + model-based safety filter" |

**Net assessment**: elements 2, 3, 4, 6 are engineering-complete; element 1 is built but needs the training-integration study; element 5 is the research.
