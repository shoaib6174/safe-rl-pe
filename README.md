# Multi-Agent Adversarial RL for Partially Observable Pursuit-Evasion

A production-grade research codebase for training adversarial policies in 1v1 pursuit-evasion under strict partial observability. Built with **Stable-Baselines3**, **Gymnasium**, and **PyTorch**, featuring a custom self-play orchestrator, masking curriculum, best-response exploitability testing, and distributed training across GPU workstations.

> **Note:** This is an active research project targeting CoRL 2026. The codebase reflects 3+ months of iterative development, 40+ training runs, and systematic ablation studies.

---

## What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| **Deep RL Engineering** | Custom SAC/PPO training loops, opponent pool with PFSP sampling, curriculum annealing |
| **Multi-Agent Systems** | Adversarial self-play with alternating freeze, forced switching, collapse rollback |
| **Reproducible Research** | 736+ unit tests, structured worklogs, decision logs, experiment tracking |
| **Distributed Training** | Multi-machine coordination (niro-1 + niro-2), rsync-based artifact sync |
| **Data Analysis** | Trajectory analysis, best-response exploitability testing, statistical aggregation |
| **Visualization** | GPU-accelerated GIF generation, TensorBoard logging, sensing overlay plots |
| **System Design** | Modular env wrappers, adapter pattern for opponent policies, clean CLI interfaces |

---

## Architecture

```
safe-rl-pe/
├── envs/                      # Gymnasium environments + wrappers
│   ├── pursuit_evasion_env.py # Core PE env: unicycle dynamics, obstacles, FOV sensing
│   ├── partial_obs_wrapper.py # Dict obs (obs_history + lidar + state)
│   ├── opponent_adapter.py    # PartialObsOpponentAdapter: frozen opponent with own sensors
│   └── wrappers.py            # SingleAgentPEWrapper, curriculum control
├── training/                  # Training orchestration
│   ├── amsdrl.py             # AMS-DRL self-play: alternate freeze, PFSP pool, health monitor
│   └── opponent_pool.py      # Prioritized Fictitious Self-Play sampling
├── scripts/                   # Runnable experiments
│   ├── train_amsdrl.py       # Main self-play launcher (40+ runs executed)
│   ├── train_br_sac.py       # Best-response exploitability tester
│   ├── analyze_exploitability.py # Verdict computation (L1-L4 / H1-H5 classifier)
│   └── visualize_both_learned_gif_gpu.py # GPU-rendered trajectory analysis
├── tests/                     # 736+ tests across envs, training, safety
├── docs/                      # Research docs, decisions, worklogs
└── paper/                     # CoRL 2026 submission (LaTeX)
```

---

## Key Technical Contributions

### 1. Stabilization Mechanisms for PO Self-Play

Standard self-play collapses under partial observability (our baseline: **0% escape rate**). We developed three mechanisms that extend stable co-evolution past 10M steps:

| Mechanism | Purpose | Impact |
|-----------|---------|--------|
| **Visibility Reward** (`w_vis=0.2`) | Explicit LOS-maintenance signal for pursuer | Without it: **collapse to 0%** |
| **Forced Switching** (every 2M steps) | Breaks freeze-sticking where one role trains indefinitely | Without it: **pursuer-dominant 64% CR** |
| **Search Staleness Reward** (`w_search=0.0001`) | Grid-based coverage bonus for systematic search | Under evaluation (ABL-3) |
| **Masking Curriculum** | Anneals `p_full_obs` from 1.0 → 0.0 over 5M steps | Without it: under evaluation (ABL-4) |

### 2. Best-Response Exploitability Testing

Most self-play papers claim "convergence" without verification. We implemented **fresh-init BR testing**:

- Train a new SAC agent from scratch against a **frozen** opponent snapshot
- Compare BR performance against the self-play baseline
- Classify into L1-L4 (per-seed) and H1-H5 (cohort) hypotheses

**Result (H3):** The self-play fixed point is **not** a Nash equilibrium. A BR evader achieves lower capture rates than the self-play evader (gap +0.08–0.10 > ε=0.05), revealing a curriculum-induced exploitability hole.

### 3. Modular, Tested Infrastructure

```python
# Example: BR trainer usage
python scripts/train_br_sac.py \
    --frozen_opponent_path results/BR_frozen/s48/evader \
    --frozen_role evader \
    --total_steps 1_500_000 \
    --eval_freq 250_000 \
    --seed 148 \
    --output_dir results/BR_1
```

- **TDD throughout:** Every new script includes unit tests (`test_br_sac_loading.py`, `test_exploitability_analyzer.py`)
- **Frozen-invariance test:** SHA-256 hash verification ensures opponent params don't mutate during training
- **Smoke tests:** 50K-step validation before launching 1.5M-step production runs

---

## Results Snapshot

| Experiment | Seeds | Key Finding |
|------------|-------|-------------|
| SP17b (full config) | 48, 49, 50 | Stable co-evolution to M2550; curriculum-end drift post `p_full=0` |
| ABL-1 (no vis reward) | 60 | **Total collapse** — CR → 0% |
| ABL-2 (no forced switch) | 61 | **Freeze-sticking** — pursuer dominates at 64% CR |
| ABL-3 (no search) | 62 | Running |
| ABL-4 (no curriculum) | 63 | Running |
| BR-1..BR-4 | 48, 49 | H3 verdict: evader-side exploitability, replicated |
| BR-5..BR-6 | 50 | Running (3rd seed replication) |

---

## Running the Code

### Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Training

```bash
# Self-play with full config
./venv/bin/python scripts/train_amsdrl.py \
    --algorithm sac --fov_angle 90 --sensing_radius 8.0 \
    --combined_masking --w_vis_pursuer 0.2 --w_search 0.0001 \
    --masking_curriculum --force_switch_steps 2000000 \
    --seed 100 --output results/SP17b_s100

# Best-response test
./venv/bin/python scripts/train_br_sac.py \
    --frozen_opponent_path results/BR_frozen/s48/pursuer \
    --frozen_role pursuer --seed 248 --output_dir results/BR_2
```

### Testing

```bash
./venv/bin/python -m pytest tests/ -v
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| RL Framework | Stable-Baselines3 (SAC, PPO) |
| Environment | Gymnasium + custom Dict obs space |
| Compute | PyTorch 2.10, CUDA 12.8, RTX 5090 |
| Visualization | matplotlib, pygame, GPU-accelerated GIF |
| Experiment Tracking | TensorBoard, `history.json` schema |
| Testing | pytest, SHA-256 param hashing |
| Infra | Multi-machine SSH + rsync, nohup background jobs |

---

## Research Artifacts

- **36 papers** reviewed across safe RL, CBFs, self-play, and sim-to-real
- **40+ training runs** logged with structured `history.json` outputs
- **Decision log** (`docs/decisions.md`): 36 architectural decisions with rationale
- **Session worklogs** (`docs/worklogs/`): Per-session detailed records
- **CoRL 2026 paper** (`paper/main.tex`): In submission

---

## About the Author

This project was built as part of a research initiative on safe deep RL for robotics. The codebase reflects industry-grade practices applied to academic research: modular architecture, comprehensive testing, distributed training infrastructure, and rigorous empirical validation.

For questions or collaboration, please open an issue.
