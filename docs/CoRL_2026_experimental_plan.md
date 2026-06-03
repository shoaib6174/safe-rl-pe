# CoRL 2026 Experimental Plan — Option A (Hybrid)

**Deadline:** May 28, 2026 (paper submission)  
**Reviews:** August 4, 2026  
**Days remaining:** ~27 days  
**Strategy:** Hybrid allocation across niro-1 (local) and niro-2 (remote)

---

## 1. Hardware Allocation

| Machine | GPU | Role | Max Processes | Why |
|---------|-----|------|---------------|-----|
| **niro-2** | RTX 5090 | Convergence cohort | 4 | Same environment for all seeds; disk on /mnt/data2 (128GB free) |
| **niro-1** | RTX 5090 | Ablations + baselines | 4 | Isolated from convergence; if issues arise, cohort is safe |

**Total parallelism:** 8 processes → 11 runs in ~1.4 batches → **~2.3 days wall time**

---

## 2. Experimental Matrix

### 2.1 Group A: Convergence Cohort (niro-2)

**Need 5 converged seeds on IDENTICAL config.**

**SP17b configuration:**
```bash
--algorithm sac
--fov_angle 90 --sensing_radius 8.0 --combined_masking
--arena_width 20 --arena_height 20 --max_steps 1200
--n_obstacles_min 0 --n_obstacles_max 3
--pursuer_v_max 1.0 --evader_v_max 1.0
--lr 3e-4 --ent_coef 0.03
--micro_phase_steps 2048 --eval_interval 150
--alternate_freeze --freeze_switch_threshold 0.65
--force_switch_steps 2000000          # 2M forced switch
--w_vis_pursuer 0.2
--w_search 0.0001 --t_stale 100
--collapse_threshold 0.1 --collapse_streak_limit 5
--pfsp
--masking_curriculum --p_full_anneal_steps 5000000
--max_total_steps 30000000
```

| Seed | Status | Action | Output Dir |
|------|--------|--------|------------|
| 48 | RUNNING, CR=0.51 | Keep | `results/SP17b_s48` |
| 49 | RUNNING, CR=0.52 | Keep | `results/SP17b_s49` |
| 50 | NOT STARTED | **Launch on niro-2** | `results/SP17b_s50` |
| 51 | NOT STARTED | **Launch on niro-2** | `results/SP17b_s51` |
| 52 | NOT STARTED | **Launch on niro-2 Day 2** | `results/SP17b_s52` |

**Note:** s45 (SP17, 3M switch) is also running on niro-2. Keep it as a comparison but do NOT include in the convergence cohort.

### 2.2 Group B: Ablation Studies (niro-1)

| ID | Name | Config Change | Seed | Output Dir | Priority |
|----|------|--------------|------|------------|----------|
| ABL-1 | No visibility reward | `--w_vis_pursuer 0` | 60 | `results/ABL_no_vis_s60` | **HIGH** |
| ABL-2 | No forced switch | `--force_switch_steps 99999999` | 61 | `results/ABL_no_switch_s61` | **HIGH** |
| ABL-3 | No search reward | `--w_search 0` | 62 | `results/ABL_no_search_s62` | MEDIUM |
| ABL-4 | No masking curriculum | Remove `--masking_curriculum` | 63 | `results/ABL_no_curriculum_s63` | MEDIUM |

### 2.3 Group C: Baselines (niro-1)

| ID | Name | Config | Seed | Output Dir | Priority |
|----|------|--------|------|------------|----------|
| BASE-1 | Full observability | Remove `--combined_masking` | 70 | `results/BASE_full_obs_s70` | **HIGH** |
| BASE-2 | Greedy pursuer | Eval only — load best evader | — | `results/BASE_greedy` | MEDIUM |

### 2.4 Group D: Behavior Analysis (Post-Training)

Run on ALL converged models from Group A.

| Analysis | Tool | Output |
|----------|------|--------|
| Trajectory visualization | `scripts/visualize_both_learned_gif_gpu.py` | GIF + PNG |
| Deterministic eval | `eval_deterministic.py` | Metrics JSON |
| Strategy classification | Manual review | Category counts |

---

## 3. niro-2 Launch Commands (Convergence)

```bash
# s50
nohup .venv/bin/python -u scripts/train_amsdrl.py \
  --algorithm sac --fov_angle 90 --sensing_radius 8.0 --combined_masking \
  --arena_width 20 --arena_height 20 --max_steps 1200 \
  --n_obstacles_min 0 --n_obstacles_max 3 \
  --pursuer_v_max 1.0 --evader_v_max 1.0 \
  --lr 3e-4 --ent_coef 0.03 \
  --micro_phase_steps 2048 --eval_interval 150 \
  --alternate_freeze --freeze_switch_threshold 0.65 \
  --force_switch_steps 2000000 \
  --w_vis_pursuer 0.2 \
  --w_search 0.0001 --t_stale 100 \
  --collapse_threshold 0.1 --collapse_streak_limit 5 \
  --pfsp \
  --masking_curriculum --p_full_anneal_steps 5000000 \
  --max_total_steps 30000000 \
  --seed 50 \
  --output results/SP17b_s50 \
  > results/SP17b_s50.log 2>&1 &

# s51 (same command, change seed to 51, output to SP17b_s51)
# s52 (same command, change seed to 52, output to SP17b_s52)
```

---

## 4. niro-1 Launch Commands (Ablations + Baselines)

**Prerequisite:** PyTorch must be upgraded to CUDA 12.8.

```bash
# ABL-1: No visibility reward
nohup ./venv/bin/python -u scripts/train_amsdrl.py \
  --algorithm sac --fov_angle 90 --sensing_radius 8.0 --combined_masking \
  --arena_width 20 --arena_height 20 --max_steps 1200 \
  --n_obstacles_min 0 --n_obstacles_max 3 \
  --pursuer_v_max 1.0 --evader_v_max 1.0 \
  --lr 3e-4 --ent_coef 0.03 \
  --micro_phase_steps 2048 --eval_interval 150 \
  --alternate_freeze --freeze_switch_threshold 0.65 \
  --force_switch_steps 2000000 \
  --w_vis_pursuer 0 \
  --w_search 0.0001 --t_stale 100 \
  --collapse_threshold 0.1 --collapse_streak_limit 5 \
  --pfsp \
  --masking_curriculum --p_full_anneal_steps 5000000 \
  --max_total_steps 30000000 \
  --seed 60 \
  --output results/ABL_no_vis_s60 \
  > results/ABL_no_vis_s60.log 2>&1 &

# ABL-2: No forced switch
# Same as above but: --force_switch_steps 99999999 --seed 61 --output results/ABL_no_switch_s61

# ABL-3: No search reward
# Same as SP17b but: --w_search 0 --seed 62 --output results/ABL_no_search_s62

# BASE-1: Full observability
# Same as SP17b but: remove --combined_masking --seed 70 --output results/BASE_full_obs_s70
```

---

## 5. Timeline

### Week 1: May 1–7 (Launch Phase)

| Day | Action | Deliverable |
|-----|--------|-------------|
| **May 1** | Kill s43. Launch s50, s51 on niro-2. Fix niro-1 PyTorch. Launch ABL-1, ABL-2 on niro-1. | 7 processes running |
| **May 2** | Launch s52 on niro-2. Launch ABL-3, BASE-1 on niro-1. | 10 processes running |
| **May 3** | Start LaTeX skeleton (Intro + Related Work). | Paper skeleton v0.1 |
| **May 4** | Start plotting automation (`plot_convergence.py`). | Script draft |
| **May 5** | Monitor runs for early collapse. | Collapse report |
| **May 6** | Write Methods section. | Methods draft |
| **May 7** | Weekly review — kill collapsed, relaunch if needed. | Updated run status |

### Week 2: May 8–14 (First Results)

| Day | Action | Deliverable |
|-----|--------|-------------|
| **May 8** | First evals from s50, s51 (~3M steps). | Early convergence data |
| **May 9** | ABL-1, ABL-2 should show divergence. | Ablation early results |
| **May 10** | Generate convergence curves. | Figure: convergence_overview.pdf |
| **May 11** | Write Experiments section framework. | Experiments draft |
| **May 12** | Behavior analysis on s48, s49 (already converged). | Trajectory visualizations |
| **May 13** | Write Results section with partial data. | Results draft v0.1 |
| **May 14** | Checkpoint: 3+ seeds showing CR in 0.35–0.65? | Go/no-go decision |

### Week 3: May 15–21 (Analysis + Full Draft)

| Day | Action | Deliverable |
|-----|--------|-------------|
| **May 15** | All convergence seeds should have 5M+ steps. | Cohort complete |
| **May 16** | Complete behavior analysis on all converged models. | Strategy classification |
| **May 17** | Generate all publication-quality figures. | Figure set (6+) |
| **May 18** | Complete first full paper draft. | Draft v0.9 |
| **May 19** | Internal review. | Review notes |
| **May 20** | Revise based on review. | Draft v1.0 |
| **May 21** | Final figure polish. | Camera-ready figures |

### Week 4: May 22–28 (Final Sprint)

| Day | Action | Deliverable |
|-----|--------|-------------|
| **May 22** | Final experiments complete. | Final results table |
| **May 23** | Complete Results section. | Results final |
| **May 24** | Polish Introduction + Related Work. | Front matter final |
| **May 25** | Abstract + conclusion. | Paper complete |
| **May 26** | Format check (8 pages, double-blind). | Formatted draft |
| **May 27** | Final proofread, supplement prep. | Submission-ready |
| **May 28** | **SUBMIT TO CORL** | Submitted paper |

---

## 6. Go / No-Go Checkpoints

### Checkpoint 1: May 7
**Criteria:** At least 7 processes running without immediate collapse.  
**If FAIL:** Relaunch with adjusted hyperparameters.

### Checkpoint 2: May 14
**Criteria:** At least 3 seeds showing CR in 0.35–0.65 range at 3M+ steps.  
**If FAIL:** Accept fewer seeds + stronger ablation story.

### Checkpoint 3: May 21
**Criteria:** Full paper draft complete with all available data.  
**If FAIL:** Pivot to ICRA 2027.

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| niro-1 PyTorch fix fails | Use CPU training (slower but functional) or run all on niro-2 |
| Seed sensitivity (only 1-2 converge) | Launch 5 seeds; accept 3 as minimum |
| Ablations all collapse | Publish as negative result — shows components ARE critical |
| GPU saturation on niro-2 | Cap at 4 processes; monitor `nvidia-smi` |
| Writing takes >2 weeks | Start Day 1; write in parallel with experiments |
| CoRL reject | Use feedback for stronger ICRA 2027 submission |

---

## 8. Paper Figures & Tables

### Figures
| Figure | Data Source | Tool |
|--------|-------------|------|
| Fig 1: Convergence curves | `history.json` (all seeds) | `plot_convergence.py` |
| Fig 2: NE gap evolution | `history.json` | `plot_ne_gap.py` |
| Fig 3: Ablation comparison | `history.json` (ablations) | `plot_ablation.py` |
| Fig 4: Trajectory grid | Checkpoints + env | `visualize_both_learned_gif_gpu.py` |
| Fig 5: Capture time distribution | `history.json` | `plot_capture_time.py` |
| Fig 6: Strategy heatmap | Trajectory data | Custom script |

### Tables
| Table | Content |
|-------|---------|
| Table 1 | Hyperparameters |
| Table 2 | Convergence results (5 seeds: CR, escape rate, mean capture time, NE gap) |
| Table 3 | Ablation results (CR with/without each component) |
| Table 4 | Baseline comparison (learned vs greedy vs full-obs) |
| Table 5 | Behavioral strategy classification |

---

## 9. Immediate Next Steps (This Session)

1. [ ] **Kill s43** on niro-2 (not converged to true NE)
2. [ ] **Launch s50, s51** on niro-2
3. [ ] **Fix niro-1 PyTorch** (upgrade to CUDA 12.8)
4. [ ] **Launch ABL-1, ABL-2** on niro-1
5. [ ] **Start LaTeX skeleton**
6. [ ] **Start plotting script**

---

## 10. Session Handoff Notes

**For next session, check:**
1. All running processes: `ps aux | grep train_amsdrl | grep -v grep` (both machines)
2. Latest logs: `tail -20 results/SP17b_s*/results/SP17b_s*.log`
3. History: `python3 -c "import json; data=json.load(open('results/SP17b_sXX/history.json')); print(data['history'][-1])"`
4. GPU status: `nvidia-smi` (both machines)
5. Disk status: `df -h` (both machines)

**If a run collapsed:**
- Check `history.json` for last CR
- If CR < 0.1 or > 0.9 for 5+ evals: kill and optionally relaunch with new seed
- Document in `docs/CoRL_2026_experimental_plan.md` under "Issues & Blockers"

**If niro-1 PyTorch still fails:**
- Try CPU training: `export CUDA_VISIBLE_DEVICES=""` before launch
- Fallback: move all runs to niro-2 (4 parallel, 5-day wall time)
