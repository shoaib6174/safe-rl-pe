# Future Work Plan: Strengthening CoRL 2026 Evidence

**Date:** 2026-05-02
**Session:** S88 continuation
**Status:** Approved for immediate execution

## Objective

Launch ABL-3, ABL-4, and BR-5/BR-6 to strengthen the empirical foundation
for Approach B ("stabilization mechanisms + limits") before the CoRL 2026
deadline (May 28). The verdict from S88 (H3, Framing C) showed the paper's
central claim is defensible but needs more evidence to be competitive.

---

## Group D: ABL-3 (No Search Reward)

**Purpose:** Prove search staleness reward is critical for stable co-evolution.

**Config change:** `w_search=0` (remove `--w_search 0.0001`).

**Expected behavior:** Without search reward, the pursuer may fail to
systematically cover the arena when the evader is not visible. This could
cause either: (a) evader hides indefinitely → CR collapses, or (b) pursuer
learns a suboptimal search pattern → CR plateaus below SP17b.

**Launch:**
```bash
nohup ./venv/bin/python -u scripts/train_amsdrl.py \
    --algorithm sac --fov_angle 90 --sensing_radius 8.0 --combined_masking \
    --arena_width 20 --arena_height 20 --max_steps 1200 \
    --n_obstacles_min 0 --n_obstacles_max 3 \
    --pursuer_v_max 1.0 --evader_v_max 1.0 \
    --lr 3e-4 --ent_coef 0.03 --micro_phase_steps 2048 --eval_interval 150 \
    --alternate_freeze --freeze_switch_threshold 0.65 \
    --force_switch_steps 2000000 \
    --w_vis_pursuer 0.2 \
    --w_search 0 \
    --t_stale 100 \
    --collapse_threshold 0.1 --collapse_streak_limit 5 \
    --pfsp --masking_curriculum --p_full_anneal_steps 5000000 \
    --max_total_steps 30000000 \
    --seed 62 \
    --output results/ABL_no_search_s62 \
    > results/ABL_no_search_s62.log 2>&1 &
```

**Machine:** niro-1 (GPU available, ~30GB free).

---

## Group E: ABL-4 (No Masking Curriculum)

**Purpose:** Prove the masking curriculum eases learning by annealing from
full to partial observability.

**Config change:** Remove `--masking_curriculum`. `p_full_obs` defaults to
0.0 (full partial observability from the start).

**Expected behavior:** Without curriculum, agents face full PO immediately.
Historical evidence (SP15) shows this causes collapse or very slow learning.
ABL-4 should either collapse early or achieve much lower peak CR than SP17b.

**Launch:**
```bash
nohup ./venv/bin/python -u scripts/train_amsdrl.py \
    --algorithm sac --fov_angle 90 --sensing_radius 8.0 --combined_masking \
    --arena_width 20 --arena_height 20 --max_steps 1200 \
    --n_obstacles_min 0 --n_obstacles_max 3 \
    --pursuer_v_max 1.0 --evader_v_max 1.0 \
    --lr 3e-4 --ent_coef 0.03 --micro_phase_steps 2048 --eval_interval 150 \
    --alternate_freeze --freeze_switch_threshold 0.65 \
    --force_switch_steps 2000000 \
    --w_vis_pursuer 0.2 \
    --w_search 0.0001 \
    --t_stale 100 \
    --collapse_threshold 0.1 --collapse_streak_limit 5 \
    --pfsp \
    --p_full_anneal_steps 5000000 \
    --max_total_steps 30000000 \
    --seed 63 \
    --output results/ABL_no_curriculum_s63 \
    > results/ABL_no_curriculum_s63.log 2>&1 &
```

**Note:** `--masking_curriculum` is intentionally omitted. All other params
match SP17b exactly.

**Machine:** niro-2 (GPU available, ~30GB free; run alongside s49 and s50).

---

## Group F: BR-5/BR-6 on s50 (3rd Seed Replication)

**Purpose:** Replicate the BR exploitability test on a 3rd seed to strengthen
the H3 claim from "2 seeds" to "3 seeds."

**Snapshot:** s50 M2550 (already exists on niro-2):
- `/home/niro-2/Codes/safe-rl-pe/results/SP17b_s50/checkpoints/pursuer/milestone_phase2550_pursuer`
- `/home/niro-2/Codes/safe-rl-pe/results/SP17b_s50/checkpoints/evader/milestone_phase2550_evader`

**Baseline:** s50 M2550 CR = 0.460 (from trajectory analysis).

**Expected behavior:** Same pattern as s48/s49:
- BR-5 (trained pursuer vs frozen s50 evader): gap negative (evader is strong)
- BR-6 (trained evader vs frozen s50 pursuer): gap positive, > ε=0.05
  (pursuer is exploitable)

**Launch on niro-1:**
```bash
# BR-5: trained pursuer vs frozen s50 evader
nohup ./venv/bin/python -u scripts/train_br_sac.py \
    --frozen_opponent_path results/BR_frozen/s50/evader \
    --frozen_role evader \
    --total_steps 1500000 \
    --eval_freq 250000 \
    --n_eval_episodes 200 \
    --seed 150 \
    --output_dir results/BR_5 \
    > results/BR_5.log 2>&1 &

# BR-6: trained evader vs frozen s50 pursuer
nohup ./venv/bin/python -u scripts/train_br_sac.py \
    --frozen_opponent_path results/BR_frozen/s50/pursuer \
    --frozen_role pursuer \
    --total_steps 1500000 \
    --eval_freq 250000 \
    --n_eval_episodes 200 \
    --seed 250 \
    --output_dir results/BR_6 \
    > results/BR_6.log 2>&1 &
```

**Note:** BR_frozen/s50/ must be created by copying the M2550 milestones.

---

## Timeline

```
May 2 (now)     Launch ABL-3, ABL-4, BR-5, BR-6
May 2–3         BR-5/BR-6 complete (~4.5h each)
May 3           Run analyzer with 3-seed data; update verdict
May 7–8         ABL-3/ABL-4 reach M2550; check for divergence
May 10          Decision gate:
                - If ABL-3/4 show clear divergence + BR-5/6 replicate H3
                  → Proceed with CoRL submission (strong evidence)
                - If evidence is mixed
                  → Evaluate whether paper is borderline; consider ICRA 2027
May 10–21       Write paper body (Intro, Methods, Results, Figures)
May 21          Paper draft complete
May 21–28       Polish, internal review, submit
```

---

## Decision Criteria

### ABL-3 (no search)
| Outcome | Interpretation |
|---------|---------------|
| Collapse (CR < 0.1) or freeze-sticking | Search reward is **critical** — strong ablation |
| Stable but lower CR (0.2–0.4) | Search reward helps but is not critical |
| Similar to SP17b (0.45–0.55) | Search reward is **not critical** — weakens ablation story |

### ABL-4 (no curriculum)
| Outcome | Interpretation |
|---------|---------------|
| Collapse (CR < 0.1) or early death | Curriculum is **critical** — strong ablation |
| Stable but lower CR (0.2–0.4) | Curriculum helps but is not critical |
| Similar to SP17b (0.45–0.55) | Curriculum is **not critical** — weakens story |

### BR-5/BR-6 (s50 replication)
| Outcome | Interpretation |
|---------|---------------|
| Both match s48/s49 pattern (L3) | H3 replicated across 3 seeds — **very strong** |
| Mixed (e.g., L2 or L1) | Seed-dependent NE quality — shifts to H5 framing |
| Both L4 (two-sided exploit) | Much worse than expected — shifts to H4 framing |

### Overall paper strength
| Condition | Strength |
|-----------|----------|
| ABL-3 collapse + ABL-4 collapse + BR-5/6 replicate H3 | **Strong** — submit to CoRL with confidence |
| 2 of 3 conditions met | **Moderate** — submit but acknowledge limitations |
| 1 of 3 conditions met | **Weak** — consider ICRA 2027 or major rewrite |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| niro-2 GPU saturated (s49 + s50 + ABL-4 + BR-5/6) | Monitor nvidia-smi; kill s49 if BR-5/6 need its slot |
| ABL-3/ABL-4 take longer than 7 days to reach M2550 | Accept earlier-phase data if divergence is already visible |
| BR-5/BR-6 smoke failure | Same debugging protocol as S88 (check frozen load, p_full=0) |
| Paper draft slips past May 21 | Hard deadline: stop writing May 25 regardless of polish level |

---

## Definition of Done

This plan is done when:
- [ ] ABL-3 launched on niro-1
- [ ] ABL-4 launched on niro-2
- [ ] s50 M2550 snapshotted to results/BR_frozen/s50/
- [ ] BR-5 launched on niro-1
- [ ] BR-6 launched on niro-1
- [ ] All 5 runs confirmed running (clean startup, no tracebacks)
