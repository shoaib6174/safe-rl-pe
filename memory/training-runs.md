# Training Runs Log

## EMA Seeds 4-6 (Post Pop-FSP Pivot)

| Run | Seeds | Episodes | Status | Launched | Notes |
|-----|-------|----------|--------|----------|-------|
| EMA tau=0.001 | 4, 5, 6 | 5000 | **COMPLETE** | 2026-05-25 | All collapsed: s4=evader+NaN, s5=pursuer, s6=evader. |
| EMA checkpoint sweep | 4, 5, 6 | 20 ckpts × 100 ep | **RUNNING** | 2026-05-31 | Robustness + oscillation analysis per checkpoint. |

## GI Vanilla 20K Baseline

| Run | Seeds | Episodes | Status | Launched | Notes |
|-----|-------|----------|--------|----------|-------|
| GI vanilla SP | 10, 11, 12 | 20000 | **RUNNING** | 2026-05-25 | Pure self-play, no EMA. Matching G&I paper duration (20K vs prior 5K). |

## Pop-FSP v2 (Recency + Asymmetric + Gradual Mixing)

| Run | Seeds | Episodes | Status | Launched | Notes |
|-----|-------|----------|--------|----------|-------|
| Pop-FSP v2 | 0, 1, 2 | 5000 | **KILLED** | 2026-05-24 | Recency λ=0.15, evader pop=16, gradual mixing 100-500. All pursuer lock-in by ep 660. |

### Early Metrics (Ep ~166)
- **Seed 0**: CR oscillating 0.0-0.5, PA~1440, EA~-1260 (evader-dominant early)
- **Seed 1**: CR oscillating 0.0-1.0, PA~1200, EA~-700 (more balanced)
- **Seed 2**: CR=0.0 consistently, PA~1160, EA~-710 (evader-dominant early)

---

## Historical Runs

### Pop-FSP Baseline (S100)
| Run | Seed | Episodes | Status | Final CR | Notes |
|-----|------|----------|--------|----------|-------|
| Pop-FSP 5K | 0 | 5000 | **COMPLETE** | 1.000 | Oscillation eliminated, pursuer-dominant |
| Pop-FSP 500 | 0 | 500 | **COMPLETE** | 0.000 | Baseline smoke test |

### EMA Comparison (S97-S99)
| Run | Seed | Episodes | Status | Final CR | Notes |
|-----|------|----------|--------|----------|-------|
| EMA tau=0.001 | 0 | 5000 | **COMPLETE** | 0.623 | Best balance achieved |
| EMA tau=0.001 | 1 | 5000 | **COMPLETE** | 0.002 | Evader collapse |
| EMA tau=0.001 | 2 | 5000 | **COMPLETE** | 0.004 | Evader collapse |
| EMA tau=0.001 | 3 | 5000 | **COMPLETE** | 0.170 | LR=0.0005, partial collapse |
| Vanilla | 0 | 5000 | **COMPLETE** | 1.000 | Pursuer-dominant |
| Vanilla | 1 | 5000 | **COMPLETE** | 0.004 | Evader collapse |
| Vanilla | 2 | 5000 | **COMPLETE** | 0.004 | Evader collapse |

### GI Reproduction (S92-S96)
| Run | Seed | Episodes | Status | Final CR | Notes |
|-----|------|----------|--------|----------|-------|
| GI-v4-vectorized | 7, 8, 9 | 5000 | **COMPLETE** | - | Paper reproduction verified |
| GI-v3-belief-noamp | 1-6 | 5000 | **COMPLETE** | - | BiMDN experiments |
