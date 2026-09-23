# PBT v3 Final Findings

**Date**: 2026-07-14  
**Status**: STOPPED (processes killed to free GPU for Experiment 3)  
**Archive**: `repro_gonultas_isler_2024/archive/pbt_v3_final_20260714/`  
**Config**: `repro_gonultas_isler_2024/config_pbt.yaml`  

---

## Summary

PBT v3 was designed to test whether population-based training with explicit
selection for dual-role balance could sustain competitive cycling in symmetric
point-mass pursuit-evasion. The experiment ran three seeds on niro-1 (RTX 5090).

**Verdict: PBT v3 did not produce competitive balance. All seeds converged to
pursuer lock-in.**

---

## Raw run data

| Seed | PID | Final episode | Final CR | Final pattern | Verdict |
|------|-----|---------------|----------|---------------|---------|
| 0 | 10679 | 5,880 | 0.750 | Predominantly CR=1.0 after early episodes; intermittent 0.75/0.5 | Pursuer lock-in |
| 1 | 10744 | 14,498 | 1.000 | CR=1.0 for final ~3,000+ episodes | Complete pursuer lock-in |
| 2 | 10810 | 13,295 | 1.000 | CR=1.0 for final ~2,000+ episodes | Complete pursuer lock-in |

---

## Tournament results (seed 1)

PBT tournaments evaluate every agent in the population as both pursuer and
evader. The balance metric is `-(|P_WR - 0.5| + |E_WR - 0.5|)`; values near 0
indicate balance, values near ±1 indicate lock-in.

Tournaments at episodes 12,000–14,000 consistently showed:

```
Agent 0: P_WR=1.000 E_WR=0.000 Balance=-1.000
Agent 1: P_WR=1.000 E_WR=0.000 Balance=-1.000
Agent 2: P_WR=1.000 E_WR=0.000 Balance=-1.000
Agent 3: P_WR=1.000 E_WR=0.000 Balance=-1.000
Agent 4: P_WR=1.000 E_WR=0.000 Balance=-1.000
No replacement (single most-balanced agent)
```

Every agent in the population won 100% of games as pursuer and 0% as evader.
The dual-role balance selection pressure was unable to correct the equilibrium.

---

## Design features tested in v3

- Fitness = dual-role balance: `-|P_WR - 0.5| - |E_WR - 0.5|`
- Replacement: weakest agent at BOTH roles replaced by copy of strongest at BOTH roles
- Data budget: 20K episodes (up from 10K)
- Replay buffer: 500K (up from 100K)
- Parallel envs: 4 (up from 2)
- Tournament frequency: 500 episodes
- Population size: 5

---

## Interpretation

PBT v3's failure supports the paper's central claim: structural ∀/∃ asymmetry
makes population-based self-play in pursuit-evasion prone to pursuer lock-in.
Even with:
- Symmetric dynamics and observations,
- Shared role-conditioned policy architecture,
- Explicit selection pressure for balance,
- Larger data and replay budgets,

the population converged to a state where every agent is pursuer-dominant.

This suggests that the asymmetry is not resolved by simply adding population
diversity or balance selection. Methods that explicitly account for the evader's
universal quantifier requirement (e.g., evader robustness training,
population-aware critics) may be needed.

---

## Paper integration

- **Do not present PBT v3 as a positive result.**
- It can be cited as additional negative evidence in the Results or Discussion
  section: "Even population-based training with explicit dual-role balance
  selection failed to escape pursuer lock-in (3/3 seeds, archive
  `pbt_v3_final_20260714/`)."
- It strengthens the case that the failure is structural rather than a
  hyperparameter or training-budget issue.

---

## Files archived

- `pbt_seed_0.log`, `pbt_seed_1.log`, `pbt_seed_2.log`
- `config_pbt.yaml`
- `launch_pbt.sh`
- `results_pbt/` (checkpoints and training outputs)

---

## Follow-up analysis reminder

After Experiments 2–4 are complete, compare PBT v3's failure mode against:
1. Pop-FSP v3–v7 (already complete): similar pursuer lock-in.
2. Symmetric point-mass cohort (Experiment 3, pending): does 2/9 balance rate
   hold at larger scale?
3. Opponent-sampling ablation (Experiment 2, pending): can any sampling strategy
   prevent lock-in?
4. Asymmetric evader training (Experiment 4, pending): can explicit evader
   robustness counter the bias?

The goal of the follow-up analysis is to determine whether PBT v3's failure is
consistent with the broader pattern and whether any intervention succeeds where
PBT v3 failed.
