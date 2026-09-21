# S17 — Phase 2 redesigned: a forward-offering recovery, judged on the task (2026-09-21, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D129** (Phase 2: every arm failed a primary metric I mis-chose — immobility counts rotation,
which is the recovery's own mechanism; `REC` drove deadlock to 0.000 in all nine cells; `OBS` and `GAM` each
help partially; `PEN` backfires by teaching the agent to stop asking to move; `MEM` confounded by A4),
**D130** (the long spins are not a trap: a straight command moves in 100 % of 326 samples taken deep inside
them, and the recovery never offers one).

## 1. What changed and why

Three corrections to S16, each forced by a measurement rather than by taste.

1. **The primary metric is no longer immobility.** It counted a remedy's mechanism as the disease. This round
   judges on the **task** (capture), on **deadlock episodes**, and on a new **spin fraction** that exists
   specifically so that no arm can win by rotating on the spot — the failure mode S16's metrics hid.
2. **The recovery offers forward candidates.** D130 showed a straight command escapes 100 % of sampled long
   spins while the old recovery only ever proposed rotations.
3. **Nothing is re-trained that need not be.** `BASE`, `OBS` and `GAM` are the S16 checkpoints, re-gated on a
   fresh stream under the new metrics. Only the combination arm is new. This is three training runs, not
   fifteen, and it keeps every comparison against policies whose S16 behaviour is already on record.

## 2. `RecoveryShieldV2`, the forward-offering recovery

When the immobile run reaches **20** steps, form the candidate set — in this fixed priority order —
`(v_max, 0)`, `(0.5·v_max, 0)`, the policy's own nominal, `(0, +ω_max)`, `(0, −ω_max)` — filter each through
the ordinary CBF shield, and execute the **first whose filtered output moves** (`v_exec > 0.05·v_max`). If none
moves, execute the rotation with the larger `|ω_exec|`, exactly as V1 did. Record on every step which category
fired: `forward`, `policy`, `rotation`, or `none`.

Safety is the ordinary argument and nothing more: every candidate is filtered by the same CBF shield as any
other action, so H-SAFE covers it with no special pleading (the claim corrected in S16-A1 is not repeated).

**Per-step logging, required by D130:** the runner records the policy's nominal and the executed command as
separate fields, so the attribution D130 could not resolve is a measurement here, not an inference.

## 3. Arms

| arm | policies | recovery | new training? |
|---|---|---|---|
| `BASE` | S16 `ckpts_BASE-{60,61,62}` | none | no |
| `OBS` | S16 `ckpts_OBS-{60,61,62}` | none | no |
| `GAM` | S16 `ckpts_GAM-{60,61,62}` | none | no |
| `OBSGAM` | **new**: escape features AND γ = 0.999, else the S16 `BASE` recipe, seeds 60/61/62, 5 M steps | none | **yes** |
| `RECF-BASE` | S16 `BASE` | V2 | no |
| `RECF-OBS` | S16 `OBS` | V2 | no |
| `RECF-OBSGAM` | `OBSGAM` | V2 | no |

Gate: stream **1 740 000+**, N = **500**, opponents `ATK4-45`, `HUG7`, `GREEDY`, all arms on the same episodes.
A fresh `RecoveryShieldV2` per episode (the S16 requirement, unchanged).

## 4. Hypotheses (fixed now)

**H1 (task).** An arm's `GreedyEvader` capture is significantly above `BASE`'s on the same seed (McNemar,
p < 0.05), on **≥ 2 of 3** seeds.
**H2 (deadlock).** Its deadlock-episode rate against `ATK4-45` is **< 0.05** on **≥ 2 of 3** seeds.
**H3 (no spinning).** Its **spin fraction** — steps with `v_exec ≤ 0.05·v_max` AND `|ω_exec| ≥ 0.5·ω_max`,
over all gate steps — is **< 0.05** on **≥ 2 of 3** seeds.
**An arm SOLVES iff H1, H2 and H3 all hold.** `BASE` is the reference and is not eligible to solve.
**H-SAFE:** 0 pursuer contact steps in every cell.

**Pre-registered expectations, so the round can contradict me:** `RECF-BASE` solves; `OBSGAM` does not solve
alone but beats both `OBS` and `GAM` on H1; `RECF-OBSGAM` is the best arm overall. I expect S16's `REC` to
have been failing H3 had it been measured, and this round measures it.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **`RECF-*` solve, bare learned arms do not** | The remedy is an engineering one: offer the filtered agent a forward escape. That is the paper's practical recommendation, now with a metric that cannot be won by spinning. |
| **`RECF-BASE` fails H3** | The forward-offering recovery still spins; D130's 100 % result did not transfer to closed loop, and the mechanism needs re-diagnosing before any remedy is proposed. |
| **`OBSGAM` solves alone** | The fix is a learning one after all — expose the admissible set and lengthen the horizon — and the scripted recovery is unnecessary. |
| **No arm solves** | Report as a failure. The three metrics are then known to be jointly unsatisfiable by anything tried, which is itself a result about filtered agents. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, retuned or re-seeded after any number of this round is seen. Arms, seeds,
budget, streams and the three bars (0.05 deadlock, 0.05 spin fraction, p < 0.05 capture) are fixed above.

## 6. Artefacts

`results/derisk/s17_arm_*.json`, `ckpts_OBSGAM-*/`, `s17_round.{json,png}`; runner
`scripts/derisk_s17_fixes_b.py`, run from a frozen worktree.
