# S5 — Step 10 remedy: a shield-aware structured learner (2026-09-15, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83, D97 (shield v2), D89 (reportable margins; stall / deadlock split),
D111 (STR14), **D112** (S4: the shield cuts STR14 0.728 → 0.396; the learned search had exploited
free wall contact; three seeds trained under the shield reach only 0.42–0.43, NOT RECOVERED,
with intervention 0.61–0.63, stall 0.41–0.54, deadlock episodes 0.37–0.46).

## 1. Why this round exists

S4 established the failure mode of shield-in-training: the learner never sees the shield acting.
Its observation is the same 11 numbers whether the filter is passing its command through or
pinning it against an obstacle, so it cannot learn to avoid the states where the filter stalls
it, and the ±1 terminal reward arrives too rarely to attribute a 40-step stall. R3's claim is
that shield-*aware* training recovers what shield-at-deploy lost. This round tests the two
cheapest shield-aware ingredients, alone and together.

**H3 (remedy):** a structured learner trained under the shield with (A) the shield's state in its
observation, (B) a per-step stall penalty, or (C) both, produces a **shielded** pursuer that beats
the **shielded teacher** (0.426 on the primary stream, D112), on ≥ 2 of 3 seeds, at zero violations.

## 2. Components

- `pe2.structured.SearchMDP(…, shield_obs=False, stall_penalty=0.0)`:
  - `shield_obs=True` appends three features to the learner's observation (Box (14,), bounds
    ±2): the last executed step's **intervention flag** (0/1), its **`min_clearance`** in metres
    clipped to [−0.5, 2.0], and the **current stall-run length / DEADLOCK_STEPS** clipped to
    [0, 2]. All three come from the shield's own `filter` info and `ShieldMetrics` of the step
    that produced the decision-point observation; without a shield they are 0 / 2.0 / 0. The
    scripted law never sees them (it acts on the 11-dim observation as before).
  - `stall_penalty=w` subtracts `w` from the learner's span reward for every env step in the
    span on which the shield reported `stalled` (executed speed ≤ 5 % of v_max while the
    nominal wanted to move). **Not potential-based by design**: the objective must change so
    that shield-pinned states are worth avoiding, not merely learnt faster. `w = 1e-3`, fixed:
    at S4's STRSH stall rate (≈ 0.5) over a 700-step episode the penalty totals ≈ 0.35, below
    the ±1 terminal, so a capture still dominates. Evaluation is always the unshaped capture
    rate.
- Runner `scripts/derisk_s5_remedy.py`.

## 3. Protocol

- Everything else = S4's STRSH configuration exactly: `SearchMDP(cfg, GreedyEvader,
  seed0 = seed × 1000, hold = HOLD, r_acq = 0.0, shield = CBFShield(cfg))`, PPO from scratch,
  γ = 0.99, SB3 defaults, **10 M env steps**, probes every 1 M (N = 200, shielded) on
  **1 110 000+** (fresh; ≤ 1 100 199 consumed), no selection — the final checkpoint is the arm's
  policy. `OMP_NUM_THREADS=2` everywhere; provenance recorded.
- **Gate:** the final checkpoint, **shielded (δ = 0)**, on the S4 primary stream **1 090 000+**
  (N = 500) so the S4 cells pair exactly: references `TEACHER:GREEDY12:SHIELD` (0.426, the
  verdict bar), `STR14:GREEDY12:SHIELD` (0.396), `TEACHER:GREEDY12:NONE` (0.558), and the three
  S4 `STRSH` finals (0.420 / 0.432 / 0.416, the no-remedy baseline). Gate rule = S2 §3
  (`gate_pass`).
- **H-SAFE** (kill criterion): 0 contact steps across every arm's training, probes and gate.

## 4. The arms

| arm | shield_obs | stall_penalty | seeds | purpose |
|---|---|---|---|---|
| **RA** | yes | 0 | 19, 20, 21 | ingredient A: the learner sees the shield |
| **RB** | no | 1e-3 | 19, 20, 21 | ingredient B: stalling costs |
| **RC** | yes | 1e-3 | 19, 20, 21 | both |
| **RZ** | yes | 0, **all rewards zeroed** | 19 | negative control (D83): the augmented observation with no signal must not beat the shielded teacher |

Ten processes, ≈ 3.5 h concurrent.

## 5. Verdicts (fixed now)

- Per arm × seed: `final_pass_vs_TEACHER_SHIELD` (the verdict), and reported `vs STR14_SHIELD`,
  `vs TEACHER_NONE`, vs each `STRSH` final; probe curves; shield cost rates.
- **H3(arm) PASSES iff ≥ 2 of 3 seeds pass vs `TEACHER:GREEDY12:SHIELD`.** H3 overall PASSES iff
  any arm passes. RZ passing ⇒ every arm `UNDETERMINED (harness)`. A missing seed ⇒ that arm
  `INCOMPLETE`.
- **Artefact rule:** among passing arms, the seed with the highest gate rate; ties → arm order
  RC, RA, RB, then lowest seed. Written as `artefact`.
- **Cost reading (reported):** stall, deadlock-episode, intervention and infeasibility rates of
  each final vs STRSH's — did the remedy reduce the pinning, or find captures despite it?
- **Ingredient reading (reported, no gate):** RA vs RB vs RC pass counts and mean gate rates.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H3 PASSES** | A safe learned pursuer that beats the shielded teacher exists: R3's first evidence in the adversarial setting, and the artefact for Step 9 (the attacker faces both STR14-SHIELD and this policy). Which ingredient did it is the reported reading. |
| **H3 FAILS, RZ fails** | Neither cheap ingredient recovers the loss at this budget; Step 10 needs the learned feasibility constraint (Xiao et al.) or a search policy that plans around the barrier surfaces explicitly. Step 9 proceeds against STR14-SHIELD alone. |
| **RZ passes** | Harness invalid; diagnose. |
| **H-SAFE fails** | Stop; census; user ruling. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is
seen. `w = 1e-3`, the three features and their clips, seeds 19–21, the streams and the rules are
fixed here.

## 7. Not in scope

No attacker (S6 / Step 9), no learned feasibility constraint, no change to the shield, no
selection, no anchor.

## 8. Artefacts

`results/derisk/s5_arm_{ARM}{seed}.json`, `s5_probe_*.json`, `ckpts_R{A,B,C}{seed}/`,
`ckpts_RZ19/`, `s5_round.{json,png}` (probe curves per arm against the S4 bars; gate rates per
seed; cost table vs STRSH). Signed in the worklog.
