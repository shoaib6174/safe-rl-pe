# S8 — Step 10, third iteration: a capture-preserving best response to the structured attack (2026-09-17, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D113** (RC19: shield-state features recover capture, 0.564 vs Greedy), **D115** (structured attacker
pins RC19-SHIELD 0.58–0.61), **D116** (S7: training from scratch against a uniform family removes the pin,
deadlock 0.09–0.22, but the pursuer disengages — Greedy capture 0.13–0.33; fresh attackers reach only 0.14–0.16),
**D110** (KL anchors stop drift without stopping learning, β ∈ {0.1, 0.01}), **D117** (paper strategy).

## 1. Why this round exists

S7's gate was one-sided: it tested deadlock and only *labelled* capture. The learner found the cheap best response —
stand off from everything. D116's table shows capture against the attackers themselves is ≈ 0.10 for every
pursuer; what S7 lost is capture against the ordinary evaders. This round prices both objectives and moves the
starting point: **fine-tune RC19** (which already captures Greedy at 0.55) against a **task-weighted** family,
with a **KL anchor** to RC19 so the capture behaviour is protected while the pin-avoidance is learned.

**H4 (two-sided):** an anchored fine-tune of RC19 against the weighted family has, seed-paired on a fresh gate,
(i) a LOWER deadlock-episode rate than RC19 against ≥ 2 of 3 held-out attackers AND (ii) a Greedy capture rate
NOT significantly below RC19's — on ≥ 2 of 3 seeds of the primary family.
**H-RACE-3 (stage 2, reported):** fresh attackers trained against the S8 artefact reach a pooled deadlock rate
whose Wilson upper bound is below 0.58 (S6b's floor against RC19); also reported against S7's 0.147.
**H-SAFE:** 0 pursuer contact steps everywhere.

## 2. Components

- `pe2.evade.MixtureOpponentFactory(members, seed, weights=None)` — optional per-member sampling weights
  (normalised; `None` = uniform, bit-identical to S7's behaviour).
- Anchored fine-tune on `SearchMDP`: `KLAnchoredPPO.load(RC19, env=SearchMDP(...RC config...), seed, device)`,
  `beta` set, `set_anchor(model.policy)` BEFORE any step (the anchor is RC19 itself). Validity check **V-A0:**
  with `beta = 0` the anchored learner's first 4 096 env steps must match plain `PPO.load(RC19, env=...)` with the
  same seed bit-for-bit in `num_timesteps`, rollout returns and policy parameters (the S2 tripwire logic, on this
  env); a mismatch halts the round.
- Runner `scripts/derisk_s8_remedy3.py` (S7's machinery: `gate_pursuer`, race via S6, prefixes `s8`, `s8race`).

## 3. Protocol — stage 1

- Weighted family (per episode): `GreedyEvader` **0.50**, `ObstacleHuggingEvader` **0.20**, structured `SplitHug`
  **0.10**, structured `CTL-D125/126/127` **0.20 total (1/15 each)**. Fixed here.
- Learner: RC configuration unchanged (`shield=CBFShield(cfg)`, `shield_obs=True`, `stall_penalty=1e-3`,
  `r_acq=0`, γ 0.99, hold 10), **initialised from RC19**, `KLAnchoredPPO` with anchor = RC19.
- Arms (7): **primary** `FT-A1-{34,35,36}` β = 0.1; **secondary** `FT-A2-{34,35,36}` β = 0.01; **control**
  `SCR-W34` = from scratch (S7's learner) on the weighted family, seed 34 — separates "start from RC19" from
  "weight the family". Budget **5 M env steps** each (a fine-tune; S2's anchored arms plateaued well inside this),
  probes every 500 k (N = 200) vs `GreedyEvader` shielded on **1 210 000+**. `OMP_NUM_THREADS=2`.
- **Gate stream 1 200 000+** (N = 500), shielded, paired within opponent: held-out `CTL-D225/226/227`
  (`ATT25/26/27`), `GreedyEvader`, `ObstacleHuggingEvader`. **References RC19 AND RCM28** are measured on this
  stream before any arm (`s8_start.json`); an arm refuses to run without both.

## 4. Verdicts — stage 1 (fixed now)

- Deadlock test per held-out attacker: `paired_indicator_test(rc19_ind, s8_ind)` (RC19 deadlocks more, p < 0.05).
- Capture guard: McNemar on the Greedy capture indicator, S8 vs RC19; the guard FAILS iff RC19 is significantly
  better (p < 0.05, b > c in RC19's favour). (Same test as S7's label, now a pass condition.)
- **H4(seed) PASSES** iff ≥ 2/3 deadlock tests pass AND the capture guard does not fail.
- **H4 PASSES** iff ≥ 2 of the 3 primary seeds pass. The secondary family and the control get the same per-seed
  verdicts, reported; if the primary fails and the secondary passes 2/3, the round is `PASS (β = 0.01)` and
  says so. Missing arm ⇒ `INCOMPLETE`.
- **Artefact:** among passing arms (primary first, secondary if the primary failed), the lowest mean held-out
  deadlock; ties → highest Greedy capture; ties → lowest seed.
- Also reported (no verdict): each arm vs RCM28 on the same stream (capture and deadlock), the anchor KL trace.
- H-SAFE fails ⇒ stop.

## 5. Stage 2 — the race (only if H4 passes)

Three fresh structured attackers (S6b protocol, CTL, terminal reward), seeds **37, 38, 39**, probes
**1 220 000+**, 10 M env steps, vs the S8 artefact shielded; gate **1 230 000+** (N = 500). H-RACE-3: pooled
Wilson-hi < 0.58, per-arm intervals beside it; S7's 0.147 reported as the comparison point.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H4 PASSES, H-RACE-3 holds** | A safe learned pursuer that resists the structured attack AND keeps its capture exists; the arms race converges without disengagement. The paper's R3 becomes a method. |
| **H4 PASSES, H-RACE-3 fails** | Capture kept, old attackers resisted, fresh ones adapt; the price of hunting is exploitability — reported as the trade-off curve. |
| **H4 FAILS (deadlock ok, capture lost)** | The anchor did not protect capture; the trade-off is structural at this budget: RCM28 (disengaged) and RC19 (huntable) are the two ends and no anchored path joins them. |
| **H4 FAILS (capture kept, still pinned)** | The anchor held too tightly; report with the KL trace; β is not re-tuned (stop rule). |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number is seen. Weights, β, seeds,
streams and rules are fixed here. Consumed streams before this round: ≤ 1 190 999.

## 7. Artefacts

`results/derisk/s8_preflight.json` (V-A0), `s8_start.json`, `s8_arm_{FT-A1-34,…,SCR-W34}.json`, `s8_probe_*`,
`ckpts_FT-*/`, `ckpts_SCR-W34/`, `s8_round.{json,png}`; stage 2 `s8race_arm_ATK3-{37,38,39}.json`,
`ckpts_ATK3-*/`, `s8_race_start.json`, `s8_race.json`. Signed in the worklog.
