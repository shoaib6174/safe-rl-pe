# S10 — confirmatory round for S8's existence proof: β = 0.01 anchored fine-tune on fresh seeds (2026-09-17, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D119** (S8: H4 fails by rule; one β = 0.01 seed, FT-A2-35, passes both sides — deadlock
0.47/0.47/0.43 vs RC19 0.60/0.56/0.58 with Greedy capture 0.672 > 0.568; the β = 0.1 arms never left RC19,
KL ≤ 0.1), **D111** (S3: a single selected success was confirmed on five fresh seeds), **D117** (paper plan).

## 1. Why this round exists

S8 produced one policy that resists the structured attack and hunts better than RC19. One seed of three is an
existence proof, not a result. This round asks whether the β = 0.01 configuration produces such a policy
reliably. Nothing about the learner changes; only the seeds are new and the number of them is larger.

**H5:** the S8 secondary configuration (anchored fine-tune of RC19, β = 0.01, weighted family, 5 M env steps)
passes S8's two-sided per-arm test (deadlock lower than RC19 on ≥ 2 of 3 held-out attackers AND Greedy capture
not significantly below RC19) on **≥ 3 of 5 fresh seeds**.
**H-RACE-4 (stage 2, reported):** fresh attackers trained against the S10 artefact reach a pooled deadlock rate
whose Wilson upper bound is below 0.58; S7's 0.147 reported beside it.
**H-SAFE:** 0 pursuer contact steps everywhere.

## 2. Protocol

- Arms `FT-B-{40, 41, 42, 43, 44}`: exactly S8's `FT-A2` learner (RC configuration, init RC19, anchor RC19,
  β = 0.01, `WEIGHTS` and `MIXTURE` as S8, 5 M env steps, probes every 500 k vs Greedy shielded on
  **1 260 000+**). `OMP_NUM_THREADS=2`; the round runs from a frozen worktree.
- **Gate stream 1 250 000+** (N = 500): held-out `ATT25/26/27`, `GreedyEvader`, `ObstacleHuggingEvader`.
  References measured before any arm: **RC19**, **RCM28**, and **FT-A2-35** (S8's passing seed, reported as
  a fourth row — it is NOT an arm and casts no vote).
- Per-arm test = S8's `_arm_verdict` unchanged (paired deadlock tests vs RC19; capture guard via `gate_pass`).

## 3. Verdicts (fixed now)

- **H5 PASSES** iff ≥ 3 of 5 arms have per-arm verdict `PASS`. `INCOMPLETE` if any arm lacks a final record.
- **Artefact:** among passing arms, lowest mean held-out deadlock; ties → highest Greedy capture; ties → lowest
  seed. FT-A2-35 is not eligible.
- Reported without verdict: FT-A2-35's row on this stream (does the S8 seed reproduce out of sample?), each
  arm vs RCM28, anchor-KL traces, the count of arms with `FAIL (still pinned)` vs `FAIL (capture lost)`.
- H-SAFE fails ⇒ stop.

## 4. Stage 2 (only if H5 passes)

Three fresh structured attackers (S6b protocol, CTL), seeds **45, 46, 47**, probes **1 270 000+**, 10 M env
steps, vs the S10 artefact shielded; gate **1 280 000+** (N = 500). H-RACE-4 as §1; per-arm intervals beside
the pooled one.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **H5 PASSES** | The trade-off is not structural: an anchored fine-tune reliably yields a safe pursuer that resists the structured attack AND hunts. R3 becomes a method; stage 2 measures the race. |
| **H5 FAILS, 1–2/5 pass** | The configuration finds the policy sometimes; report the rate with the KL traces; selection across seeds (as S2's H-SEL) is the honest next question, not a re-tune. |
| **H5 FAILS, 0/5** | FT-A2-35 was a fluke; D119's reading is withdrawn to "one observation". |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number is seen. Consumed streams
before this round: ≤ 1 249 999.

## 6. Artefacts

`results/derisk/s10_start.json`, `s10_arm_FT-B-*.json`, `s10_probe_*`, `ckpts_FT-B-*/`, `s10_round.{json,png}`;
stage 2 `s10race_arm_ATK4-*.json`, `ckpts_ATK4-*/`, `s10_race_start.json`, `s10_race.json`.
