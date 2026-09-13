# S2 — the CP lead: is the mid-run improvement real, selectable, and holdable? (2026-09-13, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), D108 (one-step-TD collapse; learnability
untested), **D109** (S1: CP — PPO-GAE λ = 1.0 from the PPO clone — rose from 0.470 to 0.645 at
200k, the first separated improvement over a start in this project, then decayed to 0.246),
D100/D102 (anchor continuum lessons). Autopsy of the decay in `docs/worklogs/2026-09-12_S125.md`.

## 1. Why this round exists

S1's control arm is the lead. Its probe curve on 1 010 000+ was 0.640 0.645 0.585 0.605 0.510
0.550 0.560 0.380 0.285 0.265 from a 0.470 start (129/200 at 200k, Wilson [0.577, 0.708]). The
pre-registration judged the final probe, so S1 claimed nothing. The autopsy of the ten CP
checkpoints on the clone's own states:

| step | capture (N=100) | mean v-action | reversing | ω saturated | visible | log-std (v, ω) | KL(clone‖π) |
|---|---|---|---|---|---|---|---|
| clone | 0.47 | 1.00 | 0.00 | 0.00 | 0.199 | −1.00, −1.00 | 0 |
| 100k | 0.59 | 0.88 | 0.00 | 0.02 | 0.207 | −1.03, −1.14 | 0.57 |
| 200k | 0.56 | 0.82 | 0.00 | 0.01 | 0.219 | −1.03, −1.18 | 0.91 |
| 400k | 0.62 | 0.76 | 0.00 | 0.01 | 0.206 | −1.19, −1.19 | 1.74 |
| 600k | 0.57 | 0.61 | 0.02 | 0.03 | 0.209 | −1.38, −1.23 | 3.88 |
| 800k | 0.27 | 0.46 | 0.25 | 0.16 | 0.123 | −1.75, −1.18 | 9.02 |
| 1M | 0.18 | 0.48 | 0.19 | 0.23 | 0.089 | −1.96, −1.52 | 17.68 |

The decay is **not** entropy collapse (σ stays ≈ 0.15–0.35; contrast SAC's log-std −14 in D108)
and not the critic (PPO's value is only a baseline at λ = 1). It is **unbounded drift from the
clone**: KL grows monotonically with no plateau; capture is high while KL ≈ 0.6–1.7 and gone
once KL passes ≈ 5–9. That names the remedy: a KL anchor to the clone (D102's mechanism, now
with a reason to expect it to work — the improved region *exists*, at bounded divergence).

Two hypotheses, two ways of not fooling ourselves:

- **H-SEL** — the improvement is real and reproducible across training seeds, and a
  pre-registered selection rule on a selection stream picks a checkpoint that beats the clone on
  a **fresh gate stream**.
- **H-ANC** — an anchor to the clone holds the improved region, so the **final** checkpoint
  beats the clone with no selection at all.

The forking-path risk of "pick the best of ten probes" is measured directly by a negative
control (§4 NULL) rather than argued away.

## 2. Components

- `pe2.anchored_ppo.KLAnchoredPPO` — subclass of SB3 `PPO` mirroring `KLAnchoredSAC`:
  `set_anchor(policy)` freezes a deep copy; `train()` is SB3's `PPO.train` with one added term,
  `beta * KL(π_θ(·|s) ‖ π_anchor(·|s))` on the rollout observations, via the existing
  `gaussian_kl` (PPO's Gaussian has a state-independent log-std, taken from the policy's
  distribution). `beta = 0` MEASURES the KL and logs `train/anchor_kl` but does not price it and
  is bit-identical to plain PPO (tested as for SAC). `last_anchor_kl` is exported so the probe
  can record it per checkpoint. `use_sde` refused; negative `beta` refused.
- `pe2.exploitability.finetune_ppo_anchored(cfg, role, opp, model_path, total_timesteps, seed,
  beta, device, callback, env_wrap=None, custom_objects=None)` — loads the PPO clone into a
  `KLAnchoredPPO` (same `PPO_KWARGS` and `custom_objects={"gae_lambda": 1.0}` as S1's CP),
  anchors it to the loaded policy BEFORE any step, trains. `env_wrap` wraps the TRAINING env
  only (used by NULL).
- `pe2.rewards.ZeroRewardWrapper` — every reward 0.0 (the NULL arm's signal).

## 3. Protocol

- Role: pursuer; opponent frozen `GreedyEvader`, `EnvConfig()` defaults.
- Prior: the existing PPO clone `results/derisk/ppo_bc_searching_pursuer.zip` (S1: 300 demos of
  `SearchingPursuer` on 1 020 000+, `train_bc_ppo` 50 epochs; 0.470 on 1 010 000+).
- Learner: S1's CP configuration exactly — `PPO_KWARGS` with `gae_lambda = 1.0` — plus the
  anchor term where `beta > 0`. **Budget 1 000 000 env steps per arm**, probes every
  **100 000** at N = 200.
- Streams (fresh; ≤ 1 030 999 consumed, including the autopsy on 1 010 000+):
  **selection/probe 1 040 000+**, **gate 1 050 000+**. Training seeds **7, 8, 9** (wrapper
  `seed0 = seed × 1000`).
- **Bars on the gate stream, N = 500, measured BEFORE any arm trains**, written to
  `results/derisk/s2_start.json`: `GATE_PPO_CLONE` (the prior), `GATE_SAC_CLONE`
  (`results/derisk/sac_bc_searching_pursuer.zip`, D106), `GATE_TEACHER` (`SearchingPursuer`),
  each with its full paired log kept for McNemar. An arm refuses to run without it.
- **Selection rule (fixed):** among an arm's ten probes on 1 040 000+, the checkpoint with the
  highest win rate; ties → the earliest. Applied identically to every arm, including NULL.
- **Gate (fixed):** the selected checkpoint AND the final (1M) checkpoint are each scored on
  1 050 000+ at N = 500, paired with the bars. A checkpoint **PASSES vs X** iff the exact
  two-sided McNemar test vs X has p < 0.05 with more discordant wins than losses AND its Wilson
  lower bound exceeds X's point estimate on the gate stream. Reported for X ∈ {PPO clone, SAC
  clone, teacher}; **the verdict uses X = PPO clone** (the arm's own start); "beats the
  teacher" (X = `SearchingPursuer`) is reported alongside and is the paper's number if it holds.

## 4. The arms

| arm | β | reward | seeds | purpose |
|---|---|---|---|---|
| **SEL** | 0 (measured only) | terminal ±1 | 7, 8, 9 | **H-SEL**: reproduce the rise, select, gate. Seed 7 also checks whether S1's CP curve reproduces bit-for-bit (reported, not gated). |
| **ANC-0.1** | 0.1 | terminal ±1 | 7, 8, 9 | **H-ANC** at the stronger weight |
| **ANC-0.01** | 0.01 | terminal ±1 | 7, 8, 9 | **H-ANC** at the weaker weight — two points on the continuum (D100/D102) |
| **NULL** | 0 | **all zero** (`ZeroRewardWrapper`) | 7 | **negative control for selection**: the same selection rule on a run with no signal measures how much "improvement" best-of-ten manufactures. Must FAIL the gate vs the PPO clone (selected checkpoint). |
| **SBLONG** | — | S1's SB (terminal only, structured) | 7 | **side arm, no claim**: S1's SB continued to **10 000 000 env steps** (≈ matched to CP's ≈ 488 iterations), probes every 1 000 000 on **1 010 000+** (S1's stream, so S1's bars apply), verdict by S1's `structured_verdict` against S1's `s1_start.json` bars. Closes amendment A1's update-count caveat on the record. |

Eleven processes, concurrent; ≈ 20 min each for the 1M arms, ≈ 3–4 h for SBLONG.

## 5. Verdicts (fixed now)

Per 1M arm: `selected_pass_vs_clone`, `final_pass_vs_clone`, and the same vs teacher, plus the
selected step, the probe curve, and the per-probe `anchor_kl`.

- **H-SEL PASSES** iff ≥ 2 of 3 SEL seeds' *selected* checkpoints pass vs the PPO clone.
- **H-ANC(β) PASSES** iff ≥ 2 of 3 seeds' *final* checkpoints pass vs the PPO clone at that β.
- **Harness:** if NULL's selected checkpoint passes vs the PPO clone, H-SEL is `UNDETERMINED
  (selection manufactures improvement)` — H-ANC readings (no selection) are unaffected.
- **Determinism note:** SEL-7 vs S1's CP probe curve is reported as reproduced / not reproduced;
  it changes no verdict.
- SBLONG: `structured_verdict` (IMPROVED / LEARNED / FAILED) against S1's bars; reported only.
- A crashed arm is `UNDETERMINED` with its probe history preserved.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H-ANC passes (either β)** | A learned pursuer that improves on its clone with no selection exists. The artefact for the paper is the anchored PPO pursuer; if it also beats the teacher, that is the headline number. Route C withdrawn. Next pre-registration: the shield on this pursuer (Step 8 as planned). Prefer the β whose final checkpoint scores higher; report both. |
| **H-SEL passes, H-ANC fails at both β** | The rise is real and reproducible but the anchor did not hold it: report the selection-based artefact honestly (it needs the gate) and pre-register the next remedy from the ANC drift traces (β too strong → learning suppressed; too weak → drift). Route C withdrawn on H-SEL. |
| **H-ANC passes, H-SEL fails** | The rise is not reliably selectable across seeds but the anchor holds it: same as row 1, with the seed-variance stated. |
| **Both fail, NULL fails** | The 0.645 was seed-7-specific or a stream artefact. Report it as such; Route C stands re-scoped (D108/D109). SBLONG is read on its own. |
| **NULL passes** | Selection manufactures improvement at this probe density; H-SEL is not read; only H-ANC's final-checkpoint readings count. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is
seen. β ∈ {0.1, 0.01}, the selection rule, the gate rule, the seeds and the streams are fixed by
this document. No checkpoint other than "selected by the rule" and "final" is ever gated.

**Amendment A1 (2026-09-13, before `--measure-start`; from the whole-branch review; no threshold, seed, stream or budget changes):**
- **The anchor-KL traces are not commensurable with the §1 autopsy table.** §1 reports KL(clone ‖ π) on the *clone's own* states; the implemented and logged quantity (`anchor_kl`) is KL(π_θ ‖ π_anchor) on the *rollout* observations — the reverse divergence on a drifting state distribution, exactly as D102's `KLAnchoredSAC` defines it. The 0.6–1.7 / 5–9 landmarks of §1 must NOT be read off the figure's anchor-KL panel; the traces are read only relative to each other (β = 0 vs 0.1 vs 0.01).
- **Determinism note.** SEL7 vs S1's CP is a per-mark comparison of checkpoint weights. CPU PyTorch training is not thread-count invariant, so the comparison is conclusive only at S1's thread count. SEL7 is therefore launched **alone first with `OMP_NUM_THREADS` unset** (S1's launcher set none; torch default), then the remaining ten arms at `OMP_NUM_THREADS=2`. Thread count is recorded in every arm record. A "not reproduced" result at a different thread count is inconclusive, not a finding.
- **`≥ 2 of 3` with a missing seed.** The runner reads a hypothesis as `INCOMPLETE` if any of its three seed records is missing (crashed), even when two present seeds pass — more conservative than the literal §5 wording; recorded here so it is not read as a rule change after the fact.
- Bars are keyed `PPO_CLONE` / `SAC_CLONE` / `TEACHER` in the artefacts (the spec's `GATE_*` names).

**Amendment A2 (2026-09-13 17:25, after the bars, before any arm record; launcher error corrected):** amendment A1 stated that S1's CP ran at the torch-default thread count. That was a factual error: S1's four arms were launched with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` (session transcript, S125). SEL7 was therefore mis-launched with the variable unset and ran 100k env steps (one probe seen: 104/200 = 0.520 on 1 040 000+; ~8 min) before the error was noticed. It was stopped; its partial artefacts are archived, not deleted, under `results/derisk/archive_s2_sel7_omp_default_aborted/`. SEL7 is relaunched at `OMP_NUM_THREADS=2`, the same as S1's CP and as every other S2 arm, so the determinism check is conclusive and all eleven arms share one thread count. The one probe value seen played no part in this decision, which is forced by the thread-count discovery alone; no threshold, seed, stream, rule or budget changes.

## 7. What this round does NOT do

No shield, no evader family, no belief state, no structured-learner redesign (SBLONG is S1's SB
unchanged, longer), no third learner family. If H-ANC passes, the hidden-state defect
(D79/D105) is still the pursuer's known ceiling and is the next question after the shield.

## 8. Artefacts

`results/derisk/s2_start.json` (bars + paired gate logs), `s2_probe_{ARM}.json`,
`s2_arm_{ARM}.json` (selected + final gate results, verdict), `s2_round.{json,png}`, checkpoints
under `results/derisk/ckpts_{ARM}/` with ARM ∈ {SEL7, SEL8, SEL9, ANC01_7, …, ANC001_9, NULL7,
SBLONG}. Figure: all 1M-arm probe curves by family, the three gate bars, selected checkpoints
marked; a second panel with `anchor_kl` per family; SBLONG on its own axis against S1's bars.
Signed in the worklog.
