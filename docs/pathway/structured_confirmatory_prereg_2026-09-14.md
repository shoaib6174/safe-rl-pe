# S3 — confirmatory round: the structured learner across seeds (2026-09-14, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), D109 (S1: structured learner FAILED at
1 M env steps ≈ 36 PPO iterations, with the update-count caveat fixed in advance), **D110** (S2:
SBLONG — S1's SB continued to 10 M env steps — IMPROVED, 0.680 [0.638, 0.719] vs the orbit
teacher's 0.464; H-SEL PASS; Route C withdrawn), D106/D107 (the teacher and its clone).

## 1. Why this round exists

SBLONG was one training seed, pre-registered as a side arm with no claim attached. Its final
checkpoint beat the orbit bar by 22 points with no selection and no anchor, on S1's stream. A
lead from one seed is not a result. This round asks the confirmatory question with fresh seeds,
a fresh paired gate, and the same no-selection rule, and it fixes in advance how the paper's
pursuer is chosen so the shield round inherits one policy without a post-hoc choice.

**Hypothesis H-STR:** the structured learner — `SearchMDP` (fixed intercept law, D106's chase
and last-seen branches), PPO from scratch on the search sub-policy, terminal ±1 reward only,
10 M env steps — produces a pursuer that beats its scripted teacher, reliably across training
seeds.

## 2. Protocol

- Role: pursuer; opponent frozen `GreedyEvader`, `EnvConfig()` defaults (as every prior round).
- Learner: **exactly SBLONG's configuration** (spec S2 §4, D110): `SearchMDP(cfg, GreedyEvader,
  seed0 = seed × 1000, hold = HOLD, r_acq = 0.0, zero_reward = False)`; `PPO("MlpPolicy", …,
  seed = seed, gamma = 0.99)` with every other hyper-parameter at the SB3 default, saved as the
  arm's init before training (`s3_init_{seed}.zip`); **budget 10 000 000 env steps**, enforced
  in env steps by `StructuredEvalProbe`; probes every **1 000 000** at N = 200. **No selection:
  the final checkpoint is the arm's policy.**
- Streams (fresh; ≤ 1 050 999 consumed): **probes 1 060 000+**, **gate 1 070 000+** (N = 500),
  **family 1 080 000+** (N = 200 per evader, the same seeds for every policy within an evader).
- Thread count: every process at `OMP_NUM_THREADS=2` (amendment S2-A2), recorded in provenance.
- **Bars on the gate stream, N = 500, measured BEFORE any arm trains**, written to
  `results/derisk/s3_start.json` with their paired logs: `TEACHER` (`SearchingPursuer`, **the
  verdict bar**), `PPO_CLONE`, `SAC_CLONE`, `SEL7_100K` (S2's best selected checkpoint,
  `results/derisk/ckpts_SEL7/SEL7_100000.zip`, 0.616 on S2's gate), `GREEDY_FULLSTATE`
  (`GreedyPursuer`, the full-state reference — it reads the true evader pose). An arm refuses to
  run without this file; `--measure-start` refuses to re-run.
- **Gate rule (S2 §3, unchanged):** a checkpoint PASSES vs X iff the exact two-sided McNemar
  test vs X has p < 0.05 with more discordant wins than losses AND its Wilson lower bound exceeds
  X's point estimate on the gate stream.

## 3. The arms

| arm | seed | reward | purpose |
|---|---|---|---|
| **STR-11, STR-12, STR-13, STR-14, STR-15** | 11–15 | terminal ±1 | **the test**: five fresh seeds |
| **SBLONG** (existing) | 7 | terminal ±1 | the arm that generated the hypothesis: its final checkpoint `ckpts_SBLONG/SBLONG_10000000.zip` is gated on the new stream and **reported, not counted** |
| **SZLONG-11** | 11 | all zero (`zero_reward = True`) | **negative control** at the confirmatory budget: unrewarded drift for 10 M steps must not beat the teacher |

Seven processes concurrent (six training); ≈ 40 min.

## 4. Verdicts (fixed now)

- Per arm: `final_pass_vs_teacher` (the verdict), and reported `final_pass_vs` each of
  `PPO_CLONE`, `SAC_CLONE`, `SEL7_100K`, `GREEDY_FULLSTATE`, with the full gate record.
- **H-STR PASSES iff ≥ 4 of the 5 fresh seeds' final checkpoints pass vs `TEACHER`.**
- **Harness:** SZLONG-11's final checkpoint must NOT pass vs `TEACHER`; if it does, H-STR is
  `UNDETERMINED (harness)`.
- A missing seed record (crash) makes H-STR `INCOMPLETE` (conservative, as in S2-A1).
- **Artefact rule:** if H-STR passes, **the paper's pursuer is the passing fresh seed with the
  highest gate rate vs `TEACHER`** (ties → lowest seed). Written to `s3_round.json` as
  `artefact`. SBLONG is not eligible (it generated the hypothesis).
- **Secondary, reported only:** how many of the passing seeds also beat `SEL7_100K` (the best
  fine-tuned PPO checkpoint) and `GREEDY_FULLSTATE`.

## 5. Characterisation and generalisation (reported only, no gate)

For the artefact policy, `SBLONG`, and `TEACHER`:
- **Behaviour on the gate stream** (from instrumented rollouts): fraction of env steps with the
  evader visible, steps to first sight (median / p90), search-stretch length (median / p90),
  fraction of steps in the search branch, mean commanded speed and |ω| during search.
- **Opponent family** on 1 080 000+, N = 200 per evader, paired within each evader:
  `GreedyEvader`, `FovLimitedEvader`, `ObstacleHuggingEvader`, `OpenSpaceEvader` — capture rate
  with Wilson CI and McNemar vs `TEACHER` per evader. D107: `OpenSpaceEvader` is uncatchable
  (≈ 0.00–0.03 for every policy), so no family-level aggregate is computed; the reading is
  "beats the teacher on k of the 3 catchable evaders", reported.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H-STR PASSES** | The structured learner is the paper's learned pursuer: a policy that improves on its scripted teacher without selection, anchoring, or the possibility of destroying the capture law. Next pre-registration: **the shield on the artefact policy** (Step 8 as planned), with the S1-family evaluator. The characterisation table is the paper's "what was learned" section. |
| **H-STR fails, SZLONG fails** | SBLONG's 0.68 was seed-specific; report the seed distribution; the structured learner is a 1-of-6 result and the selection-based PPO artefact (D110 H-SEL) is the fallback pursuer for the shield round. |
| **SZLONG passes** | The harness is invalid at this budget (unrewarded drift beats the teacher inside the fixed law); nothing is read; diagnose before any further round. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is
seen. Seeds 11–15, the budget, the streams, the gate rule, the ≥ 4/5 rule and the artefact rule
are fixed by this document.

## 7. What this round does NOT do

No shield, no change to the learner or the observation (the hidden-state defect D79/D105 stays
the known ceiling), no anchor, no selection, no reward shaping.

## 8. Artefacts

`results/derisk/s3_start.json`, `s3_init_{seed}.zip`, `s3_probe_{ARM}.json`, `s3_arm_{ARM}.json`,
`s3_round.{json,png}` (probe curves of all six training arms against the five bars; a second panel
with the family table), `s3_characterisation.json`, checkpoints under `ckpts_{ARM}/` with ARM ∈
{STR11…STR15, SZLONG11}. Signed in the worklog.
