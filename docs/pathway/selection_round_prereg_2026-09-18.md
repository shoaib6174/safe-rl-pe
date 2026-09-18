# S12 — selection round: does "train six, select one, confirm" reliably deliver the resist-and-hunt pursuer? (2026-09-18, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D120** (S10: the β = 0.01 anchored fine-tune reaches the frontier on 3/5 fresh seeds; with S8, 4/8;
artefact FT-B-41; fresh attackers cap at 0.41), **D110** (S2 H-SEL: selecting a checkpoint on one stream and
confirming it on another is a valid, pre-registered procedure), **D117** (paper plan).

## 1. Why this round exists

The recipe works on about half its seeds. A method the paper can recommend must say what to do about that. The
natural answer is a selection procedure: train a small batch, gate every member on a selection stream, keep
the best by the pre-registered criterion, and confirm the kept one on a stream it has never seen. This round
runs that procedure **twice, independently**, and asks whether both runs deliver.

**H6:** the procedure "train 6 seeds of the S10 recipe → select by the two-sided criterion on a selection
stream → confirm on a fresh stream" delivers a policy that passes the two-sided test against RC19 on the
confirmation stream in **both** independent batches.
**H6-rate (reported):** the per-seed frontier rate over all β = 0.01 seeds run to date (S8: 3, S10: 5, S12: 12
→ 20 seeds) with a Wilson interval.
**H-SAFE:** 0 pursuer contact steps everywhere.

## 2. Protocol

- Recipe: exactly S10's arm (RC configuration, init RC19, anchor RC19, β = 0.01, `WEIGHTS`/`MIXTURE` as S8,
  5 M env steps, probes every 500 k vs Greedy shielded on **1 320 000+**). Frozen worktree, `OMP_NUM_THREADS=2`.
- **Batch A** seeds `48, 49, 50, 51, 52, 53` (arms `SEL-A-{s}`); **Batch B** seeds `54, 55, 56, 57, 58, 59`
  (arms `SEL-B-{s}`). Twelve arms, run concurrently.
- **Selection stream 1 290 000+** (N = 500): every arm is gated against the held-out attackers `ATT25/26/27`,
  `GreedyEvader` and `ObstacleHuggingEvader`, paired with **RC19** measured on the same stream before any arm.
- **Selection rule (per batch):** among arms whose per-arm two-sided verdict on the selection stream is `PASS`
  (S8's `_arm_verdict`: deadlock lower than RC19 on ≥ 2/3 held-out attackers AND Greedy capture not
  significantly below RC19), the lowest mean held-out deadlock; ties → highest Greedy capture; ties → lowest
  seed. If a batch has no passing arm, it selects the arm with the lowest mean held-out deadlock among those
  whose capture guard holds, and the batch's confirmation is expected to FAIL (reported as such).
- **Confirmation stream 1 300 000+** (N = 500): the two selected policies, **RC19** and **FT-B-41** (S10's
  artefact, reference only) are gated; the selected policy is tested against RC19 with the same two-sided
  test. Nothing selected on 1 290 000+ is ever judged on it.
- Fresh attackers are NOT trained in this round (the race was measured in S10; the selected policies are the
  same recipe).

## 3. Verdicts (fixed now)

- **H6 PASSES** iff both selected policies pass the two-sided test on the confirmation stream. `PASS (1/2)` if
  exactly one does. `FAIL` if neither. `INCOMPLETE` if any of the 12 arms lacks a selection gate or either
  confirmation gate is missing.
- **Artefact:** the selected policy with the lower mean held-out deadlock on the confirmation stream, if it
  passed; if only one passed, that one; the paper's artefact is then the better of {this, FT-B-41} on the
  confirmation stream, reported, not re-selected.
- Reported without verdict: per-seed selection-stream verdicts (12), the selection → confirmation shrinkage of
  each selected policy (deadlock and capture on both streams), FT-B-41 on the confirmation stream, anchor-KL
  traces, the pooled frontier rate with Wilson interval.
- H-SAFE fails ⇒ stop.

## 4. What each outcome fixes

| result | direction |
|---|---|
| **H6 PASSES** | The method is "train six, select, confirm", with a measured per-seed rate and a two-for-two confirmation: the paper's R3 procedure. |
| **PASS (1/2)** | The procedure delivers about half the time at batch size six; the paper reports the rate and recommends a larger batch, without re-running. |
| **FAIL** | Selection on one stream does not transfer: the frontier policies of S8/S10 were stream-specific; the D120 reading is withdrawn to "frontier reachable, not selectable". |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number is seen. Consumed streams
before this round: ≤ 1 289 999.

## 5. Artefacts

`results/derisk/s12_start.json` (RC19 on the selection stream), `s12_arm_SEL-{A,B}-*.json`, `s12_probe_*`,
`ckpts_SEL-*/`, `s12_selection.json`, `s12_confirm_start.json` (RC19, FT-B-41 on the confirmation stream),
`s12_confirm_SEL-*.json`, `s12_round.{json,png}`. Signed in the worklog.
