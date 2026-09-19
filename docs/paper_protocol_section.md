# Pre-registered experimentation as method — section draft (2026-09-17)

Priority 5 of the paper plan (D117). Every experimental claim in this paper was produced under the protocol
below; the artefacts are in the repository (`docs/pathway/*_prereg_*.md`, `docs/decisions.md`).

## Protocol

1. **Pre-registration before any number.** Each round has a specification committed to git before any arm
   runs. It fixes the hypothesis, the arms, the seeds, the evaluation streams, the sample sizes, the statistical
   tests, the pass thresholds and the interpretation of every outcome (a "what each result fixes" table). Seventeen
   such specifications cover the rounds reported here.
2. **Bars are measured, not inherited.** Reference policies are re-evaluated on the round's own fresh stream
   before any arm starts; an arm refuses to run without that file. Numbers are never carried across streams.
3. **Fresh, paired, disjoint streams.** Every evaluation uses episode seeds from a stream no training or
   earlier evaluation touched (a running watermark, ≤ 1 249 999 at S10). Within a round all policies see the
   same episodes, so comparisons are paired: McNemar / exact binomial on the discordant episodes, with Wilson
   intervals reported alongside.
4. **Stop rule.** Nothing is re-run, extended, retuned or re-seeded after a number is seen. A crashed arm is
   `UNDETERMINED` with its probe history preserved. Amendments are permitted only before the affected number
   exists and are recorded in the specification with a timestamp.
5. **Controls in every round.** Each round carries a negative or harness control (zero-reward learner,
   unrewarded structured attacker, scripted comparator, from-scratch control) whose failure validates the
   measurement; a round whose control does not behave is reported as harness-invalid, not as a result (S6 → S6b).
6. **Safety as a kill criterion.** Every round counts pursuer contact steps in training, probes, gates and
   comparators; a single contact step under the shield stops the programme. Total: zero contact steps over
   roughly 500 M shielded environment steps across S4–S10.
7. **Frozen code.** Rounds run from a detached git worktree at the pre-registered commit, with provenance
   (commit, library versions, thread counts) written into every record.
8. **A decision log.** Each round's outcome is one numbered entry (D108–D119) that states the verdict by the
   pre-registered rule first, then the reading, then every number with its source file.

## What it bought

- Two rounds that "worked" were correctly downgraded: S1's structured learner was `FAILED` by its rule and only
  later shown to be an update-count artefact (D109 → D110); S8's single passing seed is an existence proof, not
  a result, and triggered a confirmatory round instead of a claim (D119 → S10).
- One round that "worked" was correctly re-read: S7 passed both pre-registered tests and the paper still says
  the defender won by disengaging, because the capture guard was a reported label (D116); the next round made
  it a pass condition.
- Two harness failures were caught before interpretation (S6's comparator, S9's pairing assumption).

## Cost

About one round-trip of writing per round (specification, review, amendment log) and a strict no-retune rule
that left β = 0.01's one-in-three success rate as a measured fact rather than a tuned-away one.

## Limitation, added 2026-09-19 (D125)

This protocol did not catch its own biggest error. Fifteen pre-registered specifications, negative controls,
seed-paired gates, stop rules and adversarial reviews all passed while the programme measured an artefact: a
memoryless policy frozen in an absorbing observation loop, unable to escape an immobility the filter would have
released the moment it asked to turn the other way. Every statistical safeguard here operates on numbers that
were already the wrong numbers. The error was found by a person watching the rendered behaviour and saying the
pursuer looks stuck, which took one viewing. The lesson is not that pre-registration failed but that it is
orthogonal to construct validity: it disciplines inference from a measurement, and says nothing about whether
the measurement means what you think. A behavioural inspection gate — watch the policy act, on unselected
seeds, before building interpretation on its summary statistics — belongs in the protocol alongside the
statistical ones.
