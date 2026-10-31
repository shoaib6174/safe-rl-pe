# Paper A (tactic game, RA-L) — recommendations received from session `safe-rl-pe-f0` on 2026-11-02 (RECORDED, NOT REVIEWED)

User instruction (2026-11-02): these were written for the tactic-game paper; record them for Paper A and do not review now.
Paper A work resumes after the L4DC deadline (Nov 13). Nothing below has been checked, adopted or acted on; S25 stays
blind (no rates opened, no change to the sweep or its GIF-before-summary protocol).

## The six recommendations, verbatim in substance
1. **Safety proposition.** From a contact-free state u = 0 leaves the pose unchanged, so every hard row holds; the
   hard-admissible pool is never empty, the backup tier is unreachable, each executed action has h(x+) ≥ (1−γ)h(x) > 0,
   and by induction the pursuer stays contact-free; `RecoveryShieldV2` inherits this because it executes only filter
   outputs; for δ > 0 the start must satisfy h_δ > 0. Scope: true state, known static map, pursuer only. Proposed to
   replace the word "certified". (Flag for Paper B's §4 as a possible Lemma — to be checked against `pe2/shield.py`,
   including the relaxed tier, when B's draft reaches that section; not checked now.)
2. **Orbit-speed condition.** An evader orbiting at radius R + a holds angular separation from a pursuer kept at
   R + a + δ when ρ ≥ ρ* = (R + a)/(R + a + δ); ignores turn-around time; not a game result.
3. **Recast in control language:** a learned supervisor switching between closed-loop controllers, the filter as a
   state-dependent input constraint.
4. **Build the tactic paper around a value-vs-margin figure from S25**; ORBIT-SEARCH and per-ratio training to a
   sentence each; pre-registration to one paragraph.
5. **Possible risk to D157:** at ρ = 1.2, δ = 0, ρ* = 1, so geometry alone may favour the evader and "the filter sets the
   value" may be overstated; S25's NONE cells test this; fallback thesis: ratio and margin jointly decide when the
   orbit wins. (Note for later: this is what S25's H-VAL-1 / H-VAL-3 were written to decide — see
   `value_sweep_prereg_s25_2026-11-01.md` §4; no new amendment made.)
6. **Proposed next steps:** (a) record the geometric prediction in the S25 prereg while blind; (b) fix the scripted
   evader and finish the derivation; (c) draft around whichever thesis S25 supports.

## Their evidence (scripted pilots, no training, no S25 records opened)
Script `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/e5d31664-d84c-4ca3-a3f1-8adc0fd9707e/scratchpad/l4dc_checks.py`
(scratchpad — may be cleared). A: `RecoveryShieldV2` 12 000 steps at δ ∈ {0, 0.1, 0.2}: all exact tier, 0 contacts.
B: capture fraction, n = 12 per cell, 600 steps, one obstacle, R ∈ {0.3, 0.6, 1.0}, columns ρ = 0.8/0.9/1.0/1.1/1.2 —
greedy none 1.00/.75/.25/.25/.67; greedy v2 d0 .92/.42/.17/.25/.25; greedy v2 d.2 .50/0/0/0/.67; arc none
1.00/.92/.17/.67/.67; arc v2 d0 1.00/.83/.08/.17/.17; arc v2 d.2 .08/0/0/0/0. Their own caveats: capture rises again
at ρ > 1 (suspected evader turn-rate limit, unverified); greedy rows fit the margin story less well; 600 steps is
half the real horizon.

## Open questions they asked (to answer after Nov 13)
Q1 is #1 usable for Paper B's formal section (check the relaxed tier); Q2 is #5 a real threat to D157; Q3 is #6(a) a
legitimate blind amendment to the S25 prereg; Q4 anything wrong or already settled by a D-number.
