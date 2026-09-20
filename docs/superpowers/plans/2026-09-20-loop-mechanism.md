# Loop Mechanism (S15) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument episodes at the per-step level, extract every stall entry, and compute the four pre-registered measurements M1–M4 with their predictions P1–P4.

**Architecture:** `scripts/derisk_s15_mechanism.py`, standalone (this round needs PER-STEP traces, which `evaluate_structured` does not return, so it drives its own episode loop rather than reusing the S9 sweep machinery). The loop mirrors the diagnostic at `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/v3_stoch.py` and `v1_qp.py`: a `SingleAgentWrapper(role="pursuer")` driven by a `StructuredPursuerPolicy`, with the attacker as a `StructuredEvaderPolicy` or a scripted class.

**Tech Stack:** as the other derisk runners; numpy, scipy not needed.

**Spec:** `docs/pathway/loop_mechanism_prereg_2026-09-20.md` (binding; §2 defines every threshold).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Pre-registered, never tunable: policies `FT-B-41` (`results/derisk/ckpts_FT-B-41/FT-B-41_5000000.zip`) and `RC19` (`results/derisk/ckpts_RC19/RC19_10000000.zip`), both `shield_obs=True`; opponents `ATK4-45` (`results/derisk/ckpts_ATK4-45/ATK4-45_10000000.zip` as `StructuredEvaderPolicy(model, cfg, hold=10, d_pos=D_POS)`) and `GREEDY` (`GreedyEvader`); `N = 200`; `SEED0 = 1_710_000`; `CBFShield(cfg)` at defaults; `HOLDS = (10, 1)` (10 for M1–M3, both for M4); `GAMMA = 0.99`; terminal reward +1 capture / −1 timeout.
- Definitions fixed by the spec and not to be adjusted: **stall entry** = first step of a run of ≥ 10 consecutive immobile steps (`sinfo["immobile"]`); **escape** = ≥ 5 consecutive steps with `v_exec > 0.1 * cfg.v_max` starting within 40 steps of the entry; prediction bars `P1_RETURN = 0.05`, `P1_CAPTURE = 0.15`, `P2_ACTION = 0.01`, `P3_RATIO = 0.8`, `P4_ABS = 0.05`.
- Per-step trace fields needed: `immobile`, `stalled`, `v_exec`, the observation vector, the emitted normalised action, evader visibility (`obs[6] > 0.5`), and the policy's per-state σ (see below).
- σ per state: get the policy distribution for the observation, e.g. `model.policy.get_distribution(obs_tensor).distribution.scale` — verify the exact accessor against the installed SB3 and say what you used. If per-state σ is not available for this policy class (a state-independent `log_std` would make M3 vacuous), **report that plainly and mark P3 `N/A (state-independent sigma)` rather than inventing a per-state number** — that is itself the answer to M3.
- H-SAFE: count pursuer contact steps; any contact is reported and the round stops.
- Never write under `results/derisk/` from tests or a smoke; redirect `OUT` to `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s15_smoke/`. Do not launch the real round; the controller does, from a frozen worktree.

---

### Task 1: the runner and its tests

**Files:** Create `scripts/derisk_s15_mechanism.py`; Test `tests/pe2/test_derisk_s15_mechanism.py`.

**Interfaces:** `CELLS` (policy × opponent × hold; 4 cells at hold 10 plus 4 at hold 1 = 8), `run_cell(name, device)` writing `s15_cells/<name>.json` atomically with `{name, policy, opponent, hold, n, seed0, entries: [...], per_episode: [...], immobile_step_rate, deadlock_episode_rate, capture_rate, violations, sigma_stalled, sigma_free, provenance}` where each entry in `entries` is `{episode, step, escaped, steps_to_escape, outcome, steps_remaining, return_to_go, action_deltas_median, obs_deltas_median, visible_fraction}`; `m1_verdict`, `m2_verdict`, `m3_verdict`, `m4_verdict` implementing P1–P4 exactly; `hsafe`; `direction`; `summarize`; `write_figure`; CLI `--cells [names] | --summarize [--device]`.

- [ ] **Step 1: write the failing tests.** Cover: the 8 cells and their naming; stall-entry extraction on a hand-built boolean trace (a run of 9 immobile steps is NOT an entry, 10 is; two entries separated by motion are two entries; an escape at exactly 40 steps counts, at 41 does not; 4 moving steps is not an escape, 5 is); `return_to_go` arithmetic on a hand-computed example (capture at step 900 from an entry at step 600 with γ = 0.99 → `0.99**300`, timeout → `−0.99**k`); each of `m1..m4_verdict` on fakes, both the holding and the failing side, and `m3_verdict` returning `N/A (state-independent sigma)` when σ is reported as state-independent; `hsafe` naming the cell; `direction` rows per spec §4; skip-existing.
- [ ] **Step 2: run them, verify they fail.**
- [ ] **Step 3: implement.** Drive episodes yourself; do not try to bend `evaluate_structured` into returning traces.
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: scratch smoke** — `N=6`, 2 cells (one learned opponent, one scripted), then `--summarize`: no exception, 0 violations, entries extracted, figure written, nothing under `results/derisk/`. Paste the output and state how many stall entries 6 episodes produced.
- [ ] **Step 6:** its tests green, ruff, commit `feat(derisk): S15 loop-mechanism runner`.

## Self-review
- §2 M1–M4 → the four verdict functions with the exact bars; §3 protocol → constants and the entry/escape definitions; §4 → `direction`; §5 → prefix `s15`. The per-state σ caveat is handled explicitly rather than faked.
