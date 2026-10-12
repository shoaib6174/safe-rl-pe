"""`pe2/selector_mdp.py` + `scripts/derisk_s24_1_train.py` — S24.1 Step 1.1.

Spec: `docs/pathway/tactics_prereg_s24_1_2026-10-13.md` sections 2, 3, 5 AND
Amendment A1 (A1.1 `OBS_LEN` 55 with the shield/escape features kept by NOMEM;
A1.2 a switch needs both spans to be >= 1 env step) AND Amendment A2 (A2.1
no tactic that would be DONE on entry is offered; A2.4 the probe rows carry the
training env's counters, `SystemExit` aborts, and the recorder's span scope is
live).

Everything here is either a constant the pre-registration fixes or one of the
properties the round's construct gates rest on:

- the observation layout: its length, its blocks, and that the blocks tile it
  exactly (a drifted layout silently re-labels the policy's inputs);
- CG-T1: `action_masks()` IS `applicable()` of the tactic each action resolves
  to, and an inapplicable pick is refused AND counted;
- the reward is the env's terminal +-1 summed over the span and nothing else
  (prereg section 3, D138) — no shaping anywhere in the chain;
- the decision points are the current tactic's DONE and the T = 40 cap, which
  is `pe2.selectors._Capped`'s and not a second copy of it;
- `NOMEM` zeroes EXACTLY the memory block: the control differs from the full
  selector in the inputs and in nothing else, including the dynamics;
- `play_episode` is deterministic for a fixed `predict` (the paired-seed
  protocol every gate in this project uses);
- the training script's arms, budget, streams and hyper-parameters are the
  pre-registered ones, it prints no capture rate, and one 1 000-decision
  `MaskablePPO` smoke learns and writes the probe JSON schema.

No real 3 M run: the smoke is 1 000 decisions with a tiny rollout.
"""

import importlib.util
import inspect
import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.recogniser import FEATURES, PURSUER_TACTICS
from pe2.selector_mdp import (
    ACTIONS,
    MEMORY_SLICES,
    N_ACTIONS,
    OBS_LAYOUT,
    OBS_LEN,
    OBS_SLICES,
    SelectorMDP,
    count_switches,
    done_on_entry,
    spans_with_length,
)
from pe2.selectors import DECISION_PERIOD

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s24_1_train.py"
)
SEED = 2_350_000


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s24_1_train", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _greedy_predict(index: int):
    """A `predict` in the `MaskablePPO` contract that wants `index` and falls
    back to the first applicable action when it is masked out."""

    def predict(obs, deterministic=True, action_masks=None):
        if action_masks is not None and not action_masks[index]:
            return int(np.flatnonzero(action_masks)[0]), None
        return index, None

    return predict


def _roll(env, n_decisions, rng):
    """`n_decisions` uniformly-random APPLICABLE decisions; stops at the end of
    the episode. Returns the (observation, mask, reward, env_steps) per step."""
    out = []
    for _ in range(n_decisions):
        mask = env.action_masks()
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, term, trunc, info = env.step(action)
        out.append((obs, mask, reward, info["env_steps"], term or trunc))
        if term or trunc:
            break
    return out


# ----------------------------------------------------------------------------
# the observation layout
# ----------------------------------------------------------------------------
def test_observation_length_is_pinned():
    """A1.1: 55 = 1 visible + 2 range/bearing + 3 dead reckoning + 4 own pose
    + 6 two obstacles + 1 wall + 11 features/valid + 6 current tactic
    + 15 last three + 1 speed ratio + 3 shield + 2 escape."""
    assert OBS_LEN == 55
    env = SelectorMDP(1.2)
    assert env.observation_space.shape == (OBS_LEN,)
    obs, _info = env.reset(seed=SEED)
    assert obs.shape == (OBS_LEN,)
    assert obs.dtype == np.float32


def test_layout_tiles_the_vector_exactly():
    assert sum(w for _n, w, _m in OBS_LAYOUT) == OBS_LEN
    covered = np.zeros(OBS_LEN, dtype=int)
    for block in OBS_SLICES.values():
        covered[block] += 1
    assert (covered == 1).all()  # no gap, no overlap
    assert OBS_SLICES["features"].stop - OBS_SLICES["features"].start == len(FEATURES) + 1
    assert OBS_SLICES["last_tactics"].stop - OBS_SLICES["last_tactics"].start == 3 * N_ACTIONS


def test_shield_and_escape_blocks_are_the_games_own_features():
    """A1.1: read off `SearchMDP.shield_features()` / `escape_features()`, so
    the definitions and the clipping are S16/S17's, not a second copy."""
    env = SelectorMDP(1.2)
    obs, _ = env.reset(seed=SEED)
    assert (obs[OBS_SLICES["shield"]] == env.game.shield_features()).all()
    assert (obs[OBS_SLICES["escape"]] == env.game.escape_features()).all()
    # (0, 2, 0) is `shield_features`'s "no shield acted" default, and the
    # escape pair is a real admissible range, never the unshielded (-1, 1)
    # sentinel unless the filter truly admits every turn.
    assert obs[OBS_SLICES["shield"]].tolist() == [0.0, pytest.approx(2.0), 0.0]
    rng = np.random.default_rng(6)
    seen = set()
    for step in _roll(env, 60, rng):
        shield = step[0][OBS_SLICES["shield"]]
        assert -0.5 <= shield[1] <= 2.0 and 0.0 <= shield[2] <= 2.0
        escape = step[0][OBS_SLICES["escape"]]
        assert (escape >= -1.0).all() and (escape <= 1.0).all()
        seen.add(tuple(np.round(shield, 6)))
    assert len(seen) > 1, "the shield block never moved: it is not live"


def test_action_space_is_the_preregistered_order():
    assert ACTIONS == ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
    assert ACTIONS is PURSUER_TACTICS  # imported, never restated
    assert SelectorMDP(1.2).action_space.n == 5


def test_speed_ratio_is_only_exposed_not_read():
    """Prereg section 2: ratio 1.2. The env takes it as a constructor argument
    and the observation's last entry IS it."""
    for ratio in (0.8, 1.0, 1.2):
        env = SelectorMDP(ratio)
        obs, _ = env.reset(seed=SEED)
        assert obs[OBS_SLICES["speed_ratio"]] == pytest.approx(ratio)
        assert env.cfg.evader_speed_ratio == ratio


# ----------------------------------------------------------------------------
# CG-T1: the mask
# ----------------------------------------------------------------------------
def test_masks_are_applicable_of_the_resolved_tactic():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        raw, game_env = env._obs_raw, env.game.inner.env
        visible = bool(raw[6] > 0.5)
        mask = env.action_masks()
        for i, tactic in enumerate(env._candidates(raw)):
            if tactic is None:
                assert not mask[i]
                continue
            applicable = bool(tactic.applicable(raw, game_env))
            gate = not done_on_entry(tactic, raw, game_env, env.cfg)  # A2.1
            if ACTIONS[i] == "COUNTER-ORBIT":  # reading 2: gated on visibility
                assert mask[i] == (applicable and visible and gate)
            else:
                assert mask[i] == (applicable and gate)
        # the independent facts the mask must carry: the two law tactics are
        # library-applicable always and offered whenever they have a branch to
        # act on (A2.1), and SWEEP is offered exactly while the evader is unseen
        for name in ("INTERCEPT", "LEAD"):
            tactic = env._candidates(raw)[ACTIONS.index(name)]
            assert tactic.applicable(raw, game_env)
            assert bool(mask[ACTIONS.index(name)]) == (
                not done_on_entry(tactic, raw, game_env, env.cfg)
            )
        assert bool(mask[ACTIONS.index("SWEEP")]) == (not visible)
        assert mask.any()  # a masked-out decision would deadlock MaskablePPO
        checked += 1
        _obs, _r, term, trunc, _i = env.step(int(rng.choice(np.flatnonzero(mask))))
        if term or trunc:
            break
    assert checked >= 5


def test_inapplicable_action_raises_and_is_counted():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    mask = env.action_masks()
    blocked = np.flatnonzero(~mask)
    assert len(blocked), "this seed's first decision admits every tactic"
    assert env.inapplicable_picks == 0
    with pytest.raises(AssertionError, match="not applicable"):
        env.step(int(blocked[0]))
    assert env.inapplicable_picks == 1


def test_a_full_episode_of_applicable_picks_logs_zero_violations_of_the_mask():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(1)
    _roll(env, 400, rng)
    assert env.inapplicable_picks == 0  # CG-T1


# ----------------------------------------------------------------------------
# the reward (prereg section 3, D138)
# ----------------------------------------------------------------------------
def test_reward_is_the_summed_terminal_and_nothing_else():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(2)
    rolled = _roll(env, 2000, rng)
    assert rolled[-1][4], "the episode must have ended inside the roll"
    rewards = [r for _o, _m, r, _s, _d in rolled]
    assert all(r == 0.0 for r in rewards[:-1])  # no shaping in any span
    assert abs(rewards[-1]) == pytest.approx(1.0)  # the env's terminal +-1
    assert sum(rewards) == pytest.approx(rewards[-1])


def test_no_shaping_ingredient_is_switched_on():
    env = SelectorMDP(1.2)
    game = env.game
    assert game.stall_penalty == 0.0
    assert game.intervention_penalty == 0.0
    assert game.zero_reward is False
    assert game._driver is game.inner  # reward_wrapper=None (v0b's assembly)
    # `SearchMDP._shaped` adds `r_acq` per acquisition onset and is the only
    # place shaping could enter; this env never routes a reward through it.
    for method in (SelectorMDP.step, SelectorMDP._advance):
        assert "_shaped" not in inspect.getsource(method)


# ----------------------------------------------------------------------------
# the decision points
# ----------------------------------------------------------------------------
def test_the_cap_is_the_selectors_module_and_a_span_never_exceeds_it():
    """Reading 3: T = 40 is `pe2.selectors._Capped`, not a second copy."""
    assert DECISION_PERIOD == 40
    assert "DECISION_PERIOD" in inspect.getsource(SelectorMDP._observe)
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(3)
    rolled = _roll(env, 300, rng)
    steps = [s for _o, _m, _r, s, _d in rolled]
    spans = np.diff([0, *steps])
    assert (spans >= 1).all()  # reading 4: every decision advances the episode
    assert spans.max() <= DECISION_PERIOD + 1


def test_a_decision_point_fires_at_forty_and_on_done():
    """`LEAD` is never DONE while the law has something to act on, so its span
    is the T = 40 cap; `SWEEP` is DONE the step the evader becomes visible, so
    its span is shorter at least once."""
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    lead, sweep = ACTIONS.index("LEAD"), ACTIONS.index("SWEEP")
    capped, early = [], []
    previous = 0
    for _ in range(60):
        mask = env.action_masks()
        action = sweep if mask[sweep] else lead
        _o, _r, term, trunc, info = env.step(action)
        span = info["env_steps"] - previous
        previous = info["env_steps"]
        (capped if span == DECISION_PERIOD else early).append((ACTIONS[action], span))
        if term or trunc:
            break
    assert capped, "no span ran to the 40-step cap"
    assert early, "no span ended on a tactic's own DONE"


def test_spans_of_the_pursuer_history_match_the_decisions():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(4)
    rolled = _roll(env, 50, rng)
    # One span per decision, plus the zero-length ones a DONE-on-entry tactic
    # leaves behind (reading 4); never fewer than one per decision.
    assert len(env.game.law.history) >= len(rolled)
    assert env.n_decisions == len(rolled)
    assert env.n_switches() <= len(env.game.law.history) - 1


# ----------------------------------------------------------------------------
# the NOMEM control (prereg section 3)
# ----------------------------------------------------------------------------
def test_nomem_zeroes_exactly_the_memory_block():
    """The same episode, the same actions: the memory blocks are zero in NOMEM
    and every other entry is bit-identical to the full selector's."""
    full, nomem = SelectorMDP(1.2, memory=True), SelectorMDP(1.2, memory=False)
    a, _ = full.reset(seed=SEED)
    b, _ = nomem.reset(seed=SEED)
    memory = np.zeros(OBS_LEN, dtype=bool)
    for block in MEMORY_SLICES:
        memory[block] = True
    assert memory.sum() == 35  # 3 dead reckoning + 11 features + 6 + 15
    # A1.1: the shield and escape blocks are observable, not memory.
    for name in ("shield", "escape"):
        assert not memory[OBS_SLICES[name]].any()
    rng = np.random.default_rng(5)
    moved = False
    for _ in range(60):
        assert (b[memory] == 0.0).all()
        assert (a[~memory] == b[~memory]).all()
        moved = moved or bool((a[memory] != 0.0).any())
        mask = full.action_masks()
        assert (mask == nomem.action_masks()).all()  # the dynamics are untouched
        action = int(rng.choice(np.flatnonzero(mask)))
        a, ra, ta, tra, ia = full.step(action)
        b, rb, tb, trb, ib = nomem.step(action)
        assert (ra, ta, tra, ia["env_steps"]) == (rb, tb, trb, ib["env_steps"])
        if ta or tra:
            break
    assert moved, "the full selector's memory block never became non-zero"
    # A1.1 again, on the live vector: NOMEM keeps the five new entries, and they
    # are not all zero by accident.
    kept = np.concatenate([b[OBS_SLICES["shield"]], b[OBS_SLICES["escape"]]])
    assert (kept != 0.0).any()


# ----------------------------------------------------------------------------
# A2.1: no decision may spend zero tactic steps
# ----------------------------------------------------------------------------
def test_a_never_seen_state_masks_the_law_tactics_and_offers_sweep():
    """A2.1: `INTERCEPT`/`LEAD` return None on entry when the evader is unseen
    and their law's last-seen is exhausted, so they are not offered; `SWEEP`'s
    `applicable()` holds exactly there, so the mask is never empty."""
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    raw = env._obs_raw
    assert raw[6] <= 0.5, "this seed starts with the evader visible"
    for tactic in (env.pursuer.intercept, env.pursuer.lead):
        assert tactic.law.last_seen is None  # never seen
        assert tactic.applicable(raw, env.game.inner.env)  # library-applicable
        assert done_on_entry(tactic, raw, env.game.inner.env, env.cfg)
    mask = env.action_masks()
    assert not mask[ACTIONS.index("INTERCEPT")]
    assert not mask[ACTIONS.index("LEAD")]
    assert mask[ACTIONS.index("SWEEP")]
    assert mask.any()


def test_done_on_entry_is_each_tactics_own_termination_test():
    """The predicate must not be a re-derivation: `INTERCEPT`/`LEAD` are asked
    about their OWN law's `last_seen`, and `AMBUSH` about the tactic's own
    visible-and-within-`AMBUSH_CLOSE` test (v0's rule-3 guard)."""
    from pe2.scripted import LAST_SEEN_TOL
    from pe2.tactics import AMBUSH_CLOSE

    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    raw, game_env = env._obs_raw.copy(), env.game.inner.env
    own = game_env.pursuer_pose
    lead = env.pursuer.lead
    # a last-seen point further than the tolerance: the law still has a branch
    lead.law.last_seen = np.asarray(own[:2]) + np.array([LAST_SEEN_TOL + 0.5, 0.0])
    assert not done_on_entry(lead, raw, game_env, env.cfg)
    # ... and one inside the tolerance: the law clears it and returns None
    lead.law.last_seen = np.asarray(own[:2]) + np.array([LAST_SEEN_TOL * 0.5, 0.0])
    assert done_on_entry(lead, raw, game_env, env.cfg)
    # each law tactic is asked about its own law, not a shared one
    env.pursuer.intercept.law.last_seen = None
    assert done_on_entry(env.pursuer.intercept, raw, game_env, env.cfg)
    # AMBUSH: only while the evader is VISIBLE and inside AMBUSH_CLOSE
    trap = env.pursuer.ambushes[0]
    raw[6] = 1.0
    game_env.evader_pose = np.array([own[0] + AMBUSH_CLOSE * 0.5, own[1], 0.0])
    assert done_on_entry(trap, raw, game_env, env.cfg)
    game_env.evader_pose = np.array([own[0] + AMBUSH_CLOSE + 1.0, own[1], 0.0])
    assert not done_on_entry(trap, raw, game_env, env.cfg)
    raw[6] = 0.0  # invisible: the tactic waits, it is not DONE
    game_env.evader_pose = np.array([own[0] + AMBUSH_CLOSE * 0.5, own[1], 0.0])
    assert not done_on_entry(trap, raw, game_env, env.cfg)
    # SWEEP and COUNTER-ORBIT are never reported DONE on entry by the predicate
    assert not done_on_entry(env.pursuer.sweep, raw, game_env, env.cfg)
    assert not done_on_entry(env.pursuer.counter_orbits[0], raw, game_env, env.cfg)


def test_four_hundred_decisions_leave_no_zero_length_span():
    """A2.1's invariant, measured: over a 400-decision random-applicable roll
    (several episodes), no span is zero-length and reading 4's search-policy
    branch never fires."""
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(11)
    zero_length = 0
    search_steps = 0
    decisions = 0
    episodes = 0
    while decisions < 400:
        for _step in _roll(env, 400 - decisions, rng):
            decisions += 1
        zero_length += sum(s["zero_length"] for s in env.spans())
        search_steps += int(env.game.usage["pursuer"].get("(search policy)", 0))
        search_steps += int(env.game.search_steps)
        episodes += 1
        if decisions < 400:
            env.reset(seed=SEED + episodes)
    assert episodes >= 1 and decisions == 400
    assert zero_length == 0, f"{zero_length} zero-length spans in 400 decisions"
    assert search_steps == 0, f"{search_steps} search-policy env steps"
    assert env.inapplicable_picks == 0


# ----------------------------------------------------------------------------
# A1.2: the switch count
# ----------------------------------------------------------------------------
def test_spans_with_length_flags_zero_length_spans():
    history = [("LEAD", 0, 40), ("INTERCEPT", 40, 40), ("SWEEP", 40, None)]
    spans = spans_with_length(history, step_index=55)
    assert [s["length"] for s in spans] == [40, 0, 15]
    assert [s["zero_length"] for s in spans] == [False, True, False]
    assert spans[-1]["end"] == 55  # the OPEN span is closed at `step_index`


def test_zero_length_spans_are_not_switches():
    """A1.2: `LEAD` -> (0-step `INTERCEPT`) -> `LEAD` is NOT two switches; it is
    none. Without the rule CG-T2's bar of 2 would be met by the DONE-on-entry
    spans alone."""
    history = [
        ("LEAD", 0, 40),
        ("INTERCEPT", 40, 40),
        ("INTERCEPT", 40, 40),
        ("LEAD", 40, 80),
    ]
    assert count_switches(spans_with_length(history, 80)) == 0
    naive = [("LEAD", 0, 40), ("INTERCEPT", 40, 41), ("LEAD", 41, 81)]
    assert count_switches(spans_with_length(naive, 81)) == 2  # real spans count
    # the family, not the instance: `COUNTER-ORBIT(0)` -> `COUNTER-ORBIT(1)`
    same = [("COUNTER-ORBIT(0)", 0, 20), ("COUNTER-ORBIT(1)", 20, 40)]
    assert count_switches(spans_with_length(same, 40)) == 0
    assert count_switches([]) == 0


def test_the_env_switch_count_uses_the_a1_2_rule():
    env = SelectorMDP(1.2)
    env.reset(seed=SEED)
    rng = np.random.default_rng(7)
    _roll(env, 60, rng)
    spans = env.spans()
    assert env.n_switches() == count_switches(spans)
    naive = count_switches([dict(s, zero_length=False) for s in spans])
    assert env.n_switches() <= naive
    if any(s["zero_length"] for s in spans):
        assert env.n_switches() < naive, "the rule made no difference to measure"


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def test_play_episode_is_deterministic_and_carries_the_gate_fields():
    env = SelectorMDP(1.2)
    first = env.play_episode(_greedy_predict(ACTIONS.index("LEAD")), SEED)
    second = SelectorMDP(1.2).play_episode(_greedy_predict(ACTIONS.index("LEAD")), SEED)
    for key in ("outcome", "steps", "violations", "n_switches", "n_decisions", "picks"):
        assert first[key] == second[key], key
    for key in (
        "seed",
        "role",
        "outcome",
        "steps",
        "final_distance",
        "los_breaks",
        "violations",
        "shield",
        "branch",
        "tactics",
        "n_switches",
        "inapplicable_picks",
        "picks",
        "pursuer_spans",
        "n_zero_length_spans",
    ):
        assert key in first, key
    assert first["n_zero_length_spans"] == sum(
        s["zero_length"] for s in first["pursuer_spans"]
    )
    assert first["n_switches"] == count_switches(first["pursuer_spans"])
    assert set(first["tactics"]) == {"evader", "pursuer"}
    assert first["inapplicable_picks"] == 0
    assert sum(first["picks"].values()) == first["n_decisions"]
    assert "n_steps" in first["branch"]  # the recorder's branch summary


def test_play_episode_replays_the_same_seed_twice_in_one_env():
    env = SelectorMDP(1.2)
    predict = _greedy_predict(ACTIONS.index("INTERCEPT"))
    a = env.play_episode(predict, SEED + 3)
    b = env.play_episode(predict, SEED + 3)
    assert (a["outcome"], a["steps"], a["n_switches"]) == (
        b["outcome"],
        b["steps"],
        b["n_switches"],
    )


def test_the_recorder_span_scope_is_live():
    """A2.4: this env drives `_step_inner` itself, so without setting
    `_in_span` and accumulating `_span_total` every step record reads
    `in_span: False` and `span_intervention_total` is identically 0."""
    env = SelectorMDP(1.2)
    env.reset(seed=SEED + 4)  # a seed whose episode the filter really acts on
    rng = np.random.default_rng(12)
    _roll(env, 400, rng)
    summary = env.game.branch_summary()
    assert summary["intervention_total"] > 0.0, "the shield never intervened"
    assert summary["span_intervention_total"] > 0.0
    # Every env step of this game falls inside a decision's span (the first
    # decision is taken at env step 0), so the two must agree and the
    # pre-decision remainder must be zero.
    assert summary["span_intervention_total"] == pytest.approx(
        summary["intervention_total"], rel=1e-6
    )
    assert summary["pre_decision_intervention"] == pytest.approx(0.0, abs=1e-9)
    assert all(rec["in_span"] for rec in env.game.step_records)


# ----------------------------------------------------------------------------
# the training script
# ----------------------------------------------------------------------------
def test_the_arms_and_the_budget_are_the_preregistered_ones():
    assert set(R.ARMS) == {
        "SEL-80",
        "SEL-81",
        "SEL-82",
        "NOMEM-80",
        "NOMEM-81",
        "NOMEM-82",
    }
    assert R.SEEDS == (80, 81, 82)
    assert all(R.ARMS[f"SEL-{s}"]["memory"] for s in R.SEEDS)
    assert not any(R.ARMS[f"NOMEM-{s}"]["memory"] for s in R.SEEDS)
    assert (R.TOTAL, R.EVERY, R.PROBE_SEED0, R.PROBE_N) == (
        3_000_000,
        250_000,
        2_350_000,
        200,
    )
    assert R.TOTAL % R.EVERY == 0
    assert (R.RATIO, R.GAMMA, R.OPPONENT) == (1.2, 0.99, "TACTIC-V0")


def test_the_streams_are_disjoint():
    """Training `seed * 1000`, probe 2 350 000+, gate 2 400 000+ (prereg 4)."""
    train = {R.ARMS[a]["seed"] * 1000 + i for a in R.ARMS for i in range(1000)}
    probe = set(range(R.PROBE_SEED0, R.PROBE_SEED0 + R.PROBE_N))
    gate = set(range(2_400_000, 2_400_500))
    assert not (train & probe) and not (train & gate) and not (probe & gate)


def test_the_learner_is_s17_base_with_the_masked_head():
    env = R.make_env("SEL-80")
    assert env.memory is True
    assert env.game.inner.seed0 == 80_000
    model = R.make_learner("SEL-80", env)
    assert type(model).__name__ == "MaskablePPO"
    assert model.gamma == 0.99
    assert model.seed == 80
    # S17's `make_learner` is "SB3 defaults apart from the arm's gamma": every
    # other hyper-parameter must still be the library default.
    from sb3_contrib import MaskablePPO

    defaults = inspect.signature(MaskablePPO.__init__).parameters
    for name in ("n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef"):
        want = defaults[name].default
        got = getattr(model, name)
        assert got == want or callable(got), name


def test_train_evidence_reads_the_training_env(tmp_path, monkeypatch):
    """A2.4: the two counters come off the TRAINING env through the VecEnv, so
    a crashed arm's last probe row still carries CG-T1 / H-SAFE evidence."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: R.make_env("SEL-80")])
    vec.reset()
    evidence = R.train_evidence(vec)
    assert set(evidence) == {"train_inapplicable_picks", "train_violations"}
    assert evidence == {"train_inapplicable_picks": 0, "train_violations": 0}
    inner = vec.envs[0].unwrapped if hasattr(vec.envs[0], "unwrapped") else vec.envs[0]
    inner.inapplicable_picks = 7
    assert R.train_evidence(vec)["train_inapplicable_picks"] == 7


def test_a_systemexit_from_the_assembly_aborts_the_launch(tmp_path, monkeypatch):
    """A2.4: an assembly guard's refusal is about the CONFIGURATION, so it must
    abort instead of writing six UNDETERMINED records."""
    monkeypatch.setattr(R, "OUT", tmp_path)

    def guard(*_a, **_k):
        raise SystemExit("the gate record now reads PASS")

    monkeypatch.setattr(R, "train", guard)
    with pytest.raises(SystemExit):
        R.run(["SEL-80", "SEL-81"], "cpu")
    assert not list(tmp_path.glob("s24_1_arm_*.json"))

    def crash(*_a, **_k):
        raise RuntimeError("this arm crashed")

    monkeypatch.setattr(R, "train", crash)
    R.run(["SEL-80"], "cpu")  # a real crash is still UNDETERMINED, not an abort
    record = json.loads((tmp_path / "s24_1_arm_SEL-80.json").read_text())
    assert record["verdict"] == "UNDETERMINED"
    assert "this arm crashed" in record["reason"]


def test_the_stream_note_is_recorded():
    """A2.4, recorded and not changed: SB3 seeds the first reset with the PPO
    seed, so `seed * 1000` is spent on episode 0 and the declared stream starts
    one later."""
    source = _PATH.read_text()
    assert "episode 0 runs on the PPO seed" in source
    assert "seed*1000+1" in source


def test_the_script_prints_no_capture_rate():
    """Prereg section 6: `--cells` prints no rate. The only `print` that fires
    while an arm trains must not carry a measured quantity."""
    source = _PATH.read_text()
    for line in source.splitlines():
        if "print(" not in line and "f\"[" not in line:
            continue
        assert "capture_rate" not in line, line
        assert "win_rate" not in line, line


def test_resume_is_arm_granular(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    assert R._resume_from("SEL-80") == (None, 0, [])
    d = tmp_path / "ckpts_S24-1-SEL-80"
    d.mkdir(parents=True)
    for mark in (250_000, 500_000):
        (d / f"S24-1-SEL-80_{mark}.zip").write_bytes(b"")
    R._write_json(
        tmp_path / "s24_1_probe_SEL-80.json",
        {"arm": "SEL-80", "checkpoints": [{"mark": 250_000}, {"mark": 500_000}]},
    )
    ckpt, mark, rows = R._resume_from("SEL-80")
    assert mark == 500_000
    assert ckpt.name == "S24-1-SEL-80_500000.zip"
    assert [r["mark"] for r in rows] == [250_000, 500_000]
    (d / f"S24-1-SEL-80_{R.TOTAL}.zip").write_bytes(b"")
    assert R._resume_from("SEL-80")[0] is None  # finished: nothing to resume


def test_provenance_pins_the_prereg_and_the_frozen_baseline():
    p = R.provenance()
    assert p["prereg"] == "docs/pathway/tactics_prereg_s24_1_2026-10-13.md"
    assert p["prereg_sha256"] and len(p["prereg_sha256"]) == 64
    assert p["baseline_record"].endswith("s24_tactic_game_v0b.json")
    assert p["baseline_sha256"] and len(p["baseline_sha256"]) == 64
    assert p["pe2_file"].endswith("pe2/__init__.py")
    assert p["obs_len"] == OBS_LEN
    assert p["label_branch"].startswith("labels: none")


def test_maskable_ppo_smoke_learns_and_writes_the_probe_schema(tmp_path, monkeypatch):
    from sb3_contrib import MaskablePPO

    monkeypatch.setattr(R, "OUT", tmp_path)
    env = R.make_env("SEL-80")
    model = MaskablePPO(
        "MlpPolicy", env, seed=80, gamma=R.GAMMA, n_steps=128, batch_size=64, verbose=0
    )
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000
    assert env.inapplicable_picks == 0  # CG-T1 through a real rollout
    monkeypatch.setattr(R, "PROBE_N", 2)
    probes = R._Probe("SEL-80", [])
    probes.at(model, env.env_steps, R.EVERY, R.train_evidence(model.get_env()))
    row = probes.rows[0]
    for key in (
        "train_inapplicable_picks",
        "train_violations",
        "opponent",
        "n",
        "k",
        "capture_rate",
        "n_switches_mean",
        "n_zero_length_spans_mean",
        "usage_steps",
        "picks",
        "inapplicable_picks",
        "violations",
    ):
        assert key in row, key
    assert row["n"] == 2
    assert row["train_inapplicable_picks"] == 0  # CG-T1 through a real rollout
    assert set(row["picks"]) == set(ACTIONS)
    written = json.loads(R._probe_file("SEL-80").read_text())
    assert written["checkpoints"][0]["mark"] == R.EVERY
    assert written["checkpoints"][0]["ckpt"].endswith(f"S24-1-SEL-80_{R.EVERY}.zip")


def test_cfg_defaults_are_the_frozen_ones():
    """Prereg section 2: `EnvConfig()` at ratio 1.2, two obstacles."""
    env = SelectorMDP(1.2)
    assert env.cfg == EnvConfig(evader_speed_ratio=1.2)
    assert env.cfg.n_obstacles == 2
    assert type(env.game.shield).__name__ == "RecoveryShieldV2"
