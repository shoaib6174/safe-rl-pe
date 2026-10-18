"""S24.2 Step E.1 — the evader seat's semi-MDP, and CG-E1.

Spec: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 1 + amendment A1.

CG-E1 is the load-bearing one. With `EvaderRulesV0` as the decision source the
new env must replay the stored `SEL-81_TACTIC-V0` episode (seed 2 400 000) BIT
FOR BIT — outcome, step count, violations, the pursuer's 13-span log and the
evader's 55-span log — against a record produced by `SelectorMDP` driving the
same game from the OTHER seat. That is the whole proof of A1.3's decision
mechanics: the pursuer's command computed once and held, the evader consulted
before the step and handed in through the one-shot proxy, and the frozen
checkpoint driven through `SelectorMDP`'s own observation/mask/candidate code.
A difference is a DEFECT, never a tolerance to widen.

The rest holds the seat's own contract: the observation layout, the mask (never
empty, never admitting a tactic that is inapplicable or DONE on entry), the
40-step cap, the zero-sum reward, the pursuer mixture and `play_episode`.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.evader_selector_mdp import (
    EV_OBS_LAYOUT,
    EV_OBS_LEN,
    EV_OBS_SLICES,
    EVADER_ACTIONS,
    EvaderSelectorMDP,
    done_on_entry,
)
from pe2.learned_select import LearnedPursuerSelect
from pe2.recogniser import EVADER_TACTICS
from pe2.selectors import DECISION_PERIOD

REPO = pathlib.Path(__file__).resolve().parents[2]
GATE_SEED0 = 2_400_000  # the S24.1 gate stream (that prereg, section 4)
RECORD = REPO / "results/derisk/s24_1_cells/SEL-81_TACTIC-V0.json"
CKPT = REPO / "results/derisk/ckpts_S24-1-SEL-81/S24-1-SEL-81_3000000.zip"
SEL_81 = ("ckpt", str(CKPT), "v0")  # library v0 is SEL-81's (commit 17d150b)
WALK_SEED0 = 2_500_000  # a stream that is NOT the gate's, for the walks


def _env(**kwargs) -> EvaderSelectorMDP:
    """`derisk_s24_1_gate.py::cell_config()`'s env config — `EnvConfig()`,
    whose ratio default IS the round's 1.2 — with nothing else varied."""
    cfg = EnvConfig()
    assert cfg.evader_speed_ratio == 1.2
    return EvaderSelectorMDP(cfg.evader_speed_ratio, cfg=cfg, **kwargs)


def _spans(log_or_record) -> list[list]:
    """A span log as plain lists, so a JSON round trip compares equal."""
    return [list(span) for span in log_or_record]


def _first_admissible(obs, deterministic=True, action_masks=None):
    """A deterministic masked `predict` shim: the lowest admissible action.
    Deterministic without a policy, so determinism tests measure the ENV."""
    return int(np.flatnonzero(action_masks)[0]), None


# ----------------------------------------------------------------------------
# CG-E1
# ----------------------------------------------------------------------------
def test_cg_e1_replays_the_stored_sel_81_tactic_v0_episode():
    record = json.loads(RECORD.read_text())
    assert (record["row"], record["opponent"]) == ("SEL-81", "TACTIC-V0")
    assert record["seed0"] == GATE_SEED0
    assert CKPT.exists(), f"CG-E1 needs the frozen checkpoint at {CKPT}"

    env = _env(pursuer=SEL_81, seed0=GATE_SEED0, decision_source="rules")
    log = env.play_episode(None, GATE_SEED0)

    assert (log["outcome"] == "capture") is bool(record["capture_ind"][0])
    assert log["steps"] == record["steps"][0] == 454
    assert log["violations"] == record["violations"][0] == 0
    assert _spans(log["tactics"]["pursuer"]) == _spans(record["tactics"][0]["pursuer"])
    assert len(log["tactics"]["pursuer"]) == 13
    # the EVADER half of the same stored record: the seat this step adds
    assert _spans(log["tactics"]["evader"]) == _spans(record["tactics"][0]["evader"])
    assert log["n_decisions"] == len(record["tactics"][0]["evader"])
    assert log["pursuer_name"] == CKPT.stem
    assert log["inapplicable_picks"] == 0


def test_cg_e1_drives_the_checkpoint_through_the_pursuer_selectors_own_code():
    """The replay above would also pass with a hard-coded pursuer, so the seat
    is checked for what A1.2 requires: a `LearnedPursuerSelect` over a
    `SelectorMDP` whose game IS this env's, taking one decision per span."""
    env = _env(pursuer=SEL_81, seed0=GATE_SEED0, decision_source="rules")
    (name, select), = env.seats
    assert isinstance(select, LearnedPursuerSelect)
    assert select.sel.game is env.game and select.sel.rules is env.rules
    assert select.sel.library == "v0" and select.sel.obs_len == 55
    log = env.play_episode(None, GATE_SEED0)
    assert env.game.law.select is select
    # one pursuer decision per pursuer span, and the picks are the families
    assert select.sel.n_decisions == len(log["tactics"]["pursuer"]) == 13
    assert sum(select.sel.picks.values()) == 13
    assert set(select.sel.picks) <= set(select.sel.actions)
    assert select.sel.inapplicable_picks == 0
    assert name == CKPT.stem


# ----------------------------------------------------------------------------
# the observation
# ----------------------------------------------------------------------------
def test_the_observation_layout_tiles_the_vector_and_is_the_mirror():
    assert EVADER_ACTIONS is EVADER_TACTICS  # imported, never restated
    widths = {name: width for name, width, _m in EV_OBS_LAYOUT}
    assert widths == {
        "hidden": 1,
        "pursuer_range_bearing": 2,
        "own_pose": 4,
        "obstacles": 6,
        "wall": 1,
        "features": 11,
        "current_tactic": 6,
        "last_tactics": 15,
        "speed_ratio": 1,
    }
    assert EV_OBS_LEN == 47 == sum(widths.values())
    covered = np.zeros(EV_OBS_LEN, dtype=int)
    for block in EV_OBS_SLICES.values():
        covered[block] += 1
    assert (covered == 1).all()
    # the prereg's absences, asserted rather than argued
    assert {"dead_reckoning", "shield", "escape"}.isdisjoint(widths)
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    obs, _info = env.reset(seed=WALK_SEED0)
    assert obs.shape == (EV_OBS_LEN,) and obs.dtype == np.float32
    assert env.observation_space.shape == (EV_OBS_LEN,)
    assert env.action_space.n == 5
    assert obs[EV_OBS_SLICES["speed_ratio"]] == pytest.approx(1.2)


def test_memory_false_zeroes_exactly_the_memory_blocks():
    """The ablation control: the three memory blocks read 0 and the block
    POSITIONS stay, so the two arms have an identical policy shape."""
    memory_names = [name for name, _w, mem in EV_OBS_LAYOUT if mem]
    assert memory_names == ["features", "current_tactic", "last_tactics"]
    rng = np.random.default_rng(0)
    full = _env(pursuer="RULES-V0", seed0=WALK_SEED0, memory=True)
    ablated = _env(pursuer="RULES-V0", seed0=WALK_SEED0, memory=False)
    seen_nonzero = False
    for env in (full, ablated):
        obs, _info = env.reset(seed=WALK_SEED0)
        for _ in range(6):
            mask = env.action_masks()
            obs, _r, term, trunc, _i = env.step(
                int(rng.choice(np.flatnonzero(mask)))
            )
            if env is ablated:
                for name in memory_names:
                    assert not obs[EV_OBS_SLICES[name]].any(), name
            else:
                seen_nonzero |= any(
                    bool(obs[EV_OBS_SLICES[name]].any()) for name in memory_names
                )
            if term or trunc:
                break
    assert seen_nonzero, "memory=True never populated a memory block"
    # the observable blocks are NOT zeroed by the flag
    obs_full, _ = full.reset(seed=WALK_SEED0)
    obs_abl, _ = ablated.reset(seed=WALK_SEED0)
    for name, _w, mem in EV_OBS_LAYOUT:
        if not mem:
            assert obs_full[EV_OBS_SLICES[name]] == pytest.approx(
                obs_abl[EV_OBS_SLICES[name]]
            ), name


def test_hidden_is_the_pursuers_flag_and_the_range_is_always_known():
    """Reading 5: `hidden` is the complement of the PURSUER's `obs[6]`, and the
    pursuer's range/bearing is full state, so it is populated on hidden steps
    too (unlike the pursuer seat's, which zeroes an unseen opponent)."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    rng = np.random.default_rng(1)
    hidden_seen = visible_seen = 0
    for episode in range(6):
        obs, _info = env.reset(seed=WALK_SEED0 + episode)
        for _ in range(12):
            hidden = bool(env._obs_p[6] <= 0.5)
            assert bool(obs[EV_OBS_SLICES["hidden"]][0] > 0.5) is hidden
            assert obs[EV_OBS_SLICES["pursuer_range_bearing"]][0] > 0.0
            hidden_seen += hidden
            visible_seen += not hidden
            mask = env.action_masks()
            obs, _r, term, trunc, _i = env.step(
                int(rng.choice(np.flatnonzero(mask)))
            )
            if term or trunc:
                break
    assert hidden_seen and visible_seen


# ----------------------------------------------------------------------------
# the mask (CG-T1 in this seat) and DONE-on-entry
# ----------------------------------------------------------------------------
def test_a_thousand_masked_steps_are_all_applicable_and_all_spend_a_step():
    """CG-T1 in the evader's seat: the mask is never empty, never admits a
    tactic that is inapplicable or DONE on entry, `inapplicable_picks` stays 0,
    and every decision spends at least one env step (A2.1's invariant, which is
    what the DONE-on-entry half of the mask exists for)."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    rng = np.random.default_rng(2)
    decisions = episode = 0
    while decisions < 1000:
        env.reset(seed=WALK_SEED0 + episode)
        episode += 1
        while decisions < 1000:
            obs, game_env = env._obs_e(), env.game.inner.env
            mask = env.action_masks()
            assert mask.shape == (5,)
            assert mask.any()  # a masked-out decision would deadlock the learner
            for i, tactic in enumerate(env._candidates(obs)):
                if not mask[i]:
                    continue
                assert tactic is not None
                assert tactic.applicable(obs, game_env)
                assert not done_on_entry(tactic, game_env)
            _o, _r, term, trunc, info = env.step(
                int(rng.choice(np.flatnonzero(mask)))
            )
            decisions += 1
            assert info["span_steps"] >= 1, info
            if term or trunc:
                break
    assert env.inapplicable_picks == 0
    assert env.violations_total == 0  # H-SAFE over the whole walk


def test_the_mask_refuses_an_inapplicable_pick_and_counts_it():
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    rng = np.random.default_rng(3)
    for episode in range(20):
        env.reset(seed=WALK_SEED0 + episode)
        for _ in range(20):
            mask = env.action_masks()
            if not mask.all():
                refused = int(np.flatnonzero(~mask)[0])
                before = env.inapplicable_picks
                with pytest.raises(AssertionError, match=EVADER_ACTIONS[refused]):
                    env.step(refused)
                assert env.inapplicable_picks == before + 1  # counted, then refused
                return
            _o, _r, term, trunc, _i = env.step(
                int(rng.choice(np.flatnonzero(mask)))
            )
            if term or trunc:
                break
    pytest.fail("no decision reached where some tactic was masked out")


def test_done_on_entry_is_each_tactics_own_termination_test_for_every_family():
    """`done_on_entry` is checked against the only ground truth there is — the
    tactic's own `act` on its first call of a span — at 400 decisions, with
    BOTH branches required for every family that has one. The probe calls the
    WRAPPED tactic (`_Capped.inner`), because the 40-step cap is deliberately
    not part of the predicate (a span boundary `enter()`s and zeroes it)."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    rng = np.random.default_rng(4)
    seen: dict[str, set[bool]] = {name: set() for name in EVADER_ACTIONS}
    decisions = episode = 0
    while decisions < 400:
        env.reset(seed=WALK_SEED0 + episode)
        episode += 1
        while decisions < 400:
            obs, game_env = env._obs_e(), env.game.inner.env
            for name, tactic in zip(EVADER_ACTIONS, env._candidates(obs), strict=True):
                if tactic is None:
                    continue
                predicted = done_on_entry(tactic, game_env)
                assert predicted is (tactic.inner.act(obs, game_env) is None), name
                seen[name].add(predicted)
            mask = env.action_masks()
            _o, _r, term, trunc, _i = env.step(
                int(rng.choice(np.flatnonzero(mask)))
            )
            decisions += 1
            if term or trunc:
                break
    assert seen["FLEE"] == {False}  # `GreedyEvader` always returns a command
    for name in ("ORBIT", "HOP", "WALL-SLIDE", "HOLD-HIDDEN"):
        assert seen[name] == {True, False}, (name, seen[name])


def test_done_on_entry_refuses_a_name_outside_the_library():
    class _Alien:
        name = "COUNTER-ORBIT(0)"

    with pytest.raises(ValueError, match="COUNTER-ORBIT"):
        done_on_entry(_Alien(), None)


# ----------------------------------------------------------------------------
# the decision period, the reward, the proxy
# ----------------------------------------------------------------------------
def test_flee_spans_end_at_the_forty_step_cap():
    """The decision period is `pe2.selectors._Capped`'s, the device the
    baseline ran under — not a second cap. `FLEE` is never DONE, so every one
    of its spans that the episode does not end must be exactly 40 env steps."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    index = EVADER_ACTIONS.index("FLEE")
    env.reset(seed=WALK_SEED0)
    lengths = []
    while True:
        assert env.action_masks()[index]  # always applicable
        _o, _r, term, trunc, info = env.step(index)
        lengths.append(int(info["span_steps"]))
        if term or trunc:
            break
    assert len(lengths) >= 2
    assert all(n == DECISION_PERIOD for n in lengths[:-1]), lengths
    assert lengths[-1] <= DECISION_PERIOD
    assert [s["length"] for s in env.spans()] == lengths


def test_the_reward_is_the_negative_of_the_games_own():
    """A1.1, pure zero-sum: the span reward is minus the sum of the rewards the
    game's `_step_inner` emitted over that span — asserted by intercepting
    `_step_inner` rather than by recomputing the env's reward."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    inner, pursuer_rewards = env.game._step_inner, []

    def spy(action, branch="law"):
        out = inner(action, branch)
        pursuer_rewards.append(float(out[1]))
        return out

    env.game._step_inner = spy
    rng = np.random.default_rng(5)
    env.reset(seed=WALK_SEED0)
    total = 0.0
    while True:
        before = len(pursuer_rewards)
        mask = env.action_masks()
        _o, reward, term, trunc, _i = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert reward == pytest.approx(-sum(pursuer_rewards[before:]))
        total += reward
        if term or trunc:
            break
    assert total == pytest.approx(-sum(pursuer_rewards))
    assert total == pytest.approx(-1.0 if env.game._info["outcome"] == "capture" else 0.0)


def test_the_proxy_is_consulted_exactly_once_per_env_step():
    """A1.3's hand-off: the wrapper's opponent is the one-shot proxy, it is
    asked exactly once per env step, and it never answers twice from one
    stash."""
    env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    rng = np.random.default_rng(6)
    env.reset(seed=WALK_SEED0)
    assert env.game.inner._opponent is env._proxy
    assert env.game.inner._opponent.uses_truth is env.game.evader.uses_truth
    assert env._proxy.n_calls == 0 and env.env_steps == 0
    while True:
        mask = env.action_masks()
        _o, _r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert env._proxy.n_calls == env.env_steps == info["env_steps"]
        if term or trunc:
            break
    with pytest.raises(AssertionError, match="stashed"):
        env._proxy.act(None, None)


# ----------------------------------------------------------------------------
# the pursuer seats
# ----------------------------------------------------------------------------
def test_the_scripted_seats_are_the_games_own_select_and_the_baseline_row():
    """A1.2: `RULES-V0` is the assembly's own `PursuerRulesV0` table and
    `LEAD-1.0` is `fixed(rules["pursuer"].lead)`, the S24.1 gate's BASELINE
    row — the same objects, not copies of them."""
    rules_env = _env(pursuer="RULES-V0", seed0=WALK_SEED0)
    assert rules_env.seats[0][1] is rules_env.rules["pursuer"]
    rules_env.reset(seed=WALK_SEED0)
    assert rules_env.game.law.select is rules_env.rules["pursuer"]

    lead_env = _env(pursuer="LEAD-1.0", seed0=WALK_SEED0)
    log = lead_env.play_episode(_first_admissible, WALK_SEED0)
    assert {span["name"] for span in log["pursuer_spans"]} == {"LEAD"}
    assert log["pursuer_name"] == "LEAD-1.0"


def test_an_unknown_seat_is_refused():
    with pytest.raises(ValueError, match="unknown pursuer seat"):
        _env(pursuer="GREEDY")
    with pytest.raises(FileNotFoundError):
        _env(pursuer=("ckpt", "results/derisk/ckpts_nope/nope.zip", "v0"))
    with pytest.raises(ValueError, match="decision_source"):
        _env(pursuer="RULES-V0", decision_source="oracle")


def test_a_three_seat_mixture_draws_each_and_is_reproducible():
    """A per-episode uniform draw from the episode RNG, which `reset` reseeds
    from the episode's own seed — so the seat a seed gets is a property of the
    seed, reproducible across env instances, and the unseeded training stream
    is still deterministic from `seed0`."""
    seats = ["RULES-V0", "LEAD-1.0", SEL_81]
    names = []
    env = _env(pursuer=seats, seed0=WALK_SEED0)
    assert len(env.seats) == 3
    for i in range(30):
        _obs, info = env.reset(seed=WALK_SEED0 + i)
        names.append(info["pursuer"])
        assert env.game.law.select is dict(env.seats)[info["pursuer"]]
    assert set(names) == {"RULES-V0", "LEAD-1.0", CKPT.stem}
    assert all(names.count(n) >= 1 for n in set(names))
    again = _env(pursuer=seats, seed0=WALK_SEED0)
    assert [again.reset(seed=WALK_SEED0 + i)[1]["pursuer"] for i in range(30)] == names
    # and the unseeded stream (SB3's vec-env reset path) is deterministic from
    # `seed0`, so a training run's mixture is reproducible without seeds
    streams = []
    for _ in range(2):
        stream_env = _env(pursuer=seats, seed0=WALK_SEED0)
        streams.append([stream_env.reset()[1]["pursuer"] for _ in range(8)])
    assert streams[0] == streams[1]
    assert len(set(streams[0])) > 1


# ----------------------------------------------------------------------------
# play_episode and a real rollout
# ----------------------------------------------------------------------------
def test_play_episode_is_deterministic_and_carries_the_schema():
    logs = [
        _env(pursuer="RULES-V0", seed0=WALK_SEED0).play_episode(
            _first_admissible, WALK_SEED0
        )
        for _ in range(2)
    ]
    assert logs[0].keys() == logs[1].keys()
    assert logs[0] == logs[1]
    log = logs[0]
    for key in (
        "outcome",
        "steps",
        "violations",
        "pursuer_spans",
        "evader_spans",
        "n_switches",
        "picks",
        "inapplicable_picks",
        "branch",
        "pursuer_name",
    ):
        assert key in log, key
    assert log["role"] == "evader" and log["shielded_role"] == "pursuer"
    assert log["pursuer_name"] == "RULES-V0"
    assert sum(log["picks"].values()) == log["n_decisions"] == len(log["evader_spans"])
    assert log["violations"] == 0  # H-SAFE
    assert log["steps"] == sum(s["length"] for s in log["evader_spans"])


def test_maskable_ppo_smoke_runs_a_thousand_steps(capsys):
    from sb3_contrib import MaskablePPO

    env = _env(pursuer="RULES-V0", seed0=82_000)
    model = MaskablePPO(
        "MlpPolicy", env, seed=82, gamma=0.99, n_steps=128, batch_size=64, verbose=0
    )
    start = time.perf_counter()
    model.learn(total_timesteps=1000)
    elapsed = time.perf_counter() - start
    assert model.num_timesteps >= 1000
    assert model.observation_space.shape == (EV_OBS_LEN,)
    assert model.action_space.n == 5
    assert env.inapplicable_picks == 0  # CG-T1 through a real rollout
    assert env.violations_total == 0  # H-SAFE through a real rollout
    assert set(env.picks) <= set(EVADER_ACTIONS)
    with capsys.disabled():
        print(
            f"\n[E.1 smoke] {model.num_timesteps} decisions in {elapsed:.1f}s = "
            f"{model.num_timesteps / elapsed:.1f} decisions/s; "
            f"{env.env_steps} env steps = {env.env_steps / elapsed:.1f} env steps/s"
        )
