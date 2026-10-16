"""`pe2/flat_mdp.py` + `scripts/derisk_s24_flat_train.py` — S24-FLAT Step F.1.

Spec: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md` sections 2, 3
(`FLAT-s`) and 5 (CG-F1, CG-F3).

Everything here is either a constant the pre-registration fixes or one of the
properties the round's construct gate rests on. The paper's central comparison
is "the selector learns where flat learning did not", so the whole content of
this file is that the two envs are the SAME GAME:

- CG-F1: the `EnvConfig`, the whole v0b assembly and every observation entry
  agree with `SelectorMDP`'s on an identical reset state, and the only
  observation difference anywhere is the 21 tactic entries, which are
  identically 0 here and go non-zero for the selector;
- the action is held exactly `hold` = 10 env steps, acquisition or not (the
  one place `SearchMDP`'s hold pattern is deliberately not copied: FLAT has no
  law to hand control to);
- the reward is the env's terminal +-1 summed over the span and nothing else;
- every env step carries the recorder's learner branch, so FLAT's immobile and
  pinned-run numbers are the selector's quantities in the selector's fields;
- `play_episode` is deterministic for a fixed `predict` and answers the
  selector's log schema, with the three tactic-selection fields answered as
  "there were none";
- the training script's arms, budget, streams, paths and hyper-parameters are
  the pre-registered ones, it prints no capture rate, CG-F1 failure aborts the
  launch, a crash is UNDETERMINED, and one 1 000-decision PPO smoke learns and
  writes the probe JSON schema.

No real 3 M run: the smoke is 1 000 decisions with a tiny rollout.
"""

import importlib.util
import inspect
import json
import pathlib
import time

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.flat_mdp import (
    CG_SEED,
    N_TACTIC_ENTRIES,
    TACTIC_INDICES,
    FlatMDP,
)
from pe2.selector_mdp import OBS_LEN, OBS_SLICES, SelectorMDP
from pe2.structured import HOLD

_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "derisk_s24_flat_train.py"
)
SEED = 2_290_000  # construct stream (review item 4: never a probe/gate stream)
STRAIGHT = np.array([1.0, 0.0], dtype=np.float32)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s24_flat_train", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _fixed_predict(action=STRAIGHT):
    """A `predict` in the `PPO.predict` contract (no `action_masks`)."""

    def predict(obs, deterministic=True):
        return np.asarray(action, dtype=np.float32), None

    return predict


def _roll(env, n_decisions, action=STRAIGHT):
    """`n_decisions` identical commands; stops at the end of the episode."""
    out = []
    for _ in range(n_decisions):
        obs, reward, term, trunc, info = env.step(action)
        out.append((obs, reward, info["env_steps"], info["span_steps"], term or trunc))
        if term or trunc:
            break
    return out


# ----------------------------------------------------------------------------
# CG-F1: the same game, one variable changed
# ----------------------------------------------------------------------------
def test_the_tactic_block_is_the_twenty_one_entries_of_the_two_tactic_slices():
    assert N_TACTIC_ENTRIES == 21
    assert list(TACTIC_INDICES) == [
        *range(OBS_SLICES["current_tactic"].start, OBS_SLICES["current_tactic"].stop),
        *range(OBS_SLICES["last_tactics"].start, OBS_SLICES["last_tactics"].stop),
    ]
    # The layout objects are the selector's own, not a second copy.
    assert FlatMDP(1.2).observation_space.shape == (OBS_LEN,)


def test_observation_equals_the_selectors_on_an_identical_reset_state():
    flat, sel = FlatMDP(1.2), SelectorMDP(1.2)
    a, _i = flat.reset(seed=SEED)
    b, _i = sel.reset(seed=SEED)
    assert a.shape == b.shape == (OBS_LEN,)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b)  # every entry, not just the non-tactic ones


def test_the_tactic_entries_stay_zero_while_the_rest_of_the_vector_lives():
    flat = FlatMDP(1.2)
    obs, _i = flat.reset(seed=SEED)
    seen = [obs]
    seen += [o for o, _r, _s, _l, _d in _roll(flat, 20)]
    index = list(TACTIC_INDICES)
    others = [i for i in range(OBS_LEN) if i not in set(index)]
    for o in seen:
        assert not np.any(np.asarray(o)[index]), "a tactic entry moved"
    stack = np.asarray(seen)
    assert np.any(stack[:, others].std(axis=0) > 0), "the rest of the vector is dead"


def test_cg_f1_holds_and_the_only_difference_is_the_action_and_the_block():
    cg = FlatMDP(1.2, seed0=80_000).config_diff_vs_selector()
    assert cg["config_diff"] == []  # CG-F1's "list of differing EnvConfig fields"
    assert cg["assembly_diff"] == []
    assert cg["obs_diff_indices_on_identical_reset"] == []
    assert cg["obs_len"] == [OBS_LEN, OBS_LEN]
    assert cg["flat_tactic_entries_nonzero"] == 0
    assert cg["selector_tactic_entries_nonzero"] > 0  # live for the selector
    assert cg["action_space"]["flat"].startswith("Box")
    assert cg["action_space"]["selector"] == "Discrete(5)"
    assert cg["cg_seed"] == CG_SEED
    assert cg["ok"] is True


def test_cg_f1_does_not_consume_the_built_envs_stream_or_counters():
    """Reading 6: the trainer asserts CG-F1 on the TRAINING env before the
    first step, so it must not reset it, spend an episode of its stream or
    move the counters the budget and H-SAFE are read off."""
    env = FlatMDP(1.2, seed0=80_000)
    before = (env.env_steps, env.violations_total, env.game.inner._episode_count)
    env.config_diff_vs_selector()
    assert (env.env_steps, env.violations_total, env.game.inner._episode_count) == before


def test_cfg_and_assembly_are_the_frozen_ones():
    """Prereg section 2: `EnvConfig()` at ratio 1.2, `RecoveryShieldV2`, the
    v0b `hold`, the scripted tactical evader."""
    env = FlatMDP(1.2)
    assert env.cfg == EnvConfig(evader_speed_ratio=1.2)
    assert env.cfg.n_obstacles == 2
    assert type(env.game.shield).__name__ == "RecoveryShieldV2"
    assert env.hold == env.game.hold == HOLD == 10
    assert type(env.game.evader).__name__ == "TacticOpponent"
    assert type(env.game.rules["evader"]).__name__ == "EvaderRulesV0"


def test_there_is_no_action_mask():
    """CG-F3: CG-T1 and A2.1 do not apply to FLAT."""
    env = FlatMDP(1.2)
    env.reset(seed=SEED)
    with pytest.raises(AssertionError, match="no action mask"):
        env.action_masks()
    assert env.action_space.shape == (2,)
    assert float(env.action_space.low[0]) == -1.0
    assert float(env.action_space.high[0]) == 1.0


# ----------------------------------------------------------------------------
# the hold
# ----------------------------------------------------------------------------
def test_the_action_is_held_exactly_ten_env_steps():
    env = FlatMDP(1.2)
    env.reset(seed=SEED)
    rolled = _roll(env, 30, np.array([0.4, -0.7], dtype=np.float32))
    for i, (_o, _r, env_steps, span, done) in enumerate(rolled):
        if done:
            break
        assert span == 10, i
        assert env_steps == 10 * (i + 1)
    records = env.game.step_records[-10:]
    held = [float(np.float32(0.4)), float(np.float32(-0.7))]  # the float32 command
    assert [rec["nominal"] for rec in records] == [held] * 10
    assert all(rec["branch"] == "search" for rec in records)


def test_the_hold_does_not_break_on_an_acquisition():
    """Reading 2 / the prereg: "at EVERY decision regardless of visibility".
    `SearchMDP.step` breaks the hold when the evader comes into view because
    the intercept law takes over there; FLAT has no law, so the span must run
    its full 10 steps across an acquisition."""
    env = FlatMDP(1.2)
    env.reset(seed=2_290_000)
    found = None
    for _ in range(200):
        start = len(env.game.step_records)
        _o, _r, term, trunc, info = env.step(np.array([1.0, 0.35], dtype=np.float32))
        visible = [rec["visible"] for rec in env.game.step_records[start:]]
        if not visible[0] and any(visible):
            found = (info["span_steps"], visible)
            break
        if term or trunc:
            break
    assert found is not None, "no span contained an acquisition"
    assert found[0] == 10, found


def test_reward_is_the_summed_terminal_and_nothing_else():
    env = FlatMDP(1.2)
    env.reset(seed=SEED)
    rolled = _roll(env, 200, np.array([1.0, 0.9], dtype=np.float32))
    assert rolled[-1][4], "the episode must have ended inside the roll"
    rewards = [r for _o, r, _s, _l, _d in rolled]
    assert all(r == 0.0 for r in rewards[:-1])  # no shaping in any span
    assert abs(rewards[-1]) == pytest.approx(1.0)  # the env's terminal +-1
    assert sum(rewards) == pytest.approx(rewards[-1])


def test_no_shaping_ingredient_is_switched_on():
    env = FlatMDP(1.2)
    game = env.game
    assert game.stall_penalty == 0.0
    assert game.intervention_penalty == 0.0
    assert game.zero_reward is False
    assert game._driver is game.inner  # reward_wrapper=None (v0b's assembly)
    # `SearchMDP._shaped` adds `r_acq` per acquisition onset and is the only
    # place shaping could enter; this env never routes a reward through it.
    for method in (FlatMDP.step, FlatMDP.play_episode):
        assert "_shaped" not in inspect.getsource(method)


# ----------------------------------------------------------------------------
# the recorder
# ----------------------------------------------------------------------------
def test_every_env_step_carries_the_learner_branch_and_the_live_span_scope():
    env = FlatMDP(1.2)
    env.reset(seed=SEED + 4)  # a seed whose episode the filter really acts on
    _roll(env, 60)
    summary = env.game.branch_summary()
    assert summary["steps_by_branch"]["law"] == 0
    assert summary["steps_by_branch"]["search"] == summary["n_steps"] > 0
    assert all(rec["branch"] == "search" for rec in env.game.step_records)
    # A2.4: without `_in_span` / `_span_total` the recorder's split is dead.
    assert all(rec["in_span"] for rec in env.game.step_records)
    assert summary["intervention_total"] > 0.0, "the shield never intervened"
    assert summary["span_intervention_total"] == pytest.approx(
        summary["intervention_total"], rel=1e-6
    )
    assert summary["pre_decision_intervention"] == pytest.approx(0.0, abs=1e-9)
    # The fields H-FLAT-3 is read off exist and are comparable to the
    # selector's learner steps.
    for key in ("immobile_by_branch", "longest_pinned_run", "pinned_runs"):
        assert key in summary


# ----------------------------------------------------------------------------
# play_episode
# ----------------------------------------------------------------------------
def test_play_episode_is_deterministic_and_carries_the_log_schema():
    first = FlatMDP(1.2).play_episode(_fixed_predict(), SEED)
    second = FlatMDP(1.2).play_episode(_fixed_predict(), SEED)
    for key in ("outcome", "steps", "n_steps", "violations", "n_decisions"):
        assert first[key] == second[key], key
    for key in (  # `SelectorMDP.play_episode`'s schema
        "seed",
        "role",
        "outcome",
        "steps",
        "final_distance",
        "los_breaks",
        "shielded_role",
        "violations",
        "shield",
        "branch",
        "tactics",
        "usage",
        "n_decisions",
        "n_switches",
        "inapplicable_picks",
        "picks",
        "pursuer_spans",
        "n_zero_length_spans",
        "recovery",
        "n_steps",
        "mean_span_steps",
    ):
        assert key in first, key
    assert first["pursuer_spans"] == []  # no tactic selection happened
    assert first["n_switches"] is None  # not 0: nothing could have switched
    assert first["picks"] == {}
    assert first["inapplicable_picks"] == 0
    assert first["n_zero_length_spans"] == 0
    assert first["n_steps"] == first["steps"]
    assert set(first["tactics"]) == {"evader", "pursuer"}
    assert first["tactics"]["pursuer"] == []  # the law is never consulted
    assert first["tactics"]["evader"], "the scripted evader ran no tactic"
    assert "n_steps" in first["branch"]  # the recorder's branch summary
    assert first["mean_span_steps"] == pytest.approx(
        first["steps"] / first["n_decisions"]
    )


def test_play_episode_replays_the_same_seed_twice_in_one_env():
    env = FlatMDP(1.2)
    predict = _fixed_predict(np.array([0.8, 0.2], dtype=np.float32))
    a = env.play_episode(predict, SEED + 3)
    b = env.play_episode(predict, SEED + 3)
    assert (a["outcome"], a["steps"], a["violations"]) == (
        b["outcome"],
        b["steps"],
        b["violations"],
    )


# ----------------------------------------------------------------------------
# the training script
# ----------------------------------------------------------------------------
def test_the_arms_and_the_budget_are_the_preregistered_ones():
    assert set(R.ARMS) == {"FLAT-80", "FLAT-81", "FLAT-82", "FLAT40-80", "FLAT40-81", "FLAT40-82"}
    assert R.SEEDS == (80, 81, 82)
    assert (R.TOTAL, R.EVERY, R.PROBE_SEED0, R.PROBE_N) == (
        3_000_000,
        250_000,
        2_550_000,
        200,
    )
    assert R.TOTAL % R.EVERY == 0
    assert (R.RATIO, R.GAMMA, R.OPPONENT) == (1.2, 0.99, "TACTIC-V0")


def test_the_streams_are_disjoint():
    """Training `seed * 1000`, probe 2 550 000+, gate 2 400 000+ (prereg 4)."""
    train = {R.ARMS[a]["seed"] * 1000 + i for a in R.ARMS for i in range(1000)}
    probe = set(range(R.PROBE_SEED0, R.PROBE_SEED0 + R.PROBE_N))
    gate = set(range(2_400_000, 2_400_500))
    assert not (train & probe) and not (train & gate) and not (probe & gate)


def test_the_paths_carry_the_round_and_the_arm_exactly_once():
    """Reading 4."""
    assert R._label("FLAT-80") == "S24-FLAT-80"
    assert R._ckpt_dir("FLAT-80").name == "ckpts_S24-FLAT-80"
    assert R._probe_file("FLAT-80").name == "s24_flat_probe_FLAT-80.json"
    assert R._arm_file("FLAT-80").name == "s24_flat_arm_FLAT-80.json"


def test_the_learner_is_s17_base_with_the_continuous_head():
    env = R.make_env("FLAT-80")
    assert type(env).__name__ == "FlatMDP"
    assert env.game.inner.seed0 == 80_000
    model = R.make_learner("FLAT-80", env)
    assert type(model).__name__ == "PPO"
    assert model.gamma == 0.99
    assert model.seed == 80
    # S17's `make_learner` is "SB3 defaults apart from the arm's gamma": every
    # other hyper-parameter must still be the library default.
    from stable_baselines3 import PPO

    defaults = inspect.signature(PPO.__init__).parameters
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gae_lambda",
        "clip_range",
        "ent_coef",
    ):
        want = defaults[name].default
        got = getattr(model, name)
        assert got == want or callable(got), name


def test_train_evidence_reads_the_training_env():
    """H-SAFE's counter comes off the TRAINING env through the VecEnv, so a
    crashed arm's last probe row still carries it."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: R.make_env("FLAT-80")])
    vec.reset()
    assert R.train_evidence(vec) == {"train_violations": 0}
    vec.envs[0].game.violations_total = 3  # the env's counter is a property
    assert R.train_evidence(vec)["train_violations"] == 3


def test_cg_f1_failure_aborts_the_launch(monkeypatch, tmp_path):
    """Prereg section 5: CG-F1 is asserted before the first step, and a failure
    is a refusal about the CONFIGURATION — it must abort, not become three
    UNDETERMINED records."""
    env = R.make_env("FLAT-80")
    assert R.assert_cg_f1(env)["ok"] is True
    monkeypatch.setattr(
        env,
        "config_diff_vs_selector",
        lambda **_kw: {
            "ok": False,
            "config_diff": ["evader_speed_ratio"],
            "assembly_diff": [],
            "obs_diff_indices_on_identical_reset": [],
            "flat_tactic_entries_nonzero": 0,
            "selector_tactic_entries_nonzero": 5,
        },
    )
    with pytest.raises(SystemExit, match="CG-F1 FAILED"):
        R.assert_cg_f1(env)


def test_a_systemexit_aborts_and_a_crash_writes_undetermined(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)

    def guard(*_a, **_k):
        raise SystemExit("CG-F1 FAILED")

    monkeypatch.setattr(R, "train", guard)
    with pytest.raises(SystemExit):
        R.run(["FLAT-80", "FLAT-81"], "cpu")
    assert not list(tmp_path.glob("s24_flat_arm_*.json"))

    def crash(*_a, **_k):
        raise RuntimeError("this arm crashed")

    monkeypatch.setattr(R, "train", crash)
    R._write_json(
        tmp_path / "s24_flat_probe_FLAT-80.json",
        {"arm": "FLAT-80", "checkpoints": [{"mark": 250_000}]},
    )
    R.run(["FLAT-80"], "cpu")  # a real crash is UNDETERMINED, not an abort
    record = json.loads((tmp_path / "s24_flat_arm_FLAT-80.json").read_text())
    assert record["verdict"] == "UNDETERMINED"
    assert "this arm crashed" in record["reason"]
    assert [r["mark"] for r in record["checkpoints"]] == [250_000]  # the history


def test_resume_is_arm_granular(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    assert R._resume_from("FLAT-80") == (None, 0, [])
    d = tmp_path / "ckpts_S24-FLAT-80"
    d.mkdir(parents=True)
    for mark in (250_000, 500_000):
        (d / f"S24-FLAT-80_{mark}.zip").write_bytes(b"")
    R._write_json(
        tmp_path / "s24_flat_probe_FLAT-80.json",
        {"arm": "FLAT-80", "checkpoints": [{"mark": 250_000}, {"mark": 500_000}]},
    )
    ckpt, mark, rows = R._resume_from("FLAT-80")
    assert mark == 500_000
    assert ckpt.name == "S24-FLAT-80_500000.zip"
    assert [r["mark"] for r in rows] == [250_000, 500_000]
    (d / f"S24-FLAT-80_{R.TOTAL}.zip").write_bytes(b"")
    assert R._resume_from("FLAT-80")[0] is None  # finished: nothing to resume


def test_provenance_pins_the_prereg_and_the_frozen_baseline():
    p = R.provenance()
    assert p["prereg"] == "docs/pathway/tactics_prereg_s24_flat_2026-10-14.md"
    assert p["prereg_sha256"] and len(p["prereg_sha256"]) == 64
    assert p["baseline_record"].endswith("s24_tactic_game_v0b.json")
    assert p["baseline_sha256"] and len(p["baseline_sha256"]) == 64
    assert p["git_commit"] and len(p["git_commit"]) == 40
    assert p["pe2_file"].endswith("pe2/__init__.py")
    assert p["obs_len"] == OBS_LEN
    assert p["ratio"] == 1.2
    assert p["omp_num_threads"] is not None
    assert p["label_branch"].startswith("labels: none")


def test_the_script_prints_no_capture_rate():
    """The stop rule only holds if training cannot show a number."""
    source = _PATH.read_text()
    for line in source.splitlines():
        if "print(" not in line and 'f"[' not in line:
            continue
        assert "capture_rate" not in line, line
        assert "win_rate" not in line, line


def test_ppo_smoke_learns_and_writes_the_probe_schema(tmp_path, monkeypatch, capsys):
    from stable_baselines3 import PPO

    monkeypatch.setattr(R, "OUT", tmp_path)
    env = R.make_env("FLAT-80")
    model = PPO(
        "MlpPolicy", env, seed=80, gamma=R.GAMMA, n_steps=128, batch_size=64, verbose=0
    )
    t0 = time.time()
    model.learn(total_timesteps=1000)
    elapsed = time.time() - t0
    assert model.num_timesteps >= 1000
    assert env.env_steps >= 10 * 1000  # one decision is 10 env steps
    monkeypatch.setattr(R, "PROBE_N", 2)
    probes = R._Probe("FLAT-80", [])
    probes.at(model, env.env_steps, R.EVERY, R.train_evidence(model.get_env()))
    row = probes.rows[0]
    for key in (
        "train_violations",
        "opponent",
        "n",
        "k",
        "capture_rate",
        "mean_steps",
        "mean_span_steps",
        "n_decisions_mean",
        "immobile_step_rate",
        "pinned_run_episode_fraction",
        "pinned_step_fraction",
        "violations",
        "episodes_with_violations",
    ):
        assert key in row, key
    assert row["n"] == 2
    assert row["train_violations"] == 0  # H-SAFE through a real rollout
    written = json.loads(R._probe_file("FLAT-80").read_text())
    assert written["checkpoints"][0]["mark"] == R.EVERY
    assert written["checkpoints"][0]["ckpt"].endswith(f"S24-FLAT-80_{R.EVERY}.zip")
    with capsys.disabled():
        print(
            f"\n[smoke] 1000 PPO decisions = {env.env_steps} env steps in "
            f"{elapsed:.1f}s: {env.env_steps / elapsed:.0f} env steps/s "
            f"({1000 / elapsed:.0f} decisions/s)"
        )
