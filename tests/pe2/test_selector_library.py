"""S24-OS Step O.1 — the pursuer library is a FLAG, and `v0` is still frozen.

Spec: `docs/pathway/tactics_prereg_s24_os_2026-10-17.md` sections 1 and 2.

Two things are load-bearing here.

- **CG-O1 (identity).** `SelectorMDP(library="v0")` must replay an S24.1 record
  bit for bit: the first gate episode (seed 2 400 000) of the cell
  `SEL-81_TACTIC-V0`, driven by that arm's frozen final checkpoint, with the
  same outcome, step count, violation count and pursuer span log as the stored
  cell record. This is the construct gate that says the flag did not move the
  measured artefacts; a difference is a DEFECT in the flag, never a tolerance
  to widen. `tests/pe2/test_selector_mdp.py` passing unchanged is the other
  half of CG-O1 and lives there.
- **`v1` is the same env with one more action.** Six actions, `OBS_LEN` 59
  (the two one-hot blocks and nothing else grow), the sixth masked out
  whenever the evader is visible, and a real `MaskablePPO` rollout over it.

Two readings, because the brief and the implemented interface differ:

1. **The checkpoint path is `derisk_s24_1_gate.py::ckpt_path`'s**, i.e.
   `ckpts_S24-1-SEL-81/S24-1-SEL-81_3000000.zip`. The brief writes
   `ckpts_S24-SEL-81/S24-SEL-81_3000000.zip`, which is not a file on disk; that
   script already names the same discrepancy against its own brief and resolves
   it the same way ("the implemented interface wins, because that is the file on
   disk").
2. **`seed0` is inert for this replay** and is passed only to name the stream:
   `SelectorMDP.play_episode` seeds every episode explicitly, and the wrapper
   falls back to `seed0` only when the seed is None. The stored record was
   produced through `build_cell_env`, which leaves `seed0` at its default; the
   episode is therefore the same either way, and the test asserts that.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.recogniser import PURSUER_TACTICS
from pe2.selector_mdp import (
    ACTIONS,
    LIBRARIES,
    MEMORY_SLICES,
    N_ACTIONS,
    OBS_LAYOUT,
    OBS_LEN,
    OBS_SLICES,
    SelectorMDP,
    done_on_entry,
    layout,
    obs_len_for,
)
from pe2.tactics import OrbitSearch as OrbitSearchTactic

REPO = pathlib.Path(__file__).resolve().parents[2]
GATE_SEED0 = 2_400_000  # the S24.1 gate stream (that prereg, section 4)
RECORD = REPO / "results/derisk/s24_1_cells/SEL-81_TACTIC-V0.json"
CKPT = REPO / "results/derisk/ckpts_S24-1-SEL-81/S24-1-SEL-81_3000000.zip"  # reading 1
V0_LEN, V1_LEN = 55, 59


def _cell_env(library: str = "v0", **kwargs) -> SelectorMDP:
    """`derisk_s24_1_gate.py::build_cell_env` for the cell `SEL-81_TACTIC-V0`:
    `cell_config()` (`EnvConfig()`, whose ratio default IS the round's 1.2),
    `memory=True` for a `SEL` row, and the `TACTIC-V0` opponent — which that
    script's `_set_opponent` leaves at the assembly's own scripted evader, so
    nothing but the library flag is passed here."""
    cfg = EnvConfig()
    assert cfg.evader_speed_ratio == 1.2
    return SelectorMDP(
        cfg.evader_speed_ratio, memory=True, cfg=cfg, library=library, **kwargs
    )


def _spans(log_or_record) -> list[list]:
    """A span log as plain lists, so a JSON round trip compares equal."""
    return [list(span) for span in log_or_record]


# ----------------------------------------------------------------------------
# CG-O1: library v0 is the frozen one
# ----------------------------------------------------------------------------
def test_cg_o1_v0_replays_the_frozen_s24_1_cell_episode():
    from sb3_contrib import MaskablePPO

    record = json.loads(RECORD.read_text())
    # the record is the cell this gate claims to replay, not a namesake
    assert (record["row"], record["opponent"]) == ("SEL-81", "TACTIC-V0")
    assert record["kind"] == "learned" and record["memory"] is True
    assert record["seed0"] == GATE_SEED0 and record["n_episodes"] == 500
    assert CKPT.exists(), f"CG-O1 needs the frozen final checkpoint at {CKPT}"

    policy = MaskablePPO.load(CKPT, device="cpu")
    env = _cell_env("v0", seed0=GATE_SEED0)  # reading 2
    log = env.play_episode(policy.predict, GATE_SEED0)

    assert (log["outcome"] == "capture") is bool(record["capture_ind"][0])
    assert log["steps"] == record["steps"][0]
    assert log["violations"] == record["violations"][0]
    assert _spans(log["tactics"]["pursuer"]) == _spans(record["tactics"][0]["pursuer"])
    # the pick histogram over the same episode, for free: the spans agreeing
    # while the picks did not would mean the action map had moved under them
    assert log["n_decisions"] == len(record["tactics"][0]["pursuer"])
    assert log["inapplicable_picks"] == 0  # CG-T1, inherited


def test_cg_o1_does_not_depend_on_seed0():
    """Reading 2, asserted rather than argued: the stored record was produced
    with `seed0` at its default and this gate names the stream explicitly."""
    from sb3_contrib import MaskablePPO

    policy = MaskablePPO.load(CKPT, device="cpu")
    named = _cell_env("v0", seed0=GATE_SEED0).play_episode(policy.predict, GATE_SEED0)
    default = _cell_env("v0").play_episode(policy.predict, GATE_SEED0)
    assert (named["outcome"], named["steps"], named["violations"]) == (
        default["outcome"],
        default["steps"],
        default["violations"],
    )
    assert _spans(named["tactics"]["pursuer"]) == _spans(default["tactics"]["pursuer"])


def test_the_module_constants_are_still_v0s():
    """Every existing import — trainers, gates, `pe2/flat_mdp.py`, the S24.1
    tests — reads these names and must keep reading S24.1's values."""
    assert LIBRARIES["v0"] is PURSUER_TACTICS
    assert ACTIONS is PURSUER_TACTICS  # imported, never restated
    assert (N_ACTIONS, OBS_LEN) == (5, V0_LEN)
    assert (ACTIONS, N_ACTIONS, OBS_LAYOUT, OBS_SLICES, MEMORY_SLICES, OBS_LEN) == (
        layout("v0")
    )
    assert obs_len_for() == obs_len_for("v0") == V0_LEN
    assert obs_len_for("v1") == V1_LEN
    # the recogniser's label space is NOT extended (D147/D148: it is the
    # recogniser's own); the selector's action list stops aliasing it
    assert PURSUER_TACTICS == ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
    assert LIBRARIES["v1"] == (*PURSUER_TACTICS, "ORBIT-SEARCH")
    with pytest.raises(ValueError):
        layout("v2")


def test_flat_mdp_still_builds_on_the_v0_library():
    from pe2.flat_mdp import FlatMDP

    env = FlatMDP(1.2)
    assert env.library == "v0"
    assert env.obs_len == OBS_LEN
    assert env.observation_space.shape == (V0_LEN,)


# ----------------------------------------------------------------------------
# library v1: one more action, four more observation entries
# ----------------------------------------------------------------------------
def test_v1_is_six_actions_and_fifty_nine_observation_entries():
    env = SelectorMDP(1.2, library="v1")
    assert env.library == "v1"
    assert env.actions == LIBRARIES["v1"]
    assert (env.n_actions, env.obs_len) == (6, V1_LEN)
    assert env.action_space.n == 6
    assert env.observation_space.shape == (V1_LEN,)
    obs, _info = env.reset(seed=GATE_SEED0)
    assert obs.shape == (V1_LEN,)
    # the layout grew in the two one-hot blocks and nowhere else
    v0 = {name: width for name, width, _m in layout("v0")[2]}
    v1 = {name: width for name, width, _m in layout("v1")[2]}
    assert set(v0) == set(v1)
    assert {k: v1[k] - v0[k] for k in v1 if v1[k] != v0[k]} == {
        "current_tactic": 1,
        "last_tactics": 3,
    }
    # the blocks still tile the vector exactly, and the memory block is the
    # same set of blocks as v0's
    covered = np.zeros(V1_LEN, dtype=int)
    for block in env.obs_slices.values():
        covered[block] += 1
    assert (covered == 1).all()
    assert [name for name, _w, mem in layout("v1")[2] if mem] == [
        name for name, _w, mem in OBS_LAYOUT if mem
    ]
    # and a v0 env built afterwards is untouched by the flag
    assert SelectorMDP(1.2).obs_len == V0_LEN


def test_v1_builds_its_orbit_search_instances_and_the_game_resets_them():
    env = SelectorMDP(1.2, library="v1")
    searches = env.rules_extra["orbit_search"]
    assert len(searches) == env.cfg.n_obstacles
    assert [t.name for t in searches] == [
        f"ORBIT-SEARCH({k})" for k in range(env.cfg.n_obstacles)
    ]
    assert all(isinstance(t.inner, OrbitSearchTactic) for t in searches)
    # the belief is the pursuer memory's own object, by reference: the tactic
    # must not carry a second, never-updated belief
    assert all(t.belief is env.pursuer.memory.belief for t in searches)
    # registered where `TacticLaw._reset_common` will reach them, and NOT in
    # the v0 rule table's own list (whose five instances stay the baseline's)
    assert all(any(t is x for x in env.game.law.tactics) for t in searches)
    assert not any(
        any(t is x for x in env.pursuer.tactics) for t in searches
    )
    searches[0].steps = 7
    env.reset(seed=GATE_SEED0)
    assert searches[0].steps == 0
    assert searches[0].env is env.game.inner.env  # the map arrived
    assert env.rules_extra == {"orbit_search": searches}
    assert SelectorMDP(1.2).rules_extra == {}  # v0 builds none


def test_v1_masks_orbit_search_out_whenever_the_evader_is_visible():
    """The brief's gate: the sixth mask entry is False on every visible step,
    and on hidden steps it is exactly the tactic's own `applicable` (masked
    "like SWEEP"). A2.1's `done_on_entry` is what gates the visible case."""
    env = SelectorMDP(1.2, library="v1")
    index = env.actions.index("ORBIT-SEARCH")
    assert index == 5
    rng = np.random.default_rng(0)
    visible_seen = hidden_seen = offered = 0
    for episode in range(4):
        env.reset(seed=GATE_SEED0 + episode)
        for _ in range(40):
            raw, game_env = env._obs_raw, env.game.inner.env
            visible = bool(raw[6] > 0.5)
            mask = env.action_masks()
            assert mask.shape == (6,)
            assert mask.any()  # a masked-out decision would deadlock the learner
            tactic = env._candidates(raw)[index]
            assert tactic is not None  # two obstacles: the family has instances
            if visible:
                assert not mask[index]
                assert done_on_entry(tactic, raw, game_env, env.cfg)
                visible_seen += 1
            else:
                assert not done_on_entry(tactic, raw, game_env, env.cfg)
                assert bool(mask[index]) == bool(tactic.applicable(raw, game_env))
                hidden_seen += 1
                offered += bool(mask[index])
            _obs, _r, term, trunc, _i = env.step(int(rng.choice(np.flatnonzero(mask))))
            if term or trunc:
                break
    assert visible_seen and hidden_seen
    # and the sixth action is not DEAD: a never-offered one would make v1 a
    # bigger observation around the same five tactics
    assert offered
    assert env.inapplicable_picks == 0  # CG-T1 over the whole walk


def test_v1_refuses_an_inapplicable_orbit_search_pick():
    """CG-T1 for the new action: the mask makes the pick impossible, and the
    env raises rather than quietly running a DONE-on-entry tactic."""
    env = SelectorMDP(1.2, library="v1")
    index = env.actions.index("ORBIT-SEARCH")
    rng = np.random.default_rng(1)
    for _ in range(60):
        env.reset(seed=GATE_SEED0)
        mask = env.action_masks()
        if not mask[index]:
            break
        env.step(int(rng.choice(np.flatnonzero(mask))))
    else:  # pragma: no cover — the gate stream always offers such a state
        pytest.fail("no decision reached where ORBIT-SEARCH is masked out")
    before = env.inapplicable_picks
    with pytest.raises(AssertionError, match="ORBIT-SEARCH"):
        env.step(index)
    assert env.inapplicable_picks == before + 1  # counted, then refused


def test_v1_maskable_ppo_smoke_runs_a_thousand_steps():
    from sb3_contrib import MaskablePPO

    env = SelectorMDP(1.2, library="v1", seed0=81_000)
    model = MaskablePPO(
        "MlpPolicy", env, seed=81, gamma=0.99, n_steps=128, batch_size=64, verbose=0
    )
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000
    assert model.observation_space.shape == (V1_LEN,)
    assert model.action_space.n == 6
    assert env.inapplicable_picks == 0  # CG-T1 through a real rollout
    assert env.violations_total == 0  # H-SAFE through a real rollout
    assert set(env.picks) <= set(env.actions)
