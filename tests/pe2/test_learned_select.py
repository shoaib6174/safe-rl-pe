"""`pe2/learned_select.py` — `LearnedPursuerSelect` (E.1) has its own coverage
inside `tests/pe2/test_evader_selector_mdp.py` (CG-E1's two tests). This file
holds E.3's mirror, `LearnedEvaderSelect`, and CG-A1: the mirror of CG-E1 from
the OTHER seat — a pursuer driven by `PursuerRulesV0` (the game's default) and
an evader driven by the frozen `EVS-80` checkpoint through `LearnedEvaderSelect`
must reproduce the stage-1 record `EVS-80_RULES-V0` episode 0 bit for bit.

Spec: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` and
`/home/niro-1/s24_briefs/brief_e_3.md` deliverable 1.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.evader_selector_mdp import EvaderSelectorMDP
from pe2.learned_select import LearnedEvaderSelect
from pe2.selector_mdp import _game_module
from pe2.structured import OrbitSearch

REPO = pathlib.Path(__file__).resolve().parents[2]
GATE_SEED0 = 2_400_000  # the S24.2 gate stream (that prereg, section 2)
RECORD = REPO / "results/derisk/s24_2_cells/EVS-80_RULES-V0.json"
CKPT = REPO / "results/derisk/ckpts_S24-2-EVS-80/S24-2-EVS-80_3000000.zip"


def _built():
    """A fresh game, a `LearnedEvaderSelect` over it, and the evader seat
    installed — the exact assembly CG-A1 and E.3's `SEL3` trainer both use."""
    from sb3_contrib import MaskablePPO

    cfg = EnvConfig()
    assert cfg.evader_speed_ratio == 1.2
    game, rules = _game_module().build_game(cfg)
    sel = EvaderSelectorMDP.over_game(game, rules, cfg=cfg)
    model = MaskablePPO.load(CKPT, device="cpu")
    select = LearnedEvaderSelect(sel, model, name=CKPT.stem)
    game.evader.select = select
    return game, rules, sel, select


def _spans(log_or_record) -> list[list]:
    return [list(span) for span in log_or_record]


# ----------------------------------------------------------------------------
# CG-A1 — the mirror of CG-E1, from the pursuer's (SelectorMDP) side
# ----------------------------------------------------------------------------
class TestCgA1:
    def test_replays_the_stored_evs_80_rules_v0_episode(self):
        record = json.loads(RECORD.read_text())
        assert record["name"] == "EVS-80_RULES-V0"
        assert record["seed0"] == GATE_SEED0
        assert CKPT.exists(), f"CG-A1 needs the frozen checkpoint at {CKPT}"

        game, _rules, sel, _select = _built()
        # the pursuer seat is PursuerRulesV0, the game's own default — NOT
        # overridden, so `game.law.select is rules["pursuer"]` throughout.
        assert game.law.select is _rules["pursuer"]
        cfg = EnvConfig()
        GAME = _game_module()
        log = game.play_episode(OrbitSearch(cfg).predict, GATE_SEED0)
        tactics = {
            "evader": GAME._history(game.evader),
            "pursuer": GAME._history(game.law),
        }

        assert (log["outcome"] == "capture") is bool(record["capture_ind"][0])
        assert log["steps"] == record["steps"][0] == 1200
        assert log["violations"] == record["violations"][0] == 0
        assert _spans(tactics["pursuer"]) == _spans(record["tactics"][0]["pursuer"])
        assert _spans(tactics["evader"]) == _spans(record["tactics"][0]["evader"])
        assert sel.n_decisions == len(record["tactics"][0]["evader"]) == 30
        assert sel.inapplicable_picks == 0

    def test_the_evader_is_driven_through_the_evader_selectors_own_code(self):
        """The replay above would also pass with a hard-coded evader, so this
        checks what A1.2's mirror requires: a `LearnedEvaderSelect` over an
        `EvaderSelectorMDP` whose game IS the game being played, taking one
        decision per evader span."""
        game, rules, sel, select = _built()
        assert isinstance(sel, EvaderSelectorMDP)
        assert sel.game is game and sel.rules is rules
        assert sel.obs_len == 47 and sel.n_actions == 5
        assert game.evader.select is select
        cfg = EnvConfig()
        game.play_episode(OrbitSearch(cfg).predict, GATE_SEED0)
        assert sel.n_decisions == len(game.evader.history) == 30
        assert sum(sel.picks.values()) == 30
        assert set(sel.picks) <= set(sel.actions)
        assert sel.inapplicable_picks == 0


# ----------------------------------------------------------------------------
# the callable's own contract
# ----------------------------------------------------------------------------
class _FakeModel:
    """A stand-in `MaskablePPO`: fixed obs/action shapes, and `predict`
    returns whatever index the test wants, so the refusal path is exercised
    without a real policy."""

    class _Space:
        def __init__(self, shape=None, n=None):
            self.shape = shape
            self.n = n

    def __init__(self, obs_len=47, n_actions=5, index=0):
        self.observation_space = self._Space(shape=(obs_len,))
        self.action_space = self._Space(n=n_actions)
        self._index = index

    def predict(self, obs, action_masks=None, deterministic=True):
        return self._index, None


class TestConstructorValidation:
    def test_refuses_an_obs_len_mismatch(self):
        cfg = EnvConfig()
        game, rules = _game_module().build_game(cfg)
        sel = EvaderSelectorMDP.over_game(game, rules, cfg=cfg)
        with pytest.raises(ValueError, match="observation shape"):
            LearnedEvaderSelect(sel, _FakeModel(obs_len=55), name="bad")

    def test_refuses_an_action_count_mismatch(self):
        cfg = EnvConfig()
        game, rules = _game_module().build_game(cfg)
        sel = EvaderSelectorMDP.over_game(game, rules, cfg=cfg)
        with pytest.raises(ValueError, match="actions"):
            LearnedEvaderSelect(sel, _FakeModel(n_actions=6), name="bad")


class TestResetAndRefusal:
    def test_reset_clears_picks_and_n_decisions(self):
        game, _rules, sel, select = _built()
        cfg = EnvConfig()
        game.play_episode(OrbitSearch(cfg).predict, GATE_SEED0)
        assert sel.n_decisions and sum(sel.picks.values())
        select.reset()
        assert sel.n_decisions == 0
        assert sum(sel.picks.values()) == 0

    def test_a_masked_action_raises_and_counts_inapplicable_picks(self):
        cfg = EnvConfig()
        game, rules = _game_module().build_game(cfg)
        sel = EvaderSelectorMDP.over_game(game, rules, cfg=cfg)
        game._begin(GATE_SEED0)  # a live decision point exists
        mask = sel.action_masks()
        bad_index = int(np.flatnonzero(~mask)[0])
        select = LearnedEvaderSelect(sel, _FakeModel(index=bad_index), name="bad")
        game.evader.select = select
        assert sel.inapplicable_picks == 0
        with pytest.raises(AssertionError, match="refuses"):
            select(game.inner._last_joint_obs["evader"], game.inner.env, game.evader.history)
        assert sel.inapplicable_picks == 1
