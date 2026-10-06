"""pe2/recorder.py + scripts/replay_s18_branch_split.py — the shared instrument.

Plan: `docs/pathway/agent_intelligence_plan_2026-10-05.md` section 1.

Four properties, and nothing built on the recorder may be read unless all four
hold:

- NO INTERVENTION GOES UNATTRIBUTED. The law's and the learner's shares sum to
  the episode's total, and the share taken inside learner decision spans sums
  to the number S18 recorded — so a branch split can be compared to an S18
  record without a conversion.
- A PIN IS A RUN WITHIN ONE COMPONENT. A branch change ends a by-branch run;
  the branch-agnostic run spans it. The law pinned for 70 steps, the learner for
  15, then the law for 120 is not one 205-step LAW pin, and the record must say
  all four numbers.
- WITHOUT A SHIELD THE SHIELD FIELDS ARE ZERO, and the branches are still tagged
  (the attribution is a property of the pursuer, not of the filter).
- THE RECORDER DOES NOT CHANGE THE EPISODE. Same seed, same log.

The replay script's own test is on its COMPARISON logic. Loading real
checkpoints here would make the suite depend on a 5 M-step training artefact
and on the frozen worktree; the reproduction itself is run by hand (and by the
paper's Phase 0), not by pytest.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.recorder import MIN_PINNED_RUN, BranchTaggedSearchMDP, branch_summary
from pe2.scripted import GreedyEvader
from pe2.shield import CBFShield
from pe2.structured import HOLD, SearchMDP

SEED = 1_900_000  # S18's gate stream; the recorder is replayed over it
FULL = np.array([1.0, 0.0], dtype=np.float32)


class _Full:
    """Always asks for full speed — the case that separates the REQUEST from
    the EXECUTION, and the one that pins against a wall."""

    def predict(self, obs, deterministic=True):
        return FULL, None


def _mdp(shield: bool = True, **kw):
    cfg = EnvConfig()
    return BranchTaggedSearchMDP(
        cfg,
        GreedyEvader,
        hold=HOLD,
        shield=CBFShield(cfg) if shield else None,
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=None,
        **kw,
    )


# ----------------------------------------------------------------------------
# no intervention goes unattributed
# ----------------------------------------------------------------------------
class TestBranchSumIdentity:
    def test_the_branches_sum_to_the_episodes_total(self):
        mdp = _mdp()
        log = mdp.play_episode(_Full().predict, SEED)
        b = log["branch"]
        assert b["intervention_total"] > 0.0  # a wall-seeking policy is corrected
        assert (
            b["intervention_by_branch"]["law"] + b["intervention_by_branch"]["search"]
            == b["intervention_total"]
        )

    def test_the_in_span_branches_sum_to_s18s_recorded_quantity(self):
        """`span_intervention_total` is accumulated exactly as S18 accumulated
        `info["intervention_magnitude"]`, and the per-step records inside
        decision spans must reconstruct it — otherwise the branch split cannot
        be compared to an S18 record."""
        mdp = _mdp()
        log = mdp.play_episode(_Full().predict, SEED)
        in_span = math.fsum(
            s["intervention_magnitude"] for s in mdp.step_records if s["in_span"]
        )
        assert in_span == pytest.approx(
            log["branch"]["span_intervention_total"], rel=1e-12, abs=1e-9
        )

    def test_the_law_steps_before_the_first_decision_are_outside_s18s_scope(self):
        """`SearchMDP.step` zeroes `span_intervention` on entry, so the law
        steps `play_episode` plays BEFORE the first learner decision are in no
        `info["intervention_magnitude"]`. The recorder keeps them, and names the
        difference rather than quietly absorbing it."""
        mdp = _mdp()
        b = mdp.play_episode(_Full().predict, SEED + 1)["branch"]
        assert b["pre_decision_intervention"] == (
            b["intervention_total"] - b["span_intervention_total"]
        )
        assert b["pre_decision_intervention"] >= 0.0

    def test_the_magnitude_is_the_base_classs_own_increment(self):
        """Not a second copy of the norm: the per-step value is the increment of
        `SearchMDP.span_intervention`, so it cannot drift from the quantity the
        penalty is built from."""
        mdp = _mdp()
        mdp._begin(SEED)
        before = mdp.span_intervention
        mdp._step_inner(FULL, "law")
        assert mdp.step_records[-1]["intervention_magnitude"] == (
            mdp.span_intervention - before
        )

    def test_steps_by_branch_covers_every_env_step(self):
        mdp = _mdp()
        b = mdp.play_episode(_Full().predict, SEED)["branch"]
        assert (
            b["steps_by_branch"]["law"] + b["steps_by_branch"]["search"]
            == (b["n_steps"])
        )
        assert b["n_steps"] > 0


# ----------------------------------------------------------------------------
# a pin is a run within one component
# ----------------------------------------------------------------------------
class TestPinnedRuns:
    """The failure S18 could not attribute: the LAW drives into a wall it can
    see and sits pinned. Driven here through `_step_inner` directly — the
    pursuer asks for full speed with zero turn rate, the filter clamps it at the
    wall, and the request never changes (D125's absorbing loop) — with a
    `search` span inserted between two `law` spans so the run-breaking rule is
    exercised on real shield output rather than a hand-written fixture."""

    PHASES = (("law", 120), ("search", 15), ("law", 120))

    def _drive(self):
        mdp = _mdp()
        mdp._begin(SEED)
        for branch, n in self.PHASES:
            for _ in range(n):
                mdp._step_inner(FULL, branch)
        return mdp, mdp.branch_summary()

    def test_the_law_is_recorded_as_pinned(self):
        _m, b = self._drive()
        assert b["longest_pinned_run_by_branch"]["law"] > 0
        assert b["immobile_by_branch"]["law"] > 0

    def test_a_branch_change_breaks_the_by_branch_run(self):
        """The three spans are three runs by branch, and the law's longest is
        strictly shorter than the branch-agnostic run that spans the change."""
        _m, b = self._drive()
        runs = b["pinned_runs"]
        assert [r["branch"] for r in runs] == ["law", "search", "law"]
        assert b["longest_pinned_run"] == sum(r["length"] for r in runs)
        assert b["longest_pinned_run"] > b["longest_pinned_run_by_branch"]["law"]
        assert b["branch_at_start"] == "law"

    def test_the_runs_are_contiguous_and_indexed_into_the_episode(self):
        mdp, b = self._drive()
        for r in b["pinned_runs"]:
            window = mdp.step_records[r["start"] : r["start"] + r["length"]]
            assert all(s["immobile"] for s in window)
            assert {s["branch"] for s in window} == {r["branch"]}


class TestRunBookkeeping:
    """`branch_summary` on hand-written step records, where every run length is
    known by construction."""

    @staticmethod
    def _steps(spec: str) -> list[dict]:
        """`spec` is one character per step: `L`/`S` an immobile law/search
        step, `l`/`s` a moving one."""
        return [
            {
                "branch": "law" if c in "Ll" else "search",
                "in_span": True,
                "immobile": c.isupper(),
                "stalled": c.isupper(),
                "intervention_magnitude": 0.0,
            }
            for c in spec
        ]

    def test_a_moving_step_ends_both_kinds_of_run(self):
        b = branch_summary(self._steps("LLLlLL"), 0.0)
        assert b["longest_pinned_run_by_branch"]["law"] == 3
        assert b["longest_pinned_run"] == 3

    def test_a_branch_change_ends_only_the_by_branch_run(self):
        b = branch_summary(self._steps("LLLSSLL"), 0.0)
        assert b["longest_pinned_run_by_branch"] == {"law": 3, "search": 2}
        assert b["longest_pinned_run"] == 7
        assert b["branch_at_start"] == "law"

    def test_only_runs_at_least_the_threshold_are_kept(self):
        spec = "L" * MIN_PINNED_RUN + "l" + "L" * (MIN_PINNED_RUN - 1)
        kept = branch_summary(self._steps(spec), 0.0)["pinned_runs"]
        assert kept == [{"branch": "law", "start": 0, "length": MIN_PINNED_RUN}]

    def test_an_episode_with_no_immobile_step(self):
        b = branch_summary(self._steps("llss"), 0.0)
        assert b["longest_pinned_run"] == 0
        assert b["branch_at_start"] is None
        assert b["pinned_runs"] == []
        assert b["longest_pinned_run_by_branch"] == {"law": 0, "search": 0}

    def test_an_empty_record(self):
        b = branch_summary([], 0.0)
        assert b["n_steps"] == 0
        assert b["longest_pinned_run"] == 0
        assert b["intervention_total"] == 0.0


# ----------------------------------------------------------------------------
# without a shield
# ----------------------------------------------------------------------------
class TestUnshielded:
    def test_the_shield_fields_are_zero_and_the_branches_are_still_tagged(self):
        mdp = _mdp(shield=False)
        b = mdp.play_episode(_Full().predict, SEED)["branch"]
        assert b["n_steps"] > 0
        assert b["intervention_by_branch"] == {"law": 0.0, "search": 0.0}
        assert b["intervention_total"] == 0.0
        assert b["stalled_by_branch"] == {"law": 0, "search": 0}
        assert b["immobile_by_branch"] == {"law": 0, "search": 0}
        assert b["longest_pinned_run"] == 0
        assert b["pinned_runs"] == []
        assert sum(b["steps_by_branch"].values()) == b["n_steps"]
        assert all(
            s["intervention"] is False
            and s["min_clearance"] == 0.0
            and s["intervention_magnitude"] == 0.0
            for s in mdp.step_records
        )

    def test_both_branches_are_seen_without_a_shield(self):
        """Attribution is a property of the pursuer, not of the filter: an
        unshielded episode still splits into the two components."""
        mdp = _mdp(shield=False)
        seen = set()
        for i in range(5):
            mdp.play_episode(_Full().predict, SEED + i)
            seen |= {s["branch"] for s in mdp.step_records}
        assert seen == {"law", "search"}


# ----------------------------------------------------------------------------
# the recorder does not change the episode
# ----------------------------------------------------------------------------
class TestObservationFree:
    def _plain(self, seed: int) -> dict:
        cfg = EnvConfig()
        mdp = SearchMDP(
            cfg,
            GreedyEvader,
            hold=HOLD,
            shield=CBFShield(cfg),
            shield_obs=True,
            escape_obs=True,
            reward_wrapper=None,
        )
        return mdp.play_episode(_Full().predict, seed)

    @pytest.mark.parametrize("seed", [SEED, SEED + 1, SEED + 2])
    def test_the_log_is_the_plain_envs_log(self, seed):
        got = _mdp().play_episode(_Full().predict, seed)
        assert "branch" in got
        assert {k: v for k, v in got.items() if k != "branch"} == self._plain(seed)

    def test_the_record_is_per_episode(self):
        """`_begin` resets the record, so a second episode's summary is not the
        first one's continued."""
        mdp = _mdp()
        first = mdp.play_episode(_Full().predict, SEED)["branch"]
        second = mdp.play_episode(_Full().predict, SEED + 1)["branch"]
        assert len(mdp.step_records) == second["n_steps"]
        assert first["n_steps"] > 0 and second["n_steps"] > 0

    def test_an_unknown_branch_is_an_error_not_a_silent_third_column(self):
        mdp = _mdp()
        mdp._begin(SEED)
        with pytest.raises(ValueError, match="unknown branch"):
            mdp._step_inner(FULL, "planner")


# ----------------------------------------------------------------------------
# the replay script's comparison logic
# ----------------------------------------------------------------------------
_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "replay_s18_branch_split.py"
)


def _replay():
    spec = importlib.util.spec_from_file_location("replay_s18_branch_split", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RP = _replay()

# `s18_arm_SPARSE-PEN-70.json -> final.gate.GREEDY`, trimmed to the two fields
# the check compares. The real values, so the fixture cannot be a rounder number
# than the thing it stands for.
RECORDED = {
    "capture_rate": 0.596,
    "intervention": {"total": 125376.45209560267, "n_episodes": 500},
}


class TestReproductionComparison:
    def test_an_exact_replay_passes(self):
        out = RP.compare_reproduction(
            {
                "intervention_total": RECORDED["intervention"]["total"],
                "capture_rate": RECORDED["capture_rate"],
                "n_episodes": 500,
            },
            RECORDED,
        )
        assert out["verdict"] == "PASS"
        assert all(c["exact"] for c in out["checks"].values())
        assert out["checks"]["intervention_total"]["difference"] == 0.0

    def test_one_ulp_of_intervention_fails_and_reports_the_difference(self):
        """No tolerance to fall back to: same code, same seeds, a deterministic
        policy and a memoryless filter, so anything but equality is a finding."""
        off = math.nextafter(RECORDED["intervention"]["total"], math.inf)
        out = RP.compare_reproduction(
            {
                "intervention_total": off,
                "capture_rate": RECORDED["capture_rate"],
                "n_episodes": 500,
            },
            RECORDED,
        )
        assert out["verdict"] == "FAIL"
        assert out["checks"]["intervention_total"]["exact"] is False
        assert out["checks"]["intervention_total"]["difference"] != 0.0
        assert out["checks"]["capture_rate"]["exact"] is True

    def test_a_different_capture_rate_fails(self):
        out = RP.compare_reproduction(
            {
                "intervention_total": RECORDED["intervention"]["total"],
                "capture_rate": 0.594,
                "n_episodes": 500,
            },
            RECORDED,
        )
        assert out["verdict"] == "FAIL"
        assert out["checks"]["capture_rate"]["difference"] == pytest.approx(-0.002)

    def test_a_short_replay_fails_on_the_episode_count(self):
        out = RP.compare_reproduction(
            {
                "intervention_total": RECORDED["intervention"]["total"],
                "capture_rate": RECORDED["capture_rate"],
                "n_episodes": 499,
            },
            RECORDED,
        )
        assert out["verdict"] == "FAIL"
        assert out["checks"]["n_episodes"]["difference"] == -1


class TestAggregate:
    """The pooled reading, including plan section 4's S21 trigger."""

    @staticmethod
    def _summary(law_run: int, search_run: int, iv_law: float, iv_search: float):
        return {
            "intervention_by_branch": {"law": iv_law, "search": iv_search},
            "steps_by_branch": {"law": 10, "search": 20},
            "stalled_by_branch": {"law": 1, "search": 2},
            "immobile_by_branch": {"law": law_run, "search": search_run},
            "longest_pinned_run_by_branch": {"law": law_run, "search": search_run},
            "pre_decision_intervention": 0.5,
        }

    def test_shares_and_the_takeover_trigger(self):
        agg = RP.aggregate(
            [
                self._summary(50, 0, 3.0, 1.0),
                self._summary(10, 5, 1.0, 1.0),
                self._summary(0, 60, 0.0, 2.0),
                self._summary(45, 0, 0.0, 0.0),
            ]
        )
        assert agg["n_episodes"] == 4
        assert agg["intervention_by_branch"] == {"law": 4.0, "search": 4.0}
        assert agg["intervention_share_by_branch"]["law"] == pytest.approx(0.5)
        # two of four episodes carry a law pin of at least 40 steps
        assert agg["episode_frac_law_pin_ge_40"] == pytest.approx(0.5)
        assert agg["episode_frac_search_pin_ge_40"] == pytest.approx(0.25)
        assert agg["longest_pinned_run_by_branch"]["law"]["max"] == 50
        assert agg["pre_decision_intervention"] == pytest.approx(2.0)

    def test_an_all_zero_round_does_not_divide_by_zero(self):
        agg = RP.aggregate([self._summary(0, 0, 0.0, 0.0)])
        assert agg["intervention_share_by_branch"] == {"law": 0.0, "search": 0.0}
        assert agg["episode_frac_law_pin_ge_40"] == 0.0
