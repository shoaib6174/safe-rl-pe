"""Round N1 (internal S27): branch attribution of the archived deadlock episodes.

Spec: docs/pathway/paper_b_prereg_n1_branch_attribution_2026-11-03.md.
The pure functions are tested on synthetic per-step records; the smoke class
replays three archived episodes of three cells and requires them to equal the
archive (CG-N1-1 in miniature) and the recomputed deadlock flag to equal the
environment's (CG-N1-2)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "paper_b_branch_attribution", ROOT / "scripts" / "paper_b_branch_attribution.py"
)
N1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N1)

K = N1.K


def rec(branch: str, visible: bool, stalled: bool, immobile: bool | None = None) -> dict:
    return {
        "branch": branch,
        "visible": visible,
        "stalled": stalled,
        "immobile": stalled if immobile is None else immobile,
    }


def free(n: int) -> list[dict]:
    return [rec("search", False, False, False) for _ in range(n)]


# ----------------------------------------------------------------------------
class TestComponent:
    def test_law_with_the_evader_visible_is_chase(self):
        assert N1.component(rec("law", True, True)) == "chase"

    def test_law_with_the_evader_out_of_view_is_last_seen(self):
        assert N1.component(rec("law", False, True)) == "last_seen"

    def test_search_is_head_whatever_the_visibility(self):
        assert N1.component(rec("search", False, True)) == "head"
        assert N1.component(rec("search", True, True)) == "head"

    def test_an_unknown_branch_is_refused(self):
        with pytest.raises(ValueError):
            N1.component(rec("teleport", False, True))


class TestStallRuns:
    def test_maximal_runs_with_censoring(self):
        assert N1.stall_runs([False, True, True, False, True]) == [(1, 2, False), (4, 1, True)]

    def test_no_stall_no_run(self):
        assert N1.stall_runs([False] * 7) == []

    def test_an_episode_that_is_one_run(self):
        assert N1.stall_runs([True] * 5) == [(0, 5, True)]


class TestEpisodeAttribution:
    def test_a_run_one_short_of_k_is_not_a_deadlock(self):
        out = N1.episode_attribution(free(3) + [rec("law", False, True)] * (K - 1) + free(2))
        assert out["deadlock"] is False
        assert out["first_k"] is None
        assert out["k_runs"] == []

    def test_a_last_seen_run_is_attributed_last_seen_at_k(self):
        out = N1.episode_attribution(free(5) + [rec("law", False, True)] * (K + 5))
        assert out["deadlock"] is True
        fk = out["first_k"]
        assert fk["component_at_k"] == "last_seen"
        assert fk["start"] == 5 and fk["length"] == K + 5 and fk["censored"] is True
        assert fk["crossing"] is False

    def test_the_component_is_read_at_the_kth_step_of_a_crossing_run(self):
        steps = free(2) + [rec("search", False, True)] * 10 + [rec("law", False, True)] * (K - 5) + free(1)
        out = N1.episode_attribution(steps)
        fk = out["first_k"]
        assert fk["start"] == 2 and fk["length"] == 10 + K - 5
        assert fk["component_at_k"] == "last_seen"  # index start + K - 1 lies in the law part
        assert fk["crossing"] is True
        assert fk["composition"] == {"chase": 0, "last_seen": K - 5, "head": 10}

    def test_a_head_run(self):
        out = N1.episode_attribution([rec("search", False, True)] * (K + 10) + free(3))
        assert out["first_k"]["component_at_k"] == "head"
        assert out["first_k"]["censored"] is False

    def test_the_first_k_run_decides_and_every_k_run_is_listed(self):
        steps = [rec("law", False, True)] * K + free(4) + [rec("search", False, True)] * (K + 1)
        out = N1.episode_attribution(steps)
        assert out["first_k"]["component_at_k"] == "last_seen"
        assert [r["component_at_k"] for r in out["k_runs"]] == ["last_seen", "head"]
        assert out["any_head_k"] is True

    def test_step_counts_by_component(self):
        steps = (
            [rec("law", True, False, False)] * 3
            + [rec("law", False, True)] * 4
            + [rec("search", False, False, True)] * 2
        )
        out = N1.episode_attribution(steps)
        assert out["steps_by_component"] == {"chase": 3, "last_seen": 4, "head": 2}
        assert out["stalled_by_component"] == {"chase": 0, "last_seen": 4, "head": 0}
        assert out["immobile_by_component"] == {"chase": 0, "last_seen": 4, "head": 2}


class TestAggregate:
    def test_shares_and_counts(self):
        eps = [
            N1.episode_attribution([rec("law", False, True)] * K),
            N1.episode_attribution([rec("law", False, True)] * K),
            N1.episode_attribution([rec("search", False, True)] * K),
            N1.episode_attribution(free(K)),
        ]
        agg = N1.aggregate(eps)
        assert agg["n_episodes"] == 4 and agg["n_deadlock"] == 3
        assert agg["first_k"] == {"chase": 0, "last_seen": 2, "head": 1}
        assert agg["first_k_share"]["last_seen"] == pytest.approx(2 / 3)
        assert agg["first_k_share"]["head"] == pytest.approx(1 / 3)

    def test_no_deadlock_gives_no_share(self):
        agg = N1.aggregate([N1.episode_attribution(free(K))])
        assert agg["n_deadlock"] == 0
        assert agg["first_k_share"] is None


# ----------------------------------------------------------------------------
class TestCells:
    def test_forty_cells_literal(self):
        assert len(N1.CELLS) == 40
        assert sum(c["set"] == "A" for c in N1.CELLS.values()) == 36
        assert sum(c["set"] == "B" for c in N1.CELLS.values()) == 4

    def test_set_a_is_the_four_bare_families_by_three_seeds_by_three_opponents(self):
        a = {n for n, c in N1.CELLS.items() if c["set"] == "A"}
        want = {
            f"{f}-{s}_{o}"
            for f in ("BASE", "OBS", "GAM", "OBSGAM")
            for s in (60, 61, 62)
            for o in ("ATK4-45", "HUG7", "GREEDY")
        }
        assert a == want

    def test_set_b_is_the_two_discovery_policies_plain(self):
        b = {n for n, c in N1.CELLS.items() if c["set"] == "B"}
        assert b == {"STR14_ATK4-45", "RC19_ATK4-45", "STR14_GREEDY", "RC19_GREEDY"}


class TestReproductionCompare:
    ARCH = {
        "capture_ind": [True, False, True],
        "deadlock_ind": [False, True, False],
        "capture_rate": 2 / 3,
        "deadlock_episode_rate": 1 / 3,
        "stall_step_rate": 0.25,
        "immobile_step_rate": 0.5,
    }

    def test_identical_passes(self):
        out = N1.compare_reproduction(dict(self.ARCH), self.ARCH)
        assert out["pass"] is True

    def test_one_flipped_episode_fails_and_is_named(self):
        rep = dict(self.ARCH, deadlock_ind=[False, False, False])
        out = N1.compare_reproduction(rep, self.ARCH)
        assert out["pass"] is False
        assert out["deadlock_ind_mismatch"] == [1]

    def test_a_rate_that_differs_in_the_last_bit_fails(self):
        rep = dict(self.ARCH, stall_step_rate=0.2500000000000001)
        out = N1.compare_reproduction(rep, self.ARCH)
        assert out["pass"] is False

    def test_a_prefix_compares_lists_only(self):
        rep = {"capture_ind": [True, False], "deadlock_ind": [False, True]}
        out = N1.compare_reproduction(rep, self.ARCH, prefix=2)
        assert out["pass"] is True


class TestStopRule:
    def test_a_cell_with_a_record_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(N1, "CELLS_DIR", tmp_path)
        (tmp_path / "BASE-60_ATK4-45.json").write_text("{}")
        with pytest.raises(SystemExit):
            N1.run_cell("BASE-60_ATK4-45")

    def test_summarize_refuses_without_the_watch_note(self, tmp_path, monkeypatch):
        spec = tmp_path / "spec.md"
        spec.write_text("no watch note here")
        monkeypatch.setattr(N1, "SPEC", spec)
        with pytest.raises(SystemExit):
            N1.summarize()


# ----------------------------------------------------------------------------
_HAVE_ARCHIVE = (ROOT / "results" / "derisk" / "s17_arm_BASE-60.json").exists() and (
    ROOT / "results" / "derisk" / "s26_b2_cells" / "STR14_GREEDY_PLAIN.json"
).exists()


@pytest.mark.skipif(not _HAVE_ARCHIVE, reason="archived records and checkpoints not present")
class TestSmokeReplay:
    @pytest.mark.parametrize("cell", ["BASE-60_GREEDY", "OBS-61_HUG7", "STR14_GREEDY"])
    def test_three_episodes_equal_the_archive(self, cell, monkeypatch):
        monkeypatch.chdir(ROOT)
        out = N1.replay(cell, n_episodes=3)
        arch = N1.archived(cell)
        cmp = N1.compare_reproduction(out["replayed"], arch, prefix=3)
        assert cmp["pass"], cmp
        assert out["cg_n1_2"]["mismatch"] == []
        assert out["cg_n1_4"]["pass"], out["cg_n1_4"]
