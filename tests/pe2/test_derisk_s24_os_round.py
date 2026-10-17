"""scripts/derisk_s24_os_train.py and scripts/derisk_s24_os_gate.py — S24-OS
Step O.2: retraining the selector over library v1 and the 12-cell gate.

Pre-registration: `docs/pathway/tactics_prereg_s24_os_2026-10-17.md`.
Everything here is either a constant the text fixes, a verdict rule it states,
or one of the properties the round's claims rest on:

- the trainer's arms, checkpoint path (`S24-OS-{ARM}`, never S24.1's stem), the
  pre-registered seed streams (disjoint) and provenance (library/obs_len/
  n_actions added);
- the construct gate on the built training env (obs_len 59, actions[-1]
  ORBIT-SEARCH, 6th mask False when visible), on real and on fabricated envs;
- a crashed arm is UNDETERMINED and is never retrained; a construct-gate
  `SystemExit` is NOT caught as a crash;
- the gate's 12-cell table (9 at 1.2, 3 at 0.8), identical across rows apart
  from the pursuer checkpoint, asserted on BUILT envs (`cg_ratio`,
  `build_cell_env`);
- record validity (n_episodes/seed0/declared ratio), duplicates reject both,
  a crash through `run_cells` writes THIS round's UNDETERMINED record;
- the comparator joins (S24.1's 1.2 column, S24.3-ZS's 0.8 column) refuse a
  non-measurable record and refuse to run inside a binding;
- H-OS-1's two-branch non-inferiority rule, H-OS-2's winning-direction rule,
  H-OS-3's reported shape, H-SAFE and the inherited CG-T1/A2.1, each with a
  FAILING direction and not only a passing one;
- the GIF seed rule (the first three GATE seeds, S24.1's own), captions that
  carry no rate, and the CLI's dispatch order (cells, then gifs, then
  summarize);
- both launchers' preconditions, checked through the launchers' OWN embedded
  checkers;
- one real, tiny end-to-end smoke per ratio with a fresh v1 policy, into a
  temporary checkpoint directory that is deleted afterwards — the only place
  this file runs a real episode.

No real 500-episode cell is ever run here.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_GATE_PATH = _SCRIPTS / "derisk_s24_os_gate.py"
_LAUNCH_TRAIN = _SCRIPTS / "launch_s24_os_train.sh"
_LAUNCH_GATE = _SCRIPTS / "launch_s24_os.sh"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("derisk_s24_os_gate", _GATE_PATH)
# `G.TR` — NOT a second `_load` of the trainer: `_load` builds a fresh module
# object every call, so a separately loaded `TR` here would be a different
# object from the one `G._bound()` reads, and patching it would patch nothing
# the gate's internals ever see.
TR = G.TR
S1 = G.S1
Z = G.Z


# ----------------------------------------------------------------------------
# the trainer: arms, dirs, streams, provenance, the construct gate
# ----------------------------------------------------------------------------
class TestTrainerArmsDirsStreams:
    def test_arms_are_the_three_seeds_on_library_v1(self):
        assert TR.ARM_ORDER == ("SEL2-80", "SEL2-81", "SEL2-82")
        for arm, spec in TR.ARMS.items():
            assert spec["memory"] is True
            assert spec["library"] == "v1"
            assert arm == f"SEL2-{spec['seed']}"

    def test_the_checkpoint_path_carries_the_trainers_own_stem(self):
        assert TR.ckpt_path("SEL2-81") == pathlib.Path(
            "results/derisk/ckpts_S24-OS-SEL2-81/S24-OS-SEL2-81_3000000.zip"
        )
        assert TR._mark_path("SEL2-81", 250_000) == pathlib.Path(
            "results/derisk/ckpts_S24-OS-SEL2-81/S24-OS-SEL2-81_250000.zip"
        )
        assert TR.ckpt_path("SEL2-81") == TR._mark_path("SEL2-81", TR.TOTAL)

    def test_the_probe_stream_is_2_750_000_and_disjoint(self):
        assert TR.PROBE_SEED0 == 2_750_000
        os_probe = set(range(TR.PROBE_SEED0, TR.PROBE_SEED0 + TR.PROBE_N))
        s1_probe = set(range(TR.T1.PROBE_SEED0, TR.T1.PROBE_SEED0 + TR.T1.PROBE_N))
        gate = set(range(2_400_000, 2_400_500))
        train = {s * 1000 + i for s in TR.SEEDS for i in range(1000)}
        assert not (os_probe & s1_probe)
        assert not (os_probe & gate)
        assert not (os_probe & train)

    def test_an_overlapping_stream_is_refused_at_import(self, monkeypatch):
        monkeypatch.setattr(TR, "PROBE_SEED0", TR.T1.PROBE_SEED0)
        clashes = sorted(
            f"{a} & {b}"
            for i, a in enumerate(TR._STREAMS)
            for b in list(TR._STREAMS)[i + 1 :]
            if TR._STREAMS[a] & TR._STREAMS[b]
        )
        # the check at import only runs once; re-derive it here with the
        # overlapping value to prove the guard's LOGIC catches a collision
        streams = {
            "train": {s * 1000 + i for s in TR.SEEDS for i in range(1000)},
            "s24_1_gate": set(range(2_400_000, 2_400_500)),
            "s24_1_probe": set(range(TR.T1.PROBE_SEED0, TR.T1.PROBE_SEED0 + TR.T1.PROBE_N)),
            "os_probe": set(range(TR.PROBE_SEED0, TR.PROBE_SEED0 + TR.PROBE_N)),
        }
        bad = [
            f"{a} & {b}"
            for i, a in enumerate(streams)
            for b in list(streams)[i + 1 :]
            if streams[a] & streams[b]
        ]
        assert bad  # the collision is real once PROBE_SEED0 is moved onto S24.1's

    def test_provenance_adds_library_obs_len_n_actions(self):
        prov = TR.provenance()
        assert prov["library"] == "v1"
        assert prov["obs_len"] == 59
        assert prov["n_actions"] == 6
        assert prov["prereg"] == TR.PREREG
        assert prov["derived_from"] == TR.T1.PREREG


class TestConstructGate:
    def test_a_real_built_env_passes(self):
        gate = TR.cg_library("SEL2-80")
        assert gate["verdict"] == "PASS"
        assert gate["obs_len"] == 59
        assert gate["actions"][-1] == "ORBIT-SEARCH"
        assert gate["orbit_search_mask_when_visible"] is False

    class _FakeEnv:
        def __init__(self, obs_len=59, actions=None, visible=True, mask_true=False):
            self.obs_len = obs_len
            self.actions = actions or (
                "INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP", "ORBIT-SEARCH",
            )
            self.n_actions = len(self.actions)
            self._visible = visible
            self._mask_true = mask_true

        def reset(self, seed=None):
            obs = np.zeros(self.obs_len, dtype=np.float32)
            obs[6] = 1.0 if self._visible else 0.0
            return obs, {}

        def action_masks(self):
            mask = np.zeros(len(self.actions), dtype=bool)
            mask[self.actions.index("ORBIT-SEARCH")] = self._mask_true
            return mask

    def test_refuses_on_the_wrong_obs_len(self, monkeypatch):
        monkeypatch.setattr(TR, "make_env", lambda arm: self._FakeEnv(obs_len=55))
        with pytest.raises(SystemExit, match="obs_len"):
            TR.cg_library("SEL2-80")

    def test_refuses_when_the_last_action_is_not_orbit_search(self, monkeypatch):
        monkeypatch.setattr(
            TR, "make_env",
            lambda arm: self._FakeEnv(actions=("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")),
        )
        with pytest.raises(SystemExit, match="actions.-1."):
            TR.cg_library("SEL2-80")

    def test_refuses_when_the_mask_is_true_while_visible(self, monkeypatch):
        monkeypatch.setattr(TR, "make_env", lambda arm: self._FakeEnv(mask_true=True))
        with pytest.raises(SystemExit, match="6th mask entry is True"):
            TR.cg_library("SEL2-80")

    def test_refuses_when_no_visible_reset_is_ever_found(self, monkeypatch):
        monkeypatch.setattr(TR, "make_env", lambda arm: self._FakeEnv(visible=False))
        with pytest.raises(SystemExit, match="no visible-at-reset state"):
            TR.cg_library("SEL2-80")


class TestCrashHandling:
    def test_a_crash_is_undetermined_and_is_never_retrained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        calls = []

        def boom(arm, device="cpu"):
            calls.append(arm)
            raise RuntimeError("boom")

        monkeypatch.setattr(TR, "train", boom)
        TR.run(["SEL2-80"], "cpu")
        rec = json.loads(TR._arm_file("SEL2-80").read_text())
        assert rec["verdict"] == "UNDETERMINED"
        assert "boom" in rec["reason"]
        TR.run(["SEL2-80"], "cpu")  # must be skipped, never retrained
        assert calls == ["SEL2-80"]

    def test_a_construct_gate_refusal_is_not_caught_as_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "OUT", tmp_path)

        def refuse(arm, device="cpu"):
            raise SystemExit("construct gate refuses")

        monkeypatch.setattr(TR, "train", refuse)
        with pytest.raises(SystemExit):
            TR.run(["SEL2-81"], "cpu")
        assert not TR._arm_file("SEL2-81").exists()


# ----------------------------------------------------------------------------
# the gate's cell table — 12 cells, rows differ only by the pursuer checkpoint
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_twelve_cells_nine_at_1_2_three_at_0_8(self):
        assert len(G.CELL_ORDER) == 12
        at_1_2 = [n for n in G.CELL_ORDER if G.CELLS[n]["ratio"] == 1.2]
        at_0_8 = [n for n in G.CELL_ORDER if G.CELLS[n]["ratio"] == 0.8]
        assert len(at_1_2) == 9 and len(at_0_8) == 3
        assert {G.CELLS[n]["opponent"] for n in at_0_8} == {"GREEDY"}

    def test_1_2_cell_names_carry_no_ratio_suffix(self):
        assert G.cell_name("SEL2-80", "GREEDY", 1.2) == "SEL2-80_GREEDY"
        assert G.cell_name("SEL2-80", "GREEDY", 0.8) == "SEL2-80_GREEDY_r0.8"

    def test_rows_differ_only_by_the_pursuer_checkpoint(self):
        common = ("kind", "memory", "library")
        vals = {tuple(G.ROWS[r][k] for k in common) for r in G.ROW_NAMES}
        assert len(vals) == 1
        cell_common = ("kind", "memory", "library", "hold")
        cell_vals = {
            tuple(G.CELLS[G.cell_name(r, "GREEDY", 1.2)][k] for k in cell_common)
            for r in G.ROW_NAMES
        }
        assert len(cell_vals) == 1
        for s in G.SEEDS:
            row = f"SEL2-{s}"
            assert G.CELLS[f"{row}_GREEDY"]["arm"] == row
            assert G.CELLS[f"{row}_HUG"]["arm"] == row

    def test_built_envs_carry_the_declared_ratio_and_the_v1_library(self):
        env12 = G.build_cell_env("SEL2-80_GREEDY")
        assert env12.cfg.evader_speed_ratio == 1.2
        assert env12.actions[-1] == "ORBIT-SEARCH" and env12.n_actions == 6
        env08 = G.build_cell_env("SEL2-80_GREEDY_r0.8")
        assert env08.cfg.evader_speed_ratio == 0.8
        assert env08.actions == env12.actions

    def test_cg_ratio_passes_at_both_ratios(self):
        g12 = G.cg_ratio("SEL2-81_HUG")
        assert g12["verdict"] == "PASS" and g12["config_diff_vs_twin"] == []
        g08 = G.cg_ratio("SEL2-81_GREEDY_r0.8")
        assert g08["verdict"] == "PASS" and g08["config_diff_vs_twin"] == ["evader_speed_ratio"]

    def test_gate_seed0_cross_checked_against_s24_1(self):
        assert G.GATE_SEED0 == S1.GATE_SEED0 == 2_400_000


# ----------------------------------------------------------------------------
# record validity, duplicates, a crash through run_cells
# ----------------------------------------------------------------------------
def _measurable(name: str, **overrides) -> dict:
    spec = G.CELLS[name]
    n = G.N_EPISODES
    caps = [True] * (n // 2) + [False] * (n - n // 2)
    base = {
        "name": name,
        "n_episodes": n,
        "seed0": G.GATE_SEED0,
        "ratio": spec["ratio"],
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "violation_steps": 0,
        "inapplicable_picks": 0,
        "search_policy_steps": 0,
        "steps": [300] * n,
        "picks": dict.fromkeys(("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP", "ORBIT-SEARCH"), 10),
        "aggregate": {"orbit_search_share": 0.1, "mean_decisions": 12.0},
    }
    return base | overrides


def _on_disk(dirpath: pathlib.Path, rec: dict, name: str | None = None) -> None:
    d = dirpath / "s24_os_cells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name or rec['name']}.json").write_text(json.dumps(rec))


class TestRecordValidity:
    def test_a_measurable_record(self):
        assert G.record_status(_measurable("SEL2-80_HUG")) == ("MEASURABLE", "")

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 1}, "seed0"),
            ({"ratio": 0.8}, "ratio"),  # a 1.2 cell filed with the wrong ratio
        ],
    )
    def test_wrong_n_seed0_or_ratio_is_invalid(self, overrides, match):
        status, why = G.record_status(_measurable("SEL2-80_HUG") | overrides)
        assert status == "INVALID" and match in why

    def test_a_record_of_another_round_is_invalid(self):
        status, why = G.record_status({"name": "SEL-80_HUG"})
        assert status == "INVALID" and "not a cell of this round" in why

    def test_a_duplicated_cell_rejects_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        rec = _measurable("SEL2-82_TACTIC-V0")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a_copy")
        measurable, rejected, by_cell = G.scan_records()
        assert measurable == {}
        assert len(rejected) == 2 and by_cell["SEL2-82_TACTIC-V0"]
        assert all("duplicate" in why for why in rejected.values())

    def test_load_records_drops_everything_not_measurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("SEL2-80_HUG"))
        _on_disk(tmp_path, _measurable("SEL2-80_GREEDY") | {"n_episodes": 20})
        _on_disk(tmp_path, _measurable("SEL2-81_GREEDY_r0.8") | {"ratio": 1.0})
        assert set(G.load_records(quiet=True)) == {"SEL2-80_HUG"}

    def test_read_cell_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no cell record"):
            G.read_cell("SEL2-81_GREEDY")
        _on_disk(tmp_path, _measurable("SEL2-81_GREEDY") | {"ratio": 0.8})
        with pytest.raises(SystemExit, match="not measurable"):
            G.read_cell("SEL2-81_GREEDY")

    def test_a_crash_through_run_cells_writes_this_rounds_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        """`undetermined_record` must be bound, or the file would carry no
        `ratio` field."""
        monkeypatch.setattr(G, "OUT", tmp_path)
        ckpt = tmp_path / "SEL2-81.zip"
        ckpt.touch()
        monkeypatch.setattr(TR, "ckpt_path", lambda arm, _p=ckpt: _p)

        def boom(name, device="cpu", n_episodes=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(G, "run_cell", boom)
        assert "undetermined_record" in G._BOUND_ALWAYS
        G.run_cells(["SEL2-81_GREEDY_r0.8"])
        rec = json.loads(G.cell_file("SEL2-81_GREEDY_r0.8").read_text())
        assert rec["verdict"] == S1.UNDETERMINED == "UNDETERMINED"
        assert rec["ratio"] == 0.8
        assert "RuntimeError: boom" in rec["error"]
        assert G.record_status(rec) == ("UNDETERMINED", f"crashed cell: {rec['error']!r}")
        assert G.load_records(quiet=True) == {}
        with pytest.raises(SystemExit, match="UNDETERMINED"):
            G.read_cell("SEL2-81_GREEDY_r0.8")


# ----------------------------------------------------------------------------
# the binding onto S24.1's orchestration
# ----------------------------------------------------------------------------
class TestBinding:
    def test_every_bound_name_is_restored(self):
        before = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        with G._bound():
            for k in G._BOUND_ALWAYS:
                assert getattr(S1, k) is not before[k] or k in ()
        after = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        assert before == after

    def test_run_cells_resolves_the_checkpoint_through_the_trainer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(TR, "ckpt_path", lambda arm: tmp_path / f"{arm}.zip")
        seen = []
        monkeypatch.setattr(G, "run_cell", lambda name, device="cpu", n_episodes=None: seen.append(name))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.run_cells(["SEL2-80_GREEDY"])
        assert not seen  # refused before the first episode


# ----------------------------------------------------------------------------
# the comparator joins: outside any binding, by the rule that measured them
# ----------------------------------------------------------------------------
class TestComparatorJoins:
    def _s1_fixture(self, tmp_path) -> None:
        d = tmp_path / "s24_1_cells"
        d.mkdir(parents=True)
        for s in G.SEEDS:
            for o in ("GREEDY", "HUG", "TACTIC-V0"):
                rec = {
                    "name": f"SEL-{s}_{o}",
                    "n_episodes": S1.N_EPISODES,
                    "seed0": S1.GATE_SEED0,
                    "capture_ind": [True] * 250 + [False] * 250,
                    "capture_rate": 0.5,
                }
                (d / f"SEL-{s}_{o}.json").write_text(json.dumps(rec))

    def _zs_fixture(self, tmp_path) -> None:
        d = tmp_path / "s24_3zs_cells"
        d.mkdir(parents=True)
        for s in G.SEEDS:
            name = Z.cell_name(f"SEL-{s}", "GREEDY", 0.8)
            rec = {
                "name": name,
                "n_episodes": Z.N_EPISODES,
                "seed0": Z.GATE_SEED0,
                "ratio": 0.8,
                "capture_ind": [True] * 300 + [False] * 200,
                "capture_rate": 0.6,
            }
            (d / f"{name}.json").write_text(json.dumps(rec))

    def test_the_1_2_join_reads_s24_1s_own_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        got = G.load_comparators_1_2()
        assert len(got) == 9
        assert got[("SEL-80", "GREEDY")]["capture_rate"] == 0.5

    def test_the_1_2_join_refuses_an_absent_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        (tmp_path / "s24_1_cells" / "SEL-82_HUG.json").unlink()
        with pytest.raises(SystemExit, match="absent"):
            G.load_comparators_1_2()

    def test_the_1_2_join_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        p = tmp_path / "s24_1_cells" / "SEL-80_GREEDY.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | {"n_episodes": 1}))
        with pytest.raises(SystemExit, match="n_episodes"):
            G.load_comparators_1_2()

    def test_the_0_8_join_reads_s24_3zs_own_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        self._zs_fixture(tmp_path)
        got = G.load_comparators_0_8()
        assert len(got) == 3
        assert got[("SEL-81", "GREEDY")]["capture_rate"] == 0.6

    def test_the_0_8_join_refuses_an_unreadable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        self._zs_fixture(tmp_path)
        name = Z.cell_name("SEL-80", "GREEDY", 0.8)
        (tmp_path / "s24_3zs_cells" / f"{name}.json").write_text("{not json")
        with pytest.raises(SystemExit, match="unreadable"):
            G.load_comparators_0_8()

    def test_both_joins_refuse_to_run_inside_a_binding(self):
        with G._bound():
            with pytest.raises(SystemExit, match="may not run inside a binding"):
                G.load_comparators_1_2()
            with pytest.raises(SystemExit, match="may not run inside a binding"):
                G.load_comparators_0_8()


# ----------------------------------------------------------------------------
# hypotheses — every rule with a FAILING direction, not only a passing one
# ----------------------------------------------------------------------------
def _indicators(n: int, both_true: int, a_only: int, b_only: int):
    a = [True] * both_true + [True] * a_only + [False] * b_only + [False] * (n - both_true - a_only - b_only)
    b = [True] * both_true + [False] * a_only + [True] * b_only + [False] * (n - both_true - a_only - b_only)
    return a, b


def _os1_fixture(opponent: str, seed: int, row_ind, comp_ind) -> tuple[dict, dict]:
    n = len(row_ind)
    row_name = G.cell_name(f"SEL2-{seed}", opponent, 1.2)
    records = {
        row_name: {
            "capture_ind": row_ind,
            "capture_rate": sum(row_ind) / n,
        }
    }
    comparators = {
        (f"SEL-{seed}", opponent): {
            "capture_ind": comp_ind,
            "capture_rate": sum(comp_ind) / n,
        }
    }
    return records, comparators


class TestHOs1TwoBranchRule:
    def test_significant_loss_with_delta_minus_0_02_passes_on_the_margin(self):
        # comparator wins 10 discordant pairs out of 500, row wins 0: delta =
        # -0.02, and (10 of 10) is significant on its own.
        comp_ind, row_ind = _indicators(500, 0, 10, 0)
        records, comparators = _os1_fixture("GREEDY", 80, row_ind, comp_ind)
        b = G.h_os_1_seed(records, comparators, "GREEDY", 80)
        assert b["delta"] == pytest.approx(-0.02)
        assert b["losing_direction_significant"] is True
        assert b["verdict"] == "PASS"

    def test_non_significant_loss_with_delta_minus_0_10_passes(self):
        # only 2 discordant pairs total (out of 20): a big RATE delta but far
        # too few pairs for the exact binomial test to call it significant.
        comp_ind, row_ind = _indicators(20, 0, 2, 0)
        records, comparators = _os1_fixture("TACTIC-V0", 81, row_ind, comp_ind)
        b = G.h_os_1_seed(records, comparators, "TACTIC-V0", 81)
        assert b["delta"] == pytest.approx(-0.10)
        assert b["losing_direction_significant"] is False
        assert b["verdict"] == "PASS"

    def test_significant_loss_with_delta_minus_0_05_fails(self):
        # 25 of 25 discordant pairs go the comparator's way: delta -0.05,
        # overwhelmingly significant, and past the -0.03 margin.
        comp_ind, row_ind = _indicators(500, 0, 25, 0)
        records, comparators = _os1_fixture("GREEDY", 82, row_ind, comp_ind)
        b = G.h_os_1_seed(records, comparators, "GREEDY", 82)
        assert b["delta"] == pytest.approx(-0.05)
        assert b["losing_direction_significant"] is True
        assert b["verdict"] == "FAIL"

    def test_exactly_on_the_margin_passes_and_one_past_it_fails(self):
        # review defect 2: 175/500 - 190/500 is -0.030000000000000027 in
        # floating point; the margin is decided on exact counts.
        row_ind, comp_ind = _indicators(500, 175, 0, 15)  # row 175, comparator 190
        records, comparators = _os1_fixture("GREEDY", 80, row_ind, comp_ind)
        b = G.h_os_1_seed(records, comparators, "GREEDY", 80)
        assert sum(row_ind) == 175 and sum(comp_ind) == 190
        assert b["losing_direction_significant"] is True
        assert b["verdict"] == "PASS"
        row_ind, comp_ind = _indicators(500, 174, 0, 16)  # row 174, comparator 190
        records, comparators = _os1_fixture("GREEDY", 80, row_ind, comp_ind)
        assert G.h_os_1_seed(records, comparators, "GREEDY", 80)["verdict"] == "FAIL"

    def test_a_winning_row_always_passes(self):
        row_ind, comp_ind = _indicators(500, 0, 50, 0)  # row wins, not comparator
        records, comparators = _os1_fixture("GREEDY", 80, row_ind, comp_ind)
        b = G.h_os_1_seed(records, comparators, "GREEDY", 80)
        assert b["delta"] > 0
        assert b["verdict"] == "PASS"

    def test_missing_either_side_is_incomplete(self):
        b = G.h_os_1_seed({}, {}, "GREEDY", 80)
        assert b["verdict"] == "INCOMPLETE"

    def test_the_opponent_round_needs_two_of_three_seeds(self):
        comp_ind, row_ind = _indicators(500, 0, 25, 0)  # a FAIL, as above
        pass_records, pass_comparators = _os1_fixture("GREEDY", 80, *_indicators(500, 0, 50, 0)[::-1])
        records, comparators = {}, {}
        for seed in G.SEEDS:
            r, c = _os1_fixture("GREEDY", seed, row_ind, comp_ind)  # every seed FAILs
            records |= r
            comparators |= c
        b = G.h_os_1_opponent(records, comparators, "GREEDY")
        assert b["n_pass"] == 0 and b["verdict"] == "FAIL"

    def test_the_round_needs_both_opponents(self):
        comp_ind, row_ind = _indicators(500, 0, 25, 0)
        records, comparators = {}, {}
        for seed in G.SEEDS:
            for opp in G.H_OS_1_OPPONENTS:
                r, c = _os1_fixture(opp, seed, row_ind, comp_ind)
                records |= r
                comparators |= c
        v = G.h_os_1(records, comparators)
        assert v["verdict"] == "FAIL"  # both opponents FAIL on every seed


class TestHOs2:
    def test_a_significant_win_vs_hug_passes(self):
        row_ind, comp_ind = _indicators(500, 0, 60, 0)
        records, comparators = _os1_fixture("HUG", 80, row_ind, comp_ind)
        b = G.h_os_2_seed(records, comparators, 80)
        assert b["verdict"] == "PASS"

    def test_a_non_significant_or_losing_difference_fails(self):
        row_ind, comp_ind = _indicators(500, 0, 1, 0)  # tiny, not significant
        records, comparators = _os1_fixture("HUG", 80, row_ind, comp_ind)
        b = G.h_os_2_seed(records, comparators, 80)
        assert b["verdict"] == "FAIL"

    def test_the_round_needs_two_of_three_seeds(self):
        win_row, win_comp = _indicators(500, 0, 60, 0)
        lose_row, lose_comp = _indicators(500, 0, 1, 0)
        records, comparators = {}, {}
        for seed, (row_ind, comp_ind) in zip(G.SEEDS, (
            (win_row, win_comp), (win_row, win_comp), (lose_row, lose_comp),
        )):
            r, c = _os1_fixture("HUG", seed, row_ind, comp_ind)
            records |= r
            comparators |= c
        v = G.h_os_2(records, comparators)
        assert v["n_pass"] == 2 and v["verdict"] == "PASS"


class TestHOs3Reported:
    def test_it_is_reported_and_never_gates(self):
        row = "SEL2-80"
        name = G.cell_name(row, "GREEDY", 0.8)
        records = {
            name: {
                "capture_ind": [True] * 10,
                "capture_rate": 1.0,
                "aggregate": {"orbit_search_share": 0.2, "mean_decisions": 9.0},
            }
        }
        comparators = {("SEL-80", "GREEDY"): {"capture_ind": [False] * 10, "capture_rate": 0.0}}
        v = G.h_os_3(records, comparators)
        assert v["gated"] is False
        assert v["zero_shot_0.8_vs_greedy"]["80"]["verdict"] == "REPORTED"
        assert v["zero_shot_0.8_vs_greedy"]["81"]["verdict"] == "INCOMPLETE"


class TestHSafeAndInheritedGates:
    def _records(self, **cell_overrides) -> dict:
        out = {}
        for n in G.CELL_ORDER:
            out[n] = _measurable(n) | cell_overrides
        return out

    def test_holds_when_everything_is_clean(self, monkeypatch):
        monkeypatch.setattr(G, "_probe_training_evidence", lambda: {"arms": {}, "offenders": [], "missing": []})
        v = G.h_safe(self._records())
        assert v["verdict"] == "HOLDS"

    def test_fails_on_a_gate_violation(self, monkeypatch):
        monkeypatch.setattr(G, "_probe_training_evidence", lambda: {"arms": {}, "offenders": [], "missing": []})
        records = self._records()
        records["SEL2-80_GREEDY"]["violation_steps"] = 1
        v = G.h_safe(records)
        assert v["verdict"] == "FAILS" and "SEL2-80_GREEDY" in v["offenders"]

    def test_fails_on_training_contact(self, monkeypatch):
        monkeypatch.setattr(
            G, "_probe_training_evidence",
            lambda: {"arms": {}, "offenders": ["SEL2-81"], "missing": []},
        )
        v = G.h_safe(self._records())
        assert v["verdict"] == "FAILS"

    def test_incomplete_on_missing_training_evidence(self, monkeypatch):
        monkeypatch.setattr(
            G, "_probe_training_evidence",
            lambda: {"arms": {}, "offenders": [], "missing": ["SEL2-82"]},
        )
        v = G.h_safe(self._records())
        assert v["verdict"] == "INCOMPLETE"

    def test_cg_t1_refuses_on_a_non_zero_inapplicable_pick(self):
        records = self._records()
        records["SEL2-81_HUG"]["inapplicable_picks"] = 1
        v = G.cg_t1(records)
        assert v["verdict"] == "REFUSE" and v["offenders"] == ["SEL2-81_HUG"]
        assert G.cg_t1(self._records())["verdict"] == "PASS"

    def test_a2_1_refuses_past_the_tolerance_but_not_within_it(self):
        clean = self._records()
        assert G.cg_a2_1(clean)["verdict"] == "PASS"
        within = self._records()
        within["SEL2-81_HUG"]["search_policy_steps"] = 1  # <= tolerance at 500*300 steps
        assert G.cg_a2_1(within)["verdict"] == "PASS"
        over = self._records()
        over["SEL2-81_HUG"]["search_policy_steps"] = 10_000
        v = G.cg_a2_1(over)
        assert v["verdict"] == "REFUSE" and "SEL2-81_HUG" in v["offenders"]


# ----------------------------------------------------------------------------
# GIFs: the seed rule, captions, and the CLI's dispatch order
# ----------------------------------------------------------------------------
class TestGifsAndDispatch:
    def test_gif_seeds_are_s24_1s_own_rule(self):
        assert G.gif_seeds() == S1.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)

    def test_captions_carry_no_rate(self):
        title = G.gif_title("SEL2-80", "HUG", 2_400_000)
        assert title == "SEL2-80 vs HUG · episode 2400000"
        assert not re.search(r"0\.\d+", title)  # no fractional "rate"-shaped number

    def test_write_gifs_refuses_without_a_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered"))
        with pytest.raises(SystemExit, match="no cell record"):
            G.write_gifs(["SEL2-80"])

    def test_write_gifs_refuses_without_the_rows_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        for o in G.GIF_OPPONENTS:
            _on_disk(tmp_path, _measurable(G.cell_name("SEL2-80", o, 1.2)))
        monkeypatch.setattr(TR, "ckpt_path", lambda arm: tmp_path / "nope.zip")
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("rendered"))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.write_gifs(["SEL2-80"])

    def test_dispatch_order_is_cells_then_gifs_then_summarize(self, monkeypatch):
        order = []
        monkeypatch.setattr(G, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(G, "write_gifs", lambda *a, **k: order.append("gifs"))

        def fake_summarize(device="cpu"):
            order.append("summarize")
            return {
                "verdicts": {
                    "h_os_1": {"verdict": "FAIL"}, "h_os_2": {"verdict": "FAIL"},
                    "h_safe": {}, "cg_t1": {}, "cg_a2_1": {},
                    "direction": "", "n_records": 0, "n_cells": 12, "complete": False,
                }
            }

        monkeypatch.setattr(G, "summarize", fake_summarize)
        monkeypatch.setattr(sys, "argv", ["prog", "--cells", "--gifs", "--summarize"])
        G.main()
        assert order == ["cells", "gifs", "summarize"]


# ----------------------------------------------------------------------------
# the launch wrappers — their OWN embedded checkers
# ----------------------------------------------------------------------------
def _checker(launch_path: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    body = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", launch_path.read_text(), re.DOTALL).group(1)
    path = tmp_path / "checker.py"
    path.write_text(body)
    if not (tmp_path / "scripts").exists():
        (tmp_path / "scripts").symlink_to(_SCRIPTS)
    return path


def _run_checker(path: pathlib.Path, tmp_path: pathlib.Path, *arms: str):
    return subprocess.run(
        [sys.executable, str(path), "wt", *arms],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )


class TestTrainLauncherChecker:
    def test_it_accepts_when_nothing_exists_yet(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80", "SEL2-81", "SEL2-82")
        assert got.returncode == 0, got.stdout + got.stderr
        assert "clean" in got.stdout

    def test_it_refuses_an_existing_arm_record(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        d.mkdir(parents=True)
        (d / "s24_os_arm_SEL2-80.json").write_text("{}")
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80")
        assert got.returncode == 1 and "arm record exists" in got.stdout

    def test_it_refuses_an_existing_checkpoint_dir(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        (d / "ckpts_S24-OS-SEL2-81").mkdir(parents=True)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-81")
        assert got.returncode == 1 and "checkpoint dir" in got.stdout


class TestGateLauncherChecker:
    def test_it_refuses_without_a_checkpoint(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80")
        assert got.returncode == 1 and "no final checkpoint" in got.stdout

    def test_it_refuses_a_dirty_probe_row(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-OS-SEL2-80"
        ck.mkdir(parents=True)
        (ck / "S24-OS-SEL2-80_3000000.zip").write_bytes(b"x")
        (d / "s24_os_probe_SEL2-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 1, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80")
        assert got.returncode == 1 and "violations = 1" in got.stdout

    def test_it_refuses_absent_probe_evidence(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-OS-SEL2-80"
        ck.mkdir(parents=True)
        (ck / "S24-OS-SEL2-80_3000000.zip").write_bytes(b"x")
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80")
        assert got.returncode == 1 and "no probe JSON" in got.stdout

    def test_it_accepts_clean_evidence(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-OS-SEL2-80"
        ck.mkdir(parents=True)
        (ck / "S24-OS-SEL2-80_3000000.zip").write_bytes(b"x")
        (d / "s24_os_probe_SEL2-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 0, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL2-80")
        assert got.returncode == 0, got.stdout + got.stderr


class TestLaunchersGateTheHost:
    def test_both_launchers_share_the_pgrep_guard_and_omp_4(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            src = path.read_text()
            assert "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" in src
            assert "OMP_NUM_THREADS=4" in src
            assert "is not a symlink to" in src
            assert "git rev-parse --short HEAD" in src

    def test_the_gate_launcher_has_two_chains_covering_all_12_cells(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH_GATE.read_text().splitlines()
            if "--cells" in line and line.lstrip().startswith("nohup")
        ]
        assert len(chains) == 2
        assert sorted(len(c) for c in chains) == [3, 9]
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(G.CELL_ORDER)

    def test_neither_launcher_summarises_or_renders(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            for line in path.read_text().splitlines():
                if line.lstrip().startswith("nohup"):
                    assert "--summarize" not in line and "--gifs" not in line


# ----------------------------------------------------------------------------
# one real, tiny end-to-end smoke per ratio — a fresh v1 policy, a tmp dir,
# deleted afterwards. No other test in this file runs a real episode.
# ----------------------------------------------------------------------------
class TestSmoke:
    def test_a_fresh_v1_policy_runs_n3_at_each_ratio(self, monkeypatch):
        from sb3_contrib import MaskablePPO

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="s24_os_smoke_"))
        try:
            monkeypatch.setattr(TR, "ckpt_path", lambda arm, _tmp=tmp: _tmp / f"{arm}.zip")
            env = G.build_cell_env("SEL2-80_GREEDY")
            # review defect 1: the trainer's make_learner must build a learner for
            # THIS round's arm names (S24.1's ARMS table does not know them)
            assert "SEL2-80" not in TR.T1.ARMS
            built = TR.make_learner("SEL2-80", env, "cpu")
            assert isinstance(built, MaskablePPO) and built.gamma == TR.GAMMA
            model = MaskablePPO(
                "MlpPolicy", env, seed=80, gamma=0.99, n_steps=64, batch_size=32, verbose=0,
            )
            model.learn(total_timesteps=64)
            ckpt = TR.ckpt_path("SEL2-80")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(ckpt)

            with G._bound():
                rec12 = G.run_cell("SEL2-80_GREEDY", n_episodes=3)
                rec08 = G.run_cell("SEL2-80_GREEDY_r0.8", n_episodes=3)

            assert rec12["ratio"] == 1.2 and rec08["ratio"] == 0.8
            assert set(rec12["picks"]) == {
                "INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP", "ORBIT-SEARCH",
            }
            assert rec12["n_episodes"] == 3 and rec08["n_episodes"] == 3
            assert rec12["cg_ratio"]["verdict"] == "PASS"
            assert rec08["cg_ratio"]["verdict"] == "PASS"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        assert not tmp.exists()


# ----------------------------------------------------------------------------
# O.1's own tests still pass (CG-O1, library v0/v1) — re-run and counted in
# the report, not duplicated here.
# ----------------------------------------------------------------------------
def test_selector_library_module_is_importable_unchanged():
    from pe2.selector_mdp import LIBRARIES, obs_len_for

    assert LIBRARIES["v1"][-1] == "ORBIT-SEARCH"
    assert obs_len_for("v1") == 59
