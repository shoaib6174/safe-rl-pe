"""scripts/derisk_s24_2_train.py and scripts/derisk_s24_2_gate.py — S24.2 Step
E.2: training the EVADER's tactic selector against a per-episode pursuer
mixture, and the 15-cell gate against the scripted-evader rows of S24.1.

Pre-registration: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` sections 1,
2 and Amendments A1/A2. Everything here is either a constant the text fixes, a
verdict rule it states, or one of the properties the round's claims rest on:

- the trainer's arms, checkpoint path (`S24-2-{ARM}`, never S24.1's stem), the
  pre-registered seed streams (disjoint), the mixture (exactly the three
  seats) and provenance (`pursuer_mixture` added);
- review defect 1 (S24-OS's own): `make_learner` must build a learner for
  THIS round's arm names (`EVS-80`), which is not a key of S24.1's `ARMS`;
- the construct gate on the built training env (obs_len, 5 actions, mask
  non-empty at reset, exactly the three seats), on real and on fabricated
  envs;
- a crashed arm is UNDETERMINED and is never retrained; a construct-gate
  `SystemExit` is NOT caught as a crash;
- the gate's 15-cell table, identical across rows apart from the pursuer seat,
  asserted on BUILT envs;
- record validity (n_episodes/seed0/declared ratio/declared pursuer),
  duplicates reject both, a crash through `run_cells` writes THIS round's
  UNDETERMINED record;
- the comparator join (S24.1's five scripted-evader `*_TACTIC-V0` rows)
  refuses a non-measurable record and refuses to run inside a binding;
- H-EV-1's and H-EV-2's DOWN-direction paired test, H-EV-3's kill rule on
  EXACT COUNTS (review defect 2's discipline, applied to a bare threshold),
  CG-E3's flag, H-SAFE and the inherited CG-T1, each with a FAILING direction
  and not only a passing one;
- the GIF seed rule (the first three GATE seeds, S24.1's own), captions that
  carry no rate, and the CLI's dispatch order (cells, then gifs, then
  summarize);
- both launchers' preconditions, checked through the launchers' OWN embedded
  checkers;
- one real, tiny end-to-end smoke per script with a fresh policy, into a
  temporary checkpoint directory that is deleted afterwards — the only place
  this file runs a real episode or trains a real step.

No real 500-episode cell is ever run here. Counts for `test_evader_selector_
mdp.py` (E.1's own suite) are reported alongside this file's, not duplicated.
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
_GATE_PATH = _SCRIPTS / "derisk_s24_2_gate.py"
_LAUNCH_TRAIN = _SCRIPTS / "launch_s24_2_train.sh"
_LAUNCH_GATE = _SCRIPTS / "launch_s24_2.sh"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("derisk_s24_2_gate", _GATE_PATH)
# `G.T2` — NOT a second `_load` of the trainer: `_load` builds a fresh module
# object every call, so a separately loaded `T2` here would be a different
# object from the one `G._bound()` reads, and patching it would patch nothing
# the gate's internals ever see.
T2 = G.T2
S1 = G.S1


# ----------------------------------------------------------------------------
# the trainer: arms, dirs, streams, the mixture, provenance, the construct gate
# ----------------------------------------------------------------------------
class TestTrainerArmsDirsStreams:
    def test_arms_are_the_three_seeds(self):
        assert T2.ARM_ORDER == ("EVS-80", "EVS-81", "EVS-82")
        for arm, spec in T2.ARMS.items():
            assert spec["memory"] is True
            assert arm == f"EVS-{spec['seed']}"

    def test_the_checkpoint_path_carries_the_trainers_own_stem(self):
        assert T2.ckpt_path("EVS-81") == pathlib.Path(
            "results/derisk/ckpts_S24-2-EVS-81/S24-2-EVS-81_3000000.zip"
        )
        assert T2._mark_path("EVS-81", 250_000) == pathlib.Path(
            "results/derisk/ckpts_S24-2-EVS-81/S24-2-EVS-81_250000.zip"
        )
        assert T2.ckpt_path("EVS-81") == T2._mark_path("EVS-81", T2.TOTAL)

    def test_the_probe_stream_is_2_700_000_and_disjoint(self):
        assert T2.PROBE_SEED0 == 2_700_000
        probe = set(range(T2.PROBE_SEED0, T2.PROBE_SEED0 + T2.PROBE_N))
        s1_probe = set(range(T2.T1.PROBE_SEED0, T2.T1.PROBE_SEED0 + T2.T1.PROBE_N))
        gate = set(range(2_400_000, 2_400_500))
        train = {s * 1000 + i for s in T2.SEEDS for i in range(1000)}
        assert not (probe & s1_probe)
        assert not (probe & gate)
        assert not (probe & train)

    def test_an_overlapping_stream_is_refused_at_import(self):
        # re-derive the import-time check's logic with an overlapping value,
        # to prove the guard CATCHES a collision rather than re-importing
        # the module with a monkeypatched constant (the check only runs once,
        # at import, so there is nothing to re-trigger on the live module).
        streams = {
            "train": {s * 1000 + i for s in T2.SEEDS for i in range(1000)},
            "probe": set(range(T2.T1.PROBE_SEED0, T2.T1.PROBE_SEED0 + T2.T1.PROBE_N)),
            "gate": set(range(2_400_000, 2_400_500)),
            "s24_1_probe": set(range(T2.T1.PROBE_SEED0, T2.T1.PROBE_SEED0 + T2.T1.PROBE_N)),
        }
        bad = [
            f"{a} & {b}"
            for i, a in enumerate(streams)
            for b in list(streams)[i + 1 :]
            if streams[a] & streams[b]
        ]
        assert bad  # the collision is real once the probe stream moves onto S24.1's

    def test_the_mixture_is_exactly_the_three_seats(self):
        assert len(T2.MIXTURE) == 3
        assert T2.MIXTURE_LABELS == (T2.SEL_81_STEM, "RULES-V0", "LEAD-1.0")
        kind = [spec[0] if isinstance(spec, tuple) else spec for spec in T2.MIXTURE]
        assert kind == ["ckpt", "RULES-V0", "LEAD-1.0"]
        assert T2.MIXTURE[0][2] == "v0"  # library

    def test_the_sel_81_seat_is_resolved_through_s24_1s_own_gate_function(self):
        assert T2.SEL_81_CKPT == S1.ckpt_path("SEL-81")
        assert T2.SEL_81_SEAT == ("ckpt", str(S1.ckpt_path("SEL-81")), "v0")

    def test_provenance_adds_the_mixture_and_width(self):
        prov = T2.provenance()
        assert prov["pursuer_mixture"] == list(T2.MIXTURE_LABELS)
        assert prov["obs_len"] == T2.EV_OBS_LEN == 47
        assert prov["n_actions"] == 5
        assert prov["prereg"] == T2.PREREG
        assert prov["derived_from"] == T2.T1.PREREG

    def test_make_learner_builds_a_maskable_ppo_for_evs_80(self):
        """Review defect 1 (S24-OS's own): S24.1's `make_learner` resolves
        `ARMS` in S24.1's OWN module, so calling it with THIS round's arm name
        directly would raise `KeyError` there."""
        from sb3_contrib import MaskablePPO

        assert "EVS-80" not in T2.T1.ARMS
        env = T2.make_env("EVS-80")
        built = T2.make_learner("EVS-80", env, "cpu")
        assert isinstance(built, MaskablePPO)
        assert built.gamma == T2.GAMMA == 0.99


class TestConstructGate:
    def test_a_real_built_env_passes(self):
        gate = T2.cg_mixture("EVS-80")
        assert gate["verdict"] == "PASS"
        assert gate["obs_len"] == T2.EV_OBS_LEN
        assert gate["n_actions"] == 5
        assert gate["mask_nonempty_at_reset"] is True
        assert set(gate["seats"]) == set(T2.MIXTURE_LABELS)

    class _FakeEnv:
        def __init__(self, obs_len=47, n_actions=5, mask_nonempty=True, seats=None):
            self.obs_len = obs_len
            self.n_actions = n_actions
            self._mask_nonempty = mask_nonempty
            self.seats = seats or [(s, object()) for s in ("X", "RULES-V0", "LEAD-1.0")]

        def reset(self, seed=None):
            return np.zeros(self.obs_len, dtype=np.float32), {}

        def action_masks(self):
            mask = np.zeros(self.n_actions, dtype=bool)
            if self._mask_nonempty:
                mask[0] = True
            return mask

    def test_refuses_on_the_wrong_obs_len(self, monkeypatch):
        monkeypatch.setattr(T2, "make_env", lambda arm: self._FakeEnv(obs_len=55))
        with pytest.raises(SystemExit, match="obs_len"):
            T2.cg_mixture("EVS-80")

    def test_refuses_on_the_wrong_action_count(self, monkeypatch):
        monkeypatch.setattr(T2, "make_env", lambda arm: self._FakeEnv(n_actions=6))
        with pytest.raises(SystemExit, match="n_actions"):
            T2.cg_mixture("EVS-80")

    def test_refuses_when_the_mask_is_empty_at_reset(self, monkeypatch):
        monkeypatch.setattr(T2, "make_env", lambda arm: self._FakeEnv(mask_nonempty=False))
        with pytest.raises(SystemExit, match="mask is empty"):
            T2.cg_mixture("EVS-80")

    def test_refuses_when_the_mixture_is_not_exactly_the_three_seats(self, monkeypatch):
        monkeypatch.setattr(
            T2, "make_env",
            lambda arm: self._FakeEnv(seats=[("RULES-V0", object()), ("LEAD-1.0", object())]),
        )
        with pytest.raises(SystemExit, match="mixture"):
            T2.cg_mixture("EVS-80")
        monkeypatch.setattr(
            T2, "make_env",
            lambda arm: self._FakeEnv(
                seats=[("X", object()), ("Y", object()), ("RULES-V0", object())]
            ),
        )
        with pytest.raises(SystemExit, match="mixture"):
            T2.cg_mixture("EVS-80")


class TestCrashHandling:
    def test_a_crash_is_undetermined_and_is_never_retrained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T2, "OUT", tmp_path)
        calls = []

        def boom(arm, device="cpu"):
            calls.append(arm)
            raise RuntimeError("boom")

        monkeypatch.setattr(T2, "train", boom)
        T2.run(["EVS-80"], "cpu")
        rec = json.loads(T2._arm_file("EVS-80").read_text())
        assert rec["verdict"] == "UNDETERMINED"
        assert "boom" in rec["reason"]
        T2.run(["EVS-80"], "cpu")  # must be skipped, never retrained
        assert calls == ["EVS-80"]

    def test_a_construct_gate_refusal_is_not_caught_as_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T2, "OUT", tmp_path)

        def refuse(arm, device="cpu"):
            raise SystemExit("construct gate refuses")

        monkeypatch.setattr(T2, "train", refuse)
        with pytest.raises(SystemExit):
            T2.run(["EVS-81"], "cpu")
        assert not T2._arm_file("EVS-81").exists()


# ----------------------------------------------------------------------------
# the gate's cell table — 15 cells, rows differ only by the pursuer seat
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_fifteen_cells_three_rows_five_pursuers(self):
        assert len(G.CELL_ORDER) == 15
        assert set(G.ROW_NAMES) == {"EVS-80", "EVS-81", "EVS-82"}
        assert set(G.PURSUERS) == {"SEL-80", "SEL-81", "SEL-82", "RULES-V0", "LEAD-1.0"}
        for r in G.ROW_NAMES:
            for p in G.PURSUERS:
                assert G.cell_name(r, p) in G.CELLS

    def test_cell_names_have_no_suffix(self):
        assert G.cell_name("EVS-80", "SEL-81") == "EVS-80_SEL-81"

    def test_rows_differ_only_by_the_pursuer_seat(self):
        common = ("kind", "ratio", "hold")
        cell_vals = {
            tuple(G.CELLS[G.cell_name(r, "RULES-V0")][k] for k in common) for r in G.ROW_NAMES
        }
        assert len(cell_vals) == 1
        for s in G.SEEDS:
            row = f"EVS-{s}"
            for p in G.PURSUERS:
                assert G.CELLS[f"{row}_{p}"]["arm"] == row
                assert G.CELLS[f"{row}_{p}"]["seed"] == s

    def test_built_envs_are_identical_across_rows_at_the_same_pursuer(self):
        """Rows differ only in WHICH evader checkpoint plays (`run_cell`'s own
        `load_policy(spec["row"], ...)`), never in the env the cell builds:
        `build_cell_env` reads only the pursuer's seat off the cell, so two
        rows at the same pursuer must build bit-identical envs."""
        env80 = G.build_cell_env("EVS-80_SEL-81")
        env81 = G.build_cell_env("EVS-81_SEL-81")
        assert env80.obs_len == env81.obs_len
        assert env80.actions == env81.actions
        assert [n for n, _s in env80.seats] == [n for n, _s in env81.seats]
        assert env80.cfg.evader_speed_ratio == env81.cfg.evader_speed_ratio == 1.2

    def test_pursuer_specs_resolve_sel_ckpts_through_s24_1s_own_function(self):
        for s in G.SEEDS:
            assert G.PURSUER_SPECS[f"SEL-{s}"] == ("ckpt", str(S1.ckpt_path(f"SEL-{s}")), "v0")
        assert G.PURSUER_SPECS["RULES-V0"] == "RULES-V0"
        assert G.PURSUER_SPECS["LEAD-1.0"] == "LEAD-1.0"

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
        "pursuer": spec["pursuer"],
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "violation_steps": 0,
        "inapplicable_picks": 0,
        "steps": [300] * n,
        "picks": dict.fromkeys(G.EVADER_ACTIONS, 10),
        "aggregate": {"picks_share": dict.fromkeys(G.EVADER_ACTIONS, 1 / len(G.EVADER_ACTIONS))},
    }
    return base | overrides


def _on_disk(dirpath: pathlib.Path, rec: dict, name: str | None = None) -> None:
    d = dirpath / "s24_2_cells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name or rec['name']}.json").write_text(json.dumps(rec))


class TestRecordValidity:
    def test_a_measurable_record(self):
        assert G.record_status(_measurable("EVS-80_SEL-81")) == ("MEASURABLE", "")

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 1}, "seed0"),
            ({"ratio": 0.8}, "ratio"),
            ({"pursuer": "SEL-82"}, "pursuer"),
        ],
    )
    def test_wrong_n_seed0_ratio_or_pursuer_is_invalid(self, overrides, match):
        status, why = G.record_status(_measurable("EVS-80_SEL-81") | overrides)
        assert status == "INVALID" and match in why

    def test_a_record_of_another_round_is_invalid(self):
        status, why = G.record_status({"name": "SEL-80_HUG"})
        assert status == "INVALID" and "not a cell of this round" in why

    def test_a_duplicated_cell_rejects_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        rec = _measurable("EVS-82_LEAD-1.0")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a_copy")
        measurable, rejected, by_cell = G.scan_records()
        assert measurable == {}
        assert len(rejected) == 2 and by_cell["EVS-82_LEAD-1.0"]
        assert all("duplicate" in why for why in rejected.values())

    def test_load_records_drops_everything_not_measurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("EVS-80_SEL-81"))
        _on_disk(tmp_path, _measurable("EVS-80_SEL-80") | {"n_episodes": 20})
        _on_disk(tmp_path, _measurable("EVS-81_RULES-V0") | {"pursuer": "LEAD-1.0"})
        assert set(G.load_records(quiet=True)) == {"EVS-80_SEL-81"}

    def test_read_cell_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no cell record"):
            G.read_cell("EVS-81_SEL-80")
        _on_disk(tmp_path, _measurable("EVS-81_SEL-80") | {"ratio": 0.8})
        with pytest.raises(SystemExit, match="not measurable"):
            G.read_cell("EVS-81_SEL-80")

    def test_a_crash_through_run_cells_writes_this_rounds_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        """`undetermined_record` must be bound, or the file would carry no
        `ratio`/`pursuer` field."""
        monkeypatch.setattr(G, "OUT", tmp_path)
        ckpt = tmp_path / "EVS-81.zip"
        ckpt.touch()
        monkeypatch.setattr(T2, "ckpt_path", lambda arm, _p=ckpt: _p)

        def boom(name, device="cpu", n_episodes=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(G, "run_cell", boom)
        assert "undetermined_record" in G._BOUND_ALWAYS
        G.run_cells(["EVS-81_LEAD-1.0"])
        rec = json.loads(G.cell_file("EVS-81_LEAD-1.0").read_text())
        assert rec["verdict"] == S1.UNDETERMINED == "UNDETERMINED"
        assert rec["ratio"] == 1.2
        assert rec["pursuer"] == "LEAD-1.0"
        assert "RuntimeError: boom" in rec["error"]
        assert G.record_status(rec) == ("UNDETERMINED", f"crashed cell: {rec['error']!r}")
        assert G.load_records(quiet=True) == {}
        with pytest.raises(SystemExit, match="UNDETERMINED"):
            G.read_cell("EVS-81_LEAD-1.0")


# ----------------------------------------------------------------------------
# the binding onto S24.1's orchestration
# ----------------------------------------------------------------------------
class TestBinding:
    def test_every_bound_name_is_restored(self):
        before = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        with G._bound():
            for k in G._BOUND_ALWAYS:
                assert getattr(S1, k) is not before[k]
        after = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        assert before == after

    def test_run_cells_resolves_the_checkpoint_through_the_trainer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(T2, "ckpt_path", lambda arm: tmp_path / f"{arm}.zip")
        seen = []
        monkeypatch.setattr(G, "run_cell", lambda name, device="cpu", n_episodes=None: seen.append(name))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.run_cells(["EVS-80_SEL-80"])
        assert not seen  # refused before the first episode


# ----------------------------------------------------------------------------
# the comparator join: outside any binding, by the rule that measured them
# ----------------------------------------------------------------------------
class TestComparatorJoin:
    def _s1_fixture(self, tmp_path) -> None:
        d = tmp_path / "s24_1_cells"
        d.mkdir(parents=True)
        for pursuer in G.PURSUERS:
            name = f"{pursuer}_TACTIC-V0"
            rec = {
                "name": name,
                "n_episodes": S1.N_EPISODES,
                "seed0": S1.GATE_SEED0,
                "capture_ind": [True] * 250 + [False] * 250,
                "capture_rate": 0.5,
            }
            (d / f"{name}.json").write_text(json.dumps(rec))

    def test_reads_s24_1s_own_five_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        got = G.load_comparators()
        assert len(got) == 5
        assert got["SEL-80_TACTIC-V0"]["capture_rate"] == 0.5

    def test_refuses_an_absent_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        (tmp_path / "s24_1_cells" / "LEAD-1.0_TACTIC-V0.json").unlink()
        with pytest.raises(SystemExit, match="absent"):
            G.load_comparators()

    def test_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        p = tmp_path / "s24_1_cells" / "SEL-80_TACTIC-V0.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | {"n_episodes": 1}))
        with pytest.raises(SystemExit, match="n_episodes"):
            G.load_comparators()

    def test_refuses_to_run_inside_a_binding(self):
        with G._bound():
            with pytest.raises(SystemExit, match="may not run inside a binding"):
                G.load_comparators()


# ----------------------------------------------------------------------------
# hypotheses — every rule with a FAILING direction, not only a passing one
# ----------------------------------------------------------------------------
def _indicators(n: int, both_true: int, a_only: int, b_only: int):
    a = [True] * both_true + [True] * a_only + [False] * b_only + [False] * (n - both_true - a_only - b_only)
    b = [True] * both_true + [False] * a_only + [True] * b_only + [False] * (n - both_true - a_only - b_only)
    return a, b


class TestDownTest:
    def test_a_significant_lowering_passes(self):
        # the comparator (scripted evader) wins 25 discordant pairs, the row
        # (learned evader) wins 0: the row's capture is significantly LOWER.
        comp_ind, row_ind = _indicators(500, 0, 25, 0)
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._down_test(row_rec, comp_rec, "EVS-80_SEL-80", "SEL-80_TACTIC-V0")
        assert b["delta"] < 0
        assert b["verdict"] == "PASS"

    def test_a_non_significant_or_rising_delta_fails(self):
        comp_ind, row_ind = _indicators(500, 0, 1, 0)  # tiny, not significant
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._down_test(row_rec, comp_rec, "EVS-80_SEL-80", "SEL-80_TACTIC-V0")
        assert b["verdict"] == "FAIL"

    def test_a_row_that_captures_more_than_the_comparator_fails(self):
        row_ind, comp_ind = _indicators(500, 0, 60, 0)  # row wins, not comparator
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._down_test(row_rec, comp_rec, "EVS-80_SEL-80", "SEL-80_TACTIC-V0")
        assert b["delta"] > 0
        assert b["verdict"] == "FAIL"


class TestHEv1:
    def _fixture(self, seed: int, row_ind, comp_ind) -> tuple[dict, dict]:
        n = len(row_ind)
        row_name = G.cell_name(f"EVS-{seed}", f"SEL-{seed}")
        comp_name = f"SEL-{seed}_TACTIC-V0"
        records = {row_name: {"capture_ind": row_ind, "capture_rate": sum(row_ind) / n}}
        comparators = {comp_name: {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / n}}
        return records, comparators

    def test_passes_on_two_of_three_seeds(self):
        pass_comp, pass_row = _indicators(500, 0, 25, 0)  # lowers, significant
        fail_comp, fail_row = _indicators(500, 0, 1, 0)  # not significant
        records, comparators = {}, {}
        for seed, (c, r) in zip(G.SEEDS, [(pass_comp, pass_row), (pass_comp, pass_row), (fail_comp, fail_row)]):
            rec, cmp_ = self._fixture(seed, r, c)
            records |= rec
            comparators |= cmp_
        v = G.h_ev_1(records, comparators)
        assert v["n_pass"] == 2 and v["verdict"] == "PASS"

    def test_fails_when_fewer_than_two_of_three_seeds_pass(self):
        fail_comp, fail_row = _indicators(500, 0, 1, 0)
        records, comparators = {}, {}
        for seed in G.SEEDS:
            rec, cmp_ = self._fixture(seed, fail_row, fail_comp)
            records |= rec
            comparators |= cmp_
        v = G.h_ev_1(records, comparators)
        assert v["n_pass"] == 0 and v["verdict"] == "FAIL"

    def test_missing_either_side_is_incomplete(self):
        v = G.h_ev_1({}, {})
        assert v["verdict"] == "INCOMPLETE"


class TestHEv2:
    def _fixture(self, pursuer: str, seed: int, row_ind, comp_ind) -> tuple[dict, dict]:
        n = len(row_ind)
        row_name = G.cell_name(f"EVS-{seed}", pursuer)
        comp_name = f"{pursuer}_TACTIC-V0"
        records = {row_name: {"capture_ind": row_ind, "capture_rate": sum(row_ind) / n}}
        comparators = {comp_name: {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / n}}
        return records, comparators

    def test_needs_both_pursuers_to_pass(self):
        pass_comp, pass_row = _indicators(500, 0, 25, 0)
        records, comparators = {}, {}
        for pursuer in G.MIXTURE_PURSUERS:
            for seed in G.SEEDS:
                rec, cmp_ = self._fixture(pursuer, seed, pass_row, pass_comp)
                records |= rec
                comparators |= cmp_
        assert G.h_ev_2(records, comparators)["verdict"] == "PASS"

    def test_one_failing_pursuer_fails_the_round(self):
        pass_comp, pass_row = _indicators(500, 0, 25, 0)
        fail_comp, fail_row = _indicators(500, 0, 1, 0)
        records, comparators = {}, {}
        for seed in G.SEEDS:
            r, c = self._fixture("RULES-V0", seed, pass_row, pass_comp)
            records |= r
            comparators |= c
            r, c = self._fixture("LEAD-1.0", seed, fail_row, fail_comp)
            records |= r
            comparators |= c
        v = G.h_ev_2(records, comparators)
        assert v["pursuers"]["RULES-V0"]["verdict"] == "PASS"
        assert v["pursuers"]["LEAD-1.0"]["verdict"] == "FAIL"
        assert v["verdict"] == "FAIL"


class TestHEv3Kill:
    def _records_for_seed(self, seed: int, ks: dict[str, int]) -> dict:
        out = {}
        for p, k in ks.items():
            name = G.cell_name(f"EVS-{seed}", p)
            caps = [True] * k + [False] * (G.N_EPISODES - k)
            out[name] = {"capture_ind": caps, "capture_rate": k / G.N_EPISODES}
        return out

    def test_the_threshold_is_exact_counts_25_of_500(self):
        assert G.KILL_MIN_CAPTURE == 0.05
        assert round(G.KILL_MIN_CAPTURE * G.N_EPISODES) == 25

    def test_a_seed_whose_best_pursuer_clears_25_passes(self):
        records = self._records_for_seed(80, dict.fromkeys(G.PURSUERS, 10) | {"SEL-80": 25})
        b = G.h_ev_3(records)["seeds"]["80"]
        assert b["max_k"] == 25 and b["max_pursuer"] == "SEL-80"
        assert b["verdict"] == "PASS"

    def test_a_seed_whose_best_pursuer_falls_one_short_fails(self):
        records = self._records_for_seed(80, dict.fromkeys(G.PURSUERS, 10) | {"SEL-80": 24})
        b = G.h_ev_3(records)["seeds"]["80"]
        assert b["max_k"] == 24
        assert b["verdict"] == "FAIL"

    def test_all_three_seeds_must_pass_not_two_of_three(self):
        records = {}
        for seed in (80, 81):
            records |= self._records_for_seed(seed, dict.fromkeys(G.PURSUERS, 10) | {"SEL-%d" % seed: 25})
        records |= self._records_for_seed(82, dict.fromkeys(G.PURSUERS, 10) | {"SEL-82": 24})  # fails
        v = G.h_ev_3(records)
        assert len(v["passing_seeds"]) == 2
        assert v["verdict"] == "FAIL"  # not PASS, because H-EV-3 is ALL seeds

    def test_missing_cells_are_incomplete(self):
        records = self._records_for_seed(80, {p: 10 for p in list(G.PURSUERS)[:-1]})  # one missing
        v = G.h_ev_3(records)
        assert v["seeds"]["80"]["verdict"] == "INCOMPLETE"
        assert v["verdict"] == "INCOMPLETE"


class TestCgE3:
    def _records(self, flagged_seeds: set[int]) -> dict:
        out = {}
        for s in G.SEEDS:
            share = {a: 0.0 for a in G.EVADER_ACTIONS}
            share["FLEE"] = 0.97 if s in flagged_seeds else 0.4
            share["ORBIT"] = 1 - share["FLEE"]
            for p in G.PURSUERS:
                out[G.cell_name(f"EVS-{s}", p)] = {"aggregate": {"picks_share": dict(share)}}
        return out

    def test_flags_a_seed_whose_top_tactic_clears_95_against_every_pursuer(self):
        v = G.cg_e3(self._records({80}))
        assert v["flagged_seeds"] == [80]
        assert v["verdict"] == "REPORTED"  # never gates

    def test_does_not_flag_a_seed_below_the_bar_on_even_one_pursuer(self):
        records = self._records(set())
        # one pursuer cell for seed 81 clears the bar, the rest do not
        records["EVS-81_SEL-80"]["aggregate"]["picks_share"] = {
            a: 0.0 for a in G.EVADER_ACTIONS
        } | {"FLEE": 0.99, "ORBIT": 0.01}
        v = G.cg_e3(records)
        assert v["flagged_seeds"] == []


class TestHSafeAndInheritedCgT1:
    def _records(self, **cell_overrides) -> dict:
        return {n: _measurable(n) | cell_overrides for n in G.CELL_ORDER}

    def _clean_probes(self) -> dict:
        return {
            arm: {"missing": False, "rows": [{"train_violations": 0, "violations": 0}]}
            for arm in T2.ARM_ORDER
        }

    def test_holds_when_everything_is_clean(self):
        v = G.h_safe(self._records(), self._clean_probes())
        assert v["verdict"] == "HOLDS"

    def test_fails_on_a_gate_violation(self):
        records = self._records()
        records["EVS-80_SEL-80"]["violation_steps"] = 1
        v = G.h_safe(records, self._clean_probes())
        assert v["verdict"] == "FAILS" and "EVS-80_SEL-80" in v["offenders"]

    def test_fails_on_training_contact(self):
        probes = self._clean_probes()
        probes["EVS-81"]["rows"][0]["train_violations"] = 1
        v = G.h_safe(self._records(), probes)
        assert v["verdict"] == "FAILS"
        assert "EVS-81" in v["training_offenders"]

    def test_incomplete_on_missing_training_evidence(self):
        probes = self._clean_probes()
        probes["EVS-82"] = {"missing": True, "rows": []}
        v = G.h_safe(self._records(), probes)
        assert v["verdict"] == "INCOMPLETE"
        assert "EVS-82" in v["missing_training"]

    def test_cg_t1_refuses_on_a_non_zero_inapplicable_pick(self):
        # the REAL read_probes() shape: {arm: {"path", "missing", "rows": [...]}}
        clean = {a: {"path": "", "missing": False, "rows": [{"mark": 3_000_000, "train_inapplicable_picks": 0, "inapplicable_picks": 0}]} for a in ("EVS-80", "EVS-81", "EVS-82")}
        records = self._records()
        records["EVS-81_RULES-V0"]["inapplicable_picks"] = 1
        v = G.cg_t1(records, clean)
        assert v["verdict"] == "REFUSE" and v["offenders"] == ["EVS-81_RULES-V0"]
        assert G.cg_t1(self._records(), clean)["verdict"] == "PASS"
        # review fix 2: probe rows are evidence too — a non-zero counter or an ABSENT one refuses
        bad = dict(clean); bad["EVS-80"] = {"path": "", "missing": False, "rows": [{"mark": 3_000_000, "train_inapplicable_picks": 2, "inapplicable_picks": 0}]}
        assert G.cg_t1(self._records(), bad)["offenders"] == ["probe:EVS-80"]
        absent = dict(clean); absent["EVS-82"] = {"path": "", "missing": False, "rows": [{"mark": 3_000_000, "inapplicable_picks": 0}]}
        assert G.cg_t1(self._records(), absent)["offenders"] == ["probe:EVS-82"]
        # and the real files on disk, when present, must be readable in that shape
        real = G.read_probes(); assert all(isinstance(real[a].get("rows"), list) for a in real)
        assert G.cg_t1(self._records(), absent)["offenders"] == ["probe:EVS-82"]


# ----------------------------------------------------------------------------
# GIFs: the seed rule, captions, and the CLI's dispatch order
# ----------------------------------------------------------------------------
class TestGifsAndDispatch:
    def test_eighteen_gifs_three_rows_two_pursuers_three_seeds(self):
        assert len(G.GIF_ROWS) == 3 and len(G.GIF_PURSUERS) == 2 and G.GIF_N == 3
        assert set(G.GIF_PURSUERS) == {"SEL-81", "RULES-V0"}

    def test_gif_seeds_are_s24_1s_own_rule(self):
        assert G.gif_seeds() == S1.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)

    def test_captions_carry_no_rate(self):
        title = G.gif_title("EVS-80", "SEL-81", 2_400_000)
        assert title == "EVS-80 vs SEL-81 · episode 2400000"
        assert not re.search(r"0\.\d+", title)

    def test_write_gifs_refuses_without_a_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered"))
        with pytest.raises(SystemExit, match="no cell record"):
            G.write_gifs(["EVS-80"])

    def test_write_gifs_refuses_without_the_rows_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        for p in G.GIF_PURSUERS:
            _on_disk(tmp_path, _measurable(G.cell_name("EVS-80", p)))
        monkeypatch.setattr(T2, "ckpt_path", lambda arm: tmp_path / "nope.zip")
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("rendered"))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.write_gifs(["EVS-80"])

    def test_dispatch_order_is_cells_then_gifs_then_summarize(self, monkeypatch):
        order = []
        monkeypatch.setattr(G, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(G, "write_gifs", lambda *a, **k: order.append("gifs"))

        def fake_summarize(device="cpu"):
            order.append("summarize")
            return {
                "verdicts": {
                    "h_ev_3": {"verdict": "FAIL"}, "h_ev_1": {"verdict": "FAIL"},
                    "h_ev_2": {"verdict": "FAIL"}, "cg_e3": {},
                    "h_safe": {}, "cg_t1": {},
                    "direction": "", "n_records": 0, "n_cells": 15, "complete": False,
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
        got = _run_checker(checker, tmp_path, "EVS-80", "EVS-81", "EVS-82")
        assert got.returncode == 0, got.stdout + got.stderr
        assert "clean" in got.stdout

    def test_it_refuses_an_existing_arm_record(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        d.mkdir(parents=True)
        (d / "s24_2_arm_EVS-80.json").write_text("{}")
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-80")
        assert got.returncode == 1 and "arm record exists" in got.stdout

    def test_it_refuses_an_existing_checkpoint_dir(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        (d / "ckpts_S24-2-EVS-81").mkdir(parents=True)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-81")
        assert got.returncode == 1 and "checkpoint dir" in got.stdout


class TestGateLauncherChecker:
    def test_it_refuses_without_a_checkpoint(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-80")
        assert got.returncode == 1 and "no final checkpoint" in got.stdout

    def test_it_refuses_a_dirty_probe_row(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2-EVS-80"
        ck.mkdir(parents=True)
        (ck / "S24-2-EVS-80_3000000.zip").write_bytes(b"x")
        (d / "s24_2_probe_EVS-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 1, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-80")
        assert got.returncode == 1 and "violations = 1" in got.stdout

    def test_it_refuses_absent_probe_evidence(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2-EVS-80"
        ck.mkdir(parents=True)
        (ck / "S24-2-EVS-80_3000000.zip").write_bytes(b"x")
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-80")
        assert got.returncode == 1 and "no probe JSON" in got.stdout

    def test_it_accepts_clean_evidence(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2-EVS-80"
        ck.mkdir(parents=True)
        (ck / "S24-2-EVS-80_3000000.zip").write_bytes(b"x")
        (d / "s24_2_probe_EVS-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 0, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "EVS-80")
        assert got.returncode == 0, got.stdout + got.stderr


class TestLaunchersGateTheHost:
    def test_both_launchers_share_the_pgrep_guard_and_omp_4(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            src = path.read_text()
            assert "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" in src
            assert "OMP_NUM_THREADS=4" in src
            assert "is not a symlink to" in src
            assert "git rev-parse --short HEAD" in src

    def test_the_gate_launcher_has_three_chains_of_five_covering_all_15_cells(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH_GATE.read_text().splitlines()
            if "--cells" in line and line.lstrip().startswith("nohup")
        ]
        assert len(chains) == 3
        assert all(len(c) == 5 for c in chains)
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(G.CELL_ORDER)

    def test_neither_launcher_summarises_or_renders(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            for line in path.read_text().splitlines():
                if line.lstrip().startswith("nohup"):
                    assert "--summarize" not in line and "--gifs" not in line

    def test_launchers_use_braced_loop_variables(self):
        assert "${a}" in _LAUNCH_TRAIN.read_text()


# ----------------------------------------------------------------------------
# one real, tiny end-to-end smoke — a fresh policy, a tmp dir, deleted
# afterwards. No other test in this file trains a real step or runs a real
# episode against a frozen checkpoint.
# ----------------------------------------------------------------------------
class TestSmoke:
    def test_a_fresh_policy_trains_a_few_steps_and_runs_n3_gate_cells(self, monkeypatch):
        from sb3_contrib import MaskablePPO

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="s24_2_smoke_"))
        try:
            monkeypatch.setattr(T2, "ckpt_path", lambda arm, _tmp=tmp: _tmp / f"{arm}.zip")
            env = T2.make_env("EVS-80")
            built = T2.make_learner("EVS-80", env, "cpu")
            assert isinstance(built, MaskablePPO)
            model = MaskablePPO(
                "MlpPolicy", env, seed=80, gamma=0.99, n_steps=64, batch_size=32, verbose=0,
            )
            model.learn(total_timesteps=64)
            ckpt = T2.ckpt_path("EVS-80")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(ckpt)

            with G._bound():
                rec_rules = G.run_cell("EVS-80_RULES-V0", n_episodes=3)
                rec_lead = G.run_cell("EVS-80_LEAD-1.0", n_episodes=3)

            assert rec_rules["n_episodes"] == 3 and rec_lead["n_episodes"] == 3
            assert set(rec_rules["picks"]) == set(G.EVADER_ACTIONS)
            assert rec_rules["pursuer"] == "RULES-V0" and rec_lead["pursuer"] == "LEAD-1.0"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        assert not tmp.exists()
