"""scripts/derisk_s24_3zs.py — the S24.3-ZS zero-shot speed sweep.

Pre-registration: `docs/pathway/tactics_prereg_s24_3zs_2026-10-13.md`.
Everything here is either a constant that text fixes, a verdict rule it states,
or one of the properties the round's claims rest on:

- the cell table really is 6 x 3 x 2, and CG-Z1 holds on the BUILT envs at both
  ratios: the ratio is the only config difference from the 1.2 twin, the
  selector's observation entry carries it, and the assembly is otherwise
  bit-for-bit S24.1's (asserted against S24.1's own 1.2 build);
- the 1.2 column is READ from S24.1's records and REFUSED whole if any joined
  record is absent, unreadable or not measurable by S24.1's own rule;
- record validity also covers the DECLARED RATIO — a 0.8 record filed as a 1.0
  cell is INVALID — and duplicates reject both files;
- `_bound` restores every S24.1 global it rebinds, and `_ORIG` really holds
  S24.1's own versions (so a delegating call cannot re-enter this module);
- every hypothesis is tested with its FAILING direction, not only its passing
  one: an inverted H-ZS-1, an inverted and a non-significant H-ZS-2 step, an
  inverted H-ZS-3 profile;
- the decision rule's three branches, on synthetic tables;
- CG-Z1 and the inherited CG-T1/A2.1 REFUSE the summary before any rate;
- CG-Z2: the watched seeds are the first three GATE seeds (a rule, not a pick),
  the pass refuses without a measurable record, no caption carries a measured
  number, and `--gifs` dispatches before `--summarize`;
- the launch wrapper refuses on its own preconditions, checked through the
  wrapper's OWN checker.

No cell is ever run here: nothing calls `run_cell`, and the only envs built are
CG-Z1's.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

from pe2.config import EnvConfig
from pe2.selector_mdp import ACTIONS, OBS_SLICES, SelectorMDP
from pe2.structured import HOLD

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s24_3zs.py"
_LAUNCH = _PATH.parent / "launch_s24_3zs.sh"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s24_3zs", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
S1 = R.S1
N = 500  # episodes in the synthetic records below — the round's own N
ALL = ("GREEDY", "HUG", "TACTIC-V0")
EVEN = dict.fromkeys(ACTIONS, 20)
CHASEY = dict.fromkeys(ACTIONS, 0) | {"INTERCEPT": 400, "LEAD": 400, "SWEEP": 200}
PATIENTY = dict.fromkeys(ACTIONS, 0) | {"INTERCEPT": 100, "LEAD": 100, "SWEEP": 800}


# ----------------------------------------------------------------------------
# synthetic records, in the shape `run_cell` writes
# ----------------------------------------------------------------------------
def _caps(rate: float, n: int = N) -> list[bool]:
    k = round(rate * n)
    return [i < k for i in range(n)]


def _cg_z1(name: str) -> dict:
    spec = R.CELLS[name]
    return {
        "gate": "CG-Z1",
        "cell": name,
        "ratio": spec["ratio"],
        "config_diff_vs_twin": ["evader_speed_ratio"],
        "cfg_ratio": spec["ratio"],
        "obs_ratio": spec["ratio"] if spec["kind"] == "learned" else None,
        "obs_checked": spec["kind"] == "learned",
        "verdict": "PASS",
    }


def _cell(name: str, rate: float, picks: dict | None = None, n: int = N) -> dict:
    """One record with only the fields the verdicts read. The per-episode array
    and the rate agree, so a verdict that recomputed one from the other would
    still say the same thing."""
    spec = R.CELLS.get(name)
    caps = _caps(rate, n)
    picks = dict(EVEN) if picks is None else picks
    base = {
        "name": name,
        "n_episodes": n,
        "seed0": R.GATE_SEED0,
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "wilson_capture": R._ci(sum(caps), n),
        "steps": [300] * n,
        "violations": [0] * n,
        "violation_steps": 0,
        "picks": {a: int(picks.get(a, 0)) for a in ACTIONS},
        "inapplicable_picks": 0,
        "search_policy_steps": 0,
    }
    if spec is None:  # a 1.2 cell: S24.1's record, re-keyed by the join
        return base | {"ratio": R.BASE_RATIO}
    return base | {
        "row": spec["row"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": spec["kind"],
        "ratio": spec["ratio"],
        "cg_z1": _cg_z1(name),
    }


def _rates(sel=0.60, lead=0.40, rules=0.45, nomem=0.50) -> dict:
    return {"SEL": sel, R.BASELINE_ROW: lead, R.RULES_ROW: rules, R.CONTROL_ROW: nomem}


def _round_records(
    per_ratio: dict | None = None, picks: dict | None = None
) -> tuple[dict, dict]:
    """`(records, base)` for the whole round. `per_ratio` maps a ratio to a
    `_rates()` dict; `picks` maps a ratio to the SEL pick histogram."""
    per_ratio = per_ratio or {0.8: _rates(0.70), 1.0: _rates(0.60), 1.2: _rates(0.50)}
    picks = picks or {0.8: CHASEY, 1.0: dict(EVEN), 1.2: PATIENTY}
    records: dict = {}
    base: dict = {}
    for ratio, rates in per_ratio.items():
        for row in R.ROW_NAMES:
            rate = rates["SEL"] if row.startswith("SEL-") else rates[row]
            for o in ALL:
                name = R.cell_name(row, o, ratio)
                p = picks[ratio] if row.startswith("SEL-") else dict(EVEN)
                rec = _cell(name, rate, p)
                (base if ratio == R.BASE_RATIO else records)[name] = rec
    return records, base


def _on_disk(out: pathlib.Path, rec: dict, name: str | None = None) -> pathlib.Path:
    d = out / "s24_3zs_cells"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name or rec['name']}.json"
    path.write_text(json.dumps(rec))
    return path


def _measurable(name: str) -> dict:
    return _cell(name, 0.5, n=R.N_EPISODES)


# ----------------------------------------------------------------------------
# the cell table and the pre-registered constants
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_thirtysix_cells_six_rows_three_opponents_two_ratios(self):
        assert len(R.CELLS) == 36 == 6 * 3 * 2
        assert R.ROW_NAMES == (
            "SEL-80", "SEL-81", "SEL-82", "NOMEM-81", "LEAD-1.0", "RULES-V0",
        )
        assert list(R.OPPONENTS) == ["GREEDY", "HUG", "TACTIC-V0"]
        assert R.RATIOS == (0.8, 1.0) and R.BASE_RATIO == 1.2
        assert R.ALL_RATIOS == (0.8, 1.0, 1.2)
        assert R.SEL_ROWS == ("SEL-80", "SEL-81", "SEL-82")
        assert R.LEARNED_ROWS == ("SEL-80", "SEL-81", "SEL-82", "NOMEM-81")
        assert set(R.CELLS) == {
            f"{r}_{o}_r{x:.1f}" for r in R.ROW_NAMES for o in ALL for x in R.RATIOS
        }
        # the row set is a SUBSET of S24.1's, with S24.1's own specs
        assert set(R.ROW_NAMES) <= set(S1.ROWS)
        for row in R.ROW_NAMES:
            assert R.ROWS[row] == S1.ROWS[row]

    def test_the_cell_carries_its_ratio_rather_than_encoding_it_in_the_name(self):
        spec = R.CELLS["LEAD-1.0_HUG_r1.0"]
        assert (spec["row"], spec["opponent"], spec["ratio"]) == ("LEAD-1.0", "HUG", 1.0)
        assert spec["hold"] == HOLD == 10
        # `LEAD-1.0` and `r1.0` both contain a separator: nothing may parse back
        assert R.cell_name("LEAD-1.0", "HUG", 1.0) == "LEAD-1.0_HUG_r1.0"

    def test_pre_registered_constants(self):
        assert R.N_EPISODES == 500
        assert R.GATE_SEED0 == 2_400_000 == S1.GATE_SEED0
        assert (R.P_GATE, R.P_CHI2, R.PASS_MIN) == (0.05, 0.01, 2)
        assert R.TIE == 0.02
        assert R.SEEDS == (80, 81, 82)
        assert R.BALANCED_TARGET == 0.5
        assert R.D150_SCRIPTED == {0.8: 1.00, 1.0: 0.68, 1.2: 0.68}
        assert R.CHASE == ("INTERCEPT", "LEAD")
        assert set(R.CHASE) | set(R.PATIENT) == set(ACTIONS)  # they partition
        assert R.OMP_THREADS == "4"
        assert R.ARTEFACT_PREFIX == "s24_3zs"
        assert R.SCRIPTED_CHECK_OPPONENT == "TACTIC-V0"

    def test_the_artefact_paths_are_this_rounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.cell_file("SEL-80_HUG_r0.8") == tmp_path / "s24_3zs_cells" / "SEL-80_HUG_r0.8.json"
        assert R.round_file("json") == tmp_path / "s24_3zs_round.json"
        assert R.gifs_dir() == tmp_path / "s24_3zs_gifs"
        # ... and the checkpoint path is still Step 1.1's layout, imported
        assert R.ckpt_path is S1.ckpt_path
        assert R.ckpt_path("SEL-81") == pathlib.Path(
            "results/derisk/ckpts_S24-1-SEL-81/S24-1-SEL-81_3000000.zip"
        )

    def test_the_prereg_exists_and_provenance_hashes_it(self):
        assert R.spec_path().exists()
        p = R.provenance("cuda")
        assert p["spec"] == "docs/pathway/tactics_prereg_s24_3zs_2026-10-13.md"
        assert p["spec_sha256"] == S1._sha256(R.spec_path())
        assert p["ratios"] == [0.8, 1.0] and p["base_ratio"] == 1.2
        assert p["gate_stream"] == 2_400_000
        assert p["joined_from"] == S1.PREREG
        assert p["device"] == "cuda"


# ----------------------------------------------------------------------------
# CG-Z1 on the BUILT envs (prereg section 5)
# ----------------------------------------------------------------------------
class TestCGZ1:
    def test_the_config_diff_is_the_ratio_and_nothing_else(self):
        assert R.config_diff(EnvConfig()) == []
        assert R.config_diff(EnvConfig(evader_speed_ratio=0.8)) == ["evader_speed_ratio"]
        assert R.config_diff(EnvConfig(evader_speed_ratio=0.8, max_steps=99)) == [
            "evader_speed_ratio",
            "max_steps",
        ]
        for x in R.RATIOS:
            cfg = R.cell_config(x)
            assert cfg.evader_speed_ratio == x
            assert R.config_diff(cfg) == ["evader_speed_ratio"]

    @pytest.mark.parametrize("ratio", [0.8, 1.0])
    def test_a_built_cell_differs_from_its_1_2_twin_only_in_the_ratio(self, ratio):
        """The round's whole attribution: build one learned and one scripted row
        at this ratio and assert against S24.1's OWN 1.2 build."""
        for row in ("SEL-81", "RULES-V0"):
            twin = S1.build_cell_env(S1.cell_name(row, "TACTIC-V0"))
            got = R.build_cell_env(R.cell_name(row, "TACTIC-V0", ratio))
            a = twin.game if S1.ROWS[row]["kind"] == "learned" else twin
            b = got.game if R.ROWS[row]["kind"] == "learned" else got
            assert type(b) is type(a)
            assert b.cfg.evader_speed_ratio == ratio
            assert a.cfg.evader_speed_ratio == R.BASE_RATIO
            assert R.config_diff(b.cfg) == ["evader_speed_ratio"]
            assert b.hold == a.hold == HOLD
            assert (b.shield_obs, b.escape_obs) == (a.shield_obs, a.escape_obs)
            assert (b.stall_penalty, b.intervention_penalty) == (0.0, 0.0)
            assert b._driver is b.inner  # reward_wrapper=None, as the twin
            assert type(b.shield).__name__ == type(a.shield).__name__ == "RecoveryShieldV2"
            assert (b.shield.margin, b.shield.gamma) == (a.shield.margin, a.shield.gamma)
            assert [t.name for t in b.law.tactics] == [t.name for t in a.law.tactics]
            assert [t.limit for t in b.law.tactics] == [t.limit for t in a.law.tactics]

    @pytest.mark.parametrize("ratio", [0.8, 1.0])
    def test_the_observation_entry_carries_the_out_of_range_ratio(self, ratio):
        name = R.cell_name("SEL-80", "GREEDY", ratio)
        env = R.build_cell_env(name)
        assert isinstance(env, SelectorMDP)
        obs, _info = env.reset(seed=R.GATE_SEED0)
        # float32: 0.8 is not exactly representable, so compare at the
        # observation's own precision (this is what `_f32_eq` exists for)
        assert R._f32_eq(float(obs[OBS_SLICES["speed_ratio"]][0]), ratio)
        assert not R._f32_eq(float(obs[OBS_SLICES["speed_ratio"]][0]), 1.2)
        block = R.cg_z1(name)
        assert block["verdict"] == "PASS"
        assert R._f32_eq(block["obs_ratio"], ratio) and block["cfg_ratio"] == ratio
        assert block["obs_checked"] is True
        assert block["config_diff_vs_twin"] == ["evader_speed_ratio"]
        assert block["twin_ratio"] == 1.2

    def test_a_scripted_row_has_no_observation_entry_and_says_so(self):
        block = R.cg_z1("RULES-V0_HUG_r0.8")
        assert block["verdict"] == "PASS"
        assert block["obs_checked"] is False and block["obs_ratio"] is None
        assert block["cfg_ratio"] == 0.8

    def test_a_built_env_at_the_wrong_ratio_refuses_before_any_episode(self, monkeypatch):
        import types

        monkeypatch.setattr(
            R,
            "build_cell_env",
            lambda *a, **k: types.SimpleNamespace(cfg=EnvConfig(evader_speed_ratio=1.2)),
        )
        with pytest.raises(SystemExit, match="CG-Z1 REFUSES"):
            R.cg_z1("RULES-V0_HUG_r0.8")

    def test_a_wrong_observation_entry_refuses_even_when_the_config_is_right(
        self, monkeypatch
    ):
        import types

        import numpy as np

        obs = np.zeros(64, dtype=np.float32)
        obs[OBS_SLICES["speed_ratio"]] = 1.2  # the twin's value, not this cell's
        monkeypatch.setattr(
            R,
            "build_cell_env",
            lambda *a, **k: types.SimpleNamespace(
                game=types.SimpleNamespace(cfg=EnvConfig(evader_speed_ratio=0.8)),
                reset=lambda **kw: (obs, {}),
            ),
        )
        with pytest.raises(SystemExit, match="observation's ratio entry"):
            R.cg_z1("SEL-80_HUG_r0.8")

    def test_run_cells_sweeps_the_gate_before_the_first_episode(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(R, "cg_z1", lambda n, d="cpu": seen.append(n))
        monkeypatch.setitem(
            R._ORIG, "run_cells", lambda *a, **k: seen.append("CHAIN") or []
        )
        R.run_cells(["RULES-V0_HUG_r0.8", "SEL-80_HUG_r1.0"])
        assert seen == ["RULES-V0_HUG_r0.8", "SEL-80_HUG_r1.0", "CHAIN"]

    def test_an_unknown_cell_name_refuses(self):
        with pytest.raises(SystemExit, match="unknown cell"):
            R.run_cells(["SEL-80_HUG"])  # S24.1's name, not this round's


# ----------------------------------------------------------------------------
# the 1.2 join (prereg section 3, reading 2)
# ----------------------------------------------------------------------------
def _base_on_disk(out: pathlib.Path, rows=None, **overrides) -> None:
    d = out / "s24_1_cells"
    d.mkdir(parents=True, exist_ok=True)
    for row in R.ROW_NAMES if rows is None else rows:
        for o in ALL:
            rec = {
                "name": f"{row}_{o}",
                "n_episodes": S1.N_EPISODES,
                "seed0": S1.GATE_SEED0,
                "capture_rate": 0.5,
                "capture_ind": _caps(0.5, S1.N_EPISODES),
                "picks": dict(EVEN),
            } | overrides
            (d / f"{row}_{o}.json").write_text(json.dumps(rec))


class TestBaseJoin:
    def test_it_reads_s24_1s_real_records_and_rekeys_them(self):
        base = R.load_base_records()
        assert len(base) == 18 == len(R.ROW_NAMES) * len(ALL)
        assert set(base) == {R.cell_name(r, o, 1.2) for r in R.ROW_NAMES for o in ALL}
        for name, rec in base.items():
            assert rec["ratio"] == 1.2
            assert rec["n_episodes"] == 500 and rec["seed0"] == 2_400_000
            assert rec["name"] != name  # S24.1's own name is kept inside
            assert S1.record_status(rec)[0] == "MEASURABLE"

    def test_the_join_uses_s24_1s_own_validity_rule(self):
        assert R._ORIG["record_status"] is S1.record_status
        assert R._ORIG["cell_file"] is S1.cell_file
        assert R.s24_1_cell_file("SEL-80", "HUG").name == "SEL-80_HUG.json"

    def test_it_refuses_an_absent_record_rather_than_joining_a_partial_column(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        _base_on_disk(tmp_path, rows=R.ROW_NAMES[:-1])
        with pytest.raises(SystemExit, match="absent"):
            R.load_base_records()

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 2_350_000}, "seed0"),
            ({"verdict": "UNDETERMINED", "n_episodes": 0}, "UNDETERMINED"),
        ],
    )
    def test_it_refuses_a_non_measurable_record(
        self, tmp_path, monkeypatch, overrides, match
    ):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        _base_on_disk(tmp_path)
        d = tmp_path / "s24_1_cells"
        rec = json.loads((d / "SEL-82_HUG.json").read_text()) | overrides
        (d / "SEL-82_HUG.json").write_text(json.dumps(rec))
        with pytest.raises(SystemExit, match=match):
            R.load_base_records()

    def test_it_refuses_an_unreadable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        _base_on_disk(tmp_path)
        (tmp_path / "s24_1_cells" / "SEL-80_GREEDY.json").write_text("{not json")
        with pytest.raises(SystemExit, match="unreadable"):
            R.load_base_records()

    def test_it_refuses_to_run_inside_a_binding(self):
        """Inside `_bound` the validity rule and the cells directory would be
        THIS round's, so the column would be checked against the wrong rule and
        read from the wrong place."""
        with pytest.raises(SystemExit, match="may not run inside a binding"), R._bound():
            R.load_base_records()
        with pytest.raises(SystemExit, match="may not run inside a binding"), R._one(
            "SEL-80_HUG_r0.8"
        ):
            R.load_base_records()
        assert len(R.load_base_records()) == 18  # ... and outside one it works

    def test_nothing_in_the_join_writes_to_s24_1s_directory(self):
        d = pathlib.Path("results/derisk/s24_1_cells")
        before = {p: p.stat().st_mtime_ns for p in sorted(d.glob("*.json"))}
        R.load_base_records()
        assert {p: p.stat().st_mtime_ns for p in sorted(d.glob("*.json"))} == before


# ----------------------------------------------------------------------------
# record validity (section 6, with the DECLARED RATIO added)
# ----------------------------------------------------------------------------
class TestRecordValidity:
    def test_a_measurable_record(self):
        assert R.record_status(_measurable("SEL-80_HUG_r0.8")) == ("MEASURABLE", "")

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 2_350_000}, "seed0"),
            ({"ratio": 1.0}, "ratio"),  # a 0.8 record filed as its own cell
            ({"ratio": None}, "ratio"),
        ],
    )
    def test_the_wrong_n_stream_or_ratio_is_invalid(self, overrides, match):
        status, why = R.record_status(_measurable("SEL-80_HUG_r0.8") | overrides)
        assert status == "INVALID" and match in why

    def test_a_record_of_another_round_is_invalid(self):
        status, why = R.record_status({"name": "SEL-80_HUG"})
        assert status == "INVALID" and "not a cell of this round" in why

    def test_a_crash_through_run_cells_writes_this_rounds_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        """The crash path is S24.1's `run_cells`, which reaches for the
        `undetermined_record` global — so that global has to be bound to THIS
        round's wrapper, or the file on disk would carry no `ratio` and
        `record_status` would call the crash INVALID instead of UNDETERMINED."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "cg_z1", lambda n, d="cpu": _cg_z1(n))

        def boom(name, device="cpu", n_episodes=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(R, "run_cell", boom)
        assert "undetermined_record" in R._BOUND_ALWAYS
        R.run_cells(["RULES-V0_HUG_r0.8"])
        rec = json.loads(R.cell_file("RULES-V0_HUG_r0.8").read_text())
        assert rec["verdict"] == R.UNDETERMINED == "UNDETERMINED"
        assert rec["ratio"] == 0.8 and rec["n_episodes"] == 0
        assert "RuntimeError: boom" in rec["error"]
        assert "NEVER re-run" in rec["note"]
        assert rec["provenance"]["spec"] == R.PREREG  # this round's, not S24.1's
        assert R.record_status(rec) == ("UNDETERMINED", f"crashed cell: {rec['error']!r}")
        # ... and the crashed cell is never re-run or read as a measurement
        assert R.load_records(quiet=True) == {}
        with pytest.raises(SystemExit, match="UNDETERMINED"):
            R.read_cell("RULES-V0_HUG_r0.8")

    def test_load_records_drops_everything_that_is_not_measurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        good = _measurable("SEL-80_HUG_r0.8")
        _on_disk(tmp_path, good)
        _on_disk(tmp_path, _measurable("SEL-80_GREEDY_r0.8") | {"n_episodes": 20})
        _on_disk(tmp_path, _measurable("SEL-81_HUG_r1.0") | {"ratio": 0.8})
        assert set(R.load_records(quiet=True)) == {"SEL-80_HUG_r0.8"}

    def test_a_duplicated_cell_rejects_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable("RULES-V0_HUG_r1.0")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a_copy")
        measurable, rejected, by_cell = R.scan_records()
        assert measurable == {}
        assert len(rejected) == 2 and by_cell["RULES-V0_HUG_r1.0"]
        assert all("duplicate" in why for why in rejected.values())

    def test_read_cell_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no cell record"):
            R.read_cell("SEL-81_GREEDY_r0.8")
        _on_disk(tmp_path, _measurable("SEL-81_GREEDY_r0.8") | {"ratio": 1.0})
        with pytest.raises(SystemExit, match="not measurable"):
            R.read_cell("SEL-81_GREEDY_r0.8")


# ----------------------------------------------------------------------------
# the binding of S24.1's machinery (the module docstring's one unusual thing)
# ----------------------------------------------------------------------------
class TestBinding:
    def test_every_bound_name_is_restored(self):
        before = {k: getattr(S1, k) for k in R._BOUND_KEYS}
        with R._bound(0.8):
            inside = {k: getattr(S1, k) for k in R._BOUND_KEYS}
            assert all(inside[k] is not before[k] for k in R._BOUND_KEYS)
        assert {k: getattr(S1, k) for k in R._BOUND_KEYS} == before

    def test_it_restores_even_when_the_call_raises(self):
        before = {k: getattr(S1, k) for k in R._BOUND_KEYS}
        with pytest.raises(ValueError, match="x"), R._bound(1.0):
            raise ValueError("x")
        assert {k: getattr(S1, k) for k in R._BOUND_KEYS} == before

    def test_nesting_restores_the_outer_binding(self):
        before = S1.CELLS
        with R._bound():
            outer = S1.CELLS
            assert len(outer) == 36
            with R._one("SEL-80_HUG_r0.8"):
                assert list(S1.CELLS) == ["SEL-80_HUG_r0.8"]
                assert S1.cell_config().evader_speed_ratio == 0.8
            assert S1.CELLS is outer
        assert S1.CELLS is before

    def test_orig_holds_s24_1s_own_versions_so_a_delegate_cannot_recurse(self):
        assert R._ORIG["run_cell"] is not R.run_cell
        assert R._ORIG["cell_name"]("SEL-80", "GREEDY") == "SEL-80_GREEDY"
        assert R._ORIG["cell_config"]().evader_speed_ratio == 1.2
        assert R._ORIG["provenance"] is not R.provenance
        assert set(R._BOUND_KEYS) <= set(R._ORIG)
        for key in R._BOUND_KEYS:
            assert hasattr(S1, key), key

    def test_the_per_ratio_names_are_bound_only_when_a_ratio_is_given(self):
        before = S1.cell_name
        with R._bound():
            assert S1.cell_name is before  # no ratio: the 1.2 namer stands
        with R._bound(1.0):
            assert S1.cell_name("SEL-80", "HUG") == "SEL-80_HUG_r1.0"
            assert S1.gif_title("SEL-81", "GREEDY", 7) == R.gif_title("SEL-81", "GREEDY", 1.0, 7)

    def test_s24_1s_helpers_are_imported_not_reimplemented(self):
        assert R._write_json is S1._write_json
        assert R._ci is S1._ci
        assert R.load_policy is S1.load_policy
        assert R.require_ckpt is S1.require_ckpt
        assert R.chi2_table is S1.chi2_table
        assert R.gif_seeds is S1.gif_seeds
        assert R.GAME is sys.modules["s24_tactic_game"]
        assert R.SEEDS is S1.SEEDS and R.OPPONENTS is S1.OPPONENTS


# ----------------------------------------------------------------------------
# H-ZS-1 (section 4) — with its failing directions
# ----------------------------------------------------------------------------
class TestHZs1:
    def test_it_passes_at_both_ratios_when_every_seed_beats_the_law(self):
        records, _base = _round_records()
        v = R.h_zs_1(records)
        assert v["verdict"] == "PASS" and v["failing_ratios"] == []
        assert list(v["ratios"]) == ["r0.8", "r1.0"]
        for block in v["ratios"].values():
            assert block["verdict"] == "PASS" and block["n_pass"] == 3
            for seed in R.SEEDS:
                b = block["seeds"][str(seed)]
                assert b["comparator"] == "LEAD-1.0" and b["n_pass"] == 3
                assert all(
                    x["direction"] == f"SEL-{seed} > LEAD-1.0"
                    for x in b["opponents"].values()
                )

    def test_an_inverted_direction_is_a_fail_and_the_direction_is_reported(self):
        records, _base = _round_records(
            {0.8: _rates(sel=0.20, lead=0.70), 1.0: _rates(0.60, 0.40)}
        )
        v = R.h_zs_1(records)
        assert v["verdict"] == "FAIL" and v["failing_ratios"] == ["r0.8"]
        b = v["ratios"]["r0.8"]["seeds"]["80"]
        assert b["verdict"] == "FAIL" and b["n_pass"] == 0
        for o, x in b["opponents"].items():
            assert x["direction"] == "SEL-80 < LEAD-1.0", o
            assert x["delta"] < 0 and x["p"] < R.P_GATE  # significant, wrong way
        assert v["ratios"]["r1.0"]["verdict"] == "PASS"

    def test_the_round_needs_both_ratios(self):
        records, _base = _round_records(
            {0.8: _rates(0.70, 0.40), 1.0: _rates(0.41, 0.40)}  # 1.0: n.s.
        )
        v = R.h_zs_1(records)
        assert v["ratios"]["r0.8"]["verdict"] == "PASS"
        assert v["ratios"]["r1.0"]["verdict"] == "FAIL"
        assert v["verdict"] == "FAIL" and v["failing_ratios"] == ["r1.0"]
        assert v["rule"] == "PASS at BOTH ratios (section 4)"

    def test_a_missing_cell_is_incomplete_not_a_fail(self):
        records, _base = _round_records()
        del records[R.cell_name("SEL-81", "HUG", 1.0)]
        v = R.h_zs_1(records)
        assert v["verdict"] == "INCOMPLETE"
        assert v["ratios"]["r1.0"]["seeds"]["81"]["verdict"] == "INCOMPLETE"

    def test_two_of_three_opponents_and_two_of_three_seeds_is_the_bar(self):
        records, _base = _round_records()
        # break one opponent for one seed: still PASS (2/3 opponents, 3/3 seeds)
        name = R.cell_name("SEL-80", "HUG", 0.8)
        records[name] = _cell(name, 0.10, CHASEY)
        assert R.h_zs_1(records)["ratios"]["r0.8"]["verdict"] == "PASS"
        # break two opponents for two seeds: the ratio FAILS
        for seed in (80, 81):
            for o in ("HUG", "GREEDY"):
                n = R.cell_name(f"SEL-{seed}", o, 0.8)
                records[n] = _cell(n, 0.10, CHASEY)
        assert R.h_zs_1(records)["ratios"]["r0.8"]["verdict"] == "FAIL"


# ----------------------------------------------------------------------------
# H-ZS-2 (section 4) — the ordering, unpaired
# ----------------------------------------------------------------------------
class TestHZs2:
    def test_a_monotone_significant_ordering_passes(self):
        records, base = _round_records()
        v = R.h_zs_2(records, base)
        assert v["verdict"] == "PASS" and v["n_violations"] == 0
        cell = v["cells"]["SEL-80_GREEDY"]
        assert cell["captures"] == {"r0.8": 0.70, "r1.0": 0.60, "r1.2": 0.50}
        assert [s["verdict"] for s in cell["steps"]] == ["PASS", "PASS"]
        assert all("unpaired" in s["test"] for s in cell["steps"])

    def test_a_tie_within_the_tolerance_passes_either_way(self):
        for a, b in ((0.50, 0.49), (0.49, 0.50)):
            step = R._step(_cell("x", a), _cell("y", b), 0.8, 1.0)
            assert step["verdict"] == "PASS" and step["tie_within"] is True
            assert abs(step["delta"]) <= R.TIE

    def test_an_inverted_ordering_is_a_violation_and_names_its_direction(self):
        records, base = _round_records(
            {0.8: _rates(0.30), 1.0: _rates(0.70), 1.2: _rates(0.50)}
        )
        v = R.h_zs_2(records, base)
        assert v["verdict"] == "FAIL" and v["n_violations"] > 0
        bad = v["violations"][0]
        assert bad["direction"] == "r0.8 < r1.0" and bad["delta"] < -R.TIE
        assert bad["significant_drop"] is False
        # only the SEL rows moved in this fixture; the comparators are flat
        assert {b["row"] for b in v["violations"]} == set(R.SEL_ROWS)

    def test_a_large_but_non_significant_drop_is_a_violation(self):
        """Prereg: each adjacent drop SIGNIFICANT or within 0.02. A 0.10 drop on
        12 episodes is neither."""
        step = R._step(_cell("x", 0.50, n=12), _cell("y", 0.40, n=12), 0.8, 1.0)
        assert step["delta"] > R.TIE
        assert step["significant_drop"] is False and step["tie_within"] is False
        assert step["verdict"] == "FAIL"

    def test_the_unpaired_test_is_newcombes_and_not_disjoint_intervals(self):
        """The runner's own justification, made checkable: at N = 500 a 0.600 vs
        0.520 difference is resolved (a two-proportion test puts it at p = 0.01)
        while the two Wilson intervals still overlap, so disjointness would have
        manufactured an H-ZS-2 violation out of a correctly ordered step."""
        a, b = _cell("x", 0.60), _cell("y", 0.52)
        lo_a, _hi_a = R._wilson(a)
        _lo_b, hi_b = R._wilson(b)
        assert lo_a < hi_b  # the intervals OVERLAP
        step = R._step(a, b, 0.8, 1.0)
        assert step["significant_drop"] is True and step["verdict"] == "PASS"
        assert step["delta_ci"][0] > 0
        assert "Newcombe" in step["test"]

    def test_a_missing_1_2_record_is_incomplete(self):
        records, base = _round_records()
        del base[R.cell_name("RULES-V0", "HUG", 1.2)]
        v = R.h_zs_2(records, base)
        assert v["verdict"] == "INCOMPLETE"
        assert v["cells"]["RULES-V0_HUG"]["missing"] == ["RULES-V0_HUG_r1.2"]

    def test_every_row_and_opponent_is_covered(self):
        records, base = _round_records()
        v = R.h_zs_2(records, base)
        assert set(v["cells"]) == {f"{r}_{o}" for r in R.ROW_NAMES for o in ALL}
        assert len(v["cells"]) == 18


# ----------------------------------------------------------------------------
# H-ZS-3 (section 4) — the profile shift, labelled as section 4 words it
# ----------------------------------------------------------------------------
class TestHZs3:
    def test_a_chase_shift_towards_the_slower_evader_is_a_suggestive_pass(self):
        records, base = _round_records()
        v = R.h_zs_3(records, base)
        assert v["verdict"] == "SUGGESTIVE-PASS"
        assert v["passing_rows"] == list(R.SEL_ROWS)
        row = v["rows"]["SEL-81"]
        assert row["verdict"] == "PASS"
        assert row["chi2"]["p"] < R.P_CHI2 and row["chi2"]["verdict"] == "OK"
        assert row["chase_higher_at_0.8"] is True
        assert row["patient_lower_at_0.8"] is True
        assert row["shares"]["r0.8"]["chase"] > row["shares"]["r1.2"]["chase"]
        assert row["pooled_over"] == list(ALL)

    def test_the_reversed_direction_fails_and_is_labelled_per_ratio_training(self):
        records, base = _round_records(picks={0.8: PATIENTY, 1.0: dict(EVEN), 1.2: CHASEY})
        v = R.h_zs_3(records, base)
        assert v["verdict"] == "FAIL→PER-RATIO-TRAINING"
        assert v["passing_rows"] == []
        row = v["rows"]["SEL-80"]
        assert row["chi2"]["p"] < R.P_CHI2  # the profile DID move, the wrong way
        assert row["chase_higher_at_0.8"] is False
        assert row["verdict"] == "FAIL"

    def test_an_unmoved_profile_fails_on_the_chi_square(self):
        records, base = _round_records(
            picks={x: dict(EVEN) for x in R.ALL_RATIOS}
        )
        v = R.h_zs_3(records, base)
        assert v["verdict"] == "FAIL→PER-RATIO-TRAINING"
        assert v["rows"]["SEL-82"]["chi2"]["differs"] is False

    def test_one_failing_seed_still_clears_the_two_of_three_bar(self):
        records, base = _round_records()
        for o in ALL:  # SEL-82 moves the wrong way at 0.8
            records[R.cell_name("SEL-82", o, 0.8)] = _cell(
                R.cell_name("SEL-82", o, 0.8), 0.70, PATIENTY
            )
        v = R.h_zs_3(records, base)
        assert v["verdict"] == "SUGGESTIVE-PASS" and v["n_pass"] == 2
        assert v["rows"]["SEL-82"]["verdict"] == "FAIL"

    def test_the_control_is_reported_and_gates_nothing(self):
        records, base = _round_records()
        v = R.h_zs_3(records, base)
        assert v["control"]["row"] == R.CONTROL_ROW == "NOMEM-81"
        assert R.CONTROL_ROW not in v["rows"]  # reading 5: the SEL seeds only
        assert v["control"]["verdict"] == "FAIL"  # EVEN picks: no shift
        assert v["verdict"] == "SUGGESTIVE-PASS"
        assert "OUT OF ITS TRAINING RANGE" in v["weak_by_construction"]

    def test_a_missing_ratio_is_incomplete(self):
        records, base = _round_records()
        del records[R.cell_name("SEL-80", "GREEDY", 0.8)]
        assert R.h_zs_3(records, base)["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------
# H-SAFE and the construct gates
# ----------------------------------------------------------------------------
class TestHSafeAndGates:
    def test_h_safe_holds_on_a_complete_round_with_no_contact(self):
        records, _base = _round_records()
        v = R.h_safe(records)
        assert v["verdict"] == "HOLDS" and v["violation_steps"] == 0
        assert len(v["gated_cells"]) == 36 and v["missing"] == []
        assert "D151" in v["training"]

    def test_one_contact_step_anywhere_fails_the_round(self):
        records, _base = _round_records()
        name = R.cell_name("SEL-82", "HUG", 1.0)
        records[name] = records[name] | {"violation_steps": 1}
        v = R.h_safe(records)
        assert v["verdict"] == "FAILS" and v["offenders"] == [name]

    def test_an_incomplete_round_is_incomplete_not_holds(self):
        records, _base = _round_records()
        del records[R.cell_name("SEL-80", "HUG", 0.8)]
        assert R.h_safe(records)["verdict"] == "INCOMPLETE"

    def test_cg_z1_round_passes_on_records_that_carry_their_gate(self):
        records, base = _round_records()
        g = R.cg_z1_round(records, base)
        assert g["verdict"] == "PASS" and g["n_checked"] == 36
        assert g["base_column"]["n_records"] == 18
        assert "refuses to run under any ratio but 1.2" in g["base_column"]["note"]

    @pytest.mark.parametrize(
        "mutate",
        [
            {"cg_z1": None},
            {"cg_z1": {"verdict": "PASS", "cfg_ratio": 1.2, "obs_checked": False,
                       "config_diff_vs_twin": ["evader_speed_ratio"]}},
            {"cg_z1": {"verdict": "PASS", "cfg_ratio": 0.8, "obs_checked": True,
                       "obs_ratio": 1.2, "config_diff_vs_twin": ["evader_speed_ratio"]}},
            {"cg_z1": {"verdict": "PASS", "cfg_ratio": 0.8, "obs_checked": False,
                       "config_diff_vs_twin": ["evader_speed_ratio", "max_steps"]}},
            {"ratio": 1.0},
        ],
    )
    def test_cg_z1_round_refuses_a_cell_whose_gate_did_not_hold(self, mutate):
        records, base = _round_records()
        name = R.cell_name("SEL-80", "HUG", 0.8)
        records[name] = records[name] | mutate
        g = R.cg_z1_round(records, base)
        assert g["verdict"] == "REFUSE" and name in g["offenders"]

    def test_the_two_inherited_gates_are_separate_sub_blocks(self):
        records, _base = _round_records()
        g = R.cg_inherited(records)
        assert g["verdict"] == "PASS" and g["n_learned_cells"] == 24
        assert g["cg_t1"]["verdict"] == g["cg_a2_1"]["verdict"] == "PASS"
        assert set(g["cg_t1"]["inapplicable_picks"]) == {
            n for n in R.CELL_ORDER if R.CELLS[n]["kind"] == "learned"
        }
        # the per-cell half ONLY: nothing trains here, so there are no probes
        assert "not read" in g["cg_t1"]["training_probes"]
        assert "per-cell half only" in g["cg_t1"]["gate"]

    def test_the_mask_refuses_on_its_own_sub_block(self):
        records, _base = _round_records()
        name = R.cell_name("SEL-81", "GREEDY", 1.0)
        bad = R.cg_inherited(
            dict(records, **{name: records[name] | {"inapplicable_picks": 1}})
        )
        assert bad["verdict"] == "REFUSE" and bad["offenders"] == [name]
        assert bad["cg_t1"]["verdict"] == "REFUSE" and bad["cg_t1"]["offenders"] == [name]
        assert bad["cg_a2_1"]["verdict"] == "PASS"  # the other gate is untouched

    def test_a5s_tolerance_is_inherited_and_every_non_zero_is_named(self):
        records, _base = _round_records()
        name = R.cell_name("SEL-80", "HUG", 0.8)
        rec = records[name] | {"search_policy_steps": 1, "steps": [740] * 500}
        one = R.cg_inherited(dict(records, **{name: rec}))
        assert one["verdict"] == "PASS"
        assert one["cg_a2_1"]["verdict"] == "PASS"
        assert one["cg_a2_1"]["non_zero"] == [name]
        assert one["cg_a2_1"]["tolerance"][name] == 4
        many = R.cg_inherited(dict(records, **{name: rec | {"search_policy_steps": 5000}}))
        assert many["verdict"] == "REFUSE" and many["offenders"] == [name]
        assert many["cg_a2_1"]["verdict"] == "REFUSE"
        assert many["cg_t1"]["verdict"] == "PASS"  # the mask still holds

    def test_a_scripted_row_is_outside_the_inherited_guards(self):
        """Reading 5 of S24.1: the comparator rows reach the search policy
        whenever their law is DONE on entry, which is the comparator working."""
        records, _base = _round_records()
        name = R.cell_name("LEAD-1.0", "HUG", 0.8)
        records[name] = records[name] | {"search_policy_steps": 40_000}
        assert R.cg_inherited(records)["verdict"] == "PASS"


# ----------------------------------------------------------------------------
# the scripted check and the decision rule (section 4)
# ----------------------------------------------------------------------------
class TestScriptedCheckAndDecision:
    def _records(self, scripted: dict, **kw):
        per = {
            x: _rates(rules=scripted[x], **kw.get(x, {}))
            for x in R.ALL_RATIOS
        }
        for x in R.ALL_RATIOS:
            per[x]["SEL"] = 0.70 - 0.10 * R.ALL_RATIOS.index(x)
            per[x][R.BASELINE_ROW] = 0.40 - 0.10 * R.ALL_RATIOS.index(x)
        return _round_records(per)

    def test_it_reproduces_d150s_direction(self):
        records, base = self._records({0.8: 0.98, 1.0: 0.68, 1.2: 0.68})
        sc = R.scripted_check(records, base)
        assert sc["verdict"] == "REPRODUCES" and sc["direction_matches"] is True
        assert sc["opponent"] == "TACTIC-V0" and sc["row"] == "RULES-V0"
        assert sc["capture"] == {"r0.8": 0.98, "r1.0": 0.68, "r1.2": 0.68}
        assert sc["d150"] == {"r0.8": 1.0, "r1.0": 0.68, "r1.2": 0.68}
        assert set(sc["per_opponent"]) == set(ALL)  # reading 6: reported alongside

    def test_it_reports_a_different_direction_rather_than_asserting_one(self):
        records, base = self._records({0.8: 0.40, 1.0: 0.68, 1.2: 0.68})
        sc = R.scripted_check(records, base)
        assert sc["verdict"] == "DIFFERS" and sc["direction_matches"] is False

    def test_the_balanced_ratio_is_the_scripted_capture_nearest_a_half(self):
        records, base = self._records({0.8: 0.98, 1.0: 0.68, 1.2: 0.52})
        sc = R.scripted_check(records, base)
        assert sc["balanced_ratio"] == 1.2 and sc["balanced_capture"] == 0.52
        records, base = self._records({0.8: 0.55, 1.0: 0.68, 1.2: 0.90})
        assert R.scripted_check(records, base)["balanced_ratio"] == 0.8

    def test_branch_one_no_per_ratio_training(self):
        records, base = _round_records()
        v = R.verdicts(records, base)
        assert v["h_zs_1"]["verdict"] == "PASS"
        assert v["h_zs_3"]["verdict"] == "SUGGESTIVE-PASS"
        d = v["decision"]
        assert d["verdict"] == "NO PER-RATIO TRAINING" and d["per_ratio_training"] == []
        assert d["balanced_ratio"] == v["scripted_check"]["balanced_ratio"]
        assert "NOT needed to claim graceful transfer" in d["statement"]

    def test_branch_two_per_ratio_training_at_the_failing_ratio(self):
        records, base = _round_records(
            {0.8: _rates(0.70, 0.40), 1.0: _rates(0.41, 0.40), 1.2: _rates(0.50, 0.30)}
        )
        v = R.verdicts(records, base)
        assert v["h_zs_1"]["failing_ratios"] == ["r1.0"]
        assert v["h_zs_3"]["verdict"] == "SUGGESTIVE-PASS"
        d = v["decision"]
        assert d["verdict"] == "PER-RATIO TRAINING AT r1.0"
        assert d["per_ratio_training"] == [1.0]
        assert d["reason"] == "H-ZS-1 fail at a ratio"
        assert d["failing_ratios"] == ["r1.0"]
        assert "3 seeds) before S24.2" in d["statement"]

    def test_branch_three_per_ratio_training_regardless_overrides_branch_one(self):
        records, base = _round_records(picks={x: dict(EVEN) for x in R.ALL_RATIOS})
        v = R.verdicts(records, base)
        assert v["h_zs_1"]["verdict"] == "PASS"  # H-ZS-1 passed...
        assert v["h_zs_3"]["verdict"] == "FAIL→PER-RATIO-TRAINING"
        d = v["decision"]
        assert d["verdict"] == "PER-RATIO TRAINING REGARDLESS"
        # a LIST in every branch, never a string, and the reason says which rule
        assert d["per_ratio_training"] == [0.8, 1.0] == list(R.RATIOS)
        assert d["reason"] == "REGARDLESS (H-ZS-3 fail)"
        assert d["failing_ratios"] == v["h_zs_1"]["failing_ratios"] == []
        assert "override" in d["statement"]
        assert d["balanced_ratio"] is not None  # reported in every branch

    def test_an_incomplete_round_reads_no_decision(self):
        records, base = _round_records()
        del records[R.cell_name("SEL-80", "HUG", 0.8)]
        d = R.verdicts(records, base)["decision"]
        assert d["verdict"] == "INCOMPLETE" and d["per_ratio_training"] is None
        assert d["reason"] == "INCOMPLETE"
        assert "not read from a partial round" in d["statement"]

    def test_the_direction_paragraph_states_the_round(self):
        records, base = _round_records()
        text = R.direction(R.verdicts(records, base))
        for needed in (
            "H-ZS-1 (round) PASS",
            "H-ZS-2 PASS",
            "SUGGESTIVE",
            "UNPAIRED",
            "Newcombe",
            "DECISION:",
            "in-distribution",
        ):
            assert needed in text, needed
        # A1.1: the narrative must not misname the statistic
        assert "disjoint" not in text
        assert "Newcombe" in R.h_zs_2(records, base)["rule"]
        assert "disjoint" not in R.h_zs_2(records, base)["rule"]

    def test_the_precedence_is_safety_then_gates_then_completeness(self):
        records, base = _round_records()
        name = R.cell_name("SEL-80", "HUG", 0.8)
        hurt = dict(records, **{name: records[name] | {"violation_steps": 3}})
        assert "H-SAFE FAILS" in R.direction(R.verdicts(hurt, base))
        gated = dict(records, **{name: records[name] | {"inapplicable_picks": 2}})
        assert "REFUSES" in R.direction(R.verdicts(gated, base))
        del records[name]
        assert "incomplete" in R.direction(R.verdicts(records, base))


# ----------------------------------------------------------------------------
# CG-Z2 (prereg section 5): the watched episodes
# ----------------------------------------------------------------------------
class TestGifs:
    def test_the_seeds_are_the_first_three_gate_seeds_and_a_rule(self):
        assert R.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)
        assert R.GIF_N == 3
        assert R.gif_seeds() == tuple(R.GATE_SEED0 + i for i in range(R.GIF_N))

    def test_twelve_gifs_one_row_two_opponents_two_ratios_three_seeds(self):
        assert R.GIF_ROWS == ("SEL-81",)
        assert R.GIF_OPPONENTS == ("TACTIC-V0", "GREEDY")
        assert len(R.GIF_ROWS) * len(R.GIF_OPPONENTS) * len(R.RATIOS) * R.GIF_N == 12

    def test_gif_paths_carry_the_ratio(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.gif_path("SEL-81", "GREEDY", 0.8, 2_400_001) == (
            tmp_path / "s24_3zs_gifs" / "SEL-81_GREEDY_r0.8_2400001.gif"
        )

    def test_the_caption_carries_no_measured_number(self):
        title = R.gif_title("SEL-81", "GREEDY", 0.8, 2_400_000)
        assert title == "SEL-81 vs GREEDY · ratio 0.8 · episode 2400000"
        rec = _measurable("SEL-81_GREEDY_r0.8")
        for forbidden in (
            "capture", "switch", "pick", "measured", "N=", "PASS", "FAIL",
            f"{rec['capture_rate']:.3f}",
        ):
            assert forbidden not in title
        assert "rec" not in R.gif_title.__code__.co_varnames

    def test_the_frame_label_is_the_v0b_games_own_two_role_caption(self):
        env = R.build_cell_env(
            "RULES-V0_TACTIC-V0_r0.8", recorder=True, label=R.gif_title("RULES-V0", "TACTIC-V0", 0.8, 1)
        )
        assert re.fullmatch(r"EVADER: .+ \| PURSUER: .+", env.caption("law"))
        greedy = R.build_cell_env("RULES-V0_GREEDY_r1.0", recorder=True, label="x")
        assert greedy.caption("law") == "EVADER: GREEDY | PURSUER: UNKNOWN"
        for forbidden in ("capture", "rate", "PASS", "0.8", "ratio"):
            assert forbidden not in greedy.caption("law")

    def test_it_refuses_without_a_measurable_record_at_every_ratio(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered"))
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs()
        # ... and a half-covered set still refuses, before the first frame
        for o in R.GIF_OPPONENTS:
            for s in R.gif_seeds():
                _on_disk(tmp_path, _measurable(R.cell_name("SEL-81", o, 0.8)))
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs()

    def test_it_refuses_without_the_rows_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(S1, "OUT", tmp_path)
        for o in R.GIF_OPPONENTS:
            for x in R.RATIOS:
                _on_disk(tmp_path, _measurable(R.cell_name("SEL-81", o, x)))
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("rendered"))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.write_gifs()

    def test_an_unknown_row_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown CG-Z2 row"):
            R.write_gifs(["SEL-80"])

    def test_existing_gifs_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.gifs_dir().mkdir(parents=True)
        for o in R.GIF_OPPONENTS:
            for x in R.RATIOS:
                for s in R.gif_seeds():
                    R.gif_path("SEL-81", o, x, s).write_bytes(b"")
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("re-rendered"))
        assert R.write_gifs() == []


# ----------------------------------------------------------------------------
# CLI dispatch order and an empty round
# ----------------------------------------------------------------------------
class TestCli:
    def test_gifs_are_written_before_the_table_is_summarised(self, monkeypatch):
        order = []
        monkeypatch.setattr(R, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(R, "write_gifs", lambda *a, **k: order.append("gifs"))
        monkeypatch.setattr(
            R,
            "summarize",
            lambda *a, **k: (
                order.append("summarize"),
                {"verdicts": R.verdicts({}, {}), "direction": "x"},
            )[1],
        )
        monkeypatch.setattr(
            "sys.argv", ["derisk_s24_3zs.py", "--summarize", "--gifs", "--cells"]
        )
        R.main()
        assert order == ["cells", "gifs", "summarize"]

    def test_summarize_survives_a_round_with_no_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        blob = R.summarize()
        assert blob["verdicts"]["n_records"] == 0
        assert blob["verdicts"]["n_base_records"] == 18  # the real 1.2 column
        assert "incomplete" in blob["direction"]
        assert (tmp_path / "s24_3zs_round.json").exists()
        assert (tmp_path / "s24_3zs_round.png").exists()

    def test_summarize_refuses_while_a_construct_gate_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        records, base = _round_records()
        name = R.cell_name("SEL-80", "HUG", 0.8)
        monkeypatch.setattr(
            R, "load_records", lambda **k: dict(records, **{name: records[name] | {"cg_z1": None}})
        )
        monkeypatch.setattr(R, "load_base_records", lambda: base)
        with pytest.raises(SystemExit, match="REFUSES the summary"):
            R.summarize()
        assert not (tmp_path / "s24_3zs_round.json").exists()

    def test_a_non_measurable_1_2_column_refuses_the_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(S1, "OUT", tmp_path)  # an empty S24.1 cells dir
        with pytest.raises(SystemExit, match="the 1.2 column"):
            R.summarize()


# ----------------------------------------------------------------------------
# the launch wrapper
# ----------------------------------------------------------------------------
def _launch_checker(tmp_path: pathlib.Path) -> pathlib.Path:
    """The wrapper's OWN 1.2-column checker, extracted from the heredoc so the
    refusal conditions are tested rather than described."""
    body = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", _LAUNCH.read_text(), re.DOTALL).group(1)
    path = tmp_path / "checker.py"
    path.write_text(body)
    return path


class TestLauncher:
    def _run(self, tmp_path):
        return subprocess.run(
            [sys.executable, str(_launch_checker(tmp_path)), str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_it_accepts_a_complete_measurable_1_2_column(self, tmp_path):
        _base_on_disk(tmp_path / "results" / "derisk")
        got = self._run(tmp_path)
        assert got.returncode == 0, got.stdout
        assert "18 S24.1 cells" in got.stdout

    def test_it_refuses_an_absent_record(self, tmp_path):
        _base_on_disk(tmp_path / "results" / "derisk", rows=R.ROW_NAMES[:-1])
        got = self._run(tmp_path)
        assert got.returncode == 1 and "absent" in got.stdout

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes=20"),
            ({"seed0": 1}, "seed0=1"),
            ({"verdict": "UNDETERMINED"}, "UNDETERMINED"),
        ],
    )
    def test_it_refuses_a_non_measurable_record(self, tmp_path, overrides, match):
        root = tmp_path / "results" / "derisk"
        _base_on_disk(root)
        p = root / "s24_1_cells" / "SEL-80_GREEDY.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | overrides))
        got = self._run(tmp_path)
        assert got.returncode == 1 and "REFUSED" in got.stdout and match in got.stdout

    def test_the_wrapper_gates_the_worktree_the_symlink_and_the_checkpoints(self):
        src = _LAUNCH.read_text()
        for needed in (
            "REFUSED: another round is using the host",
            "no frozen worktree",
            "is not a symlink to",
            "without the gate runner",
            "S24-1-${a}_3000000.zip",
            "OMP_NUM_THREADS=4",
            "git rev-parse --short HEAD",
            "confirm by results/derisk/s24_3zs_cells/*.json, not by PIDs",
        ):
            assert needed in src, needed
        assert "ARMS=(SEL-80 SEL-81 SEL-82 NOMEM-81)" in src  # this round's four

    def test_the_four_chains_cover_every_cell_exactly_once(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH.read_text().splitlines()
            if "--cells" in line
        ]
        assert len(chains) == 4
        assert all(len(c) == 9 for c in chains)
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(R.CELL_ORDER)
        assert len(set(flat)) == 36

    def test_the_wrapper_never_summarises_or_renders(self):
        """A rate may not be read by the launch: CG-Z2 comes first and both are
        run by hand."""
        for line in _LAUNCH.read_text().splitlines():
            if line.startswith("nohup"):
                assert "--summarize" not in line and "--gifs" not in line


# ----------------------------------------------------------------------------
# the one end-to-end property the reuse rests on: a record of S24.1's schema
# ----------------------------------------------------------------------------
class TestRecordSchema:
    def test_a_two_episode_scripted_smoke_writes_s24_1s_schema_plus_the_ratio(
        self, tmp_path, monkeypatch
    ):
        """NOT a cell: two episodes, into a tmp dir, and `record_status` calls it
        INVALID so it can never become a number. What it proves is the claim the
        1.2 join rests on — S24.1's `run_cell` wrote this record, so the two
        columns have one schema and differ only by `ratio` and the CG-Z1 block."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        before = {k: getattr(S1, k) for k in R._BOUND_KEYS}
        rec = R.run_cell("RULES-V0_GREEDY_r0.8", n_episodes=2)
        assert {k: getattr(S1, k) for k in R._BOUND_KEYS} == before  # restored
        real = json.loads(
            pathlib.Path("results/derisk/s24_1_cells/RULES-V0_GREEDY.json").read_text()
        )
        assert set(rec) - {"ratio", "cg_z1"} == set(real)
        assert rec["ratio"] == 0.8 and rec["cg_z1"]["verdict"] == "PASS"
        assert rec["seed0"] == R.GATE_SEED0 and rec["kind"] == "scripted"
        assert rec["violation_steps"] == 0  # H-SAFE's quantity, on 2 episodes
        assert rec["provenance"]["spec"] == R.PREREG
        assert rec["provenance"]["ratios"] == [0.8, 1.0]
        # a smoke is NOT a measurable record and may never become a number
        assert R.record_status(rec)[0] == "INVALID"
        R.write_cell(rec)
        assert R.load_records(quiet=True) == {}
