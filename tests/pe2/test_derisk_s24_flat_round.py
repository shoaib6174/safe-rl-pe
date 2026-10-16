"""`scripts/derisk_s24_flat_gate.py` + `scripts/launch_s24_flat.sh` — S24-FLAT
Step F.3, the gate: 27 cells (`FLAT`/`FLAT40`/`STRUCT` x 3 seeds x 3
opponents), joined to the S24.1 selector records.

Pre-registration: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md`,
sections 2 to 6 INCLUDING Amendment A1. Covered here, each either a constant
the text fixes, a verdict rule it states, or a property the round's claims
rest on:

- the cell table is 27 (9 rows x 3 opponents): built envs for FLAT/FLAT40/
  STRUCT differ only in the family's own construction (action space, hold,
  env class) and the three cells of one row differ only in the opponent;
- record validity, duplicate rejection and UNDETERMINED (written, never
  re-run), through the `_bound()` rebinding of S24.1's own machinery;
- the S24.1 join (`SEL`/`NOMEM`/`LEAD-1.0`/`RULES-V0`) is READ and REFUSED
  whole if any record is absent, unreadable or not measurable, and REFUSES
  to run inside a binding;
- CG-F1's `hold` augmentation for `FLAT40` (A1.2: `assembly_diff ==
  ["hold"]`, which F.1's own `config_diff_vs_selector()` does not produce);
- every hypothesis (H-FLAT-1 per family and combined, H-FLAT-2, H-SAFE) is
  tested with its FAILING direction, not only its passing one, and with the
  INCOMPLETE guard; `n_switches` is `None` throughout and never averaged;
- CG-F2: the watched seeds are the first three GATE seeds (a rule, not a
  pick), the frame caption is the literal `PURSUER: FLAT` / `PURSUER:
  STRUCT`, the pass refuses without a measurable record or checkpoint, and
  `--gifs` dispatches before `--summarize`;
- the launch wrapper refuses on its own preconditions, checked by RUNNING it
  against fixtures with every `nohup` line replaced, so a refusal is tested
  rather than described and nothing can launch.

No real cells and no real training: the only envs built are CG-F1's and one
N = 3 smoke per row family (FLAT, FLAT40, STRUCT), each with a FRESH,
untrained policy, into a tmp checkpoint directory that is deleted afterward.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
_GATE_PATH = _SCRIPTS / "derisk_s24_flat_gate.py"
_LAUNCH = _SCRIPTS / "launch_s24_flat.sh"


def _module(name: str, path: pathlib.Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _module("derisk_s24_flat_gate", _GATE_PATH)
S1 = G.S1


# ----------------------------------------------------------------------------
# synthetic records, in the shape `run_cell` writes
# ----------------------------------------------------------------------------
def _caps(rate: float, n: int) -> list[bool]:
    k = round(rate * n)
    return [i < k for i in range(n)]


def _opponent_of(name: str) -> str:
    for o in G.OPPONENTS:
        if name.endswith(f"_{o}"):
            return o
    raise ValueError(f"no known opponent suffix in {name!r}")


def _cell(name: str, rate: float, n: int = 10, cg_f1_ok: bool = True) -> dict:
    """One record carrying only the fields the verdict layer reads."""
    caps = _caps(rate, n)
    spec = G.CELLS.get(name)
    base = {
        "name": name,
        "n_episodes": n,
        "seed0": G.GATE_SEED0,
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "wilson_capture": S1._ci(sum(caps), n),
        "steps": [10] * n,
        "violations": [0] * n,
        "violation_steps": 0,
        "n_switches": [None] * n,
        "picks": {},
        "inapplicable_picks": 0,
        "n_decisions": [5] * n,
        "branch": [],
        "shield": [],
        "aggregate": {"capture_rate": sum(caps) / n},
        # `S1._paired` reads this off EVERY record, joined comparator cells
        # (SEL/NOMEM/LEAD-1.0/RULES-V0) included.
        "in_distribution": G.OPPONENTS[_opponent_of(name)]["in_distribution"],
    }
    if spec is None:  # a joined S24.1 comparator cell
        return base
    return base | {
        "row": spec["row"],
        "family": spec["family"],
        "seed": spec["seed"],
        "opponent": spec["opponent"],
        "kind": "learned",
        "hold": spec["hold"],
        "cg_f1": {"ok": cg_f1_ok, "family": spec["family"]},
    }


def _round_records(flat: float = 0.2, struct: float = 0.2, sel: float = 0.9, n: int = 20) -> tuple[dict, dict]:
    """`(records, combined)` for the whole round: 27 own cells, plus
    `SEL`/`NOMEM`/`LEAD-1.0`/`RULES-V0` joined under S24.1's own cell_name."""
    records = {name: _cell(name, struct if spec["family"] == "STRUCT" else flat, n=n)
               for name, spec in G.CELLS.items()}
    combined = dict(records)
    for row in (*G.SEL_ROWS, *G.NOMEM_ROWS, S1.BASELINE_ROW, S1.RULES_ROW):
        for o in G.OPPONENTS:
            name = S1.cell_name(row, o)
            combined[name] = _cell(name, sel, n=n)
    return records, combined


def _probes(train_violations: int = 0, violations: int = 0) -> dict:
    return {
        row: {"rows": [{"train_violations": train_violations, "violations": violations}]}
        for row in G.ROW_NAMES
    }


def _arm_jsons(tmp_path, monkeypatch, skip: str | None = None, **overrides) -> None:
    """One clean `{PREFIX}_arm_{row}.json` per row (item 1's `_check_arm_record`),
    under `FLAT_TR`/`STRUCT_TR`'s OWN `OUT`, redirected to `tmp_path` so
    nothing here reads the real, in-progress `results/derisk/`."""
    monkeypatch.setattr(G.FLAT_TR, "OUT", tmp_path)
    monkeypatch.setattr(G.STRUCT_TR, "OUT", tmp_path)
    for row in G.ROW_NAMES:
        if row == skip:
            continue
        family = G.ROWS[row]["family"]
        tr = G.STRUCT_TR if family == "STRUCT" else G.FLAT_TR
        path = tr._arm_file(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "arm": row,
            "verdict": "MEASURED",
            "resumed_from": None,
            "cg_f1": {"ok": True},
            "hold": 40 if family == "FLAT40" else G.HOLD,
        } | overrides.get(row, {})
        path.write_text(json.dumps(rec))


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_it_is_27_cells_9_rows_x_3_opponents(self):
        assert len(G.CELLS) == 27
        assert len(G.ROW_NAMES) == 9
        assert len(G.OPPONENTS) == 3

    def test_every_row_appears_in_exactly_3_cells(self):
        from collections import Counter

        counts = Counter(spec["row"] for spec in G.CELLS.values())
        assert counts == {row: 3 for row in G.ROW_NAMES}

    def test_families_and_seeds(self):
        assert G.FLAT_FAMILIES == ("FLAT", "FLAT40")
        assert G.ALL_FAMILIES == ("FLAT", "FLAT40", "STRUCT")
        for family in G.ALL_FAMILIES:
            for seed in G.SEEDS:
                assert f"{family}-{seed}" in G.ROWS

    def test_flat40s_declared_hold_is_40_and_flats_is_the_game_hold(self):
        assert G.CELLS["FLAT40-80_GREEDY"]["hold"] == 40
        assert G.CELLS["FLAT-80_GREEDY"]["hold"] == G.HOLD
        assert G.CELLS["STRUCT-80_GREEDY"]["hold"] == G.HOLD

    def test_built_flat_and_flat40_envs_share_assembly_except_hold(self):
        flat = G.build_cell_env("FLAT-80_GREEDY")
        flat40 = G.build_cell_env("FLAT40-80_GREEDY")
        assert type(flat) is type(flat40)
        assert flat.action_space == flat40.action_space
        assert type(flat.game.shield) is type(flat40.game.shield)
        assert flat.cfg.evader_speed_ratio == flat40.cfg.evader_speed_ratio == 1.2
        assert flat.hold != flat40.hold  # the ONE declared difference (A1.2)
        assert flat.game.hold == flat40.game.hold  # the underlying game's own hold

    def test_built_struct_env_shares_ratio_and_shield_class_with_flat(self):
        flat = G.build_cell_env("FLAT-80_GREEDY")
        struct = G.build_cell_env("STRUCT-80_GREEDY")
        assert struct.cfg.evader_speed_ratio == flat.cfg.evader_speed_ratio == 1.2
        # STRUCT's gate env builds shield=None; run_cell assigns a fresh one
        # per episode (reading 3) rather than at construction.
        assert struct.shield is None

    def test_cells_within_a_row_differ_only_in_opponent(self):
        for row in G.ROW_NAMES:
            envs = {o: G.build_cell_env(G.cell_name(row, o)) for o in G.OPPONENTS}
            ratios = {o: e.cfg.evader_speed_ratio for o, e in envs.items()}
            assert len(set(ratios.values())) == 1


# ----------------------------------------------------------------------------
# CG-F1 (the hold augmentation, A1.2)
# ----------------------------------------------------------------------------
class TestCgF1:
    def test_flat_has_no_assembly_diff(self):
        cg = G.flat_cg_f1("FLAT")
        assert cg["config_diff"] == []
        assert cg["assembly_diff"] == []
        assert cg["ok"] is True
        assert cg["flat_hold"] == G.HOLD
        assert cg["own_hold_differs"] is False

    def test_flat40_has_no_assembly_diff_either(self):
        """The A1.2 correction (prereg, 2026-10-16): `config_diff_vs_selector`
        reads the GAME's `hold` (10 for every arm), so `FLAT40` reads
        `assembly_diff == []` too — not `["hold"]` as A1.2 first said. Its own
        40-step span is recorded separately as `flat_hold`."""
        cg = G.flat_cg_f1("FLAT40")
        assert cg["config_diff"] == []
        assert cg["assembly_diff"] == []
        assert cg["ok"] is True
        assert cg["flat_hold"] == 40
        assert cg["game_hold"] == G.HOLD
        assert cg["own_hold_differs"] is True

    def test_struct_cg_f1_asserts_on_the_built_gate_env(self):
        cg = G.struct_cg_f1()
        assert cg["family"] == "STRUCT"
        assert cg["ok"] is True
        assert cg["shield"]["class"] == "RecoveryShieldV2"

    def test_cg_f1_is_cached_per_family(self):
        G._CG_F1_CACHE.clear()
        a = G.cg_f1("FLAT40")
        b = G.cg_f1("FLAT40")
        assert a is b

    def test_cg_f1_summary_refuses_without_a_stored_block(self, tmp_path, monkeypatch):
        _arm_jsons(tmp_path, monkeypatch)  # clean arm records; the per-cell block is the offender
        records = {"FLAT-80_GREEDY": _cell("FLAT-80_GREEDY", 0.3)}
        del records["FLAT-80_GREEDY"]["cg_f1"]
        out = G.cg_f1_summary(records)
        assert out["verdict"] == "REFUSE"
        assert out["offenders"] == ["FLAT-80_GREEDY"]

    def test_cg_f1_summary_passes_with_an_ok_block(self, tmp_path, monkeypatch):
        _arm_jsons(tmp_path, monkeypatch)
        records = {"FLAT-80_GREEDY": _cell("FLAT-80_GREEDY", 0.3)}
        out = G.cg_f1_summary(records)
        assert out["verdict"] == "PASS"
        assert out["offenders"] == []
        assert out["arm_records"]["FLAT-80"] == {"ok": True, "why": ""}
        assert all(block["ok"] for block in out["arm_records"].values())

    def test_cg_f1_summary_flags_a_mismatch_within_a_family_as_an_offender(self, tmp_path, monkeypatch):
        """Item 7: a cell whose stored `cg_f1` block disagrees with another
        cell of the SAME family (both individually `ok`) is itself an
        offender — not silently accepted as the family's first block."""
        _arm_jsons(tmp_path, monkeypatch)
        records = {
            "FLAT-80_GREEDY": _cell("FLAT-80_GREEDY", 0.3),
            "FLAT-81_GREEDY": _cell("FLAT-81_GREEDY", 0.3),
        }
        records["FLAT-81_GREEDY"]["cg_f1"] = {"ok": True, "family": "FLAT", "config_diff": ["ratio"]}
        out = G.cg_f1_summary(records)
        assert out["verdict"] == "REFUSE"
        assert "FLAT-81_GREEDY" in out["offenders"]
        assert "FLAT-80_GREEDY" not in out["offenders"]

    # --- item 1: the nine arm JSONs are re-read, not just the per-cell block ---
    def test_cg_f1_summary_refuses_on_a_missing_arm_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G.FLAT_TR, "OUT", tmp_path)
        monkeypatch.setattr(G.STRUCT_TR, "OUT", tmp_path)  # no arm files written at all
        records = {"FLAT-80_GREEDY": _cell("FLAT-80_GREEDY", 0.3)}
        out = G.cg_f1_summary(records)
        assert out["verdict"] == "REFUSE"
        assert "arm:FLAT-80" in out["offenders"]
        assert out["arm_records"]["FLAT-80"]["ok"] is False
        assert "absent" in out["arm_records"]["FLAT-80"]["why"]

    @pytest.mark.parametrize(
        ("override", "why_substr"),
        [
            ({"verdict": "UNDETERMINED"}, "not MEASURED"),
            ({"resumed_from": "some/ckpt.zip"}, "not None"),
            ({"cg_f1": None}, "cg_f1 absent"),
            ({"cg_f1": {"ok": False}}, "cg_f1 absent"),
            ({"hold": 10}, "not 40"),  # FLAT40 only
        ],
    )
    def test_cg_f1_summary_flags_each_arm_offender_kind(self, tmp_path, monkeypatch, override, why_substr):
        _arm_jsons(tmp_path, monkeypatch, **{"FLAT40-80": override})
        records = {"FLAT-80_GREEDY": _cell("FLAT-80_GREEDY", 0.3)}
        out = G.cg_f1_summary(records)
        assert out["verdict"] == "REFUSE"
        assert "arm:FLAT40-80" in out["offenders"]
        assert why_substr in out["arm_records"]["FLAT40-80"]["why"]
        # every OTHER row's arm record is still clean
        assert all(
            out["arm_records"][row]["ok"] for row in G.ROW_NAMES if row != "FLAT40-80"
        )


# ----------------------------------------------------------------------------
# CG-F3 (documentary): item 6's CG-T2 clause
# ----------------------------------------------------------------------------
class TestCgF3:
    def test_note_states_cg_t2_labels_do_not_apply(self):
        out = G.cg_f3()
        assert out["verdict"] == "N/A (documentary)"
        assert "CG-T2" in out["note"]
        assert "do not apply to FLAT/FLAT40/STRUCT rows" in out["note"]


# ----------------------------------------------------------------------------
# record validity, duplicates, UNDETERMINED (through the `_bound()` rebinding)
# ----------------------------------------------------------------------------
class TestRecordValidity:
    def test_measurable(self):
        rec = {"name": "FLAT-80_GREEDY", "n_episodes": G.N_EPISODES, "seed0": G.GATE_SEED0}
        assert G.record_status(rec) == ("MEASURABLE", "")

    def test_wrong_n_episodes_is_invalid(self):
        rec = {"name": "FLAT-80_GREEDY", "n_episodes": 1, "seed0": G.GATE_SEED0}
        status, why = G.record_status(rec)
        assert status == "INVALID" and "n_episodes" in why

    def test_wrong_seed0_is_invalid(self):
        rec = {"name": "FLAT-80_GREEDY", "n_episodes": G.N_EPISODES, "seed0": 0}
        status, why = G.record_status(rec)
        assert status == "INVALID" and "seed0" in why

    def test_unknown_name_is_invalid(self):
        rec = {"name": "NOT-A-CELL", "n_episodes": G.N_EPISODES, "seed0": G.GATE_SEED0}
        status, _why = G.record_status(rec)
        assert status == "INVALID"

    def test_undetermined_verdict_stays_undetermined(self):
        rec = {"name": "FLAT-80_GREEDY", "verdict": S1.UNDETERMINED, "error": "boom"}
        assert G.record_status(rec)[0] == S1.UNDETERMINED

    def test_undetermined_record_carries_row_family_opponent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        rec = G.undetermined_record("FLAT40-81_HUG", ValueError("boom"), "cpu")
        assert rec["verdict"] == S1.UNDETERMINED
        assert rec["row"] == "FLAT40-81"
        assert rec["family"] == "FLAT40"
        assert rec["opponent"] == "HUG"
        assert rec["n_episodes"] == 0
        assert G.record_status(rec)[0] == S1.UNDETERMINED

    def test_scan_records_rejects_duplicates_both(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        G.cells_dir().mkdir(parents=True)
        rec = {"name": "FLAT-80_GREEDY", "n_episodes": G.N_EPISODES, "seed0": G.GATE_SEED0}
        (G.cells_dir() / "a.json").write_text(json.dumps(rec))
        (G.cells_dir() / "b.json").write_text(json.dumps(rec))
        measurable, rejected, by_cell = G.scan_records()
        assert "FLAT-80_GREEDY" not in measurable
        assert set(rejected) == {"a.json", "b.json"}
        assert by_cell["FLAT-80_GREEDY"] == ["a.json", "b.json"]

    def test_run_cells_refuses_an_unknown_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown cell"):
            G.run_cells(["NOT-A-CELL"])

    def test_run_cells_refuses_a_non_measurable_file_before_any_episode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        G.cells_dir().mkdir(parents=True)
        (G.cells_dir() / "FLAT-80_GREEDY.json").write_text(
            json.dumps({"name": "FLAT-80_GREEDY", "n_episodes": 1, "seed0": G.GATE_SEED0})
        )
        with pytest.raises(SystemExit, match="REFUSED before any episode ran"):
            G.run_cells(["FLAT-80_GREEDY"])

    def test_run_cells_raises_before_the_first_episode_on_a_flat_cg_f1_failure(self, tmp_path, monkeypatch):
        """Item 3: unlike STRUCT (whose own `assert_cg_f1` already raises),
        `flat_cg_f1` is a pure function that never raises on its own — the
        docstring's promise ("raises on failure") only held for STRUCT before
        this fix."""
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(G, "require_ckpt", lambda row: pathlib.Path("unused"))
        monkeypatch.setattr(
            G, "cg_f1",
            lambda family: {
                "ok": False, "family": family, "config_diff": ["x"], "assembly_diff": [],
                "obs_diff_indices_on_identical_reset": [], "flat_tactic_entries_nonzero": 0,
                "selector_tactic_entries_nonzero": 0,
            },
        )
        ran: list[str] = []
        monkeypatch.setattr(G, "run_cell", lambda *a, **k: ran.append(a[0]))
        with pytest.raises(SystemExit, match="CG-F1 FAILED for FLAT"):
            G.run_cells(["FLAT-80_GREEDY"])
        assert ran == []  # no episode ran


# ----------------------------------------------------------------------------
# the S24.1 join
# ----------------------------------------------------------------------------
class TestS24_1Join:
    def test_it_refuses_inside_a_binding(self):
        with G._bound():
            with pytest.raises(SystemExit, match="may not run inside a binding"):
                G.load_s24_1_comparators()

    def test_it_succeeds_outside_a_binding_on_the_real_column(self):
        # results/derisk/s24_1_cells/ already holds S24.1's real, measured
        # column in this repo; this is a READ-ONLY check that the join finds
        # all 24 comparator cells it needs, never writing anything.
        out = G.load_s24_1_comparators()
        assert len(out) == 24
        for row in G.COMPARATOR_ROWS:
            for o in G.OPPONENTS:
                assert S1.cell_name(row, o) in out

    def test_it_refuses_whole_on_a_missing_comparator(self, monkeypatch, tmp_path):
        # `S1.OUT`, not `S1.cells_dir` itself: replacing the FUNCTION object
        # would make `_assert_outside_binding`'s identity check (correctly)
        # think a binding is active, since that is exactly what `_bound()`
        # does. Redirecting the directory `cells_dir()` reads from is not a
        # binding.
        monkeypatch.setattr(S1, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="REFUSED"):
            G.load_s24_1_comparators()

    def test_it_refuses_whole_on_a_not_measurable_comparator(self, monkeypatch, tmp_path):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        d = S1.cells_dir()
        d.mkdir(parents=True)
        for row in (*G.SEL_ROWS, *G.NOMEM_ROWS, S1.BASELINE_ROW, S1.RULES_ROW):
            for o in G.OPPONENTS:
                name = S1.cell_name(row, o)
                (d / f"{name}.json").write_text(
                    json.dumps({"name": name, "n_episodes": 1, "seed0": S1.GATE_SEED0})
                )
        with pytest.raises(SystemExit, match="REFUSED"):
            G.load_s24_1_comparators()


# ----------------------------------------------------------------------------
# hypotheses — passing AND failing directions, plus INCOMPLETE
# ----------------------------------------------------------------------------
class TestHypotheses:
    def test_h_flat_1_passes_when_sel_beats_both_families(self):
        _records, combined = _round_records(flat=0.1, struct=0.1, sel=0.9)
        out = G.h_flat_1(combined)
        assert out["verdict"] == "PASS"
        assert set(out["passing_families"]) == {"FLAT", "FLAT40"}
        for family in G.FLAT_FAMILIES:
            assert out["families"][family]["verdict"] == "PASS"

    def test_h_flat_1_fails_when_sel_does_not_beat_either_family(self):
        _records, combined = _round_records(flat=0.9, struct=0.1, sel=0.1)
        out = G.h_flat_1(combined)
        assert out["verdict"] == "FAIL"
        assert out["passing_families"] == []
        # the direction is reported even on failure (S1._paired's own rule)
        seed_block = out["families"]["FLAT"]["seeds"]["80"]
        for opp_block in seed_block["opponents"].values():
            assert opp_block["direction"] in ("SEL-80 < FLAT-80", "SEL-80 == FLAT-80")

    def test_h_flat_1_requires_both_families_not_just_one(self):
        # SEL beats FLAT but not FLAT40: A1.2 says the claim needs BOTH.
        records = {}
        combined = {}
        for name, spec in G.CELLS.items():
            rate = 0.9 if spec["family"] == "FLAT40" else 0.1
            if spec["family"] == "STRUCT":
                rate = 0.1
            rec = _cell(name, rate, n=20)
            records[name] = rec
            combined[name] = rec
        for row in (*G.SEL_ROWS, *G.NOMEM_ROWS, S1.BASELINE_ROW, S1.RULES_ROW):
            for o in G.OPPONENTS:
                name = S1.cell_name(row, o)
                combined[name] = _cell(name, 0.9, n=20)
        out = G.h_flat_1(combined)
        assert out["families"]["FLAT"]["verdict"] == "PASS"
        assert out["families"]["FLAT40"]["verdict"] == "FAIL"
        assert out["verdict"] == "FAIL"  # NOT a pass: both families are required

    def test_h_flat_1_is_incomplete_when_a_cell_is_missing(self):
        _records, combined = _round_records(flat=0.1, sel=0.9)
        del combined["FLAT-80_GREEDY"]
        out = G.h_flat_1(combined)
        assert out["families"]["FLAT"]["verdict"] == "INCOMPLETE"
        assert out["verdict"] == "INCOMPLETE"

    def test_h_flat_2_passes_and_fails(self):
        _r, combined_pass = _round_records(struct=0.1, sel=0.9)
        assert G.h_flat_2(combined_pass)["verdict"] == "PASS"
        _r, combined_fail = _round_records(struct=0.9, sel=0.1)
        assert G.h_flat_2(combined_fail)["verdict"] == "FAIL"

    def test_n_switches_is_none_throughout_and_never_averaged(self):
        records, _combined = _round_records()
        for rec in records.values():
            assert rec["n_switches"] == [None] * rec["n_episodes"]
            assert "mean_switches" not in rec["aggregate"]

    def test_h_safe_holds_with_clean_records_and_probes(self):
        records, _combined = _round_records()
        out = G.h_safe(records, _probes())
        assert out["verdict"] == "HOLDS"

    def test_h_safe_fails_on_a_gate_violation(self):
        records, _combined = _round_records()
        records["FLAT-80_GREEDY"]["violation_steps"] = 1
        out = G.h_safe(records, _probes())
        assert out["verdict"] == "FAILS"
        assert "FLAT-80_GREEDY" in out["offenders"]

    def test_h_safe_fails_on_a_training_violation(self):
        records, _combined = _round_records()
        out = G.h_safe(records, _probes(train_violations=1))
        assert out["verdict"] == "FAILS"
        assert out["training_offenders"]

    def test_h_safe_is_incomplete_on_absent_training_evidence(self):
        """A3.1: absent evidence is an offender, never a zero."""
        records, _combined = _round_records()
        probes = _probes()
        probes["STRUCT-81"] = {"rows": [], "missing": True}
        out = G.h_safe(records, probes)
        assert out["verdict"] == "INCOMPLETE"
        assert "STRUCT-81" in out["missing_training"]

    def test_pooled_immobile_and_pinned_reads_branch_for_flat(self):
        name = "FLAT-80_GREEDY"
        rec = _cell(name, 0.3, n=2)
        rec["steps"] = [100, 100]
        rec["branch"] = [
            {"immobile_by_branch": {"law": 0, "search": 10}, "longest_pinned_run": 50},
            {"immobile_by_branch": {"law": 0, "search": 0}, "longest_pinned_run": 0},
        ]
        records = {G.cell_name("FLAT-80", o): rec for o in G.OPPONENTS}
        out = G.pooled_immobile_and_pinned(records, "FLAT-80")
        assert out["source"].startswith("recorder branch_summary")
        assert out["immobile_step_rate"] == pytest.approx(10 / 200 * 1)  # pooled below
        assert out["pinned_run_episode_fraction"] == pytest.approx(0.5)

    def test_pooled_immobile_and_pinned_reads_shield_for_struct(self):
        rec = _cell("STRUCT-80_GREEDY", 0.3, n=1)
        rec["steps"] = [100]
        rec["shield"] = [{"n_immobile": 5, "longest_immobile_run": 50}]
        records = {G.cell_name("STRUCT-80", o): rec for o in G.OPPONENTS}
        out = G.pooled_immobile_and_pinned(records, "STRUCT-80")
        assert "ShieldMetrics.summary()" in out["source"]
        assert out["pinned_run_episode_fraction"] == 1.0

    def test_pooled_immobile_and_pinned_is_none_without_cells(self):
        assert G.pooled_immobile_and_pinned({}, "FLAT-80") is None

    def test_h_flat_3_is_reported_not_gated(self):
        _records, combined = _round_records()
        out = G.h_flat_3(combined)
        assert out["verdict"] == "REPORTED"
        assert out["gated"] is False
        for block in out["vs_comparators"].values():
            assert block["gated"] is False
            assert block["row"].startswith("FLAT-")  # reading 4: FLAT-s only

    def test_direction_reports_h_safe_first(self):
        records, combined = _round_records()
        records["FLAT-80_GREEDY"]["violation_steps"] = 1
        v = G.verdicts(records, combined, _probes())
        assert v["h_safe"]["verdict"] == "FAILS"
        assert G.direction(v).startswith("H-SAFE FAILS")

    def test_direction_reports_cg_f1_before_h_flat_1(self):
        records, combined = _round_records()
        del records["FLAT-80_GREEDY"]["cg_f1"]
        v = G.verdicts(records, combined, _probes())
        assert v["cg_f1"]["verdict"] == "REFUSE"
        assert G.direction(v).startswith("CG-F1 REFUSES")

    def test_direction_names_the_section_4_refutation_on_0_of_3(self, tmp_path, monkeypatch):
        _arm_jsons(tmp_path, monkeypatch)  # clean arm records: CG-F1 must not REFUSE
        records, combined = _round_records(flat=0.3, struct=0.3, sel=0.1)
        v = G.verdicts(records, combined, _probes())
        assert "withdrawn to 'a selector matches flat learning" in G.direction(v)

    def test_direction_reports_decisions_and_env_steps_per_decision(self, tmp_path, monkeypatch):
        _arm_jsons(tmp_path, monkeypatch)
        records, combined = _round_records()
        v = G.verdicts(records, combined, _probes())
        assert "A1.1's cadence" in G.direction(v)
        for row in G.ROW_NAMES:
            assert v["cadence"][row]["env_steps_per_decision"] == pytest.approx(2.0)

    def test_cadence_table_includes_the_selector_rows(self, tmp_path, monkeypatch):
        """Item 4: A1.1's cadence table (and `direction()`'s cadence line)
        must carry `SEL-80/81/82` too, from the JOINED S24.1 records — the
        selector's own `n_decisions`/`steps` so the ratio can be read off one
        artefact."""
        _arm_jsons(tmp_path, monkeypatch)
        records, combined = _round_records()
        v = G.verdicts(records, combined, _probes())
        for row in G.SEL_ROWS:
            assert v["cadence"][row] is not None
            assert v["cadence"][row]["env_steps_per_decision"] == pytest.approx(2.0)
        direction_text = G.direction(v)
        for row in G.SEL_ROWS:
            assert row in direction_text


# ----------------------------------------------------------------------------
# CG-F2: GIF seed rule, captions, dispatch order
# ----------------------------------------------------------------------------
class TestGifsAndCaptions:
    def test_gif_seeds_is_the_first_three_gate_seeds(self):
        assert G.gif_seeds() == (G.GATE_SEED0, G.GATE_SEED0 + 1, G.GATE_SEED0 + 2)
        assert G.gif_seeds() == S1.gif_seeds()

    def test_gif_rows_and_opponents_give_42_gifs(self):
        """Item 5 (A1.4): FLAT40-80/81/82 join the watched rows — 7 rows
        (not 4) x 2 opponents x 3 seeds. (The prereg's own figure, "36", does
        not match 7 x 2 x 3 = 42; this asserts the arithmetic actually
        implemented, not the prereg's stated total.)"""
        assert set(G.GIF_ROWS) == {
            "FLAT-80", "FLAT-81", "FLAT-82",
            "FLAT40-80", "FLAT40-81", "FLAT40-82",
            "STRUCT-81",
        }
        assert len(G.GIF_ROWS) * len(G.GIF_OPPONENTS) * G.GIF_N == 42

    def test_flat_caption_is_the_literal_pursuer_flat(self):
        env = G.build_cell_env("FLAT-80_TACTIC-V0", recorder=True, label="t")
        env.reset(seed=1)  # FlatMDP's own reset — never `env.game.reset()` (reading 1, F.1)
        caption = env.game.caption("tag")
        assert "PURSUER: FLAT" in caption
        assert caption.startswith("EVADER: ")

    def test_flat_caption_against_a_scripted_opponent_uses_its_literal_name(self):
        env = G.build_cell_env("FLAT-80_GREEDY", recorder=True, label="t")
        env.reset(seed=1)
        assert env.game.caption("tag") == "EVADER: GREEDY | PURSUER: FLAT"

    def test_struct_caption_is_the_literal_pursuer_struct(self):
        env = G.build_cell_env("STRUCT-80_HUG", recorder=True, label="t")
        env.reset(seed=1)
        assert env.caption("tag") == "EVADER: HUG | PURSUER: STRUCT"

    def test_write_gifs_refuses_an_unknown_row(self):
        with pytest.raises(SystemExit, match="unknown CG-F2 row"):
            G.write_gifs(["NOT-A-ROW"])

    def test_write_gifs_refuses_without_a_measurable_cell_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no cell record"):
            G.write_gifs(["FLAT-80"])


class TestMainDispatchOrder:
    def test_cells_then_gifs_then_summarize(self, monkeypatch):
        order = []
        monkeypatch.setattr(G, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(G, "write_gifs", lambda *a, **k: order.append("gifs"))

        def fake_summarize(device="cpu"):
            order.append("summarize")
            return {"verdicts": {
                "h_flat_1": {"verdict": "FAIL", "passing_families": []},
                "h_flat_2": {"verdict": "FAIL"},
                "h_safe": {},
                "cg_f1": {},
                "n_records": 0, "n_cells": 27, "complete": False,
            }, "direction": "x"}

        monkeypatch.setattr(G, "summarize", fake_summarize)
        monkeypatch.setattr(sys, "argv", ["prog", "--cells", "--gifs", "--summarize"])
        monkeypatch.setattr(pathlib.Path, "mkdir", lambda *a, **k: None)
        G.main()
        assert order == ["cells", "gifs", "summarize"]

    def test_summarize_alone_does_not_run_cells_or_gifs(self, monkeypatch):
        calls = []
        monkeypatch.setattr(G, "run_cells", lambda *a, **k: calls.append("cells"))
        monkeypatch.setattr(G, "write_gifs", lambda *a, **k: calls.append("gifs"))
        monkeypatch.setattr(G, "summarize", lambda device="cpu": {"verdicts": {
            "h_flat_1": {"verdict": "FAIL", "passing_families": []},
            "h_flat_2": {"verdict": "FAIL"}, "h_safe": {}, "cg_f1": {},
            "n_records": 0, "n_cells": 27, "complete": False,
        }, "direction": "x"})
        monkeypatch.setattr(sys, "argv", ["prog", "--summarize"])
        monkeypatch.setattr(pathlib.Path, "mkdir", lambda *a, **k: None)
        G.main()
        assert calls == []


class TestSummarizeWritesRejectedRecords:
    def test_summarize_writes_rejected_records(self, tmp_path, monkeypatch):
        """Item 7: `summarize()` must write `rejected_records` into the round
        JSON, as S24.1's own `summarize` does (`derisk_s24_1_gate.py:1578`).
        Every io boundary is stubbed to a controlled fake so nothing here
        reads or writes the real, in-progress `results/derisk/`."""
        monkeypatch.setattr(G, "OUT", tmp_path)
        _arm_jsons(tmp_path, monkeypatch)  # clean arm records: CG-F1 must not REFUSE
        records, combined = _round_records()
        comparators = {k: v for k, v in combined.items() if k not in records}
        rejected = {"dup_a.json": "duplicate of a measurable record", "dup_b.json": "duplicate"}
        monkeypatch.setattr(G, "load_records", lambda quiet=False: records)
        monkeypatch.setattr(G, "scan_records", lambda: (dict(records), rejected, {}))
        monkeypatch.setattr(G, "load_s24_1_comparators", lambda: comparators)
        monkeypatch.setattr(G, "read_probes", lambda: _probes())
        monkeypatch.setattr(G, "write_figure", lambda *a, **k: None)
        blob = G.summarize()
        assert blob["rejected_records"] == rejected


# ----------------------------------------------------------------------------
# N = 3 smoke: one per row family, fresh policy, tmp checkpoint dir, deleted
# ----------------------------------------------------------------------------
class TestSmoke:
    @pytest.mark.parametrize("name", ["FLAT-80_GREEDY", "FLAT40-80_TACTIC-V0", "STRUCT-80_HUG"])
    def test_run_cell_end_to_end_with_a_fresh_policy(self, name, tmp_path, monkeypatch):
        from stable_baselines3 import PPO

        fresh: dict = {}

        def fake_load_policy(row, device="cpu"):
            if row not in fresh:
                env = G.build_cell_env(G.cell_name(row, "GREEDY"))
                fresh[row] = PPO("MlpPolicy", env, seed=0, device=device)
            return fresh[row]

        monkeypatch.setattr(G, "load_policy", fake_load_policy)
        rec = G.run_cell(name, n_episodes=3)
        assert rec["n_episodes"] == 3
        assert rec["n_switches"] == [None, None, None]
        assert rec["picks"] == {}
        assert rec["inapplicable_picks"] == 0
        assert len(rec["capture_ind"]) == 3
        assert rec["cg_f1"]["ok"] is True
        assert (tmp_path / "unused").exists() is False  # nothing written to disk here


# ----------------------------------------------------------------------------
# the launch wrapper — RUN against fixtures, with every `nohup` line replaced
# (the pattern `tests/pe2/test_derisk_s24_3pr.py` uses for its own launchers)
# ----------------------------------------------------------------------------
def _fixture(script: pathlib.Path, tmp: pathlib.Path) -> pathlib.Path:
    """The wrapper with its roots pointed at a tmp tree, its `pgrep` guard
    dropped (a pytest command line matching `derisk_s2[34]_` would trip it) and
    every `nohup` replaced by an echo, so the REFUSALS are exercised and
    nothing can launch."""
    out = []
    for line in script.read_text().splitlines():
        if line.startswith("MAIN="):
            line = f"MAIN={tmp}"
        elif line.startswith("WT="):
            line = f"WT={tmp}/wt"
        elif line.startswith("V="):
            line = f"V={sys.executable}"
        elif line.startswith("if pgrep"):
            continue
        elif line.startswith("mkdir -p /home/niro-1"):
            line = f"mkdir -p {tmp}/logs"
        elif line.lstrip().startswith("nohup"):
            line = "echo LAUNCHED"
        out.append(line)
    body = "\n".join(out)
    assert "nohup" not in body  # nothing in a fixture may launch
    path = tmp / script.name
    path.write_text(body)
    return path


def _run_fixture(script: pathlib.Path, tmp: pathlib.Path):
    return subprocess.run(
        ["zsh", str(_fixture(script, tmp))], capture_output=True, text=True, check=False
    )


def _worktree(tmp: pathlib.Path) -> None:
    (tmp / "wt" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "wt" / "scripts" / "derisk_s24_flat_gate.py").write_text("# frozen\n")
    (tmp / "results" / "derisk").mkdir(parents=True, exist_ok=True)


def _ckpts(tmp: pathlib.Path, skip: str | None = None) -> None:
    for arm in G.ROW_NAMES:
        if arm == skip:
            continue
        d = tmp / "results" / "derisk" / f"ckpts_S24-{arm}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"S24-{arm}_3000000.zip").write_bytes(b"")


def _probe_jsons(tmp: pathlib.Path, skip: str | None = None, **overrides) -> None:
    for arm in G.ROW_NAMES:
        if arm == skip:
            continue
        prefix = "s24_struct" if G.ROWS[arm]["family"] == "STRUCT" else "s24_flat"
        rows = [{"mark": 3_000_000, "violations": 0, "train_violations": 0} | overrides.get(arm, {})]
        (tmp / "results" / "derisk" / f"{prefix}_probe_{arm}.json").write_text(
            json.dumps({"arm": arm, "checkpoints": rows})
        )


class TestLauncher:
    def test_it_refuses_without_the_frozen_worktree(self, tmp_path):
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "no frozen worktree" in got.stdout

    def test_it_refuses_a_worktree_without_the_gate_runner(self, tmp_path):
        (tmp_path / "wt" / "scripts").mkdir(parents=True)
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "without the gate runner" in got.stdout

    def test_it_refuses_a_missing_final_checkpoint(self, tmp_path):
        _worktree(tmp_path)
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "no final 3M checkpoint" in got.stdout

    def test_it_refuses_a_missing_checkpoint_for_any_of_the_nine_arms(self, tmp_path):
        _worktree(tmp_path)
        _ckpts(tmp_path, skip="STRUCT-82")
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1
        assert "no final 3M checkpoint for arm STRUCT-82" in got.stdout

    def test_it_refuses_a_missing_probe_json(self, tmp_path):
        _worktree(tmp_path)
        _ckpts(tmp_path)
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "no probe JSON for arm" in got.stdout

    def test_it_refuses_a_struct_probe_json_at_the_flat_prefix(self, tmp_path):
        """The two families' probe files live under different prefixes
        (`s24_flat_probe_*` vs `s24_struct_probe_*`); a STRUCT probe filed
        under the flat prefix must still read as missing."""
        _worktree(tmp_path)
        _ckpts(tmp_path)
        _probe_jsons(tmp_path, skip="STRUCT-80")
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "no probe JSON for arm STRUCT-80" in got.stdout

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ({"train_violations": 1}, "train_violations = 1"),
            ({"violations": 2}, "violations = 2"),
        ],
    )
    def test_it_refuses_unclean_probe_rows(self, tmp_path, override, match):
        _worktree(tmp_path)
        _ckpts(tmp_path)
        _probe_jsons(tmp_path, **{"STRUCT-81": override})
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "REFUSED" in got.stdout and match in got.stdout

    def test_it_refuses_a_probe_row_that_does_not_carry_the_counter(self, tmp_path):
        """A3.1: absent evidence is an offender, never a zero."""
        _worktree(tmp_path)
        _ckpts(tmp_path)
        _probe_jsons(tmp_path)
        path = tmp_path / "results" / "derisk" / "s24_flat_probe_FLAT-81.json"
        blob = json.loads(path.read_text())
        del blob["checkpoints"][0]["train_violations"]
        path.write_text(json.dumps(blob))
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1
        assert "train_violations absent" in got.stdout
        assert "absent evidence is an offender" in got.stdout

    def test_it_refuses_an_empty_probe_file(self, tmp_path):
        _worktree(tmp_path)
        _ckpts(tmp_path)
        _probe_jsons(tmp_path)
        path = tmp_path / "results" / "derisk" / "s24_struct_probe_STRUCT-82.json"
        path.write_text(json.dumps({"arm": "STRUCT-82", "checkpoints": []}))
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 1 and "no probe rows" in got.stdout

    def test_it_starts_four_chains_on_a_clean_tree(self, tmp_path):
        _worktree(tmp_path)
        _ckpts(tmp_path)
        _probe_jsons(tmp_path)
        got = _run_fixture(_LAUNCH, tmp_path)
        assert got.returncode == 0, got.stdout + got.stderr
        assert got.stdout.count("LAUNCHED") == 4

    def test_the_wrapper_gates_the_host_the_symlink_and_the_threads(self):
        src = _LAUNCH.read_text()
        for needed in (
            "REFUSED: another round is using the host",
            "derisk_s2[34]_",
            "is not a symlink to",
            "OMP_NUM_THREADS=4",
            "git rev-parse --short HEAD",
            "confirm by results/derisk/s24_flat_cells/*.json, not by PIDs",
        ):
            assert needed in src, needed
        assert "pgrep -f" in src and r"\|" not in src.split("pgrep -f")[1].split("\n")[0]
        assert len(G.ROW_NAMES) == 9  # the binding count A1.2 fixes, not the brief's older six

    def test_the_four_chains_cover_all_27_cells_exactly_once(self):
        src = _LAUNCH.read_text()
        cells: list[str] = []
        for line in src.splitlines():
            if "--cells" in line and line.lstrip().startswith("nohup"):
                cells += line.split("--cells", 1)[1].split(">", 1)[0].split()
        assert sorted(cells) == sorted(G.CELL_ORDER)
        assert len(cells) == 27
