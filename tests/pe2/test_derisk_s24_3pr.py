"""`scripts/derisk_s24_3pr_train.py` + `scripts/derisk_s24_3pr_gate.py` —
S24.3-PR, per-ratio training of the tactic selector at 0.8 and 1.0.

Pre-registration: `docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md`.
Everything here is either a constant that text fixes, a verdict rule it states,
or one of the properties the round's claims rest on:

- the REBINDING really is a rebinding: the six arms, their ratios, their probe
  streams and their artefact paths are this round's, every other constant of
  the trainer is S24.1's object, and every bound name is restored;
- a crash through S24.1's own training loop writes a PR-named UNDETERMINED
  record that carries the `ratio` (the 3zs review's lesson about
  `undetermined_record`, applied to the trainer's `_crash_record`);
- CG-P1 holds on the BUILT training env at 0.8 — the ratio is the only
  `EnvConfig` difference from an S24.1 arm's env and the observation entry
  carries it at float32 — and a SECOND config difference is REFUSED;
- the gate's cell table is 6 x 3 = 18, each row at ITS ratio only;
- both joined columns are READ and REFUSED whole if any record is absent,
  unreadable, not measurable or at the wrong ratio, and both REFUSE to run
  inside a binding;
- every hypothesis is tested with its FAILING direction, not only its passing
  one, and with the VACUOUS / INCOMPLETE guards;
- CG-P3: the watched seeds are the first three GATE seeds (a rule, not a pick),
  the pass refuses without a measurable record, no caption carries a measured
  number, and `--gifs` dispatches before `--summarize`;
- both launch wrappers refuse on their own preconditions, checked by RUNNING
  them against fixtures with every `nohup` line replaced, so a refusal is
  tested rather than described and nothing can launch.

No real training and no real cells: the only training is one 1 000-step smoke
into a tmp directory, and the only envs built are CG-P1's and CG-Z1's.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import types

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.selector_mdp import ACTIONS, OBS_SLICES
from pe2.structured import HOLD

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
_TRAIN_PATH = _SCRIPTS / "derisk_s24_3pr_train.py"
_GATE_PATH = _SCRIPTS / "derisk_s24_3pr_gate.py"
_LAUNCH_TRAIN = _SCRIPTS / "launch_s24_3pr_train.sh"
_LAUNCH_GATE = _SCRIPTS / "launch_s24_3pr.sh"


def _module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _module("derisk_s24_3pr_gate", _GATE_PATH)
# The gate loads the trainer by path, so THAT module object is the one whose
# arms, ratios and checkpoint paths the gate actually reads; the tests below
# drive it rather than a second copy.
TR = R.TR
T1, Z, S1 = TR.T1, R.Z, R.S1
assert TR.__file__ == str(_TRAIN_PATH)

N = 500  # episodes in the synthetic records below — the round's own N
ALL = ("GREEDY", "HUG", "TACTIC-V0")
EVEN = dict.fromkeys(ACTIONS, 20)
CHASEY = dict.fromkeys(ACTIONS, 0) | {"INTERCEPT": 400, "LEAD": 400, "SWEEP": 200}
PATIENTY = dict.fromkeys(ACTIONS, 0) | {"INTERCEPT": 100, "LEAD": 100, "SWEEP": 800}
ONE_TACTIC = dict.fromkeys(ACTIONS, 0) | {"LEAD": 900}
NO_DECISIONS = dict.fromkeys(ACTIONS, 0)


# ----------------------------------------------------------------------------
# synthetic records, in the shape the frozen `run_cell` writes
# ----------------------------------------------------------------------------
def _caps(rate: float, n: int = N) -> list[bool]:
    k = round(rate * n)
    return [i < k for i in range(n)]


def _cell(
    name: str, rate: float, picks: dict | None = None, n: int = N, ratio: float | None = None
) -> dict:
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
        "in_distribution": False,
    }
    if spec is None:  # a joined comparator cell: S24.3-ZS's or S24.1's record
        return base | {"ratio": ratio}
    return base | {
        "row": spec["row"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": "learned",
        "ratio": spec["ratio"],
        "cg_z1": {
            "gate": "CG-Z1",
            "cell": name,
            "ratio": spec["ratio"],
            "config_diff_vs_twin": ["evader_speed_ratio"],
            "cfg_ratio": spec["ratio"],
            "obs_ratio": spec["ratio"],
            "obs_checked": True,
            "verdict": "PASS",
        },
    }


def _round_records(
    pr: float = 0.70,
    zs: float = 0.60,
    law: float = 0.40,
    pr_picks: dict | None = None,
    zs12_picks: dict | None = None,
) -> tuple[dict, dict]:
    """`(records, joined)` for the whole round: 18 PR cells, the zero-shot
    column at 0.8/1.0 and the 1.2 column."""
    pr_picks = CHASEY if pr_picks is None else pr_picks
    zs12_picks = PATIENTY if zs12_picks is None else zs12_picks
    records = {n: _cell(n, pr, pr_picks) for n in R.CELL_ORDER}
    joined: dict = {}
    for row in R.ZS_ROWS:
        for ratio in R.RATIOS:
            for o in ALL:
                name = R.cell_name(row, o, ratio)
                rate = zs if row.startswith("SEL-") else law
                joined[name] = _cell(name, rate, dict(EVEN), ratio=ratio)
    for row in (*Z.ROW_NAMES,):
        for o in ALL:
            name = R.cell_name(row, o, R.BASE_RATIO)
            picks = zs12_picks if row.startswith("SEL-") else dict(EVEN)
            joined[name] = _cell(name, 0.5, picks, ratio=R.BASE_RATIO)
    return records, joined


def _probes(**overrides) -> dict:
    out = {
        row: {
            "path": f"s24_3pr_probe_{row}.json",
            "missing": False,
            "rows": [
                {
                    "mark": TR.TOTAL,
                    "ratio": R.row_ratio(row),
                    "train_inapplicable_picks": 0,
                    "train_violations": 0,
                    "inapplicable_picks": 0,
                    "violations": 0,
                }
            ],
        }
        for row in R.ROW_NAMES
    }
    return out | overrides


def _arms(**overrides) -> dict:
    out = {
        arm: {
            "path": f"s24_3pr_arm_{arm}.json",
            "missing": False,
            "record": {
                "arm": arm,
                "ratio": R.row_ratio(arm),
                "cg_p1": {
                    "verdict": "PASS",
                    "config_diff_vs_twin": ["evader_speed_ratio"],
                    "cfg_ratio": R.row_ratio(arm),
                    "obs_ratio": R.row_ratio(arm),
                },
            },
        }
        for arm in R.ROW_NAMES
    }
    return out | overrides


def _v(records=None, joined=None, probes=None, arms=None, rejected=None) -> dict:
    if records is None:
        records, joined_ = _round_records()
        joined = joined_ if joined is None else joined
    return R.verdicts(
        records,
        joined,
        _probes() if probes is None else probes,
        {} if rejected is None else rejected,
        _arms() if arms is None else arms,
    )


def _on_disk(out: pathlib.Path, rec: dict, name: str | None = None) -> pathlib.Path:
    d = out / "s24_3pr_cells"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name or rec['name']}.json"
    path.write_text(json.dumps(rec))
    return path


def _measurable(name: str) -> dict:
    return _cell(name, 0.5, n=R.N_EPISODES)


# ============================================================================
# Deliverable 1 — the trainer, as a rebinding of S24.1's
# ============================================================================
class TestTrainerArms:
    def test_the_six_arms_are_the_preregistered_ones(self):
        assert R.ROW_NAMES == tuple(TR.ARMS) == (
            "SELr0.8-80", "SELr0.8-81", "SELr0.8-82",
            "SELr1.0-80", "SELr1.0-81", "SELr1.0-82",
        )
        assert TR.RATIOS == (0.8, 1.0) and TR.BASE_RATIO == 1.2
        assert TR.SEEDS == (80, 81, 82) and TR.SEEDS is T1.SEEDS
        for arm, spec in TR.ARMS.items():
            assert spec["memory"] is True  # section 3: memory ON on every arm
            assert spec["family"] == "SEL"
            assert arm == f"SELr{spec['ratio']:.1f}-{spec['seed']}"
        assert sorted({s["ratio"] for s in TR.ARMS.values()}) == [0.8, 1.0]
        assert sorted({s["seed"] for s in TR.ARMS.values()}) == [80, 81, 82]

    def test_everything_but_the_ratio_is_s24_1s_object(self):
        """Section 2: "Nothing else changes between a 0.8/1.0 arm and S24.1's
        1.2 arms" — so the budget, the probe cadence, gamma and the opponent are
        READ from the frozen trainer, not restated."""
        assert (TR.TOTAL, TR.EVERY, TR.PROBE_N) == (T1.TOTAL, T1.EVERY, T1.PROBE_N)
        assert (TR.TOTAL, TR.EVERY, TR.PROBE_N) == (3_000_000, 250_000, 200)
        assert (TR.GAMMA, TR.OPPONENT) == (T1.GAMMA, T1.OPPONENT) == (0.99, "TACTIC-V0")
        assert T1.RATIO == 1.2  # the twin CG-P1 asserts against
        assert TR.OMP_THREADS == "4"

    def test_the_probe_streams_are_per_ratio_and_disjoint(self):
        assert TR.PROBE_SEED0_BY_RATIO == {0.8: 2_450_000, 1.0: 2_500_000}
        assert TR.probe_seed0("SELr0.8-81") == 2_450_000
        assert TR.probe_seed0("SELr1.0-81") == 2_500_000
        # the module's own guard, re-stated as the property it protects
        train = {s * 1000 + i for s in TR.SEEDS for i in range(1000)}
        gate = set(range(2_400_000, 2_400_500))
        streams = [
            train,
            gate,
            set(range(T1.PROBE_SEED0, T1.PROBE_SEED0 + TR.PROBE_N)),
            *(
                set(range(s0, s0 + TR.PROBE_N))
                for s0 in TR.PROBE_SEED0_BY_RATIO.values()
            ),
        ]
        for i, a in enumerate(streams):
            for b in streams[i + 1 :]:
                assert not (a & b)

    def test_the_artefact_paths_are_this_rounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        assert TR.arm_file("SELr0.8-80") == tmp_path / "s24_3pr_arm_SELr0.8-80.json"
        assert TR.probe_file("SELr1.0-82") == tmp_path / "s24_3pr_probe_SELr1.0-82.json"
        assert TR.ckpt_dir("SELr0.8-81") == tmp_path / "ckpts_S24-3PR-SELr0.8-81"
        # reading 1: the DIRECTORY is this round's, the file stem is S24.1's,
        # because the stem is hardcoded inside the frozen probe loop
        assert TR.ckpt_path("SELr0.8-81") == (
            tmp_path / "ckpts_S24-3PR-SELr0.8-81" / "S24-1-SELr0.8-81_3000000.zip"
        )
        # the stem this round inherits is hardcoded in the frozen probe loop
        assert 'f"S24-1-{arm}_{mark}.zip"' in (_SCRIPTS / "derisk_s24_1_train.py").read_text()

    def test_under_the_binding_s24_1s_own_helpers_resolve_to_this_rounds(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        with TR._bound("SELr0.8-80"):
            assert T1._arm_file("SELr0.8-80") == TR.arm_file("SELr0.8-80")
            assert T1._probe_file("SELr0.8-80") == TR.probe_file("SELr0.8-80")
            assert T1._ckpt_dir("SELr0.8-80") == TR.ckpt_dir("SELr0.8-80")
            assert T1.RATIO == 0.8 and T1.PROBE_SEED0 == 2_450_000
            assert T1.ARMS is TR.ARMS and T1.PREREG == TR.PREREG
        with TR._bound("SELr1.0-82"):
            assert T1.RATIO == 1.0 and T1.PROBE_SEED0 == 2_500_000

    def test_the_prereg_exists_and_provenance_pins_it(self):
        assert TR.spec_path().exists()
        p = TR.provenance()
        assert p["prereg"] == "docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md"
        assert p["prereg_sha256"] == T1._sha256(TR.spec_path())
        assert len(p["prereg_sha256"]) == 64
        assert p["derived_from"] == T1.PREREG
        assert p["ratios"] == [0.8, 1.0] and p["base_ratio"] == 1.2
        assert p["round"] == TR.ROUND
        # the frozen baseline record is still S24.1's provenance, untouched
        assert p["baseline_record"].endswith("s24_tactic_game_v0b.json")
        assert p["obs_len"] == 55

    def test_the_script_prints_no_capture_rate(self):
        for line in _TRAIN_PATH.read_text().splitlines():
            if "print(" not in line:
                continue
            assert "capture_rate" not in line, line
            assert "win_rate" not in line, line


class TestTrainerBinding:
    def test_every_bound_name_is_restored(self):
        before = {k: getattr(T1, k) for k in TR._BOUND_KEYS}
        with TR._bound("SELr0.8-80"):
            inside = {k: getattr(T1, k) for k in TR._BOUND_KEYS}
            assert all(inside[k] != before[k] for k in TR._BOUND_KEYS)
        assert {k: getattr(T1, k) for k in TR._BOUND_KEYS} == before

    def test_it_restores_even_when_the_call_raises(self):
        before = {k: getattr(T1, k) for k in TR._BOUND_KEYS}
        with pytest.raises(ValueError, match="x"), TR._bound("SELr1.0-81"):
            raise ValueError("x")
        assert {k: getattr(T1, k) for k in TR._BOUND_KEYS} == before

    def test_the_per_arm_names_are_bound_only_when_an_arm_is_given(self):
        with TR._bound():
            assert T1.RATIO == 1.2  # no arm: S24.1's own ratio stands
            assert T1.PROBE_SEED0 == 2_350_000
            assert T1.ARMS is TR.ARMS  # the always-bound names are bound

    def test_orig_holds_s24_1s_own_versions_so_a_wrapper_cannot_recurse(self):
        assert TR._ORIG["train"] is not TR.train
        assert TR._ORIG["probe"] is not TR.probe
        assert TR._ORIG["_crash_record"] is not TR._crash_record
        assert TR._ORIG["provenance"] is not TR.provenance
        assert TR._ORIG["run"] is not TR.run
        assert set(TR._BOUND_KEYS) <= set(TR._ORIG)
        for key in TR._BOUND_KEYS:
            assert hasattr(T1, key), key

    def test_the_undetermined_record_is_bound_too(self):
        """The 3zs review's lesson: without this the crash record would be filed
        in S24.1's shape, with no `ratio` in it."""
        assert "_crash_record" in TR._BOUND_ALWAYS
        with TR._bound("SELr0.8-80"):
            assert T1._crash_record is TR._crash_record


class TestTrainerRecords:
    def test_a_crash_through_the_loop_writes_a_pr_named_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        monkeypatch.setattr(TR, "cg_p1", lambda arm, device="cpu": {"verdict": "PASS"})

        def crash(*_a, **_k):
            raise RuntimeError("this arm crashed")

        monkeypatch.setitem(TR._ORIG, "train", crash)
        TR.run(["SELr0.8-81"], "cpu")
        path = tmp_path / "s24_3pr_arm_SELr0.8-81.json"
        assert path.exists()  # PR-named, not S24.1-named
        assert not (tmp_path / "s24_1_arm_SELr0.8-81.json").exists()
        rec = json.loads(path.read_text())
        assert rec["verdict"] == "UNDETERMINED"
        assert "this arm crashed" in rec["reason"]
        assert rec["ratio"] == 0.8  # the field the binding exists for
        assert rec["probe_seed0"] == 2_450_000
        assert rec["round"] == TR.ROUND
        assert rec["prereg"] == TR.PREREG
        assert rec["provenance"]["prereg"] == TR.PREREG

    def test_a_cg_p1_refusal_aborts_the_launch_instead_of_filing_records(
        self, tmp_path, monkeypatch
    ):
        """A2.4: a configuration refusal is not a crashed arm."""
        monkeypatch.setattr(TR, "OUT", tmp_path)

        def refuse(*_a, **_k):
            raise SystemExit("CG-P1 REFUSES")

        monkeypatch.setattr(TR, "cg_p1", refuse)
        with pytest.raises(SystemExit, match="CG-P1 REFUSES"):
            TR.run(["SELr0.8-80", "SELr0.8-81"], "cpu")
        assert not list(tmp_path.glob("s24_3pr_arm_*.json"))

    def test_an_existing_record_is_skipped_arm_granularly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        TR._write_json(TR.arm_file("SELr1.0-80"), {"verdict": "MEASURED"})
        monkeypatch.setitem(
            TR._ORIG, "train", lambda *a, **k: pytest.fail("a finished arm re-ran")
        )
        TR.run(["SELr1.0-80"], "cpu")

    def test_the_probe_row_carries_the_ratio_and_its_stream(self, monkeypatch):
        monkeypatch.setitem(
            TR._ORIG, "probe", lambda model, arm, n, seed0: {"k": 1, "n": 2}
        )
        row = TR.probe(None, "SELr1.0-82")
        assert row == {"k": 1, "n": 2, "ratio": 1.0, "probe_seed0": 2_500_000}

    def test_the_arm_record_carries_the_ratio_and_the_cg_p1_block(self, monkeypatch):
        monkeypatch.setattr(
            TR, "cg_p1", lambda arm, device="cpu": {"verdict": "PASS", "arm": arm}
        )
        monkeypatch.setitem(
            TR._ORIG, "train", lambda arm, device: {"arm": arm, "verdict": "MEASURED"}
        )
        rec = TR.train("SELr0.8-82")
        assert rec["ratio"] == 0.8 and rec["probe_seed0"] == 2_450_000
        assert rec["cg_p1"] == {"verdict": "PASS", "arm": "SELr0.8-82"}
        assert rec["prereg"] == TR.PREREG and rec["round"] == TR.ROUND


# ----------------------------------------------------------------------------
# CG-P1 on the BUILT training env (prereg section 6)
# ----------------------------------------------------------------------------
class TestCGP1:
    def test_the_config_diff_is_the_ratio_and_nothing_else(self):
        assert TR.config_diff(EnvConfig()) == []
        assert TR.config_diff(EnvConfig(evader_speed_ratio=0.8)) == ["evader_speed_ratio"]
        assert TR.config_diff(EnvConfig(evader_speed_ratio=0.8, max_steps=99)) == [
            "evader_speed_ratio",
            "max_steps",
        ]

    def test_it_holds_on_a_built_env_at_0_8(self):
        block = TR.cg_p1("SELr0.8-81")
        assert block["verdict"] == "PASS"
        assert block["ratio"] == 0.8 and block["twin_ratio"] == 1.2
        assert block["config_diff_vs_twin"] == ["evader_speed_ratio"]
        assert block["cfg_ratio"] == 0.8
        # float32: 0.8 is not exactly representable, so the entry is compared at
        # the observation's own precision
        assert TR._f32_eq(block["obs_ratio"], 0.8)
        assert not TR._f32_eq(block["obs_ratio"], 1.2)
        assert block["obs_ratio"] != 0.8  # ... and a bare `==` would refuse it
        assert block["memory"] is True and block["env_seed0"] == 81_000

    def test_the_built_env_is_s24_1s_assembly_with_one_field_moved(self):
        got = TR.make_env("SELr0.8-80")
        twin = T1.make_env("SEL-80")
        assert type(got) is type(twin)
        assert got.game.cfg.evader_speed_ratio == 0.8
        assert twin.game.cfg.evader_speed_ratio == 1.2
        assert TR.config_diff(got.game.cfg) == ["evader_speed_ratio"]
        assert got.memory == twin.memory is True
        assert got.game.inner.seed0 == twin.game.inner.seed0 == 80_000
        assert got.game.hold == twin.game.hold == HOLD
        assert type(got.game.shield).__name__ == type(twin.game.shield).__name__
        assert [t.name for t in got.game.law.tactics] == [
            t.name for t in twin.game.law.tactics
        ]
        assert got.observation_space.shape == twin.observation_space.shape

    def test_a_second_config_difference_is_refused(self, monkeypatch):
        """The round's whole attribution: a capture difference is the SPEED's
        only if nothing else moved."""
        monkeypatch.setattr(
            TR,
            "make_env",
            lambda arm: types.SimpleNamespace(
                memory=True,
                game=types.SimpleNamespace(
                    cfg=EnvConfig(evader_speed_ratio=0.8, max_steps=99),
                    inner=types.SimpleNamespace(seed0=80_000),
                ),
                reset=lambda **kw: (np.full(55, 0.8, dtype=np.float32), {}),
            ),
        )
        with pytest.raises(SystemExit, match="CG-P1 REFUSES"):
            TR.cg_p1("SELr0.8-80")

    def test_a_wrong_observation_entry_is_refused_even_when_the_config_is_right(
        self, monkeypatch
    ):
        obs = np.zeros(55, dtype=np.float32)
        obs[OBS_SLICES["speed_ratio"]] = 1.2  # the twin's value, not this arm's
        monkeypatch.setattr(
            TR,
            "make_env",
            lambda arm: types.SimpleNamespace(
                memory=True,
                game=types.SimpleNamespace(
                    cfg=EnvConfig(evader_speed_ratio=0.8),
                    inner=types.SimpleNamespace(seed0=80_000),
                ),
                reset=lambda **kw: (obs, {}),
            ),
        )
        with pytest.raises(SystemExit, match="observation's ratio entry"):
            TR.cg_p1("SELr0.8-80")


# ============================================================================
# Deliverable 2 — the gate, as a rebinding of the 3zs runner
# ============================================================================
class TestCellTable:
    def test_eighteen_cells_six_rows_three_opponents_one_ratio_each(self):
        assert len(R.CELLS) == 18 == 6 * 3
        assert list(R.OPPONENTS) == ["GREEDY", "HUG", "TACTIC-V0"]
        assert set(R.CELLS) == {
            R.cell_name(r, o, R.row_ratio(r)) for r in R.ROW_NAMES for o in ALL
        }
        for name, spec in R.CELLS.items():
            assert spec["ratio"] == R.row_ratio(spec["row"])
            assert spec["kind"] == "learned" and spec["memory"] is True
            assert spec["hold"] == HOLD == 10
            assert name == f"{spec['row']}_{spec['opponent']}_r{spec['ratio']:.1f}"
        # reading 1: a row is gated at ITS ratio only — no 6 x 3 x 2 sweep
        assert len({s["ratio"] for s in R.CELLS.values()}) == 2
        assert sum(1 for s in R.CELLS.values() if s["ratio"] == 0.8) == 9

    def test_the_rows_are_the_trainers_arms_and_the_comparators_are_not(self):
        assert R.ROW_NAMES == tuple(TR.ARMS)
        assert R.ZS_ROWS == ("SEL-80", "SEL-81", "SEL-82", "LEAD-1.0", "RULES-V0")
        assert not set(R.ROW_NAMES) & set(R.ZS_ROWS)
        assert set(R.ZS_ROWS) <= set(Z.ROW_NAMES)  # they are 3zs's own rows
        assert R.pr_row(0.8, 81) == "SELr0.8-81" in TR.ARMS

    def test_pre_registered_constants_are_the_frozen_rounds(self):
        assert (R.GATE_SEED0, R.N_EPISODES) == (2_400_000, 500)
        assert (R.GATE_SEED0, R.N_EPISODES) == (Z.GATE_SEED0, Z.N_EPISODES)
        assert (R.P_GATE, R.P_CHI2, R.PASS_MIN) == (0.05, 0.01, 2)
        assert R.RATIOS == (0.8, 1.0) and R.BASE_RATIO == 1.2
        assert R.HEADLINE_RATIO == 0.8  # section 5, reading 2
        assert R.BASELINE_ROW == "LEAD-1.0" and R.RULES_ROW == "RULES-V0"
        assert R.CHASE == ("INTERCEPT", "LEAD")
        assert set(R.CHASE) | set(R.PATIENT) == set(ACTIONS)
        assert R.ARTEFACT_PREFIX == "s24_3pr" and R.OMP_THREADS == "4"

    def test_the_artefact_paths_are_this_rounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.cell_file("SELr0.8-80_HUG_r0.8") == (
            tmp_path / "s24_3pr_cells" / "SELr0.8-80_HUG_r0.8.json"
        )
        assert R.round_file("json") == tmp_path / "s24_3pr_round.json"
        assert R.gifs_dir() == tmp_path / "s24_3pr_gifs"
        assert R.gif_path("SELr1.0-81", "GREEDY", 2_400_001) == (
            tmp_path / "s24_3pr_gifs" / "SELr1.0-81_GREEDY_r1.0_2400001.gif"
        )

    def test_the_checkpoint_path_is_the_trainers_own(self):
        """Reading 7: binding the trainer's function rather than restating the
        name is what keeps the two from drifting."""
        with R._bound():
            assert S1.ckpt_path is TR.ckpt_path
            assert S1.require_ckpt is R.require_ckpt
            assert S1.ckpt_path("SELr1.0-80") == TR.ckpt_path("SELr1.0-80")
        assert R.require_ckpt is S1.require_ckpt

    def test_the_prereg_exists_and_provenance_hashes_it(self):
        assert R.spec_path().exists()
        p = R.provenance("cuda")
        assert p["spec"] == "docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md"
        assert p["spec_sha256"] == S1._sha256(R.spec_path())
        assert p["ratios"] == [0.8, 1.0] and p["base_ratio"] == 1.2
        assert p["gate_stream"] == 2_400_000
        assert p["trained_by"] == TR.PREREG
        assert p["joined_from"] == [Z.PREREG, S1.PREREG]
        assert p["device"] == "cuda"


class TestGateBinding:
    def test_every_bound_name_is_restored(self):
        before = {
            Z: {k: getattr(Z, k) for k in R._BOUND_Z},
            S1: {k: getattr(S1, k) for k in R._BOUND_S1},
        }
        with R._bound(0.8):
            for mod, was in before.items():
                for k in was:
                    assert getattr(mod, k) is not was[k], k
        for mod, was in before.items():
            assert {k: getattr(mod, k) for k in was} == was

    def test_it_restores_even_when_the_call_raises(self):
        before = {k: getattr(Z, k) for k in R._BOUND_Z} | {
            f"S1.{k}": getattr(S1, k) for k in R._BOUND_S1
        }
        with pytest.raises(ValueError, match="x"), R._bound(1.0):
            raise ValueError("x")
        after = {k: getattr(Z, k) for k in R._BOUND_Z} | {
            f"S1.{k}": getattr(S1, k) for k in R._BOUND_S1
        }
        assert after == before

    def test_the_binding_reaches_s24_1_through_the_3zs_runners_own_bound(self):
        with R._bound(0.8):
            assert S1.CELLS is R.CELLS  # pushed down two levels
            assert S1.CELL_ORDER == R.CELL_ORDER
            assert S1.cell_name("SELr0.8-80", "HUG") == "SELr0.8-80_HUG_r0.8"
            assert S1.cell_config().evader_speed_ratio == 0.8
            assert S1.record_status is Z.record_status
            assert S1.provenance is R.provenance
            assert S1.ROWS is R.ROWS and S1.LEARNED_ROWS == R.ROW_NAMES
            assert S1.probe_file is TR.probe_file

    def test_nesting_restores_the_outer_binding(self):
        before = Z.CELLS
        with R._bound():
            outer = Z.CELLS
            assert len(outer) == 18
            with R._one("SELr1.0-82_HUG_r1.0"):
                assert list(Z.CELLS) == ["SELr1.0-82_HUG_r1.0"]
                assert Z.cell_config(1.0).evader_speed_ratio == 1.0
            assert Z.CELLS is outer
        assert Z.CELLS is before

    def test_orig_holds_the_frozen_versions(self):
        assert R._ORIG_Z["run_cell"] is not R.run_cell
        assert R._ORIG_Z["CELLS"] is not R.CELLS and len(R._ORIG_Z["CELLS"]) == 36
        assert R._ORIG_Z["ROW_NAMES"] == Z.ROW_NAMES
        assert R._ORIG_S1["_paired"] is S1._paired
        assert set(R._BOUND_Z) <= set(R._ORIG_Z)
        assert set(R._BOUND_S1) <= set(R._ORIG_S1)

    def test_the_frozen_helpers_are_imported_not_reimplemented(self):
        assert R.cell_name is Z.cell_name
        assert R.cell_config is Z.cell_config and R.config_diff is Z.config_diff
        assert R._f32_eq is Z._f32_eq and R._r is Z._r
        assert R.chi2_table is S1.chi2_table and R.gif_seeds is S1.gif_seeds
        assert R.load_policy is S1.load_policy
        assert R._write_json is S1._write_json and R._ci is S1._ci
        assert R.GAME is sys.modules["s24_tactic_game"]
        assert R.OPPONENTS is Z.OPPONENTS

    def test_the_gate_cells_are_run_by_the_frozen_run_cell(self, tmp_path, monkeypatch):
        # Into a tmp record root: the sweep skips cells that ALREADY have a
        # measurable record, so this property is only visible on an empty one
        # (the equivalent 3zs test reads the real directory and stopped being
        # able to see it once that round's cells were measured).
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        seen: list = []
        monkeypatch.setattr(Z, "cg_z1", lambda n, d="cpu": seen.append(("cg_z1", n)))
        monkeypatch.setitem(
            Z._ORIG, "run_cells", lambda *a, **k: seen.append(("CHAIN",)) or []
        )
        R.run_cells(["SELr0.8-80_HUG_r0.8", "SELr1.0-82_GREEDY_r1.0"])
        assert seen == [
            ("cg_z1", "SELr0.8-80_HUG_r0.8"),
            ("cg_z1", "SELr1.0-82_GREEDY_r1.0"),
            ("CHAIN",),
        ]

    def test_an_unknown_cell_name_refuses(self):
        with pytest.raises(SystemExit, match="unknown cell"):
            R.run_cells(["SEL-80_HUG_r0.8"])  # the 3zs round's name
        with pytest.raises(SystemExit, match="unknown cell"):
            R.run_cells(["SELr0.8-80_HUG_r1.0"])  # the wrong ratio for the row


# ----------------------------------------------------------------------------
# CG-Z1 on this round's built EVALUATION envs
# ----------------------------------------------------------------------------
class TestCGZ1:
    def test_a_built_cell_carries_its_ratio_in_the_config_and_the_observation(self):
        block = R.cg_z1("SELr0.8-81_GREEDY_r0.8")
        assert block["verdict"] == "PASS" and block["cfg_ratio"] == 0.8
        assert block["obs_checked"] is True
        assert R._f32_eq(block["obs_ratio"], 0.8)
        assert block["config_diff_vs_twin"] == ["evader_speed_ratio"]
        assert R.cg_z1("SELr1.0-80_HUG_r1.0")["cfg_ratio"] == 1.0

    def test_the_eval_env_is_the_trained_arms_env_with_the_opponent_moved(self):
        env = R.build_cell_env("SELr0.8-82_GREEDY_r0.8")
        train_env = TR.make_env("SELr0.8-82")
        assert type(env) is type(train_env)
        assert env.game.cfg.evader_speed_ratio == 0.8
        assert R.config_diff(env.game.cfg) == ["evader_speed_ratio"]
        assert env.memory is True
        assert env.game.hold == train_env.game.hold == HOLD


# ----------------------------------------------------------------------------
# the two joins (prereg section 4, reading 4)
# ----------------------------------------------------------------------------
def _zs_on_disk(out: pathlib.Path, rows=None, ratios=None, **overrides) -> None:
    d = out / "s24_3zs_cells"
    d.mkdir(parents=True, exist_ok=True)
    for row in R.ZS_ROWS if rows is None else rows:
        for ratio in R.RATIOS if ratios is None else ratios:
            for o in ALL:
                name = R.cell_name(row, o, ratio)
                rec = {
                    "name": name,
                    "n_episodes": Z.N_EPISODES,
                    "seed0": Z.GATE_SEED0,
                    "ratio": ratio,
                    "capture_rate": 0.5,
                    "capture_ind": _caps(0.5, Z.N_EPISODES),
                    "picks": dict(EVEN),
                } | overrides
                (d / f"{name}.json").write_text(json.dumps(rec))


class TestJoins:
    def test_both_columns_read_the_real_records_on_disk(self):
        zs = R.load_zs_records()
        assert len(zs) == 30 == len(R.ZS_ROWS) * len(R.RATIOS) * len(ALL)
        assert set(zs) == {
            R.cell_name(r, o, x) for r in R.ZS_ROWS for o in ALL for x in R.RATIOS
        }
        for name, rec in zs.items():
            assert rec["ratio"] == R.CELLS.get(name, {"ratio": rec["ratio"]})["ratio"]
            assert rec["n_episodes"] == 500 and rec["seed0"] == 2_400_000
            assert Z.record_status(rec)[0] == "MEASURABLE"
        base = R.load_base_records()
        assert len(base) == 18 and all(r["ratio"] == 1.2 for r in base.values())
        joined = R.load_joined()
        assert len(joined) == 48 and not set(joined) & set(R.CELL_ORDER)

    def test_the_joins_use_the_rules_that_measured_the_records(self):
        assert R._ORIG_Z["record_status"] is Z.record_status
        assert R._ORIG_Z["load_base_records"] is Z.load_base_records
        assert R.zs_cell_file("SEL-80", "HUG", 0.8).name == "SEL-80_HUG_r0.8.json"
        assert R.zs_cell_file("SEL-80", "HUG", 0.8).parent.name == "s24_3zs_cells"

    def test_the_zero_shot_join_refuses_an_absent_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        _zs_on_disk(tmp_path, rows=R.ZS_ROWS[:-1])
        with pytest.raises(SystemExit, match="absent"):
            R.load_zs_records()

    def test_the_zero_shot_join_refuses_a_missing_ratio(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        _zs_on_disk(tmp_path, ratios=(0.8,))
        with pytest.raises(SystemExit, match="absent"):
            R.load_zs_records()

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 2_350_000}, "seed0"),
            ({"verdict": "UNDETERMINED", "n_episodes": 0}, "UNDETERMINED"),
            ({"ratio": 1.2}, "ratio"),  # a 1.2 record filed as a 0.8 comparator
        ],
    )
    def test_the_zero_shot_join_refuses_a_non_measurable_or_wrong_ratio_record(
        self, tmp_path, monkeypatch, overrides, match
    ):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        _zs_on_disk(tmp_path)
        p = tmp_path / "s24_3zs_cells" / "SEL-81_HUG_r0.8.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | overrides))
        with pytest.raises(SystemExit, match=match):
            R.load_zs_records()

    def test_the_zero_shot_join_refuses_an_unreadable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "OUT", tmp_path)
        _zs_on_disk(tmp_path)
        (tmp_path / "s24_3zs_cells" / "SEL-80_GREEDY_r1.0.json").write_text("{not json")
        with pytest.raises(SystemExit, match="unreadable"):
            R.load_zs_records()

    def test_the_1_2_join_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="REFUSED"):
            R.load_base_records()  # an empty S24.1 cells dir

    @pytest.mark.parametrize("join", ["load_zs_records", "load_base_records", "load_joined"])
    def test_a_join_refuses_to_run_inside_a_binding(self, join):
        """With the globals rebound, `record_status` and `cell_file` would be
        THIS round's: the comparator column would be checked against the wrong
        rule and read from the wrong directory."""
        with R._bound(), pytest.raises(SystemExit, match="may not run inside a binding"):
            getattr(R, join)()
        with R._bound(0.8), pytest.raises(SystemExit, match="inside a binding"):
            getattr(R, join)()

    def test_nothing_in_a_join_writes_to_the_frozen_directories(self):
        src = _GATE_PATH.read_text()
        body = src.split("def load_zs_records")[1].split("def load_joined")[0]
        for forbidden in ("write_text", "_write_json", "mkdir", "unlink", "replace"):
            assert forbidden not in body, forbidden


# ----------------------------------------------------------------------------
# record validity and the stop rule
# ----------------------------------------------------------------------------
class TestRecordValidity:
    def test_a_measurable_record(self):
        assert R.record_status(_measurable("SELr0.8-80_HUG_r0.8")) == ("MEASURABLE", "")

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [({"n_episodes": 20}, "n_episodes"), ({"seed0": 7}, "seed0"), ({"ratio": 1.0}, "ratio")],
    )
    def test_the_wrong_n_stream_or_ratio_is_invalid(self, overrides, match):
        rec = _measurable("SELr0.8-80_HUG_r0.8") | overrides
        status, why = R.record_status(rec)
        assert status == "INVALID" and match in why

    def test_a_record_of_another_round_is_invalid(self):
        assert R.record_status({"name": "SEL-80_HUG_r0.8"})[0] == "INVALID"

    def test_a_crash_through_run_cells_writes_this_rounds_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        name = "SELr1.0-81_HUG_r1.0"
        monkeypatch.setattr(R, "cg_z1", lambda *a, **k: {"verdict": "PASS"})
        monkeypatch.setattr(Z, "cg_z1", lambda *a, **k: {"verdict": "PASS"})
        monkeypatch.setattr(
            S1, "run_cell", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setitem(Z._ORIG, "run_cell", S1.run_cell)
        monkeypatch.setattr(S1, "require_ckpt", lambda row: tmp_path)
        R.run_cells([name])
        rec = json.loads((tmp_path / "s24_3pr_cells" / f"{name}.json").read_text())
        assert rec["verdict"] == "UNDETERMINED" and rec["ratio"] == 1.0
        assert "boom" in rec["error"]
        assert rec["provenance"]["spec"] == R.PREREG
        assert R.record_status(rec)[0] == "UNDETERMINED"
        assert R.load_records(quiet=True) == {}

    def test_a_duplicated_cell_rejects_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        rec = _measurable("SELr0.8-80_GREEDY_r0.8")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="copy")
        measurable, rejected, _by_cell = R.scan_records()
        assert measurable == {} and len(rejected) == 2

    def test_read_cell_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        name = "SELr0.8-80_GREEDY_r0.8"
        _on_disk(tmp_path, _cell(name, 0.5, n=2))
        with pytest.raises(SystemExit, match="not measurable"):
            R.read_cell(name)


# ----------------------------------------------------------------------------
# H-PR-1 (section 5) — with its failing directions
# ----------------------------------------------------------------------------
class TestHPr1:
    def test_it_passes_when_every_trained_seed_beats_its_zero_shot_twin(self):
        records, joined = _round_records(pr=0.70, zs=0.60)
        v = R.h_pr_1(records | joined)
        assert v["verdict"] == "PASS" and v["failing_ratios"] == []
        assert list(v["ratios"]) == ["r0.8", "r1.0"]
        assert v["headline_ratio"] == 0.8
        block = v["ratios"]["r0.8"]["seeds"]["81"]
        assert block["row"] == "SELr0.8-81" and block["comparator"] == "SEL-81"
        assert block["n_pass"] == 3 and block["ratio"] == 0.8
        for o, x in block["opponents"].items():
            assert x["direction"] == "SELr0.8-81 > SEL-81", o
            assert x["p"] < R.P_GATE

    def test_an_inverted_direction_fails_and_the_direction_is_reported(self):
        records, joined = _round_records(pr=0.40, zs=0.60)
        v = R.h_pr_1(records | joined)
        assert v["verdict"] == "FAIL"
        assert v["failing_ratios"] == ["r0.8", "r1.0"]
        x = v["ratios"]["r0.8"]["seeds"]["80"]["opponents"]["HUG"]
        assert x["verdict"] == "FAIL" and x["delta"] < 0
        assert x["direction"] == "SELr0.8-80 < SEL-80"

    def test_the_rounds_verdict_is_the_headline_ratios(self):
        """Reading 2: section 5 names a pass at 0.8 alone as the headline."""
        records, joined = _round_records(pr=0.70, zs=0.60)
        all_r = records | joined
        for o in ALL:
            for s in R.SEEDS:
                name = R.cell_name(R.pr_row(1.0, s), o, 1.0)
                all_r[name] = _cell(name, 0.30, CHASEY)
        v = R.h_pr_1(all_r)
        assert v["ratios"]["r0.8"]["verdict"] == "PASS"
        assert v["ratios"]["r1.0"]["verdict"] == "FAIL"
        assert v["verdict"] == "PASS" and v["failing_ratios"] == ["r1.0"]

    def test_a_missing_cell_is_incomplete_not_a_fail(self):
        records, joined = _round_records()
        del records[R.cell_name("SELr0.8-80", "HUG", 0.8)]
        v = R.h_pr_1(records | joined)
        assert v["verdict"] == "INCOMPLETE" and v["incomplete_ratios"] == ["r0.8"]

    def test_a_missing_zero_shot_comparator_is_incomplete(self):
        records, joined = _round_records()
        del joined[R.cell_name("SEL-82", "GREEDY", 1.0)]
        v = R.h_pr_1(records | joined)
        assert v["ratios"]["r1.0"]["verdict"] == "INCOMPLETE"

    def test_two_of_three_opponents_and_two_of_three_seeds_is_the_bar(self):
        records, joined = _round_records(pr=0.70, zs=0.60)
        all_r = records | joined
        # seed 80 loses one opponent (2/3 still passes), seed 81 loses two
        all_r[R.cell_name("SELr0.8-80", "HUG", 0.8)] = _cell(
            R.cell_name("SELr0.8-80", "HUG", 0.8), 0.60, CHASEY
        )
        for o in ("HUG", "GREEDY"):
            n = R.cell_name("SELr0.8-81", o, 0.8)
            all_r[n] = _cell(n, 0.60, CHASEY)
        block = R.h_pr_1(all_r)["ratios"]["r0.8"]
        assert block["seeds"]["80"]["verdict"] == "PASS"
        assert block["seeds"]["81"]["verdict"] == "FAIL"
        assert block["verdict"] == "PASS" and block["n_pass"] == 2


# ----------------------------------------------------------------------------
# H-PR-2 (section 5) — the profile, with its failing directions and guards
# ----------------------------------------------------------------------------
class TestHPr2:
    def test_a_chase_shift_against_the_1_2_record_passes(self):
        records, joined = _round_records(pr_picks=CHASEY, zs12_picks=PATIENTY)
        v = R.h_pr_2(records | joined)
        assert v["verdict"] == "PASS" and v["headline_ratio"] == 0.8
        b = v["ratios"]["r0.8"]["seeds"]["80"]
        assert b["row"] == "SELr0.8-80" and b["comparator"] == "SEL-80"
        assert b["chase_higher"] is True
        assert b["chase_share"] == pytest.approx(0.8)
        assert b["chase_share_comparator"] == pytest.approx(0.2)
        assert b["chi2"]["p"] < R.P_CHI2
        assert b["pooled_over"] == list(ALL)  # the three opponents, pooled

    def test_the_reversed_direction_fails_even_with_a_significant_chi_square(self):
        records, joined = _round_records(pr_picks=PATIENTY, zs12_picks=CHASEY)
        v = R.h_pr_2(records | joined)
        assert v["verdict"] == "FAIL"
        b = v["ratios"]["r0.8"]["seeds"]["81"]
        assert b["chase_higher"] is False and b["chi2"]["differs"] is True
        assert b["verdict"] == "FAIL"

    def test_an_unmoved_profile_fails_on_the_chi_square(self):
        records, joined = _round_records(pr_picks=dict(EVEN), zs12_picks=dict(EVEN))
        b = R.h_pr_2(records | joined)["ratios"]["r0.8"]["seeds"]["80"]
        assert b["chi2"]["differs"] is False and b["verdict"] == "FAIL"

    def test_a_row_with_no_decisions_is_vacuous_and_never_a_pass(self):
        records, joined = _round_records(pr_picks=NO_DECISIONS)
        b = R.h_pr_2(records | joined)["ratios"]["r0.8"]["seeds"]["80"]
        assert b["verdict"] == "VACUOUS"
        assert "cannot be compared" in b["note"]
        assert R.h_pr_2(records | joined)["verdict"] != "PASS"

    def test_the_same_single_tactic_on_both_sides_is_a_fail_not_a_vacuum(self):
        records, joined = _round_records(pr_picks=ONE_TACTIC, zs12_picks=ONE_TACTIC)
        b = R.h_pr_2(records | joined)["ratios"]["r0.8"]["seeds"]["80"]
        assert b["verdict"] == "FAIL"
        assert "same single tactic" in b["note"]

    def test_one_failing_seed_still_clears_the_two_of_three_bar(self):
        records, joined = _round_records(pr_picks=CHASEY, zs12_picks=PATIENTY)
        all_r = records | joined
        for o in ALL:
            n = R.cell_name("SELr0.8-82", o, 0.8)
            all_r[n] = _cell(n, 0.70, PATIENTY)
        block = R.h_pr_2(all_r)["ratios"]["r0.8"]
        assert block["seeds"]["82"]["verdict"] == "FAIL"
        assert block["verdict"] == "PASS" and block["n_pass"] == 2

    def test_the_1_0_ratio_is_reported_and_gates_nothing(self):
        """Reading 3 / section 5: "The same for 1.0 reported, not gated"."""
        records, joined = _round_records(pr_picks=CHASEY, zs12_picks=PATIENTY)
        all_r = records | joined
        for o in ALL:
            for s in R.SEEDS:
                n = R.cell_name(R.pr_row(1.0, s), o, 1.0)
                all_r[n] = _cell(n, 0.70, PATIENTY)
        v = R.h_pr_2(all_r)
        assert v["ratios"]["r1.0"]["verdict"] == "FAIL"
        assert v["ratios"]["r1.0"]["gated"] is False
        assert v["ratios"]["r0.8"]["gated"] is True
        assert v["verdict"] == "PASS"  # the round's bar is 0.8's

    def test_a_missing_1_2_record_is_incomplete(self):
        records, joined = _round_records()
        del joined[R.cell_name("SEL-80", "HUG", R.BASE_RATIO)]
        b = R.h_pr_2(records | joined)["ratios"]["r0.8"]["seeds"]["80"]
        assert b["verdict"] == "INCOMPLETE"
        assert R.cell_name("SEL-80", "HUG", 1.2) in b["missing"]


# ----------------------------------------------------------------------------
# H-PR-3 (section 5)
# ----------------------------------------------------------------------------
class TestHPr3:
    def test_it_passes_when_every_trained_row_beats_the_law_at_its_ratio(self):
        records, joined = _round_records(pr=0.70, law=0.40)
        v = R.h_pr_3(records | joined)
        assert v["verdict"] == "PASS" and v["comparator"] == "LEAD-1.0"
        b = v["ratios"]["r1.0"]["seeds"]["82"]
        assert b["row"] == "SELr1.0-82" and b["comparator"] == "LEAD-1.0"
        assert b["opponents"]["GREEDY"]["direction"] == "SELr1.0-82 > LEAD-1.0"

    def test_an_inverted_direction_fails(self):
        records, joined = _round_records(pr=0.30, law=0.55)
        v = R.h_pr_3(records | joined)
        assert v["verdict"] == "FAIL"
        assert v["failing_ratios"] == ["r0.8", "r1.0"]

    def test_the_round_needs_both_ratios(self):
        """Reading 2: section 5 names no headline for H-PR-3."""
        records, joined = _round_records(pr=0.70, law=0.40)
        all_r = records | joined
        for o in ALL:
            for s in R.SEEDS:
                n = R.cell_name(R.pr_row(0.8, s), o, 0.8)
                all_r[n] = _cell(n, 0.20, CHASEY)
        v = R.h_pr_3(all_r)
        assert v["ratios"]["r1.0"]["verdict"] == "PASS"
        assert v["verdict"] == "FAIL" and v["failing_ratios"] == ["r0.8"]
        assert "BOTH ratios" in v["rule"]

    def test_a_missing_law_cell_is_incomplete(self):
        records, joined = _round_records()
        del joined[R.cell_name("LEAD-1.0", "TACTIC-V0", 0.8)]
        assert R.h_pr_3(records | joined)["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------
# H-SAFE and the construct gates (sections 5 and 6)
# ----------------------------------------------------------------------------
class TestHSafeAndGates:
    def test_h_safe_holds_on_a_complete_round_with_no_contact(self):
        records, _joined = _round_records()
        v = R.h_safe(records, _probes())
        assert v["verdict"] == "HOLDS" and v["violation_steps"] == 0
        assert v["offenders"] == [] and v["missing"] == []
        assert v["gated_cells"] == list(R.CELL_ORDER) and len(v["gated_cells"]) == 18
        assert v["missing_training"] == []
        assert set(v["training"]) == set(R.ROW_NAMES)

    def test_one_contact_step_in_a_gate_cell_fails_the_round(self):
        records, _joined = _round_records()
        name = R.CELL_ORDER[4]
        records[name] = records[name] | {"violation_steps": 1}
        v = R.h_safe(records, _probes())
        assert v["verdict"] == "FAILS" and v["offenders"] == [name]

    def test_training_contact_fails_it_too(self):
        records, _joined = _round_records()
        probes = _probes()
        probes["SELr1.0-80"]["rows"][0]["train_violations"] = 2
        v = R.h_safe(records, probes)
        assert v["verdict"] == "FAILS" and v["training_offenders"] == ["SELr1.0-80"]

    def test_absent_training_evidence_is_an_offender_never_a_zero(self):
        """S24.1's A3.1, inherited (section 6)."""
        records, _joined = _round_records()
        probes = _probes(**{"SELr0.8-80": {"path": "x", "missing": True, "rows": []}})
        v = R.h_safe(records, probes)
        assert v["verdict"] == "INCOMPLETE"
        assert v["missing_training"] == ["SELr0.8-80"]
        gate = R.cg_inherited(records, probes)
        assert gate["verdict"] == "REFUSE"
        assert "SELr0.8-80" in gate["cg_t1"]["absent_training_evidence"]

    def test_an_incomplete_round_is_incomplete_not_holds(self):
        records, _joined = _round_records()
        del records[R.CELL_ORDER[0]]
        assert R.h_safe(records, _probes())["verdict"] == "INCOMPLETE"

    def test_cg_t1_refuses_on_a_training_pick_and_on_a_cell_pick(self):
        records, _joined = _round_records()
        probes = _probes()
        probes["SELr0.8-82"]["rows"][0]["train_inapplicable_picks"] = 1
        assert R.cg_inherited(records, probes)["verdict"] == "REFUSE"
        probes = _probes()
        name = R.CELL_ORDER[0]
        records[name] = records[name] | {"inapplicable_picks": 3}
        gate = R.cg_inherited(records, probes)
        assert gate["verdict"] == "REFUSE" and name in gate["cg_t1"]["offenders"]

    def test_a2_1_is_inherited_at_a5s_tolerance_and_names_every_non_zero(self):
        records, _joined = _round_records()
        name = R.CELL_ORDER[1]
        records[name] = records[name] | {"search_policy_steps": 1}  # 150 000 env steps
        gate = R.cg_inherited(records, _probes())
        assert gate["verdict"] == "PASS"  # <= 1 per 100 000 env steps (A5)
        assert name in gate["cg_a2_1"]["non_zero"]
        records[name] = records[name] | {"search_policy_steps": 99}
        gate = R.cg_inherited(records, _probes())
        assert gate["verdict"] == "REFUSE" and name in gate["cg_a2_1"]["offenders"]

    def test_cg_p1_is_re_read_from_the_arm_records(self):
        v = R.cg_p1_round(_arms())
        assert v["verdict"] == "PASS" and v["n_checked"] == 6
        assert v["offenders"] == [] and set(v["blocks"]) == set(R.ROW_NAMES)

    def test_cg_p1_refuses_an_absent_block(self):
        arms = _arms()
        arms["SELr1.0-82"]["record"].pop("cg_p1")
        v = R.cg_p1_round(arms)
        assert v["verdict"] == "REFUSE" and v["absent_evidence"] == ["SELr1.0-82"]
        arms = _arms()
        arms["SELr0.8-80"] = {"path": "x", "missing": True}
        assert R.cg_p1_round(arms)["absent_evidence"] == ["SELr0.8-80"]

    @pytest.mark.parametrize(
        "mutate",
        [
            {"verdict": "FAIL"},
            {"config_diff_vs_twin": ["evader_speed_ratio", "max_steps"]},
            {"cfg_ratio": 1.2},
            {"obs_ratio": 1.2},
        ],
    )
    def test_cg_p1_refuses_a_block_that_did_not_hold(self, mutate):
        arms = _arms()
        arms["SELr0.8-81"]["record"]["cg_p1"] |= mutate
        v = R.cg_p1_round(arms)
        assert v["verdict"] == "REFUSE" and v["failed"] == ["SELr0.8-81"]

    def test_cg_p1_refuses_an_arm_record_whose_ratio_disagrees(self):
        arms = _arms()
        arms["SELr1.0-80"]["record"]["ratio"] = 0.8
        assert R.cg_p1_round(arms)["failed"] == ["SELr1.0-80"]

    def test_cg_p1_reads_the_arm_files_this_round_writes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "OUT", tmp_path)
        assert R.cg_p1_round()["absent_evidence"] == list(R.ROW_NAMES)
        for arm in R.ROW_NAMES:
            TR._write_json(
                TR.arm_file(arm),
                {"ratio": R.row_ratio(arm), "cg_p1": _arms()[arm]["record"]["cg_p1"]},
            )
        assert R.cg_p1_round()["verdict"] == "PASS"

    def test_cg_z1_round_reads_the_cells_own_gate_blocks(self):
        records, joined = _round_records()
        v = R.cg_z1_round(records, joined)
        assert v["verdict"] == "PASS" and v["n_checked"] == 18
        name = R.CELL_ORDER[2]
        broken = dict(records, **{name: records[name] | {"cg_z1": None}})
        assert R.cg_z1_round(broken, joined)["verdict"] == "REFUSE"
        wrong = dict(
            records,
            **{name: records[name] | {"cg_z1": records[name]["cg_z1"] | {"cfg_ratio": 1.2}}},
        )
        assert R.cg_z1_round(wrong, joined)["verdict"] == "REFUSE"


# ----------------------------------------------------------------------------
# the refutation paragraph (section 5)
# ----------------------------------------------------------------------------
class TestDirection:
    def test_it_states_the_round_when_everything_passes(self):
        text = R.direction(_v())
        assert "H-PR-1 (round, ratio 0.8) PASS" in text
        assert "H-PR-2 (round, ratio 0.8) PASS" in text
        assert "H-PR-3 (round) PASS" in text
        assert "the zero-shot SEL-81" in text
        assert "paired by seed at the same ratio" in text
        assert "in-distribution" in text

    def test_h_pr_1_failing_at_0_8_emits_section_5s_refutation(self):
        records, joined = _round_records(pr=0.40, zs=0.60)
        text = R.direction(_v(records, joined))
        assert "SEEING THE RATIO DOES NOT HELP" in text
        assert "graceful transfer (D152)" in text
        assert "the ratio input is inert" in text

    def test_h_pr_1_passing_and_h_pr_2_failing_emits_the_other_refutation(self):
        records, joined = _round_records(pr_picks=PATIENTY, zs12_picks=CHASEY)
        v = _v(records, joined)
        assert v["h_pr_1"]["verdict"] == "PASS" and v["h_pr_2"]["verdict"] == "FAIL"
        text = R.direction(v)
        assert "OTHER THAN CHASING MORE" in text
        assert "the mechanism stays open" in text
        assert "SEEING THE RATIO DOES NOT HELP" not in text

    def test_the_precedence_is_safety_then_gates_then_completeness(self):
        records, joined = _round_records()
        name = R.CELL_ORDER[0]
        hurt = dict(records, **{name: records[name] | {"violation_steps": 1}})
        assert "H-SAFE FAILS" in R.direction(_v(hurt, joined))
        gated = dict(records, **{name: records[name] | {"inapplicable_picks": 2}})
        assert "REFUSES" in R.direction(_v(gated, joined))
        arms = _arms()
        arms["SELr0.8-80"]["record"].pop("cg_p1")
        assert "CG-P1" in R.direction(_v(records, joined, arms=arms))
        del records[name]
        assert "incomplete" in R.direction(_v(records, joined))


# ----------------------------------------------------------------------------
# CG-P3 (prereg section 6): the watched episodes
# ----------------------------------------------------------------------------
class TestGifs:
    def test_the_seeds_are_the_first_three_gate_seeds_and_a_rule(self):
        assert R.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)
        assert R.GIF_N == 3
        assert R.gif_seeds() == tuple(R.GATE_SEED0 + i for i in range(R.GIF_N))
        assert R.gif_seeds is S1.gif_seeds  # no scan may choose the episodes

    def test_twelve_gifs_two_rows_two_opponents_three_seeds(self):
        assert R.GIF_ROWS == ("SELr0.8-81", "SELr1.0-81")  # section 6, CG-P3
        assert R.GIF_OPPONENTS == ("TACTIC-V0", "GREEDY")
        assert len(R.GIF_ROWS) * len(R.GIF_OPPONENTS) * R.GIF_N == 12
        assert all(row in R.ROW_NAMES for row in R.GIF_ROWS)
        assert {R.row_ratio(r) for r in R.GIF_ROWS} == {0.8, 1.0}

    def test_the_caption_carries_no_measured_number(self):
        title = R.gif_title("SELr0.8-81", "GREEDY", 2_400_001)
        assert title == "SELr0.8-81 vs GREEDY · ratio 0.8 · episode 2400001"
        rec = _measurable("SELr0.8-81_GREEDY_r0.8")
        for forbidden in (
            "capture", "switch", "pick", "measured", "N=", "PASS", "FAIL",
            f"{rec['capture_rate']:.3f}",
        ):
            assert forbidden not in title
        assert R.gif_title("SELr1.0-81", "TACTIC-V0", 2_400_000).endswith("ratio 1.0 · episode 2400000")

    def test_the_frame_label_is_the_v0b_games_own_two_role_caption(self):
        env = R.build_cell_env(
            "SELr0.8-81_TACTIC-V0_r0.8",
            recorder=True,
            label=R.gif_title("SELr0.8-81", "TACTIC-V0", 1),
        )
        assert re.fullmatch(r"EVADER: .+ \| PURSUER: .+", env.game.caption("law"))
        for forbidden in ("capture", "rate", "PASS"):
            assert forbidden not in env.game.caption("law")

    def test_it_refuses_without_a_measurable_record_for_every_watched_cell(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        monkeypatch.setattr(
            R, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered")
        )
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs()
        # ... and a half-covered set still refuses, before the first frame
        for o in R.GIF_OPPONENTS:
            _on_disk(tmp_path, _measurable(R.cell_name("SELr0.8-81", o, 0.8)))
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs()

    def test_it_refuses_without_the_rows_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)
        monkeypatch.setattr(TR, "OUT", tmp_path)
        for row in R.GIF_ROWS:
            for o in R.GIF_OPPONENTS:
                _on_disk(tmp_path, _measurable(R.cell_name(row, o, R.row_ratio(row))))
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("rendered"))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.write_gifs()

    def test_an_unknown_row_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown CG-P3 row"):
            R.write_gifs(["SELr0.8-80"])

    def test_existing_gifs_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.gifs_dir().mkdir(parents=True)
        for row in R.GIF_ROWS:
            for o in R.GIF_OPPONENTS:
                for s in R.gif_seeds():
                    R.gif_path(row, o, s).write_bytes(b"")
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("re-rendered"))
        assert R.write_gifs() == []


# ----------------------------------------------------------------------------
# CLI dispatch order and the summary's refusals
# ----------------------------------------------------------------------------
class TestCli:
    def test_gifs_are_written_before_the_table_is_summarised(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(R, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(R, "write_gifs", lambda *a, **k: order.append("gifs"))
        monkeypatch.setattr(
            R,
            "summarize",
            lambda *a, **k: (
                order.append("summarize"),
                {"verdicts": _v(), "direction": "x"},
            )[1],
        )
        monkeypatch.setattr(
            "sys.argv", ["derisk_s24_3pr_gate.py", "--summarize", "--gifs", "--cells"]
        )
        R.main()
        assert order == ["cells", "gifs", "summarize"]

    def test_summarize_writes_the_round_artefacts(self, tmp_path, monkeypatch):
        records, joined = _round_records()
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "load_records", lambda **k: records)
        monkeypatch.setattr(R, "load_joined", lambda: joined)
        monkeypatch.setattr(R, "read_probes", _probes)
        monkeypatch.setattr(R, "scan_records", lambda: ({}, {}, {}))
        monkeypatch.setattr(R, "read_arm_records", _arms)
        blob = R.summarize()
        assert blob["verdicts"]["n_records"] == 18
        assert blob["verdicts"]["n_joined"] == len(joined)
        assert blob["verdicts"]["complete"] is True
        assert (tmp_path / "s24_3pr_round.json").exists()
        assert (tmp_path / "s24_3pr_round.png").exists()
        # the 500-long arrays never reach the round blob
        for cell in blob["cells"].values():
            assert not set(cell) & set(R._BULK_KEYS)

    @pytest.mark.parametrize("gate", ["cg_p1", "cg_z1", "cg_inherited"])
    def test_summarize_refuses_while_a_construct_gate_refuses(
        self, tmp_path, monkeypatch, gate
    ):
        records, joined = _round_records()
        probes = _probes()
        arms = _arms()
        if gate == "cg_p1":
            arms["SELr0.8-80"]["record"].pop("cg_p1")
        elif gate == "cg_z1":
            name = R.CELL_ORDER[0]
            records[name] = records[name] | {"cg_z1": None}
        else:
            probes["SELr0.8-80"]["rows"][0]["train_inapplicable_picks"] = 1
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "load_records", lambda **k: records)
        monkeypatch.setattr(R, "load_joined", lambda: joined)
        monkeypatch.setattr(R, "read_probes", lambda: probes)
        monkeypatch.setattr(R, "scan_records", lambda: ({}, {}, {}))
        monkeypatch.setattr(R, "read_arm_records", lambda: arms)
        with pytest.raises(SystemExit, match="REFUSES the summary"):
            R.summarize()
        assert not (tmp_path / "s24_3pr_round.json").exists()

    def test_summarize_refuses_without_training_evidence(self, tmp_path, monkeypatch):
        records, joined = _round_records()
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "load_records", lambda **k: records)
        monkeypatch.setattr(R, "load_joined", lambda: joined)
        monkeypatch.setattr(
            R,
            "read_probes",
            lambda: _probes(
                # CG-T1 is checked BEFORE H-SAFE, so this row carries the mask's
                # counters and is missing only `train_violations` (A3.1).
                **{
                    "SELr1.0-81": {
                        "path": "x",
                        "missing": False,
                        "rows": [
                            {
                                "mark": 1,
                                "train_inapplicable_picks": 0,
                                "inapplicable_picks": 0,
                            }
                        ],
                    }
                }
            ),
        )
        monkeypatch.setattr(R, "scan_records", lambda: ({}, {}, {}))
        monkeypatch.setattr(R, "read_arm_records", _arms)
        with pytest.raises(SystemExit, match="H-SAFE REFUSES"):
            R.summarize()
        assert not (tmp_path / "s24_3pr_round.json").exists()

    def test_a_non_measurable_comparator_column_refuses_the_summary(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(Z, "OUT", tmp_path)  # an empty 3zs cells dir
        with pytest.raises(SystemExit, match="zero-shot column"):
            R.summarize()


# ----------------------------------------------------------------------------
# the launch wrappers — RUN against fixtures, with every `nohup` line replaced
# ----------------------------------------------------------------------------
def _fixture(script: pathlib.Path, tmp: pathlib.Path) -> pathlib.Path:
    """The wrapper with its roots pointed at a tmp tree, its `pgrep` guard
    dropped (a pytest command line matching `derisk_s24_...` would trip it) and
    every `nohup` replaced by an echo, so the REFUSALS are exercised and nothing
    can launch."""
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


def _worktree(tmp: pathlib.Path, runner: str) -> None:
    (tmp / "wt" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "wt" / "scripts" / runner).write_text("# frozen\n")
    (tmp / "results" / "derisk").mkdir(parents=True, exist_ok=True)


def _ckpts(tmp: pathlib.Path) -> None:
    for arm in R.ROW_NAMES:
        d = tmp / "results" / "derisk" / f"ckpts_S24-3PR-{arm}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"S24-1-{arm}_3000000.zip").write_bytes(b"")


def _probe_json(tmp: pathlib.Path, **overrides) -> None:
    for arm in R.ROW_NAMES:
        rows = [
            {
                "mark": 3_000_000,
                "ratio": R.row_ratio(arm),
                "inapplicable_picks": 0,
                "train_inapplicable_picks": 0,
                "violations": 0,
                "train_violations": 0,
            }
            | overrides.get(arm, {})
        ]
        (tmp / "results" / "derisk" / f"s24_3pr_probe_{arm}.json").write_text(
            json.dumps({"arm": arm, "checkpoints": rows})
        )


class TestTrainLauncher:
    def test_it_refuses_without_the_frozen_worktree(self, tmp_path):
        got = _run_fixture(_LAUNCH_TRAIN, tmp_path)
        assert got.returncode == 1 and "no frozen worktree" in got.stdout
        assert "LAUNCHED" not in got.stdout

    def test_it_refuses_a_worktree_without_the_trainer(self, tmp_path):
        (tmp_path / "wt" / "scripts").mkdir(parents=True)
        got = _run_fixture(_LAUNCH_TRAIN, tmp_path)
        assert got.returncode == 1 and "without the trainer" in got.stdout

    def test_it_refuses_an_existing_checkpoint_dir(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_train.py")
        (tmp_path / "results" / "derisk" / "ckpts_S24-3PR-SELr1.0-80").mkdir(parents=True)
        got = _run_fixture(_LAUNCH_TRAIN, tmp_path)
        assert got.returncode == 1
        assert "ckpts_S24-3PR-SELr1.0-80 exists" in got.stdout
        assert "never re-run an arm" in got.stdout

    def test_it_refuses_an_existing_arm_record(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_train.py")
        (tmp_path / "results" / "derisk" / "s24_3pr_arm_SELr0.8-82.json").write_text("{}")
        got = _run_fixture(_LAUNCH_TRAIN, tmp_path)
        assert got.returncode == 1 and "s24_3pr_arm_SELr0.8-82.json exists" in got.stdout

    def test_it_starts_the_six_arms_on_a_clean_tree(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_train.py")
        got = _run_fixture(_LAUNCH_TRAIN, tmp_path)
        assert got.returncode == 0, got.stdout + got.stderr
        for arm in R.ROW_NAMES:
            assert f"started {arm}" in got.stdout
        assert got.stdout.count("LAUNCHED") == 6

    def test_the_wrapper_gates_the_host_the_symlink_and_the_threads(self):
        src = _LAUNCH_TRAIN.read_text()
        for needed in (
            "REFUSED: another round is using the host",
            "derisk_s2[34]_",
            "is not a symlink to",
            "OMP_NUM_THREADS=3",
            "git rev-parse --short HEAD",
            "confirm by results/derisk/ckpts_S24-3PR-*/ checkpoints, not by PIDs",
        ):
            assert needed in src, needed
        assert "ARMS=(SELr0.8-80 SELr0.8-81 SELr0.8-82 SELr1.0-80 SELr1.0-81 SELr1.0-82)" in src
        for line in src.splitlines():
            if line.lstrip().startswith("nohup"):
                assert "--arms $a" in line and "scripts/derisk_s24_3pr_train.py" in line


class TestGateLauncher:
    def test_it_refuses_without_the_frozen_worktree(self, tmp_path):
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "no frozen worktree" in got.stdout

    def test_it_refuses_a_missing_final_checkpoint(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "no final 3M checkpoint" in got.stdout

    def test_it_refuses_a_missing_probe_json(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        _ckpts(tmp_path)
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "no probe JSON for arm" in got.stdout

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ({"train_violations": 1}, "train_violations = 1"),
            ({"inapplicable_picks": 2}, "inapplicable_picks = 2"),
            ({"ratio": 1.2}, "ratio = 1.2"),
        ],
    )
    def test_it_refuses_unclean_probe_rows(self, tmp_path, override, match):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        _ckpts(tmp_path)
        _probe_json(tmp_path, **{"SELr0.8-81": override})
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "REFUSED" in got.stdout and match in got.stdout

    def test_it_refuses_a_probe_row_that_does_not_carry_the_counter(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        _ckpts(tmp_path)
        _probe_json(tmp_path)
        p = tmp_path / "results" / "derisk" / "s24_3pr_probe_SELr1.0-82.json"
        blob = json.loads(p.read_text())
        blob["checkpoints"][0].pop("train_violations")
        p.write_text(json.dumps(blob))
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "absent evidence is an offender" in got.stdout

    def test_it_refuses_an_arm_with_no_probe_rows(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        _ckpts(tmp_path)
        _probe_json(tmp_path)
        p = tmp_path / "results" / "derisk" / "s24_3pr_probe_SELr0.8-80.json"
        p.write_text(json.dumps({"arm": "SELr0.8-80", "checkpoints": []}))
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 1 and "no probe rows" in got.stdout

    def test_it_starts_three_chains_on_clean_evidence(self, tmp_path):
        _worktree(tmp_path, "derisk_s24_3pr_gate.py")
        _ckpts(tmp_path)
        _probe_json(tmp_path)
        got = _run_fixture(_LAUNCH_GATE, tmp_path)
        assert got.returncode == 0, got.stdout + got.stderr
        assert "CG-P2 and H-SAFE clean over 6 arms' probe rows" in got.stdout
        assert got.stdout.count("LAUNCHED") == 3

    def test_the_three_chains_cover_every_cell_exactly_once(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH_GATE.read_text().splitlines()
            if "--cells" in line and line.startswith("nohup")
        ]
        assert len(chains) == 3 and all(len(c) == 6 for c in chains)
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(R.CELL_ORDER)
        assert len(set(flat)) == 18

    def test_the_wrapper_never_summarises_or_renders(self):
        """A rate may not be read by the launch: CG-P3 comes first and both are
        run by hand."""
        for line in _LAUNCH_GATE.read_text().splitlines():
            if line.startswith("nohup"):
                assert "--summarize" not in line and "--gifs" not in line
        assert "then: --gifs (CG-P3, 12 GIFs) BEFORE --summarize" in _LAUNCH_GATE.read_text()

    def test_the_wrapper_names_the_trainers_checkpoint_stem(self):
        src = _LAUNCH_GATE.read_text()
        assert "ckpts_S24-3PR-$a/S24-1-${a}_3000000.zip" in src
        assert "OMP_NUM_THREADS=4" in src
        assert "confirm by results/derisk/s24_3pr_cells/*.json, not by PIDs" in src


# ----------------------------------------------------------------------------
# the one end-to-end property the rebinding rests on: S24.1's loop really runs
# ----------------------------------------------------------------------------
class TestSmoke:
    def test_a_thousand_step_maskable_ppo_smoke_at_0_8(self, tmp_path, monkeypatch):
        """NOT a training run: 1 000 steps into a tmp directory, which the
        fixture deletes. What it proves is the path — S24.1's `_Probe` writes
        this round's checkpoint and probe file, the probe row carries the
        `ratio`, and the mask holds through a real rollout at 0.8 (CG-T1)."""
        from sb3_contrib import MaskablePPO

        monkeypatch.setattr(TR, "OUT", tmp_path)
        arm = "SELr0.8-80"
        env = TR.make_env(arm)
        assert env.game.cfg.evader_speed_ratio == 0.8
        model = MaskablePPO(
            "MlpPolicy", env, seed=80, gamma=TR.GAMMA, n_steps=128, batch_size=64, verbose=0
        )
        model.learn(total_timesteps=1000)
        assert model.num_timesteps >= 1000
        assert env.inapplicable_picks == 0  # CG-T1 through a real rollout at 0.8
        assert env.violations_total == 0  # H-SAFE through a real rollout at 0.8
        with TR._bound(arm):
            monkeypatch.setattr(T1, "PROBE_N", 2)
            probes = T1._Probe(arm, [])
            probes.at(model, env.env_steps, TR.EVERY, T1.train_evidence(model.get_env()))
        row = probes.rows[0]
        assert row["ratio"] == 0.8 and row["probe_seed0"] == 2_450_000
        assert row["seed0"] == 2_450_000 and row["n"] == 2
        assert row["train_inapplicable_picks"] == 0 and row["train_violations"] == 0
        assert set(row["picks"]) == set(ACTIONS)
        written = json.loads(TR.probe_file(arm).read_text())
        assert written["checkpoints"][0]["mark"] == TR.EVERY
        assert written["checkpoints"][0]["ckpt"] == str(
            TR.ckpt_dir(arm) / f"S24-1-{arm}_{TR.EVERY}.zip"
        )
        assert TR.ckpt_dir(arm).is_dir()
        # the resume path finds what the probe wrote, under the same binding
        with TR._bound(arm):
            ckpt, mark, rows = T1._resume_from(arm)
        assert mark == TR.EVERY and ckpt.parent == TR.ckpt_dir(arm)
        assert [r["mark"] for r in rows] == [TR.EVERY]
