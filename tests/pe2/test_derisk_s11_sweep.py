"""scripts/derisk_s11_filter.py — the S11 second-filter round's own decisions.

Spec: docs/pathway/second_filter_prereg_2026-09-18.md (+ amendment S11-A1, which
sets the horizon levels to 1, 4, 16, 32).

No test here loads a checkpoint: the enumeration, the verdict rules and the cell
bookkeeping are driven on fabricated records. The two cheap preflight checks
(V-P (i) identity and V-P (iii) brute force) ARE run for real, at a small state
count — they need no model and they are the checks that bind the runner to
`pe2.shield.PredictiveShield` and `pe2.env.pursuer_step`.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s11_filter.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s11_filter", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n):
    """n paired indicators whose first round(rate * n) entries are True, so a
    higher-rate list strictly contains a lower-rate one (every discordant pair
    points the same way and the paired test sees the direction under test)."""
    k = round(rate * n)
    return [i < k for i in range(n)]


def _record(name, deadlock=0.5, violations=0, n_ep=300):
    spec = R.CELLS[name]
    ind = _ind(deadlock, n_ep)
    return {
        "name": name,
        "pursuer": spec["pursuer"],
        "opponent": spec["opponent"],
        "filter": spec["filter"],
        "horizon": spec["horizon"],
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "shield_obs": R.PURSUERS[spec["pursuer"]]["shield_obs"],
        "n_episodes": n_ep,
        "seed0": R.SEED0,
        "deadlock_ind": ind,
        "capture_ind": _ind(0.2, n_ep),
        "deadlock_rate": sum(ind) / n_ep,
        "capture_rate": 0.2,
        "escape_rate": 0.8,
        "outcomes": {"capture": round(0.2 * n_ep), "timeout": n_ep - round(0.2 * n_ep)},
        "stall_step_rate": 0.3,
        "intervention_rate": 0.4,
        "backup_rate": 0.0,
        "violations": violations,
        "min_clearance_floor": 0.01,
        "min_clearance_hard_floor": 0.02,
        "wilson_deadlock": R._ci(sum(ind), n_ep),
    }


def _records(rate_fn, violations_fn=None):
    """A full 68-cell record set whose deadlock rate is rate_fn(spec)."""
    return {
        name: _record(
            name,
            rate_fn(spec),
            0 if violations_fn is None else violations_fn(spec),
        )
        for name, spec in R.CELLS.items()
    }


# Deadlock under a structured attacker is above every scripted probe (FILT-1),
# grows with the horizon (FILT-2) and orders RCM28 < FT-B-41 < RC19 = STR14 at
# every horizon (FILT-3): the whole round CONFIRMED.
_ORDER = {"RCM28": 0.00, "FT-B-41": 0.06, "RC19": 0.12, "STR14": 0.12}


def _all_confirm(spec):
    base = 0.20 if spec["opponent"] in R.STRUCTURED else 0.05
    grow = 0.05 * R.HORIZONS.index(spec["horizon"]) if spec["horizon"] else 0.0
    return base + grow + _ORDER[spec["pursuer"]]


def _flat(spec):
    return 0.30 if spec["opponent"] in R.STRUCTURED else 0.05


# ---------------------------------------------------------------- enumeration
class TestCells:
    def test_sixty_four_predictive_cells_plus_four_cbf_base_cells(self):
        assert len(R.PURSUERS) == 4
        assert len(R.OPPONENTS) == 4
        assert R.HORIZONS == (1, 4, 16, 32)
        predictive = [s for s in R.CELLS.values() if s["filter"] == "predictive"]
        cbf = [s for s in R.CELLS.values() if s["filter"] == "cbf"]
        assert len(predictive) == 64
        assert len(cbf) == 4
        assert len(R.CELLS) == 68

    def test_every_pursuer_opponent_pair_has_all_four_horizons(self):
        for p in R.PURSUERS:
            for o in R.OPPONENTS:
                hs = sorted(
                    s["horizon"]
                    for s in R.CELLS.values()
                    if s["pursuer"] == p
                    and s["opponent"] == o
                    and s["filter"] == "predictive"
                )
                assert hs == sorted(R.HORIZONS)

    def test_cbf_base_cells_are_ft_b_41_only_and_carry_no_horizon(self):
        cbf = [s for s in R.CELLS.values() if s["filter"] == "cbf"]
        assert {s["pursuer"] for s in cbf} == {"FT-B-41"}
        assert {s["opponent"] for s in cbf} == set(R.OPPONENTS)
        for s in cbf:
            assert s["horizon"] is None
            assert (s["delta"], s["n"], s["gamma"]) == (0.0, 2, 0.2)

    def test_predictive_cells_are_delta_zero_two_obstacles(self):
        for s in R.CELLS.values():
            if s["filter"] == "predictive":
                assert (s["delta"], s["n"]) == (0.0, 2)

    def test_cell_names_carry_the_filter_and_the_horizon(self):
        assert (
            R.cell_name("RC19", "ATT25", "predictive", 4)
            == "RC19__ATT25__pred_H4_d0.00_n2"
        )
        assert (
            R.cell_name("FT-B-41", "GREEDY", "cbf")
            == "FT-B-41__GREEDY__cbf_d0.00_n2_g0.2"
        )
        for name, spec in R.CELLS.items():
            assert name == R.cell_name(
                spec["pursuer"], spec["opponent"], spec["filter"], spec["horizon"]
            )

    def test_pre_registered_constants(self):
        assert R.SEED0 == 1_330_000
        assert R.N_EPISODES == 300
        assert R.MAX_WORKERS == 6
        assert (R.DELTA, R.N_OBSTACLES, R.GAMMA) == (0.0, 2, 0.2)
        assert R.H_FILT3 == 4
        assert R.SCORING == ("RC19", "RCM28", "STR14")
        assert R.REPORT_ONLY == "FT-B-41"
        assert R.STRUCTURED == ("ATT25", "ATK2-31")
        assert R.SCRIPTED == ("HUG7", "GREEDY")

    def test_ft_b_41_is_the_five_million_step_checkpoint_with_shield_obs(self):
        spec = R.PURSUERS["FT-B-41"]
        assert spec["shield_obs"] is True
        assert spec["ckpt"].name == "FT-B-41_5000000.zip"
        assert spec["ckpt"].parent.name == "ckpts_FT-B-41"


# ------------------------------------------------------------- shield factory
class TestShieldFactory:
    def test_predictive_cell_builds_a_predictive_shield_at_its_horizon(self):
        from pe2.config import EnvConfig
        from pe2.shield import PredictiveShield

        cfg = EnvConfig()
        name = R.cell_name("RC19", "ATT25", "predictive", 16)
        shield = R.shield_factory(R.CELLS[name], cfg)()
        assert isinstance(shield, PredictiveShield)
        assert shield.horizon == 16
        assert shield.margin == 0.0

    def test_cbf_cell_builds_a_plain_cbf_shield_at_gamma_zero_point_two(self):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield, PredictiveShield

        cfg = EnvConfig()
        name = R.cell_name("FT-B-41", "HUG7", "cbf")
        shield = R.shield_factory(R.CELLS[name], cfg)()
        assert isinstance(shield, CBFShield)
        assert not isinstance(shield, PredictiveShield)
        assert (shield.gamma, shield.margin) == (0.2, 0.0)


# --------------------------------------------------------------------- FILT-1
class TestFilt1:
    def test_attackers_above_the_best_scripted_probe_confirm_every_pursuer(self):
        f = R.verdicts(_records(_all_confirm))["filt1"]
        assert f["verdict"] == "CONFIRMED"
        assert f["n_confirmed"] == 3
        for p in R.SCORING:
            assert f["pursuers"][p]["verdict"] == "CONFIRMED", p
            for h in R.HORIZONS:
                block = f["pursuers"][p]["horizons"][f"H{h}"]
                assert block["verdict"] == "CONFIRMED"
                for a in R.STRUCTURED:
                    assert block["attackers"][a]["pass"] is True

    def test_a_horizon_flat_surface_still_confirms_existence(self):
        # FILT-1 asks only "is the attacker above the best scripted probe"; a
        # surface that does not grow with H (FILT-2's question) still confirms it.
        f = R.verdicts(_records(_flat))["filt1"]
        assert f["verdict"] == "CONFIRMED"

    def test_attackers_at_the_scripted_rate_fail(self):
        f = R.verdicts(_records(lambda spec: 0.30))["filt1"]
        assert f["verdict"] == "NOT CONFIRMED"
        assert f["n_confirmed"] == 0

    def test_one_failing_horizon_fails_the_pursuer(self):
        def rate(spec):
            if (
                spec["pursuer"] == "RC19"
                and spec["opponent"] == "ATT25"
                and spec["horizon"] == 16
            ):
                return 0.05  # down to the scripted level
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt1"]
        assert f["pursuers"]["RC19"]["verdict"] == "NOT CONFIRMED"
        assert f["pursuers"]["RC19"]["horizons"]["H16"]["verdict"] == "NOT CONFIRMED"
        assert f["pursuers"]["RC19"]["horizons"]["H4"]["verdict"] == "CONFIRMED"
        # two of three scoring pursuers still confirm -> overall CONFIRMED
        assert f["verdict"] == "CONFIRMED"
        assert f["n_confirmed"] == 2

    def test_two_failing_pursuers_fail_the_round(self):
        def rate(spec):
            if spec["pursuer"] in ("RC19", "STR14") and spec["opponent"] == "ATT25":
                return 0.05
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt1"]
        assert f["n_confirmed"] == 1
        assert f["verdict"] == "NOT CONFIRMED"

    def test_the_probe_is_the_higher_scripted_rate_of_the_two(self):
        def rate(spec):
            if spec["opponent"] == "GREEDY":
                return 0.25
            if spec["opponent"] == "HUG7":
                return 0.05
            return 0.30

        f = R.verdicts(_records(rate))["filt1"]
        block = f["pursuers"]["RC19"]["horizons"]["H4"]
        assert block["best_scripted_opponent"] == "GREEDY"
        assert block["attackers"]["ATT25"]["rate_b"] == pytest.approx(0.25)

    def test_ft_b_41_is_reported_but_casts_no_vote(self):
        def rate(spec):
            if spec["pursuer"] == "FT-B-41" and spec["opponent"] in R.STRUCTURED:
                return 0.05
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt1"]
        assert f["pursuers"]["FT-B-41"]["scored"] is False
        assert f["pursuers"]["FT-B-41"]["verdict"] == "NOT CONFIRMED"
        assert f["n_confirmed"] == 3
        assert f["verdict"] == "CONFIRMED"

    def test_missing_cells_make_it_incomplete(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RC19", "HUG7", "predictive", 4)]
        f = R.verdicts(recs)["filt1"]
        assert f["pursuers"]["RC19"]["horizons"]["H4"]["verdict"] == "INCOMPLETE"
        assert f["pursuers"]["RC19"]["verdict"] == "INCOMPLETE"
        assert f["verdict"] == "INCOMPLETE"


# --------------------------------------------------------------------- FILT-2
class TestFilt2:
    def test_growth_with_the_horizon_confirms(self):
        f = R.verdicts(_records(_all_confirm))["filt2"]
        assert f["verdict"] == "CONFIRMED"
        assert f["n_confirmed"] == 3
        assert f["levels"] == list(R.HORIZONS)
        cell = f["pursuers"]["RC19"]["opponents"]["ATT25"]
        assert cell["verdict"] == "CONFIRMED"
        assert cell["rho"] > 0
        assert cell["top_vs_bottom"]["test"] == "paired_indicator"
        assert cell["cells"][0].endswith("pred_H1_d0.00_n2")
        assert cell["cells"][-1].endswith("pred_H32_d0.00_n2")

    def test_flat_in_the_horizon_does_not_confirm(self):
        f = R.verdicts(_records(_flat))["filt2"]
        assert f["verdict"] == "NOT CONFIRMED"
        assert f["pursuers"]["RC19"]["opponents"]["ATT25"]["rho"] == 0.0

    def test_both_attackers_are_required(self):
        def rate(spec):
            if spec["opponent"] == "ATK2-31":
                return _flat(spec)
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt2"]
        assert f["verdict"] == "NOT CONFIRMED"
        for p in R.SCORING:
            assert f["pursuers"][p]["verdict"] == "NOT CONFIRMED"
            assert f["pursuers"][p]["opponents"]["ATT25"]["verdict"] == "CONFIRMED"

    def test_two_of_three_scoring_pursuers_carry_the_round(self):
        def rate(spec):
            return _flat(spec) if spec["pursuer"] == "STR14" else _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt2"]
        assert f["verdict"] == "CONFIRMED"
        assert f["n_confirmed"] == 2
        assert f["pursuers"]["STR14"]["verdict"] == "NOT CONFIRMED"

    def test_rho_alone_is_not_enough(self):
        # rho = 1 but H = 32 beats H = 1 by a single episode: the paired test vetoes.
        def rate(spec):
            if spec["opponent"] not in R.STRUCTURED or not spec["horizon"]:
                return 0.05
            return 0.30 + 0.001 * R.HORIZONS.index(spec["horizon"])

        cell = R.verdicts(_records(rate))["filt2"]["pursuers"]["RC19"]["opponents"][
            "ATT25"
        ]
        assert cell["rho"] > 0
        assert cell["top_vs_bottom"]["pass"] is False
        assert cell["verdict"] == "NOT CONFIRMED"

    def test_scripted_opponents_are_reported_without_a_verdict(self):
        f = R.verdicts(_records(_all_confirm))["filt2"]
        for o in R.SCRIPTED:
            cell = f["pursuers"]["RC19"]["opponents"][o]
            assert cell["verdict"] is None
            assert "rho" in cell and "rates" in cell

    def test_missing_cells_make_it_incomplete(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RCM28", "ATT25", "predictive", 32)]
        f = R.verdicts(recs)["filt2"]
        assert f["pursuers"]["RCM28"]["opponents"]["ATT25"]["verdict"] == "INCOMPLETE"
        assert f["verdict"] == "INCOMPLETE"


# --------------------------------------------------------------------- FILT-3
class TestFilt3:
    def test_the_pre_registered_ordering_confirms(self):
        f = R.verdicts(_records(_all_confirm))["filt3"]
        assert f["verdict"] == "CONFIRMED"
        assert f["horizon"] == 4
        for a in R.STRUCTURED:
            block = f["attackers"][a]
            assert block["verdict"] == "CONFIRMED"
            assert block["tests"]["FT-B-41>RCM28"]["pass"] is True
            assert block["tests"]["RC19>FT-B-41"]["pass"] is True
            assert block["cells"]["RC19"].endswith("pred_H4_d0.00_n2")

    def test_str14_is_reported_and_casts_no_vote(self):
        def rate(spec):
            return 0.9 if spec["pursuer"] == "STR14" else _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt3"]
        assert f["verdict"] == "CONFIRMED"
        assert f["attackers"]["ATT25"]["reported"]["STR14"] == pytest.approx(0.9)

    def test_a_broken_inequality_fails_the_round(self):
        def rate(spec):
            # FT-B-41 above RC19: the upper inequality breaks.
            if spec["pursuer"] == "FT-B-41":
                return _all_confirm(spec) + 0.20
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt3"]
        assert f["verdict"] == "NOT CONFIRMED"
        for a in R.STRUCTURED:
            assert f["attackers"][a]["tests"]["FT-B-41>RCM28"]["pass"] is True
            assert f["attackers"][a]["tests"]["RC19>FT-B-41"]["pass"] is False

    def test_one_attacker_is_enough_to_fail_it(self):
        def rate(spec):
            if spec["pursuer"] == "FT-B-41" and spec["opponent"] == "ATK2-31":
                return _all_confirm(spec) + 0.20
            return _all_confirm(spec)

        f = R.verdicts(_records(rate))["filt3"]
        assert f["attackers"]["ATT25"]["verdict"] == "CONFIRMED"
        assert f["attackers"]["ATK2-31"]["verdict"] == "NOT CONFIRMED"
        assert f["verdict"] == "NOT CONFIRMED"

    def test_missing_cells_make_it_incomplete(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("FT-B-41", "ATT25", "predictive", 4)]
        f = R.verdicts(recs)["filt3"]
        assert f["attackers"]["ATT25"]["verdict"] == "INCOMPLETE"
        assert f["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------ CBF base
class TestCbfBase:
    def test_the_four_cbf_cells_are_reported_without_a_verdict(self):
        base = R.verdicts(_records(_all_confirm))["cbf_base"]
        assert set(base["rates"]) == set(R.OPPONENTS)
        assert "verdict" not in base


# ---------------------------------------------------------------------- H-SAFE
class TestHSafe:
    def test_holds_on_a_clean_full_round(self):
        h = R.verdicts(_records(_all_confirm))["hsafe"]
        assert h["verdict"] == "HOLDS"
        assert h["violation_steps"] == 0
        assert h["offenders"] == []

    def test_fails_and_names_the_offending_cell(self):
        bad = R.cell_name("STR14", "GREEDY", "predictive", 32)

        def viol(spec):
            return 3 if spec["name"] == bad else 0

        h = R.verdicts(_records(_all_confirm, violations_fn=viol))["hsafe"]
        assert h["verdict"] == "FAILS"
        assert h["violation_steps"] == 3
        assert h["offenders"] == [bad]

    def test_incomplete_when_cells_are_missing_and_none_offend(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RC19", "ATT25", "predictive", 1)]
        assert R.verdicts(recs)["hsafe"]["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------- cell bookkeeping
def _pass_preflight(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "s11_preflight.json").write_text(json.dumps({"pass": True}))


class TestRunCells:
    def test_skips_cells_that_already_have_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_preflight(tmp_path)
        done = R.cell_name("RC19", "ATT25", "predictive", 1)
        todo = R.cell_name("RC19", "ATT25", "predictive", 4)
        R.cells_dir().mkdir(parents=True)
        (R.cells_dir() / f"{done}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells([done, todo], workers=1)
        assert called == [todo]
        assert json.loads((R.cells_dir() / f"{todo}.json").read_text())["name"] == todo
        assert (R.cells_dir() / f"{done}.json").read_text() == "{}"

    def test_no_names_runs_every_missing_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_preflight(tmp_path)
        R.cells_dir().mkdir(parents=True)
        done = list(R.CELLS)[:5]
        for name in done:
            (R.cells_dir() / f"{name}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells(None, workers=1)
        assert len(called) == len(R.CELLS) - 5
        assert set(called).isdisjoint(done)

    def test_unknown_cell_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_preflight(tmp_path)
        with pytest.raises(SystemExit):
            R.run_cells(["NOPE"], workers=1)

    def test_cells_refuse_to_run_without_a_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "run_cell", lambda name, device="cpu": None)
        with pytest.raises(SystemExit, match="preflight"):
            R.run_cells(None, workers=1)

    def test_cells_refuse_to_run_on_a_failed_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "s11_preflight.json").write_text(
            json.dumps({"pass": False, "checks": {}})
        )
        monkeypatch.setattr(R, "run_cell", lambda name, device="cpu": None)
        with pytest.raises(SystemExit, match="preflight"):
            R.run_cells(None, workers=1)

    def test_records_are_written_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        name = R.cell_name("RC19", "ATT25", "predictive", 4)
        R.cells_dir().mkdir(parents=True)
        R.write_cell(_record(name))
        assert json.loads((R.cells_dir() / f"{name}.json").read_text())["name"] == name
        assert list(R.cells_dir().glob("*.tmp")) == []


# -------------------------------------------------------------------- preflight
class TestPreflight:
    def test_identity_at_horizon_zero_holds(self):
        out = R.preflight_identity(n_states=40)
        assert out["pass"] is True
        assert out["n_states"] == 40
        assert out["mismatches"] == 0
        assert out["horizon"] == 0
        # the only change H = 0 may make is the float32 action-grid snap
        assert out["max_deviation_from_clip"] <= R.IDENTITY_TOL

    def test_the_identity_reference_is_the_shields_own_snap(self):
        """V-P (i)'s reference is transcribed in the runner from the env-side
        functions, NOT read from `PredictiveShield._snap`, so the check cannot
        pass by agreeing with itself. This pins the two together: they must be
        bit-identical, and if the filter's snap convention ever changes, this
        fires here rather than silently failing the round's preflight."""
        import numpy as np

        from pe2.config import EnvConfig
        from pe2.shield import PredictiveShield

        cfg = EnvConfig()
        shield = PredictiveShield(cfg, horizon=0, margin=0.0)
        rng = np.random.default_rng(11)
        for _ in range(2000):
            action = (
                float(rng.uniform(-0.5, cfg.v_max + 0.5)),
                float(rng.uniform(-cfg.omega_max - 1.0, cfg.omega_max + 1.0)),
            )
            clipped = shield._clip_nominal(action, cfg.v_max)
            assert R._executed_nominal(action, cfg, cfg.v_max) == shield._snap(
                *clipped, cfg.v_max
            )

    def test_brute_force_accept_decisions_agree(self):
        out = R.preflight_bruteforce(n_pairs=40)
        assert out["pass"] is True
        assert out["n_pairs"] == 40
        assert out["disagreements"] == 0
        # the sample must exercise both answers, or it proves nothing
        assert 0 < out["n_accepted"] < 40
        assert out["n_aimed_at_surface"] > 0
        assert sorted(out["horizons"]) == sorted(R.HORIZONS)

    def test_run_preflight_writes_all_three_checks_and_the_verdict(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R,
            "preflight_guarantee",
            lambda n_episodes=200, device="cpu": {
                "pass": True,
                "n_episodes": n_episodes,
                "violation_steps": 0,
            },
        )
        blob = R.run_preflight(n_states=20, n_pairs=20)
        assert blob["pass"] is True
        assert set(blob["checks"]) == {"identity", "guarantee", "bruteforce"}
        on_disk = json.loads((tmp_path / "s11_preflight.json").read_text())
        assert on_disk["pass"] is True
        assert on_disk["seed0"] == R.PREFLIGHT_SEED0
        assert "provenance" in on_disk
        assert R.require_preflight()["pass"] is True

    def test_a_failing_check_fails_the_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R,
            "preflight_guarantee",
            lambda n_episodes=200, device="cpu": {
                "pass": False,
                "violation_steps": 2,
            },
        )
        blob = R.run_preflight(n_states=20, n_pairs=20)
        assert blob["pass"] is False
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()


# -------------------------------------------------------------------- summarize
class TestSummarize:
    def test_writes_round_json_and_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        for rec in _records(_all_confirm).values():
            R.write_cell(rec)
        summary = R.summarize()
        assert (tmp_path / "s11_round.json").exists()
        assert (tmp_path / "s11_round.png").exists()
        blob = json.loads((tmp_path / "s11_round.json").read_text())
        assert blob["verdicts"]["filt1"]["verdict"] == "CONFIRMED"
        assert blob["verdicts"]["filt2"]["verdict"] == "CONFIRMED"
        assert blob["verdicts"]["filt3"]["verdict"] == "CONFIRMED"
        assert summary["verdicts"]["hsafe"]["verdict"] == "HOLDS"
        assert len(blob["cells"]) == 68
        assert "deadlock_ind" not in next(iter(blob["cells"].values()))

    def test_a_partial_round_still_draws(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        recs = _records(_all_confirm)
        for name in list(recs)[:6]:
            R.write_cell(recs[name])
        blob = R.summarize()
        assert (tmp_path / "s11_round.png").exists()
        assert blob["verdicts"]["complete"] is False
