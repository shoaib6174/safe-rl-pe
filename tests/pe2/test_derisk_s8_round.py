"""scripts/derisk_s8_remedy3.py — the third remedy round's own decisions."""

import importlib.util
import json
import pathlib
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s8_remedy3.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s8_remedy3", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _gate(deadlock, capture):
    return {
        "deadlock_ind": _ind(deadlock),
        "capture_ind": _ind(capture),
        "pursuer_violations": 0,
        "pursuer": {
            "deadlock_episode_rate": deadlock,
            "stall_step_rate": 0.3,
            "intervention_rate": 0.4,
        },
        "capture_rate": capture,
    }


def _starts(att=(0.63, 0.62, 0.62), greedy=0.55):
    return {
        "RC19": {
            "ATT25": _gate(att[0], 0.12),
            "ATT26": _gate(att[1], 0.12),
            "ATT27": _gate(att[2], 0.10),
            "GREEDY": _gate(0.33, greedy),
            "HUG7": _gate(0.33, 0.41),
        },
        "RCM28": {
            "ATT25": _gate(0.10, 0.10),
            "ATT26": _gate(0.09, 0.09),
            "ATT27": _gate(0.10, 0.10),
            "GREEDY": _gate(0.04, 0.13),
            "HUG7": _gate(0.05, 0.15),
        },
    }


def _arm(name, att, greedy_capture):
    return {
        "arm": name,
        "seed": R.ARMS[name]["seed"],
        "family": R.ARMS[name]["family"],
        "train_violations": 0,
        "probe_violations": 0,
        "final": {
            "pursuer_violations": 0,
            "gate": {
                "ATT25": _gate(att[0], 0.1),
                "ATT26": _gate(att[1], 0.1),
                "ATT27": _gate(att[2], 0.1),
                "GREEDY": _gate(0.2, greedy_capture),
                "HUG7": _gate(0.2, 0.35),
            },
        },
        "checkpoints": [],
    }


def _full(
    primary=((0.30, 0.32, 0.31), (0.28, 0.30, 0.29), (0.60, 0.61, 0.62)),
    primary_cap=(0.54, 0.53, 0.55),
    secondary=((0.20, 0.22, 0.21),) * 3,
    secondary_cap=(0.30, 0.31, 0.29),
    ctl=(0.10, 0.09, 0.10),
    ctl_cap=0.13,
):
    r = {}
    for s, a, c in zip(R.SEEDS, primary, primary_cap, strict=True):
        r[f"FT-A1-{s}"] = _arm(f"FT-A1-{s}", a, c)
    for s, a, c in zip(R.SEEDS, secondary, secondary_cap, strict=True):
        r[f"FT-A2-{s}"] = _arm(f"FT-A2-{s}", a, c)
    r["SCR-W34"] = _arm("SCR-W34", ctl, ctl_cap)
    return r


class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (34, 35, 36) and R.BETAS == {"FT-A1": 0.1, "FT-A2": 0.01}
        assert R.TOTAL == 5_000_000 and R.EVERY == 500_000
        assert (R.PROBE_SEED0, R.GATE_SEED0) == (1_210_000, 1_200_000)
        assert R.WEIGHTS == pytest.approx((0.5, 0.2, 0.1, 1 / 15, 1 / 15, 1 / 15))
        assert R.MIXTURE == (
            "GREEDY",
            "HUG7",
            "SPLITHUG",
            "CTL-D125",
            "CTL-D126",
            "CTL-D127",
        )
        assert set(R.ARMS) == {
            "FT-A1-34",
            "FT-A1-35",
            "FT-A1-36",
            "FT-A2-34",
            "FT-A2-35",
            "FT-A2-36",
            "SCR-W34",
        }
        assert R.ARMS["FT-A1-34"] == {"seed": 34, "family": "FT-A1", "beta": 0.1}
        assert R.ARMS["FT-A2-36"] == {"seed": 36, "family": "FT-A2", "beta": 0.01}
        assert R.ARMS["SCR-W34"] == {"seed": 34, "family": "SCR", "beta": None}
        assert (
            R.SEEDS2,
            R.PROBE2_SEED0,
            R.GATE2_SEED0,
            R.RACE_BAR,
            R.S7_RACE_RATE,
        ) == ((37, 38, 39), 1_220_000, 1_230_000, 0.58, 0.147)
        assert (R.GATE_N, R.EVAL_N, R.PASS_MIN, R.P_GATE) == (500, 200, 2, 0.05)
        assert R.HELD_OUT == ("ATT25", "ATT26", "ATT27")
        assert R.GATE_OPPONENTS == ("ATT25", "ATT26", "ATT27", "GREEDY", "HUG7")
        assert R.WEIGHTS[0] + R.WEIGHTS[1] + R.WEIGHTS[2] == pytest.approx(0.8)
        assert sum(R.WEIGHTS) == pytest.approx(1.0)

    def test_the_s7_module_is_pointed_at_the_s8_stream(self):
        """S7's `gate_pursuer` reads GATE_SEED0/GATE_N/OUT as its OWN module
        globals, so a reused S7 must be re-pointed at the S8 round."""
        assert R.S7.GATE_SEED0 == R.GATE_SEED0 and R.S7.GATE_N == R.GATE_N
        assert R.S7.OUT == R.OUT

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s8_start.json")
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s8_preflight.json")
        with pytest.raises(SystemExit, match="preflight missing"):
            R.require_preflight()
        (tmp_path / "s8_preflight.json").write_text('{"pass": false}')
        with pytest.raises(SystemExit, match="V-A0"):
            R.require_preflight()
        (tmp_path / "s8_preflight.json").write_text('{"pass": true}')
        R.require_preflight()  # satisfied: no raise

        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        entry = {"deadlock_ind": [], "capture_ind": []}
        one_ref = {"RC19": dict.fromkeys(R.GATE_OPPONENTS, entry)}
        (tmp_path / "s8_start.json").write_text(json.dumps(one_ref))
        with pytest.raises(SystemExit, match="RCM28"):
            R.load_start()
        both = {r: dict.fromkeys(R.GATE_OPPONENTS, entry) for r in ("RC19", "RCM28")}
        (tmp_path / "s8_start.json").write_text(json.dumps(both))
        assert set(R.load_start()) == {"RC19", "RCM28"}
        partial = json.loads(json.dumps(both))
        del partial["RCM28"]["HUG7"]
        (tmp_path / "s8_start.json").write_text(json.dumps(partial))
        with pytest.raises(SystemExit, match="RCM28/HUG7"):
            R.load_start()

        R.require_arm_not_run("FT-A1-34")  # nothing on disk yet
        (tmp_path / "s8_arm_FT-A1-34.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("FT-A1-34")

    def test_stage2_requires_the_stage1_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="run --summarize first"):
            R._artefact_ckpt()
        (tmp_path / "s8_round.json").write_text(
            json.dumps({"summary": {"hypotheses": {"artefact": None}}})
        )
        with pytest.raises(SystemExit, match="no S8 artefact"):
            R._artefact_ckpt()
        (tmp_path / "s8_round.json").write_text(
            json.dumps({"summary": {"hypotheses": {"artefact": "FT-A1-35"}}})
        )
        (tmp_path / "s8_arm_FT-A1-35.json").write_text(
            json.dumps({"final": {"ckpt": str(tmp_path / "ck.zip")}})
        )
        assert R._artefact_ckpt() == tmp_path / "ck.zip"


class TestVerdicts:
    def test_h4_two_sided_primary_passes(self):
        h = R.h4_verdicts(_full(), _starts())
        assert h["per_arm"]["FT-A1-34"]["verdict"] == "PASS"
        assert h["per_arm"]["FT-A1-36"]["verdict"] == "FAIL (still pinned)"
        assert h["per_arm"]["FT-A2-34"]["verdict"] == "FAIL (capture lost)"
        assert h["per_arm"]["SCR-W34"]["verdict"] == "FAIL (capture lost)"
        # lowest mean held-out deadlock among the passing primaries
        assert h["H4"] == "PASS" and h["artefact"] == "FT-A1-35"

    def test_secondary_rescue(self):
        h = R.h4_verdicts(
            _full(
                primary_cap=(0.30, 0.30, 0.30),
                secondary=((0.30, 0.31, 0.32),) * 3,
                secondary_cap=(0.54, 0.54, 0.54),
            ),
            _starts(),
        )
        assert h["H4"] == "PASS (β = 0.01)" and h["artefact"].startswith("FT-A2-")

    def test_fail_and_incomplete(self):
        assert R.h4_verdicts(_full(primary_cap=(0.3, 0.3, 0.3)), _starts())["H4"] == (
            "FAIL"
        )
        res = _full()
        del res["FT-A1-35"]
        assert R.h4_verdicts(res, _starts())["H4"] == "INCOMPLETE"
        assert R.h4_verdicts(res, _starts())["artefact"] is None

    def test_artefact_tiebreak_capture_then_seed(self):
        h = R.h4_verdicts(
            _full(primary=((0.30, 0.30, 0.30),) * 3, primary_cap=(0.53, 0.55, 0.55)),
            _starts(),
        )
        assert h["artefact"] == "FT-A1-35"

    def test_fail_flavours_and_fail_mode(self):
        """A both-clauses failure is labelled as such, and the round's fail_mode
        is the primary family's majority flavour (spec section 6's two rows)."""
        h = R.h4_verdicts(
            _full(primary=((0.60, 0.61, 0.62),) * 3, primary_cap=(0.30, 0.30, 0.30)),
            _starts(),
        )
        assert h["per_arm"]["FT-A1-34"]["verdict"] == "FAIL (both)"
        h2 = R.h4_verdicts(_full(primary_cap=(0.30, 0.30, 0.55)), _starts())
        # seeds 34/35 keep their deadlock win but lose capture; 36 is still pinned
        assert h2["H4"] == "FAIL" and h2["fail_mode"] == "capture lost"
        h3 = R.h4_verdicts(
            _full(primary=((0.60, 0.61, 0.62),) * 3, primary_cap=(0.55, 0.55, 0.55)),
            _starts(),
        )
        assert h3["fail_mode"] == "still pinned"
        assert R.h4_verdicts(_full(), _starts())["fail_mode"] is None

    def test_reported_vs_rcm28(self):
        """Spec section 4: each arm vs RCM28 on the same stream, no verdict."""
        h = R.h4_verdicts(_full(), _starts())
        v = h["per_arm"]["FT-A1-34"]["vs_rcm28"]
        assert v["greedy_capture"] == pytest.approx(0.54)
        assert v["rcm28_greedy_capture"] == pytest.approx(0.13)
        assert set(v["heldout_deadlock"]) == set(R.HELD_OUT)
        assert v["rcm28_heldout_deadlock"]["ATT25"] == pytest.approx(0.10)


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        res = _full()
        res["FT-A1-34"]["train_violations"] = 1
        assert R.hsafe_verdict(res, _starts())["verdict"] == "FAILS"
        res = _full()
        res["SCR-W34"]["probe_violations"] = 2
        h = R.hsafe_verdict(res, _starts())
        assert h["verdict"] == "FAILS" and h["offenders"] == ["SCR-W34"]
        res = _full()
        res["FT-A2-35"]["final"]["pursuer_violations"] = 3
        assert R.hsafe_verdict(res, _starts())["pursuer_violation_steps"] == 3
        res = _full()
        del res["FT-A1-36"]
        h = R.hsafe_verdict(res, _starts())
        assert h["verdict"] == "INCOMPLETE" and h["missing"] == ["FT-A1-36"]

    def test_hsafe_counts_both_references(self):
        st = _starts()
        st["RC19"]["HUG7"]["pursuer_violations"] = 2
        h = R.hsafe_verdict(_full(), st)
        assert h["verdict"] == "FAILS" and h["offenders"] == ["reference:RC19/HUG7"]
        st = _starts()
        st["RCM28"]["ATT27"]["pursuer_violations"] = 5
        h = R.hsafe_verdict(_full(), st)
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 5
        assert h["offenders"] == ["reference:RCM28/ATT27"]

    def test_hsafe_counts_a_crash_records_partial_gate(self):
        res = _full()
        del res["SCR-W34"]["final"]  # crashed inside the gate
        res["SCR-W34"]["gate_violations"] = 1
        assert R.hsafe_verdict(res, _starts())["verdict"] == "FAILS"

    def test_hsafe_folds_in_stage_2(self):
        assert R.hsafe_verdict(_full(), _starts(), None)["verdict"] == "HOLDS"
        race = {"pursuer_violation_steps": 1, "offenders": ["ATK3-37"]}
        h = R.hsafe_verdict(_full(), _starts(), race)
        assert h["verdict"] == "FAILS" and "race:ATK3-37" in h["offenders"]
        clean = {"pursuer_violation_steps": 0, "offenders": []}
        assert R.hsafe_verdict(_full(), _starts(), clean)["verdict"] == "HOLDS"


class TestDirection:
    def _d(self, **kw):
        h = {"hsafe": "HOLDS", "H4": "PASS", "fail_mode": None, "H_RACE": None}
        return R.direction(h | kw)

    def test_hsafe_fails_stops_the_round(self):
        assert "Stop" in self._d(hsafe="FAILS")
        assert "Stop" in self._d(hsafe="FAILS", H4="FAIL", fail_mode="capture lost")

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(H4="INCOMPLETE")

    def test_pass_rows(self):
        assert "converges" in self._d(H_RACE="HOLDS")
        assert "exploitability" in self._d(H_RACE="FAILS")
        assert "stage 2 incomplete" in self._d(H_RACE="INCOMPLETE")
        assert "--race-start" in self._d(H_RACE=None)
        # the secondary family's pass takes the same rows
        assert "converges" in self._d(H4="PASS (β = 0.01)", H_RACE="HOLDS")

    def test_fail_rows(self):
        capture_lost = self._d(H4="FAIL", fail_mode="capture lost")
        assert "anchor did not protect capture" in capture_lost
        assert "RCM28" in capture_lost and "RC19" in capture_lost
        pinned = self._d(H4="FAIL", fail_mode="still pinned")
        assert "anchor held too tightly" in pinned and "not re-tuned" in pinned
        mixed = self._d(H4="FAIL", fail_mode="mixed")
        assert "capture" in mixed and "pinned" in mixed


class TestRace:
    """The S8 race is S6's machinery under S8's overrides."""

    def test_race_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "RACE_START_PATH", tmp_path / "s8_race_start.json")
        S6 = types.SimpleNamespace()
        R.race_setup(S6, tmp_path / "artefact.zip")
        assert S6.OUT == tmp_path
        assert S6.START_PATH == tmp_path / "s8_race_start.json"
        assert S6.ARTEFACT_PREFIX == "s8race"
        assert (S6.PROBE_SEED0, S6.GATE_SEED0) == (1_220_000, 1_230_000)
        assert (S6.GATE_N, S6.EVAL_N) == (R.GATE_N, R.EVAL_N)
        assert (S6.TOTAL_LONG, S6.EVERY_LONG) == (10_000_000, 1_000_000)
        assert S6.SEEDS == (37, 38, 39)
        assert set(S6.ARMS) == {"ATK3-37", "ATK3-38", "ATK3-39"}
        assert S6.ARMS["ATK3-37"] == {
            "family": "CTL",
            "defender": "D3",
            "seed": 37,
            "beta": 0.0,
            "zero_reward": False,
        }
        assert S6.DEFENDERS["D3"]["ckpt"] == tmp_path / "artefact.zip"
        assert S6.DEFENDERS["D3"]["shield_obs"] is True

    def test_race_start_validation_reads_the_s8_file(self, tmp_path, monkeypatch):
        path = tmp_path / "s8_race_start.json"
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "RACE_START_PATH", path)
        S6 = types.SimpleNamespace()
        R.race_setup(S6, tmp_path / "artefact.zip")
        with pytest.raises(SystemExit, match="race comparators missing"):
            S6.load_start()
        entry = {"pursuer_violations": 0, "pursuer": {"deadlock_episode_rate": 0.5}}
        path.write_text(json.dumps({"D3": {"HUG7": entry}}))
        with pytest.raises(SystemExit, match="D3/GREEDY"):
            S6.load_start()
        path.write_text(json.dumps({"D3": {"HUG7": entry, "GREEDY": entry}}))
        assert set(S6.load_start()["D3"]) == {"HUG7", "GREEDY"}

    def _race_arm(self, rate=0.45, viol_train=0, viol_gate=0):
        return {
            "train_pursuer_violations": viol_train,
            "probe_pursuer_violations": 0,
            "final": {
                "deadlock_ind": _ind(rate),
                "pursuer": {"deadlock_episode_rate": rate, "escape_rate": 0.7},
                "pursuer_violations": viol_gate,
            },
        }

    def test_race_summary_adds_the_s7_comparison(self):
        arms = {
            f"ATK3-{s}": self._race_arm(rate=r)
            for s, r in zip(R.SEEDS2, (0.10, 0.12, 0.11), strict=True)
        }
        r = R.race_summary(arms)
        assert r["H_RACE"] == "HOLDS" and r["race_bar"] == 0.58
        assert r["s7_race_rate"] == 0.147 and r["below_s7"] is True
        hot = {f"ATK3-{s}": self._race_arm(rate=0.30) for s in R.SEEDS2}
        r2 = R.race_summary(hot)
        assert r2["below_s7"] is False and r2["H_RACE"] == "HOLDS"
        dirty = dict(hot, **{"ATK3-38": self._race_arm(rate=0.30, viol_train=4)})
        assert R.race_summary(dirty)["pursuer_violation_steps"] == 4

    def test_race_summary_is_incomplete_without_all_three(self):
        arms = {f"ATK3-{s}": self._race_arm() for s in R.SEEDS2[:2]}
        assert R.race_summary(arms)["H_RACE"] == "INCOMPLETE"
        assert R.race_summary({})["below_s7"] is False


class TestSummary:
    def test_summarize_reports_both_references_and_the_direction(self):
        s = R.summarize(_full(), _starts())
        assert s["spec"].endswith("shield_remedy_3_prereg_2026-09-17.md")
        assert s["hypotheses"]["H4"] == "PASS"
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert set(s["references"]) == {"RC19", "RCM28"}
        assert s["references"]["RC19"]["GREEDY"]["capture"] == pytest.approx(0.55)
        assert set(s["arms"]) == set(R.ARMS)
        assert s["complete"] is True and "--race-start" in s["direction"]

    def test_write_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        results, starts = _full(), _starts()
        p = tmp_path / "s8_round.png"
        R.write_figure(results, starts, R.summarize(results, starts), p)
        assert p.exists() and p.stat().st_size > 0


class TestAnchorKLTrace:
    """The anchored arms' KL trace: SB3's logger clears `train/anchor_kl`, so the
    round samples `last_anchor_kl` off the model whenever a probe row appears."""

    def _cb(self, rows, trace, model):
        pytest.importorskip("stable_baselines3")
        cb = R._anchor_kl_callback(rows, trace)
        cb.model = model
        return cb

    def test_one_sample_per_probe_row(self):
        rows, trace = [], []
        model = types.SimpleNamespace(last_anchor_kl=None)
        cb = self._cb(rows, trace, model)
        assert cb._on_step() is True and trace == []  # no row yet, no sample
        rows.append({"steps": 501_000, "mark": 500_000})
        model.last_anchor_kl = 0.25
        cb._on_step()
        assert trace == [{"steps": 501_000, "mark": 500_000, "anchor_kl": 0.25}]
        model.last_anchor_kl = 0.31
        cb._on_step()
        assert len(trace) == 1  # no new row, no new sample
        rows.append({"steps": 1_000_400, "mark": 1_000_000})
        cb._on_step()
        assert trace[-1] == {
            "steps": 1_000_400,
            "mark": 1_000_000,
            "anchor_kl": 0.31,
        }

    def test_the_unanchored_control_records_none(self):
        rows, trace = [{"steps": 7, "mark": 5}], []
        self._cb(rows, trace, object())._on_step()  # plain PPO: no last_anchor_kl
        assert trace == [{"steps": 7, "mark": 5, "anchor_kl": None}]


class TestArtefactNamespace:
    def test_load_results_ignores_the_stage_2_arms(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s8_arm_FT-A1-34.json").write_text('{"arm": "FT-A1-34"}')
        (tmp_path / "s8race_arm_ATK3-37.json").write_text('{"arm": "ATK3-37"}')
        assert set(R.load_results()) == {"FT-A1-34"}
